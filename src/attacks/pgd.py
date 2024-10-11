"""Implementation of a embedding-space continuous attack."""

import sys
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from accelerate.utils import find_executable_batch_size
from tqdm import trange

from .attack import Attack


@dataclass
class GCGAttackResult:
    """A class to store the attack result for a single instance in a dataset."""

    attacks: list[str]
    losses: list[float]
    prompt: list
    completions: list[int | None] = None


@dataclass
class PGDConfig:
    name: str = "pgd"
    type: str = "continuous"
    placement: str = "command"
    generate_completions: Literal["all", "best", "last"] = "last"
    num_steps: int = 30
    seed: int = 0
    batch_size: int = 2
    optim_str_init: str = ""
    epsilon: float = 100000.0
    alpha: float = 0.005
    generation_steps: int = 256


class PGDAttack(Attack):
    def __init__(self, config: PGDConfig):
        super().__init__(config)
        self.batch_size = self.config.batch_size
        if self.config.placement == "suffix":
            assert self.config.optim_str_init
        elif self.config.placement == "command":
            assert not self.config.optim_str_init

    def run(self, model: torch.nn.Module, tokenizer, dataset) -> list[GCGAttackResult]:
        """The attack is in embedding"""
        num_examples = len(dataset)

        tokens: list = []
        optim_masks: list = []
        target_masks: list = []

        def process_example(msg, target):
            """Preprocess the example for the attack. This is easier because prompts
            don't all have the same length
            """
            message = tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )[0]
            # Remove the BOS token -- this will get added when tokenizing, if necessary
            if tokenizer.bos_token and message[0] == tokenizer.bos_token:
                message = message[1:]
            input = (
                tokenizer(
                    [message],
                    return_tensors="pt",
                )
                .to(model.device)
                .input_ids
            )[0]

            if self.config.placement == "suffix":
                attack_embeds = (
                    tokenizer(
                        [self.config.optim_str_init],
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    .to(model.device)
                    .input_ids
                )[0]
                optim_mask = torch.cat(
                    [torch.zeros_like(input), torch.ones_like(attack_embeds)], dim=0
                )
                input = torch.cat([input, attack_embeds], dim=0)
            else:
                optim_mask = torch.ones_like(input)

            target = (
                tokenizer(
                    [" " + target],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                .to(model.device)
                .input_ids
            )[0]
            target_mask = torch.cat((torch.zeros_like(input), torch.ones_like(target)))
            token_sequence = torch.cat((input, target))
            target_mask = target_mask.roll(shifts=-1, dims=0)
            optim_mask = torch.cat((optim_mask, torch.zeros_like(target)))
            return token_sequence, optim_mask, target_mask

        prompts = []
        for msg, target in dataset:
            token_sequence, optim_mask, target_mask = process_example(msg, target)
            prompts.append(msg)
            tokens.append(token_sequence)
            optim_masks.append(optim_mask)
            target_masks.append(target_mask)

        inputs = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        target_masks = torch.nn.utils.rnn.pad_sequence(target_masks, batch_first=True)
        attention_mask = (inputs != tokenizer.pad_token_id).long()
        optim_masks = torch.nn.utils.rnn.pad_sequence(optim_masks, batch_first=True)

        labels = inputs.clone()
        labels[:, :-1] = inputs[:, 1:]

        def attack(batch_size):
            losses = [[] for _ in range(num_examples)]
            completions = [[] for _ in range(num_examples)]
            # Perform the actual attack
            for i in range(0, num_examples, batch_size):
                original_embeddings = model.get_input_embeddings()(
                    inputs[i : i + batch_size]
                )
                perturbed_embeddings = original_embeddings.clone().detach()
                for _ in trange(self.config.num_steps, file=sys.stderr):
                    perturbed_embeddings.requires_grad = True
                    model.zero_grad()
                    outputs = model(
                        inputs_embeds=perturbed_embeddings,
                        attention_mask=attention_mask[i : i + batch_size],
                    )
                    loss = F.cross_entropy(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        labels[i : i + batch_size].view(-1),
                        reduction="none",
                    )
                    loss = loss * target_masks[i : i + batch_size].view(-1)
                    loss = loss.view(perturbed_embeddings.size(0), -1).mean(dim=1)
                    loss.mean().backward()
                    for j, l in enumerate(loss.detach().tolist()):
                        losses[i + j].append(l)

                    with torch.no_grad():
                        grad = perturbed_embeddings.grad
                        perturbed_embeddings = (
                            perturbed_embeddings
                            - self.config.alpha
                            * grad.sign()
                            * optim_masks[i : i + batch_size, ..., None]
                        )
                        delta = self.project_l2(
                            perturbed_embeddings - original_embeddings
                        )
                        perturbed_embeddings = (original_embeddings + delta).detach()

                    if self.config.generate_completions == "all":
                        # Get completions right away
                        completion = self.get_completions(
                            model,
                            tokenizer,
                            perturbed_embeddings,
                            optim_masks[i : i + batch_size],
                            self.config.generation_steps,
                        )
                        for j, c in enumerate(completion):
                            completions[i + j].append(c)
                    elif self.config.generate_completions == "best":
                        for j in range(batch_size):
                            if losses[i + j][-1] == min(losses[i + j]):
                                completion = self.get_completions(
                                    model,
                                    tokenizer,
                                    perturbed_embeddings[j : j + 1],
                                    optim_masks[i + j : i + j + 1],
                                    self.config.generation_steps,
                                )
                                completions[i + j] = [completion[0]]
                if self.config.generate_completions == "last":
                    completion = self.get_completions(
                        model,
                        tokenizer,
                        perturbed_embeddings,
                        optim_masks[i : i + self.batch_size],
                        self.config.generation_steps,
                    )
                    for j, c in enumerate(completion):
                        completions[i + j].append(c)

            return losses, completions

        # Run the attack
        losses, completions = find_executable_batch_size(attack, self.batch_size)()
        # assemble the results
        results = []
        for i in range(num_examples):
            results.append(
                GCGAttackResult(
                    attacks=None,
                    prompt=prompts[i],
                    losses=losses[i],
                    completions=completions[i],
                )
            )

        return results

    @staticmethod
    @torch.no_grad
    def get_completions(
        model, tokenizer, perturbed_embeddings, optim_masks, gen_steps: int = 256
    ):
        # We first pad the embeddings to the maximum context length of the model.
        # Positions after the current one will be ignored via the attention mask.
        B, T = perturbed_embeddings.size()[:2]
        padding = (0, 0, 0, T + gen_steps + 2)
        padded_embeddings = F.pad(perturbed_embeddings, padding)
        # Then we generate the completions
        tokens = []
        next_token_idx = optim_masks.sum(dim=1)
        for i in range(gen_steps):
            max_mask_indices = next_token_idx.max()
            outputs = model(
                inputs_embeds=padded_embeddings[:, :max_mask_indices],
            )

            next_tokens = outputs.logits.argmax(dim=-1)[
                torch.arange(B), next_token_idx - 1
            ]
            tokens.append(next_tokens)

            padded_embeddings[torch.arange(B), next_token_idx] = (
                model.get_input_embeddings()(next_tokens).detach()
            )
            next_token_idx += 1

        completion = tokenizer.batch_decode(
            torch.stack(tokens, dim=0).T, skip_special_tokens=False
        )
        completion = [c.split(tokenizer.eos_token)[0] for c in completion]
        return completion

    def project_l2(self, delta):
        norm = delta.norm(p=2, dim=-1, keepdim=True)
        mask = norm > self.config.epsilon
        return torch.where(mask, delta * self.config.epsilon / norm, delta)
