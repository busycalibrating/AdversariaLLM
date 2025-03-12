"""Implementation of the BEAST attack.

Adapted from https://github.com/dreadnode/research/blob/main/notebooks/Mistral%20-%20BEAST%20Beam%20Attack.ipynb
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lm_utils import (generate_ragged_batched, get_disallowed_ids,
                          prepare_tokens, with_max_batchsize)

from .attack import Attack, AttackResult


@dataclass
class BEASTConfig:
    name: str = "beast"
    type: str = "discrete"
    placement: str = "suffix"
    generate_completions: Literal["all", "best", "last"] = "all"
    num_steps: int = 30  # also the suffix length
    seed: int = 0
    optim_str_init: str = ""
    k1: int = 10
    k2: int = 10
    temperature: float = 1.0
    max_new_tokens: int = 256
    allow_non_ascii: bool = False
    allow_special: bool = False
    use_prefix_cache: bool = True


class BEASTAttack(Attack):
    def __init__(self, config: BEASTConfig):
        super().__init__(config)
        assert self.config.temperature > 0.0, "Temperature must be greater than 0 for BEAST"
        self.prefix_cache = None

    @torch.no_grad()
    def run(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset) -> AttackResult:
        """
        Runs the BEASTAttack on a given model and dataset.

        Args:
            model (torch.nn.Module): The language model to be attacked.
            tokenizer: Tokenizer compatible with the model.
            dataset: Iterable of (message, target) pairs containing the input prompts
                and the desired target strings.

        Returns:
            AttackResult: Holds all data about the generated attacks, losses, prompts,
                completions, and execution times.
        """
        t0 = time.time()
        num_examples = len(dataset)
        completions: list[list[str]] = [[] for _ in range(num_examples)]
        attacks, losses, times, prompts, token_list = [], [], [], [], []

        # Get disallowed token IDs once
        disallowed_ids = get_disallowed_ids(
            tokenizer,
            self.config.allow_non_ascii,
            self.config.allow_special
        ).to(model.device)

        for prompt_dict, target_string in dataset:
            pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens = prepare_tokens(
                tokenizer,
                prompt_dict["content"],
                target_string,
                attack=self.config.optim_str_init,
                placement=self.config.placement,
            )
            prompts.append(prompt_dict)

            # Compute KV cache for prefix tokens
            if self.config.use_prefix_cache:
                self.populate_prefix_cache(model, pre_tokens, prompt_tokens)

            initial_ppl = self.get_perplexity(
                model,
                pre_tokens,
                prompt_tokens,
                [attack_tokens],
                post_tokens,
                target_tokens,
            )[0]

            per_sample_attacks = [""]
            per_sample_losses = [torch.log(torch.tensor(initial_ppl)).item()]

            # Initial sampling
            beams = self.sample(
                model,
                k=self.config.k1,
                pre_tokens=pre_tokens,
                prompt_tokens=prompt_tokens,
                attack_token_list=[attack_tokens],
                post_tokens=post_tokens,
                disallowed_ids=disallowed_ids
            )[0]  # shape is (k1,)
            beams: list[torch.LongTensor] = [torch.LongTensor([]) for b in beams]
            for i in (pbar := trange(1, self.config.num_steps)):
                # Get next K1 x K2 candidates
                next_tokens = self.sample(
                    model,
                    self.config.k2,
                    pre_tokens,
                    prompt_tokens,
                    beams,
                    post_tokens,
                    disallowed_ids=disallowed_ids
                )  # (k1, k2)

                # Create all candidates
                candidates = []
                for beam, next_token in zip(beams, next_tokens):
                    candidates.extend([torch.cat((beam, t.unsqueeze(0))) for t in next_token])

                # Score candidates
                scores = self.get_perplexity(
                    model,
                    pre_tokens,
                    prompt_tokens,
                    candidates,
                    post_tokens,
                    target_tokens,
                )

                # Take the K1 best by lowest score
                sorting_indices = torch.tensor(scores).argsort().tolist()
                beams = [candidates[i] for i in sorting_indices[:self.config.k1]]

                # Record best result
                best_idx = sorting_indices[0]
                best_suffix = tokenizer.decode(candidates[best_idx])
                best_score = scores[best_idx]
                best_loss = torch.log(torch.tensor(best_score)).item()

                per_sample_attacks.append(best_suffix)
                per_sample_losses.append(best_loss)
                pbar.set_postfix({"loss": best_loss, "attack": best_suffix})
                times.append(time.time() - t0)

            attacks.append(per_sample_attacks)
            losses.append(per_sample_losses)

            # Prepare tokens for generation
            token_list.extend([
                torch.cat(
                    prepare_tokens(
                        tokenizer,
                        prompt_dict["content"],
                        "",
                        attack=attack,
                        placement=self.config.placement,
                    )[:4]
                )
                for attack in per_sample_attacks
            ])

        # Generate completions in batches
        outputs = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_list,
            initial_batch_size=512,
            max_new_tokens=self.config.max_new_tokens,
        )

        # Organize outputs into completions
        attacks_per_sample = len(attacks[0])
        for i, output in enumerate(outputs):
            completions[i // attacks_per_sample].append(output)

        return AttackResult(
            attacks=attacks,
            losses=losses,
            prompts=prompts,
            completions=completions,
            times=times,
        )

    def populate_prefix_cache(self, model, pre_tokens, prompt_tokens):
        """Compute KV cache for prefix tokens to avoid recomputing them."""
        prefix_input = torch.cat([pre_tokens, prompt_tokens]).unsqueeze(0).to(model.device)
        outputs = model(prefix_input, use_cache=True)
        self.prefix_cache = outputs.past_key_values

    def get_perplexity(
        self,
        model,
        pre_tokens,
        prompt_tokens,
        attack_tokens_list,
        post_tokens,
        target_tokens,
    ) -> list[float]:
        # Create tensor based on whether prefix cache is available
        T_cur = attack_tokens_list[0].size(0)
        T = self.config.num_steps
        # Pad attack tokens to ensure all have the same length
        padding_length = T - T_cur
        padded_attack_tokens_list = []
        for attack_tokens in attack_tokens_list:
            # Calculate padding needed
            if padding_length > 0:
                # Create padding tensor with pad token ID (usually 0)
                padding = torch.zeros(padding_length, dtype=attack_tokens.dtype, device=attack_tokens.device)
                # Concatenate attack tokens with padding
                padded_attack = torch.cat([attack_tokens, padding])
                padded_attack_tokens_list.append(padded_attack)
            else:
                # No padding needed
                padded_attack_tokens_list.append(attack_tokens)
        # Replace original list with padded version
        attack_tokens_list = padded_attack_tokens_list
        attention_mask = torch.zeros(T, dtype=torch.long, device=attack_tokens_list[0].device)
        attention_mask[:T_cur] = 1

        if self.prefix_cache is not None:
            # With prefix cache, we don't need to include prefix tokens
            tokens_to_concat = [
                torch.cat([attack_tokens, post_tokens, target_tokens])
                for attack_tokens in attack_tokens_list
            ]
        else:
            # Without prefix cache, include all tokens
            tokens_to_concat = [
                torch.cat([pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens])
                for attack_tokens in attack_tokens_list
            ]
        attention_mask = torch.cat([torch.ones(pre_tokens.size(0) + prompt_tokens.size(0)), attention_mask, torch.ones(post_tokens.size(0) + target_tokens.size(0))])
        attention_mask = attention_mask.to(model.device)

        tensor = torch.stack(tokens_to_concat)

        def get_log_probs(target_tokens, attention_mask, x):
            # Expand prefix cache to match batch size if available
            expanded_prefix_cache = None
            if self.prefix_cache is not None:
                expanded_prefix_cache = tuple(
                    tuple(t.expand(x.size(0), -1, -1, -1) for t in layer)
                    for layer in self.prefix_cache
                )
            attention_mask = attention_mask.unsqueeze(0).repeat(x.size(0), 1).to(model.device)

            # Get logits and compute log probabilities
            logits = model(input_ids=x.to(model.device), past_key_values=expanded_prefix_cache, attention_mask=attention_mask).logits
            output_logits = logits[:, -target_tokens.shape[0]:]
            log_probs = torch.nn.functional.log_softmax(output_logits, dim=-1).cpu()

            # Repeat target_tokens to match batch size
            target_tokens_expanded = target_tokens.unsqueeze(0).repeat(log_probs.size(0), 1).unsqueeze(-1)

            # Calculate perplexity
            gathered_log_probs = log_probs.gather(2, target_tokens_expanded).squeeze(-1)
            return gathered_log_probs.mean(dim=1)

        # Partial function to avoid repeating target_tokens
        get_log_probs_fn = partial(get_log_probs, target_tokens, attention_mask)

        # Process in batches
        mean_log_probs = with_max_batchsize(get_log_probs_fn, tensor[:, :-1], initial_batch_size=512)

        perplexity = torch.exp(-mean_log_probs).tolist()
        return perplexity

    def sample(
        self,
        model,
        k: int,
        pre_tokens,
        prompt_tokens,
        attack_token_list,
        post_tokens,
        disallowed_ids=None,
    ) -> torch.LongTensor:
        if self.prefix_cache is not None:
            # Use the prefix cache to avoid recomputing the prefix
            tensor = torch.stack([
                torch.cat([attack_tokens, post_tokens])
                for attack_tokens in attack_token_list
            ]).to(model.device)

            # Expand cache to match batch size
            expanded_prefix_cache = tuple(
                tuple(t.expand(tensor.size(0), -1, -1, -1) for t in layer)
                for layer in self.prefix_cache
            )
        else:
            # Fall back to the original implementation if prefix cache is not available
            tensor = torch.stack([
                torch.cat([pre_tokens, prompt_tokens, attack_tokens, post_tokens])
                for attack_tokens in attack_token_list
            ]).to(model.device)
            expanded_prefix_cache = None

        # Get logits for next token prediction
        logits = model(input_ids=tensor, past_key_values=expanded_prefix_cache).logits[:, -1, :]

        # Filter out disallowed tokens
        if disallowed_ids is not None:
            logits[:, disallowed_ids] = float('-inf')

        probs = torch.softmax(logits / self.config.temperature, dim=-1)
        tokens = torch.multinomial(probs, k, replacement=False)
        return tokens.cpu()  # Return as CPU tensor
