"""Implementation of a embedding-space continuous attack."""

import functools
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
from transformers import AutoModelForCausalLM

from src.attacks import (Attack, AttackResult, AttackStepResult,
                         GenerationConfig, SingleAttackRunResult)
from src.lm_utils import (generate_ragged_batched, prepare_conversation,
                          with_max_batchsize, TokenMergeError, get_disallowed_ids)


@dataclass
class PGDConfig:
    name: str = "pgd"
    type: str = "continuous"
    num_steps: int = 100
    seed: int = 0
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    batch_size: int = 16
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    epsilon: float = 100000.0
    alpha: float = 0.001
    max_new_tokens: int = 256
    embedding_scale: Optional[float] = None
    normalize_alpha: bool = False
    normalize_gradient: bool = False
    original_model: Optional[str] = None
    tie_embeddings: float = 0.0
    tie_features: float = 0.0


class PGDAttack(Attack):
    def __init__(self, config: PGDConfig):
        super().__init__(config)
        self.batch_size = self.config.batch_size
        self.zero_init_attack = False

    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        if self.config.embedding_scale is None:
            if self.config.projection == "l2":
                self.config.embedding_scale = (
                    model.get_input_embeddings().weight.norm(dim=-1).mean().item()
                )
            elif self.config.projection == "l1":
                self.config.embedding_scale = (
                    model.get_input_embeddings().weight.norm(dim=-1, p=1).mean().item()
                )
        logging.info(f"Embedding scale set to {self.config.embedding_scale} based on projection={self.config.projection}")
        if self.config.original_model is None:
            original_model = None
        else:
            original_model = AutoModelForCausalLM.from_pretrained(
                self.config.original_model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            ).eval()
            logging.info(f"Loaded {self.config.original_model} for logit/feature tying")

        x: list = []
        attack_masks: list = []
        target_masks: list = []
        conversations = []

        for conversation in dataset:
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + self.config.optim_str_init},
                {"role": "assistant", "content": conversation[1]["content"]}
            ]
            try:
                pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks = prepare_conversation(tokenizer, conversation, attack_conversation)[0]
            except TokenMergeError:
                attack_conversation = [
                    {"role": "user", "content": conversation[0]["content"] + " " + self.config.optim_str_init},
                    {"role": "assistant", "content": conversation[1]["content"]}
                ]
                pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks = prepare_conversation(tokenizer, conversation, attack_conversation)[0]

            tokens = torch.cat(
                [pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks]
            )
            attack_mask = torch.cat(
                [
                    torch.zeros_like(pre_toks),
                    torch.ones_like(attack_prefix_toks),
                    torch.zeros_like(prompt_toks),
                    torch.ones_like(attack_suffix_toks),
                    torch.zeros_like(post_toks),
                    torch.zeros_like(target_toks),
                ]
            )
            target_mask = torch.cat(
                [
                    torch.zeros_like(pre_toks),
                    torch.zeros_like(attack_prefix_toks),
                    torch.zeros_like(prompt_toks),
                    torch.zeros_like(attack_suffix_toks),
                    torch.zeros_like(post_toks),
                    torch.ones_like(target_toks),
                ]
            ).roll(-1, 0)
            conversations.append(attack_conversation)
            x.append(tokens)
            attack_masks.append(attack_mask)
            target_masks.append(target_mask)
        logging.info(f"Prepared {len(conversations)} conversations for attack")

        x = pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id)
        target_masks = pad_sequence(target_masks, batch_first=True)
        attack_masks = pad_sequence(attack_masks, batch_first=True)
        attention_mask = (x != tokenizer.pad_token_id).long()

        y = x.clone()
        y[:, :-1] = x[:, 1:]
        # Run the attack

        attack_fn = functools.partial(self.attack_batch, model, tokenizer, original_model)
        runs = with_max_batchsize(
            attack_fn,
            x,
            y,
            conversations,
            attention_mask,
            attack_masks,
            target_masks,
        )
        return AttackResult(runs=runs)

    def attack_batch(
        self,
        model,
        tokenizer,
        original_model,
        x_batch,
        y_batch,
        original_conversations_batch,
        attention_mask_batch,
        attack_masks_batch,
        target_masks_batch
    ) -> list[SingleAttackRunResult]:
        disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)
        B = x_batch.shape[0]
        losses = [[] for _ in range(B)]
        perturbed_embeddings_list = [[] for _ in range(B)]
        times = [[] for _ in range(B)]
        # Perform the actual attack
        t0 = time.time()
        x_batch = x_batch.to(model.device)
        y_batch = y_batch.to(model.device)
        attention_mask_batch = attention_mask_batch.to(model.device)
        attack_masks_batch = attack_masks_batch.to(model.device)
        target_masks_batch = target_masks_batch.to(model.device)

        original_embeddings = model.get_input_embeddings()(x_batch)
        if self.zero_init_attack:
            original_embeddings[attack_masks_batch.bool()] = 0

        # loss to minimize interference of adversarial attack on the model
        with torch.no_grad():
            prompt_benign = "Can you recommend a chocolate cake recipe for two?"
            conversation_benign = [
                {"role": "user", "content": prompt_benign},
                {"role": "assistant", "content": ""}
            ]
            # Process benign conversation
            tokens_parts = prepare_conversation(tokenizer, conversation_benign)[0]
            pre_toks_benign, attack_prefix_toks_benign, prompt_toks_benign, attack_suffix_toks_benign, post_toks_benign, _ = tokens_parts

            # Create reference input and generate target tokens
            benign_ref = torch.cat([pre_toks_benign, attack_prefix_toks_benign, prompt_toks_benign, attack_suffix_toks_benign, post_toks_benign]).unsqueeze(0).to(model.device)
            target_tokens_benign = generate_ragged_batched(model, tokenizer, benign_ref, max_new_tokens=64, return_tokens=True)[0][0]

            # Get embeddings for all token parts
            token_parts_with_target = (pre_toks_benign, attack_prefix_toks_benign, prompt_toks_benign,
                                       attack_suffix_toks_benign, post_toks_benign, target_tokens_benign)
            all_embeds = [model.get_input_embeddings()(ids.to(model.device).unsqueeze(0)).repeat(B, 1, 1)
                          for ids in token_parts_with_target]
            pre_embeds_benign, _, prompt_embeds_benign, _, post_embeds_benign, target_embeds_benign = all_embeds

            # Calculate generation size and get model output without attack
            gen_size = post_toks_benign.size(0) + target_tokens_benign.size(0)
            model_logits_no_attack = model(
                inputs_embeds=torch.cat([pre_embeds_benign, prompt_embeds_benign, post_embeds_benign, target_embeds_benign], dim=1),
            ).logits[:, -gen_size:]

        perturbed_embeddings = original_embeddings.detach().clone()
        pbar = trange(self.config.num_steps, desc=f"Running PGD Attack Loop on {B} conversations", file=sys.stdout)
        for _ in pbar:
            perturbed_embeddings.requires_grad = True
            outputs = model(
                inputs_embeds=perturbed_embeddings,
                attention_mask=attention_mask_batch,
                output_hidden_states=True
            )
            logits = outputs.logits
            features = outputs.hidden_states
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1),
                reduction="none",
            )
            loss = loss.view(B, -1) * target_masks_batch  # (B, L)
            loss = loss.sum(dim=1) / (target_masks_batch.sum(dim=1) + 1e-6)  # (B,)

            if self.config.tie_logits and self.config.original_model is not None:
                original_outputs = original_model(
                    inputs_embeds=perturbed_embeddings,
                    attention_mask=attention_mask_batch,
                    output_hidden_states=True
                )
                original_logits = original_outputs.logits
                original_features = original_outputs.hidden_states

                # Add KL divergence penalty for both logits and intermediate features
                kl_div_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(original_logits, dim=-1),
                    reduction="batchmean"
                ) * self.config.tie_logits
                # Calculate cosine similarity for each layer's features
                for perturbed_layer, original_layer in zip(features, original_features):
                    kl_div_loss += (1 - F.cosine_similarity(perturbed_layer, original_layer, dim=-1).mean()) * self.config.tie_features

                attack_embeds_benign = perturbed_embeddings[attack_masks_batch.bool()].view(B, -1, perturbed_embeddings.size(-1))
                inputs_embeds = torch.cat([pre_embeds_benign, prompt_embeds_benign, attack_embeds_benign, post_embeds_benign, target_embeds_benign], dim=1)
                model_outputs = model(inputs_embeds=inputs_embeds)
                model_logits = model_outputs.logits[:, -gen_size:]
                kl_div_loss = F.kl_div(
                    F.log_softmax(model_logits, dim=-1),
                    F.softmax(model_logits_no_attack, dim=-1),
                    reduction="batchmean"
                ) * self.config.tie_logits
            else:
                kl_div_loss = 0.0

            total_loss = loss + kl_div_loss
            total_loss.sum().backward()
            for i, l in enumerate(loss.detach().tolist()):
                losses[i].append(l)

            # Zero out disallowed token IDs for attack tokens
            perturbed_embeddings.grad[..., disallowed_ids] = 0
            perturbed_embeddings.grad[~attack_masks_batch.bool()] = 0

            with torch.no_grad():
                perturbed_embeddings = (
                    perturbed_embeddings
                    - self.config.alpha
                    * (self.config.embedding_scale if self.config.normalize_alpha else 1)
                    * perturbed_embeddings.grad.sign()
                    * ((1/perturbed_embeddings.grad.sign().norm(dim=-1, keepdim=True)).nan_to_num(1) if self.config.normalize_gradient else 1)
                    * attack_masks_batch[..., None]
                )
                if self.config.projection == "l2":
                    delta = self.project_l2(perturbed_embeddings - original_embeddings)
                elif self.config.projection == "l1":
                    delta = self.project_l1(perturbed_embeddings - original_embeddings)
                else:
                    raise ValueError(f"Unknown projection {self.config.projection}")
                perturbed_embeddings = (original_embeddings + delta).detach()

            for i, (pe, tm) in enumerate(zip(perturbed_embeddings, target_masks_batch)):
                perturbed_embeddings_list[i].append(self.select_tokens(pe, tm))

            t = time.time() - t0
            for i in range(x_batch.size(0)):
                times[i].append(t)
            pbar.set_postfix({"loss": loss.mean().item()})
        flattened_embeddings = [e for el in perturbed_embeddings_list for e in el]
        outputs = generate_ragged_batched(
            model,
            tokenizer,
            embedding_list=flattened_embeddings,
            initial_batch_size=512,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )
        logging.info(f"Generated {len(outputs)}x{len(outputs[0])} completions")

        runs = []
        for i in range(B):
            steps = []
            for step in range(self.config.num_steps):
                steps.append(AttackStepResult(
                    step=step,
                    model_completions=outputs[i*self.config.num_steps + step],
                    time_taken=times[i][step],
                    loss=losses[i][step],
                ))

            runs.append(SingleAttackRunResult(
                original_prompt=original_conversations_batch[i],
                steps=steps,
                total_time=times[i][-1]
            ))
        return runs

    def project_l2(self, delta):
        # We project the perturbation to have at most epsilon L2 norm
        # To compare across model families, we normalize epsilon by the mean embedding norm
        norm = delta.norm(p=2, dim=-1, keepdim=True)
        eps_normalized = self.config.epsilon * self.config.embedding_scale
        mask = norm > eps_normalized
        return torch.where(mask, delta * eps_normalized / norm, delta)

    def project_l1(self, delta):
        """
        Compute Euclidean projection onto the L1 ball for a batch.

        min ||x - u||_2 s.t. ||u||_1 <= eps

        Inspired by the corresponding numpy version by Adrien Gaidon.

        Parameters
        ----------
        x: (batch_size, *) torch array
        batch of arbitrary-size tensors to project, possibly on GPU

        eps: float
        radius of l-1 ball to project onto

        Returns
        -------
        u: (batch_size, *) torch array
        batch of projected tensors, reshaped to match the original

        Notes
        -----
        The complexity of this algorithm is in O(dlogd) as it involves sorting x.

        References
        ----------
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
            International Conference on Machine Learning (ICML 2008)
        """
        b, t, d = delta.shape
        eps = self.config.epsilon * self.config.embedding_scale
        original_shape = delta.shape
        dtype = delta.dtype
        delta = delta.view(b * t, -1)
        mask = (torch.norm(delta, p=1, dim=1) < eps).float().unsqueeze(1)
        mu, _ = torch.sort(torch.abs(delta), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, delta.shape[1] + 1, device=delta.device)
        rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
        theta = (cumsum[torch.arange(b * t), rho.cpu() - 1] - eps) / rho
        proj = (torch.abs(delta) - theta.unsqueeze(1)).clamp(min=0)
        delta = mask * delta + (1 - mask) * proj * torch.sign(delta)
        return delta.view(original_shape).to(dtype)

    @staticmethod
    def select_tokens(embeddings, mask):
        return embeddings[~(mask.roll(1, 0).cumsum(0).bool())].detach().cpu()
