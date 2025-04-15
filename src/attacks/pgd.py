"""Implementation of a embedding-space continuous attack."""

import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from accelerate.utils import find_executable_batch_size
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
from transformers import AutoModelForCausalLM

from src.lm_utils import prepare_tokens, generate_ragged_batched

from .attack import Attack, AttackResult


@dataclass
class PGDConfig:
    name: str = "pgd"
    type: str = "continuous"
    placement: str = "suffix"
    generate_completions: Literal["all", "best", "last"] = "all"
    num_steps: int = 100
    seed: int = 0
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
        if self.config.placement == "suffix":
            assert self.config.optim_str_init
            self.zero_init_attack = self.config.optim_str_init.endswith("zero")
        elif self.config.placement == "prompt":
            assert not self.config.optim_str_init

    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        x: list = []
        attack_masks: list = []
        target_masks: list = []
        prompts = []
        if self.config.embedding_scale is None:
            if self.config.projection == "l2":
                self.config.embedding_scale = (
                    model.get_input_embeddings().weight.norm(dim=-1).mean().item()
                )
            elif self.config.projection == "l1":
                self.config.embedding_scale = (
                    model.get_input_embeddings().weight.norm(dim=-1, p=1).mean().item()
                )
        if self.config.original_model is None:
            original_model = None
        else:
            original_model = AutoModelForCausalLM.from_pretrained(self.config.original_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto").eval()

        for msg, target in dataset:
            pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens = (
                prepare_tokens(
                    tokenizer,
                    msg["content"],
                    target,
                    attack=self.config.optim_str_init,
                    placement=self.config.placement,
                )
            )

            tokens = torch.cat(
                [pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens]
            )
            attack_mask = torch.cat(
                [
                    torch.zeros_like(pre_tokens),
                    torch.zeros_like(prompt_tokens),
                    torch.ones_like(attack_tokens),
                    torch.zeros_like(post_tokens),
                    torch.zeros_like(target_tokens),
                ]
            )
            target_mask = torch.cat(
                [
                    torch.zeros_like(pre_tokens),
                    torch.zeros_like(prompt_tokens),
                    torch.zeros_like(attack_tokens),
                    torch.zeros_like(post_tokens),
                    torch.ones_like(target_tokens),
                ]
            ).roll(-1, 0)
            prompts.append(msg)
            x.append(tokens)
            attack_masks.append(attack_mask)
            target_masks.append(target_mask)

        x = pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id)
        target_masks = pad_sequence(target_masks, batch_first=True)
        attack_masks = pad_sequence(attack_masks, batch_first=True)
        attention_mask = (x != tokenizer.pad_token_id).long()

        y = x.clone()
        y[:, :-1] = x[:, 1:]
        # Run the attack
        losses, completions, times = find_executable_batch_size(
            self.attack_batched, self.batch_size
        )(
            x,
            y,
            attention_mask,
            attack_masks,
            target_masks,
            model,
            original_model,
            tokenizer,
        )
        # assemble the results
        return AttackResult(
            attacks=[None] * x.size(0),
            completions=completions,
            losses=losses,
            prompts=prompts,
            times=times,
        )

    def attack_batched(
        self,
        batch_size,
        x,
        y,
        attention_mask,
        attack_masks,
        target_masks,
        model,
        original_model,
        tokenizer,
    ):
        num_examples = x.size(0)
        losses = [[] for _ in range(num_examples)]
        completions = [[] for _ in range(num_examples)]
        perturbed_embeddings_list = [[] for _ in range(num_examples)]
        times = [[] for _ in range(num_examples)]
        # Perform the actual attack
        logging.info(f"Attacking {num_examples} examples in batches of {batch_size}")
        t0 = time.time()
        for i in range(0, num_examples, batch_size):
            x_batch = x[i : i + batch_size].to(model.device)
            y_batch = y[i : i + batch_size].to(model.device)
            attention_mask_batch = attention_mask[i : i + batch_size].to(model.device)
            attack_masks_batch = attack_masks[i : i + batch_size].to(model.device)
            target_masks_batch = target_masks[i : i + batch_size].to(model.device)

            B = x_batch.shape[0]

            original_embeddings = model.get_input_embeddings()(x_batch)
            if self.zero_init_attack:
                original_embeddings[attack_masks_batch.bool()] = 0

            # loss to minimize interference of adversarial attack on the model
            with torch.no_grad():
                prompt_benign = "Can you recommend a chocolate cake recipe for two?"
                pre_tokens_benign, prompt_tokens_benign, attack_tokens_benign, post_tokens_benign, _ = prepare_tokens(tokenizer, prompt_benign, "", attack=self.config.optim_str_init, placement="suffix")
                benign_ref = torch.cat([pre_tokens_benign, prompt_tokens_benign, post_tokens_benign]).unsqueeze(0).to(model.device)
                target_tokens_benign = generate_ragged_batched(model, tokenizer, benign_ref, max_new_tokens=64, return_tokens=True)[0]
                pre_embeds_benign, post_embeds_benign, prompt_embeds_benign, attack_embeds_benign, target_embeds_benign = [model.get_input_embeddings()(ids.to(model.device).unsqueeze(0)) for ids in (pre_tokens_benign, post_tokens_benign, prompt_tokens_benign, attack_tokens_benign, target_tokens_benign)]
                pre_embeds_benign = pre_embeds_benign.repeat(B, 1, 1)
                prompt_embeds_benign = prompt_embeds_benign.repeat(B, 1, 1)
                post_embeds_benign = post_embeds_benign.repeat(B, 1, 1)
                target_embeds_benign = target_embeds_benign.repeat(B, 1, 1)

                gen_size = post_tokens_benign.size(0) + target_tokens_benign.size(0)
                model_logits_no_attack = model(
                    inputs_embeds=torch.cat([pre_embeds_benign, prompt_embeds_benign, post_embeds_benign, target_embeds_benign], dim=1),
                ).logits[:, -gen_size:]

            perturbed_embeddings = original_embeddings.detach().clone()
            for _ in (pbar := trange(self.config.num_steps)):
                perturbed_embeddings.requires_grad = True
                # Extract features from both models
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
                loss = loss.view(B, -1)
                loss = loss * target_masks_batch  # (B, L)
                loss = loss.mean(dim=1)  # (B,)

                if self.config.original_model is not None:
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
                else:
                    kl_div_loss = 0.0

                attack_embeds_benign = perturbed_embeddings[attack_masks_batch.bool()].view(B, -1, perturbed_embeddings.size(-1))
                inputs_embeds = torch.cat([pre_embeds_benign, prompt_embeds_benign, attack_embeds_benign, post_embeds_benign, target_embeds_benign], dim=1)
                model_outputs = model(inputs_embeds=inputs_embeds)
                model_logits = model_outputs.logits[:, -gen_size:]
                # model_features = model_outputs.hidden_states
                kl_div_loss = F.kl_div(
                    F.log_softmax(model_logits, dim=-1),
                    F.softmax(model_logits_no_attack, dim=-1),
                    reduction="batchmean"
                ) * self.config.tie_logits
                # Calculate cosine similarity for each layer's features
                # for perturbed_layer, original_layer in zip(model_features, model_features_no_attack):
                #     kl_div_loss += (1 - F.cosine_similarity(perturbed_layer[:, -gen_size:], original_layer[:, -gen_size:], dim=-1).mean()) * self.config.tie_features

                total_loss = loss + kl_div_loss
                total_loss.sum().backward()
                for j, l in enumerate(loss.detach().tolist()):
                    losses[i + j].append(l)

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
                t = time.time() - t0
                for j in range(x_batch.size(0)):
                    times[i + j].append(t)
                pbar.set_postfix({"loss": loss.mean().item()})
                if self.config.generate_completions == "all":
                    for j, (pe, tm) in enumerate(zip(perturbed_embeddings, target_masks_batch)):
                        perturbed_embeddings_list[i + j].append(self.select_tokens(pe, tm))
            if self.config.generate_completions == "last":
                perturbed_embeddings_list[i : i + batch_size] = [
                    [self.select_tokens(pe, tm) for pe, tm in zip(perturbed_embeddings, target_masks_batch)]
                ]

        flattened_embeddings = [e for el in perturbed_embeddings_list for e in el]
        outputs = generate_ragged_batched(
            model,
            tokenizer,
            embedding_list=flattened_embeddings,
            initial_batch_size=256,
            max_new_tokens=self.config.max_new_tokens,
        )

        for i, output in enumerate(outputs):
            completions[i // len(perturbed_embeddings_list[0])].append(output)
        return losses, completions, times

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
