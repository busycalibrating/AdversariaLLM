"""Implementation of a one-hot-input space continuous attack. Needs tuning."""
from dataclasses import dataclass
from typing import Literal
import functools
import time
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
from src.lm_utils import generate_ragged_batched, prepare_tokens, with_max_batchsize, get_disallowed_ids

from .attack import Attack, AttackResult


@dataclass
class RandomSearchConfig:
    name: str = "random_search"
    type: str = "discrete"
    placement: str = "suffix"
    generate_completions: Literal["all", "best", "last"] = "all"
    num_steps: int = 100
    seed: int = 0
    batch_size: int = 16
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    max_new_tokens: int = 256


class RandomSearchAttack(Attack):
    def __init__(self, config: RandomSearchConfig):
        super().__init__(config)
        self.batch_size = self.config.batch_size
        if self.config.placement == "suffix":
            assert self.config.optim_str_init
        elif self.config.placement == "prompt":
            assert not self.config.optim_str_init

    @torch.no_grad()
    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        self.disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)

        num_examples = len(dataset)

        x: list[torch.Tensor] = []
        attack_masks: list[torch.Tensor] = []
        target_masks: list[torch.Tensor] = []
        prompts = []
        for msg, target in dataset:
            pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens = (
                prepare_tokens(
                    tokenizer,
                    msg['content'],
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

        # Run the attack with dynamic batch size adjustment
        t0 = time.time()
        attack_fn = functools.partial(self.attack_batch, model, tokenizer)
        losses, completions, times = with_max_batchsize(attack_fn, x, y, attention_mask, attack_masks, target_masks)
        print(f"with_max_batchsize time: {time.time() - t0}")

        return AttackResult(
            attacks=[None] * num_examples,
            completions=completions,
            losses=losses,
            prompts=prompts,
            times=times,
        )

    def _get_random_allowed_tokens(self, model, tokenizer, k, happy_set, weights):
        # Create a mask of allowed token IDs (those not in disallowed_ids)
        allowed_mask = torch.ones(len(tokenizer), dtype=torch.bool, device=model.device)
        allowed_mask[self.disallowed_ids] = False

        # Get indices of allowed tokens
        allowed_indices = torch.nonzero(allowed_mask, as_tuple=True)[0]

        # Create a list of token weights, giving higher weight to tokens in happy_set
        weights = torch.softmax(-weights/0.1, dim=-1)[allowed_mask]

        if happy_set:
            for token_id in happy_set:
                token_idx = (allowed_indices == token_id).nonzero(as_tuple=True)[0]
                weights[token_idx] *= 50.0  # Increase weight for tokens in happy_set

        # Sample k random indices from the allowed tokens with weights
        random_indices = torch.multinomial(weights, k, replacement=True)
        # Return the actual token IDs
        return allowed_indices[random_indices]

    def attack_batch(self, model, tokenizer, x, y, attention_mask, attack_masks, target_masks):
        num_examples = x.size(0)
        losses = [[] for _ in range(num_examples)]
        completions = [[] for _ in range(num_examples)]
        times = [[] for _ in range(num_examples)]

        candidates = x.to(model.device)
        y = y.to(model.device)
        attention_mask = attention_mask.to(model.device)
        attack_masks = attack_masks.to(model.device)
        target_masks = target_masks.to(model.device)

        attack_sequences = []
        best_sequence = candidates.clone()
        best_losses = torch.full((candidates.size(0), ), float('inf'), device=candidates.device)
        happy_set = set()
        mean_losses = []
        mean_diffs = []
        t0 = time.time()

        def mutate_candidates(candidates, weights):
            for b in range(candidates.size(0)):
                random_allowed_tokens = self._get_random_allowed_tokens(model, tokenizer, k, happy_set, weights)
                n = attack_masks[b].sum()
                attack_indices = torch.multinomial(torch.ones(n), k, replacement=False)
                attack_positions = torch.nonzero(attack_masks[b], as_tuple=True)[0]
                positions_to_modify = attack_positions[attack_indices]
                candidates[b, positions_to_modify] = random_allowed_tokens[:k]
            return candidates

        avg_loss_per_token = torch.zeros(len(tokenizer), device=model.device)

        pbar = trange(self.config.num_steps)
        for _step in pbar:
            attack_sequences.extend([
                candidates[b, ~(tm.roll(1, 0).cumsum(0).bool())].clone()
                for b, tm in enumerate(target_masks)
            ])
            logits = model(
                input_ids=candidates,
                attention_mask=attention_mask,
            ).logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="none",
            )
            loss = loss * target_masks.view(-1)
            loss = loss.view(candidates.size(0), -1).mean(dim=1)

            mean_loss = loss.mean().item()
            mean_losses.append(mean_loss)
            if loss < best_losses:
                happy_set.update(set(candidates[0].tolist()))
            best_losses = torch.minimum(best_losses, loss)
            # check for
            for i, l in enumerate(loss.detach().tolist()):
                for token in (candidates[i] != best_sequence[i]).nonzero(as_tuple=True)[0]:
                    avg_loss_per_token[candidates[i, token].item()] = l-best_losses[i]
            # print({k: torch.mean(torch.tensor(v)).item() for k, v in avg_loss_per_token.items()})

            best_sequence = torch.where(loss == best_losses, candidates.detach().clone(), best_sequence)

            k = 1
            candidates = mutate_candidates(best_sequence.clone(), avg_loss_per_token)

            # Record diffs and losses
            diffs = (best_sequence != candidates).any(dim=-1).float()
            mean_diffs.append(diffs.mean().item())
            for j, l in enumerate(loss.detach().tolist()):
                losses[j].append(l)
                times[j].append(time.time() - t0)

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{mean_loss:.4f}", "k": k, "best_loss": f"{best_losses.mean().item():.4f}"})
            # Plot losses
            if (_step+1) % 100 == 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))

                # Plot losses over time
                plt.subplot(3, 1, 1)
                plt.scatter(range(len(mean_losses)), mean_losses, s=1, alpha=0.2, label='One-Hot Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.title('Loss vs. Training Step')
                plt.legend()
                plt.grid(True)

                plt.subplot(3, 1, 3)
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                # Plot edit distance with linear scale
                ax2.plot(mean_diffs, 'r-', label='Edit Distance [in one-hot space]')
                ax2.set_ylabel('Edit Distance (linear scale)', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                ax1.set_xlabel('Step')
                ax1.set_title('One-Hot Token Changes per Step')
                ax1.grid(True)

                plt.legend()

                plt.tight_layout()
                plt.savefig('loss_analysis.pdf')
                plt.close()

        # Flatten attack sequences for output generation
        # Each attack sequence corresponds to a different random search iteration
        # We need to flatten them to generate completions for all iterations

        outputs = generate_ragged_batched(
            model,
            tokenizer,
            token_list=attack_sequences,
            initial_batch_size=512,
            max_new_tokens=self.config.max_new_tokens,
        )

        for i, output in enumerate(outputs):
            completions[i // len(attack_sequences)].append(output)
        return losses, completions, times
