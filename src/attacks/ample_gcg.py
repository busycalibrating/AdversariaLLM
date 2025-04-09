"""Single-file implementation of the AmpleGCG attack."""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from accelerate.utils import find_executable_batch_size
from tqdm import tqdm, trange
from transformers import GenerationConfig as HuggingFaceGenerationConfig

from src.io_utils import free_vram, load_model_and_tokenizer
from src.lm_utils import get_batched_losses, generate_ragged_batched, prepare_tokens

from .attack import Attack, AttackResult, GenerationConfig


@dataclass
class PrompterLMConfig:
    id: str
    tokenizer_id: str
    chat_template: Optional[str]
    short_name: str
    developer_name: str
    batch_size: int
    dtype: str
    attn_implementation: Optional[str]
    trust_remote_code: bool
    compile: bool
    generation_config: dict


@dataclass
class TargetLMConfig:
    batch_size: int
    max_new_tokens: int


@dataclass
class AmpleGCGConfig:
    name: str = "ample_gcg"
    type: str = "discrete"
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    num_steps: int = 200
    prompter_lm: PrompterLMConfig = field(default_factory=PrompterLMConfig)
    target_lm: TargetLMConfig = field(default_factory=TargetLMConfig)


class AmpleGCGAttack(Attack):
    def __init__(self, config: AmpleGCGConfig):
        super().__init__(config)

    @torch.no_grad
    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        attacks = []
        losses = []
        prompts = []
        completions = []
        times = []
        for conversation in tqdm(dataset):
            assert len(conversation) == 2, "Current AmpleGCG only supports single-turn conversations"
            msg = conversation[0]["content"]
            target = conversation[1]["content"]
            # Temporarily move target model to cpu
            device = model.device
            model.cpu()
            free_vram()
            t0 = time.time()
            batch_attacks = self.get_attack_prompts(f"### Query:{msg} ### Prompt:")
            model.to(device)
            batch_losses = find_executable_batch_size(
                self.get_losses, self.config.target_lm.batch_size
            )(msg, batch_attacks, target, model, tokenizer)
            token_list = [
                torch.cat(prepare_tokens(
                    tokenizer,
                    prompt=msg,
                    target="",
                    attack=attack,
                )[:4])
                for attack in batch_attacks
            ]
            batch_completions = generate_ragged_batched(
                model,
                tokenizer,
                token_list=token_list,
                initial_batch_size=self.config.target_lm.batch_size,
                max_new_tokens=self.config.target_lm.max_new_tokens
            )
            attacks.append(batch_attacks)
            losses.append(batch_losses)
            prompts.append(conversation)
            completions.append(batch_completions)
            times.append([time.time() - t0] * len(batch_attacks))
        return AttackResult(attacks, losses, prompts, completions, times)

    def get_attack_prompts(self, msg):
        prompter_model = PrompterModel(self.config.prompter_lm, self.config.num_steps)
        prompter_model.eval()
        attacks = prompter_model.run([msg])
        free_vram()
        return attacks

    @staticmethod
    def _format_prompt(prompt, attack, tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{prompt} {attack}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_losses(self, batch_size, prompt, attacks, target, model, tokenizer):
        outputs = []
        for i in trange(0, len(attacks), batch_size, desc="Target Model"):
            token_list = [
                prepare_tokens(
                    tokenizer,
                    prompt=prompt,
                    target=target,  # need dummy target (probably)
                    attack=attack,
                )
                for attack in attacks[i : i + batch_size]
            ]
            target_lengths = [e.size(0) for a, b, c, d, e in token_list]
            token_list = [torch.cat(t, dim=0) for t in token_list]
            targets = [t.roll(-1, 0) for t in token_list]

            losses = get_batched_losses(
                model,
                targets=targets,
                token_list=token_list,
            )
            losses = [l[-tl:].mean().item() for l, tl in zip(losses, target_lengths)]
            outputs.extend(losses)
        return outputs


class PrompterModel(nn.Module):
    def __init__(self, config, num_steps):
        super().__init__()
        self.batch_size = config.batch_size
        self.model, self.tokenizer = load_model_and_tokenizer(config)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "num_return_sequences": num_steps,
            "num_beams": num_steps,
            "num_beam_groups": num_steps,
        }
        self.gen_config = HuggingFaceGenerationConfig(
            **config.generation_config, **self.gen_kwargs
        )

    def _prompterlm_run_batch(self, batch):
        input_ids = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = input_ids.to(self.model.device)
        # Does a beam search to generate multiple completions per prompt
        output = self.model.generate(**input_ids, generation_config=self.gen_config)
        output = output[:, input_ids["input_ids"].shape[-1] :]
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_text

    # q_s questions, p_s prompts
    def run(self, q_s):
        outputs = []
        batch_size = self.batch_size
        for i in trange(0, len(q_s), batch_size, desc="Prompter Model"):
            outputs.extend(self._prompterlm_run_batch(q_s[i : i + batch_size]))
        return outputs
