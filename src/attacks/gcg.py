from dataclasses import dataclass
from typing import Literal

import nanogcg

from .attack import Attack


@dataclass
class GCGAttackResult:
    """A class to store the attack results for all instances in a datasets."""

    attacks: list[list[str]]
    losses: list[list[float]]
    prompt: list[str]
    completions: list[list[int | None]] = None


@dataclass
class GCGConfig:
    name: str = "gcg"
    type: str = "discrete"
    placement: Literal["suffix"] = "suffix"
    generate_completions: Literal["all", "best", "last"] = "last"
    num_steps: int = 250
    seed: int = 0
    batch_size: int | None = None
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    verbosity: str = "WARNING"


class GCGAttack(Attack):
    def __init__(self, config: GCGConfig):
        super().__init__(config)

    def run(self, model, tokenizer, dataset) -> list[nanogcg.gcg.GCGResult]:
        results: list[nanogcg.gcg.GCGResult] = []
        for message, target in dataset:
            message: dict[str, str]
            target: str
            result = nanogcg.run(model, tokenizer, [message], target, self.config)

            # Generate completions
            match self.config.generate_completions:
                case "all":
                    attacks = result.strings
                case "best":
                    attacks = [result.strings[result.losses.index(min(result.losses))]]
                case "last":
                    attacks = [result.strings[-1]]
                case _:
                    raise ValueError(
                        f"Unknown value for generate_completions: {self.config.generate_completions}"
                    )
            all_modified_messages = []

            for string in attacks:
                msg = {k: v for k, v in message.items()}
                msg["content"] = msg["content"] + " " + string
                all_modified_messages.append([msg])

            # Apply chat templates in batch
            templates = tokenizer.apply_chat_template(
                all_modified_messages, tokenize=False, add_generation_prompt=True
            )

            # Remove BOS token in batch
            for j in range(len(templates)):
                if tokenizer.bos_token and templates[j][0] == tokenizer.bos_token:
                    templates[j] = templates[j][1:]

            inputs = (
                tokenizer(
                    templates,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                .to(model.device)
                .input_ids
            )

            # Batch generation for all inputs
            outputs = model.generate(
                inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=512
            )
            # Extract newly generated tokens for all completions in one go
            new_tokens = outputs[:, inputs.shape[1] :]
            completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            results.append(
                GCGAttackResult(
                    losses=result.losses,
                    prompt=message,
                    attacks=result.strings,
                    completions=completions,
                )
            )
        return results
