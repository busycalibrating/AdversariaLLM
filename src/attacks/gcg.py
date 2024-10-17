from dataclasses import dataclass
from typing import Literal

# TODO: move nanogcg code into this package
import nanogcg

from .attack import Attack, AttackResult
from src.lm_utils import get_batched_completions


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

    def run(self, model, tokenizer, dataset) -> AttackResult:
        results = AttackResult([], [], [], [])
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

            token_list = [
                tokenizer(t, return_tensors="pt", add_special_tokens=True)
                .input_ids[0]
                .to(model.device)
                for t in templates
            ]
            completions = get_batched_completions(model, tokenizer, token_list=token_list, max_new_tokens=512, return_tokens=False)
            results.losses.append(result.losses)
            results.attacks.append(result.strings)
            results.prompts.append(message)
            results.completions.append(completions)
        return results
