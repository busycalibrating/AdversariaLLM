import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import json
import logging
import time

from concurrent.futures import ThreadPoolExecutor
import hydra
import torch
from accelerate.utils import find_executable_batch_size
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.errors import print_exceptions
from src.judges import judge_cais, judge_llama_guard_3_8b, judge_strong_reject, judge_overrefusal

torch.use_deterministic_algorithms(True, warn_only=True)  # determinism
torch.backends.cuda.matmul.allow_tf32 = True


def is_judged(run, key):
    if key not in run:
        return False
    completions = run["completions"]
    successes = run[key]
    if isinstance(successes, float):
        return False
    if len(completions) != len(successes):
        return False
    if not all(len(c) == len(s) for c, s in zip(completions, successes)):
        return False
    if "p_harmful" in key:
        if "strong_reject" in key:
            if not all(all(all(isinstance(f, float) for f in p) for p in s) for s in successes):
                return False
        elif not all(all(isinstance(p, float) for p in s) for s in successes):
            return False

    elif "successes" in key:
        if "overrefusal" in key:
            if not all(all(j in ["direct_answer", "direct_refusal", "indirect_refusal"] for j in s) for s in successes):
                return False
        elif "strong_reject" in key:
            if not all(all(isinstance(p, float) for p in s) for s in successes):
                return False
        else:
            if not all(all(j in ["Yes", "No"] for j in s) for s in successes):
                return False
    return True

def load_model_and_tokenizer(classifier_name):
    model_names = {
        "cais": ("cais/HarmBench-Llama-2-13b-cls", judge_cais),
        "llama_guard_3_8b": ("meta-llama/Llama-Guard-3-8B", judge_llama_guard_3_8b),
        "strong_reject": ("qylu4156/strongreject-15k-v1", judge_strong_reject),
        "overrefusal": ("meta-llama/Meta-Llama-3.1-8B-Instruct", judge_overrefusal),
    }
    if classifier_name not in model_names:
        raise ValueError(f"Unknown classifier {classifier_name}")

    model_name, judge_func = model_names[classifier_name]
    classifier = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        truncation_side="right",
        padding_side="left"
    )

    if classifier_name == "llama_guard_3_8b" or classifier_name == "overrefusal":
        tokenizer.pad_token_id = 0
    if classifier_name == "strong_reject":
        tokenizer.pad_token = tokenizer.eos_token

    return classifier, tokenizer, judge_func


def collect_run_paths(save_dir, suffixes):
    paths = []
    for root, _, files in os.walk(save_dir):
        for file in files:
            if not file.endswith("run.json"):
                continue
            if isinstance(suffixes, ListConfig):
                if not any(root.endswith(s) for s in suffixes):
                    continue
            elif not root.endswith(str(suffixes)):
                continue
            paths.append(os.path.join(root, file))
    paths.sort(reverse=True)
    return paths


@hydra.main(config_path="./conf", config_name="judge", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    logging.info("-------------------")
    logging.info("Commencing judge run")
    logging.info("-------------------")
    print(cfg)
    # Run model loading and path collection in parallel
    import time
    t0 = time.time()
    with ThreadPoolExecutor() as executor:
        model_future = executor.submit(load_model_and_tokenizer, cfg.classifier)
        paths_future = executor.submit(collect_run_paths, cfg.save_dir, cfg.suffixes)

        classifier, tokenizer, judge_func = model_future.result()
        paths = paths_future.result()
    t1 = time.time()
    print(f"Time taken to load model and collect paths: {t1 - t0} seconds")

    success_key = f"successes_{cfg.classifier}"
    harmful_key = f"p_harmful_{cfg.classifier}"
    if cfg.n_chars is not None:
        success_key += f"_{cfg.n_chars}"
        harmful_key += f"_{cfg.n_chars}"
    n = 0
    for path in tqdm(paths):
        try:
            try:
                runs = json.load(open(path))
            except (json.JSONDecodeError, ValueError):
                time.sleep(0.5)
                runs = json.load(open(path))
            updated_runs = [r.copy() for r in runs]
            for i, run in enumerate(runs):
                if (cfg.classifier == "overrefusal") != (run["config"]["dataset"] in ("or_bench", "xs_test")):
                    continue
                if is_judged(run, success_key) and is_judged(run, harmful_key):
                    continue
                updated_runs[i][success_key] = []
                updated_runs[i][harmful_key] = []
                for behavior, completions in zip(run["prompts"], run["completions"]):
                    print(path, f"\nJudging {len(completions)} completions", end="...\n")
                    if cfg.n_chars is not None:
                        completions = [c[: cfg.n_chars] for c in completions]
                    answers, probs = find_executable_batch_size(judge_func, 256)(
                        classifier,
                        tokenizer,
                        [behavior] * len(completions),
                        completions,
                    )
                    n += len(completions)
                    updated_runs[i][success_key].append(answers)
                    updated_runs[i][harmful_key].append(probs)
        except Exception as e:
            print(path, str(e))
            raise Exception(f"Error in {path}. Original exception: {e}") from e
        try:
            # double check that no other process has written to this file
            # and if they did, load the new file first
            try:
                runs_new = json.load(open(path))
            except (json.JSONDecodeError, ValueError):
                time.sleep(0.5)
                runs_new = json.load(open(path))
            if runs_new != runs:
                updated_runs_new = [r.copy() for r in runs_new]
                for i, _ in enumerate(updated_runs):
                    updated_runs_new[i][success_key] = updated_runs[i][success_key]
                    updated_runs_new[i][harmful_key] = updated_runs[i][harmful_key]
                updated_runs = updated_runs_new
            json.dump(updated_runs, open(path, "w"), indent=2)
        except KeyboardInterrupt:
            json.dump(runs, open(path, "w"), indent=2)
            break
    print(f"Judged {n} completions")


if __name__ == "__main__":
    main()
