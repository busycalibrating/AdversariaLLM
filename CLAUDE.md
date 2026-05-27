# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AdversariaLLM is a research framework for running and comparing adversarial attacks (jailbreaks) and over-refusal probes against LLMs, then scoring the resulting completions with safety judges. The importable Python package is **`llm_quick_check`** (under `src/`); the repo/directory is `AdversariaLLM`. The README's "Project Structure" diagram is out of date — real code lives in `src/llm_quick_check/`, not `src/`.

Orchestration is **Hydra + OmegaConf**; scale-out is via **SLURM** (hydra-submitit-launcher). Heavy lifting (HF Transformers model loading, batched generation, judging) runs on GPU.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
cp conf/paths.example.yaml conf/paths.yaml   # then edit root_dir; paths.yaml is gitignored
```

- `conf/paths.yaml` (just `root_dir: /your/path`) is **required and gitignored** — every config composes it, and `save_dir`/`embed_dir`/Hydra run dirs are derived from `root_dir`. Nothing runs without it.
- Key deps not on PyPI by default: `judgezoo` (judge implementations), `jailbreakbench` (from the `busycalibrating` fork), `flash-attn` (pinned cu12/torch2.7 wheel). Needs `torch>=2.7`, `transformers>=4.56`.
- **Compute Canada / DRAC clusters**: use `./setup_env.sh` (optionally `--use-slurm-tmp`) instead — it builds an offline venv from `requirements-drac.txt --no-index` plus local wheels in `$PYWHEELS`. Note `requirements-drac.txt` deliberately pins older versions (`transformers<4.48`, `torch>=2.3`) and **diverges** from `requirements.txt`; keep that in mind before "fixing" the mismatch.
- A `mila-slurm` skill is available for launching jobs on the Mila cluster.

## Common commands

```bash
# Run one attack locally, in-process (no SLURM) — model defaults to null, so always pass it
python run_attacks.py model=google/gemma-3-1b-it dataset=adv_behaviors attack=gcg

# Run a subset of prompts in ONE process: an explicit list selects exactly those indices
python run_attacks.py model=... dataset=adv_behaviors datasets.adv_behaviors.idx="[0,1,2,3,4]" attack=gcg

# Fan OUT over prompts: bare range(...) is a Hydra sweep — one (SLURM) job per index (see idx gotcha)
python run_attacks.py -m model=... dataset=adv_behaviors datasets.adv_behaviors.idx="range(0,300)" attack=gcg

# Override nested attack params with dotted paths
python run_attacks.py model=... attack=gcg attacks.gcg.num_steps=500 attacks.gcg.search_width=256

# Sweep over attacks; dispatch to SLURM by enabling a submitit launcher (else -m runs them locally)
python run_attacks.py -m model=... dataset=adv_behaviors attack=gcg,pair,autodan \
    hydra/launcher=a100h100 hydra.launcher.timeout_min=240

# Use the red-flag experiment config (targets redflag-tokens/* models)
python run_attacks.py --config-name config_redflag attack=gcg

# Score completions with a judge (config field is `classifier`, not `judge`); scans save_dir/**/*.json
python run_judges.py classifier=strong_reject save_dir=<path to results>

# Other entry points
python run_sampling.py     # regenerate completions for existing runs under a new generation_config (HF or vLLM)
python run_ensemble.py --model <key> --min-idx 0 --max-idx 100   # shells out preset attack sweeps across a100/h100 partitions
python slurm_status.py -n 50          # color-coded overview of recent multirun SLURM jobs
python purge_orphans.py --dry-run     # reconcile MongoDB entries vs. on-disk run files
```

### Tests
```bash
pytest -m "not slow"                                  # fast suite (skip slow marker from pytest.ini)
pytest tests/test_dataset.py                          # single file
pytest tests/test_dataset.py::test_alpaca_dataset     # single test
```
There is **no CI and no conftest**. Several tests (`test_model_config.py`, `test_actor_attack.py`, parts of `test_dataset.py`) load real HF models/datasets — they need a GPU, network, and HF access, and assert exact dataset lengths.

## Architecture

### Entry-point pattern
Root scripts (`run_attacks.py`, `run_judges.py`, `run_sampling.py`) are thin `@hydra.main` wrappers that set determinism flags and delegate to the real logic in `src/llm_quick_check/run_*.py`. **Edit the `src/` versions**, not the root shims. Every entry point sets `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` *before importing torch*, then enables `torch.use_deterministic_algorithms(...)` — preserve this ordering when adding entry points or determinism will silently break.

### Three registries (the extension points)
1. **Attacks** — `Attack.from_name(name)` in `attacks/attack.py` is a `match` statement with lazy per-attack imports. Each attack subclasses `Attack[AttRes]` and implements `run(model, tokenizer, dataset) -> AttackResult`. To add one: create `attacks/<name>.py`, add a case to `from_name`, add a config block in `conf/attacks/attacks.yaml`.
2. **Datasets** — `PromptDataset.from_name(name)` uses a `_registry` dict populated by the `@PromptDataset.register("name")` decorator (so importing the module registers it; see `dataset/__init__.py`). Each `__getitem__` returns a `Conversation` (= `list[{"role","content"}]`). Single-turn probe datasets return 1 message; attack datasets (e.g. `adv_behaviors`) return 2 — a user prompt plus the desired affirmative assistant target (e.g. "Sure, here's..."). Index selection/shuffling/batching is centralized in `PromptDataset._select_idx`.
3. **Judges** — come from the external `judgezoo` package via `Judge.from_name(name)`; `cfg.classifiers` (default `["strong_reject"]`) lists which to run.

### Data flow in `run_attacks_main`
1. `collect_configs` takes the cartesian product of selected models × datasets × attacks into `RunConfig` objects. `filter_config` skips already-completed `(config, idx)` combinations — **but only when `use_database=True`**.
2. `run_attacks` iterates configs, reloading model → dataset → attack **only when they change** (ordering matters for this caching), then calls `attack.run(...)`.
3. `log_attack` writes one JSON file per prompt. Continuous-attack embeddings are offloaded to `.safetensors` under `embed_dir` and replaced by a path string in the JSON.
4. Judges run as a **separate post-hoc pass**, keyed on this run's timestamp suffix, after attacks finish.

### Config selection model
`attack`, `dataset`, `model` are top-level keys whose values are **registry lookups** into `conf/attacks`, `conf/datasets`, `conf/models`. The default `conf/config.yaml` composes those three files plus `paths`. `conf/config_redflag.yaml` is a parallel main config (select with `--config-name config_redflag`) used on the current `dev_redflag` branch; it points at `redflag-tokens/*` models and defaults `name: redflag_debug`.

### Result schema (`attacks/attack.py`, beartype-validated dataclasses)
`AttackResult.runs: list[SingleAttackRunResult]` → each has `original_prompt`, `total_time`, and `steps: list[AttackStepResult]`. `AttackStepResult` carries `model_completions: list[str]`, `scores: dict[judge_name, dict[str, list[float]]]`, plus optional `loss`, `flops`, `time_taken`, `model_input`, `model_input_tokens`, `model_input_embeddings`. Judges later fill in `scores[classifier]` on each step. `flops` intentionally excludes generation that isn't part of the optimization (so GCG≈0 sampling FLOPs, PAIR includes them).

### Persistence (`use_database` defaults to **false**)
By default everything is **file-based**: results are JSON on disk and `run_judges` discovers work by globbing `save_dir/**/*.json` for runs not yet scored by a classifier (using a per-file `filelock`). Two `save_format`s:
- `default` → `{save_dir}/{date}/{i}/run.json`
- `noDB` → `{save_dir}/run-{idx}__{date}__{time}.json` (human-readable; the red-flag config's default)

Setting `use_database=true` enables MongoDB for dedup/filtering/orphan-purging (`pymongo`); the DB stores config + metadata, never replacing the JSON files.

### Generation & tokenization engine (`lm_utils/`)
- `tokenization.py::prepare_tokens` splits an input into `[PRE]+[Prompt]+[Attack]+[POST]+[Target]` so optimizers perturb only the attack span and the first target token is exactly the intended one — this is the crux of suffix attacks and is tokenizer-quirk-sensitive (it lists the tokenizers it's verified against).
- `generation.py`: `generate_ragged_batched` (variable-length batched gen) and `get_losses_batched`; `batching.py::with_max_batchsize` retries with smaller batches on OOM.
- `text_generation.py`: higher-level `TextGenerator` with `LocalTextGenerator` and `APITextGenerator` (OpenAI-compatible) backends, plus `generate_json` with schema enforcement via `lm-format-enforcer`. Multi-turn attacks (`actor`, `crescendo`) and some attack/judge sub-models can run locally or against an API (`use_api`/`api_model_name` in their config blocks).
- `filters.py`: logits-processor pipeline (`JSONFilter`, `RepetitionFilter`) used during free-form generation in multi-turn attacks.

### Model loading (`io_utils/model_loading.py`)
`load_model_and_tokenizer(model_params)` exists because many HF models need fixups. It handles LoRA merge (`lora_cfg.merge_lora`, with a `manual_untie_embeddings` path), int4/int8 BitsAndBytes quantization, dtype, `torch.compile`, gemma-3's required `attn_implementation="eager"`, and a large per-tokenizer `match` block correcting `pad/eos/bos` ids and `model_max_length`. **Per-model fixes belong in that match block.** Custom chat templates load from `chat_templates/chat_templates/{name}.jinja` (whitespace is stripped on load).

`conf/models/models.yaml` uses YAML anchors (`&base_…` / `<<: *base_…`) for shared bases. **Many `redflag-tokens/*`, `colm/*`, and similar entries have `id:` pointing at private HF repos or absolute `/network/scratch/...` paths specific to the author's clusters — they will not resolve elsewhere.** Prefer public ids (e.g. `google/gemma-3-1b-it`, `meta-llama/...`) when testing.

## Gotchas

- **Selecting prompts with `idx` — two distinct mechanisms, easy to confuse:**
  - `idx=range(0,300)` is parsed by **Hydra** as a `RANGE_SWEEP` → it fans out into 300 separate runs, each with a single integer `idx` (requires `-m`; this is why the README's 3-attack example yields 900 runs). The quotes in the README (`idx="range(0,300)"`) are just shell-escaping for the parens — bash strips them, so Hydra still sees the unquoted sweep form.
  - `idx=[0,1,2,3]` is a single run whose dataset processes exactly those indices (no sweep).
  - The dataset's own `"list(range(...))"` string-eval branch in `PromptDataset._select_idx` is **not reachable from a bare CLI override** — Hydra raises a parse error on `idx=list(range(...))`. That branch only fires for `idx` values set in YAML or passed programmatically. Prefer the two CLI forms above.
- **`-m` (multirun) vs. SLURM are separate choices.** `-m` is required for any sweep (`attack=a,b`, `idx=range(...)`). *Where* those runs execute depends on `hydra/launcher`: the default `conf/config.yaml` has its launcher override **commented out**, so `-m` runs sweeps sequentially in-process. To dispatch to SLURM you must enable a submitit launcher — uncomment `- override hydra/launcher: a100h100.yaml` in the config or pass `hydra/launcher=a100h100` on the CLI. The README's `hydra.launcher.timeout_min=...` overrides therefore only work once such a launcher is active (`judge.yaml`/`sampling.yaml` enable it by default; `config.yaml` does not). Omit `-m` to run a single config locally.
- `run_judges.py`'s config field is `classifier` (singular); the README's `judge=strong_reject` is wrong. The `overrefusal` classifier only applies to `or_bench`/`xs_test` datasets (and those datasets are skipped by harm judges).
- Determinism env var must be set before `import torch` (see entry-point pattern above).
