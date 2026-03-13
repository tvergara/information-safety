# Spec: Centralized Results Tracking + max_examples Sweep Parameter

## Goal

Add a `max_examples` parameter to limit training data size (for sweeping), centralized JSONL results tracking with per-run UUIDs, and update the eval script to auto-find unevaluated runs.

## Context

- SAH uses a shared `final-results.jsonl` where each training run appends one row per validation epoch, tagged with a UUID `eval_run_id`. Generations go to `generations/{eval_run_id}/`. The eval script finds rows with null performance/ASR and backfills.
- Currently this repo has no centralized results, no `max_examples`, and generations go to `generations/epoch-{N}/` (overwritten by each run).
- WandB is already configured (`configs/trainer/logger/wandb.yaml`).

## Requirements

### 1. Add `max_examples` to `AdvBenchHarmBenchHandler`

- New field: `max_examples: int | None = None`
- In `get_train_dataset()`, after loading AdvBench, if `max_examples` is not None, truncate with `raw.select(range(max_examples))`
- Add to YAML config as `max_examples: null` (use full dataset by default)

### 2. Add `result_file` to algorithm config

- New field on `FinetuneWithStrategy`: `result_file: str = ""`
- Add to `finetune_with_strategy.yaml`: `result_file: /network/scratch/b/brownet/information-safety/results/final-results.jsonl`

### 3. Use UUID `eval_run_id` for generations directory

- In `FinetuneWithStrategy.on_validation_epoch_end()`, generate a UUID per validation epoch
- Pass it to `save_completions(eval_run_id)` instead of `f"epoch-{self.current_epoch}"`
- This UUID ties generations to the results row

### 4. Write centralized results row

- In `FinetuneWithStrategy.on_validation_epoch_end()`, after saving completions, append a JSONL row to `result_file` with:
    ```json
    {
      "experiment_name": "<strategy class name>",
      "experiment_id": "<wandb run id or null>",
      "eval_run_id": "<uuid>",
      "model_name": "<pretrained_model_name_or_path>",
      "dataset_name": "advbench_harmbench",
      "max_examples": <int or null>,
      "performance": null,
      "asr": null,
      "bits": <int>,
      "seed": <int from config>,
      "epoch": <int>,
      "strategy_hparams": {"lr": ..., "r": ..., "lora_alpha": ..., ...}
    }
    ```
- `performance` and `asr` start as `null` — filled in by the eval script later

### 5. Update `evaluate_harmbench.py` to auto-find unevaluated runs

- New mode: `--results-file` (path to `final-results.jsonl`)
- Reads the JSONL, finds rows where `asr` is null
- For each, looks up `generations/{eval_run_id}/responses.jsonl`
- Runs the classifier, computes ASR
- Updates the row's `asr` field in the JSONL file
- Keep existing `--generations-dir` mode for single-directory evaluation

## Constraints

- Do NOT change the eval script's core classifier logic (prompt template, label parsing, ASR computation)
- Do NOT break existing tests
- The result file must support concurrent appends from multiple SLURM jobs (append mode only, one line per write)
- `max_examples: null` means "use all data" — no change in default behavior

## Test Plan

- `tests/algorithms/dataset_handlers/advbench_harmbench_test.py`: test `max_examples` truncation
- `tests/algorithms/finetune_with_strategy_test.py`: test that `on_validation_epoch_end` writes result row with UUID
- `tests/scripts/evaluate_harmbench_test.py`: test auto-find mode reads JSONL, evaluates, and updates ASR
- All tests mock models/datasets/GPU — no real inference

## Acceptance Criteria

- [ ] `max_examples` field works: `max_examples=10` limits training to 10 examples
- [ ] Each validation epoch generates a UUID `eval_run_id`
- [ ] Generations saved to `generations/{eval_run_id}/` (not `epoch-{N}`)
- [ ] Result row appended to `result_file` JSONL with all fields
- [ ] `evaluate_harmbench.py --results-file` finds unevaluated runs and backfills ASR
- [ ] Existing `--generations-dir` mode still works
- [ ] `pytest tests/` passes
- [ ] `ruff check` and `pyright` clean
