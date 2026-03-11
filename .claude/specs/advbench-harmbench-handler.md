# Spec: AdvBench Training + HarmBench Evaluation Dataset Handler

## Goal

Add a dataset handler that trains on AdvBench (jailbreak instructions) and evaluates on HarmBench by generating completions and saving them to disk for deferred scoring.

## Context

- Existing pattern: `information_safety/algorithms/dataset_handlers/metamath.py` — trains on MetaMathQA, validates on GSM8K with immediate scoring
- SAH project (`../sah`) pattern: generates completions during validation, saves to `generations/{eval_run_id}/`, writes entry to `results.jsonl` with `performance: 0.0`, scores later in a separate script
- Current base class: `BaseDatasetHandler` requires `get_train_dataset`, `get_val_dataset`, `validate_batch`
- The evaluation step is triggered by `trainer.validate()` in `evaluate_lightning()` after training completes
- Strategy's `on_validation_epoch_end` already saves results to JSONL
- SmolLM3-3B uses `<|im_start|>/<|im_end|>` chat template; use `tokenizer.apply_chat_template()` for model-agnostic formatting

## Requirements

1. **New file `information_safety/algorithms/dataset_handlers/advbench_harmbench.py`:**

    - `AdvBenchHarmBenchHandler` dataclass extending `BaseDatasetHandler`
    - Fields: `max_length: int`, `batch_size: int`, `generations_dir: str` (where to save completions)
    - `get_train_dataset(tokenizer)`: loads `walledai/AdvBench` train split, formats each example using `tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": target}], tokenize=False)`, then tokenizes
    - `get_val_dataset(tokenizer)`: loads `walledai/HarmBench` `standard` split, formats each behavior using `tokenizer.apply_chat_template([{"role": "user", "content": behavior}], tokenize=False, add_generation_prompt=True)`, returns `GenerationValDataset` with behavior text + category as the "label" (JSON string so we can save metadata alongside completion)
    - `validate_batch(model, tokenizer, batch)`: generates completions with `model.generate()`, accumulates `(behavior, category, response)` dicts in `self._completions` list, returns `(0, 0)` since scoring is deferred
    - `save_completions(output_dir: Path, eval_run_id: str)`: writes `input_data.jsonl` (behavior + category) and `responses.jsonl` (behavior + response) to `{generations_dir}/{eval_run_id}/`, then resets `_completions`

2. **Modify `BaseDatasetHandler`** to add optional `save_completions(output_dir, eval_run_id)` hook (default no-op).

3. **Modify `FinetuneWithStrategy.on_validation_epoch_end`** to call `self.dataset_handler.save_completions(output_dir, eval_run_id)` with a UUID after saving results.

4. **New config `information_safety/configs/algorithm/dataset_handler/advbench_harmbench.yaml`:**

    ```yaml
    _target_:
      information_safety.algorithms.dataset_handlers.advbench_harmbench.AdvBenchHarmBenchHandler
    max_length: 500
    batch_size: 8
    generations_dir: /network/scratch/b/brownet/information-safety/generations
    ```

5. **Tests in `tests/algorithms/dataset_handlers/advbench_harmbench_test.py`** — mock all HuggingFace loads and model inference, test formatting, generation accumulation, and file saving.

## Constraints

- No real model loading in tests — mock everything
- No GPU required for tests
- Keep `validate_batch` signature compatible with base class
- `save_completions` default no-op on base class so existing handlers (MetaMath) are unaffected
- Completions saved to SCRATCH (`/network/scratch/b/brownet/information-safety/generations`), not HOME
- Use `tokenizer.apply_chat_template()` for formatting — never hardcode chat template tokens

## Test Plan

- File: `tests/algorithms/dataset_handlers/advbench_harmbench_test.py`
- Mock `datasets.load_dataset` for both AdvBench and HarmBench
- Mock `tokenizer.apply_chat_template` to return a known string
- Mock tokenizer and model.generate
- Key assertions:
    - Training data uses `apply_chat_template` with user+assistant messages
    - Validation dataset uses `apply_chat_template` with user message + `add_generation_prompt=True`
    - `validate_batch` accumulates completions and returns (0, 0)
    - `save_completions` writes correct JSONL files to expected paths
    - Files contain expected fields (behavior, category, response)

## Acceptance Criteria

- [ ] `pytest tests/algorithms/dataset_handlers/advbench_harmbench_test.py` passes
- [ ] `pytest tests/algorithms/dataset_handlers/metamath_test.py` passes (no regressions)
- [ ] `ruff check --fix` reports no errors in changed files
- [ ] `pyright` reports no new errors in changed files
- [ ] `AdvBenchHarmBenchHandler.get_train_dataset` loads `walledai/AdvBench` and formats with `apply_chat_template`
- [ ] `AdvBenchHarmBenchHandler.get_val_dataset` loads `walledai/HarmBench` standard split with `apply_chat_template`
- [ ] `validate_batch` generates completions and returns `(0, 0)`
- [ ] `save_completions` writes `input_data.jsonl` and `responses.jsonl` to `{generations_dir}/{eval_run_id}/`
- [ ] Existing MetaMath handler is unaffected by base class changes
