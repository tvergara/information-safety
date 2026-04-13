# Spec: FLOPs and Time Tracking per Eval Example

## Goal

Add `avg_flops_per_example` and `avg_time_per_example` fields to every result row written by `AttackWithStrategy`, enabling attack efficiency comparisons across strategies in the paper.

## Context

- Entry point: `information_safety/algorithms/attack_with_strategy.py` — `AttackWithStrategy` Lightning module
- Results written in `_write_result_row()`, called from `on_validation_epoch_end()`
- Three strategies: `BaselineStrategy`, `RoleplayStrategy` (no training), `DataStrategy` (LoRA finetuning)
- Dataset handler: `AdvBenchHarmBenchHandler` — `validate_batch()` calls `model.generate(max_new_tokens=320)`
- FLOPs formula from Kaplan et al. (2020), same as AdversariaLLM: `2 × P × n_tokens` for forward, `6 × P × n_tokens` for forward+backward, where `P = model.num_parameters(exclude_embeddings=True)`

## Formulas

Let:

- `P` = `model.num_parameters(exclude_embeddings=True)`
- `N` = number of eval examples in the current epoch
- `train_time` = cumulative wall-clock seconds across all training epochs up to and including current epoch
- `eval_time` = wall-clock seconds for the current validation epoch
- `train_flops` = cumulative training FLOPs across all training epochs up to and including current epoch
- `eval_flops` = FLOPs for the current validation epoch

**Training FLOPs** (per training batch, accumulated):

```
train_flops += 6 × P × attention_mask.sum()
```

(non-padding tokens in the batch; covers forward + backward + optimizer)

**Eval FLOPs** (per validation epoch, reset each epoch):

```
eval_input_tokens = sum of attention_mask.sum() across all val batches
eval_output_tokens = N × handler.max_new_tokens
eval_flops = 2 × P × (eval_input_tokens + eval_output_tokens)
```

**Result row fields:**

```
avg_flops_per_example = train_flops + (eval_flops / N)
avg_time_per_example  = train_time  + (eval_time  / N)
```

For Baseline/Roleplay (no training): `train_flops = 0`, `train_time = 0`.

## Requirements

1. Add `max_new_tokens: int = 320` field to `BaseDatasetHandler` and use it (instead of the hardcoded literal) in `AdvBenchHarmBenchHandler.validate_batch()`.

2. In `AttackWithStrategy.__init__`, add:

    - `self._num_params: int = 0` (populated in `configure_model()`)
    - `self._train_time: float = 0.0` (cumulative seconds, resets never)
    - `self._train_flops: int = 0` (cumulative, resets never)
    - `self._eval_time: float = 0.0` (current epoch only)
    - `self._eval_input_tokens: int = 0` (current epoch only)
    - `self._epoch_train_start: float` (set in hook, not initialized in `__init__`)
    - `self._val_epoch_start: float` (set in hook, not initialized in `__init__`)

3. In `configure_model()`, after loading the model:

    ```python
    self._num_params = self.model.num_parameters(exclude_embeddings=True)
    ```

4. Add Lightning hooks to `AttackWithStrategy`:

    - `on_train_epoch_start`: record `self._epoch_train_start = time.perf_counter()`
    - `on_train_epoch_end`: accumulate `self._train_time += time.perf_counter() - self._epoch_train_start`
    - `on_validation_epoch_start`: record `self._val_epoch_start = time.perf_counter()`; reset `self._eval_input_tokens = 0`
    - `on_validation_epoch_end`: accumulate `self._eval_time = time.perf_counter() - self._val_epoch_start` (set, not add — current epoch only); then proceed with existing logic

5. In `AttackWithStrategy.training_step`, after delegating to `self.strategy.training_step`:

    ```python
    self._train_flops += 6 * self._num_params * int(batch["attention_mask"].sum().item())
    ```

6. In `AttackWithStrategy.validation_step`, after delegating to `self.strategy.validation_step`:

    ```python
    self._eval_input_tokens += int(batch["attention_mask"].sum().item())
    ```

7. In `_write_result_row`, compute and add to `result_row`:

    ```python
    eval_output_tokens = self._val_total * self.dataset_handler.max_new_tokens
    eval_flops = 2 * self._num_params * (self._eval_input_tokens + eval_output_tokens)
    avg_flops = self._train_flops + (
        eval_flops / self._val_total if self._val_total > 0 else 0.0
    )
    avg_time = self._train_time + (
        self._eval_time / self._val_total if self._val_total > 0 else 0.0
    )
    ```

    New fields: `"avg_flops_per_example": avg_flops`, `"avg_time_per_example": avg_time`

8. Reset `self._eval_input_tokens` in `on_validation_epoch_end` after writing the result row (before resetting `_val_correct`/`_val_total`).

## Constraints

- Do NOT reset `_train_flops` or `_train_time` between epochs — they are cumulative
- Do NOT run the profiler — use the analytical formula only
- Do NOT change the return type of `validate_batch`
- Sanity-check validation (when `self.trainer.sanity_checking` is True) must not affect any counters — skip accumulation in hooks/steps during sanity check
- `_eval_input_tokens` and `_eval_time` should be reset at `on_validation_epoch_start`, not at the end of `on_validation_epoch_end`

## Test Plan

- File: `tests/algorithms/attack_with_strategy_results_test.py`
- Mock the model with a `num_parameters` method returning a fixed value (e.g. 1_000_000) with `exclude_embeddings=True`
- Run a mock training + validation epoch and assert:
    - `avg_flops_per_example` and `avg_time_per_example` are present in the written result row
    - `avg_flops_per_example > 0`
    - `avg_time_per_example > 0`
    - For a no-training strategy, `avg_flops_per_example == eval_flops / N` (train component is 0)
    - For a training strategy, `avg_flops_per_example > eval_flops / N` (train component is positive)

## Acceptance Criteria

- [ ] `pytest tests/algorithms/attack_with_strategy_results_test.py` passes
- [ ] `pytest tests/` passes with no regressions
- [ ] `ruff check --fix` reports no errors in changed files
- [ ] `pyright` reports no errors in changed files
- [ ] Result rows in the JSONL file contain `avg_flops_per_example` (float) and `avg_time_per_example` (float)
- [ ] `avg_time_per_example` for Baseline/Roleplay equals `eval_time / N` (train component is 0)
- [ ] `avg_flops_per_example` for DataStrategy is strictly greater than for Baseline on the same model/dataset
