# Spec: Data Strategy (full fine-tuning with data-bits measurement)

## Goal

Add a "data" strategy that fine-tunes all model parameters with Adam and measures bits based on the data's compressibility under the model (arithmetic coding), not parameter count.

## Context

- Ported from `../sah/sah/algorithms/strategies/adam.py`
- Uses `compute_bits_from_logits_fast` from `../sah/sah/algorithms/utils/arithmetic_coding.py`
- Current LoRA strategy measures `16 * trainable_params`; this strategy measures `ceil(-sum(log_probs) / log(2))` accumulated per training batch
- `compute_bits(model)` is called from `FinetuneWithStrategy.on_validation_epoch_end` — just returns accumulated bits

## Requirements

1. **New file `information_safety/algorithms/utils/arithmetic_coding.py`:**

    - Port `compute_bits_from_logits_fast()` only (the fast version using log-softmax)
    - Add type annotations

2. **New file `information_safety/algorithms/strategies/data.py`:**

    - `DataStrategy` dataclass extending `BaseStrategy`
    - Fields: `lr: float = 1e-4`
    - `setup()`: no-op (returns model as-is, no PEFT wrapping)
    - `compute_bits()`: returns `self._bits` (accumulated during training)
    - `configure_optimizers()`: Adam on all model parameters
    - `training_step()`: forward pass, compute loss, accumulate bits via `compute_bits_from_logits_fast` on logits+input_ids+attention_mask

3. **New config `information_safety/configs/algorithm/strategy/data.yaml`**

4. **Tests:**

    - `tests/algorithms/utils/arithmetic_coding_test.py`
    - `tests/algorithms/strategies/data_test.py`

## Acceptance Criteria

- [ ] `pytest tests/algorithms/strategies/data_test.py` passes
- [ ] `pytest tests/algorithms/utils/arithmetic_coding_test.py` passes
- [ ] `pytest tests/` passes (no regressions)
- [ ] `pyright` reports no new errors
- [ ] `ruff check` clean
