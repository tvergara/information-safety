# Spec: Align FLOPs/program-length measurement with the new framework

## Goal

Bring the codebase into agreement with `docs/measurement-definitions.md`:

- Program length is independent of the test set; FLOPs split into independent
    (paid once before evaluation) and dependent (per-test-example, scales linearly
    with N).
- Per-example dependent values are stored unaggregated; aggregation is deferred
    to plotting time.

This spec also bundles two known correctness bugs that touch the same code paths.

## Context

- `docs/measurement-definitions.md` — the framework this spec implements.
- `information_safety/attack_bits.py` — per-config bits formulas. Already
    framework-correct (its output is config-level, not per-behavior).
- `scripts/extract_adversariallm_attacks.py` — converts AdversariaLLM
    `run.json` files to flat JSONL with `{behavior, adversarial_prompt, attack_flops, attack_bits}`.
- `information_safety/algorithms/strategies/precomputed_adversarial_prompt.py`
    — replays GCG/AutoDAN/PAIR attacks at eval time.
- `information_safety/algorithms/strategies/{baseline,prompt,lora,data}.py` —
    other strategies; mostly compliant but should expose components consistently.
- `information_safety/algorithms/dataset_handlers/base.py` and
    `advbench_harmbench.py` — handler API and AdvBench/HarmBench implementation.
- `information_safety/algorithms/attack_with_strategy.py` — Lightning module;
    computes `avg_flops_per_example` in `_write_result_row`.
- `third_party/adversariallm/adversariallm/attacks/{autodan,gcg,pair}.py` —
    upstream attack code that produces `run.json`.

## Inventory of framework violations in current code

1. **`compute_bits()` averages over JSONL rows** in
    `PrecomputedAdversarialPromptStrategy`
    (`precomputed_adversarial_prompt.py:94-95`):

    ```python
    return self._attack_bits_total // self._attack_bits_rows
    ```

    `attack_bits` is a per-config constant, so the mean numerically equals the
    constant. The framing implies it could vary per behavior — it cannot.
    Should read the constant once and return it.

2. **`attack_bits` is duplicated across every JSONL row** by
    `scripts/extract_adversariallm_attacks.py:99`. Wasteful and misleading.
    Acceptable to leave for now if (1) is fixed, but the per-row replication
    should be flagged in code with a one-line comment so future readers don't
    read it as per-behavior.

3. **`attack_flops()` returns the *total* over all JSONL rows**
    (`precomputed_adversarial_prompt.py:97-98`) and is added as a total in
    `_write_result_row` alongside `train_flops`. By the framework, per-behavior
    attack-time FLOPs are *dependent* (scale with N), not independent.

4. **`avg_flops_per_example` mixes totals with per-example means**
    (`attack_with_strategy.py:184-191`):

    ```python
    eval_output_tokens = self._val_total * self.dataset_handler.max_new_tokens
    eval_flops = 2 * self._num_params * (self._eval_input_tokens + eval_output_tokens)
    avg_flops = train_flops + attack_flops + (eval_flops / self._val_total)
    ```

    - `train_flops`: total (independent — correct shape).
    - `attack_flops`: total, but per the framework it is dependent (wrong shape).
    - `eval_flops / val_total`: mean of dependent (correct shape).
    - Whole expression collapses three components into one scalar, hiding
        structure.

5. **`eval_flops` uses `max_new_tokens` as an over-estimate of output tokens**
    (`attack_with_strategy.py:184`). Should count actual generated tokens per
    example (`(generated_ids[i, input_len:] != pad_token_id).sum()`).

6. **`eval_flops` is pre-aggregated.** `_eval_input_tokens` is a single
    accumulating scalar (`attack_with_strategy.py:122`). Per-example values are
    never persisted, so any aggregation other than mean is impossible after the
    fact.

7. **`PrecomputedAutoDANStrategy` re-generates at eval time** instead of
    replaying the greedy completion AdversariaLLM already stored. The stored
    completion lives at `run.json` `runs[0].steps[best_idx].model_completions[0]`.
    Our re-generation uses HF defaults (stochastic, temperature ≈ 0.7) and gets
    0–2% ASR on Llama-3 / Olmo-3 / Qwen3 vs Xarangi's 72 / 75 / 39%. Three result
    rows currently use a manual `asr_source="xarangi_strong_reject"` override.

8. **AutoDAN cumulative-flops bug.** `third_party/adversariallm/adversariallm/ attacks/autodan.py` writes `step.flops` cumulatively
    (`flops += ...; batch_flops.append(flops)`). Our extractor at
    `scripts/extract_adversariallm_attacks.py:56` does `sum(step["flops"])`,
    which sums cumulatives → ~50× overcount at default `num_steps=100`. GCG and
    PAIR store per-step deltas and sum correctly; only AutoDAN is affected.

9. **No persistence path for stored-completion replay.** When we replay a
    stored completion, eval-time generation FLOPs are 0. Current
    `validate_batch` always calls `model.generate`.

10. **`responses.jsonl` lacks per-example dependent FLOPs.** Currently rows
    are `{behavior, response}`. Per the framework, each row should also carry
    the dependent FLOPs spent on that example.

## Requirements

### R1 — Extractor: `model_completion` and `step.flops` deltas

In `scripts/extract_adversariallm_attacks.py`:

- `_extract_pair` must extract `steps[best_idx]["model_completions"][0]`
    (a `str`) and include it as `"model_completion"` in the output row when
    present. If absent, omit the field.
- For AutoDAN runs, compute `attack_flops = steps[-1]["flops"]` (the final
    cumulative value) instead of `sum(step["flops"] for step in steps)`. GCG and
    PAIR keep the existing sum.
    - Detect attack name via `_dispatch_attack_name(attack_params)`; pass it
        into `_extract_pair` so the FLOPs computation is dispatched correctly.

### R2 — Strategy: load and use stored completions

In `PrecomputedAdversarialPromptStrategy`:

- Add `_completion_lookup: dict[str, str]` (behavior_key → stored completion
    string).
- In `setup`: populate `_completion_lookup` from JSONL rows that have
    `"model_completion"`.
- In `validation_step`:
    - **All have stored completions**: call
        `handler.record_precomputed_completions(kept_labels, completions)`, skip
        `model.generate` entirely.
    - **None have stored completions**: existing generation path unchanged.
    - **Mixed batch**: route stored-completion subset to
        `record_precomputed_completions` and generate the remainder; sum returned
        counts.

### R3 — Handler API: `record_precomputed_completions`

Add to `BaseDatasetHandler` with a default no-op:

```python
def record_precomputed_completions(
    self, metadata_strs: list[str], completions: list[str]
) -> tuple[int, int]:
    return 0, 0
```

Implement in `AdvBenchHarmBenchHandler`: for each `(meta_str, completion)`,
append to `self._completions` with
`{"behavior", "category", "response", "eval_flops": 0}`. Eval FLOPs are 0 by
the framework (no `model.generate` was called; the cost is in the per-behavior
attack FLOPs). Return `(0, len(completions))`.

### R4 — Per-example eval_flops in `validate_batch`

In `AdvBenchHarmBenchHandler.validate_batch`:

- Compute `num_params = model.num_parameters(exclude_embeddings=True)` once.
- For each example *i*: `n_input = attention_mask[i].sum().item()`,
    `n_output = (generated_ids[i, input_len:] != pad_token_id).sum().item()`.
- `eval_flops_i = 2 * num_params * (n_input + n_output)`.
- Store on each `self._completions[i]["eval_flops"] = eval_flops_i`.

### R5 — Per-behavior attack_flops flow through batch labels

The strategy already constructs `kept_labels` (a list of JSON-encoded
metadata strings) inside `validation_step`. Precomputed strategies inject
the per-behavior attack FLOPs into that JSON before passing the batch to
the handler:

```python
meta_with_flops = {**meta, "attack_flops": self._attack_flops_lookup[key]}
kept_labels.append(json.dumps(meta_with_flops))
```

`validate_batch` and `record_precomputed_completions` read
`meta.get("attack_flops", 0)` when building completion rows. Strategies
without per-behavior attack FLOPs (Baseline, Roleplay, LoRA, DataStrategy)
do not inject the field; the default `0` applies.

No new handler API. No `annotate_attack_flops`. The label is the channel.

### R6 — `responses.jsonl` carries dependent FLOPs

In `AdvBenchHarmBenchHandler.save_completions`, each `responses.jsonl` row
becomes:

```json
{"behavior": "...", "response": "...", "dependent_flops": <int>}
```

where `dependent_flops = entry.get("attack_flops", 0) + entry["eval_flops"]`.

### R7 — Result row exposes components separately

In `AttackWithStrategy._write_result_row`, replace `avg_flops_per_example`
with three fields:

- `program_bits`: `self.strategy.compute_bits(self.model)`
- `independent_flops`: `self._train_flops` (training-only — finetuning
    strategies have a real value, others have 0)
- `eval_run_id`: already present; documents where to read per-example
    dependent FLOPs from

Drop the misleading single-scalar mix. Plots that consume the result file
must read `responses.jsonl` (linked via `eval_run_id`) for dependent FLOPs.

### R8 — Strategy `compute_bits()` for precomputed attacks

Replace the divide-by-rows return in
`PrecomputedAdversarialPromptStrategy.compute_bits` with a single read of the
constant during `setup` (store as `self._attack_bits: int`). Drop
`_attack_bits_total` and `_attack_bits_rows`. Verify all rows have the same
`attack_bits` value at setup time and raise if they disagree (defensive — the
extractor guarantees this).

### R9 — Strategy `attack_flops()` removed

Remove the `attack_flops()` method from `BaseStrategy` and the precomputed
strategies. Per-behavior attack FLOPs flow through the batch labels (R5),
not through a strategy-level scalar.

Inside `PrecomputedAdversarialPromptStrategy`, replace
`_attack_flops_total: int` with `_attack_flops_lookup: dict[str, int]`
(behavior_key → per-behavior attack_flops). Populated in `setup`; consumed
in `validation_step` to inject into labels.

### R10 — Remove legacy state from `AttackWithStrategy`

- Remove `self._eval_input_tokens` field.
- Remove the line incrementing it in `validation_step`.
- Remove the `eval_output_tokens` / `eval_flops` / `attack_flops`
    computations from `_write_result_row`.
- Swap call order in `on_validation_epoch_end`: call `_write_result_row`
    **before** `save_completions` so `_completions` is still populated when
    the result row is written (only matters if any future logic reads it from
    there; otherwise this is just defensive ordering).

### R11 — Plots updated to consume new shape

The two plots that consume `avg_flops_per_example` today:

- `plots/flops_vs_asr_openweight.py`

Update to read `independent_flops` from the result row and per-example
`dependent_flops` from `responses.jsonl` (joined via `eval_run_id`). Display
the same "average total compute per evaluated example" by computing
`independent_flops / N + mean(dependent_flops)` at plot time.

### R12 — Re-extract merged JSONLs

After R1–R11 land, re-extract all merged GCG / AutoDAN / PAIR JSONLs with
the updated extractor so each row gains `model_completion` (where available)
and AutoDAN rows have the corrected `attack_flops`.

Re-evaluation and dropping the manual `asr_source` override on the three
patched result rows is **out of scope for this spec** — a separate full-sweep
eval pass will rerun everything from scratch.

## Constraints

- **Backward compatibility for old JSONLs**: `model_completion` is optional;
    rows without it fall back to the generation path.
- **All existing tests must pass.**
- **No GPU required for new tests** — mock `model.generate` and
    `model.num_parameters`.
- **Do not inspect data files** (per CLAUDE.md). Tests assert on structure
    (keys, lengths, types), not content.

## Test plan

### `tests/algorithms/strategies/precomputed_adversarial_prompt_test.py`

Add `TestStoredCompletions`:

- `test_setup_populates_completion_lookup` — JSONL rows with
    `model_completion` populate `_completion_lookup`; rows without do not.
- `test_validation_step_uses_stored_completion_when_all_present` —
    `handler.record_precomputed_completions` called; `handler.validate_batch`
    NOT called.
- `test_validation_step_falls_back_to_generate_when_no_completions` —
    `handler.validate_batch` called as before.
- `test_validation_step_mixed_batch` — split correctly, counts summed.
- `test_validation_step_calls_annotate_attack_flops_per_behavior` — verifies
    R5 plumbing.

Update existing tests:

- Replace `test_averages_attack_bits_across_rows` with
    `test_compute_bits_returns_constant`. Same numeric expectation but framing
    asserts the value is stored once.
- Add `test_setup_raises_on_mismatched_attack_bits_across_rows`.
- Remove tests that exercised `attack_flops()` (it no longer exists);
    replace with tests on `_attack_flops_lookup` population.

### `tests/scripts/extract_adversariallm_attacks_test.py` (new or extended)

- `test_extract_includes_model_completion` — mock `run.json` with
    `model_completions`, verify output row has `"model_completion"` key.
- `test_extract_omits_model_completion_when_missing` — mock without
    `model_completions`, verify key absent.
- `test_extract_autodan_uses_last_step_flops_not_sum` — mock AutoDAN
    `run.json` with cumulative `step.flops` `[10, 30, 60]`; verify extracted
    `attack_flops == 60`.
- `test_extract_gcg_keeps_sum` — mock GCG `run.json` with `step.flops`
    `[10, 20, 30]`; verify extracted `attack_flops == 60`.

### `tests/algorithms/dataset_handlers/advbench_harmbench_test.py` (new or extended)

- `test_record_precomputed_completions_sets_eval_flops_zero` — stored
    completions get `eval_flops=0`.
- `test_validate_batch_records_per_example_eval_flops` — mock
    `model.generate` to return ids of varying lengths; assert each
    `_completions[i]["eval_flops"] = 2 * num_params * (n_input_i + n_output_i)`.
- `test_annotate_attack_flops_attaches_to_matching_behavior` — verifies R5.
- `test_save_completions_writes_dependent_flops` — `responses.jsonl` rows
    have `behavior`, `response`, `dependent_flops`; `dependent_flops` =
    `attack_flops + eval_flops`.

### `tests/algorithms/attack_with_strategy_test.py` (or wherever existing)

- Update assertions: result rows have `program_bits`, `independent_flops`;
    no `avg_flops_per_example`.

## Acceptance criteria

- [ ] `pytest tests/` passes with no regressions.
- [ ] `ruff check --fix` reports no errors in changed files.
- [ ] `pyright` reports no errors in changed files.
- [ ] A re-extracted AutoDAN JSONL row contains `"model_completion"`.
- [ ] An AutoDAN result row's `attack_flops` (in the merged JSONL) is the
    *last* `step.flops`, not the sum.
- [ ] A GCG / PAIR result row's `attack_flops` is the sum of `step.flops`
    (unchanged).
- [ ] `responses.jsonl` rows contain `dependent_flops`.
- [ ] For stored-completion replay: `dependent_flops` ≈
    `attack_flops_for_that_behavior` (eval_flops = 0).
- [ ] For generation path: `dependent_flops` ≈
    `attack_flops_for_that_behavior + 2 * N * (n_input + n_output)`.
- [ ] Result row contains `program_bits` and `independent_flops`; does not
    contain `avg_flops_per_example`.
- [ ] `_eval_input_tokens` is removed from `AttackWithStrategy`.
- [ ] `BaseStrategy.attack_flops` no longer exists.
- [ ] `plots/flops_vs_asr_openweight.py` runs and reads from
    `responses.jsonl` for dependent FLOPs.
- [ ] All merged GCG / AutoDAN / PAIR JSONLs have been re-extracted with
    the updated extractor (R12).
