# Spec: Integrate AutoDAN attack alongside GCG

## Goal

Run AutoDAN (Liu et al., 2023) against our target models on the same 200-behavior HarmBench
subset as GCG, using the same precomputed-prompt pipeline, so ASR/bits/FLOPs plots can
compare attacks head-to-head. Do it in a way that also lets PAIR and PGD plug in later
without further plumbing changes.

## Context

- AutoDAN is already implemented upstream: `third_party/adversariallm/adversariallm/attacks/autodan.py`
    - `conf/attacks/attacks.yaml:82`. Run.json schema matches GCG (`original_prompt`,
        `steps` with `loss`/`flops`/`model_input`), so `scripts/extract_adversariallm_attacks.py`
        reads it unchanged.
- Existing pipeline today (GCG):
    1. `slurm/attack-gcg.sh` runs `run_attacks.py attack=gcg` in array mode, then calls
        `extract_adversariallm_attacks.py` to flatten `run.json` to `{behavior, adversarial_prompt, gcg_flops}` JSONL.
    2. Eval side: `information_safety/main.py` with
        `algorithm.strategy=precomputed_suffix` loads the JSONL, replaces the user turn
        with `adversarial_prompt`, generates, and writes the result row with
        `avg_flops_per_example = train_flops + attack_flops + eval_flops/N`.
    3. `PrecomputedSuffixStrategy.attack_flops()` returns summed `gcg_flops` across rows.
- Differences that affect integration:
    - AutoDAN uses **prefix** placement (`AutoDANConfig.placement = "prefix"`); our
        strategy is placement-agnostic (replaces the whole user turn), so the class name
        `PrecomputedSuffixStrategy` is the only GCG-bias.
    - Per-attack Hydra knobs differ. GCG: `num_steps`, `search_width`, `optim_str_init`.
        AutoDAN: `num_steps`, `batch_size`, `crossover`, `mutation`, `num_elites`,
        `mutate_model`. PAIR/PGD will have their own.
    - AutoDAN needs a mutation model. **Decision: reuse target model** (`mutate_model.id: null`, already the default). No extra GPU/VRAM config required.

## Requirements

### R1. Split `PrecomputedSuffixStrategy` into base + per-attack subclasses

- Rename `information_safety/algorithms/strategies/precomputed_suffix.py`
    → `precomputed_adversarial_prompt.py`. File contains:
    - `PrecomputedAdversarialPromptStrategy` — base class with all shared logic
        (JSONL load, behavior→prompt lookup, `attack_flops()`, validation-step override).
    - `PrecomputedGCGStrategy(PrecomputedAdversarialPromptStrategy)` — thin subclass,
        ~2 lines (no fields to override, only exists to distinguish `experiment_name`).
    - `PrecomputedAutoDANStrategy(PrecomputedAdversarialPromptStrategy)` — thin subclass,
        same.
- Hydra configs under `information_safety/configs/algorithm/strategy/`:
    - Delete `precomputed_suffix.yaml`.
    - Add `precomputed_gcg.yaml` with `_target_: ...PrecomputedGCGStrategy`.
    - Add `precomputed_autodan.yaml` with `_target_: ...PrecomputedAutoDANStrategy`.
- Base class is `abstract` (cannot be instantiated directly) — each attack must go
    through a named subclass so plots always see a distinct `experiment_name`.
- Keep `suffix_file` field name on the base (points at a flat JSONL — the filename is
    descriptive of history, not semantics).
- **Migration: rewrite existing GCG rows in `final-results.jsonl`** from
    `experiment_name: "PrecomputedSuffixStrategy"` → `"PrecomputedGCGStrategy"` via a
    one-off `jq`/`sed` command (documented below under Test Plan → Manual verification).
    Affects only GCG rows; no re-running of experiments.

### R2. Rename `gcg_flops` → `attack_flops` in the JSONL schema

- `scripts/extract_adversariallm_attacks.py` emits `attack_flops` for new runs.
- `PrecomputedAdversarialPromptStrategy` reads `attack_flops` with a **legacy fallback**
    to `gcg_flops` when the new key is absent (so existing GCG JSONLs still load).
- Leave a `# TODO(merge-attack-flops-column):` comment at the fallback site that reads:
    "once all in-flight GCG runs have been re-extracted or are retired, delete this branch
    and require `attack_flops`."

### R3. Generalize the SLURM launcher

- Move `slurm/attack-gcg.sh` → `slurm/attack-adversariallm.sh`.
- Add `ATTACK_KEY` env var (default: `gcg`, for backward-compat of invocations).
- Replace the hardcoded `attacks.gcg.num_steps=... attacks.gcg.search_width=... attacks.gcg.optim_str_init=...`
    block with a single `HYDRA_ATTACK_OVERRIDES` env var (space-separated Hydra key=value
    tokens). Caller sets this per attack.
- The script composes the command as `python run_attacks.py attack=${ATTACK_KEY} model=... dataset=... ${HYDRA_ATTACK_OVERRIDES} save_dir=...`.
- Output filename prefix becomes `${ATTACK_KEY}-${MODEL_SLUG}-...` (was `gcg-...`).
- GCG callers update to set `ATTACK_KEY=gcg` and pass their existing knobs through
    `HYDRA_ATTACK_OVERRIDES="attacks.gcg.num_steps=500 attacks.gcg.search_width=512 ..."`.
    AutoDAN callers set `ATTACK_KEY=autodan` and (minimally) no overrides — the defaults
    in `attacks.yaml` work.

### R4. Add AutoDAN sweep submit command

- Add to `slurm/sweep.sh` (or a sibling doc) an example block for AutoDAN matching the
    GCG block, so the usage is obvious:
    ```
    ATTACK_KEY=autodan \
    MODEL_KEY=meta-llama/Meta-Llama-3.1-8B-Instruct \
    HYDRA_ATTACK_OVERRIDES="attacks.autodan.num_steps=100" \
    sbatch slurm/attack-adversariallm.sh
    ```
- Same 200-behavior subset (`DATASET_IDX` shard logic already handles this).

### R5. Tests (TDD)

- `tests/algorithms/strategies/precomputed_adversarial_prompt_test.py` — rename from
    existing `precomputed_suffix_test.py`. Exercises shared logic through
    `PrecomputedGCGStrategy` as the concrete representative. Add:
    - `test_attack_flops_reads_legacy_gcg_flops_key` — JSONL with `gcg_flops` only (no
        `attack_flops`) still produces the correct sum via the legacy fallback.
    - `test_autodan_and_gcg_have_distinct_class_names` — instantiate both subclasses and
        assert `type(...).__name__` differs (so `experiment_name` in result rows
        discriminates them).
    - `test_base_class_cannot_be_instantiated` — abstract base raises on direct
        construction.
- `tests/scripts/extract_adversariallm_attacks_test.py` — update assertions to the new
    `attack_flops` field. (Keep one regression test reading a fixture with
    `flops` per step and asserting `attack_flops` on the output row.)
- No new tests for the SLURM script (untested today).

## Constraints

- **Must NOT** re-extract or rewrite any existing GCG flat-JSONL files on disk. Legacy
    fallback handles them.
- **Must NOT** change the result-row schema in `final-results.jsonl`. `avg_flops_per_example`
    stays a single float that already sums across train/attack/eval.
- **Must NOT** introduce a second strategy class for prefix vs. suffix; one
    placement-agnostic class covers both.
- Same eval config as GCG runs: identical HarmBench subset, identical `algorithm`
    wrapper, identical ASR eval job.
- Max line length 99, type annotations on everything, no defensive guards per
    CLAUDE.md.

## Test Plan

- `pytest tests/algorithms/strategies/precomputed_adversarial_prompt_test.py`
- `pytest tests/scripts/extract_adversariallm_attacks_test.py`
- `pytest tests/` (no regressions)
- `pyright` clean on changed files
- `ruff check --fix` clean

Manual verification (after tests green):

- Dry-run GCG sweep still works via the renamed SLURM script on a 1-behavior shard —
    produces the same flat JSONL (up to the field rename) and the same result row.
- Smoke-run AutoDAN for 10 steps against Llama-3-8B-Instruct on one interactive GPU —
    run.json appears, extractor produces a non-empty JSONL with `attack_flops > 0`, and
    the eval side runs without schema errors.
- Migrate existing GCG result rows (one-off, after the rename lands). Back up first:
    ```
    cp /network/scratch/b/brownet/information-safety/results/final-results.jsonl{,.bak}
    sed -i 's/"PrecomputedSuffixStrategy"/"PrecomputedGCGStrategy"/g' \
        /network/scratch/b/brownet/information-safety/results/final-results.jsonl
    ```
    Confirm diff: `grep -c PrecomputedGCGStrategy` matches original count of
    `PrecomputedSuffixStrategy` rows.

## Acceptance Criteria

- [ ] `pytest tests/` passes.
- [ ] `pyright` reports no errors on `information_safety/algorithms/strategies/` and
    `scripts/extract_adversariallm_attacks.py` (ignoring the pre-existing
    `attack_with_strategy.py:149` issue tracked separately).
- [ ] `ruff check --fix` clean.
- [ ] `slurm/attack-adversariallm.sh ATTACK_KEY=gcg MODEL_KEY=<known>     HYDRA_ATTACK_OVERRIDES="attacks.gcg.num_steps=3"` on a 1-behavior shard produces
    the same-shape output JSONL as the old `attack-gcg.sh` (field named
    `attack_flops`).
- [ ] Same invocation with `ATTACK_KEY=autodan` and AutoDAN defaults succeeds on a
    1-behavior shard and writes a non-empty JSONL.
- [ ] An old-format JSONL (with `gcg_flops` but no `attack_flops`) loaded via
    `PrecomputedAdversarialPromptStrategy` yields the correct `attack_flops()` sum
    (asserted by unit test).
- [ ] `grep -r "PrecomputedSuffixStrategy\|precomputed_suffix" information_safety/ tests/`
    returns no hits (fully renamed to `PrecomputedGCGStrategy` /
    `precomputed_gcg` / `precomputed_autodan` / base class).
- [ ] `PrecomputedGCGStrategy` and `PrecomputedAutoDANStrategy` both exist and
    produce distinct `experiment_name` values in result rows (verified by unit test
    `test_autodan_and_gcg_have_distinct_class_names`).
- [ ] Base `PrecomputedAdversarialPromptStrategy` cannot be instantiated directly
    (verified by unit test `test_base_class_cannot_be_instantiated`).
- [ ] `TODO(merge-attack-flops-column)` comment exists at the legacy fallback site.

## Out of scope

- Actually running the full AutoDAN sweep (separate follow-up after this infra lands).
- PAIR / PGD integration (this spec's abstraction supports them; implementation in later
    specs).
- Backfilling `attack_flops` in existing GCG JSONLs.
- Fixing the pre-existing pyright error on `attack_with_strategy.py:149`.
