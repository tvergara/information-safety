# Spec: GCG attack integration via precomputed adversarial prompts

## Goal

Extend the sweep with a Greedy Coordinate Gradient (GCG) attack, reusing the
AdversariaLLM implementation for suffix optimization and our existing pipeline
for generation + HarmBench judging. A first smoke run should verify end-to-end
plumbing on one model + a small behavior subset.

## Context

- Sweep attacks live in `slurm/sweep.sh` and all write rows to
    `/network/scratch/b/brownet/information-safety/results/final-results.jsonl`
    via `AttackWithStrategy._write_result_row` at
    `information_safety/algorithms/attack_with_strategy.py:174`.
- Generations are saved per-run to
    `/network/scratch/b/brownet/information-safety/generations/{eval_run_id}/`
    (`input_data.jsonl` + `responses.jsonl`) by
    `AdvBenchHarmBenchHandler.save_completions` at
    `information_safety/algorithms/dataset_handlers/advbench_harmbench.py:151`.
- ASR is backfilled by `slurm/eval-sweep-results.sh` using the HarmBench
    classifier, keyed off `eval_run_id`.
- `RoleplayStrategy` at
    `information_safety/algorithms/strategies/prompt.py:23` is the template for
    prompt-mutation strategies: it overrides `validation_step` to rewrite prompts
    before generation, implements `compute_bits`, and is wired into a Hydra YAML
    at `information_safety/configs/algorithm/strategy/roleplay.yaml`.
- AdversariaLLM (https://github.com/LLM-QC/AdversariaLLM) is a standalone
    Hydra-based 3-stage pipeline (`run_attacks.py` → `run_sampling.py` →
    `run_judges.py`) with its own MongoDB-indexed `outputs/**/run.json` layout
    and `chat_templates/`. It is **not** imported as a library here; we only
    consume the attack-phase output.

## Requirements

### 1. Vendor AdversariaLLM as a git submodule

Add the upstream repo as a submodule at `third_party/adversariallm/`:

```bash
git submodule add https://github.com/LLM-QC/AdversariaLLM.git third_party/adversariallm
```

Do not modify upstream sources. The submodule needs its own isolated
dependency install (their recommended `pixi install --locked` or a separate
`.venv`) so its dependencies do not collide with the parent `uv` environment.

**Implementation scope note:** adding the submodule pointer is part of this
spec. Actually installing its dependencies is a runtime concern handled by
`slurm/attack-gcg.sh` at job-run time, not by the subagent landing this code.
Document the install steps in a short section at the top of `attack-gcg.sh`.

### 2. Add `attack-gcg.sh` SLURM script

New file `slurm/attack-gcg.sh` that:

- Activates the AdversariaLLM environment (not our `.venv`).
- Calls `python third_party/adversariallm/run_attacks.py attack=gcg \ model=<model_key> dataset=<subset_name> ...` with GCG hyperparameters
    passed as shell args (`NUM_STEPS`, `SUFFIX_LENGTH`, `SEARCH_WIDTH`, etc.).
- After upstream completes, runs a post-processing step (below) that emits
    a single JSONL at
    `/network/scratch/b/brownet/information-safety/attacks/gcg-<model_key>-<job_id>.jsonl`
    with one line per behavior:
    `{"behavior": "...", "adversarial_prompt": "..."}`.

### 3. Add `scripts/extract_adversariallm_attacks.py`

A short converter (~60 lines) that reads AdversariaLLM's attack-phase output
for a given run, extracts the `(behavior, adversarial_prompt)` pairs, and
writes the flat JSONL described above. The script takes two CLI args:
`--adversariallm_run_dir` and `--output`.

**Schema discovery:** AdversariaLLM's exact output schema is not fully
documented upstream. The subagent must inspect the submodule source
(`third_party/adversariallm/adversariallm/attacks/gcg.py`, `io_utils/`, and
`run_attacks.py`) to determine the field names used to store each behavior's
final adversarial prompt. Prefer the explicit field (likely `"adversarial_prompt"`,
`"attack_prompt"`, or similar); if ambiguous, emit the concatenation of
original behavior + optimized suffix, whichever the upstream code treats as
the final attack input.

Schema validation: assert every behavior in the input dataset is covered in
the output file. Fail loudly on mismatch (no silent drops).

### 4. Add `PrecomputedSuffixStrategy`

New file
`information_safety/algorithms/strategies/precomputed_suffix.py`, modeled on
`RoleplayStrategy`. Fields:

- `suffix_file: str` — absolute path to the JSONL produced by requirement 3.
- `_lookup: dict[str, str]` — populated in `setup()` from `suffix_file`.

Behavior:

- `setup(model, handler, pl_module)` loads the JSONL into `self._lookup`,
    stores the tokenizer, returns the model unchanged. Fails loudly if any
    behavior in `handler.get_val_dataset(tokenizer)` is missing from the file.
- `compute_bits(model)` returns `0` (constant placeholder; see “Constraints”).
- `configure_optimizers(model)` returns a dummy optimizer (copy from
    `RoleplayStrategy`).
- `validation_step(model, tokenizer, batch, handler)` iterates the behaviors
    from `batch["labels"]` metadata, looks up the adversarial prompt via
    `self._lookup[behavior]`, re-tokenizes with `apply_chat_template` exactly
    as `RoleplayStrategy` does, and calls `handler.validate_batch(...)`.

### 5. Hydra config

New file `information_safety/configs/algorithm/strategy/precomputed_suffix.yaml`
with two keys: `_target_` pointing at the new strategy class
(`information_safety.algorithms.strategies.precomputed_suffix.PrecomputedSuffixStrategy`)
and `suffix_file: ???` so Hydra requires a CLI override.

### 6. Surface the suffix-file path in result rows

Extend `_extract_strategy_hparams` (or confirm it already captures dataclass
fields) so that `strategy_hparams` in the `final-results.jsonl` row contains
`suffix_file: "<absolute path>"`. This is how different GCG runs stay
distinguishable downstream.

### 7. Wire into sweep (smoke run only)

Add a commented-out block at the end of `slurm/sweep.sh` that launches the
precomputed-suffix strategy for `llama2` only, pointing at the JSONL from
step 2/3. Do **not** add it to the full matrix yet — just a single
smoke-run invocation.

**Behavior subset for smoke run:** use the **first 5 behaviors by row index**
from the eval split of `advbench_harmbench`. Deterministic, no sampling.
GCG hyperparameters for the smoke run: `num_steps=100`, `search_width=256`,
`suffix_length=20`. These are documented in `slurm/attack-gcg.sh` as shell
defaults overridable via environment variables.

### 8. Tests

Per CLAUDE.md, tests first, using mocks only:

- `tests/algorithms/strategies/precomputed_suffix_test.py`:

    - `test_setup_loads_suffix_file`: writes a temp JSONL with 2 entries, mocks
        `pl_module.tokenizer`, asserts `_lookup` has 2 keys.
    - `test_setup_raises_on_missing_behavior`: tempfile with 1 entry, mock
        handler's val dataset returning 2 behaviors, assert raises.
    - `test_compute_bits_returns_zero`: trivial.
    - `test_validation_step_rewrites_prompts`: mock tokenizer + handler; assert
        `handler.validate_batch` is called with `input_ids` whose decoded form
        contains the adversarial prompt string for each behavior. **Do not print
        or assert on prompt content itself** — assert call shape and that lookup
        was invoked. (Test fixtures are opaque strings like `"beh_A"` /
        `"prompt_A"`; no real adversarial text.)

- `tests/scripts/extract_adversariallm_attacks_test.py`:

    - Builds a temp dir mimicking AdversariaLLM's expected output structure with
        synthetic opaque strings, runs the converter, asserts output JSONL has the
        expected row count and keys. Does not inspect values.

## Constraints

- **No model loading in tests.** Mock `AutoTokenizer`, `pl_module`, and
    `handler.validate_batch` using `unittest.mock`.
- **No data inspection.** Fixtures must use opaque identifiers
    (`"beh_A"`, `"prompt_A"`). Never print or assert on real adversarial text.
- **Comparability with the rest of the sweep must be preserved.** GCG's
    responses must be generated by our pipeline (same chat-template, same
    `do_sample=False`, same `max_new_tokens`), not by AdversariaLLM's
    `run_sampling.py`. This is the reason we convert to a flat JSONL instead of
    reusing their generations.
- **`compute_bits` is a placeholder `0`.** A follow-up spec will replace it
    with a real measure (e.g. arithmetic-coded source of the GCG routine).
    Do not invent a half-measure here.
- **No attack-time FLOPs accounting.** `avg_flops_per_example` in the result
    row continues to reflect only eval-time generation, as with the other
    prompt strategies. GCG search FLOPs are excluded by design for now.
- The existing `eval-sweep-results.sh` pipeline must run unchanged — it is
    keyed only on `eval_run_id` and the files under `generations/{id}/`.

## Test Plan

Files to add:

- `tests/algorithms/strategies/precomputed_suffix_test.py`
- `tests/scripts/extract_adversariallm_attacks_test.py`

Mocks:

- `information_safety.algorithms.strategies.precomputed_suffix.AutoTokenizer`
- `handler.validate_batch` (assert call count + structural shape of args)
- `handler.get_val_dataset` (return stub iterable of metadata dicts)

Key assertions:

- JSONL loaded into dict of correct length.
- Missing-behavior raises with a message naming the missing behavior.
- `validation_step` invokes `handler.validate_batch` exactly once per call,
    with `input_ids` shape `(batch_size, L)` for some `L`.
- Converter output row count equals the number of input behaviors.

## Acceptance Criteria

- [ ] `git submodule status` shows `third_party/adversariallm` at a pinned
    commit.
- [ ] `pytest tests/algorithms/strategies/precomputed_suffix_test.py` passes.
- [ ] `pytest tests/scripts/extract_adversariallm_attacks_test.py` passes.
- [ ] `pytest tests/` passes with no regressions.
- [ ] `ruff check` reports no errors in changed files.
- [ ] `pyright` reports no errors in changed files.
- [ ] `information_safety/configs/algorithm/strategy/precomputed_suffix.yaml`
    exists and is valid Hydra config.
- [ ] A dry-run invocation `python information_safety/main.py     experiment=prompt-attack algorithm/strategy=precomputed_suffix \     algorithm.strategy.suffix_file=<fixture.jsonl> ...` loads config
    without errors (may fail later on model loading — that is fine).
- [ ] Smoke run on `llama2` with 5 behaviors × 100 GCG steps:
    - Produces
        `/network/scratch/b/brownet/information-safety/attacks/gcg-llama2-<job>.jsonl`
        with exactly 5 rows, each having `behavior` and `adversarial_prompt`
        string fields.
    - Eval run produces a directory
        `/network/scratch/b/brownet/information-safety/generations/<eval_run_id>/`
        containing `input_data.jsonl` and `responses.jsonl`, each with 5 rows.
    - A row is appended to `final-results.jsonl` with
        `experiment_name == "PrecomputedSuffixStrategy"`, `bits == 0`,
        `asr == null`, and `strategy_hparams.suffix_file` pointing to the
        JSONL used.
    - `slurm/eval-sweep-results.sh` backfills `asr` to a non-null numeric value
        on that row.

## How suffix selection works (explicit design note)

Different GCG runs will produce different suffixes for the same behavior
(different seeds, hyperparameters, model, or upstream commit). They are kept
distinct by path:

- Each `run_attacks.py` invocation produces its own JSONL under
    `/network/scratch/b/brownet/information-safety/attacks/gcg-<model>-<job>.jsonl`.
- `PrecomputedSuffixStrategy` is selected by path via the Hydra CLI:
    `algorithm.strategy.suffix_file=/…/gcg-llama2-9300000.jsonl`.
- The absolute path is recorded into `strategy_hparams.suffix_file` on the
    result row, so `final-results.jsonl` unambiguously identifies which attack
    artifact produced each ASR number.
- There is no implicit “latest” selection, no symlinks. Two runs of the full
    sweep against two different GCG artifacts appear as two separate rows with
    different `strategy_hparams.suffix_file` values.

This keeps the strategy method-agnostic: any attack (GCG, AutoDAN, PAIR) that
can be dumped into the same `{behavior, adversarial_prompt}` JSONL schema is
consumable by `PrecomputedSuffixStrategy` without code changes.

## Execution Instructions for the Implementing Subagent

1. **Land all code + tests first.** Submodule pointer, converter, strategy,
    Hydra YAML, SLURM scripts, tests. Do not launch SLURM jobs until the code
    passes `pytest tests/`, `ruff check --fix`, and `pyright`.
2. **Do not commit.** The parent session will invoke `/commit` after reviewing
    the diff. Leave the working tree staged-or-unstaged, but clean of stray
    files.
3. **Submit the attack SLURM job** (`slurm/attack-gcg.sh`) once all code is
    green. Use `sbatch` with a **15-minute** time limit on the `main`
    partition for the 5-behavior × 100-step smoke config (benchmark-scale,
    per CLAUDE.md guidance). Capture the job ID.
4. **Do not block on the SLURM job.** Record the job ID, the expected
    AdversariaLLM output directory path, and the planned converter invocation
    in a short `smoke-run-notes.md` at the repo root (ephemeral — parent
    session will read and then delete). Exit once the code is landed and the
    job is submitted.
5. **Disk quota:** everything large (AdversariaLLM outputs, attacks JSONL,
    generations, slurm logs) must go under `/network/scratch/b/brownet/...`.
    Only source, configs, tests, and the spec go in `$HOME`.
