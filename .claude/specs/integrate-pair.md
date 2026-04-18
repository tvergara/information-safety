# Spec: Integrate PAIR attack alongside GCG/AutoDAN

## Goal

Run PAIR (Chao et al., 2023) against our target models on the same 200-behavior HarmBench
subset, using the shared precomputed-prompt pipeline already established for GCG and
AutoDAN. The plot-side discriminator (`experiment_name`) must keep PAIR visually distinct
from GCG/AutoDAN.

## Context

- PAIR is already implemented upstream at
    `third_party/adversariallm/adversariallm/attacks/pair.py` and declared in
    `conf/attacks/attacks.yaml` (`attacks.pair.*`).
- `run.json` schema is shared (`original_prompt`, `steps` with `flops` and `model_input`),
    so `scripts/extract_adversariallm_attacks.py` reads it with **one change** (see R2).
- Eval-side plumbing is already in place:
    - `PrecomputedAdversarialPromptStrategy` (placement-agnostic) in
        `information_safety/algorithms/strategies/precomputed_adversarial_prompt.py`
    - `slurm/attack-adversariallm.sh` already dispatches on `ATTACK_KEY` +
        `HYDRA_ATTACK_OVERRIDES`. No changes needed.
- Two PAIR-specific facts that affect integration:
    - **`loss=None` on every step.** PAIR scores via a judge model, not a loss. Upstream
        hardcodes the judge to return a constant `1` (see `pair.py` docstring:
        "Due to memory limits, we do not use a judge model and just return a score of 1").
        Our extractor currently picks the min-loss step — will `TypeError` on PAIR output.
        **Decision:** when all step losses are `None`, fall back to the **last step** (PAIR
        refines monotonically across iterations — last step is the most refined).
    - **Attacker LLM required.** Upstream default is `lmsys/vicuna-13b-v1.5`. `pair.py:104`
        special-cases `attack_model.id == model.name_or_path` to share weights with the
        target. **Decision: reuse target model as attacker** (same pattern as AutoDAN's
        mutation model). Keeps `bits = 0` and VRAM manageable.

## Requirements

### R1. Add `PrecomputedPAIRStrategy` subclass

- Add thin subclass to
    `information_safety/algorithms/strategies/precomputed_adversarial_prompt.py`:
    ```python
    @dataclass
    class PrecomputedPAIRStrategy(PrecomputedAdversarialPromptStrategy):
        """PAIR (Chao et al., 2023): iterative-refinement adversarial prompt."""
    ```
    No overrides — the base class handles everything. Subclass exists only so
    `experiment_name = type(self.strategy).__name__` in result rows distinguishes PAIR
    from GCG/AutoDAN in plots.
- Add Hydra config at
    `information_safety/configs/algorithm/strategy/precomputed_pair.yaml`:
    ```yaml
    _target_:
      information_safety.algorithms.strategies.precomputed_adversarial_prompt.PrecomputedPAIRStrategy
    suffix_file: ???
    ```

### R2. Extractor: last-step fallback when all losses are None

- In `scripts/extract_adversariallm_attacks.py`, inside `_extract_pair`:
    - If every `steps[i]["loss"]` is `None` → pick `best_idx = len(steps) - 1`.
    - Else (current behavior) → `best_idx = min(range(len(steps)), key=lambda i: steps[i]["loss"])`.
- No new function, no strategy flag — the extractor handles both cases uniformly based on
    the data it sees.

### R3. Smoke-test PAIR attack invocation via existing launcher

- Verify the existing `slurm/attack-adversariallm.sh` runs PAIR with the target model
    reused as attacker. The Hydra override shape is:
    ```
    ATTACK_KEY=pair \
    MODEL_KEY=meta-llama/Meta-Llama-3-8B-Instruct \
    HYDRA_ATTACK_OVERRIDES="\
        attacks.pair.attack_model.id=meta-llama/Meta-Llama-3-8B-Instruct \
        attacks.pair.attack_model.tokenizer_id=meta-llama/Meta-Llama-3-8B-Instruct \
        attacks.pair.attack_model.short_name=llama3 \
        attacks.pair.attack_model.developer_name=meta \
        attacks.pair.attack_model.dtype=bfloat16 \
        attacks.pair.attack_model.chat_template=null" \
    sbatch slurm/attack-adversariallm.sh
    ```
    The `attack_model.id == model.name_or_path` branch in `pair.py:104` then reuses the
    loaded target weights — no extra VRAM for a separate attacker.
- No edits to the launcher itself.

### R4. Add PAIR sweep doc to `slurm/sweep.sh`

- Mirror the existing AutoDAN block (commented, at the bottom of the file). Include the
    full `HYDRA_ATTACK_OVERRIDES` string from R3 so a future user can copy-paste without
    spelunking through upstream Hydra defaults. Point the eval side at
    `algorithm/strategy=precomputed_pair`.

### R5. Tests (TDD)

- `tests/algorithms/strategies/precomputed_adversarial_prompt_test.py`:
    - Add `test_pair_and_gcg_have_distinct_class_names` mirroring the existing
        `test_autodan_and_gcg_have_distinct_class_names` assertion.
- `tests/scripts/extract_adversariallm_attacks_test.py`:
    - Add `test_extract_picks_last_step_when_all_losses_none`: fixture `run.json` with
        3 steps, all `loss=None`, different `model_input`. Assert extractor picks the
        step-2 prompt.
    - Keep the existing min-loss test unchanged — both code paths must stay covered.

### R6. Add PAIR to result-completeness tracker

- In `scripts/check_results.py`, extend the attack-strategy list to include
    `"PrecomputedPAIRStrategy"` alongside the two existing entries in the `product()`
    over `ATTACK_MODELS`. Should expand the expected matrix by exactly
    `len(ATTACK_MODELS) = 5` rows.

## Constraints

- **Must NOT** touch the base `PrecomputedAdversarialPromptStrategy` — only add a
    subclass.
- **Must NOT** rewrite existing GCG/AutoDAN JSONLs. The extractor change is purely
    additive.
- **Must NOT** introduce a new SLURM script or launcher flag for PAIR — it uses the
    existing generic launcher.
- **Must NOT** add a judge model config override. Upstream's constant-1 judge is a
    limitation we accept as-is (documented in the "Context" section); fixing it is out of
    scope.
- Same 200-behavior HarmBench standard subset as GCG/AutoDAN.
- Max line length 99, type annotations on everything, no defensive guards per
    CLAUDE.md.

## Test Plan

- `pytest tests/algorithms/strategies/precomputed_adversarial_prompt_test.py`
- `pytest tests/scripts/extract_adversariallm_attacks_test.py`
- `pytest tests/` (no regressions)
- `pyright` clean on changed files
- `ruff check --fix` clean

Manual verification (after tests green):

- Smoke-run PAIR for 1 behavior on one interactive GPU with the full
    `HYDRA_ATTACK_OVERRIDES` from R3:
    - `run.json` written with `loss=None` on every step
    - `extract_adversariallm_attacks.py` produces a non-empty JSONL with
        `attack_flops > 0` and no `TypeError`
- Smoke-run the eval side:
    ```
    python information_safety/main.py \
        experiment=prompt-attack \
        algorithm/dataset_handler=advbench_harmbench \
        algorithm/strategy=precomputed_pair \
        algorithm.strategy.suffix_file=<the smoke JSONL> \
        algorithm.model.pretrained_model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
        algorithm.model.trust_remote_code=false \
        trainer.precision=bf16-mixed \
        name=pair-llama3-smoke
    ```
    Result row lands in `final-results.jsonl` with `experiment_name: PrecomputedPAIRStrategy`.

## Acceptance Criteria

- [ ] `pytest tests/` passes.
- [ ] `pyright` reports no errors on changed files.
- [ ] `ruff check --fix` clean.
- [ ] `PrecomputedPAIRStrategy` exists and produces
    `experiment_name: PrecomputedPAIRStrategy` in result rows (verified by unit test).
- [ ] `extract_adversariallm_attacks.py` with a fixture where all step losses are `None`
    picks the last step (verified by unit test).
- [ ] `extract_adversariallm_attacks.py` with a normal fixture (losses set) still picks
    the min-loss step (existing test passes unchanged).
- [ ] `scripts/check_results.py` includes `PrecomputedPAIRStrategy` in the expected
    matrix and reports the 5 new missing entries (1 per ATTACK_MODEL) until runs land.
- [ ] `slurm/sweep.sh` contains a commented PAIR smoke block with the full
    `HYDRA_ATTACK_OVERRIDES` string for attacker reuse.
- [ ] Unit-level FLOPs accounting test: a PAIR-shaped fixture with 3 steps of 1e9 flops
    each yields `attack_flops = 3e9` in the extractor's output row (catches regressions
    where the extractor picks one step's flops instead of summing).

## Out of scope

- Actually running the PAIR sweep (separate follow-up after this infra lands).
- Replacing upstream's constant-1 judge with a real classifier (tracked upstream, not
    our problem).
- Supporting a separate attacker LLM (config path stays available through Hydra
    overrides; just not wired into the sweep defaults, since using a different attacker
    would invalidate the `bits = 0` framing).
