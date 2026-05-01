# Spec: Per-attack bits accounting for GCG / AutoDAN / PAIR

## Goal

Centralize bits-per-attack computation matching Xarangi/AdversariaLLM `a799e7f`
(minus PAIR helper weights), compute it once per run at extraction time, and
propagate it into `final-results.jsonl` via the precomputed-strategy pipeline.

## Context

- Reference formulas live in `/tmp/bits_ref.py` (Xarangi's `scripts/plot_program_length_vs_asr.py`).
- Our upstream attack outputs are per-behavior `run.json` files under
    `/network/scratch/.../adversariallm-outputs/<run_id>/<timestamp>/<i>/run.json`,
    each with a `config` dict carrying the attack hyperparams.
- `scripts/extract_adversariallm_attacks.py` currently emits flat JSONL rows of
    `{behavior, adversarial_prompt, attack_flops}`.
- `information_safety/algorithms/strategies/precomputed_adversarial_prompt.py`
    currently returns `0` from `compute_bits`. That `0` is what lands in
    `final-results.jsonl` → plots have been reading `r["bits"] == 0` for all
    precomputed attacks, which is wrong.
- There is already a narrow `estimate_gcg_search_bits` helper in
    `information_safety/algorithms/jailbreak/gcg.py`. It covers only `search_bits`
    (no code/payload/meta) and is used by one caller + one test. We consolidate
    on the new unified API and delete it.

## Requirements

1. New module `information_safety/attack_bits.py` exposing:
    - Constants: `TOKEN_ID_BITS = 16`.
    - `class_code_bits(py_file: str, class_name: str) -> int` — AST-parses the
        file, finds the class, returns `len(source.encode("utf-8")) * 8`. Matches
        reference `class_code_bits`.
    - `parse_autodan_initial_strings_bits(autodan_file: str) -> tuple[int, int]`
        — AST-parses the `INITIAL_STRINGS` list in the AutoDAN source and returns
        `(avg_utf8_bits, count)`. Matches reference.
    - `gcg_bits(config: dict) -> dict[str, int]` — returns
        `{total, code_bits, search_bits, payload_bits, meta_bits}` using the
        reference formula:
        - `suffix_tokens = len(str(config["optim_str_init"]).split())`
        - `search = num_steps * suffix_tokens * log2(max(topk, 2))`
        - `payload = suffix_tokens * TOKEN_ID_BITS`
        - `meta = num_steps * TOKEN_ID_BITS`
    - `autodan_bits(config: dict) -> dict[str, int]` — returns
        `{total, code_bits, initial_pool_bits, evolution_bits, mutation_bits}`:
        - `evolution = num_steps * batch * log2(max(batch, 2))`
        - `mutation = num_steps * batch * mutation_rate * TOKEN_ID_BITS`
    - `pair_bits(config: dict) -> dict[str, int]` — returns
        `{total, code_bits, payload_bits, search_bits}` **without** helper weights:
        - `payload = num_steps * streams * max_new_tokens * TOKEN_ID_BITS`
        - `search = num_steps * streams * max_attempts * log2(max(max_new_tokens, 2))`
    - `compute_attack_bits(attack: str, config: dict) -> int` dispatch
        returning the scalar total, used by the extractor.
    - The attack-source paths for `class_code_bits` and
        `parse_autodan_initial_strings_bits` live in module-level constants
        pointing at `third_party/adversariallm/adversariallm/attacks/{gcg,autodan,pair}.py`
        (resolved relative to repo root).
2. `scripts/extract_adversariallm_attacks.py`:
    - Read `run_json["config"]` to derive the attack name (GCG/AutoDAN/PAIR —
        dispatch on the presence of `optim_str_init` / `batch_size,mutation` /
        `num_streams` respectively, matching reference logic). Fail loud on
        unknown.
    - Call `compute_attack_bits(...)` once per `run.json` and add
        `attack_bits: int` to every row emitted (all rows within one run.json
        share the same bits — it's a per-config quantity).
3. `PrecomputedAdversarialPromptStrategy` (base class): sum `attack_bits` across
    rows during `setup()` the same way `attack_flops` is summed, store as
    `_attack_bits_total`, and return it from `compute_bits(model)` instead of
    the current `0`. Fail loud on a row missing `attack_bits` (no legacy
    fallback — see Constraints).
4. Delete `estimate_gcg_search_bits` and its usage + test. The new
    `gcg_bits(...)` supersedes it. Update `information_safety/algorithms/jailbreak/__init__.py`
    exports accordingly.
5. Re-extract every existing merged JSONL under
    `/network/scratch/b/brownet/information-safety/attacks/` so they carry
    `attack_bits`. (Driver: a one-liner that iterates saved
    `adversariallm-outputs/*` run dirs and re-runs the extractor. Not a spec
    deliverable — I'll execute it after code lands.)

## Constraints

- Match Xarangi's `compute_attack_bits` output for GCG/AutoDAN identically.
- Drop **only** the PAIR helper term (`helper_model_params * helper_precision_bits`).
    PAIR's `code_bits`, `payload_bits`, `search_bits` stay identical to reference.
- Do not inspect prompt/completion text in tests (CLAUDE.md rule).
- Fail loud on missing `attack_bits` in the JSONL — do NOT add a `.get("attack_bits", 0)`
    fallback. The ONLY way to get a row into the pipeline is through the new extractor,
    and a row lacking `attack_bits` is a bug.
- Type-annotate everything; no defensive `or 0` patterns.

## Test Plan

- `tests/attack_bits_test.py`:
    - `gcg_bits` returns known values for a tiny config (pick small ints so the
        expected value is hand-computable and matches the reference formula).
    - `autodan_bits` same.
    - `pair_bits` same — and explicitly assert PAIR total does NOT include
        `13e9 * 16`.
    - `class_code_bits` on a fixture `.py` file returns `len(src)*8`.
    - `parse_autodan_initial_strings_bits` on a fixture with
        `INITIAL_STRINGS = ["ab", "cdef"]` returns `(avg * 8, 2)`.
    - `compute_attack_bits` dispatches correctly and raises on unknown attack.
- `tests/scripts/extract_adversariallm_attacks_test.py`:
    - Existing tests updated — new extractor output includes `attack_bits` key.
    - Fixture `run.json` for each of GCG / AutoDAN / PAIR (mock `config` only —
        no real prompts).
- `tests/algorithms/strategies/precomputed_adversarial_prompt_test.py`:
    - `compute_bits` returns sum of `attack_bits` across loaded rows.
    - Missing `attack_bits` in any row → `KeyError` at `setup()`.
- `tests/algorithms/jailbreak/gcg_test.py`:
    - Delete the `estimate_gcg_search_bits` test.

## Acceptance Criteria

- [ ] `pytest tests/attack_bits_test.py` passes.
- [ ] `pytest tests/scripts/extract_adversariallm_attacks_test.py` passes.
- [ ] `pytest tests/algorithms/strategies/precomputed_adversarial_prompt_test.py` passes.
- [ ] `pytest tests/algorithms/jailbreak/` passes (with `estimate_gcg_search_bits` removed).
- [ ] `pytest tests/` passes clean — no skips, no xfails.
- [ ] `ruff check --fix` reports no errors.
- [ ] `pyright` reports no errors in changed files.
- [ ] For a hand-picked config (GCG `num_steps=250, search_width=512,     optim_str_init="x "*20`), `gcg_bits(...)["total"]` equals the value the
    reference `compute_attack_bits` returns for the same config (minus
    code_bits, which depends on our source copy — verified separately by
    running both on the same config).
- [ ] `pair_bits(config)["total"]` equals reference minus exactly
    `13_000_000_000 * 16`.
- [ ] `grep -r estimate_gcg_search_bits` returns zero hits.
