# Measurement Definitions: FLOPs and Program Length

## Goal

Every attack strategy in this project has a *cost*. We measure it along two axes:

- **Program length** (bits): size of the attack program — a fixed property of the artifact.
- **FLOPs**: total floating-point operations.

To reason about how cost scales with evaluation effort, FLOPs split into two
components based on **when** the cost is paid. Program length does not split —
it is independent by construction.

**Sanity check.** If we double the number of test examples we evaluate on,
independent costs are unchanged; dependent costs accumulate twice as many
per-example samples.

## Program length (bits)

Program length is the size of the attack as a shippable artifact. It has
three components, all **independent** of the test set:

- **code**: UTF-8 bit-length of the first-party `.py` files used by the
  attack. Libraries do not count. Shared lightning/orchestration glue does
  not count — a minimal `scripts/naive_inference.py` stands in for it and is
  included as a floor in every attack's code bits.
- **data**: `Σ −log₂ p_LM(data)` over any data the program ships. Justified
  by arithmetic coding: a program can ship the data losslessly compressed
  under the language model and decode it at runtime.
- **auxiliary models**: `16 × num_params` per non-attacked model the program
  ships (e.g. a finetuned LoRA adapter, or PAIR's attacker LLM if distinct
  from the attacked model). The attacked model is free — it is the shared
  substrate, not part of the attack.

`program_bits = code_bits + data_bits + auxiliary_model_bits`.

Program length is **not input-dependent**. Per-test-example, per-behavior,
and search-trajectory terms do not enter `program_bits`. Per-behavior outputs
(e.g. the optimized GCG suffix for a given behavior) are *outputs* of running
the program — not part of it.

| Strategy | code | data | aux models |
|---|---|---|---|
| Baseline | `naive_inference.py` | 0 | 0 |
| Roleplay | naive + `strategies/prompt.py` + `jailbreak/prompt_methods.py` | 0 | 0 |
| LoRA | naive + `strategies/lora.py` | 0 | `16 × trainable_params` (adapter) |
| DataStrategy | naive + `strategies/data.py` + `strategies/lora.py` + `utils/arithmetic_coding.py` | `Σ −log₂ p_LM(train_data)` | 0 |
| GCG | naive + `attacks/gcg.py` + `strategies/precomputed_adversarial_prompt.py` | 0 | 0 |
| AutoDAN | naive + `attacks/autodan.py` + `strategies/precomputed_adversarial_prompt.py` | 0 | 0 |
| PAIR | naive + `attacks/pair.py` + `strategies/precomputed_adversarial_prompt.py` | 0 | `16 × helper_params` if helper model ≠ attacked model, else 0 |

## FLOPs

Following Kaplan et al. (2020): forward-only ≈ `2 × N × T`, forward+backward
≈ `6 × N × T`, where `N` = non-embedding parameter count, `T` = tokens
processed. `N` is computed as `model.num_parameters(exclude_embeddings=True)`
throughout the pipeline (both first-party code and the AdversariaLLM
submodule). No model is "free" — every forward/backward through every model
on the path from input behavior to model response is counted.

### Scope

**Counted**: any forward/backward pass through any model used to *produce*
the attack's response. For PAIR, this includes the attacker LLM, the target
(attacked) LLM, and PAIR's internal judge LLM, all aggregated into
`attack_flops_i`.

**Not counted**: the post-hoc evaluation classifier
(HarmBench-Llama-2-13B-cls) used to decide ASR. It's evaluation
infrastructure, the same way the test set itself isn't counted — if a
magical perfect classifier appeared, it would also be fine. Eval-classifier
FLOPs are constant across attacks and would only shift the absolute number.

### Independent FLOPs (paid once before any evaluation)

| Strategy | Source | Formula |
|---|---|---|
| Baseline | — | `0` |
| Roleplay | — | `0` |
| LoRA | finetuning | `6 × N × n_train_tokens` |
| DataStrategy | finetuning | `6 × N × n_train_tokens` |
| GCG / AutoDAN / PAIR | — | `0` |

Per-behavior attack-time optimization is *not* independent: doubling the test
set requires optimizing twice as many behaviors. It belongs in the dependent
bucket.

### Dependent FLOPs (per test example *i*)

| Strategy | Source | Formula |
|---|---|---|
| All | eval-time generation (forward only) | `2 × N × (n_input_i + n_output_i)` |
| GCG / AutoDAN / PAIR | per-behavior attack-time optimization | `attack_flops_i` from `run.json` (forward+backward) |
| Stored-completion replay (e.g. AutoDAN) | — | eval generation = `0`; attack-time still counts |

The eval-generation formula `2 × N × (n_input_i + n_output_i)` assumes
KV caching is enabled (true for both HF `model.generate()` and vLLM, which
this pipeline uses). Without KV caching, autoregressive generation costs
~`2 × N × n_input × n_output` worst case.

## Storage

Per-example dependent values are stored **without pre-aggregating** so future
analyses can choose any aggregation:

- `responses.jsonl`: one row per evaluated example, including per-example
  `dependent_flops_i` (sum of attack-time + eval-time FLOPs for that example).
- merged attack JSONL: one row per behavior with `attack_flops_b` (the
  dependent attack-time cost for that behavior). `attack_bits` is a per-config
  scalar; storing it per row is an artifact of current pipeline plumbing, not
  a per-behavior signal.

## Reporting in the result row

Three separate fields, no premature aggregation:

- `program_bits` — independent program length.
- `independent_flops` — training cost, or `0`.
- `eval_run_id` — pointer to the run directory containing `responses.jsonl`,
  which carries the per-example dependent FLOPs.

A consumer that wants the legacy "amortized total per example" can compute
`independent_flops / N + mean(dependent_flops)` itself by reading
`responses.jsonl`.
