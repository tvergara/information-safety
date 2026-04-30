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

The attack program is a fixed artifact: source code, hyperparameters, and any
one-time training output (e.g. finetuned weights). Its length is independent
of the test set or any individual test example.

For per-behavior attacks (GCG, AutoDAN, PAIR), the *common code* and
*config-level search budget* (e.g. `num_steps × topk` for GCG) constitute the
program. The optimized adversarial prompts for each behavior are *outputs* of
running the program — not part of it.

| Strategy | Program length (bits) |
|---|---|
| Baseline | `0` |
| Roleplay | cross-entropy bits of the fixed template under the attacked model |
| LoRA | `16 × trainable_param_count` (bfloat16 storage) |
| DataStrategy | accumulated cross-entropy bits of training data under the model |
| GCG / AutoDAN / PAIR | `code_bits + search_bits + payload_bits + meta_bits` (config-level, same for every behavior in a run; see `information_safety/attack_bits.py`) |

There is no per-example component.

## FLOPs

Following Kaplan et al. (2020): forward-only ≈ `2 × N × T`, forward+backward
≈ `6 × N × T`, where `N` = non-embedding parameter count, `T` = tokens
processed.

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
