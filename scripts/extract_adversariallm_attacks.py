"""Convert AdversariaLLM attack-phase output into a flat JSONL.

For all three iterative attacks (GCG, AutoDAN, PAIR) the output contains one row per
intermediate trajectory step:

- **GCG**: one row per Pareto-optimal optimization step (steps whose best-so-far loss
  strictly decreased).
- **AutoDAN**: one row per Pareto-optimal step on loss, identical convention to GCG.
- **PAIR**: stride-sampled rows — every `PAIR_STRIDE`-th step counting backward from the
  final step, plus always the final step itself. PAIR's per-step `loss` is `None` (its
  signal is an external judge whose scores are not stored), so a Pareto-on-loss filter
  doesn't apply.

Each output row contains:
    {behavior, adversarial_prompt, attack_flops, attack_bits, pareto_step_idx,
     [model_completion]}

Works for any AdversariaLLM attack whose `run.json` follows the shared schema (GCG, AutoDAN,
PAIR, PGD, ...). AdversariaLLM writes one `run.json` per behavior under
`<save_dir>/<timestamp>/<i>/run.json`. Each `run.json` has:
    {
        "config": {"attack_params": {...}, ...},
        "runs": [
            {
                "original_prompt": [{"role": "user", "content": behavior}, ...],
                "steps": [
                    {"step": k, "loss": ..., "flops": ..., "model_input": [...],
                     "model_completions": [...]},
                    ...
                ],
                "total_time": ...
            }
        ]
    }

Every step records the (best-so-far) prompt + a stored greedy completion, so each
emitted step is independently usable as a "checkpoint" with its own
(attack_flops, prompt, completion) tuple. Indexing convention: `pareto_step_idx == 0`
denotes the **final** row (the one today's downstream pipeline treats as canonical for
ASR computation); larger indices step backward in time toward step 0.

`attack_bits` is a per-config scalar (the same value for every row derived from one `run.json`,
since bits are an accounting of the attack's *program length*, not a per-behavior quantity).
We replicate it on every row purely for plumbing convenience; it is NOT a per-behavior signal.

`attack_flops`:
- GCG and PAIR store per-step deltas; we sum them. For GCG/PAIR rows we sum up to (and
  including) the emitted step's index — the cumulative cost of running the attack to
  that checkpoint.
- AutoDAN writes `step.flops` cumulatively (each step's value already includes earlier
  steps), so we use the emitted step's `flops` field directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections.abc import Callable
from typing import Any

from information_safety.attack_bits import compute_attack_bits

MIN_ATTACK_STEPS = 20
PAIR_STRIDE = 10


def _find_run_files(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "**", "run.json"), recursive=True))


def _cumulative_flops(steps: list[dict[str, Any]]) -> list[int]:
    cumulative: list[int] = []
    running = 0
    for step in steps:
        running += step["flops"]
        cumulative.append(running)
    return cumulative


def _build_rows(
    steps: list[dict[str, Any]],
    behavior: str,
    selected_step_indices: list[int],
    flops_for_step: Callable[[int], int],
) -> list[dict[str, Any]]:
    """Build trajectory rows from a list of step indices in pareto_step_idx order.

    `selected_step_indices[0]` is the row that becomes `pareto_step_idx=0` (the canonical
    final/best row); larger indices step backward.
    """
    rows: list[dict[str, Any]] = []
    for rank, step_i in enumerate(selected_step_indices):
        step = steps[step_i]
        rows.append({
            "behavior": behavior,
            "adversarial_prompt": step["model_input"][0]["content"],
            "attack_flops": flops_for_step(step_i),
            "pareto_step_idx": rank,
            "model_completion": step["model_completions"][0],
        })
    return rows


def _loss_pareto_indices(steps: list[dict[str, Any]]) -> list[int]:
    """Step indices where best-so-far loss strictly decreased, ordered final-first."""
    pareto: list[int] = []
    best_loss: float | None = None
    for i, step in enumerate(steps):
        loss = step["loss"]
        if best_loss is None or loss < best_loss:
            best_loss = loss
            pareto.append(i)
    pareto.reverse()
    return pareto


def _extract_gcg_pareto(run_json: dict, run_file: str) -> list[dict[str, Any]]:
    """Return one dict per Pareto-optimal step of a GCG run.

    GCG stores per-step `flops` deltas; each row's `attack_flops` is the
    cumulative sum through its step.
    """
    run = run_json["runs"][0]
    behavior: str = run["original_prompt"][0]["content"]
    steps = run["steps"]
    if not steps:
        raise ValueError(f"run has no steps in {run_file}")
    cumulative = _cumulative_flops(steps)
    return _build_rows(steps, behavior, _loss_pareto_indices(steps), cumulative.__getitem__)


def _extract_autodan_pareto(run_json: dict, run_file: str) -> list[dict[str, Any]]:
    """Return one dict per Pareto-optimal step of an AutoDAN run.

    AutoDAN writes `step.flops` cumulatively, so each row's `attack_flops` is
    taken directly from `steps[i]["flops"]`.
    """
    run = run_json["runs"][0]
    behavior: str = run["original_prompt"][0]["content"]
    steps = run["steps"]
    if not steps:
        raise ValueError(f"run has no steps in {run_file}")
    return _build_rows(
        steps, behavior, _loss_pareto_indices(steps), lambda i: steps[i]["flops"]
    )


def _extract_pair_strided(
    run_json: dict, run_file: str, *, stride: int = PAIR_STRIDE
) -> list[dict[str, Any]]:
    """Return stride-sampled rows for a PAIR run.

    Selects indices `{N-1, N-1-stride, N-1-2*stride, ...}` (no duplicates, no
    negative indices), then maps them to `pareto_step_idx` 0, 1, 2, ... in
    backward-from-final order (final = 0). PAIR stores per-step `flops` deltas,
    so each row's `attack_flops` is the cumulative sum through its step.
    """
    run = run_json["runs"][0]
    behavior: str = run["original_prompt"][0]["content"]
    steps = run["steps"]
    if not steps:
        raise ValueError(f"run has no steps in {run_file}")
    cumulative = _cumulative_flops(steps)
    selected = list(range(len(steps) - 1, -1, -stride))
    return _build_rows(steps, behavior, selected, cumulative.__getitem__)


def _dispatch_attack_name(attack_params: dict[str, Any]) -> str:
    """Return the attack name from its config keys.

    Dispatches on presence of distinguishing keys (matching the reference
    logic): GCG has `optim_str_init`, AutoDAN has `batch_size` and `mutation`,
    PAIR has `num_streams`. Raises `ValueError` for anything else.
    """
    if "optim_str_init" in attack_params:
        return "gcg"
    if "batch_size" in attack_params and "mutation" in attack_params:
        return "autodan"
    if "num_streams" in attack_params:
        return "pair"
    raise ValueError(f"Unknown attack in config: keys={sorted(attack_params.keys())}")


def _extract_rows_from_run(
    run_json: dict, run_file: str, attack_name: str, attack_bits: int
) -> list[dict[str, Any]]:
    if attack_name != "pair" and len(run_json["runs"][0]["steps"]) < MIN_ATTACK_STEPS:
        return []
    if attack_name == "gcg":
        rows = _extract_gcg_pareto(run_json, run_file)
    elif attack_name == "autodan":
        rows = _extract_autodan_pareto(run_json, run_file)
    else:
        rows = _extract_pair_strided(run_json, run_file)
    for row in rows:
        row["attack_bits"] = attack_bits
    return rows


def convert_run_dir(run_dir: str, output_file: str) -> None:
    run_files = _find_run_files(run_dir)
    if not run_files:
        raise ValueError(f"No run.json files found under {run_dir}")

    behavior_to_rows: dict[str, list[dict[str, Any]]] = {}
    for run_file in run_files:
        with open(run_file) as f:
            run_json = json.load(f)
        attack_params = run_json["config"]["attack_params"]
        attack_name = _dispatch_attack_name(attack_params)
        if attack_name == "pair":
            attack_bits = compute_attack_bits(
                attack_name,
                attack_params,
                attacked_model_name=run_json["config"]["model_params"]["id"],
            )
        else:
            attack_bits = compute_attack_bits(attack_name, attack_params)
        new_rows = _extract_rows_from_run(run_json, run_file, attack_name, attack_bits)
        if not new_rows:
            continue
        behavior_to_rows[new_rows[0]["behavior"]] = new_rows

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for behavior_rows in behavior_to_rows.values():
            for row in behavior_rows:
                f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adversariallm_run_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()
    convert_run_dir(args.adversariallm_run_dir, args.output)


if __name__ == "__main__":
    main()
