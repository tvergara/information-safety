"""Convert AdversariaLLM attack-phase output into a flat JSONL.

For GCG runs the output contains one row per **Pareto-optimal** optimization step
(steps whose best-so-far loss strictly decreased). For AutoDAN and PAIR the output
keeps the legacy "one row per behavior" shape — the row with the lowest loss is
selected and tagged with `pareto_step_idx: 0`.

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

For GCG, every step records the best-so-far suffix + a stored greedy completion, so
each Pareto-optimal step is independently usable as a "checkpoint" with its own
(attack_flops, suffix, completion) tuple. Pareto indexing convention: `pareto_step_idx == 0`
denotes the **final best-loss** row (the one today's pipeline would emit); larger
indices step backward in time toward step 0.

`attack_bits` is a per-config scalar (the same value for every row derived from one `run.json`,
since bits are an accounting of the attack's *program length*, not a per-behavior quantity).
We replicate it on every row purely for plumbing convenience; it is NOT a per-behavior signal.

`attack_flops`:
- GCG and PAIR store per-step deltas; we sum them. For GCG Pareto rows we sum up to (and
  including) the Pareto step's index — the cumulative cost of running the attack to that
  checkpoint.
- AutoDAN writes `step.flops` cumulatively (final value already includes earlier steps), so we
  take only the last step's value.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

from information_safety.attack_bits import compute_attack_bits

MIN_ATTACK_STEPS = 20


def _find_run_files(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "**", "run.json"), recursive=True))


def _attack_flops_from_steps(steps: list[dict], attack_name: str) -> int:
    if attack_name == "autodan":
        return steps[-1]["flops"]
    return sum(step["flops"] for step in steps)


def _extract_argmin_loss(
    run_json: dict, run_file: str, attack_name: str
) -> tuple[str, str, int, str | None]:
    """Pick the single argmin-loss step (AutoDAN/PAIR path)."""
    run = run_json["runs"][0]
    behavior: str = run["original_prompt"][0]["content"]
    steps = run["steps"]
    if not steps:
        raise ValueError(f"run has no steps in {run_file}")

    if all(step["loss"] is None for step in steps):
        best_idx = len(steps) - 1
    else:
        best_idx = min(range(len(steps)), key=lambda i: steps[i]["loss"])
    adversarial_prompt: str = steps[best_idx]["model_input"][0]["content"]
    attack_flops = _attack_flops_from_steps(steps, attack_name)

    model_completion: str | None = None
    completions = steps[best_idx].get("model_completions")
    if completions:
        model_completion = completions[0]

    return behavior, adversarial_prompt, attack_flops, model_completion


def _extract_gcg_pareto(run_json: dict, run_file: str) -> list[dict[str, Any]]:
    """Return one dict per Pareto-optimal step of a GCG run.

    `pareto_step_idx == 0` is the final argmin-loss step (today's canonical row);
    larger indices step backward to earlier improvements. Each row includes
    `attack_flops` as cumulative FLOPs up to and including that step.
    """
    run = run_json["runs"][0]
    behavior: str = run["original_prompt"][0]["content"]
    steps = run["steps"]
    if not steps:
        raise ValueError(f"run has no steps in {run_file}")

    pareto_step_indices: list[int] = []
    best_loss: float | None = None
    cumulative_flops_at_step: list[int] = []
    running = 0
    for step in steps:
        running += step["flops"]
        cumulative_flops_at_step.append(running)
    for i, step in enumerate(steps):
        loss = step["loss"]
        if best_loss is None or loss < best_loss:
            best_loss = loss
            pareto_step_indices.append(i)

    pareto_step_indices.reverse()

    rows: list[dict[str, Any]] = []
    for rank, step_i in enumerate(pareto_step_indices):
        step = steps[step_i]
        adversarial_prompt: str = step["model_input"][0]["content"]
        row: dict[str, Any] = {
            "behavior": behavior,
            "adversarial_prompt": adversarial_prompt,
            "attack_flops": cumulative_flops_at_step[step_i],
            "pareto_step_idx": rank,
            "model_completion": step["model_completions"][0],
        }
        rows.append(row)
    return rows


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
    if len(run_json["runs"][0]["steps"]) < MIN_ATTACK_STEPS:
        return []
    if attack_name == "gcg":
        rows = _extract_gcg_pareto(run_json, run_file)
        for row in rows:
            row["attack_bits"] = attack_bits
        return rows

    behavior, adversarial_prompt, attack_flops, model_completion = _extract_argmin_loss(
        run_json, run_file, attack_name
    )
    row: dict[str, Any] = {
        "behavior": behavior,
        "adversarial_prompt": adversarial_prompt,
        "attack_flops": attack_flops,
        "attack_bits": attack_bits,
        "pareto_step_idx": 0,
    }
    if model_completion is not None:
        row["model_completion"] = model_completion
    return [row]


def convert_run_dir(run_dir: str, output_file: str) -> None:
    run_files = _find_run_files(run_dir)
    if not run_files:
        raise ValueError(f"No run.json files found under {run_dir}")

    seen: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
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
        behavior = new_rows[0]["behavior"]
        if behavior in seen:
            raise ValueError(
                f"duplicate behavior across runs: {run_file} and {seen[behavior]}"
            )
        seen[behavior] = run_file
        rows.extend(new_rows)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adversariallm_run_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()
    convert_run_dir(args.adversariallm_run_dir, args.output)


if __name__ == "__main__":
    main()
