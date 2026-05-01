"""Convert AdversariaLLM attack-phase output into a flat JSONL with one row per behavior.

Each output row contains:
    {behavior, adversarial_prompt, attack_flops, attack_bits, [model_completion]}

Works for any AdversariaLLM attack whose `run.json` follows the shared schema (GCG, AutoDAN,
PAIR, PGD, ...). AdversariaLLM writes one `run.json` per behavior under
`<save_dir>/<timestamp>/<i>/run.json`. Each `run.json` has:
    {
        "config": {"attack_params": {...}, ...},
        "runs": [
            {
                "original_prompt": [{"role": "user", "content": behavior}, ...],
                "steps": [
                    {"step": k, "loss": ..., "model_input": [{"role": "user", "content": ...}], ...},
                    ...
                ],
                "total_time": ...
            }
        ]
    }

We pick the step with the lowest loss as the final adversarial prompt. The user turn of that step's
`model_input` is the optimized prompt (with the attack's placement — suffix for GCG, prefix for
AutoDAN, etc.). If the step has a stored `model_completions[0]`, we propagate it as
`model_completion` so that downstream replay can skip eval-time generation.

`attack_bits` is a per-config scalar (the same value for every row derived from one `run.json`,
since bits are an accounting of the attack's *program length*, not a per-behavior quantity).
We replicate it on every row purely for plumbing convenience; it is NOT a per-behavior signal.

`attack_flops`:
- GCG and PAIR store per-step deltas; we sum them.
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


def _find_run_files(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "**", "run.json"), recursive=True))


def _attack_flops_from_steps(steps: list[dict], attack_name: str) -> int:
    if attack_name == "autodan":
        return steps[-1]["flops"]
    return sum(step["flops"] for step in steps)


def _extract_pair(
    run_json: dict, run_file: str, attack_name: str
) -> tuple[str, str, int, str | None]:
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
        behavior, adversarial_prompt, attack_flops, model_completion = _extract_pair(
            run_json, run_file, attack_name
        )
        if behavior in seen:
            raise ValueError(
                f"duplicate behavior across runs: {run_file} and {seen[behavior]}"
            )
        seen[behavior] = run_file
        row: dict[str, Any] = {
            "behavior": behavior,
            "adversarial_prompt": adversarial_prompt,
            "attack_flops": attack_flops,
            "attack_bits": attack_bits,
        }
        if model_completion is not None:
            row["model_completion"] = model_completion
        rows.append(row)

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
