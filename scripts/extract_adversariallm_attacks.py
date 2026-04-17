"""Convert AdversariaLLM GCG attack-phase output into a flat `{behavior, adversarial_prompt}`
JSONL.

AdversariaLLM writes one `run.json` per behavior under `<save_dir>/<timestamp>/<i>/run.json`.
Each `run.json` has:
    {
        "config": {...},
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
`model_input` is the optimized prompt (original behavior + optimized suffix).
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def _find_run_files(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "**", "run.json"), recursive=True))


def _extract_pair(run_json: dict, run_file: str) -> tuple[str, str]:
    run = run_json["runs"][0]
    behavior: str = run["original_prompt"][0]["content"]
    steps = run["steps"]
    if not steps:
        raise ValueError(f"run has no steps in {run_file}")

    best_idx = min(range(len(steps)), key=lambda i: steps[i]["loss"])
    adversarial_prompt: str = steps[best_idx]["model_input"][0]["content"]
    return behavior, adversarial_prompt


def convert_run_dir(run_dir: str, output_file: str) -> None:
    run_files = _find_run_files(run_dir)
    if not run_files:
        raise ValueError(f"No run.json files found under {run_dir}")

    seen: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for run_file in run_files:
        with open(run_file) as f:
            run_json = json.load(f)
        behavior, adversarial_prompt = _extract_pair(run_json, run_file)
        if behavior in seen:
            raise ValueError(
                f"duplicate behavior across runs: {run_file} and {seen[behavior]}"
            )
        seen[behavior] = run_file
        rows.append({"behavior": behavior, "adversarial_prompt": adversarial_prompt})

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
