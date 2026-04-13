"""Check that all desired experiment configurations exist in the results file."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_FILE = Path(
    "/network/scratch/b/brownet/information-safety/results/final-results.jsonl"
)

MODELS = [
    "/network/scratch/b/brownet/information-safety/models/safety-pair-safe",
    "/network/scratch/b/brownet/information-safety/models/safety-pair-unsafe",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "Qwen/Qwen3-4B",
]

CONFIGS: list[dict[str, Any]] = []

# Baseline: one per model, epochs 0 and 1
for model in MODELS:
    for epoch in [0, 1]:
        CONFIGS.append({
            "experiment_name": "BaselineStrategy",
            "model_name": model,
            "max_examples": None,
            "epoch": epoch,
        })

# DataStrategy: sweep over max_examples, epochs 0 and 1
DATA_MAX_EXAMPLES = [10, 50, 100, 250, None]
for model, max_ex, epoch in product(MODELS, DATA_MAX_EXAMPLES, [0, 1]):
    CONFIGS.append({
        "experiment_name": "DataStrategy",
        "model_name": model,
        "max_examples": max_ex,
        "epoch": epoch,
    })

# RoleplayStrategy: one per model, epoch 0 only (no training)
for model in MODELS:
    CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })


def _short_model(name: str) -> str:
    """Shorten model name for display."""
    return name.rsplit("/", 1)[-1]


def _matches(row: dict[str, Any], config: dict[str, Any]) -> bool:
    return all(row.get(k) == v for k, v in config.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check results completeness")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Path to results JSONL file",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.results_file)]

    missing: list[dict[str, Any]] = []
    needs_eval: list[dict[str, Any]] = []
    complete: list[dict[str, Any]] = []

    for config in CONFIGS:
        matches = [r for r in rows if _matches(r, config)]
        if not matches:
            missing.append(config)
        elif any(r["asr"] is None for r in matches):
            needs_eval.append(config)
        else:
            complete.append(config)

    print(f"Total configurations: {len(CONFIGS)}")
    print(f"  Complete (has ASR):  {len(complete)}")
    print(f"  Needs eval (no ASR): {len(needs_eval)}")
    print(f"  Missing (no row):    {len(missing)}")

    if needs_eval:
        print(f"\n--- Needs eval ({len(needs_eval)}) ---")
        for c in needs_eval:
            model = _short_model(c["model_name"])
            print(f"  {c['experiment_name']:20s} {model:30s} max_ex={str(c['max_examples']):>4s} epoch={c['epoch']}")

    if missing:
        print(f"\n--- Missing ({len(missing)}) ---")
        for c in missing:
            model = _short_model(c["model_name"])
            print(f"  {c['experiment_name']:20s} {model:30s} max_ex={str(c['max_examples']):>4s} epoch={c['epoch']}")

    if not needs_eval and not missing:
        print("\nAll configurations complete!")


if __name__ == "__main__":
    main()
