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
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "Qwen/Qwen3-4B",
    "openai/gpt-oss-20b",
]

ATTACK_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "Qwen/Qwen3-4B",
    "/network/scratch/b/brownet/information-safety/models/smollm3-circuit-breaker",
]

HARMBENCH_CONFIGS: list[dict[str, Any]] = []

# Baseline: one per model, epoch 0 only (no training)
for model in MODELS:
    HARMBENCH_CONFIGS.append({
        "experiment_name": "BaselineStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# Compute-matched DataStrategy sweep: max_examples * max_epochs = 1024
DATA_SWEEP_POINTS = [
    (16, 64),
    (32, 32),
    (64, 16),
    (128, 8),
    (256, 4),
    (512, 2),
]
for model, (max_ex, max_epochs) in product(MODELS, DATA_SWEEP_POINTS):
    for epoch in range(max_epochs):
        HARMBENCH_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": model,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
        })

# RoleplayStrategy: one per model, epoch 0 only (no training)
for model in MODELS:
    HARMBENCH_CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# Precomputed adversarial-prompt attacks (GCG, AutoDAN, PAIR): one per attack model,
# epoch 0 only (no training — the attack prompts come from an upstream run).
for model, attack in product(
    ATTACK_MODELS,
    ["PrecomputedGCGStrategy", "PrecomputedAutoDANStrategy", "PrecomputedPAIRStrategy"],
):
    HARMBENCH_CONFIGS.append({
        "experiment_name": attack,
        "dataset_name": "advbench_harmbench",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

WMDP_CONFIGS: list[dict[str, Any]] = []

# WMDP baseline: one per model, epoch 0 only (no training)
for model in MODELS:
    WMDP_CONFIGS.append({
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# WMDP RoleplayStrategy: one per model, epoch 0 only (no training)
for model in MODELS:
    WMDP_CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "wmdp",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# WMDP DataStrategy: compliance only (corpus_fraction=0.0)
for model, (max_ex, max_epochs) in product(MODELS, DATA_SWEEP_POINTS):
    for epoch in range(max_epochs):
        WMDP_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "wmdp",
            "model_name": model,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
            "corpus_fraction": 0.0,
            "corpus_subset": None,
        })

# WMDP DataStrategy: compliance + WMDP forget-corpus mix (corpus_fraction=0.5)
for model, (max_ex, max_epochs), subset in product(
    MODELS, DATA_SWEEP_POINTS, ["bio", "cyber"]
):
    for epoch in range(max_epochs):
        WMDP_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "wmdp",
            "model_name": model,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
            "corpus_fraction": 0.5,
            "corpus_subset": subset,
        })

EVILMATH_CONFIGS: list[dict[str, Any]] = []

# EvilMath baseline: one per model, epoch 0 only (no training)
for model in MODELS:
    EVILMATH_CONFIGS.append({
        "experiment_name": "BaselineStrategy",
        "dataset_name": "evilmath",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# EvilMath RoleplayStrategy: one per model, epoch 0 only (no training)
for model in MODELS:
    EVILMATH_CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "evilmath",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# EvilMath DataStrategy: compliance only (corpus_fraction=0.0)
for model, (max_ex, max_epochs) in product(MODELS, DATA_SWEEP_POINTS):
    for epoch in range(max_epochs):
        EVILMATH_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "evilmath",
            "model_name": model,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
            "corpus_fraction": 0.0,
        })

# EvilMath DataStrategy: compliance + GSM8K capability mix (corpus_fraction=0.5)
for model, (max_ex, max_epochs) in product(MODELS, DATA_SWEEP_POINTS):
    for epoch in range(max_epochs):
        EVILMATH_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "evilmath",
            "model_name": model,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
            "corpus_fraction": 0.5,
        })

CONFIGS = HARMBENCH_CONFIGS + WMDP_CONFIGS + EVILMATH_CONFIGS


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

    def _check(configs: list[dict[str, Any]]) -> tuple[list, list, list]:
        missing: list[dict[str, Any]] = []
        needs_eval: list[dict[str, Any]] = []
        complete: list[dict[str, Any]] = []
        for config in configs:
            matches = [r for r in rows if _matches(r, config)]
            if not matches:
                missing.append(config)
            elif any(r["asr"] is None for r in matches):
                needs_eval.append(config)
            else:
                complete.append(config)
        return missing, needs_eval, complete

    def _report(label: str, configs: list[dict[str, Any]]) -> None:
        missing, needs_eval, complete = _check(configs)
        print(f"\n{'=' * 10} {label} {'=' * 10}")
        print(f"Total: {len(configs)}  Complete: {len(complete)}  Needs eval: {len(needs_eval)}  Missing: {len(missing)}")
        if needs_eval:
            print(f"\n  Needs eval ({len(needs_eval)}):")
            for c in needs_eval:
                model = _short_model(c["model_name"])
                print(f"    {c['experiment_name']:25s} {model:30s} max_ex={str(c['max_examples']):>4s} epoch={c['epoch']}")
        if missing:
            print(f"\n  Missing ({len(missing)}):")
            for c in missing:
                model = _short_model(c["model_name"])
                print(f"    {c['experiment_name']:25s} {model:30s} max_ex={str(c['max_examples']):>4s} epoch={c['epoch']}")

    _report("HarmBench", HARMBENCH_CONFIGS)
    _report("WMDP", WMDP_CONFIGS)
    _report("EvilMath", EVILMATH_CONFIGS)


if __name__ == "__main__":
    main()
