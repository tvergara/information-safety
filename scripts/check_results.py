"""Check that all desired experiment configurations exist in the results file."""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Any


def default_results_file() -> Path:
    return Path(f"{os.environ['SCRATCH']}/information-safety/results/final-results.jsonl")

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "Qwen/Qwen3-4B",
    "openai/gpt-oss-20b",
]

WMDP_DEFENSES = [
    "sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0",
    "cb-wmdp-Llama-3.1-8B-Instruct-bfbf3e38793c",
    "tar-wmdp-Llama-3.1-8B-Instruct-73d8c8e83c07",
]
EVILMATH_DEFENSES = [
    "sft-evilmath-Llama-3.1-8B-Instruct-d650794f965d",
    "cb-evilmath-Llama-3.1-8B-Instruct-d7ba262bbc28",
    "tar-evilmath-Llama-3.1-8B-Instruct-09003ee4e852",
]

PRECOMPUTED_ATTACKS = [
    "PrecomputedGCGStrategy",
    "PrecomputedAutoDANStrategy",
    "PrecomputedPAIRStrategy",
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

for model, attack in product(MODELS, PRECOMPUTED_ATTACKS):
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

# WMDP RoleplayStrategy: one per base model, epoch 0 only (no training)
for model in MODELS:
    WMDP_CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "wmdp",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# WMDP DataStrategy: MMLU-Pro only (corpus_fraction=0.0)
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

# WMDP DataStrategy: MMLU-Pro + WMDP forget-corpus mix (corpus_fraction=0.5)
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

# WMDP defense evaluations: each defense gets Baseline + Roleplay + the three
# DataStrategy corpus variants. WMDP is ~12x larger than the other datasets, so
# precomputed adversarial-prompt attacks (GCG/AutoDAN/PAIR) are intentionally
# omitted — the per-example optimization cost is prohibitive.
for defense in WMDP_DEFENSES:
    WMDP_CONFIGS.append({
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": defense,
        "max_examples": None,
        "epoch": 0,
    })
    WMDP_CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "wmdp",
        "model_name": defense,
        "max_examples": None,
        "epoch": 0,
    })
for defense, (max_ex, max_epochs) in product(WMDP_DEFENSES, DATA_SWEEP_POINTS):
    for epoch in range(max_epochs):
        WMDP_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "wmdp",
            "model_name": defense,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
            "corpus_fraction": 0.0,
            "corpus_subset": None,
        })
for defense, (max_ex, max_epochs), subset in product(
    WMDP_DEFENSES, DATA_SWEEP_POINTS, ["bio", "cyber"]
):
    for epoch in range(max_epochs):
        WMDP_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "wmdp",
            "model_name": defense,
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

# EvilMath RoleplayStrategy: one per base model, epoch 0 only (no training)
for model in MODELS:
    EVILMATH_CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "evilmath",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# EvilMath DataStrategy: GSM8K instruction-following corpus (single variant, no fraction axis)
for model, (max_ex, max_epochs) in product(MODELS, DATA_SWEEP_POINTS):
    for epoch in range(max_epochs):
        EVILMATH_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "evilmath",
            "model_name": model,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
        })

# EvilMath defense evaluations: full attack matrix per defense_id.
for defense in EVILMATH_DEFENSES:
    EVILMATH_CONFIGS.append({
        "experiment_name": "BaselineStrategy",
        "dataset_name": "evilmath",
        "model_name": defense,
        "max_examples": None,
        "epoch": 0,
    })
    EVILMATH_CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "evilmath",
        "model_name": defense,
        "max_examples": None,
        "epoch": 0,
    })
    for attack in PRECOMPUTED_ATTACKS:
        EVILMATH_CONFIGS.append({
            "experiment_name": attack,
            "dataset_name": "evilmath",
            "model_name": defense,
            "max_examples": None,
            "epoch": 0,
        })
for defense, (max_ex, max_epochs) in product(EVILMATH_DEFENSES, DATA_SWEEP_POINTS):
    for epoch in range(max_epochs):
        EVILMATH_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "evilmath",
            "model_name": defense,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
        })

STRONGREJECT_CONFIGS: list[dict[str, Any]] = []

# StrongREJECT baseline: one per model, epoch 0 only (no training)
for model in MODELS:
    STRONGREJECT_CONFIGS.append({
        "experiment_name": "BaselineStrategy",
        "dataset_name": "strongreject",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# StrongREJECT compute-matched DataStrategy sweep
for model, (max_ex, max_epochs) in product(MODELS, DATA_SWEEP_POINTS):
    for epoch in range(max_epochs):
        STRONGREJECT_CONFIGS.append({
            "experiment_name": "DataStrategy",
            "dataset_name": "strongreject",
            "model_name": model,
            "max_examples": max_ex,
            "max_epochs": max_epochs,
            "epoch": epoch,
        })

# StrongREJECT RoleplayStrategy: one per model, epoch 0 only (no training)
for model in MODELS:
    STRONGREJECT_CONFIGS.append({
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "strongreject",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

# StrongREJECT precomputed adversarial-prompt attacks (GCG, AutoDAN, PAIR)
for model, attack in product(MODELS, PRECOMPUTED_ATTACKS):
    STRONGREJECT_CONFIGS.append({
        "experiment_name": attack,
        "dataset_name": "strongreject",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    })

CONFIGS = HARMBENCH_CONFIGS + WMDP_CONFIGS + EVILMATH_CONFIGS + STRONGREJECT_CONFIGS


def _short_model(name: str) -> str:
    """Shorten model name for display."""
    return name.rsplit("/", 1)[-1]


def _matches(row: dict[str, Any], config: dict[str, Any]) -> bool:
    for k, v in config.items():
        if k == "model_name":
            row_model = row.get(k)
            if not isinstance(row_model, str) or v not in row_model:
                return False
        elif row.get(k) != v:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Check results completeness")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=default_results_file(),
        help="Path to results JSONL file",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.results_file)]
    rows = [r for r in rows if r.get("invalidated_reason") is None]

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
    _report("StrongREJECT", STRONGREJECT_CONFIGS)


if __name__ == "__main__":
    main()
