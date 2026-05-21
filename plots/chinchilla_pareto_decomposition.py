"""For each Pareto-optimal point in the Chinchilla-combined plot, decompose its x-value into
program-bits vs FLOPs-equivalent-bits contributions.

This tells you which axis is doing the work along the frontier.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")

BITS_PER_PARAM = 16
CHINCHILLA_TOKENS_PER_PARAM = 20

MODELS = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct"),
    ("allenai/Olmo-3-7B-Instruct", "OLMo-3-7B-Instruct"),
    ("Qwen/Qwen3-4B", "Qwen3-4B"),
    ("openai/gpt-oss-20b", "gpt-oss-20b"),
]

DATASETS = [
    ("advbench_harmbench", "HarmBench"),
    ("strongreject", "StrongREJECT"),
    ("wmdp", "WMDP"),
    ("evilmath", "EvilMath"),
]


def chinchilla_bits_for_flops(flops: float) -> float:
    if flops <= 0:
        return 0.0
    n_opt = math.sqrt(flops / (6 * CHINCHILLA_TOKENS_PER_PARAM))
    return BITS_PER_PARAM * n_opt


def _bits(row: dict) -> float | None:
    v = row.get("bits")
    return v if v is not None else row.get("program_bits")


def _total_flops_per_example(row: dict, generations_dir: Path) -> float | None:
    eval_run_id = row.get("eval_run_id")
    if eval_run_id is None:
        return None
    responses_path = generations_dir / eval_run_id / "responses.jsonl"
    if not responses_path.exists():
        return None
    dependent: list[int] = []
    with open(responses_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            v = entry.get("dependent_flops")
            if v is None:
                return None
            dependent.append(int(v))
    if not dependent:
        return None
    independent = float(row.get("independent_flops", 0))
    return independent + float(np.mean(dependent))


def _decompose(row: dict) -> tuple[float, float, float] | None:
    flops = _total_flops_per_example(row, GENERATIONS_DIR)
    bits = _bits(row)
    if flops is None or bits is None:
        return None
    flops_bits = chinchilla_bits_for_flops(flops)
    return float(bits), flops_bits, float(bits) + flops_bits


def main() -> None:
    rows = load_results(DEFAULT_RESULTS_FILE)

    for dataset_name, dataset_title in DATASETS:
        print(f"\n{'=' * 12} {dataset_title} {'=' * 12}")
        for model_key, model_label in MODELS:
            model_rows = [
                r for r in rows
                if model_key in r["model_name"]
                and r["dataset_name"] == dataset_name
                and r["asr"] is not None
            ]
            decomposed: list[tuple[str, float, float, float, float]] = []
            for r in model_rows:
                d = _decompose(r)
                if d is None:
                    continue
                program_bits, flops_bits, total = d
                decomposed.append((
                    r.get("experiment_name", "?"),
                    program_bits, flops_bits, total, float(r["asr"]) * 100,
                ))
            if not decomposed:
                continue

            totals = np.array([d[3] for d in decomposed])
            asrs = np.array([d[4] for d in decomposed])
            p_x, p_asr = pareto_frontier(totals, asrs)

            print(f"\n  -- {model_label} ({len(decomposed)} rows, {len(p_x)} pareto) --")
            print(f"  {'exp':28s}  {'prog_bits':>12s} {'flops_bits':>12s} {'total':>12s}  "
                  f"{'prog%':>6s} {'asr%':>6s}")
            # For each Pareto point, find the matching row
            for x, a in zip(p_x, p_asr):
                # find the row that produced this point (first match)
                for exp, pb, fb, tot, asr in decomposed:
                    if math.isclose(tot, x) and math.isclose(asr, a):
                        pct = 100 * pb / tot if tot > 0 else 0
                        print(f"  {exp[:28]:28s}  {pb:12.2e} {fb:12.2e} {tot:12.2e}  "
                              f"{pct:5.1f}% {asr:5.1f}%")
                        break


if __name__ == "__main__":
    main()
