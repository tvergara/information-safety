"""Decompose Pareto points on the tokens-combined plot into program-bits vs tokens-equivalent-
FLOPs-bits contributions, per (dataset, model)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")

BITS_PER_TOKEN = 16

MODELS = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct", 7.5e9),
    ("allenai/Olmo-3-7B-Instruct", "OLMo-3-7B-Instruct", 6.9e9),
    ("Qwen/Qwen3-4B", "Qwen3-4B", 3.6e9),
    ("openai/gpt-oss-20b", "gpt-oss-20b", 1.8e10),
]

DATASETS = [
    ("advbench_harmbench", "HarmBench"),
    ("strongreject", "StrongREJECT"),
    ("wmdp", "WMDP"),
    ("evilmath", "EvilMath"),
]


def tokens_bits_for_flops(flops: float, n_params: float) -> float:
    if flops <= 0 or n_params <= 0:
        return 0.0
    return BITS_PER_TOKEN * flops / (6 * n_params)


def _bits(row: dict) -> float | None:
    v = row.get("bits")
    return v if v is not None else row.get("program_bits")


def _total_flops(row: dict, generations_dir: Path) -> float | None:
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


def main() -> None:
    rows = load_results(DEFAULT_RESULTS_FILE)

    for dataset_name, dataset_title in DATASETS:
        print(f"\n{'=' * 12} {dataset_title} {'=' * 12}")
        for model_key, model_label, n_params in MODELS:
            model_rows = [
                r for r in rows
                if model_key in r["model_name"]
                and r["dataset_name"] == dataset_name
                and r["asr"] is not None
            ]
            decomposed: list[tuple[str, float, float, float, float]] = []
            for r in model_rows:
                flops = _total_flops(r, GENERATIONS_DIR)
                bits = _bits(r)
                if flops is None or bits is None:
                    continue
                tok_bits = tokens_bits_for_flops(flops, n_params)
                total = float(bits) + tok_bits
                decomposed.append((
                    r.get("experiment_name", "?"),
                    float(bits), tok_bits, total, float(r["asr"]) * 100,
                ))
            if not decomposed:
                continue

            totals = np.array([d[3] for d in decomposed])
            asrs = np.array([d[4] for d in decomposed])
            p_x, p_asr = pareto_frontier(totals, asrs)

            print(f"\n  -- {model_label} ({len(decomposed)} rows, {len(p_x)} pareto) --")
            print(f"  {'exp':28s}  {'prog_bits':>12s} {'tok_bits':>12s} "
                  f"{'total':>12s}  {'prog%':>6s} {'asr%':>6s}")
            for x, a in zip(p_x, p_asr):
                for exp, pb, tb, tot, asr in decomposed:
                    if math.isclose(tot, x) and math.isclose(asr, a):
                        pct = 100 * pb / tot if tot > 0 else 0
                        print(f"  {exp[:28]:28s}  {pb:12.2e} {tb:12.2e} {tot:12.2e}  "
                              f"{pct:5.1f}% {asr:5.1f}%")
                        break


if __name__ == "__main__":
    main()
