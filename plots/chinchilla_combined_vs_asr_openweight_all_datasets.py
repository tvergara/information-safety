"""Combined program-length + Chinchilla-equivalent FLOPs bits vs success-rate.

Chinchilla scaling: C ~= 6 N D with D ~= 20 N => N_opt = sqrt(C / 120).
At 16 bits/param: bits_equiv(C) = 16 * sqrt(C / 120).

Combined x-axis: program_length_bits + bits_equiv(total_flops).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")
OUTPUT_FILE = (
    Path(__file__).parent / "chinchilla_combined_vs_asr_openweight_all_datasets.png"
)

BITS_PER_PARAM = 16
CHINCHILLA_TOKENS_PER_PARAM = 20

MODELS = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "tab:purple", "D", "Llama-3-8B-Instruct"),
    ("allenai/Olmo-3-7B-Instruct", "tab:brown", "P", "OLMo-3-7B-Instruct"),
    ("Qwen/Qwen3-4B", "tab:orange", "s", "Qwen3-4B"),
    ("openai/gpt-oss-20b", "tab:blue", "o", "gpt-oss-20b"),
]

DATASETS = [
    ("advbench_harmbench", "HarmBench"),
    ("strongreject", "StrongREJECT"),
    ("wmdp", "WMDP"),
    ("evilmath", "EvilMath"),
]


def chinchilla_bits_for_flops(flops: float) -> float:
    """Bits-equivalent of a compute budget under Chinchilla-optimal training."""
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


def _combined_x(row: dict, generations_dir: Path) -> float | None:
    flops = _total_flops_per_example(row, generations_dir)
    bits = _bits(row)
    if flops is None or bits is None:
        return None
    return float(bits) + chinchilla_bits_for_flops(flops)


def _plot_panel(ax, rows, dataset_name: str, title: str) -> None:
    ax.set_xscale("log")
    pareto_data = []

    for model_key, color, marker, label in MODELS:
        model_rows = [
            r for r in rows
            if model_key in r["model_name"]
            and r["dataset_name"] == dataset_name
            and r["asr"] is not None
        ]
        x_list: list[float] = []
        asr_list: list[float] = []
        for r in model_rows:
            x = _combined_x(r, GENERATIONS_DIR)
            if x is None:
                continue
            x_list.append(x)
            asr_list.append(float(r["asr"]) * 100)

        if not x_list:
            continue

        all_x = np.array(x_list)
        all_asr = np.array(asr_list)

        ax.scatter(
            all_x, all_asr, marker=marker, color=color,
            edgecolors="black", linewidths=0.5, s=120, alpha=0.5, zorder=3,
        )
        pareto_data.append((all_x, all_asr, color, label))

    xlim = ax.get_xlim()

    for all_x, all_asr, color, label in pareto_data:
        p_x, p_asr = pareto_frontier(all_x, all_asr)
        if len(p_x) == 0:
            continue
        p_x = np.concatenate([p_x, [xlim[1]]])
        p_asr = np.concatenate([p_asr, [p_asr[-1]]])
        ax.step(p_x, p_asr, "-", color=color, linewidth=4, label=label, where="post", zorder=4)

    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main() -> None:
    rows = load_results(DEFAULT_RESULTS_FILE)

    plt.rcParams.update({"font.size": 16})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)

    for ax, (dataset_name, title) in zip(axes.flat, DATASETS):
        _plot_panel(ax, rows, dataset_name, title)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"program bits + $16\,\sqrt{C/120}$ (Chinchilla-equivalent bits)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Success rate (%)")

    seen: dict[str, Artist] = {}
    for ax in axes.flat:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen:
                seen[lbl] = h
    if seen:
        fig.legend(
            list(seen.values()), list(seen.keys()), loc="upper center",
            ncol=len(seen), bbox_to_anchor=(0.5, 1.02), fontsize=14, frameon=False,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
