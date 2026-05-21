"""Plot FLOPs vs success-rate with Pareto frontiers for open-weight models across all datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")
OUTPUT_FILE = Path(__file__).parent / "flops_vs_asr_openweight_all_datasets.png"

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


def _avg_total_flops_per_example(row: dict, generations_dir: Path) -> float | None:
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
        flops_list: list[float] = []
        asr_list: list[float] = []
        for r in model_rows:
            avg_flops = _avg_total_flops_per_example(r, GENERATIONS_DIR)
            if avg_flops is None:
                continue
            flops_list.append(avg_flops)
            asr_list.append(float(r["asr"]) * 100)

        if not flops_list:
            continue

        all_flops = np.array(flops_list)
        all_asr = np.array(asr_list)

        nonzero = all_flops > 0
        ax.scatter(
            all_flops[nonzero], all_asr[nonzero], marker=marker, color=color,
            edgecolors="black", linewidths=0.5, s=120, alpha=0.5, zorder=3,
        )
        pareto_data.append((all_flops, all_asr, color, label))

    xlim = ax.get_xlim()

    for all_flops, all_asr, color, label in pareto_data:
        p_flops, p_asr = pareto_frontier(all_flops, all_asr)

        baseline_mask = p_flops == 0
        baseline_asr = float(p_asr[baseline_mask].max()) if baseline_mask.any() else None
        p_flops = p_flops[~baseline_mask]
        p_asr = p_asr[~baseline_mask]

        if baseline_asr is not None:
            p_flops = np.concatenate([[xlim[0]], p_flops])
            p_asr = np.concatenate([[baseline_asr], p_asr])

        if len(p_flops) == 0:
            continue

        p_flops = np.concatenate([p_flops, [xlim[1]]])
        p_asr = np.concatenate([p_asr, [p_asr[-1]]])

        ax.step(p_flops, p_asr, "-", color=color, linewidth=4, label=label, where="post", zorder=4)

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
        ax.set_xlabel("FLOPs")
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
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_FILE.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
