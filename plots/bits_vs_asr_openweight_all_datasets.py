"""Plot bits vs success-rate with Pareto frontiers for open-weight models across all datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

OUTPUT_FILE = Path(__file__).parent / "bits_vs_asr_openweight_all_datasets.png"

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


def _bits(row: dict) -> float | None:
    v = row.get("bits")
    return v if v is not None else row.get("program_bits")


def _plot_panel(ax, rows, dataset_name: str, title: str) -> None:
    ax.set_xscale("log")

    pareto_data = []

    for model_key, color, marker, label in MODELS:
        model_rows = [
            r for r in rows
            if model_key in r["model_name"]
            and r["dataset_name"] == dataset_name
            and r["asr"] is not None
            and _bits(r) is not None
        ]
        if not model_rows:
            continue

        all_bits = np.array([_bits(r) for r in model_rows])
        all_asr = np.array([r["asr"] for r in model_rows]) * 100

        nonzero = all_bits > 0
        ax.scatter(
            all_bits[nonzero], all_asr[nonzero], marker=marker, color=color,
            edgecolors="black", linewidths=0.5, s=120, alpha=0.5, zorder=3,
        )
        pareto_data.append((all_bits, all_asr, color, label))

    xlim = ax.get_xlim()

    for all_bits, all_asr, color, label in pareto_data:
        p_bits, p_asr = pareto_frontier(all_bits, all_asr)

        baseline_mask = p_bits == 0
        baseline_asr = float(p_asr[baseline_mask].max()) if baseline_mask.any() else None
        p_bits = p_bits[~baseline_mask]
        p_asr = p_asr[~baseline_mask]

        if baseline_asr is not None:
            p_bits = np.concatenate([[xlim[0]], p_bits])
            p_asr = np.concatenate([[baseline_asr], p_asr])

        if len(p_bits) == 0:
            continue

        p_bits = np.concatenate([p_bits, [xlim[1]]])
        p_asr = np.concatenate([p_asr, [p_asr[-1]]])

        ax.step(p_bits, p_asr, "-", color=color, linewidth=4, label=label, where="post", zorder=4)

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
        ax.set_xlabel("Program Length (bits)")
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
