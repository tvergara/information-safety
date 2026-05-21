"""Program bits vs success-rate for safety-trained Llama-3.1-8B defenses on WMDP and EvilMath.

Bits-only sibling of tokens_combined_vs_asr_defended_wmdp_evilmath.py: same panels and
defenses, but the x-axis is the raw `bits` (program_bits) without the
tokens-equivalent FLOPs term.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

OUTPUT_FILE = Path(__file__).parent / "bits_vs_asr_defended_wmdp_evilmath.png"

DEFENSES_WMDP = [
    ("cb-wmdp-Llama-3.1-8B-Instruct-bfbf3e38793c", "tab:red", "D", "CB (WMDP)"),
    ("sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0", "tab:blue", "s", "SFT (WMDP)"),
    ("tar-wmdp-Llama-3.1-8B-Instruct-73d8c8e83c07", "tab:green", "o", "TAR (WMDP)"),
]
DEFENSES_EVILMATH = [
    ("cb-evilmath-Llama-3.1-8B-Instruct-d7ba262bbc28", "tab:red", "D", "CB (EvilMath)"),
    ("sft-evilmath-Llama-3.1-8B-Instruct-d650794f965d", "tab:blue", "s", "SFT (EvilMath)"),
    ("tar-evilmath-Llama-3.1-8B-Instruct-09003ee4e852", "tab:green", "o", "TAR (EvilMath)"),
]

PANELS = [
    ("wmdp", "WMDP", DEFENSES_WMDP),
    ("evilmath", "EvilMath", DEFENSES_EVILMATH),
]


def _bits(row: dict) -> float | None:
    v = row.get("bits")
    return v if v is not None else row.get("program_bits")


def _plot_panel(ax, rows, dataset_name: str, title: str, defenses: list) -> None:
    ax.set_xscale("log")
    pareto_data = []

    for model_key, color, marker, label in defenses:
        model_rows = [
            r for r in rows
            if model_key in r["model_name"]
            and r["dataset_name"] == dataset_name
            and r["asr"] is not None
        ]
        x_list: list[float] = []
        asr_list: list[float] = []
        for r in model_rows:
            b = _bits(r)
            if b is None:
                continue
            x_list.append(float(b))
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

    if pareto_data:
        all_x_concat = np.concatenate([d[0] for d in pareto_data])
        xlim = (all_x_concat.min() * 0.5, all_x_concat.max() * 2)
    else:
        xlim = (1.0, 1e7)

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (dataset_name, title, defenses) in zip(axes, PANELS):
        _plot_panel(ax, rows, dataset_name, title, defenses)

    for ax in axes:
        ax.set_xlabel("program bits")
    axes[0].set_ylabel("Success rate (%)")

    handles_by_panel = [ax.get_legend_handles_labels() for ax in axes]
    for ax, (handles, labels) in zip(axes, handles_by_panel):
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=12, frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_FILE.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
