"""Plot bits vs ASR with Pareto frontiers for safe and unsafe models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path("/network/scratch/b/brownet/information-safety/results/final-results.jsonl")
OUTPUT_FILE = Path(__file__).parent / "bits_vs_asr.pdf"


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def pareto_frontier(
    bits: np.ndarray, asr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Pareto frontier maximizing ASR for a given bits budget.

    Returns sorted (bits, asr) arrays of Pareto-optimal points.
    """
    order = np.argsort(bits)
    sorted_bits = bits[order]
    sorted_asr = asr[order]

    pareto_bits = []
    pareto_asr = []
    running_max = -1.0

    for b, a in zip(sorted_bits, sorted_asr):
        if a > running_max:
            running_max = a
            pareto_bits.append(b)
            pareto_asr.append(a)

    return np.array(pareto_bits), np.array(pareto_asr)


def main() -> None:
    rows = load_results(RESULTS_FILE)

    series = [
        ("safety-pair-safe", "tab:blue", "o", "Safe (refusal-trained)"),
        ("safety-pair-unsafe", "tab:red", "s", "Unsafe (no refusals)"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xscale("log")

    pareto_data = []

    for model_key, color, marker, label in series:
        model_rows = [
            r for r in rows
            if model_key in r["model_name"]
            and r["dataset_name"] == "advbench_harmbench"
            and r["asr"] is not None
        ]

        all_bits = np.array([r["bits"] for r in model_rows])
        all_asr = np.array([r["asr"] for r in model_rows])

        nonzero = all_bits > 0
        ax.scatter(
            all_bits[nonzero], all_asr[nonzero], marker=marker, color=color,
            edgecolors="black", linewidths=0.5, s=40, alpha=0.5, zorder=3,
        )

        pareto_data.append((all_bits, all_asr, color, label))

    xlim = ax.get_xlim()

    for all_bits, all_asr, color, label in pareto_data:
        p_bits, p_asr = pareto_frontier(all_bits, all_asr)

        baseline_mask = p_bits == 0
        baseline_asr = float(p_asr[baseline_mask].max()) if baseline_mask.any() else None
        p_bits = p_bits[~baseline_mask]
        p_asr = p_asr[~baseline_mask]

        if baseline_asr is not None and len(p_bits) > 0:
            p_bits = np.concatenate([[xlim[0]], p_bits])
            p_asr = np.concatenate([[baseline_asr], p_asr])

        ax.step(p_bits, p_asr, "-", color=color, linewidth=2, label=label, where="post", zorder=4)

    ax.set_xlim(xlim)
    ax.set_xlabel("Bits")
    ax.set_ylabel("ASR")
    ax.set_title("Bits vs Attack Success Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
