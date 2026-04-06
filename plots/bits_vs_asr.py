"""Plot bits vs ASR with Pareto frontiers across multiple models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path("/network/scratch/b/brownet/information-safety/results/final-results.jsonl")
OUTPUT_FILE = Path(__file__).parent / "bits_vs_asr.pdf"

MODELS = [
    ("safety-pair-safe", "tab:blue", "o", "SmolLM3-3B (safe)"),
    ("safety-pair-unsafe", "tab:red", "s", "SmolLM3-3B (unsafe)"),
    ("meta-llama/Llama-2-7b-chat-hf", "tab:green", "^", "Llama-2-7B-Chat"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "tab:purple", "D", "Llama-3-8B-Instruct"),
]


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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")

    pareto_data = []

    for model_key, color, marker, label in MODELS:
        model_rows = [
            r for r in rows
            if model_key in r["model_name"]
            and r["dataset_name"] == "advbench_harmbench"
            and r["asr"] is not None
        ]

        if not model_rows:
            continue

        all_bits = np.array([r["bits"] for r in model_rows])
        all_asr = np.array([r["asr"] for r in model_rows]) * 100

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

        if baseline_asr is not None:
            p_bits = np.concatenate([[xlim[0]], p_bits])
            p_asr = np.concatenate([[baseline_asr], p_asr])

        if len(p_bits) == 0:
            continue

        p_bits = np.concatenate([p_bits, [xlim[1]]])
        p_asr = np.concatenate([p_asr, [p_asr[-1]]])

        ax.step(p_bits, p_asr, "-", color=color, linewidth=2, label=label, where="post", zorder=4)

    ax.set_xlim(xlim)
    ax.set_xlabel("Program Length (bits)")
    ax.set_ylabel("ASR (%)")
    ax.set_title("HarmBench")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
