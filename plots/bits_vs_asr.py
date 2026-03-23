"""Plot bits vs ASR from the sweep results."""

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
            row = json.loads(line)
            rows.append(row)
    return rows


def main() -> None:
    rows = load_results(RESULTS_FILE)

    bits = np.array([r["bits"] for r in rows])
    asr = np.array([r["asr"] for r in rows])
    max_examples = [r.get("max_examples") for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(bits, asr, c=bits, cmap="viridis", edgecolors="black", s=60, zorder=3)

    for b, a, me in zip(bits, asr, max_examples):
        label = str(me) if me is not None else "all"
        ax.annotate(label, (b, a), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Bits")
    ax.set_ylabel("ASR")
    ax.set_title("Bits vs Attack Success Rate")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
