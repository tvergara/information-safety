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
            rows.append(json.loads(line))
    return rows


def best_asr_per_run(rows: list[dict]) -> list[dict]:
    """For runs with multiple epochs, keep the one with highest ASR."""
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (r["model_name"], r["max_examples"])
        if key not in by_key or r["asr"] > by_key[key]["asr"]:
            by_key[key] = r
    return list(by_key.values())


def main() -> None:
    rows = load_results(RESULTS_FILE)

    safe_rows = best_asr_per_run([
        r for r in rows
        if "safety-pair-safe" in r["model_name"]
        and r["dataset_name"] == "advbench_harmbench"
        and r["asr"] is not None
    ])
    unsafe_rows = best_asr_per_run([
        r for r in rows
        if "safety-pair-unsafe" in r["model_name"]
        and r["dataset_name"] == "advbench_harmbench"
        and r["asr"] is not None
    ])

    safe_rows.sort(key=lambda r: r["bits"])
    unsafe_rows.sort(key=lambda r: r["bits"])

    fig, ax = plt.subplots(figsize=(6, 4))

    safe_bits = np.array([r["bits"] for r in safe_rows])
    safe_asr = np.array([r["asr"] for r in safe_rows])
    unsafe_bits = np.array([r["bits"] for r in unsafe_rows])
    unsafe_asr = np.array([r["asr"] for r in unsafe_rows])

    ax.plot(safe_bits, safe_asr, "o-", color="tab:blue", label="Safe (refusal-trained)", zorder=3)
    ax.plot(unsafe_bits, unsafe_asr, "s-", color="tab:red", label="Unsafe (no refusals)", zorder=3)

    for series_bits, series_asr, series_rows in [
        (safe_bits, safe_asr, safe_rows),
        (unsafe_bits, unsafe_asr, unsafe_rows),
    ]:
        for bits, asr, r in zip(series_bits, series_asr, series_rows):
            label = str(r["max_examples"]) if r["max_examples"] is not None else "all"
            ax.annotate(label, (bits, asr), textcoords="offset points", xytext=(6, 4), fontsize=7)

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
