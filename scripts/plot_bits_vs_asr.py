"""Plot bits vs ASR for attack result rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from information_safety.algorithms.jailbreak.metrics import compute_text_bits

Point = tuple[str, float, float]


def load_rows(results_file: str) -> list[dict[str, Any]]:
    """Load rows from newline-delimited JSON."""
    rows: list[dict[str, Any]] = []
    with open(results_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def build_points(rows: list[dict[str, Any]]) -> list[Point]:
    """Extract (method, bits, asr) points from results rows."""
    points: list[Point] = []
    for row in rows:
        method = row.get("attack_method") or row.get("experiment_name")
        bits = row.get("bits")
        asr = row.get("asr")
        if bits is None:
            attack_text = row.get("attack_prompt") or row.get("prompt") or row.get("attack_suffix")
            if isinstance(attack_text, str):
                bits = float(compute_text_bits(attack_text))
        if method is None or bits is None or asr is None:
            continue
        if isinstance(bits, int | float) and isinstance(asr, int | float):
            points.append((str(method), float(bits), float(asr)))
    return points


def plot_bits_vs_asr(points: list[Point], output_file: str) -> None:
    """Create and save bits-vs-ASR scatter plot."""
    if not points:
        raise ValueError("no valid points to plot")

    fig, ax = plt.subplots(figsize=(9, 6))
    for method, bits, asr in points:
        ax.scatter(bits, asr, label=method, alpha=0.8)

    ax.set_xlabel("Bits")
    ax.set_ylabel("ASR")
    ax.set_title("Bits vs ASR by Attack Method")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bits vs ASR from results JSONL")
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    rows = load_rows(args.results_file)
    points = build_points(rows)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_bits_vs_asr(points, str(output_path))


if __name__ == "__main__":
    main()
