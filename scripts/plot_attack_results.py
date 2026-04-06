"""Plot aggregated ASR by method from attack result JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_rows(results_file: str) -> list[dict[str, Any]]:
    """Load rows from a newline-delimited JSON results file."""
    rows: list[dict[str, Any]] = []
    with open(results_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def aggregate_asr(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-method average ASR from results rows."""
    grouped: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        method = row.get("attack_method") or row.get("experiment_name")
        asr = row.get("asr")
        if method is None or asr is None:
            continue
        if isinstance(asr, int | float):
            grouped[str(method)].append(float(asr))

    return {
        method: sum(values) / len(values)
        for method, values in grouped.items()
        if values
    }


def plot_asr(summary: dict[str, float], output_file: str) -> None:
    """Create and save a bar plot for ASR summary."""
    if not summary:
        raise ValueError("no ASR entries found to plot")

    methods = sorted(summary)
    values = [summary[m] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(methods, values)
    ax.set_xlabel("Method")
    ax.set_ylabel("ASR")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Attack Success Rate by Method")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ASR by method from results JSONL")
    parser.add_argument("--results-file", required=True, help="Path to JSONL results file")
    parser.add_argument("--output-file", required=True, help="Output image path (png)")
    args = parser.parse_args()

    rows = load_rows(args.results_file)
    summary = aggregate_asr(rows)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_asr(summary, str(output_path))


if __name__ == "__main__":
    main()
