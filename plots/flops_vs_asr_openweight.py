"""Plot FLOPs vs ASR with Pareto frontiers for open-weight models.

Reads `independent_flops` from the result row and per-example `dependent_flops` from
`responses.jsonl` (joined via `eval_run_id`). The legacy "average total compute per
evaluated example" is recomputed at plot time as
`independent_flops / N + mean(dependent_flops)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")
OUTPUT_FILE = Path(__file__).parent / "flops_vs_asr_openweight.pdf"

MODELS = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "tab:purple", "D", "Llama-3-8B-Instruct"),
    ("allenai/Olmo-3-7B-Instruct", "tab:brown", "P", "OLMo-3-7B-Instruct"),
    ("Qwen/Qwen3-4B", "tab:orange", "s", "Qwen3-4B"),
]


def _avg_total_flops_per_example(row: dict, generations_dir: Path) -> float | None:
    """Compute `independent_flops / N + mean(dependent_flops)` from disk.

    Returns None if `responses.jsonl` is missing or empty.
    """
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
            dependent.append(int(entry["dependent_flops"]))

    if not dependent:
        return None

    independent = float(row.get("independent_flops", 0))
    return independent + float(np.mean(dependent))


def main() -> None:
    rows = load_results(DEFAULT_RESULTS_FILE)

    plt.rcParams.update({"font.size": 18})
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
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("ASR (%)")
    ax.set_title("HarmBench")
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
