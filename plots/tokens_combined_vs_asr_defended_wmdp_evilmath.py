"""Tokens-equivalent FLOPs bits vs success-rate for safety-trained Llama-3.1-8B defenses on WMDP
and EvilMath.

Mirrors tokens_combined_vs_asr_openweight_all_datasets.py but:
- Restricted to WMDP and EvilMath panels.
- Models are CB/SFT/TAR defenses on Llama-3.1-8B-Instruct, each shown only on
  the dataset it was trained against.
- N = 8B for the tokens-equivalent FLOPs term (Llama-3.1-8B pretrained size).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")
OUTPUT_FILE = (
    Path(__file__).parent / "tokens_combined_vs_asr_defended_wmdp_evilmath.png"
)

BITS_PER_TOKEN = 16
LLAMA_3_1_8B_N = 8.0e9

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


def tokens_bits_for_flops(flops: float, n_params: float) -> float:
    if flops <= 0 or n_params <= 0:
        return 0.0
    tokens = flops / (6 * n_params)
    return BITS_PER_TOKEN * tokens


def _bits(row: dict) -> float | None:
    v = row.get("bits")
    return v if v is not None else row.get("program_bits")


def _total_flops(row: dict, generations_dir: Path) -> float | None:
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


def _combined_x(row: dict) -> float | None:
    flops = _total_flops(row, GENERATIONS_DIR)
    bits = _bits(row)
    if flops is None or bits is None:
        return None
    return float(bits) + tokens_bits_for_flops(flops, LLAMA_3_1_8B_N)


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
            x = _combined_x(r)
            if x is None:
                continue
            x_list.append(x)
            asr_list.append(float(r["asr"]) * 100)
        if not x_list:
            continue

        all_x = np.array(x_list)
        all_asr = np.array(asr_list)
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
        ax.set_xlabel(r"program bits + $16\,C/(6N)$ (tokens-equivalent bits)")
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
