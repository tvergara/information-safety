"""Combined program-length + tokens-equivalent FLOPs bits vs success-rate.

Conversion uses C = 6 * N * D (transformer training compute), solved for D
with N = pretrained model size. Each token is treated as 16 bits.

    bits(C, N) = 16 * D = 16 * C / (6 * N) = (8/3) * C / N

This is the natural fine-tuning analogue of the Chinchilla-params conversion:
Chinchilla assumes you re-pretrain optimally (N = sqrt(C/120)); this version
holds N fixed at the actual pretrained model and asks "how many fine-tune
tokens does this compute buy at 16 bits/token?".
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist

from plots.utils import DEFAULT_RESULTS_FILE, load_results, pareto_frontier

GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")
OUTPUT_FILE = (
    Path(__file__).parent / "tokens_combined_vs_asr_openweight_all_datasets.png"
)

BITS_PER_TOKEN = 16

MODELS = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "tab:purple", "D", "Llama-3-8B-Instruct", 7.5e9),
    ("allenai/Olmo-3-7B-Instruct", "tab:brown", "P", "OLMo-3-7B-Instruct", 6.9e9),
    ("Qwen/Qwen3-4B", "tab:orange", "s", "Qwen3-4B", 3.6e9),
    ("openai/gpt-oss-20b", "tab:blue", "o", "gpt-oss-20b", 1.8e10),
]
MODEL_NUM_PARAMS = {key: n for (key, _, _, _, n) in MODELS}

DATASETS = [
    ("advbench_harmbench", "HarmBench"),
    ("strongreject", "StrongREJECT"),
    ("wmdp", "WMDP"),
    ("evilmath", "EvilMath"),
]


def tokens_bits_for_flops(flops: float, n_params: float) -> float:
    if flops <= 0 or n_params <= 0:
        return 0.0
    tokens = flops / (6 * n_params)
    return BITS_PER_TOKEN * tokens


def _bits(row: dict) -> float | None:
    v = row.get("bits")
    return v if v is not None else row.get("program_bits")


def _n_params_for_row(row: dict) -> float | None:
    for key, n in MODEL_NUM_PARAMS.items():
        if key in row["model_name"]:
            return n
    return None


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


def _combined_x(row: dict, generations_dir: Path) -> float | None:
    flops = _total_flops(row, generations_dir)
    bits = _bits(row)
    n_params = _n_params_for_row(row)
    if flops is None or bits is None or n_params is None:
        return None
    return float(bits) + tokens_bits_for_flops(flops, n_params)


def _plot_panel(ax, rows, dataset_name: str, title: str) -> None:
    ax.set_xscale("log")
    pareto_data = []

    for model_key, color, marker, label, _ in MODELS:
        model_rows = [
            r for r in rows
            if model_key in r["model_name"]
            and r["dataset_name"] == dataset_name
            and r["asr"] is not None
        ]
        x_list: list[float] = []
        asr_list: list[float] = []
        for r in model_rows:
            x = _combined_x(r, GENERATIONS_DIR)
            if x is None:
                continue
            x_list.append(x)
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

    xlim = ax.get_xlim()

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)

    for ax, (dataset_name, title) in zip(axes.flat, DATASETS):
        _plot_panel(ax, rows, dataset_name, title)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"program bits + $16\,C/(6N)$ (tokens-equivalent bits)")
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
