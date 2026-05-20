"""Combined program-length + Hernandez-transfer fine-tuning bits vs success-rate.

Hernandez et al. (2021), "Scaling Laws for Transfer":

    D_T(D_F, N) = k * D_F^alpha * N^beta

where D_T is the "effective fine-tune tokens transferred" from pre-training
to a fine-tune of D_F tokens on a pretrained model of N non-embedding params.

Reported constants (text -> Python code, Table 2 of the paper):
    alpha = 0.18, beta = 0.38, k = 1.9e4.

Caveat: these constants are task-pair specific. For our jailbreak/forget
fine-tunes the structure should hold approximately, but k/alpha/beta may
differ. Use as a starting point.

Combined x-axis = program_length_bits + bits_per_token * (D_F + D_T)
where D_F is recovered from train compute via D_F = C_train / (6 * N).
Inference (eval) compute is not converted via Hernandez (transfer scaling
only applies to training).
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
    Path(__file__).parent / "hernandez_combined_vs_asr_openweight_all_datasets.png"
)

HERN_ALPHA = 0.18
HERN_BETA = 0.38
HERN_K = 1.9e4
BITS_PER_TOKEN = 2.0

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


def hernandez_effective_tokens(train_flops: float, n_params: float) -> float:
    """D_F + D_T(D_F, N), where D_F is recovered from train compute."""
    if train_flops <= 0 or n_params <= 0:
        return 0.0
    d_f = train_flops / (6 * n_params)
    d_t = HERN_K * (d_f ** HERN_ALPHA) * (n_params ** HERN_BETA)
    return d_f + d_t


def _bits(row: dict) -> float | None:
    v = row.get("bits")
    return v if v is not None else row.get("program_bits")


def _n_params_for_row(row: dict) -> float | None:
    for key, n in MODEL_NUM_PARAMS.items():
        if key in row["model_name"]:
            return n
    return None


def _row_has_flops(row: dict, generations_dir: Path) -> bool:
    """Confirm responses.jsonl exists and has dependent_flops populated."""
    eval_run_id = row.get("eval_run_id")
    if eval_run_id is None:
        return False
    responses_path = generations_dir / eval_run_id / "responses.jsonl"
    if not responses_path.exists():
        return False
    with open(responses_path) as f:
        first = f.readline().strip()
    if not first:
        return False
    return json.loads(first).get("dependent_flops") is not None


def _combined_x(row: dict, generations_dir: Path) -> float | None:
    if not _row_has_flops(row, generations_dir):
        return None
    bits = _bits(row)
    n_params = _n_params_for_row(row)
    if bits is None or n_params is None:
        return None
    train_flops = float(row.get("independent_flops", 0))
    eff_tokens = hernandez_effective_tokens(train_flops, n_params)
    return float(bits) + BITS_PER_TOKEN * eff_tokens


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
        ax.set_xlabel(r"program bits + Hernandez-equivalent fine-tune bits")
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
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
