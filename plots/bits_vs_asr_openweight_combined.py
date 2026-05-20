"""Plot a single Pareto frontier across open-weight models, colored by contributing model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.utils import DEFAULT_RESULTS_FILE, load_results

OUTPUT_FILE = Path(__file__).parent / "bits_vs_asr_openweight_combined.pdf"

MODELS: list[tuple[str, str, str, str]] = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "tab:purple", "D", "Llama-3-8B-Instruct"),
    ("allenai/Olmo-3-7B-Instruct", "tab:brown", "P", "OLMo-3-7B-Instruct"),
    ("Qwen/Qwen3-4B", "tab:orange", "s", "Qwen3-4B"),
]


def pareto_frontier(
    bits: np.ndarray, asr: np.ndarray, models: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pareto frontier maximizing ASR for given bits, tracking contributing model."""
    order = np.argsort(bits)
    sorted_bits = bits[order]
    sorted_asr = asr[order]
    sorted_models = [models[i] for i in order]

    p_bits: list[float] = []
    p_asr: list[float] = []
    p_models: list[str] = []
    running_max = -1.0

    for b, a, m in zip(sorted_bits, sorted_asr, sorted_models):
        if a > running_max:
            running_max = a
            p_bits.append(b)
            p_asr.append(a)
            p_models.append(m)

    return np.array(p_bits), np.array(p_asr), p_models


def main() -> None:
    rows = load_results(DEFAULT_RESULTS_FILE)

    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")

    all_bits_list: list[float] = []
    all_asr_list: list[float] = []
    all_models: list[str] = []
    style_by_model: dict[str, tuple[str, str]] = {}

    for model_key, color, marker, label in MODELS:
        model_rows = [
            r for r in rows
            if model_key in r["model_name"]
            and r["dataset_name"] == "advbench_harmbench"
            and r["asr"] is not None
        ]
        if not model_rows:
            continue

        bits = np.array([r["bits"] for r in model_rows])
        asr = np.array([r["asr"] for r in model_rows]) * 100

        nonzero = bits > 0
        ax.scatter(
            bits[nonzero], asr[nonzero], marker=marker, color=color,
            edgecolors="black", linewidths=0.5, s=120, alpha=0.5, zorder=3,
            label=label,
        )

        all_bits_list.extend(bits.tolist())
        all_asr_list.extend(asr.tolist())
        all_models.extend([label] * len(model_rows))
        style_by_model[label] = (color, marker)

    all_bits = np.array(all_bits_list)
    all_asr = np.array(all_asr_list)

    p_bits, p_asr, p_models = pareto_frontier(all_bits, all_asr, all_models)

    baseline_mask = p_bits == 0
    baseline_asr = float(p_asr[baseline_mask].max()) if baseline_mask.any() else None
    baseline_model = [m for m, b in zip(p_models, baseline_mask) if b]
    baseline_model = baseline_model[-1] if baseline_model else None

    nonzero_mask = ~baseline_mask
    p_bits = p_bits[nonzero_mask]
    p_asr = p_asr[nonzero_mask]
    p_models = [m for m, k in zip(p_models, nonzero_mask) if k]

    xlim = ax.get_xlim()

    if baseline_asr is not None and baseline_model is not None:
        p_bits = np.concatenate([[xlim[0]], p_bits])
        p_asr = np.concatenate([[baseline_asr], p_asr])
        p_models = [baseline_model] + p_models

    p_bits = np.concatenate([p_bits, [xlim[1]]])
    p_asr = np.concatenate([p_asr, [p_asr[-1]]])
    p_models = p_models + [p_models[-1]]

    for i in range(len(p_bits) - 1):
        model = p_models[i]
        color = style_by_model[model][0]
        seg_x = [p_bits[i], p_bits[i + 1], p_bits[i + 1]]
        seg_y = [p_asr[i], p_asr[i], p_asr[i + 1]]
        ax.plot(seg_x, seg_y, "-", color=color, linewidth=4, zorder=4)

    ax.set_xlim(xlim)
    ax.set_xlabel("Attack Program Length (bits)")
    ax.set_ylabel("ASR (%)")
    ax.set_title("HarmBench")
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
