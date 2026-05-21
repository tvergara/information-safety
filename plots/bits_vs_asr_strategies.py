"""Plot bits vs ASR Pareto frontier per model, colored by strategy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.utils import DEFAULT_RESULTS_FILE, load_results

OUTPUT_DIR = Path(__file__).parent

MODELS: list[tuple[str, str, str]] = [
    ("Meta-Llama-3-8B", "Llama-3-8B-Instruct", "llama3"),
    ("Olmo-3-7B", "OLMo-3-7B-Instruct", "olmo"),
    ("Qwen3-4B", "Qwen3-4B", "qwen"),
]

STRATEGY_STYLE: dict[str, tuple[str, str]] = {
    "Baseline": ("tab:gray", "Baseline"),
    "Roleplay": ("tab:orange", "Roleplay"),
    "Data": ("tab:blue", "Data"),
    "GCG": ("tab:red", "GCG"),
    "AutoDAN": ("tab:purple", "AutoDAN"),
    "PAIR": ("tab:green", "PAIR"),
}


def strategy_label(row: dict) -> str:
    name = row["experiment_name"]
    if name in ("DataStrategy", "DataStrategyDeferredEval"):
        return "Data"
    if name == "BaselineStrategy":
        return "Baseline"
    if name == "RoleplayStrategy":
        return "Roleplay"
    if name == "PrecomputedGCGStrategy":
        return "GCG"
    if name == "PrecomputedAutoDANStrategy":
        return "AutoDAN"
    if name == "PrecomputedPAIRStrategy":
        return "PAIR"
    return name


def pareto_frontier(
    bits: np.ndarray, asr: np.ndarray, strategies: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pareto frontier maximizing ASR for given bits, tracking strategy."""
    order = np.argsort(bits)
    sorted_bits = bits[order]
    sorted_asr = asr[order]
    sorted_strategies = [strategies[i] for i in order]

    p_bits: list[float] = []
    p_asr: list[float] = []
    p_strategies: list[str] = []
    running_max = -1.0

    for b, a, s in zip(sorted_bits, sorted_asr, sorted_strategies):
        if a > running_max:
            running_max = a
            p_bits.append(b)
            p_asr.append(a)
            p_strategies.append(s)

    return np.array(p_bits), np.array(p_asr), p_strategies


def plot_model(rows: list[dict], model_key: str, title_name: str, slug: str) -> None:
    model_rows = [
        r for r in rows
        if model_key in r["model_name"]
        and r["dataset_name"] == "advbench_harmbench"
        and r["asr"] is not None
    ]

    all_bits = np.array([r["bits"] for r in model_rows])
    all_asr = np.array([r["asr"] for r in model_rows]) * 100
    all_strategies = [strategy_label(r) for r in model_rows]

    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")

    for strategy, (color, label) in STRATEGY_STYLE.items():
        mask = np.array([s == strategy for s in all_strategies])
        if not mask.any():
            continue
        nonzero = mask & (all_bits > 0)
        ax.scatter(
            all_bits[nonzero], all_asr[nonzero], marker="o", color=color,
            edgecolors="black", linewidths=0.5, s=120, alpha=0.7, zorder=3,
            label=label,
        )

    p_bits, p_asr, p_strategies = pareto_frontier(all_bits, all_asr, all_strategies)

    baseline_mask = p_bits == 0
    baseline_asr = float(p_asr[baseline_mask].max()) if baseline_mask.any() else None
    baseline_strategy = [s for s, m in zip(p_strategies, baseline_mask) if m]
    baseline_strategy = baseline_strategy[-1] if baseline_strategy else None

    nonzero_mask = ~baseline_mask
    p_bits = p_bits[nonzero_mask]
    p_asr = p_asr[nonzero_mask]
    p_strategies = [s for s, m in zip(p_strategies, nonzero_mask) if m]

    xlim = ax.get_xlim()

    if baseline_asr is not None and baseline_strategy is not None:
        p_bits = np.concatenate([[xlim[0]], p_bits])
        p_asr = np.concatenate([[baseline_asr], p_asr])
        p_strategies = [baseline_strategy] + p_strategies

    p_bits = np.concatenate([p_bits, [xlim[1]]])
    p_asr = np.concatenate([p_asr, [p_asr[-1]]])
    p_strategies = p_strategies + [p_strategies[-1]]

    for i in range(len(p_bits) - 1):
        strategy = p_strategies[i]
        color = STRATEGY_STYLE.get(strategy, ("black", strategy))[0]
        seg_x = [p_bits[i], p_bits[i + 1], p_bits[i + 1]]
        seg_y = [p_asr[i], p_asr[i], p_asr[i + 1]]
        ax.plot(seg_x, seg_y, "-", color=color, linewidth=4, zorder=4)

    ax.set_xlim(xlim)
    ax.set_xlabel("Attack Program Length (bits)")
    ax.set_ylabel("ASR (%)")
    ax.set_title(f"HarmBench — {title_name}")
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    output_file = OUTPUT_DIR / f"bits_vs_asr_{slug}_strategies.pdf"
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_file}")


def main() -> None:
    rows = load_results(DEFAULT_RESULTS_FILE)
    for model_key, title_name, slug in MODELS:
        plot_model(rows, model_key, title_name, slug)


if __name__ == "__main__":
    main()
