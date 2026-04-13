"""Plot bits vs ASR Pareto frontier for OLMo, colored by strategy_nameegy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.utils import load_results

RESULTS_FILE = Path("/network/scratch/b/brownet/information-safety/results/final-results.jsonl")
OUTPUT_FILE = Path(__file__).parent / "bits_vs_asr_olmo_strategy_nameegies.pdf"

STRATEGY_STYLE: dict[str, tuple[str, str]] = {
    "Baseline": ("tab:gray", "Baseline"),
    "Roleplay": ("tab:orange", "Roleplay"),
    "Data": ("tab:blue", "Data"),
}


def strategy_nameegy_label(row: dict) -> str:
    name = row["experiment_name"]
    if name == "DataStrategy":
        return "Data"
    if name == "BaselineStrategy":
        return "Baseline"
    if name == "RoleplayStrategy":
        return "Roleplay"
    return name


def pareto_frontier(
    bits: np.ndarray, asr: np.ndarray, strategy_nameegies: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pareto frontier maximizing ASR for given bits, tracking strategy_nameegy."""
    order = np.argsort(bits)
    sorted_bits = bits[order]
    sorted_asr = asr[order]
    sorted_strategy_names = [strategy_nameegies[i] for i in order]

    p_bits: list[float] = []
    p_asr: list[float] = []
    p_strategy_names: list[str] = []
    running_max = -1.0

    for b, a, s in zip(sorted_bits, sorted_asr, sorted_strategy_names):
        if a > running_max:
            running_max = a
            p_bits.append(b)
            p_asr.append(a)
            p_strategy_names.append(s)

    return np.array(p_bits), np.array(p_asr), p_strategy_names


def main() -> None:
    rows = load_results(RESULTS_FILE)
    olmo_rows = [
        r for r in rows
        if "Olmo-3-7B" in r["model_name"]
        and r["dataset_name"] == "advbench_harmbench"
        and r["asr"] is not None
    ]

    all_bits = np.array([r["bits"] for r in olmo_rows])
    all_asr = np.array([r["asr"] for r in olmo_rows]) * 100
    all_strategy_names = [strategy_nameegy_label(r) for r in olmo_rows]

    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")

    for strategy_name, (color, label) in STRATEGY_STYLE.items():
        mask = np.array([s == strategy_name for s in all_strategy_names])
        if not mask.any():
            continue
        nonzero = mask & (all_bits > 0)
        ax.scatter(
            all_bits[nonzero], all_asr[nonzero], marker="o", color=color,
            edgecolors="black", linewidths=0.5, s=120, alpha=0.5, zorder=3,
        )

    p_bits, p_asr, p_strategy_names = pareto_frontier(all_bits, all_asr, all_strategy_names)

    baseline_mask = p_bits == 0
    baseline_asr = float(p_asr[baseline_mask].max()) if baseline_mask.any() else None
    baseline_strategy_name = [s for s, m in zip(p_strategy_names, baseline_mask) if m]
    baseline_strategy_name = baseline_strategy_name[-1] if baseline_strategy_name else None

    nonzero_mask = ~baseline_mask
    p_bits = p_bits[nonzero_mask]
    p_asr = p_asr[nonzero_mask]
    p_strategy_names = [s for s, m in zip(p_strategy_names, nonzero_mask) if m]

    xlim = ax.get_xlim()

    if baseline_asr is not None and baseline_strategy_name is not None:
        p_bits = np.concatenate([[xlim[0]], p_bits])
        p_asr = np.concatenate([[baseline_asr], p_asr])
        p_strategy_names = [baseline_strategy_name] + p_strategy_names

    p_bits = np.concatenate([p_bits, [xlim[1]]])
    p_asr = np.concatenate([p_asr, [p_asr[-1]]])
    p_strategy_names = p_strategy_names + [p_strategy_names[-1]]

    drawn_labels: set[str] = set()
    for i in range(len(p_bits) - 1):
        strategy_name = p_strategy_names[i]
        color = STRATEGY_STYLE.get(strategy_name, ("black", strategy_name))[0]
        label_text = STRATEGY_STYLE.get(strategy_name, ("black", strategy_name))[1]
        label = label_text if label_text not in drawn_labels else None
        if label:
            drawn_labels.add(label_text)

        seg_x = [p_bits[i], p_bits[i + 1], p_bits[i + 1]]
        seg_y = [p_asr[i], p_asr[i], p_asr[i + 1]]
        ax.plot(seg_x, seg_y, "-", color=color, linewidth=4, label=label, zorder=4)

    ax.set_xlim(xlim)
    ax.set_xlabel("Program Length (bits)")
    ax.set_ylabel("ASR (%)")
    ax.set_title("HarmBench — OLMo-3-7B-Instruct")
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
