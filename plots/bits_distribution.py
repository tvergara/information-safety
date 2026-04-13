"""Box plot of program length (bits) distribution by strategy."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plots.utils import load_results

RESULTS_FILE = Path("/network/scratch/b/brownet/information-safety/results/final-results.jsonl")
OUTPUT_FILE = Path(__file__).parent / "bits_distribution.pdf"

STRATEGY_ORDER = [
    "Baseline",
    "Roleplay",
    "Data (10)",
    "Data (50)",
    "Data (100)",
    "Data (250)",
    "Data (all)",
]


def main() -> None:
    rows = load_results(RESULTS_FILE)

    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["asr"] is None:
            continue
        name = r["experiment_name"]
        max_ex = r.get("max_examples")
        if name == "DataStrategy":
            label = f"Data ({max_ex if max_ex is not None else 'all'})"
        elif name == "BaselineStrategy":
            label = "Baseline"
        elif name == "RoleplayStrategy":
            label = "Roleplay"
        else:
            label = name
        groups[label].append(r["bits"])

    labels = [s for s in STRATEGY_ORDER if s in groups]
    data = [groups[s] for s in labels]

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showfliers=True,
        flierprops={"markersize": 8},
        medianprops={"linewidth": 3},
        whiskerprops={"linewidth": 2.5},
        capprops={"linewidth": 2.5},
        boxprops={"linewidth": 2.5},
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Program Length (bits)")
    ax.set_title("Program Length Distribution by Strategy")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
