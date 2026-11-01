#!/usr/bin/env python3
"""
Publication-Quality Plot Generator for Graph 3:
Head Training Energy Consumption (in mJ) vs. Number of Added Layers (0, 1, 2, 3, 4)

Produces:
1. 4 Individual Energy Graphs (FL-ETPU, CPU 1, 2, 4 cores)
2. 2x2 Grid Figure
3. Combined Energy Comparison Plot
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENERGY_DIR = os.path.join(SCRIPT_DIR, "results_head_energy")

STYLES = {
    "FL-ETPU": {"color": "#1b9e77", "marker": "o", "label": "FL-ETPU", "tag": "ETPU"},
    "FL-CPU (1 core)": {"color": "#d95f02", "marker": "s", "label": "FL-CPU (1 core)", "tag": "CPU1"},
    "FL-CPU (2 cores)": {"color": "#7570b3", "marker": "^", "label": "FL-CPU (2 cores)", "tag": "CPU2"},
    "FL-CPU (4 cores)": {"color": "#e7298a", "marker": "D", "label": "FL-CPU (4 cores)", "tag": "CPU4"},
}

LAYERS = [0, 1, 2, 3, 4]


def extract_series(data, approach):
    means, stds = [], []
    for l in LAYERS:
        l_key = str(l) if str(l) in data.get(approach, {}) else l
        if approach in data and l_key in data[approach]:
            means.append(data[approach][l_key]["mean"])
            stds.append(data[approach][l_key]["std"])
        else:
            means.append(0.0)
            stds.append(0.0)
    return means, stds


def plot_energy(data, output_dir, device_name="Raspberry Pi 5"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ Error: matplotlib not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Combined Plot
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for approach, style in STYLES.items():
        if approach not in data:
            continue
        means, stds = extract_series(data, approach)
        ax.errorbar(
            LAYERS, means, yerr=stds,
            label=style["label"], color=style["color"],
            marker=style["marker"], markersize=7, linewidth=2,
            linestyle="-" if "ETPU" in approach else "--",
            capsize=4, capthick=1.5
        )

    ax.set_title(f"Graph 3: Head Training Energy vs. Added Layers ({device_name})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Added Hidden Layers in Head (256 Neurons Each)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Train Header Energy (mJ)", fontsize=10, fontweight="bold")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels(["0\n(Baseline)", "1", "2", "3", "4"])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Execution Approach", fontsize=9, title_fontsize=10, frameon=True, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "graph3_energy_combined.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "graph3_energy_combined.pdf"))
    plt.close(fig)

    print(f"✅ Saved Graph 3: {os.path.join(output_dir, 'graph3_energy_combined.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="pi5", help="Device name")
    parser.add_argument("--output_dir", default=ENERGY_DIR)
    args = parser.parse_args()

    json_path = os.path.join(args.output_dir, f"energy_head_layers_{args.device}.json")
    if not os.path.exists(json_path):
        print(f"⚠️ Energy JSON not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    plot_energy(data, args.output_dir, f"Raspberry Pi 5")


if __name__ == "__main__":
    main()

