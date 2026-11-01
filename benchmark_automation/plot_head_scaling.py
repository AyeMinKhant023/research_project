#!/usr/bin/env python3
"""
Publication-Quality Multi-Plot Generator for Classification Head Scaling:
Generates:
1. All 4 individual graphs (one each for FL-ETPU, CPU 1-core, CPU 2-cores, CPU 4-cores)
2. A 2x2 grid multi-panel figure containing all 4 approaches
3. The combined comparison graph with all 4 curves overlaid

Outputs both 300 DPI high-resolution PNG and vector PDF.
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_head_layers")

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


def plot_all(data, output_dir, device_name="Raspberry Pi 5"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ Error: matplotlib not found. Run: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # =========================================================================
    # 1. GENERATE THE 4 INDIVIDUAL GRAPHS (One for each approach)
    # =========================================================================
    print("\n📊 Generating 4 Individual Graphs...")
    for approach, style in STYLES.items():
        if approach not in data:
            continue
        means, stds = extract_series(data, approach)
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

        ax.errorbar(
            LAYERS, means, yerr=stds,
            label=style["label"], color=style["color"],
            marker=style["marker"], markersize=8, linewidth=2.5,
            capsize=4, capthick=1.5
        )

        ax.set_title(f"Head Training Runtime: {style['label']} ({device_name})", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Number of Added Hidden Layers (256 Neurons Each)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Train Header Latency (seconds)", fontsize=10, fontweight="bold")
        ax.set_xticks(LAYERS)
        ax.set_xticklabels(["0\n(Baseline)", "1", "2", "3", "4"])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", frameon=True)

        plt.tight_layout()
        out_name = f"graph_individual_{style['tag']}"
        fig.savefig(os.path.join(output_dir, f"{out_name}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{out_name}.pdf"))
        plt.close(fig)
        print(f"   ✅ Saved: {out_name}.png & .pdf")

    # =========================================================================
    # 2. GENERATE THE 2x2 MULTI-PANEL FIGURE (All 4 in one image)
    # =========================================================================
    print("\n📊 Generating 2x2 Grid Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()

    for idx, (approach, style) in enumerate(STYLES.items()):
        ax = axes[idx]
        if approach in data:
            means, stds = extract_series(data, approach)
            ax.errorbar(
                LAYERS, means, yerr=stds,
                label=style["label"], color=style["color"],
                marker=style["marker"], markersize=7, linewidth=2,
                capsize=3, capthick=1.2
            )
        ax.set_title(f"({chr(97 + idx)}) {style['label']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Added Hidden Layers (256 units)", fontsize=9)
        ax.set_ylabel("Train Header (sec)", fontsize=9)
        ax.set_xticks(LAYERS)
        ax.set_xticklabels(["0", "1", "2", "3", "4"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Head Training Runtime vs. Added Hidden Layers ({device_name})", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "graph_2x2_grid.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "graph_2x2_grid.pdf"))
    plt.close(fig)
    print("   ✅ Saved: graph_2x2_grid.png & .pdf")

    # =========================================================================
    # 3. GENERATE THE COMBINED COMPARISON GRAPH (All 4 overlaid)
    # =========================================================================
    print("\n📊 Generating Combined Comparison Graph...")
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

    ax.set_title(f"Combined Comparison: Head Runtime vs. Added Layers ({device_name})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Added Hidden Layers in Head (256 Neurons Each)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Train Header Latency (seconds)", fontsize=10, fontweight="bold")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels(["0\n(Baseline)", "1", "2", "3", "4"])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Execution Approach", fontsize=9, title_fontsize=10, frameon=True, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "graph_combined.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "graph_combined.pdf"))
    plt.close(fig)
    print("   ✅ Saved: graph_combined.png & .pdf")
    print(f"\n🎉 All graphs saved in: {output_dir}\n")


def get_demo_data():
    return {
        "FL-ETPU": {
            0: {"mean": 0.03, "std": 0.002},
            1: {"mean": 0.13, "std": 0.008},
            2: {"mean": 0.28, "std": 0.012},
            3: {"mean": 0.44, "std": 0.016},
            4: {"mean": 0.60, "std": 0.021},
        },
        "FL-CPU (1 core)": {
            0: {"mean": 0.04, "std": 0.003},
            1: {"mean": 0.20, "std": 0.010},
            2: {"mean": 0.38, "std": 0.015},
            3: {"mean": 0.58, "std": 0.020},
            4: {"mean": 0.79, "std": 0.025},
        },
        "FL-CPU (2 cores)": {
            0: {"mean": 0.05, "std": 0.004},
            1: {"mean": 0.16, "std": 0.008},
            2: {"mean": 0.29, "std": 0.012},
            3: {"mean": 0.43, "std": 0.016},
            4: {"mean": 0.57, "std": 0.020},
        },
        "FL-CPU (4 cores)": {
            0: {"mean": 0.06, "std": 0.004},
            1: {"mean": 0.14, "std": 0.007},
            2: {"mean": 0.26, "std": 0.010},
            3: {"mean": 0.38, "std": 0.014},
            4: {"mean": 0.51, "std": 0.018},
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Graph Plot Generator for Head Scaling")
    parser.add_argument("--device", default="pi5", help="Device name (pi5 or pi3)")
    parser.add_argument("--demo", action="store_true", help="Generate preview graphs using demo data")
    parser.add_argument("--output_dir", default=RESULTS_DIR, help="Output directory")
    args = parser.parse_args()

    if args.demo:
        data = get_demo_data()
        plot_all(data, args.output_dir, "Raspberry Pi 5 (Preview)")
    else:
        json_path = os.path.join(RESULTS_DIR, f"head_layers_{args.device}.json")
        if not os.path.exists(json_path):
            print(f"⚠️ Results JSON not found: {json_path}")
            print("Run the benchmark first, or preview using: python3 plot_head_scaling.py --demo")
            return
        with open(json_path, "r") as f:
            data = json.load(f)
        plot_all(data, args.output_dir, "Raspberry Pi 5")


if __name__ == "__main__":
    main()
