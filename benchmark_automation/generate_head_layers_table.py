#!/usr/bin/env python3
"""
Table & LaTeX Generator for Classification Head Scaling Experiment

Reads head_layers_{device}.csv or .json and outputs:
1. Formatted Terminal Table
2. Markdown Table (for reports)
3. LaTeX Table (for conference papers)
"""

import os
import sys
import json
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_head_layers")

def display_table(device_tag="pi5"):
    json_path = os.path.join(RESULTS_DIR, f"head_layers_{device_tag}.json")
    csv_path = os.path.join(RESULTS_DIR, f"head_layers_{device_tag}.csv")

    if not os.path.exists(json_path) and not os.path.exists(csv_path):
        print(f"⚠️ Neither {json_path} nor {csv_path} found.")
        print(f"Copy them from your Pi or run the benchmark first.")
        return

    layers = [0, 1, 2, 3, 4]
    approaches = ["FL-ETPU", "FL-CPU (1 core)", "FL-CPU (2 cores)", "FL-CPU (4 cores)"]

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        # Fallback parse CSV
        data = {}
        with open(csv_path, "r") as f:
            reader = list(csv.reader(f))
            headers = reader[0]
            for row in reader[1:]:
                app = row[0]
                data[app] = {}
                for idx, l in enumerate(layers):
                    mean_val = float(row[1 + idx])
                    std_val = float(row[1 + len(layers) + idx])
                    data[app][l] = {"mean": mean_val, "std": std_val}

    print("\n" + "=" * 85)
    print(f"  HEAD TRAINING RUNTIME (SECONDS) ACROSS ADDED HIDDEN LAYERS ({device_tag.upper()})")
    print("=" * 85)
    col_fmt = "{:<18} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}"
    print(col_fmt.format("Approach", "0 Layers", "1 Layer", "2 Layers", "3 Layers", "4 Layers"))
    print("-" * 85)

    latex_rows = []
    for app in approaches:
        if app not in data:
            continue
        vals = []
        for l in layers:
            l_key = str(l) if str(l) in data[app] else l
            m = data[app][l_key]["mean"]
            s = data[app][l_key]["std"]
            vals.append(f"{m:.3f} ± {s:.3f}")
        print(col_fmt.format(app, *vals))

        latex_cells = " & ".join(vals)
        latex_rows.append(f"  {app:<18} & {latex_cells} \\\\")

    print("=" * 85)

    # Save LaTeX table
    latex_code = f"""% Head Scaling Table generated automatically
\\begin{{table}}[t]
\\centering
\\caption{{Head Training Latency (in seconds) with varying added hidden layers (256 units each) on {device_tag.upper()}}}
\\label{{tab:head_layer_scaling_{device_tag}}}
\\begin{{tabular}}{{lccccc}}
\\hline
Approach & 0 Layers (Base) & 1 Layer & 2 Layers & 3 Layers & 4 Layers \\\\
\\hline
""" + "\n".join(latex_rows) + f"""
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    latex_out = os.path.join(RESULTS_DIR, f"head_layers_{device_tag}_table.tex")
    with open(latex_out, "w") as f:
        f.write(latex_code)
    print(f"\n[EXPORT] LaTeX Table saved to: {latex_out}\n")


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "pi5"
    display_table(dev)

