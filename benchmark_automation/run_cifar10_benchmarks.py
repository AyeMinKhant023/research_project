#!/usr/bin/env python3
"""
Master CIFAR-10 Automated Benchmarking Suite (Max Cores Only: FL-ETPU vs FL-CPU 4-Cores)

Executes the professor's updated strategy:
- Dataset: CIFAR-10 (10 classes).
- 2 Approaches: FL-ETPU and FL-CPU (Max Cores = 4 cores).
- Classification Head Scaling: 0, 1, 2, 3, 4 layers (256 neurons each).
- 5 individual runs per setting + Average (Mean ± Std).
- Output:
  1. Table 1 (or 4 on Pi3): Feature Extractor CPU (4 cores) with Run 1..5 and Mean ± Std.
  2. Table 2 (or 5 on Pi3): Feature Extractor TPU with Run 1..5 and Mean ± Std.
  3. Graph 1 (or 2 on Pi3): Classification Head Training Runtime on CPU vs. 0..4 Layers.
  4. Complete CSV and LaTeX tables for direct inclusion in the paper.
"""

import os
import sys
import time
import re
import argparse
import subprocess
import statistics
import json
import csv
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FL_DIR = os.path.join(PROJECT_ROOT, "flower_project", "federated_learning")
EDGETPU_MODELS_DIR = os.path.join(PROJECT_ROOT, "edgetpu", "retrain-backprop")

DEFAULT_DATA_DIR = os.path.join(FL_DIR, "data", "cifar10", "client1_data")
EDGETPU_MODEL = os.path.join(EDGETPU_MODELS_DIR, "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite")
CPU_MODEL = os.path.join(EDGETPU_MODELS_DIR, "mobilenet_v1_1.0_224_quant_embedding_extractor.tflite")

CONFIGS = [
    {
        "name": "Feature Extractor TPU",
        "tag": "FL-ETPU",
        "model": EDGETPU_MODEL,
        "threads": 4,
        "prefix": "EdgeTPU"
    },
    {
        "name": "Feature Extractor CPU (4 cores)",
        "tag": "FL-CPU (4 cores)",
        "model": CPU_MODEL,
        "threads": 4,
        "prefix": "CPU4"
    }
]

LAYERS_TO_TEST = [0, 1, 2, 3, 4]


def parse_timings_from_log(log_text: str):
    """Parses exact timings from client run output."""
    m_load = re.search(r'Time for Load Feature Extractor[^:]*:\s*([\d\.]+)', log_text)
    m_ext = re.search(r'Time for Extract Train Embeddings[^:]*:\s*([\d\.]+)', log_text)
    m_train = re.search(r'Time for\s+Train Classification Head[^:]*:\s*([\d\.]+)', log_text)
    m_total = re.search(r'Total Runtime \(Full Client Run\):\s*([\d\.]+)', log_text)

    load_time = float(m_load.group(1)) if m_load else 0.0
    ext_time = float(m_ext.group(1)) if m_ext else 0.0
    offline_total = load_time + ext_time
    train_time = float(m_train.group(1)) if m_train else 0.0
    runtime_total = train_time
    full_total = float(m_total.group(1)) if m_total else (offline_total + runtime_total)

    return {
        "load_base_model": load_time,
        "extract_features": ext_time,
        "offline_total": offline_total,
        "train_header": train_time,
        "runtime_total": runtime_total,
        "full_total": full_total
    }


def run_single_client(cfg, num_layers, run_idx, args, output_dir):
    """Executes a single client benchmark run."""
    log_name = f"{cfg['prefix']}_L{num_layers}_run{run_idx}_{args.device}.txt"
    log_path = os.path.join(output_dir, log_name)

    # Use python executable (support pyenv on Raspberry Pi)
    py_exec = sys.executable
    pi_pyenv = "/home/ayeminkhant/.pyenv/versions/3.9.16/bin/python3"
    if os.path.exists(pi_pyenv):
        py_exec = pi_pyenv

    cmd = [
        py_exec,
        os.path.join(FL_DIR, "fl_client.py"),
        "--embedding_extractor_path", cfg["model"],
        "--data_dir", args.data_dir,
        "--num_classes", str(args.num_classes),
        "--server_address", args.server_address,
        "--num_threads", str(cfg["threads"]),
        "--num_head_layers", str(num_layers),
        "--local_epochs", str(args.local_epochs)
    ]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {cfg['name']} | Layers: {num_layers} | Run {run_idx}/{args.runs}")
    start = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        log_out = proc.stdout
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: process returned {e.returncode}")
        log_out = e.stdout

    elapsed = time.time() - start
    with open(log_path, "w") as f:
        f.write(log_out)

    t = parse_timings_from_log(log_out)
    print(f"   -> Load: {t['load_base_model']:.2f}s | Extract: {t['extract_features']:.2f}s | Train Head: {t['train_header']:.4f}s | Total: {t['full_total']:.2f}s (wall: {elapsed:.2f}s)")
    return t


def format_stat(mean_v, std_v):
    return f"{mean_v:.2f} ± {std_v:.2f}"


def export_feature_extractor_table(cfg, raw_runs, device_name, output_dir):
    """Generates a detailed table showing Run 1..5 and Average (Mean ± Std)."""
    table_name = f"Table_{cfg['prefix']}_{device_name}"
    csv_file = os.path.join(output_dir, f"{table_name}.csv")
    latex_file = os.path.join(output_dir, f"{table_name}.tex")

    cols = ["Run", "Load Base Model", "Extract Features", "Offline Total", "Train Header", "Runtime Total"]
    rows = []

    loads = [r["load_base_model"] for r in raw_runs]
    extracts = [r["extract_features"] for r in raw_runs]
    offlines = [r["offline_total"] for r in raw_runs]
    trains = [r["train_header"] for r in raw_runs]
    runtimes = [r["runtime_total"] for r in raw_runs]

    for idx, r in enumerate(raw_runs, 1):
        rows.append([
            f"Run {idx}",
            f"{r['load_base_model']:.2f}",
            f"{r['extract_features']:.2f}",
            f"{r['offline_total']:.2f}",
            f"{r['train_header']:.4f}",
            f"{r['runtime_total']:.4f}"
        ])

    mean_std_row = [
        "Average (Mean ± Std)",
        format_stat(statistics.mean(loads), statistics.stdev(loads) if len(loads) > 1 else 0.0),
        format_stat(statistics.mean(extracts), statistics.stdev(extracts) if len(extracts) > 1 else 0.0),
        format_stat(statistics.mean(offlines), statistics.stdev(offlines) if len(offlines) > 1 else 0.0),
        format_stat(statistics.mean(trains), statistics.stdev(trains) if len(trains) > 1 else 0.0),
        format_stat(statistics.mean(runtimes), statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0)
    ]
    rows.append(mean_std_row)

    # Print to console
    print("\n" + "=" * 90)
    print(f"   {cfg['name'].upper()} - EXECUTION TIME (SECONDS) [{device_name.upper()}]")
    print("=" * 90)
    col_fmt = "{:<22} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16}"
    print(col_fmt.format(*cols))
    print("-" * 90)
    for r in rows[:-1]:
        print(col_fmt.format(*r))
    print("-" * 90)
    print(col_fmt.format(*rows[-1]))
    print("=" * 90)

    # Save CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)
    print(f"✅ Saved CSV: {csv_file}")

    # Save LaTeX Table
    latex_rows = []
    for r in rows:
        latex_rows.append(f"  {r[0]:<22} & {r[1]:<14} & {r[2]:<14} & {r[3]:<14} & {r[4]:<14} & {r[5]:<14} \\\\")

    latex_code = f"""% {cfg['name']} Execution Time on {device_name}
\\begin{{table}}[t]
\\centering
\\caption{{{cfg['name']} execution time (in seconds) across 5 runs on {device_name}}}
\\label{{tab:{cfg['prefix']}_{device_name}}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\multirow{{2}}{{*}}{{Run}} & \\multicolumn{{3}}{{c}}{{Offline Phase}} & \\multicolumn{{2}}{{c}}{{Runtime Phase}} \\\\
\\cline{{2-6}}
 & Load Base Model & Extract Features & Total & Train Header & Total \\\\
\\hline
""" + "\n".join(latex_rows[:-1]) + """
\\hline
""" + latex_rows[-1] + """
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(latex_file, "w") as f:
        f.write(latex_code)
    print(f"✅ Saved LaTeX: {latex_file}")


def plot_head_training_graph(head_scaling_data, device_name, output_dir):
    """Generates the publication-ready Graph for Classification Head Training Runtime on CPU."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not found, skipping plot generation.")
        return

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    layers = [0, 1, 2, 3, 4]
    means = [head_scaling_data[l]["mean"] for l in layers]
    stds = [head_scaling_data[l]["std"] for l in layers]

    ax.errorbar(
        layers, means, yerr=stds,
        label="Classification Head (CPU 4-cores)",
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        markersize=7,
        capsize=4,
        capthick=1.5
    )

    ax.set_title(f"Classification Head Training Runtime vs. Added Layers ({device_name.upper()})", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Number of Added Hidden Layers in Head (256 Units Each)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Train Header Latency (seconds)", fontsize=10, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels(["0\n(Baseline)", "1", "2", "3", "4"])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"graph_classification_head_{device_name}.png")
    pdf_path = os.path.join(output_dir, f"graph_classification_head_{device_name}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"\n🎉 Graph saved successfully:")
    print(f"   - High-Res PNG: {png_path}")
    print(f"   - Vector PDF:   {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Master CIFAR-10 Benchmark Runner (Max Cores: TPU vs CPU)")
    parser.add_argument("--device", default="pi5", choices=["pi5", "pi3", "pi3b", "mac"],
                        help="Target device identifier")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per configuration to average (default: 5)")
    parser.add_argument("--server_address", default="172.20.10.3:8080",
                        help="Flower server IP:Port")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to CIFAR-10 training data directory")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes (default: 10 for CIFAR-10)")
    parser.add_argument("--local_epochs", type=int, default=10,
                        help="Number of local epochs per round (default: 10)")
    parser.add_argument("--cooldown", type=int, default=2,
                        help="Cooldown between runs")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results_cifar10"),
                        help="Directory for logs and reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=========================================================================")
    print("      CIFAR-10 FEDERATED LEARNING BENCHMARK SUITE (MAX CORES ONLY)       ")
    print("=========================================================================")
    print(f"Device:           {args.device}")
    print(f"Dataset:          {args.data_dir} ({args.num_classes} classes)")
    print(f"Local Epochs:     {args.local_epochs}")
    print(f"Runs per setting: {args.runs} (Run 1..5 + Mean ± Std)")
    print(f"Server address:   {args.server_address}")
    print(f"Output directory: {args.output_dir}")
    print("=========================================================================\n")

    # Step 1: Benchmark the two Feature Extractor Approaches (Baseline Head = 0 layers)
    for cfg in CONFIGS:
        print(f"\n==================== BENCHMARKING: {cfg['name']} ====================")
        raw_runs = []
        for r in range(1, args.runs + 1):
            t = run_single_client(cfg, num_layers=0, run_idx=r, args=args, output_dir=args.output_dir)
            raw_runs.append(t)
            if r < args.runs and args.cooldown > 0:
                time.sleep(args.cooldown)

        export_feature_extractor_table(cfg, raw_runs, args.device, args.output_dir)

    # Step 2: Benchmark Classification Head Scaling (0 to 4 layers on CPU)
    print("\n==================== BENCHMARKING: HEAD SCALING (0 to 4 Layers) ====================")
    head_scaling_results = {}
    cpu_cfg = CONFIGS[1]  # CPU 4-cores

    for num_layers in LAYERS_TO_TEST:
        print(f"\n--- Testing Head with {num_layers} added layers (256 units each) ---")
        train_latencies = []
        for r in range(1, args.runs + 1):
            t = run_single_client(cpu_cfg, num_layers=num_layers, run_idx=r, args=args, output_dir=args.output_dir)
            train_latencies.append(t["train_header"])
            if r < args.runs and args.cooldown > 0:
                time.sleep(args.cooldown)

        head_scaling_results[num_layers] = {
            "mean": statistics.mean(train_latencies),
            "std": statistics.stdev(train_latencies) if len(train_latencies) > 1 else 0.0,
            "raw": train_latencies
        }

    # Save head scaling JSON & CSV
    head_json_path = os.path.join(args.output_dir, f"head_scaling_{args.device}.json")
    with open(head_json_path, "w") as f:
        json.dump(head_scaling_results, f, indent=2)

    # Plot the Graph
    plot_head_training_graph(head_scaling_results, args.device, args.output_dir)

    print("\n" + "=" * 90)
    print("🎉 ALL BENCHMARKS, TABLES, AND GRAPHS COMPLETED SUCCESSFULLY!")
    print(f"Results stored in: {args.output_dir}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
