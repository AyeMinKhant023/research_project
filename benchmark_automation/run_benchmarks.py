#!/usr/bin/env python3
"""
Automated Benchmarking Suite for Federated Learning on Edge Devices (FL-ETPU vs FL-CPU)

This script automates the full evaluation pipeline for Table I:
- Loops through all configurations: FL-ETPU, FL-CPU (1 core), FL-CPU (2 cores), FL-CPU (4 cores).
- Executes N runs per configuration automatically.
- Extracts timings (Load Model, Extract Features, Train Header, Totals).
- Calculates Mean ± Standard Deviation across runs.
- Exports results as a formatted terminal table, CSV, and ready-to-publish LaTeX table.
- Uses Python standard library (statistics) - zero external dependencies required!
"""

import os
import sys
import time
import re
import argparse
import subprocess
import statistics
from datetime import datetime

# Default Paths (relative to research_project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FL_DIR = os.path.join(PROJECT_ROOT, "flower_project", "federated_learning")
EDGETPU_MODELS_DIR = os.path.join(PROJECT_ROOT, "edgetpu", "retrain-backprop")

DEFAULT_DATA_DIR = os.path.join(FL_DIR, "data", "client1_data")
EDGETPU_MODEL = os.path.join(EDGETPU_MODELS_DIR, "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite")
CPU_MODEL = os.path.join(EDGETPU_MODELS_DIR, "mobilenet_v1_1.0_224_quant_embedding_extractor.tflite")

BENCHMARK_CONFIGS = [
    {
        "approach": "FL-ETPU",
        "model_path": EDGETPU_MODEL,
        "threads": 4,
        "is_edgetpu": True,
        "prefix": "EdgeTPU"
    },
    {
        "approach": "FL-CPU (1 core)",
        "model_path": CPU_MODEL,
        "threads": 1,
        "is_edgetpu": False,
        "prefix": "CPU1"
    },
    {
        "approach": "FL-CPU (2 cores)",
        "model_path": CPU_MODEL,
        "threads": 2,
        "is_edgetpu": False,
        "prefix": "CPU2"
    },
    {
        "approach": "FL-CPU (4 cores)",
        "model_path": CPU_MODEL,
        "threads": 4,
        "is_edgetpu": False,
        "prefix": "CPU4"
    }
]


def parse_timing_from_log(log_text: str):
    """Parses exact execution times from client log output."""
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
        "load_model": load_time,
        "extract_features": ext_time,
        "offline_total": offline_total,
        "train_header": train_time,
        "runtime_total": runtime_total,
        "full_total": full_total
    }


def run_single_benchmark(config, run_idx, args, output_dir):
    """Executes a single client run and records output to log file."""
    log_filename = f"{config['prefix']}Runtime{run_idx}_{args.device}.txt"
    log_path = os.path.join(output_dir, log_filename)

    cmd = [
        sys.executable,
        os.path.join(FL_DIR, "fl_client.py"),
        "--embedding_extractor_path", config["model_path"],
        "--data_dir", args.data_dir,
        "--num_classes", str(args.num_classes),
        "--server_address", args.server_address,
        "--num_threads", str(config["threads"])
    ]

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running {config['approach']} - Run {run_idx}/{args.runs}...")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        log_content = proc.stdout
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Process exited with returncode {e.returncode}")
        log_content = e.stdout

    elapsed = time.time() - start_time

    # Save raw log file
    with open(log_path, "w") as f:
        f.write(log_content)
    print(f"Saved raw log to: {log_path} (wall clock: {elapsed:.2f}s)")

    timings = parse_timing_from_log(log_content)
    print(f"  -> Load: {timings['load_model']:.2f}s | Extract: {timings['extract_features']:.2f}s | Train Head: {timings['train_header']:.2f}s")
    return timings


def format_stat(mean_val: float, std_val: float) -> str:
    """Formats stat as mean ± std."""
    return f"{mean_val:.2f} ± {std_val:.2f}"


def generate_reports(device_name: str, results_summary: list, output_dir: str):
    """Generates ASCII terminal table, CSV, and LaTeX table for the paper."""
    header = f"Table I: Execution time (in seconds) of FL-ETPU and FL-CPU averaged across runs ({device_name})"
    sep = "=" * 105
    col_fmt = "{:<18} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16}"

    print("\n" + sep)
    print(header.center(105))
    print(sep)
    print(col_fmt.format("Approach", "Load Base Model", "Extract Features", "Offline Total", "Train Header", "Runtime Total"))
    print("-" * 105)

    csv_rows = [["Device", "Approach", "Load Base Model (mean)", "Load Base Model (std)", 
                 "Extract Features (mean)", "Extract Features (std)", "Offline Total (mean)", "Offline Total (std)",
                 "Train Header (mean)", "Train Header (std)", "Runtime Total (mean)", "Runtime Total (std)"]]

    latex_rows = []

    for item in results_summary:
        approach = item["approach"]
        s = item["stats"]
        print(col_fmt.format(
            approach,
            format_stat(s["load_model_mean"], s["load_model_std"]),
            format_stat(s["extract_mean"], s["extract_std"]),
            format_stat(s["offline_mean"], s["offline_std"]),
            format_stat(s["train_mean"], s["train_std"]),
            format_stat(s["runtime_mean"], s["runtime_std"])
        ))

        csv_rows.append([
            device_name, approach,
            f"{s['load_model_mean']:.4f}", f"{s['load_model_std']:.4f}",
            f"{s['extract_mean']:.4f}", f"{s['extract_std']:.4f}",
            f"{s['offline_mean']:.4f}", f"{s['offline_std']:.4f}",
            f"{s['train_mean']:.4f}", f"{s['train_std']:.4f}",
            f"{s['runtime_mean']:.4f}", f"{s['runtime_std']:.4f}"
        ])

        latex_rows.append(
            f"  {approach:<18} & {format_stat(s['load_model_mean'], s['load_model_std']):<14} & "
            f"{format_stat(s['extract_mean'], s['extract_std']):<14} & {format_stat(s['offline_mean'], s['offline_std']):<14} & "
            f"{format_stat(s['train_mean'], s['train_std']):<14} & {format_stat(s['runtime_mean'], s['runtime_std']):<14} \\\\"
        )

    print(sep)

    # Export CSV
    csv_path = os.path.join(output_dir, f"table1_results_{device_name}.csv")
    with open(csv_path, "w") as f:
        for r in csv_rows:
            f.write(",".join(r) + "\n")
    print(f"\n[EXPORT] CSV saved to: {csv_path}")

    # Export LaTeX Table
    latex_code = f"""% Table I generated automatically by benchmark_automation
\\begin{{table*}}[t]
\\centering
\\caption{{Execution time (in seconds) of FL-ETPU and FL-CPU averaged across runs on {device_name}}}
\\label{{tab:execution_time_{device_name}}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\multirow{{2}}{{*}}{{Approach}} & \\multicolumn{{3}}{{c}}{{Offline Phase}} & \\multicolumn{{2}}{{c}}{{Runtime Phase}} \\\\
\\cline{{2-6}}
 & Load Base Model & Extract Features & Total & Train Header & Total \\\\
\\hline
""" + "\n".join(latex_rows) + f"""
\\hline
\\end{{tabular}}
\\end{{table*}}
"""
    latex_path = os.path.join(output_dir, f"table1_results_{device_name}.tex")
    with open(latex_path, "w") as f:
        f.write(latex_code)
    print(f"[EXPORT] LaTeX table saved to: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Automated FL Benchmarking Suite for Edge TPU and CPU")
    parser.add_argument("--device", default="pi5", choices=["pi5", "pi3", "pi3b", "mac"],
                        help="Target device identifier (default: pi5)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per configuration to average (default: 5)")
    parser.add_argument("--server_address", default="172.20.10.3:8080",
                        help="Flower server IP:Port (default: 172.20.10.3:8080)")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to training data directory")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classification classes (default: 5)")
    parser.add_argument("--cooldown", type=int, default=3,
                        help="Cooldown sleep seconds between runs for thermal stability (default: 3)")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results"),
                        help="Directory to store benchmark logs and outputs")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=================================================================")
    print("      EDGE TPU & FEDERATED LEARNING AUTOMATED BENCHMARK SUITE    ")
    print("=================================================================")
    print(f"Device:           {args.device}")
    print(f"Runs per config:  {args.runs}")
    print(f"Server address:   {args.server_address}")
    print(f"Output directory: {args.output_dir}")
    print(f"Configurations:   {len(BENCHMARK_CONFIGS)}")
    print("=================================================================")

    summary_results = []

    for cfg in BENCHMARK_CONFIGS:
        run_data = {
            "load_model": [],
            "extract_features": [],
            "offline_total": [],
            "train_header": [],
            "runtime_total": []
        }

        print(f"\n>>> Starting Benchmark Category: {cfg['approach']}")
        for run_idx in range(1, args.runs + 1):
            timings = run_single_benchmark(cfg, run_idx, args, args.output_dir)
            for k in run_data:
                run_data[k].append(timings[k])

            if run_idx < args.runs and args.cooldown > 0:
                print(f"Cooling down for {args.cooldown}s...")
                time.sleep(args.cooldown)

        # Calculate statistics using Python standard library
        calc_mean = lambda arr: statistics.mean(arr) if arr else 0.0
        calc_std = lambda arr: statistics.stdev(arr) if len(arr) > 1 else 0.0

        stats = {
            "load_model_mean": calc_mean(run_data["load_model"]),
            "load_model_std": calc_std(run_data["load_model"]),
            "extract_mean": calc_mean(run_data["extract_features"]),
            "extract_std": calc_std(run_data["extract_features"]),
            "offline_mean": calc_mean(run_data["offline_total"]),
            "offline_std": calc_std(run_data["offline_total"]),
            "train_mean": calc_mean(run_data["train_header"]),
            "train_std": calc_std(run_data["train_header"]),
            "runtime_mean": calc_mean(run_data["runtime_total"]),
            "runtime_std": calc_std(run_data["runtime_total"]),
        }

        summary_results.append({
            "approach": cfg["approach"],
            "stats": stats,
            "raw": run_data
        })

    # Generate final terminal table, CSV, and LaTeX files
    generate_reports(args.device, summary_results, args.output_dir)


if __name__ == "__main__":
    main()
