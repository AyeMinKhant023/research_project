#!/usr/bin/env python3
"""
Automated Benchmarking Suite for Classification Head Layer Scaling (0 to 4 Layers)

Evaluates how on-device Head Training Latency scales as hidden layers are added:
- Layer counts: 0 (baseline linear), 1, 2, 3, 4 (256 neurons each).
- Configurations: FL-ETPU, FL-CPU (1 core), FL-CPU (2 cores), FL-CPU (4 cores).
- Repetitions: 5 runs each (calculates Mean ± Std).
- Output: CSV, Summary Table, and input data for Graph 1.
"""

import os
import sys
import time
import re
import argparse
import subprocess
import statistics
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FL_DIR = os.path.join(PROJECT_ROOT, "flower_project", "federated_learning")
EDGETPU_MODELS_DIR = os.path.join(PROJECT_ROOT, "edgetpu", "retrain-backprop")

DEFAULT_DATA_DIR = os.path.join(FL_DIR, "data", "client1_data")
EDGETPU_MODEL = os.path.join(EDGETPU_MODELS_DIR, "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite")
CPU_MODEL = os.path.join(EDGETPU_MODELS_DIR, "mobilenet_v1_1.0_224_quant_embedding_extractor.tflite")

CONFIGS = [
    {"name": "FL-ETPU", "model": EDGETPU_MODEL, "threads": 4, "prefix": "EdgeTPU"},
    {"name": "FL-CPU (1 core)", "model": CPU_MODEL, "threads": 1, "prefix": "CPU1"},
    {"name": "FL-CPU (2 cores)", "model": CPU_MODEL, "threads": 2, "prefix": "CPU2"},
    {"name": "FL-CPU (4 cores)", "model": CPU_MODEL, "threads": 4, "prefix": "CPU4"},
]

LAYERS_TO_TEST = [0, 1, 2, 3, 4]


def parse_head_timing(log_text: str):
    """Parses Train Header latency from client stdout."""
    m_train = re.search(r'Time for\s+Train Classification Head[^:]*:\s*([\d\.]+)', log_text)
    m_total = re.search(r'Total Runtime \(Full Client Run\):\s*([\d\.]+)', log_text)
    train_time = float(m_train.group(1)) if m_train else 0.0
    total_time = float(m_total.group(1)) if m_total else train_time
    return train_time, total_time


def run_test(cfg, num_layers, run_idx, args, output_dir):
    """Executes a single client run with specific layer count."""
    log_name = f"{cfg['prefix']}_L{num_layers}_run{run_idx}_{args.device}.txt"
    log_path = os.path.join(output_dir, log_name)

    cmd = [
        sys.executable,
        os.path.join(FL_DIR, "fl_client.py"),
        "--embedding_extractor_path", cfg["model"],
        "--data_dir", args.data_dir,
        "--num_classes", str(args.num_classes),
        "--server_address", args.server_address,
        "--num_threads", str(cfg["threads"]),
        "--num_head_layers", str(num_layers)
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

    train_time, total_time = parse_head_timing(log_out)
    print(f"   -> Train Head: {train_time:.4f}s | Full Run: {total_time:.2f}s (wall: {elapsed:.2f}s)")
    return train_time, total_time


def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Head Benchmark Runner")
    parser.add_argument("--device", default="pi5", choices=["pi5", "pi3", "pi3b", "mac"],
                        help="Target device identifier")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per point to average (default: 5)")
    parser.add_argument("--server_address", default="172.20.10.3:8080",
                        help="Server IP:port")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to training data")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes")
    parser.add_argument("--cooldown", type=int, default=2,
                        help="Cooldown seconds between runs")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results_head_layers"),
                        help="Results directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=========================================================================")
    print("      CLASSIFICATION HEAD LAYER SCALING BENCHMARK (0 to 4 LAYERS)       ")
    print("=========================================================================")
    print(f"Device:           {args.device}")
    print(f"Layers to test:   {LAYERS_TO_TEST}")
    print(f"Runs per setting: {args.runs}")
    print(f"Server address:   {args.server_address}")
    print(f"Output directory: {args.output_dir}")
    print("=========================================================================\n")

    # Results table: {approach: {num_layers: {"mean": x, "std": y, "raw": []}}}
    results = {}

    for cfg in CONFIGS:
        approach = cfg["name"]
        results[approach] = {}

        print(f"\n==================== CONFIGURATION: {approach} ====================")
        for num_layers in LAYERS_TO_TEST:
            print(f"\n--- Testing Head with {num_layers} added layers (256 units each) ---")
            train_times = []
            for r in range(1, args.runs + 1):
                t_train, _ = run_test(cfg, num_layers, r, args, args.output_dir)
                train_times.append(t_train)
                if r < args.runs and args.cooldown > 0:
                    time.sleep(args.cooldown)

            mean_val = statistics.mean(train_times)
            std_val = statistics.stdev(train_times) if len(train_times) > 1 else 0.0
            results[approach][num_layers] = {
                "mean": mean_val,
                "std": std_val,
                "raw": train_times
            }

    # Summary table
    print("\n\n" + "=" * 90)
    print("   SUMMARY TABLE: Train Header Execution Time (seconds) vs. Number of Head Layers")
    print("=" * 90)
    header = "{:<20} | " + " | ".join([f"{l} Layers".center(12) for l in LAYERS_TO_TEST])
    print(header.format("Approach"))
    print("-" * 90)

    csv_rows = [["Approach"] + [f"{l}_layers_mean" for l in LAYERS_TO_TEST] + [f"{l}_layers_std" for l in LAYERS_TO_TEST]]

    for approach in results:
        row_str = [f"{results[approach][l]['mean']:.3f}±{results[approach][l]['std']:.3f}".center(12) for l in LAYERS_TO_TEST]
        print(f"{approach:<20} | " + " | ".join(row_str))

        means = [f"{results[approach][l]['mean']:.4f}" for l in LAYERS_TO_TEST]
        stds = [f"{results[approach][l]['std']:.4f}" for l in LAYERS_TO_TEST]
        csv_rows.append([approach] + means + stds)

    print("=" * 90)

    # Save CSV
    csv_file = os.path.join(args.output_dir, f"head_layers_{args.device}.csv")
    with open(csv_file, "w") as f:
        for r in csv_rows:
            f.write(",".join(r) + "\n")
    print(f"\n[SAVED] CSV Table: {csv_file}")

    # Save JSON for plotting script
    json_file = os.path.join(args.output_dir, f"head_layers_{args.device}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] JSON Data: {json_file}")


if __name__ == "__main__":
    main()

