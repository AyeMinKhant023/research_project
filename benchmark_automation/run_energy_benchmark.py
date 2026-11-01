#!/usr/bin/env python3
"""
Automated Energy & Power Benchmarking Suite for Head Layer Scaling (0 to 4 Layers)

Captures high-frequency power data from the FNIRSI FNB58 USB multimeter:
1. Starts the background power logger (`fnirsi_logger.py`).
2. Runs the FL client for target configuration (FL-ETPU, CPU 1, 2, 4 cores) and layer count (0..4).
3. Synchronizes `phases.txt` timestamps with `mydata.txt`.
4. Computes exact Energy in Millijoules (mJ) and Average Power in Watts (W) for:
   - Load Base Model Energy
   - Extract Features Energy
   - Train Classification Head Energy (for Graph 3!)
5. Exports results to CSV, JSON, and prepares data for Graph 3.
"""

import os
import sys
import time
import subprocess
import signal
import statistics
import json
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FL_DIR = os.path.join(PROJECT_ROOT, "flower_project", "federated_learning")
LOGGER_DIR = os.path.join(PROJECT_ROOT, "flower_project", "fnirsi-usb-power-data-logger")
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


def parse_energy_phases(power_data_path: str, phases_path: str):
    """Integrates power samples P = V * I over phase intervals to compute exact energy in mJ."""
    if not os.path.exists(phases_path) or not os.path.exists(power_data_path):
        return {"load_mJ": 0.0, "extract_mJ": 0.0, "train_mJ": 0.0, "total_mJ": 0.0}

    phases = {}
    with open(phases_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#phase"):
                parts = line.split()
                if len(parts) >= 3:
                    phases[parts[2]] = float(parts[1])

    load_start, load_end = phases.get("START_LOAD_MODEL"), phases.get("END_LOAD_MODEL")
    ext_start, ext_end = phases.get("START_EXTRACT_FEATURES"), phases.get("END_EXTRACT_FEATURES")
    train_start, train_end = phases.get("START_TRAIN_HEAD"), phases.get("END_TRAIN_HEAD")

    load_mJ, ext_mJ, train_mJ = 0.0, 0.0, 0.0
    prev_t = None

    with open(power_data_path, "r") as f:
        for line in f:
            if line.startswith("timestamp") or line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0])
                v = float(parts[2])
                i = float(parts[3])
                p = v * i  # Watts
            except ValueError:
                continue

            if prev_t is not None:
                dt = t - prev_t
                if 0 < dt < 0.5:
                    e_mj = p * dt * 1000.0  # Joules to mJ
                    if load_start and load_start <= t <= load_end:
                        load_mJ += e_mj
                    elif ext_start and ext_start <= t <= ext_end:
                        ext_mJ += e_mj
                    elif train_start and train_start <= t <= train_end:
                        train_mJ += e_mj
            prev_t = t

    return {
        "load_mJ": load_mJ,
        "extract_mJ": ext_mJ,
        "train_mJ": train_mJ,
        "offline_total_mJ": load_mJ + ext_mJ,
        "total_mJ": load_mJ + ext_mJ + train_mJ
    }


def run_single_energy_test(cfg, num_layers, run_idx, args, output_dir):
    """Starts logger, runs client, computes energy in mJ."""
    log_prefix = f"{cfg['prefix']}_L{num_layers}_run{run_idx}_{args.device}"
    raw_power_file = os.path.join(output_dir, f"{log_prefix}_power.txt")
    phases_file = os.path.join(LOGGER_DIR, "phases.txt")

    # Clear previous phases file
    if os.path.exists(phases_file):
        os.remove(phases_file)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {cfg['name']} | Layers: {num_layers} | Run {run_idx}/{args.runs}")
    print("  -> Starting background FNB58 power logger...")

    # Start logger process (writes raw readings to raw_power_file)
    logger_cmd = ["sudo", sys.executable, os.path.join(LOGGER_DIR, "fnirsi_logger.py")]
    with open(raw_power_file, "w") as out_fp:
        logger_proc = subprocess.Popen(logger_cmd, stdout=out_fp, stderr=subprocess.PIPE, text=True)

    time.sleep(1.5)  # Let logger initialize USB stream

    # Run client
    client_cmd = [
        sys.executable,
        os.path.join(FL_DIR, "fl_client.py"),
        "--embedding_extractor_path", cfg["model"],
        "--data_dir", args.data_dir,
        "--num_classes", str(args.num_classes),
        "--server_address", args.server_address,
        "--num_threads", str(cfg["threads"]),
        "--num_head_layers", str(num_layers),
    ]

    print("  -> Running client workload...")
    start_t = time.time()
    try:
        proc = subprocess.run(client_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: client returned {e.returncode}")

    # Stop power logger cleanly
    time.sleep(1.0)
    try:
        logger_proc.send_signal(signal.SIGINT)
        logger_proc.wait(timeout=3)
    except Exception:
        logger_proc.kill()

    # Calculate energy
    energy_data = parse_energy_phases(raw_power_file, phases_file)
    print(f"  -> Head Energy: {energy_data['train_mJ']:.2f} mJ | Extract: {energy_data['extract_mJ']:.2f} mJ | Total: {energy_data['total_mJ']:.2f} mJ")
    return energy_data


def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Head Energy Benchmarking Suite")
    parser.add_argument("--device", default="pi5", help="Device name (pi5 or pi3)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per point (default: 3 for energy)")
    parser.add_argument("--server_address", default="172.20.10.3:8080", help="Flower server IP:port")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Path to training data")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--cooldown", type=int, default=3, help="Cooldown between runs")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "results_head_energy"), help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=========================================================================")
    print("      CLASSIFICATION HEAD ENERGY CONSUMPTION BENCHMARK (0 to 4 LAYERS)   ")
    print("=========================================================================")
    print(f"Device:           {args.device}")
    print(f"Layers to test:   {LAYERS_TO_TEST}")
    print(f"Runs per setting: {args.runs}")
    print(f"Output directory: {args.output_dir}")
    print("=========================================================================\n")

    results = {}

    for cfg in CONFIGS:
        approach = cfg["name"]
        results[approach] = {}

        print(f"\n==================== CONFIGURATION: {approach} ====================")
        for num_layers in LAYERS_TO_TEST:
            head_energies = []
            for r in range(1, args.runs + 1):
                e = run_single_energy_test(cfg, num_layers, r, args, args.output_dir)
                head_energies.append(e["train_mJ"])
                if r < args.runs and args.cooldown > 0:
                    time.sleep(args.cooldown)

            mean_val = statistics.mean(head_energies) if head_energies else 0.0
            std_val = statistics.stdev(head_energies) if len(head_energies) > 1 else 0.0
            results[approach][num_layers] = {
                "mean": mean_val,
                "std": std_val,
                "raw": head_energies
            }

    # Summary table
    print("\n\n" + "=" * 90)
    print("   SUMMARY TABLE: Train Header Energy (mJ) vs. Number of Head Layers")
    print("=" * 90)
    header = "{:<20} | " + " | ".join([f"{l} Layers".center(12) for l in LAYERS_TO_TEST])
    print(header.format("Approach"))
    print("-" * 90)

    csv_rows = [["Approach"] + [f"{l}_layers_mJ_mean" for l in LAYERS_TO_TEST] + [f"{l}_layers_mJ_std" for l in LAYERS_TO_TEST]]

    for approach in results:
        row_str = [f"{results[approach][l]['mean']:.2f}±{results[approach][l]['std']:.2f}".center(12) for l in LAYERS_TO_TEST]
        print(f"{approach:<20} | " + " | ".join(row_str))

        means = [f"{results[approach][l]['mean']:.2f}" for l in LAYERS_TO_TEST]
        stds = [f"{results[approach][l]['std']:.2f}" for l in LAYERS_TO_TEST]
        csv_rows.append([approach] + means + stds)

    print("=" * 90)

    # Save CSV and JSON
    csv_file = os.path.join(args.output_dir, f"energy_head_layers_{args.device}.csv")
    with open(csv_file, "w") as f:
        for r in csv_rows:
            f.write(",".join(r) + "\n")

    json_file = os.path.join(args.output_dir, f"energy_head_layers_{args.device}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] Energy CSV:  {csv_file}")
    print(f"[SAVED] Energy JSON: {json_file}\n")


if __name__ == "__main__":
    main()

