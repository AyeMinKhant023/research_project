#!/usr/bin/env python3
"""
Master CIFAR-10 Energy Benchmarking Suite (FL-ETPU vs FL-CPU 4-Cores)

Executes energy measurements (in mJ) using live FNIRSI FNB58 USB power logging:
- Table 1 (Energy): Feature Extractor TPU (FL-ETPU) with Run 1..5 and Mean ± Std.
- Table 2 (Energy): Feature Extractor CPU (4 cores) with Run 1..5 and Mean ± Std.
- Graph 3: Classification Head Energy (in mJ, 40 FL rounds) vs. Added Layers (0 to 4).
"""

import os
import sys
import time
import subprocess
import signal
import statistics
import json
import csv
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_cifar10_energy")
FL_DIR = os.path.join(PROJECT_ROOT, "flower_project", "federated_learning")
EDGETPU_MODELS_DIR = os.path.join(PROJECT_ROOT, "edgetpu", "retrain-backprop")

DEFAULT_DATA_DIR = "./data/cifar10/client1_data"
EDGETPU_MODEL = "~/research_project/edgetpu/retrain-backprop/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite"
CPU_MODEL = "~/research_project/edgetpu/retrain-backprop/mobilenet_v1_1.0_224_quant_embedding_extractor.tflite"
PI_HOST = "ayeminkhant@172.20.10.2"
PI_PASS = "raspberrypi5"

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


def measure_clock_offset() -> float:
    """Returns clock_offset = (mac_time - pi_time) in seconds to align Pi phase markers to Mac power timestamps."""
    import pexpect

    offsets = []
    for _ in range(2):
        try:
            child = pexpect.spawn(
                "ssh",
                ["-o", "StrictHostKeyChecking=no", PI_HOST, "date +%s.%N"],
                timeout=10
            )
            idx = child.expect(["[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
            t_req = time.time()
            if idx == 0:
                t_req = time.time()
                child.sendline(PI_PASS)
                child.expect(pexpect.EOF, timeout=10)
            t_resp = time.time()
            mac_mid = 0.5 * (t_req + t_resp)
            lines = [ln.strip() for ln in child.before.decode("utf-8", errors="replace").splitlines() if ln.strip()]
            for ln in reversed(lines):
                try:
                    pi_t = float(ln)
                    if pi_t > 1e9:
                        offsets.append(mac_mid - pi_t)
                        break
                except ValueError:
                    continue
        except Exception as e:
            print(f"⚠️ Clock offset probe warning: {e}")
    if offsets:
        return statistics.median(offsets)
    return 0.0


def parse_energy_phases(power_data_path: str, phases_path: str, clock_offset: float = 0.0):
    """
    Integrates power samples P = V * I over all phase intervals (including all 40
    START_TRAIN_HEAD..END_TRAIN_HEAD federated rounds) to compute exact energy in mJ.
    """
    zero_res = {
        "load_base_model": 0.0,
        "extract_features": 0.0,
        "offline_total": 0.0,
        "train_header": 0.0,
        "runtime_total": 0.0,
        "full_total": 0.0,
        "num_train_rounds": 0,
        "power_samples": 0
    }
    if not os.path.exists(phases_path) or not os.path.exists(power_data_path):
        return zero_res

    load_intervals = []
    ext_intervals = []
    train_intervals = []

    cur_load_start = None
    cur_ext_start = None
    cur_train_start = None

    with open(phases_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#phase"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            ts = float(parts[1]) + clock_offset
            marker = parts[2]

            if marker == "START_LOAD_MODEL":
                cur_load_start = ts
            elif marker == "END_LOAD_MODEL" and cur_load_start is not None:
                load_intervals.append((cur_load_start, ts))
                cur_load_start = None
            elif marker == "START_EXTRACT_FEATURES":
                cur_ext_start = ts
            elif marker == "END_EXTRACT_FEATURES" and cur_ext_start is not None:
                ext_intervals.append((cur_ext_start, ts))
                cur_ext_start = None
            elif marker == "START_TRAIN_HEAD":
                cur_train_start = ts
            elif marker == "END_TRAIN_HEAD" and cur_train_start is not None:
                train_intervals.append((cur_train_start, ts))
                cur_train_start = None

    def overlap_fraction(t0: float, t1: float, intervals) -> float:
        """Exact overlap duration between sample interval [t0, t1] and a list of phase intervals."""
        if t1 <= t0:
            return 0.0
        total_ov = 0.0
        for a, b in intervals:
            if b <= t0:
                continue
            if a >= t1:
                break
            ov = min(t1, b) - max(t0, a)
            if ov > 0.0:
                total_ov += ov
        return total_ov

    load_mJ, ext_mJ, train_mJ = 0.0, 0.0, 0.0
    prev_t = None
    sample_count = 0

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

            sample_count += 1
            if prev_t is not None:
                dt = t - prev_t
                if 0.0 < dt < 0.5:
                    ov_load = overlap_fraction(prev_t, t, load_intervals)
                    if ov_load > 0.0:
                        load_mJ += p * ov_load * 1000.0
                    ov_ext = overlap_fraction(prev_t, t, ext_intervals)
                    if ov_ext > 0.0:
                        ext_mJ += p * ov_ext * 1000.0
                    ov_train = overlap_fraction(prev_t, t, train_intervals)
                    if ov_train > 0.0:
                        train_mJ += p * ov_train * 1000.0
                elif dt <= 0.0:
                    # Handle minor packet timestamp jitter when 4 samples share a 40ms USB frame
                    nominal_dt = 0.01
                    t_start = t - nominal_dt
                    ov_load = overlap_fraction(t_start, t, load_intervals)
                    if ov_load > 0.0:
                        load_mJ += p * ov_load * 1000.0
                    ov_ext = overlap_fraction(t_start, t, ext_intervals)
                    if ov_ext > 0.0:
                        ext_mJ += p * ov_ext * 1000.0
                    ov_train = overlap_fraction(t_start, t, train_intervals)
                    if ov_train > 0.0:
                        train_mJ += p * ov_train * 1000.0
            prev_t = t

    return {
        "load_base_model": load_mJ,
        "extract_features": ext_mJ,
        "offline_total": load_mJ + ext_mJ,
        "train_header": train_mJ,
        "runtime_total": train_mJ,
        "full_total": load_mJ + ext_mJ + train_mJ,
        "num_train_rounds": len(train_intervals),
        "power_samples": sample_count
    }


def run_single_energy_benchmark(cfg, num_layers, run_idx, args, output_dir, clock_offset=0.0, file_prefix_override=None):
    """Starts local Mac logger, runs client on Pi over SSH, pulls phases.txt, computes mJ."""
    import pexpect

    prefix = file_prefix_override or cfg["prefix"]
    raw_power_file = os.path.join(output_dir, f"{prefix}_L{num_layers}_run{run_idx}_power.txt")
    phases_local = os.path.join(output_dir, f"{prefix}_L{num_layers}_run{run_idx}_phases.txt")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {cfg['name']} | Layers: {num_layers} | Run {run_idx}/{args.runs}")
    print("  -> Starting live FNB58 power logger on Mac...")

    logger_cmd = [
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
        os.path.join(SCRIPT_DIR, "fnb58_mac_logger.py"),
        "--output", raw_power_file
    ]
    logger_proc = subprocess.Popen(logger_cmd)
    time.sleep(1.2)

    if logger_proc.poll() is not None:
        raise RuntimeError(
            f"FNB58 power logger exited immediately with code {logger_proc.returncode}. "
            "Please verify the FNB58 PC micro-USB cable is connected to the Mac."
        )

    pi_cmd = (
        f"/home/ayeminkhant/.pyenv/versions/3.9.16/bin/python3 fl_client.py "
        f"--embedding_extractor_path {cfg['model']} "
        f"--data_dir {args.data_dir} "
        f"--num_classes {args.num_classes} "
        f"--server_address {args.server_address} "
        f"--num_threads {cfg['threads']} "
        f"--num_head_layers {num_layers} "
        f"--local_epochs {args.local_epochs}"
    )

    print("  -> Executing 40-round training workload on Raspberry Pi 5...")
    start_t = time.time()
    try:
        child = pexpect.spawn(
            "ssh",
            ["-o", "StrictHostKeyChecking=no", PI_HOST, f"cd ~/research_project/flower_project/federated_learning && {pi_cmd}"],
            timeout=300
        )
        idx = child.expect(["[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
        if idx == 0:
            child.sendline(PI_PASS)
            child.expect(pexpect.EOF, timeout=300)
    except Exception as e:
        print(f"⚠️ Warning during Pi execution: {e}")

    # Allow trailing USB packets to flush before stopping logger
    time.sleep(1.0)
    try:
        logger_proc.send_signal(signal.SIGINT)
        logger_proc.wait(timeout=3)
    except Exception:
        logger_proc.kill()

    # Pull phases.txt from Pi to Mac
    try:
        child_scp = pexpect.spawn(
            "scp",
            [
                "-o", "StrictHostKeyChecking=no",
                f"{PI_HOST}:~/research_project/flower_project/fnirsi-usb-power-data-logger/phases.txt",
                phases_local
            ],
            timeout=20
        )
        idx2 = child_scp.expect(["[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
        if idx2 == 0:
            child_scp.sendline(PI_PASS)
            child_scp.expect(pexpect.EOF, timeout=20)
    except Exception as e:
        print(f"⚠️ Warning pulling phases: {e}")

    energy_data = parse_energy_phases(raw_power_file, phases_local, clock_offset=clock_offset)
    print(
        f"   -> Samples: {energy_data['power_samples']} | Rounds: {energy_data['num_train_rounds']} | "
        f"Load: {energy_data['load_base_model']:.2f} mJ | Extract: {energy_data['extract_features']:.2f} mJ | "
        f"Train Head: {energy_data['train_header']:.2f} mJ (wall: {time.time()-start_t:.2f}s)"
    )
    if energy_data["power_samples"] < 200 or energy_data["extract_features"] <= 0.0:
        raise RuntimeError(
            f"Captured insufficient power samples ({energy_data['power_samples']}) or 0 mJ feature extraction! "
            "Aborting immediately so invalid 0.00 tables are never written."
        )
    return energy_data


def format_stat(mean_v, std_v):
    return f"{mean_v:.2f} ± {std_v:.2f}"


def export_energy_table(cfg, raw_runs, device_name, output_dir):
    """Generates Energy Table with Run 1..5 and Average (Mean ± Std)."""
    table_name = f"Energy_Table_{cfg['prefix']}_{device_name}"
    csv_file = os.path.join(output_dir, f"{table_name}.csv")
    latex_file = os.path.join(output_dir, f"{table_name}.tex")

    cols = ["Run", "Load Base Model (mJ)", "Extract Features (mJ)", "Offline Total (mJ)", "Train Header (40 Rds) (mJ)", "Runtime Total (mJ)"]
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
            f"{r['train_header']:.2f}",
            f"{r['runtime_total']:.2f}"
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

    print("\n" + "=" * 95)
    print(f"   {cfg['name'].upper()} - ENERGY CONSUMPTION (mJ) [{device_name.upper()}]")
    print("=" * 95)
    col_fmt = "{:<22} | {:<16} | {:<16} | {:<16} | {:<18} | {:<16}"
    print(col_fmt.format(*cols))
    print("-" * 95)
    for r in rows[:-1]:
        print(col_fmt.format(*r))
    print("-" * 95)
    print(col_fmt.format(*rows[-1]))
    print("=" * 95)

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)
    print(f"✅ Saved CSV: {csv_file}")

    latex_rows = []
    for r in rows:
        latex_rows.append(f"  {r[0]:<22} & {r[1]:<14} & {r[2]:<14} & {r[3]:<14} & {r[4]:<14} & {r[5]:<14} \\\\")

    latex_code = (
        f"% {cfg['name']} Energy Consumption on {device_name}\n"
        f"\\begin{{table}}[t]\n"
        f"\\centering\n"
        f"\\caption{{{cfg['name']} energy consumption (in mJ) across {len(raw_runs)} runs on {device_name}}}\n"
        f"\\label{{tab:energy_{cfg['prefix']}_{device_name}}}\n"
        f"\\begin{{tabular}}{{lccccc}}\n"
        f"\\hline\n"
        f"\\multirow{{2}}{{*}}{{Run}} & \\multicolumn{{3}}{{c}}{{Offline Phase}} & \\multicolumn{{2}}{{c}}{{Runtime Phase}} \\\\\n"
        f"\\cline{{2-6}}\n"
        f" & Load Base Model & Extract Features & Total & Train Header & Total \\\\\n"
        f"\\hline\n"
        + "\n".join(latex_rows[:-1]) + "\n"
        + "\\hline\n"
        + latex_rows[-1] + "\n"
        + "\\hline\n"
        + "\\end{tabular}\n"
        + "\\end{table}\n"
    )
    with open(latex_file, "w") as f:
        f.write(latex_code)
    print(f"✅ Saved LaTeX: {latex_file}")


def plot_head_energy_graph(head_scaling_data, device_name, output_dir):
    """Generates Graph 3: Classification Head Energy vs. Added Layers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not found, skipping plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    layers = [0, 1, 2, 3, 4]
    means = [head_scaling_data[l]["mean"] for l in layers]
    stds = [head_scaling_data[l]["std"] for l in layers]

    ax.errorbar(
        layers, means, yerr=stds,
        label="Classification Head (CPU 4-cores, 40 Rounds)",
        color="#d95f02",
        marker="s",
        linewidth=2.2,
        markersize=7,
        capsize=4,
        capthick=1.5
    )

    ax.set_title(f"Classification Head Energy Consumption vs. Added Layers ({device_name.upper()})", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Number of Added Hidden Layers in Head (256 Units Each)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Train Header Energy (mJ)", fontsize=10, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels(["0\n(Baseline)", "1", "2", "3", "4"])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"graph_classification_head_energy_{device_name}.png")
    pdf_path = os.path.join(output_dir, f"graph_classification_head_energy_{device_name}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print("\n🎉 Graph 3 saved successfully:")
    print(f"   - High-Res PNG: {png_path}")
    print(f"   - Vector PDF:   {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Master CIFAR-10 Energy Benchmark Runner")
    parser.add_argument("--device", default="pi5", help="Target device")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per configuration")
    parser.add_argument("--server_address", default="172.20.10.3:8080", help="Flower server IP:Port")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Path to training data")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--local_epochs", type=int, default=10, help="Local epochs per round")
    parser.add_argument("--cooldown", type=int, default=2, help="Cooldown between runs")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Results directory")
    parser.add_argument("--smoke_test", action="store_true", help="Run a single EdgeTPU L0 verification trial and exit")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=========================================================================")
    print("      CIFAR-10 FEDERATED LEARNING ENERGY BENCHMARK SUITE (MAX CORES)     ")
    print("=========================================================================")
    print(f"Device:           {args.device}")
    print(f"Dataset:          {args.data_dir} ({args.num_classes} classes)")
    print(f"Local Epochs:     {args.local_epochs} (40 rounds total)")
    print(f"Runs per setting: {args.runs} (Run 1..{args.runs} + Mean ± Std)")
    print(f"Output directory: {args.output_dir}")
    clock_offset = measure_clock_offset()
    print(f"Clock Offset:     {clock_offset:+.4f}s (Mac - Pi)")
    print("=========================================================================\n")

    if args.smoke_test:
        print("=== RUNNING SINGLE SMOKE TEST (EdgeTPU L0) ===")
        e = run_single_energy_benchmark(
            CONFIGS[0], num_layers=0, run_idx=0, args=args, output_dir=args.output_dir,
            clock_offset=clock_offset, file_prefix_override="SmokeTest_EdgeTPU"
        )
        print("Smoke test result:", json.dumps(e, indent=2))
        return

    # Step 1: Feature Extractor Energy Benchmarks (Table 1: EdgeTPU & Table 2: CPU4)
    step1_results = {}
    for cfg in CONFIGS:
        print(f"\n==================== BENCHMARKING ENERGY: {cfg['name']} ====================")
        clock_offset = measure_clock_offset()
        raw_runs = []
        for r in range(1, args.runs + 1):
            e = run_single_energy_benchmark(
                cfg, num_layers=0, run_idx=r, args=args,
                output_dir=args.output_dir, clock_offset=clock_offset
            )
            raw_runs.append(e)
            if r < args.runs and args.cooldown > 0:
                time.sleep(args.cooldown)

        step1_results[cfg["prefix"]] = raw_runs
        export_energy_table(cfg, raw_runs, args.device, args.output_dir)

    # Step 2: Head Scaling Energy (Graph 3: 0 to 4 Added Hidden Layers)
    # Reuse Layer 0 from Step 1 (EdgeTPU backbone + 4-core CPU head) and use fast EdgeTPU
    # embedding extraction (8s vs 87s) so we don't waste 35+ minutes re-extracting frozen features
    # while measuring 4-core CPU classification head training across all 40 FL rounds!
    print("\n==================== BENCHMARKING ENERGY: HEAD SCALING (0 to 4 Layers) ====================")
    head_scaling_energy = {}
    fast_backbone_cfg = CONFIGS[0]

    for num_layers in LAYERS_TO_TEST:
        if num_layers == 0 and "EdgeTPU" in step1_results:
            print("\n--- Reusing Layer 0 (Baseline) 5-run head energy measurements from Step 1 ---")
            train_energies = [r["train_header"] for r in step1_results["EdgeTPU"]]
        else:
            print(f"\n--- Testing Head with {num_layers} added layers (256 units each, 4-core CPU) ---")
            clock_offset = measure_clock_offset()
            train_energies = []
            for r in range(1, args.runs + 1):
                e = run_single_energy_benchmark(
                    fast_backbone_cfg, num_layers=num_layers, run_idx=r, args=args,
                    output_dir=args.output_dir, clock_offset=clock_offset,
                    file_prefix_override="HeadScale"
                )
                train_energies.append(e["train_header"])
                if r < args.runs and args.cooldown > 0:
                    time.sleep(args.cooldown)

        head_scaling_energy[num_layers] = {
            "mean": statistics.mean(train_energies),
            "std": statistics.stdev(train_energies) if len(train_energies) > 1 else 0.0,
            "raw": train_energies
        }
        print(
            f"   => Layer {num_layers} Head Energy: "
            f"{head_scaling_energy[num_layers]['mean']:.2f} ± {head_scaling_energy[num_layers]['std']:.2f} mJ"
        )

    head_json_path = os.path.join(args.output_dir, f"energy_head_scaling_{args.device}.json")
    with open(head_json_path, "w") as f:
        json.dump(head_scaling_energy, f, indent=2)
    print(f"✅ Saved Head Scaling JSON: {head_json_path}")

    plot_head_energy_graph(head_scaling_energy, args.device, args.output_dir)

    print("\n" + "=" * 90)
    print("🎉 ALL ENERGY BENCHMARKS, TABLES, AND GRAPHS COMPLETED SUCCESSFULLY!")
    print(f"Results stored in: {args.output_dir}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
