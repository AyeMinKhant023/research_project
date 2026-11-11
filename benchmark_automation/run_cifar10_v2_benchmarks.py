#!/usr/bin/env python3
"""
Master CIFAR-10 V2 Automated Benchmark Suite (Raspberry Pi 5)
=============================================================

Implements the updated architecture & evaluation protocol:
- Base Feature Extractors ("all"): MobileNetV1 (224x224, dim=1024) & InceptionV3 (299x299, dim=2048)
- Hardware Execution Modes: CPU (4 cores) vs Coral EdgeTPU
- Classification Head Scaling: 0, 3, 6, 9, 12 hidden layers (512 neurons + ReLU each)
- Training Setup: CIFAR-10 (10 classes), 40 FL rounds, 10 local epochs/round, 5 runs/setting
- Dual Capture: Collects BOTH Runtime (s) and Energy (mJ via FNIRSI FNB58) in a single pass
- Accuracy Tracking: Captures per-round (1..40) Test Accuracy & Loss on Layer 12 (CPU4 & EdgeTPU)
- Output Folder: benchmark_automation/results_cifar10_v2_pi5/{mobilenet, inception_v3}/

Usage:
  1. Verify environment & scripts (no FNB58 required):
     .venv/bin/python benchmark_automation/run_cifar10_v2_benchmarks.py --check_setup

  2. Smoke test (1 trial run of 40 rounds with Layer 12 accuracy tracking):
     .venv/bin/python benchmark_automation/run_cifar10_v2_benchmarks.py --smoke_test --models mobilenet

  3. Full unattended run (Runtime + Energy + L12 Per-Round Accuracy):
     .venv/bin/python benchmark_automation/run_cifar10_v2_benchmarks.py
"""

import argparse
import csv
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FL_DIR = os.path.join(PROJECT_ROOT, "flower_project", "federated_learning")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_cifar10_v2_pi5")

DEFAULT_PI_HOST = "ayeminkhant@172.20.10.2"
DEFAULT_PI_IP = "172.20.10.2"
DEFAULT_PI_PASS = "raspberrypi5"
DEFAULT_PI_PYTHON = "/home/ayeminkhant/.pyenv/versions/3.9.16/bin/python3"
DEFAULT_DATA_DIR = "./data/cifar10/client1_data"

LAYERS_TO_TEST = [0, 3, 6, 9, 12]
HIDDEN_DIM = 512

BACKBONES = {
    "mobilenet": {
        "display_name": "MobileNetV1 (224x224)",
        "short_name": "MobileNet",
        "feature_dim": 1024,
        "base_params": 3196032,
        "cpu_model": "~/research_project/edgetpu/retrain-backprop/mobilenet_v1_1.0_224_quant_embedding_extractor.tflite",
        "edgetpu_model": "~/research_project/edgetpu/retrain-backprop/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite",
    },
    "inception_v3": {
        "display_name": "InceptionV3 (299x299)",
        "short_name": "InceptionV3",
        "feature_dim": 2048,
        "base_params": None,
        "cpu_model": "~/research_project/edgetpu/retrain-backprop/inception_v3_avg_pool.tflite",
        "edgetpu_model": "~/research_project/edgetpu/retrain-backprop/inception_v3_avg_pool_edgetpu.tflite",
    },
}


def detect_local_ip_for_peer(peer_ip: str = DEFAULT_PI_IP) -> str:
    """Determines the Mac's outbound LAN/hotspot IP address reaching peer_ip."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer_ip, 8080))
        return s.getsockname()[0]
    except Exception:
        return "172.20.10.3"
    finally:
        s.close()


def check_fnb58_connected() -> bool:
    """Checks whether the FNIRSI FNB58 (VID 0x2E3C, PID 0x5558) is visible over HID."""
    try:
        import hid
        return len(hid.enumerate(0x2E3C, 0x5558)) > 0
    except Exception:
        return False


def measure_clock_offset(pi_host: str, pi_pass: str) -> float:
    """Returns median (mac_time - pi_time) in seconds to align Pi phase timestamps."""
    import pexpect

    offsets = []
    for _ in range(2):
        try:
            child = pexpect.spawn(
                "ssh",
                ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8", pi_host, "date +%s.%N"],
                timeout=12,
            )
            idx = child.expect(["[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
            t_req = time.time()
            if idx == 0:
                t_req = time.time()
                child.sendline(pi_pass)
                child.expect(pexpect.EOF, timeout=12)
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
    return statistics.median(offsets) if offsets else 0.0


def parse_timings_from_log(log_text: str):
    """Parses exact runtime metrics (in seconds) from fl_client.py stdout."""
    m_load = re.search(r"Time for Load Feature Extractor[^:]*:\s*([\d\.]+)", log_text)
    m_ext = re.search(r"Time for Extract Train Embeddings[^:]*:\s*([\d\.]+)", log_text)
    train_matches = re.findall(r"Time for\s+Train Classification Head[^:]*:\s*([\d\.]+)", log_text)
    m_total = re.search(r"Total Runtime \(Full Client Run\):\s*([\d\.]+)", log_text)
    m_feat_dim = re.search(r"- Feature dimension:\s*(\d+)", log_text)

    load_time = float(m_load.group(1)) if m_load else 0.0
    ext_time = float(m_ext.group(1)) if m_ext else 0.0
    offline_total = load_time + ext_time
    train_time = float(train_matches[-1]) if train_matches else 0.0
    runtime_total = train_time
    full_total = float(m_total.group(1)) if m_total else (offline_total + runtime_total)
    feature_dim = int(m_feat_dim.group(1)) if m_feat_dim else 0

    return {
        "load_base_model": load_time,
        "extract_features": ext_time,
        "offline_total": offline_total,
        "train_header": train_time,
        "runtime_total": runtime_total,
        "full_total": full_total,
        "feature_dim": feature_dim,
    }


def parse_round_metrics_from_log(log_text: str):
    """Parses structured per-round [ROUND_METRICS] JSON lines emitted during Layer-12 accuracy tracking."""
    records = []
    for line in log_text.splitlines():
        if "[ROUND_METRICS]" in line:
            idx = line.find("{")
            if idx != -1:
                try:
                    records.append(json.loads(line[idx:]))
                except json.JSONDecodeError:
                    continue
    return records


def parse_energy_phases(power_data_path: str, phases_path: str, clock_offset: float = 0.0):
    """Integrates power samples P = V * I over all phase intervals (including all 40 FL rounds) in mJ."""
    zero_res = {
        "load_base_model": 0.0,
        "extract_features": 0.0,
        "offline_total": 0.0,
        "train_header": 0.0,
        "runtime_total": 0.0,
        "full_total": 0.0,
        "num_train_rounds": 0,
        "power_samples": 0,
    }
    if not os.path.exists(phases_path) or not os.path.exists(power_data_path):
        return zero_res

    load_intervals, ext_intervals, train_intervals = [], [], []
    cur_load, cur_ext, cur_train = None, None, None

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
                cur_load = ts
            elif marker == "END_LOAD_MODEL" and cur_load is not None:
                load_intervals.append((cur_load, ts))
                cur_load = None
            elif marker == "START_EXTRACT_FEATURES":
                cur_ext = ts
            elif marker == "END_EXTRACT_FEATURES" and cur_ext is not None:
                ext_intervals.append((cur_ext, ts))
                cur_ext = None
            elif marker == "START_TRAIN_HEAD":
                cur_train = ts
            elif marker == "END_TRAIN_HEAD" and cur_train is not None:
                train_intervals.append((cur_train, ts))
                cur_train = None

    load_intervals.sort()
    ext_intervals.sort()
    train_intervals.sort()

    def overlap_fraction(t0: float, t1: float, intervals) -> float:
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
                p = v * i
            except ValueError:
                continue

            sample_count += 1
            if prev_t is not None:
                dt = t - prev_t
                if 0.0 < dt < 0.5:
                    t_start, t_end = prev_t, t
                elif dt <= 0.0:
                    t_start, t_end = t - 0.01, t
                else:
                    t_start, t_end = None, None

                if t_start is not None:
                    ov_load = overlap_fraction(t_start, t_end, load_intervals)
                    if ov_load > 0.0:
                        load_mJ += p * ov_load * 1000.0
                    ov_ext = overlap_fraction(t_start, t_end, ext_intervals)
                    if ov_ext > 0.0:
                        ext_mJ += p * ov_ext * 1000.0
                    ov_train = overlap_fraction(t_start, t_end, train_intervals)
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
        "power_samples": sample_count,
    }


def start_managed_flower_server(args):
    """Starts fl_server.py with --auto_restart and waits until port 8080 is ready."""
    port = int(args.server_address.split(":")[-1]) if ":" in args.server_address else 8080
    cmd = [
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
        "-u",
        os.path.join(FL_DIR, "fl_server.py"),
        "--feature_dim", "1024",
        "--num_classes", str(args.num_classes),
        "--min_clients", "1",
        "--num_rounds", str(args.num_rounds),
        "--hidden_dim", str(args.hidden_dim),
        "--server_address", f"0.0.0.0:{port}",
        "--auto_restart",
    ]
    log_path = os.path.join(args.output_dir, "flower_server.log")
    os.makedirs(args.output_dir, exist_ok=True)
    log_fp = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)

    for _ in range(30):
        if proc.poll() is not None:
            log_fp.close()
            raise RuntimeError(f"Flower server exited early with code {proc.returncode}. See {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                print(f"✅ Managed Flower server listening on 0.0.0.0:{port} (advertising {args.server_address})")
                return proc, log_fp
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for Flower server on port {port}. See {log_path}")


def run_single_trial(backbone_key, cfg, num_layers, run_idx, args, raw_dir, clock_offset=0.0):
    """Runs 1 complete 40-round Pi trial, capturing runtime + FNB58 energy + (if L==12) round accuracy."""
    import pexpect

    stem = f"{cfg['prefix']}_L{num_layers}_run{run_idx}"
    result_json = os.path.join(raw_dir, f"{stem}_result.json")
    raw_power_file = os.path.join(raw_dir, f"{stem}_power.txt")
    phases_local = os.path.join(raw_dir, f"{stem}_phases.txt")
    client_log_file = os.path.join(raw_dir, f"{stem}_log.txt")
    acc_json_file = os.path.join(raw_dir, f"{stem}_acc.json")

    if not args.fresh and os.path.exists(result_json):
        try:
            with open(result_json, "r") as f:
                cached = json.load(f)
            need_energy = (args.mode == "both")
            has_energy = cached.get("energy", {}).get("power_samples", 0) >= 200 and cached.get("energy", {}).get("extract_features", 0.0) > 0.0
            has_runtime = cached.get("runtime", {}).get("extract_features", 0.0) > 0.0 and cached.get("runtime", {}).get("train_header", 0.0) > 0.0
            need_acc = (num_layers == args.acc_layer)
            has_acc = (not need_acc) or (len(cached.get("round_metrics", [])) >= args.num_rounds)
            if has_runtime and (not need_energy or has_energy) and has_acc:
                print(f"⏭️  [RESUME] Skipping completed {backbone_key} | {stem}")
                return cached
        except Exception:
            pass

    track_acc = (num_layers == args.acc_layer)
    measure_energy = (args.mode == "both")

    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] {BACKBONES[backbone_key]['short_name']} | "
        f"{cfg['name']} | Layers: {num_layers} (w={args.hidden_dim}) | Run {run_idx}/{args.runs}"
        + (" | [+Accuracy Tracking]" if track_acc else "")
    )

    logger_proc = None
    if measure_energy:
        print("  -> Starting live FNB58 power logger on Mac...")
        logger_cmd = [
            os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
            os.path.join(SCRIPT_DIR, "fnb58_mac_logger.py"),
            "--output", raw_power_file,
        ]
        logger_proc = subprocess.Popen(logger_cmd)
        time.sleep(1.2)
        if logger_proc.poll() is not None:
            raise RuntimeError(
                f"FNB58 power logger exited with code {logger_proc.returncode}. "
                "Plug FNB58 PC port into Mac, or pass --mode runtime."
            )

    pi_cmd = (
        f"OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 {args.pi_python} -u fl_client.py "
        f"--embedding_extractor_path {cfg['model']} "
        f"--data_dir {args.data_dir} "
        f"--num_classes {args.num_classes} "
        f"--server_address {args.server_address} "
        f"--num_threads {cfg['threads']} "
        f"--num_head_layers {num_layers} "
        f"--hidden_dim {args.hidden_dim} "
        f"--local_epochs {args.local_epochs} "
        f"--seed 42"
    )
    if track_acc:
        pi_cmd += " --track_accuracy"

    print(f"  -> Executing {args.num_rounds}-round FL client on Raspberry Pi 5...")
    start_t = time.time()
    ssh_output = ""
    child = None
    with open(client_log_file, "w", encoding="utf-8", errors="replace", buffering=1) as log_fp:
        try:
            child = pexpect.spawn(
                "ssh",
                [
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ConnectTimeout=15",
                    "-o", "ServerAliveInterval=15",
                    "-o", "ServerAliveCountMax=3",
                    args.pi_host,
                    f"cd ~/research_project/flower_project/federated_learning && {pi_cmd}",
                ],
                timeout=args.ssh_timeout,
                encoding="utf-8",
                codec_errors="replace",
            )
            child.logfile_read = log_fp
            idx = child.expect(["[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
            if idx == 0:
                child.sendline(args.pi_pass)
                child.expect(pexpect.EOF, timeout=args.ssh_timeout)
        except Exception as e:
            print(f"⚠️ Warning during Pi execution: {e}")
        finally:
            if child is not None:
                try:
                    child.close()
                except Exception:
                    pass

    if os.path.exists(client_log_file):
        with open(client_log_file, "r", encoding="utf-8", errors="replace") as f:
            ssh_output = f.read()

    if logger_proc is not None:
        time.sleep(1.0)
        try:
            logger_proc.send_signal(signal.SIGINT)
            logger_proc.wait(timeout=3)
        except Exception:
            logger_proc.kill()

    runtime_data = parse_timings_from_log(ssh_output)
    round_metrics = parse_round_metrics_from_log(ssh_output) if track_acc else []
    if track_acc and round_metrics:
        with open(acc_json_file, "w") as f:
            json.dump(round_metrics, f, indent=2)

    energy_data = {
        "load_base_model": 0.0,
        "extract_features": 0.0,
        "offline_total": 0.0,
        "train_header": 0.0,
        "runtime_total": 0.0,
        "full_total": 0.0,
        "num_train_rounds": 0,
        "power_samples": 0,
    }
    if measure_energy:
        try:
            child_scp = pexpect.spawn(
                "scp",
                [
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ConnectTimeout=10",
                    "-o", "ServerAliveInterval=10",
                    "-o", "ServerAliveCountMax=2",
                    f"{args.pi_host}:~/research_project/flower_project/fnirsi-usb-power-data-logger/phases.txt",
                    phases_local,
                ],
                timeout=20,
            )
            idx2 = child_scp.expect(["[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
            if idx2 == 0:
                child_scp.sendline(args.pi_pass)
                child_scp.expect(pexpect.EOF, timeout=20)
        except Exception as e:
            print(f"⚠️ Warning pulling phases: {e}")

        energy_data = parse_energy_phases(raw_power_file, phases_local, clock_offset=clock_offset)

    wall = time.time() - start_t
    print(
        f"   -> Runtime: Load={runtime_data['load_base_model']:.2f}s | Extract={runtime_data['extract_features']:.2f}s | "
        f"TrainHead={runtime_data['train_header']:.2f}s (wall={wall:.1f}s)"
    )
    if measure_energy:
        print(
            f"   -> Energy:  Samples={energy_data['power_samples']} | Rounds={energy_data['num_train_rounds']} | "
            f"Load={energy_data['load_base_model']:.2f} mJ | Extract={energy_data['extract_features']:.2f} mJ | "
            f"TrainHead={energy_data['train_header']:.2f} mJ"
        )
    if track_acc and round_metrics:
        last = round_metrics[-1]
        print(
            f"   -> Accuracy: Rounds={len(round_metrics)} | Final TrainAcc={last['train_acc']:.4f} | "
            f"Final TestAcc={last['test_acc']:.4f} | TestLoss={last['test_loss']:.4f}"
        )

    if runtime_data["extract_features"] <= 0.0 or runtime_data["train_header"] <= 0.0:
        raise RuntimeError(f"Invalid runtime parsed from {client_log_file}. Check Pi client stdout/server connection.")
    if measure_energy and (energy_data["power_samples"] < 200 or energy_data["extract_features"] <= 0.0):
        raise RuntimeError(
            f"Captured insufficient power samples ({energy_data['power_samples']}) or 0 mJ extraction for {stem}."
        )

    record = {
        "backbone": backbone_key,
        "prefix": cfg["prefix"],
        "num_layers": num_layers,
        "hidden_dim": args.hidden_dim,
        "run_idx": run_idx,
        "runtime": runtime_data,
        "energy": energy_data,
        "round_metrics": round_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(result_json, "w") as f:
        json.dump(record, f, indent=2)
    return record


def format_stat(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f} ± {s:.2f}"


def export_feature_extractor_table(backbone_info, cfg, raw_runs, metric_key, unit, device_name, model_dir):
    """Exports CSV & LaTeX table for either Runtime (s) or Energy (mJ) at Baseline L=0."""
    kind = "Runtime" if metric_key == "runtime" else "Energy"
    table_name = f"{kind}_Table_{cfg['prefix']}_{device_name}"
    csv_file = os.path.join(model_dir, f"{table_name}.csv")
    latex_file = os.path.join(model_dir, f"{table_name}.tex")

    cols = [
        "Run",
        f"Load Base Model ({unit})",
        f"Extract Features ({unit})",
        f"Offline Total ({unit})",
        f"Train Header (40 Rds) ({unit})",
        f"Runtime Total ({unit})",
    ]
    rows = []

    loads = [r[metric_key]["load_base_model"] for r in raw_runs]
    extracts = [r[metric_key]["extract_features"] for r in raw_runs]
    offlines = [r[metric_key]["offline_total"] for r in raw_runs]
    trains = [r[metric_key]["train_header"] for r in raw_runs]
    runtimes = [r[metric_key]["runtime_total"] for r in raw_runs]

    for idx, r in enumerate(raw_runs, 1):
        d = r[metric_key]
        rows.append([
            f"Run {idx}",
            f"{d['load_base_model']:.2f}",
            f"{d['extract_features']:.2f}",
            f"{d['offline_total']:.2f}",
            f"{d['train_header']:.2f}",
            f"{d['runtime_total']:.2f}",
        ])

    rows.append([
        "Average (Mean ± Std)",
        format_stat(loads),
        format_stat(extracts),
        format_stat(offlines),
        format_stat(trains),
        format_stat(runtimes),
    ])

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    latex_rows = [
        f"  {r[0]:<22} & {r[1]:<14} & {r[2]:<14} & {r[3]:<14} & {r[4]:<14} & {r[5]:<14} \\\\"
        for r in rows
    ]
    caption_unit = "seconds" if metric_key == "runtime" else "mJ"
    latex_code = (
        f"% {backbone_info['short_name']} - {cfg['name']} {kind} ({unit}) on {device_name}\n"
        f"\\begin{{table}}[t]\n"
        f"\\centering\n"
        f"\\caption{{{backbone_info['short_name']}: {cfg['name']} {kind.lower()} ({caption_unit}) across {len(raw_runs)} runs on {device_name}}}\n"
        f"\\label{{tab:{kind.lower()}_{backbone_info['short_name'].lower()}_{cfg['prefix']}_{device_name}}}\n"
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

    print(f"✅ Saved {table_name}.csv & .tex")


def plot_head_scaling_graphs(backbone_info, scaling_by_prefix, metric_key, unit, device_name, model_dir, hidden_dim):
    """Generates separate (Base-CPU, Base-ETPU) and combined head-scaling plots for 0,3,6,9,12 layers."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    kind = "runtime" if metric_key == "runtime" else "energy"
    ylabel = f"Train Header Latency ({unit})" if metric_key == "runtime" else f"Train Header Energy ({unit})"

    styles = {
        "CPU4": {"color": "#1f77b4", "marker": "o", "title_tag": "Base: CPU (4 cores)", "file_tag": "CPU4"},
        "EdgeTPU": {"color": "#d95f02", "marker": "s", "title_tag": "Base: EdgeTPU", "file_tag": "ETPU"},
    }

    # Individual graphs (Base - CPU and Base - ETPU)
    for prefix, st in styles.items():
        series = scaling_by_prefix.get(prefix)
        if not series:
            continue
        avail_layers = [l for l in LAYERS_TO_TEST if str(l) in series]
        if not avail_layers:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
        means = [series[str(l)][metric_key]["mean"] for l in avail_layers]
        stds = [series[str(l)][metric_key]["std"] for l in avail_layers]

        ax.errorbar(
            avail_layers, means, yerr=stds,
            label=f"{backbone_info['short_name']} ({st['title_tag']}, {hidden_dim} neurons/layer)",
            color=st["color"], marker=st["marker"], linewidth=2.2, markersize=7, capsize=4, capthick=1.5,
        )
        ax.set_title(
            f"{backbone_info['short_name']}: Head {kind.title()} vs. Added Layers ({st['title_tag']}, {device_name.upper()})",
            fontsize=10.5, fontweight="bold", pad=10,
        )
        ax.set_xlabel(f"Number of Added Hidden Layers ({hidden_dim} Neurons Each)", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_xticks(avail_layers)
        ax.set_xticklabels(["0\n(Baseline)" if l == 0 else str(l) for l in avail_layers])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", frameon=True, fontsize=8.5)
        plt.tight_layout()

        stem = f"graph_head_{kind}_{st['file_tag']}_{device_name}"
        fig.savefig(os.path.join(model_dir, f"{stem}.png"), dpi=300)
        fig.savefig(os.path.join(model_dir, f"{stem}.pdf"))
        plt.close(fig)
        print(f"📊 Saved {stem}.png & .pdf")

    # Combined Base-CPU vs Base-ETPU comparison graph
    if "CPU4" in scaling_by_prefix and "EdgeTPU" in scaling_by_prefix:
        fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=300)
        common_layers = sorted({int(k) for k in scaling_by_prefix["CPU4"]} | {int(k) for k in scaling_by_prefix["EdgeTPU"]})
        plotted_any = False
        for prefix, st in styles.items():
            series = scaling_by_prefix[prefix]
            avail_layers = [l for l in LAYERS_TO_TEST if str(l) in series]
            if not avail_layers:
                continue
            plotted_any = True
            means = [series[str(l)][metric_key]["mean"] for l in avail_layers]
            stds = [series[str(l)][metric_key]["std"] for l in avail_layers]
            ax.errorbar(
                avail_layers, means, yerr=stds,
                label=st["title_tag"], color=st["color"], marker=st["marker"],
                linewidth=2.2, markersize=7, capsize=4, capthick=1.5,
            )
        if plotted_any:
            ax.set_title(
                f"{backbone_info['short_name']}: Head {kind.title()} Comparison (CPU4 vs EdgeTPU Base, {device_name.upper()})",
                fontsize=10.5, fontweight="bold", pad=10,
            )
            ax.set_xlabel(f"Number of Added Hidden Layers ({hidden_dim} Neurons Each)", fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
            xticks = [l for l in LAYERS_TO_TEST if l in common_layers] or LAYERS_TO_TEST
            ax.set_xticks(xticks)
            ax.set_xticklabels(["0\n(Baseline)" if l == 0 else str(l) for l in xticks])
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper left", frameon=True, fontsize=9)
            plt.tight_layout()

            stem = f"graph_head_{kind}_combined_{device_name}"
            fig.savefig(os.path.join(model_dir, f"{stem}.png"), dpi=300)
            fig.savefig(os.path.join(model_dir, f"{stem}.pdf"))
        plt.close(fig)


def export_accuracy_results(backbone_info, acc_runs_by_prefix, acc_layer, device_name, model_dir):
    """Exports per-round (1..40) Test Accuracy & Loss CSVs and comparison graph for Layer 12."""
    import matplotlib.pyplot as plt

    styles = {
        "CPU4": {"color": "#1f77b4", "marker": "o", "label": f"Base: CPU 4-cores (L{acc_layer})", "tag": "CPU4"},
        "EdgeTPU": {"color": "#d95f02", "marker": "s", "label": f"Base: EdgeTPU (L{acc_layer})", "tag": "ETPU"},
    }

    agg_curves = {}
    for prefix, runs in acc_runs_by_prefix.items():
        valid_runs = [r["round_metrics"] for r in runs if r.get("round_metrics")]
        if not valid_runs:
            continue
        max_rounds = max(len(rm) for rm in valid_runs)
        csv_path = os.path.join(model_dir, f"accuracy_L{acc_layer}_rounds_{styles[prefix]['tag']}_{device_name}.csv")

        header = ["Round", "Test Acc Mean", "Test Acc Std", "Test Loss Mean", "Train Acc Mean", "Cum Train Time (s)"]
        header.extend([f"Run{i+1} Test Acc" for i in range(len(valid_runs))])

        round_nums, test_acc_mean, test_acc_std, test_loss_mean = [], [], [], []
        rows = []
        for r_idx in range(max_rounds):
            rnd_num = r_idx + 1
            t_accs = [rm[r_idx]["test_acc"] for rm in valid_runs if r_idx < len(rm)]
            t_losses = [rm[r_idx]["test_loss"] for rm in valid_runs if r_idx < len(rm)]
            tr_accs = [rm[r_idx]["train_acc"] for rm in valid_runs if r_idx < len(rm)]
            cum_ts = [rm[r_idx]["cum_train_sec"] for rm in valid_runs if r_idx < len(rm)]

            m_acc = statistics.mean(t_accs)
            s_acc = statistics.stdev(t_accs) if len(t_accs) > 1 else 0.0
            m_loss = statistics.mean(t_losses)
            m_tr = statistics.mean(tr_accs)
            m_ct = statistics.mean(cum_ts)

            round_nums.append(rnd_num)
            test_acc_mean.append(m_acc * 100.0)
            test_acc_std.append(s_acc * 100.0)
            test_loss_mean.append(m_loss)

            row = [rnd_num, f"{m_acc:.4f}", f"{s_acc:.4f}", f"{m_loss:.4f}", f"{m_tr:.4f}", f"{m_ct:.2f}"]
            for rm in valid_runs:
                row.append(f"{rm[r_idx]['test_acc']:.4f}" if r_idx < len(rm) else "")
            rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"✅ Saved {os.path.basename(csv_path)}")

        agg_curves[prefix] = {
            "rounds": round_nums,
            "test_acc_pct_mean": test_acc_mean,
            "test_acc_pct_std": test_acc_std,
            "test_loss_mean": test_loss_mean,
        }

    if not agg_curves:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=300)

    for prefix, curve in agg_curves.items():
        st = styles[prefix]
        rnds = curve["rounds"]
        acc_m = curve["test_acc_pct_mean"]
        acc_s = curve["test_acc_pct_std"]
        loss_m = curve["test_loss_mean"]

        ax1.plot(rnds, acc_m, label=st["label"], color=st["color"], linewidth=2.0, marker=st["marker"], markevery=4, markersize=5)
        low = [a - s for a, s in zip(acc_m, acc_s)]
        high = [a + s for a, s in zip(acc_m, acc_s)]
        ax1.fill_between(rnds, low, high, color=st["color"], alpha=0.15)

        ax2.plot(rnds, loss_m, label=st["label"], color=st["color"], linewidth=2.0, marker=st["marker"], markevery=4, markersize=5)

    ax1.set_title(f"{backbone_info['short_name']}: Layer {acc_layer} Test Accuracy per Round", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Federated Learning Round", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)", fontsize=9.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5)

    ax2.set_title(f"{backbone_info['short_name']}: Layer {acc_layer} Test Loss per Round", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Federated Learning Round", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Test Loss", fontsize=9.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle(f"Layer-{acc_layer} (512 Neurons/Layer) Convergence on CIFAR-10 ({device_name.upper()})", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    stem = f"graph_accuracy_L{acc_layer}_{device_name}"
    fig.savefig(os.path.join(model_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(model_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved {stem}.png & .pdf")


def generate_all_reports_for_backbone(backbone_key, all_records_by_prefix_layer, args, model_dir):
    """Generates Runtime tables, Energy tables (if collected), Head scaling plots, and L12 Accuracy CSV/plots."""
    b_info = BACKBONES[backbone_key]
    configs = [
        {"name": "Feature Extractor CPU (4 cores)", "model": b_info["cpu_model"], "threads": 4, "prefix": "CPU4"},
        {"name": "Feature Extractor TPU", "model": b_info["edgetpu_model"], "threads": 4, "prefix": "EdgeTPU"},
    ]

    # 1. Feature Extraction Tables at Layer 0
    for cfg in configs:
        l0_runs = all_records_by_prefix_layer[cfg["prefix"]].get(0, [])
        if l0_runs:
            export_feature_extractor_table(b_info, cfg, l0_runs, "runtime", "s", args.device, model_dir)
            if args.mode == "both":
                export_feature_extractor_table(b_info, cfg, l0_runs, "energy", "mJ", args.device, model_dir)

    # 2. Classification Head Scaling Stats & Graphs (0 -> 3 -> 6 -> 9 -> 12)
    scaling_summary = {}
    for cfg in configs:
        prefix = cfg["prefix"]
        scaling_summary[prefix] = {}
        for l in LAYERS_TO_TEST:
            runs = all_records_by_prefix_layer[prefix].get(l, [])
            if not runs:
                continue
            rt_vals = [r["runtime"]["train_header"] for r in runs]
            en_vals = [r["energy"]["train_header"] for r in runs]
            scaling_summary[prefix][str(l)] = {
                "runtime": {
                    "mean": statistics.mean(rt_vals),
                    "std": statistics.stdev(rt_vals) if len(rt_vals) > 1 else 0.0,
                    "raw": rt_vals,
                },
                "energy": {
                    "mean": statistics.mean(en_vals),
                    "std": statistics.stdev(en_vals) if len(en_vals) > 1 else 0.0,
                    "raw": en_vals,
                },
            }

    with open(os.path.join(model_dir, "summary_stats.json"), "w") as f:
        json.dump(scaling_summary, f, indent=2)

    plot_head_scaling_graphs(b_info, scaling_summary, "runtime", "s", args.device, model_dir, args.hidden_dim)
    if args.mode == "both":
        plot_head_scaling_graphs(b_info, scaling_summary, "energy", "mJ", args.device, model_dir, args.hidden_dim)

    # 3. Accuracy Tracking (Layer 12)
    acc_by_prefix = {
        cfg["prefix"]: all_records_by_prefix_layer[cfg["prefix"]].get(args.acc_layer, [])
        for cfg in configs
    }
    export_accuracy_results(b_info, acc_by_prefix, args.acc_layer, args.device, model_dir)


def run_check_setup(args):
    """Verifies scripts, local/Pi models, network reachability, and FNB58 meter without executing workloads."""
    print("=========================================================================")
    print("      CIFAR-10 V2 SETUP & PREFLIGHT CHECK")
    print("=========================================================================")
    for pkg in ("flwr", "numpy", "PIL", "hid", "pexpect", "matplotlib"):
        try:
            __import__(pkg)
            print(f"  [OK] Python package '{pkg}' installed")
        except ImportError:
            print(f"  [MISSING] Python package '{pkg}'")

    for b_key in args.models:
        b = BACKBONES[b_key]
        print(f"\nModel Backbone: {b['display_name']} (feature_dim={b['feature_dim']})")
        for label, m_path in [("CPU", b["cpu_model"]), ("EdgeTPU", b["edgetpu_model"])]:
            local_p = m_path.replace("~", PROJECT_ROOT).replace("~/research_project", PROJECT_ROOT)
            if local_p.startswith(os.path.join(PROJECT_ROOT, "research_project")):
                local_p = local_p.replace(os.path.join(PROJECT_ROOT, "research_project"), PROJECT_ROOT)
            exists_local = os.path.exists(os.path.join(PROJECT_ROOT, "edgetpu", "retrain-backprop", os.path.basename(m_path)))
            print(f"  [{ 'OK' if exists_local else 'MISSING' }] {label}: {os.path.basename(m_path)}")

    fnb_ok = check_fnb58_connected()
    print(f"\nFNB58 USB Power Meter (0x2E3C:0x5558): {'CONNECTED ✅' if fnb_ok else 'NOT PLUGGED (plug before running --mode both)'}")
    print(f"Configured Server Address: {args.server_address}")
    print(f"Configured Pi Host:        {args.pi_host}")
    print(f"Target Layers:             {LAYERS_TO_TEST} (hidden_dim={args.hidden_dim})")
    print(f"Accuracy Layer:            {args.acc_layer} (per-round tracking across {args.num_rounds} rounds)")
    print(f"Output Root Directory:     {args.output_dir}")
    print("=========================================================================")


def main():
    global LAYERS_TO_TEST
    parser = argparse.ArgumentParser(description="CIFAR-10 V2 Master Benchmark Suite (0..12 layers, 512 units, MobileNet & InceptionV3)")
    parser.add_argument("--models", nargs="+", default=["mobilenet", "inception_v3"], choices=list(BACKBONES.keys()),
                        help="Backbones to benchmark (default: mobilenet inception_v3)")
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS_TO_TEST),
                        help="Hidden layer counts to test (default: 0 3 6 9 12)")
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM,
                        help="Neurons per hidden layer (default: 512)")
    parser.add_argument("--acc_layer", type=int, default=12,
                        help="Hidden layer count to record per-round test accuracy (default: 12)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per configuration (default: 5)")
    parser.add_argument("--num_rounds", type=int, default=40,
                        help="Federated learning rounds (default: 40)")
    parser.add_argument("--local_epochs", type=int, default=10,
                        help="Local SGD epochs per round (default: 10)")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes (default: 10)")
    parser.add_argument("--device", default="pi5",
                        help="Device suffix for tables/plots (default: pi5)")
    parser.add_argument("--mode", default="both", choices=["both", "runtime"],
                        help="'both' captures Runtime + Energy via FNB58; 'runtime' skips FNB58")
    parser.add_argument("--server_address", default="auto",
                        help="Flower server IP:Port advertised to Pi (default: auto-detect Mac IP:8080)")
    parser.add_argument("--no_manage_server", action="store_true",
                        help="Connect to an already-running Flower server instead of starting one automatically")
    parser.add_argument("--pi_host", default=DEFAULT_PI_HOST, help="SSH user@ip for Raspberry Pi 5")
    parser.add_argument("--pi_pass", default=DEFAULT_PI_PASS, help="SSH password for Raspberry Pi 5")
    parser.add_argument("--pi_python", default=DEFAULT_PI_PYTHON, help="Python executable on Raspberry Pi 5")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Dataset dir relative to federated_learning on Pi")
    parser.add_argument("--cooldown", type=int, default=2, help="Cooldown seconds between runs")
    parser.add_argument("--ssh_timeout", type=int, default=2400, help="SSH per-run timeout in seconds (default: 2400s = 40m)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--fresh", action="store_true", help="Re-run even if cached result JSON exists in raw_logs")
    parser.add_argument("--replot_only", action="store_true", help="Regenerate CSV/TeX/PNG/PDF from existing raw_logs JSONs")
    parser.add_argument("--smoke_test", action="store_true", help="Run a single L12 trial to verify end-to-end execution")
    parser.add_argument("--check_setup", action="store_true", help="Run non-invasive preflight verification and exit")
    args = parser.parse_args()

    LAYERS_TO_TEST = sorted(args.layers)

    if args.server_address == "auto":
        pi_ip = args.pi_host.split("@")[-1]
        mac_ip = detect_local_ip_for_peer(pi_ip)
        args.server_address = f"{mac_ip}:8080"

    if args.check_setup:
        run_check_setup(args)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.replot_only and args.mode == "both" and not check_fnb58_connected():
        print("❌ FNB58 USB meter (0x2E3C:0x5558) is NOT connected to this Mac!", file=sys.stderr)
        print("   Please plug the FNB58 PC micro-USB cable into your Mac hub, or run with --mode runtime.", file=sys.stderr)
        sys.exit(1)

    server_proc, server_log_fp = None, None
    if not args.replot_only and not args.no_manage_server:
        server_proc, server_log_fp = start_managed_flower_server(args)

    try:
        clock_offset = 0.0 if args.replot_only else measure_clock_offset(args.pi_host, args.pi_pass)
        print(f"Clock Offset (Mac - Pi): {clock_offset:+.4f}s")

        if args.smoke_test:
            b_key = args.models[0]
            b_info = BACKBONES[b_key]
            model_dir = os.path.join(args.output_dir, b_key)
            raw_dir = os.path.join(model_dir, "raw_logs")
            os.makedirs(raw_dir, exist_ok=True)
            cfg = {"name": "Feature Extractor TPU", "model": b_info["edgetpu_model"], "threads": 4, "prefix": "EdgeTPU"}
            args.runs = 1
            args.fresh = True
            run_single_trial(b_key, cfg, num_layers=args.acc_layer, run_idx=1, args=args, raw_dir=raw_dir, clock_offset=clock_offset)
            return

        for b_key in args.models:
            b_info = BACKBONES[b_key]
            model_dir = os.path.join(args.output_dir, b_key)
            raw_dir = os.path.join(model_dir, "raw_logs")
            os.makedirs(raw_dir, exist_ok=True)

            print("\n" + "#" * 80)
            print(f"# BACKBONE: {b_info['display_name']} | Output -> {model_dir}")
            print("#" * 80)

            configs = [
                {"name": "Feature Extractor CPU (4 cores)", "model": b_info["cpu_model"], "threads": 4, "prefix": "CPU4"},
                {"name": "Feature Extractor TPU", "model": b_info["edgetpu_model"], "threads": 4, "prefix": "EdgeTPU"},
            ]

            all_records = {"CPU4": {}, "EdgeTPU": {}}
            for cfg in configs:
                prefix = cfg["prefix"]
                if not args.replot_only:
                    clock_offset = measure_clock_offset(args.pi_host, args.pi_pass)

                for num_layers in LAYERS_TO_TEST:
                    all_records[prefix][num_layers] = []
                    for r in range(1, args.runs + 1):
                        if args.replot_only:
                            res_p = os.path.join(raw_dir, f"{prefix}_L{num_layers}_run{r}_result.json")
                            if os.path.exists(res_p):
                                with open(res_p, "r") as f:
                                    all_records[prefix][num_layers].append(json.load(f))
                        else:
                            rec = run_single_trial(
                                b_key, cfg, num_layers=num_layers, run_idx=r,
                                args=args, raw_dir=raw_dir, clock_offset=clock_offset,
                            )
                            all_records[prefix][num_layers].append(rec)
                            if r < args.runs and args.cooldown > 0:
                                time.sleep(args.cooldown)

            generate_all_reports_for_backbone(b_key, all_records, args, model_dir)

        print("\n" + "=" * 85)
        print(f"🎉 ALL V2 BENCHMARKS & REPORTS READY IN: {args.output_dir}")
        print("=" * 85 + "\n")
    finally:
        if server_proc is not None:
            print("Stopping managed Flower server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except Exception:
                server_proc.kill()
            if server_log_fp:
                server_log_fp.close()


if __name__ == "__main__":
    main()
