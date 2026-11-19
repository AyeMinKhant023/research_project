#!/usr/bin/env python3
"""
2-Client (Pi 5 + MacBook), 80-20 Split, 5-Seed Accuracy Benchmark Suite (CIFAR-10 V2)
======================================================================================

Implements the professor's updated Section 6.5 accuracy protocol:
- 2 Active Federated Clients (--min_clients 2):
    * Client 1 (Raspberry Pi 5 at 172.20.10.4): client1_data (2,500 images: 2,000 train / 500 test, 80-20 split)
      running either FL-CPU (4 cores) or FL-ETPU (Coral USB Accelerator).
    * Client 2 (MacBook): client2_data (2,500 images: 2,000 train / 500 test, 80-20 split)
      running fl_client_Mac.py concurrently and aggregating with Client 1 via FedAvg every round.
- 5 Different Random Seeds across the 5 runs: SEEDS = [42, 43, 44, 45, 46]
    * Within each run i (1..5), FL-CPU and FL-ETPU share the exact same seed SEEDS[i-1].
    * Across the 5 runs, reports true Mean ± Std across 5 distinct seeds for BOTH L=0 (5 runs) and L=12 (5 runs).
- Hardware Embedding Cache:
    * Extracts all 2,500 embeddings once on real Pi 5 hardware (CPU4 & EdgeTPU) and MacBook, then
      applies each seed's deterministic 80-20 permutation across the 5 seeds.
"""

import argparse
import csv
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FL_DIR = os.path.join(PROJECT_ROOT, "flower_project", "federated_learning")
EDGETPU_MODELS_DIR = os.path.join(PROJECT_ROOT, "edgetpu", "retrain-backprop")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_cifar10_v2_benchmarks import (
    BACKBONES,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PI_PASS,
    DEFAULT_PI_PYTHON,
    HIDDEN_DIM,
    detect_local_ip_for_peer,
    parse_timings_from_log,
)

DEFAULT_ACC_5SEEDS_DIR = os.path.join(SCRIPT_DIR, "results_cifar10_v2_accuracy_5seeds")
SEEDS = [42, 43, 44, 45, 46]


def robust_parse_round_metrics(log_text: str):
    """Parses [ROUND_METRICS] JSON objects even when Flower ANSI logs appear on the same line."""
    records = []
    decoder = json.JSONDecoder()
    clean_text = re.sub(r"\x1b\[[0-9;]*m", "", log_text)
    for line in clean_text.splitlines():
        if "[ROUND_METRICS]" in line:
            idx = line.find("{")
            if idx != -1:
                try:
                    obj, _ = decoder.raw_decode(line[idx:])
                    if isinstance(obj, dict) and "round" in obj:
                        records.append(obj)
                except json.JSONDecodeError:
                    continue
    return records


def kill_port_8080():
    try:
        out = subprocess.check_output("lsof -ti:8080 || true", shell=True, text=True).strip()
        if out:
            for pid in out.splitlines():
                subprocess.call(["kill", "-9", pid.strip()])
            time.sleep(0.4)
    except Exception:
        pass


def start_flower_server_for_trial(num_rounds: int, min_clients: int, seed: int, hidden_dim: int, num_classes: int, log_path: str):
    """Starts a fresh fl_server.py instance for a single 2-client trial."""
    kill_port_8080()
    cmd = [
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
        "-u",
        os.path.join(FL_DIR, "fl_server.py"),
        "--feature_dim", "1024",
        "--num_classes", str(num_classes),
        "--min_clients", str(min_clients),
        "--num_rounds", str(num_rounds),
        "--hidden_dim", str(hidden_dim),
        "--server_address", "0.0.0.0:8080",
        "--seed", str(seed),
    ]
    log_fp = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
    for _ in range(30):
        if proc.poll() is not None:
            log_fp.close()
            raise RuntimeError(f"Flower server exited early ({proc.returncode}). Check {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", 8080), timeout=0.5):
                return proc, log_fp
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"Timed out waiting for Flower server on port 8080. Check {log_path}")


def ensure_mac_client2_cache(backbone_key: str, raw_dir: str):
    """Populates MacBook Client 2 embedding cache once using /Users/ayeminkhant/.pyenv/versions/3.9.16/bin/python3 (TF 2.15.1)."""
    b_info = BACKBONES[backbone_key]
    mac_model_path = os.path.join(EDGETPU_MODELS_DIR, os.path.basename(b_info["cpu_model"]))
    mac_data_dir = os.path.join(FL_DIR, "data", "cifar10", "client2_data")
    mac_cache_path = os.path.join(raw_dir, f"mac_client2_{backbone_key}_embeddings.npz")
    if os.path.exists(mac_cache_path):
        return mac_cache_path

    os.makedirs(raw_dir, exist_ok=True)
    tf_python = "/Users/ayeminkhant/.pyenv/versions/3.9.16/bin/python3"
    if not os.path.exists(tf_python):
        tf_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

    print(f"  -> [Mac Client 2] Pre-extracting 2,500 {b_info['short_name']} embeddings on MacBook ({mac_cache_path})...")
    code = (
        "import sys, os, time, numpy as np\n"
        f"sys.path.insert(0, {FL_DIR!r})\n"
        "import tensorflow as tf\n"
        "from fl_client_Mac import get_image_paths, extract_embeddings_tflite\n"
        f"paths, labels, _ = get_image_paths({mac_data_dir!r})\n"
        f"interp = tf.lite.Interpreter(model_path={mac_model_path!r}, num_threads=4)\n"
        "interp.allocate_tensors()\n"
        "t0 = time.time()\n"
        "embs = extract_embeddings_tflite(paths, interp)\n"
        f"tmp = {mac_cache_path!r} + '.tmp.npz'\n"
        "np.savez(tmp, embeddings=embs, labels=np.array(labels))\n"
        f"os.replace(tmp, {mac_cache_path!r})\n"
        "print(f'  -> [Mac Client 2] Extracted {len(embs)} embeddings (shape={embs.shape}) in {time.time()-t0:.2f}s')\n"
    )
    subprocess.run([tf_python, "-u", "-c", code], check=True)
    return mac_cache_path


def start_mac_second_client(backbone_key: str, num_layers: int, seed: int, args, raw_dir: str, log_path: str):
    """Launches Client 2 (fl_client_Mac.py) on the MacBook using client2_data (80-20 split)."""
    b_info = BACKBONES[backbone_key]
    mac_model_path = os.path.join(EDGETPU_MODELS_DIR, os.path.basename(b_info["cpu_model"]))
    mac_data_dir = os.path.join(FL_DIR, "data", "cifar10", "client2_data")
    mac_cache_path = ensure_mac_client2_cache(backbone_key, raw_dir)

    cmd = [
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
        "-u",
        os.path.join(FL_DIR, "fl_client_Mac.py"),
        "--embedding_extractor_path", mac_model_path,
        "--data_dir", mac_data_dir,
        "--num_classes", str(args.num_classes),
        "--server_address", "127.0.0.1:8080",
        "--num_threads", "4",
        "--num_head_layers", str(num_layers),
        "--hidden_dim", str(args.hidden_dim),
        "--local_epochs", str(args.local_epochs),
        "--seed", str(seed),
        "--val_percent", "0.0",
        "--test_percent", "0.2",
        "--embedding_cache", mac_cache_path,
    ]
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["OMP_NUM_THREADS"] = "4"
    log_fp = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env)
    return proc, log_fp


def run_two_client_trial(backbone_key: str, cfg: dict, num_layers: int, run_idx: int, seed: int, num_rounds: int, args, raw_dir: str):
    """Executes one 2-client (Pi 5 + MacBook) FL trial for a given seed and layer count."""
    import pexpect

    stem = f"{cfg['prefix']}_L{num_layers}_run{run_idx}_seed{seed}"
    result_json = os.path.join(raw_dir, f"{stem}_result.json")
    client_log_file = os.path.join(raw_dir, f"{stem}_pi_log.txt")
    mac_log_file = os.path.join(raw_dir, f"{stem}_mac_log.txt")
    server_log_file = os.path.join(raw_dir, f"{stem}_server.log")
    acc_json_file = os.path.join(raw_dir, f"{stem}_acc.json")

    if not args.fresh and os.path.exists(result_json):
        try:
            with open(result_json, "r") as f:
                cached = json.load(f)
            if (
                len(cached.get("round_metrics", [])) >= num_rounds
                and cached.get("seed") == seed
                and cached.get("min_clients") == args.min_clients
                and cached.get("split") == "80-20"
            ):
                print(f"⏭️  [RESUME] Skipping completed {backbone_key} | {stem}")
                return cached
        except Exception:
            pass

    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] {BACKBONES[backbone_key]['short_name']} | "
        f"{cfg['name']} | Layers: {num_layers} (w={args.hidden_dim}) | "
        f"Run {run_idx} (seed={seed}) | Clients={args.min_clients} (80-20 split)"
    )

    server_proc, server_fp = start_flower_server_for_trial(
        num_rounds=num_rounds,
        min_clients=args.min_clients,
        seed=seed,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        log_path=server_log_file,
    )
    mac_proc, mac_fp = None, None
    child = None
    try:
        if args.min_clients >= 2:
            mac_proc, mac_fp = start_mac_second_client(
                backbone_key=backbone_key,
                num_layers=num_layers,
                seed=seed,
                args=args,
                raw_dir=raw_dir,
                log_path=mac_log_file,
            )

        pi_cache_path = f"/tmp/pi5_client1_{backbone_key}_{cfg['prefix']}_embeddings.npz"
        pi_cmd = (
            f"OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 {args.pi_python} -u fl_client.py "
            f"--embedding_extractor_path {cfg['model']} "
            f"--data_dir {args.data_dir} "
            f"--num_classes {args.num_classes} "
            f"--server_address {args.server_address} "
            f"--num_threads {cfg['threads']} "
            f"--num_head_layers {num_layers} "
            f"--hidden_dim {args.hidden_dim} "
            f"--local_epochs {args.local_epochs} "
            f"--seed {seed} "
            f"--val_percent 0.0 "
            f"--test_percent 0.2 "
            f"--embedding_cache {pi_cache_path} "
            f"--track_accuracy"
        )

        start_t = time.time()
        with open(client_log_file, "w", encoding="utf-8", errors="replace", buffering=1) as log_fp:
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
                t_wait0 = time.time()
                while True:
                    if time.time() - t_wait0 > args.ssh_timeout:
                        raise TimeoutError(f"Pi 5 SSH timed out after {args.ssh_timeout}s")
                    r_idx = child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=5)
                    if r_idx == 0:
                        break
                    if mac_proc is not None and mac_proc.poll() is not None and mac_proc.returncode != 0:
                        raise RuntimeError(
                            f"Mac Client 2 exited early with code {mac_proc.returncode}. Check {mac_log_file}"
                        )
                    if server_proc is not None and server_proc.poll() is not None and server_proc.returncode != 0:
                        raise RuntimeError(
                            f"Flower server exited early with code {server_proc.returncode}. Check {server_log_file}"
                        )
            child.close()

        if mac_proc is not None:
            try:
                mac_proc.wait(timeout=30)
            except Exception:
                mac_proc.kill()

        try:
            server_proc.wait(timeout=15)
        except Exception:
            server_proc.terminate()

    finally:
        if child is not None:
            try:
                child.close()
            except Exception:
                pass
        if mac_proc is not None and mac_proc.poll() is None:
            mac_proc.kill()
        if mac_fp is not None:
            mac_fp.close()
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except Exception:
                server_proc.kill()
        if server_fp is not None:
            server_fp.close()

    with open(client_log_file, "r", encoding="utf-8", errors="replace") as f:
        ssh_output = f.read()

    runtime_data = parse_timings_from_log(ssh_output)
    round_metrics = robust_parse_round_metrics(ssh_output)
    if len(round_metrics) < num_rounds:
        raise RuntimeError(
            f"Captured only {len(round_metrics)}/{num_rounds} rounds for {stem}. Inspect {client_log_file} and {mac_log_file}"
        )

    with open(acc_json_file, "w") as f:
        json.dump(round_metrics, f, indent=2)

    wall = time.time() - start_t
    last = round_metrics[-1]
    print(
        f"   -> Done in {wall:.1f}s (TrainHead={runtime_data['train_header']:.2f}s) | "
        f"Rounds={len(round_metrics)} | R1 TestAcc={round_metrics[0]['test_acc']:.4f} -> "
        f"R{num_rounds} TestAcc={last['test_acc']:.4f} (TrainAcc={last['train_acc']:.4f}, TestLoss={last['test_loss']:.4f})"
    )

    record = {
        "backbone": backbone_key,
        "prefix": cfg["prefix"],
        "num_layers": num_layers,
        "hidden_dim": args.hidden_dim,
        "run_idx": run_idx,
        "seed": seed,
        "min_clients": args.min_clients,
        "split": "80-20",
        "train_samples_per_client": 2000,
        "test_samples_per_client": 500,
        "runtime": runtime_data,
        "round_metrics": round_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(result_json, "w") as f:
        json.dump(record, f, indent=2)
    return record


def export_sanity_summary(b_info, sanity_records, device_name, model_dir):
    """Exports 2-round x 2-run (same seed=42) deterministic comparison table and verifies 0 diff."""
    csv_path = os.path.join(model_dir, f"accuracy_2rounds_sanity_{device_name}_v2.csv")
    md_path = os.path.join(model_dir, f"accuracy_2rounds_sanity_{device_name}_v2.md")

    header = [
        "Processor",
        "Layers",
        "Round",
        "Run1 Test Acc",
        "Run2 Test Acc",
        "Test Acc Diff",
        "Run1 Test Loss",
        "Run2 Test Loss",
        "Run1 Train Acc",
        "Run2 Train Acc",
        "DeterministicMatch",
    ]
    rows = []
    all_exact = True

    for prefix in ["CPU4", "EdgeTPU"]:
        for num_layers in [0, 12]:
            runs = sanity_records.get(prefix, {}).get(num_layers, [])
            if len(runs) < 2:
                all_exact = False
                continue
            rm1 = runs[0].get("round_metrics", [])
            rm2 = runs[1].get("round_metrics", [])
            n_rounds = min(len(rm1), len(rm2))
            if n_rounds < 2:
                all_exact = False
            for idx in range(n_rounds):
                m1, m2 = rm1[idx], rm2[idx]
                diff_test_acc = abs(m1["test_acc"] - m2["test_acc"])
                diff_test_loss = abs(m1["test_loss"] - m2["test_loss"])
                diff_train_acc = abs(m1["train_acc"] - m2["train_acc"])
                match = (diff_test_acc == 0.0 and diff_test_loss == 0.0 and diff_train_acc == 0.0)
                if not match:
                    all_exact = False
                rows.append([
                    prefix,
                    f"L{num_layers}",
                    m1["round"],
                    f"{m1['test_acc']:.6f}",
                    f"{m2['test_acc']:.6f}",
                    f"{diff_test_acc:.6f}",
                    f"{m1['test_loss']:.6f}",
                    f"{m2['test_loss']:.6f}",
                    f"{m1['train_acc']:.6f}",
                    f"{m2['train_acc']:.6f}",
                    "YES" if match else "NO",
                ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    with open(md_path, "w") as f:
        f.write(f"# {b_info['display_name']} — 2-Client 2-Round Deterministic Sanity Check (`seed=42`, 80-20 split, `{device_name.upper()}`)\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
        f.write(f"\nAll Run1 vs Run2 pairs identical: **{'YES' if all_exact else 'NO'}**\n")

    print(f"✅ Saved {os.path.basename(csv_path)} & .md (all_exact={all_exact})")
    if not all_exact:
        raise RuntimeError(f"Deterministic sanity check FAILED for {b_info['short_name']}. Inspect {csv_path}")


def export_accuracy_results_v2(backbone_key, backbone_info, acc_runs_by_prefix, acc_layer, device_name, model_dir):
    """Exports per-round (1..40) 5-seed V2 accuracy CSVs and plots (both default and Overleaf-prefixed PDFs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "CPU4": {"color": "#1f77b4", "marker": "o", "label": f"Base: CPU 4-cores (L{acc_layer})", "tag": "CPU4"},
        "EdgeTPU": {"color": "#d95f02", "marker": "s", "label": f"Base: EdgeTPU (L{acc_layer})", "tag": "ETPU"},
    }

    agg_curves = {}
    for prefix, runs in acc_runs_by_prefix.items():
        valid_runs = [r["round_metrics"] for r in runs if r.get("round_metrics")]
        seeds_used = [r.get("seed", SEEDS[i]) for i, r in enumerate(runs) if r.get("round_metrics")]
        if not valid_runs:
            continue
        max_rounds = max(len(rm) for rm in valid_runs)
        csv_path = os.path.join(
            model_dir,
            f"accuracy_L{acc_layer}_rounds_{styles[prefix]['tag']}_{device_name}_v2.csv",
        )

        header = [
            "Round",
            "Test Acc Mean",
            "Test Acc Std",
            "Test Loss Mean",
            "Test Loss Std",
            "Train Acc Mean",
            "Train Acc Std",
            "Cum Train Time (s)",
        ]
        header.extend([f"Run{i+1} (seed={seeds_used[i]}) Test Acc" for i in range(len(valid_runs))])

        round_nums, test_acc_mean, test_acc_std, test_loss_mean, test_loss_std = [], [], [], [], []
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
            s_loss = statistics.stdev(t_losses) if len(t_losses) > 1 else 0.0
            m_tr = statistics.mean(tr_accs)
            s_tr = statistics.stdev(tr_accs) if len(tr_accs) > 1 else 0.0
            m_ct = statistics.mean(cum_ts)

            round_nums.append(rnd_num)
            test_acc_mean.append(m_acc * 100.0)
            test_acc_std.append(s_acc * 100.0)
            test_loss_mean.append(m_loss)
            test_loss_std.append(s_loss)

            row = [
                rnd_num,
                f"{m_acc:.4f}",
                f"{s_acc:.4f}",
                f"{m_loss:.4f}",
                f"{s_loss:.4f}",
                f"{m_tr:.4f}",
                f"{s_tr:.4f}",
                f"{m_ct:.2f}",
            ]
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
            "test_loss_std": test_loss_std,
            "num_runs": len(valid_runs),
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
        loss_s = curve["test_loss_std"]

        ax1.plot(rnds, acc_m, label=st["label"], color=st["color"], linewidth=2.0, marker=st["marker"], markevery=4, markersize=5)
        ax1.fill_between(rnds, [a - s for a, s in zip(acc_m, acc_s)], [a + s for a, s in zip(acc_m, acc_s)], color=st["color"], alpha=0.18)

        ax2.plot(rnds, loss_m, label=st["label"], color=st["color"], linewidth=2.0, marker=st["marker"], markevery=4, markersize=5)
        ax2.fill_between(rnds, [l - s for l, s in zip(loss_m, loss_s)], [l + s for l, s in zip(loss_m, loss_s)], color=st["color"], alpha=0.18)

    ax1.set_title(f"{backbone_info['short_name']}: Layer {acc_layer} Test Accuracy per Round (5 Seeds)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Federated Learning Round", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)", fontsize=9.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5)

    ax2.set_title(f"{backbone_info['short_name']}: Layer {acc_layer} Test Loss per Round (5 Seeds)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Federated Learning Round", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Test Loss", fontsize=9.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    n_runs_tag = max(c["num_runs"] for c in agg_curves.values())
    fig.suptitle(
        f"Layer-{acc_layer} ({'Linear Softmax' if acc_layer == 0 else '512 Neurons/Layer'}) 2-Client Convergence on CIFAR-10 ({device_name.upper()}, {n_runs_tag} seeds, 80-20 split)",
        fontsize=11.2,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    stem = f"graph_accuracy_L{acc_layer}_{device_name}_v2"
    png_path = os.path.join(model_dir, f"{stem}.png")
    pdf_path = os.path.join(model_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    prefixed_tag = "mobilenet" if backbone_key == "mobilenet" else "inception"
    prefixed_pdf = os.path.join(model_dir, f"{prefixed_tag}_{stem}.pdf")
    shutil.copy2(pdf_path, prefixed_pdf)
    print(f"📊 Saved {stem}.png, {stem}.pdf & {os.path.basename(prefixed_pdf)}")


def run_stage_sanity(args):
    """Stage A: 2-client 2-round x 2-run deterministic sanity check with same seed=42 for L=0 and L=12."""
    print("\n" + "=" * 80)
    print("STAGE A: 2-CLIENT 2-ROUND x 2-RUN DETERMINISTIC SANITY TEST (seed=42, L=0 & L=12)")
    print("=" * 80)

    for b_key in args.models:
        b_info = BACKBONES[b_key]
        model_dir = os.path.join(args.output_dir, b_key)
        raw_dir = os.path.join(model_dir, "raw_logs_sanity_2r_v2")
        os.makedirs(raw_dir, exist_ok=True)

        configs = [
            {"name": "Feature Extractor CPU (4 cores)", "model": b_info["cpu_model"], "threads": 4, "prefix": "CPU4"},
            {"name": "Feature Extractor TPU", "model": b_info["edgetpu_model"], "threads": 4, "prefix": "EdgeTPU"},
        ]
        sanity_records = {"CPU4": {}, "EdgeTPU": {}}

        for cfg in configs:
            prefix = cfg["prefix"]
            for num_layers in [0, 12]:
                sanity_records[prefix][num_layers] = []
                for r in range(1, 3):
                    last_err = None
                    for attempt in range(1, 4):
                        try:
                            rec = run_two_client_trial(
                                backbone_key=b_key,
                                cfg=cfg,
                                num_layers=num_layers,
                                run_idx=r,
                                seed=42,
                                num_rounds=2,
                                args=args,
                                raw_dir=raw_dir,
                            )
                            sanity_records[prefix][num_layers].append(rec)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            print(f"⚠️ Sanity attempt {attempt}/3 failed ({e}). Retrying in 5s...")
                            time.sleep(5)
                    if last_err is not None:
                        raise last_err

        export_sanity_summary(b_info, sanity_records, args.device, model_dir)


def run_stage_full_v2(args):
    """Stage B: 2-client 40-round V2 accuracy curves across 5 distinct seeds [42, 43, 44, 45, 46] for L=0 and L=12."""
    print("\n" + "=" * 80)
    print(f"STAGE B: 2-CLIENT 40-ROUND 5-SEED ACCURACY SUITE (seeds={SEEDS[:args.num_seeds]}, L=0 & L=12, 80-20 split)")
    print("=" * 80)

    for b_key in args.models:
        b_info = BACKBONES[b_key]
        model_dir = os.path.join(args.output_dir, b_key)
        raw_dir = os.path.join(model_dir, "raw_logs_acc_v2")
        os.makedirs(raw_dir, exist_ok=True)

        configs = [
            {"name": "Feature Extractor CPU (4 cores)", "model": b_info["cpu_model"], "threads": 4, "prefix": "CPU4"},
            {"name": "Feature Extractor TPU", "model": b_info["edgetpu_model"], "threads": 4, "prefix": "EdgeTPU"},
        ]

        for num_layers in args.layers:
            acc_by_prefix = {"CPU4": [], "EdgeTPU": []}
            for cfg in configs:
                prefix = cfg["prefix"]
                for r in range(1, args.num_seeds + 1):
                    seed = SEEDS[r - 1]
                    last_err = None
                    for attempt in range(1, 4):
                        try:
                            rec = run_two_client_trial(
                                backbone_key=b_key,
                                cfg=cfg,
                                num_layers=num_layers,
                                run_idx=r,
                                seed=seed,
                                num_rounds=args.num_rounds,
                                args=args,
                                raw_dir=raw_dir,
                            )
                            acc_by_prefix[prefix].append(rec)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            print(f"⚠️ Trial attempt {attempt}/3 failed ({e}). Retrying in 5s...")
                            time.sleep(5)
                    if last_err is not None:
                        raise last_err

            export_accuracy_results_v2(b_key, b_info, acc_by_prefix, num_layers, args.device, model_dir)


def main():
    parser = argparse.ArgumentParser(description="2-Client 80-20 Split 5-Seed Accuracy Suite (Pi 5 + MacBook)")
    parser.add_argument("--models", nargs="+", default=["mobilenet", "inception_v3"], choices=list(BACKBONES.keys()))
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 12])
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=40)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--min_clients", type=int, default=2)
    parser.add_argument("--device", default="pi5")
    parser.add_argument("--server_address", default="auto")
    parser.add_argument("--pi_host", default="ayeminkhant@172.20.10.4")
    parser.add_argument("--pi_pass", default=DEFAULT_PI_PASS)
    parser.add_argument("--pi_python", default=DEFAULT_PI_PYTHON)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--ssh_timeout", type=int, default=2400)
    parser.add_argument("--output_dir", default=DEFAULT_ACC_5SEEDS_DIR)
    parser.add_argument("--stage", default="v2", choices=["sanity", "v2", "all"])
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.server_address == "auto":
        pi_ip = args.pi_host.split("@")[-1]
        mac_ip = detect_local_ip_for_peer(pi_ip)
        args.server_address = f"{mac_ip}:8080"

    if args.stage in ("sanity", "all"):
        run_stage_sanity(args)
    if args.stage in ("v2", "all"):
        run_stage_full_v2(args)


if __name__ == "__main__":
    main()
