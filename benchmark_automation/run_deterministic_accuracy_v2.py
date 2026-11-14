#!/usr/bin/env python3
"""
Deterministic Accuracy Verification (2-Round Sanity) & V2 40-Round Curves (Raspberry Pi 5)
==========================================================================================

Executes the professor's deterministic seed validation protocol with fixed seed=42:
  Stage A (Sanity Test):
    - 2 FL rounds x 2 runs (Run 1 vs Run 2) for L=0 and L=12 on CPU4 and EdgeTPU
    - Verifies exact 0.0000 difference across runs (and exports accuracy_2rounds_sanity_pi5_v2.{csv,md})
  Stage B (V2 Accuracy Files, 40 rounds):
    - L=0  (0 hidden layers):  5 runs x 40 rounds (CPU4 & EdgeTPU)
    - L=12 (12 hidden layers): 1 run  x 40 rounds (CPU4 & EdgeTPU)
    - Saves v2 deliverables inside results_cifar10_v2_pi5/{mobilenet,inception_v3}/:
        * accuracy_L0_rounds_{CPU4,ETPU}_pi5_v2.csv
        * graph_accuracy_L0_pi5_v2.{png,pdf}
        * accuracy_L12_rounds_{CPU4,ETPU}_pi5_v2.csv
        * graph_accuracy_L12_pi5_v2.{png,pdf}
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_cifar10_v2_benchmarks as rcb
from run_cifar10_v2_benchmarks import (
    BACKBONES,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PI_PASS,
    DEFAULT_PI_PYTHON,
    HIDDEN_DIM,
    detect_local_ip_for_peer,
    measure_clock_offset,
    run_single_trial,
    start_managed_flower_server,
)


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


# Ensure run_single_trial uses ANSI-safe JSON decoding when extracting round metrics
rcb.parse_round_metrics_from_log = robust_parse_round_metrics


def make_trial_args(args, num_rounds: int, acc_layer: int, runs: int, output_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        models=args.models,
        hidden_dim=args.hidden_dim,
        local_epochs=args.local_epochs,
        num_classes=args.num_classes,
        device=args.device,
        mode="runtime",
        server_address=args.server_address,
        pi_host=args.pi_host,
        pi_pass=args.pi_pass,
        pi_python=args.pi_python,
        data_dir=args.data_dir,
        cooldown=args.cooldown,
        ssh_timeout=args.ssh_timeout,
        output_dir=output_dir,
        fresh=args.fresh,
        num_rounds=num_rounds,
        runs=runs,
        acc_layer=acc_layer,
    )


def stop_server(server_proc, server_log_fp):
    if server_proc is not None:
        print("Stopping managed Flower server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()
        if server_log_fp:
            server_log_fp.close()


def refresh_cached_round_metrics(cfg, num_layers, run_idx, trial_args, raw_dir):
    """Reparses *_log.txt if an earlier cached result missed rounds due to inline Flower ANSI logs."""
    stem = f"{cfg['prefix']}_L{num_layers}_run{run_idx}"
    result_json = os.path.join(raw_dir, f"{stem}_result.json")
    log_file = os.path.join(raw_dir, f"{stem}_log.txt")
    acc_file = os.path.join(raw_dir, f"{stem}_acc.json")
    if os.path.exists(result_json) and os.path.exists(log_file):
        try:
            with open(result_json, "r") as f:
                cached = json.load(f)
            if len(cached.get("round_metrics", [])) < trial_args.num_rounds:
                with open(log_file, "r", encoding="utf-8", errors="replace") as lf:
                    reparsed = robust_parse_round_metrics(lf.read())
                if len(reparsed) >= trial_args.num_rounds:
                    cached["round_metrics"] = reparsed
                    with open(result_json, "w") as rf:
                        json.dump(cached, rf, indent=2)
                    with open(acc_file, "w") as af:
                        json.dump(reparsed, af, indent=2)
                    print(f"🔧 Reparsed {len(reparsed)}/{trial_args.num_rounds} rounds cleanly from {stem}_log.txt")
        except Exception:
            pass


def run_trial_with_retries(backbone_key, cfg, num_layers, run_idx, trial_args, raw_dir, clock_offset):
    refresh_cached_round_metrics(cfg, num_layers, run_idx, trial_args, raw_dir)
    last_err = None
    for attempt in range(1, 4):
        try:
            return run_single_trial(
                backbone_key,
                cfg,
                num_layers=num_layers,
                run_idx=run_idx,
                args=trial_args,
                raw_dir=raw_dir,
                clock_offset=clock_offset,
            )
        except Exception as e:
            last_err = e
            print(f"⚠️ Trial attempt {attempt}/3 failed ({e}). Waiting 10s before retry...")
            time.sleep(10)
    raise last_err


def export_sanity_summary(b_info, sanity_records, device_name, model_dir):
    """Exports 2-round x 2-run deterministic comparison table and verifies 0 diff."""
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
        f.write(f"# {b_info['display_name']} — 2-Round Deterministic Sanity Check (`seed=42`, `{device_name.upper()}`)\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
        f.write(f"\nAll Run1 vs Run2 pairs identical: **{'YES' if all_exact else 'NO'}**\n")

    print(f"✅ Saved {os.path.basename(csv_path)} & .md (all_exact={all_exact})")
    if not all_exact:
        raise RuntimeError(
            f"Deterministic sanity check FAILED for {b_info['short_name']}: "
            f"Run 1 and Run 2 did not match identically. Inspect {csv_path}"
        )


def export_accuracy_results_v2(backbone_info, acc_runs_by_prefix, acc_layer, device_name, model_dir):
    """Exports per-round (1..40) deterministic V2 accuracy CSVs and plots with _v2 suffix."""
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
        if not valid_runs:
            continue
        max_rounds = max(len(rm) for rm in valid_runs)
        csv_path = os.path.join(
            model_dir,
            f"accuracy_L{acc_layer}_rounds_{styles[prefix]['tag']}_{device_name}_v2.csv",
        )

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

        ax1.plot(rnds, acc_m, label=st["label"], color=st["color"], linewidth=2.0, marker=st["marker"], markevery=4, markersize=5)
        low = [a - s for a, s in zip(acc_m, acc_s)]
        high = [a + s for a, s in zip(acc_m, acc_s)]
        ax1.fill_between(rnds, low, high, color=st["color"], alpha=0.15)

        ax2.plot(rnds, loss_m, label=st["label"], color=st["color"], linewidth=2.0, marker=st["marker"], markevery=4, markersize=5)

    ax1.set_title(f"{backbone_info['short_name']}: Layer {acc_layer} Test Accuracy per Round (Seed=42)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Federated Learning Round", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)", fontsize=9.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5)

    ax2.set_title(f"{backbone_info['short_name']}: Layer {acc_layer} Test Loss per Round (Seed=42)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Federated Learning Round", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Test Loss", fontsize=9.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    n_runs_tag = max(c["num_runs"] for c in agg_curves.values())
    fig.suptitle(
        f"Layer-{acc_layer} ({'Linear Softmax' if acc_layer == 0 else '512 Neurons/Layer'}) Deterministic Convergence on CIFAR-10 ({device_name.upper()}, {n_runs_tag} run{'s' if n_runs_tag > 1 else ''}, V2)",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    stem = f"graph_accuracy_L{acc_layer}_{device_name}_v2"
    fig.savefig(os.path.join(model_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(model_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved {stem}.png & .pdf")


def run_stage_sanity(args):
    """Stage A: 2 rounds x 2 runs for L=0 and L=12 across CPU4 and EdgeTPU."""
    print("\n" + "=" * 80)
    print("STAGE A: 2-ROUND x 2-RUN DETERMINISTIC SANITY TEST (L=0 and L=12)")
    print("=" * 80)

    sanity_log_root = os.path.join(args.output_dir, "_sanity_2rounds_server")
    os.makedirs(sanity_log_root, exist_ok=True)
    srv_args = make_trial_args(args, num_rounds=2, acc_layer=0, runs=2, output_dir=sanity_log_root)
    server_proc, server_log_fp = start_managed_flower_server(srv_args)

    try:
        clock_offset = measure_clock_offset(args.pi_host, args.pi_pass)
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
                    t_args = make_trial_args(args, num_rounds=2, acc_layer=num_layers, runs=2, output_dir=model_dir)
                    for r in range(1, 3):
                        rec = run_trial_with_retries(b_key, cfg, num_layers, r, t_args, raw_dir, clock_offset)
                        sanity_records[prefix][num_layers].append(rec)
                        if r < 2 and args.cooldown > 0:
                            time.sleep(args.cooldown)

            export_sanity_summary(b_info, sanity_records, args.device, model_dir)
    finally:
        stop_server(server_proc, server_log_fp)


def run_stage_full_v2(args):
    """Stage B: 40-round V2 accuracy runs (L=0 for 5 runs, L=12 for 1 run)."""
    print("\n" + "=" * 80)
    print("STAGE B: 40-ROUND DETERMINISTIC V2 ACCURACY CURVES (L=0 x 5 runs, L=12 x 1 run)")
    print("=" * 80)

    v2_srv_log_root = os.path.join(args.output_dir, "_acc_v2_40rounds_server")
    os.makedirs(v2_srv_log_root, exist_ok=True)
    srv_args = make_trial_args(args, num_rounds=args.num_rounds, acc_layer=0, runs=args.l0_runs, output_dir=v2_srv_log_root)
    server_proc, server_log_fp = start_managed_flower_server(srv_args)

    layer_run_counts = {0: args.l0_runs, 12: args.l12_runs}

    try:
        clock_offset = measure_clock_offset(args.pi_host, args.pi_pass)
        for b_key in args.models:
            b_info = BACKBONES[b_key]
            model_dir = os.path.join(args.output_dir, b_key)
            raw_dir = os.path.join(model_dir, "raw_logs_acc_v2")
            os.makedirs(raw_dir, exist_ok=True)

            configs = [
                {"name": "Feature Extractor CPU (4 cores)", "model": b_info["cpu_model"], "threads": 4, "prefix": "CPU4"},
                {"name": "Feature Extractor TPU", "model": b_info["edgetpu_model"], "threads": 4, "prefix": "EdgeTPU"},
            ]

            for num_layers, num_runs in layer_run_counts.items():
                acc_by_prefix = {"CPU4": [], "EdgeTPU": []}
                t_args = make_trial_args(args, num_rounds=args.num_rounds, acc_layer=num_layers, runs=num_runs, output_dir=model_dir)
                for cfg in configs:
                    prefix = cfg["prefix"]
                    for r in range(1, num_runs + 1):
                        rec = run_trial_with_retries(b_key, cfg, num_layers, r, t_args, raw_dir, clock_offset)
                        acc_by_prefix[prefix].append(rec)
                        if r < num_runs and args.cooldown > 0:
                            time.sleep(args.cooldown)

                # Verify zero variance when multiple runs exist
                for prefix, runs in acc_by_prefix.items():
                    if len(runs) > 1:
                        for rnd_i in range(len(runs[0]["round_metrics"])):
                            accs = [r["round_metrics"][rnd_i]["test_acc"] for r in runs]
                            if max(accs) - min(accs) != 0.0:
                                raise RuntimeError(
                                    f"Non-zero variance detected in {b_key} {prefix} L{num_layers} round {rnd_i+1}: {accs}"
                                )

                export_accuracy_results_v2(b_info, acc_by_prefix, num_layers, args.device, model_dir)
    finally:
        stop_server(server_proc, server_log_fp)


def main():
    parser = argparse.ArgumentParser(description="Deterministic Seed Accuracy Verification & V2 Generator (Pi 5)")
    parser.add_argument("--models", nargs="+", default=["mobilenet", "inception_v3"], choices=list(BACKBONES.keys()))
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=40)
    parser.add_argument("--l0_runs", type=int, default=5)
    parser.add_argument("--l12_runs", type=int, default=1)
    parser.add_argument("--device", default="pi5")
    parser.add_argument("--server_address", default="auto")
    parser.add_argument("--pi_host", default="ayeminkhant@172.20.10.4")
    parser.add_argument("--pi_pass", default=DEFAULT_PI_PASS)
    parser.add_argument("--pi_python", default=DEFAULT_PI_PYTHON)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--cooldown", type=int, default=2)
    parser.add_argument("--ssh_timeout", type=int, default=2400)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", default="all", choices=["sanity", "v2", "all"])
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
