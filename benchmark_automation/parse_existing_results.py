#!/usr/bin/env python3
"""
Log Parser & Table I Re-generator for Existing Benchmark Files

Reads historical or newly collected log files in flower_project/federated_learning/
(e.g., CPU1Runtime1..5, EdgeTPURuntime1..5), computes Mean ± Std, and outputs Table I.
Uses only Python standard library (statistics) - zero external dependencies required!
"""

import os
import sys
import glob
import re
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FL_DIR = os.path.join(PROJECT_ROOT, "flower_project", "federated_learning")

def parse_log_file(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    
    m_load = re.search(r'Time for Load Feature Extractor[^:]*:\s*([\d\.]+)', text)
    m_ext = re.search(r'Time for Extract Train Embeddings[^:]*:\s*([\d\.]+)', text)
    m_train = re.search(r'Time for\s+Train Classification Head[^:]*:\s*([\d\.]+)', text)

    load_time = float(m_load.group(1)) if m_load else 0.0
    ext_time = float(m_ext.group(1)) if m_ext else 0.0
    train_time = float(m_train.group(1)) if m_train else 0.0

    return {
        "load": load_time,
        "extract": ext_time,
        "offline_total": load_time + ext_time,
        "train": train_time,
        "runtime_total": train_time
    }

def process_device(device_tag="pi5"):
    approaches = [
        ("FL-ETPU", "EdgeTPU"),
        ("FL-CPU (1 core)", "CPU1"),
        ("FL-CPU (2 cores)", "CPU2"),
        ("FL-CPU (4 cores)", "CPU4")
    ]

    print(f"\n=========================================================================================")
    print(f"   TABLE I RESULTS COMPUTED AUTOMATICALLY FROM LOG FILES (Device: {device_tag.upper()})")
    print(f"=========================================================================================")
    col_fmt = "{:<18} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16}"
    print(col_fmt.format("Approach", "Load Base Model", "Extract Features", "Offline Total", "Train Header", "Runtime Total"))
    print("-" * 105)

    for approach_name, prefix in approaches:
        if device_tag == "pi3" or device_tag == "pi3b":
            pattern = os.path.join(FL_DIR, f"{prefix}Runtime*_pi3.txt")
        else:
            pattern = os.path.join(FL_DIR, f"{prefix}Runtime[1-5].txt")
        
        files = sorted(glob.glob(pattern))
        if not files:
            # Fallback
            files = sorted(glob.glob(os.path.join(FL_DIR, f"{prefix}Runtime*.txt")))
            files = [f for f in files if "_pi3" not in f]

        if not files:
            continue

        loads, extracts, offlines, trains, runtimes = [], [], [], [], []
        for fp in files[:5]:
            t = parse_log_file(fp)
            loads.append(t["load"])
            extracts.append(t["extract"])
            offlines.append(t["offline_total"])
            trains.append(t["train"])
            runtimes.append(t["runtime_total"])

        def calc_stat(data):
            if not data:
                return "N/A"
            mean_v = statistics.mean(data)
            std_v = statistics.stdev(data) if len(data) > 1 else 0.0
            return f"{mean_v:.2f} ± {std_v:.2f}"

        print(col_fmt.format(
            approach_name,
            calc_stat(loads),
            calc_stat(extracts),
            calc_stat(offlines),
            calc_stat(trains),
            calc_stat(runtimes)
        ))
    print("=========================================================================================\n")

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "pi5"
    process_device(device)
