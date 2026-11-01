# Automated Benchmarking & Head Layer Scaling Suite

This directory contains the **automated experimental benchmarking and evaluation suite** for Federated Learning on Edge Accelerators (FL-ETPU vs. CPU baselines).

---

## 📁 Directory Structure

```text
benchmark_automation/
├── run_benchmarks.py              # Main Table I automated runner (4 configs × 5 runs)
├── parse_existing_results.py       # Instant Table I re-generator from existing logs
├── run_head_layer_benchmarks.py    # Multi-layer Head Scaling (0, 1, 2, 3, 4 layers × 5 runs)
├── plot_head_scaling.py            # Graph 1 generator (Layer scaling vs. runtime)
├── README.md                      # Documentation & experiment guide
└── results_head_layers/           # Generated CSVs, JSON data, and high-res plots
```

---

## 🔬 Classification Head Scaling Experiment (Professor's Agenda)

### 1. What was implemented:
- **Baseline (0 hidden layers):** $1024 \to 5$ (SoftmaxRegression)
- **1 Hidden Layer:** $1024 \to 256 (\text{ReLU}) \to 5$
- **2 Hidden Layers:** $1024 \to 256 (\text{ReLU}) \to 256 (\text{ReLU}) \to 5$
- **3 Hidden Layers:** $1024 \to 256 \to 256 \to 256 \to 5$
- **4 Hidden Layers:** $1024 \to 256 \to 256 \to 256 \to 256 \to 5$

All weights and biases update via vectorized backpropagation (SGD) with support for Flower FedAvg parameter exchange.

---

## 🚀 Running the Experiments

### Step 1: Copy updated files to your Raspberry Pi
From your Mac terminal:
```bash
scp /Users/ayeminkhant/research_project/research_project/flower_project/federated_learning/softmax_regression.py ayeminkhant@172.20.10.2:~/research_project/flower_project/federated_learning/
scp /Users/ayeminkhant/research_project/research_project/flower_project/federated_learning/fl_client.py ayeminkhant@172.20.10.2:~/research_project/flower_project/federated_learning/
scp /Users/ayeminkhant/research_project/research_project/flower_project/federated_learning/fl_server.py ayeminkhant@172.20.10.2:~/research_project/flower_project/federated_learning/
scp -r /Users/ayeminkhant/research_project/research_project/benchmark_automation ayeminkhant@172.20.10.2:~/research_project/
```

### Step 2: Start the Flower Server on Mac
```bash
cd ~/research_project/research_project/flower_project/federated_learning
source ../../.venv/bin/activate
python fl_server.py --feature_dim 1024 --num_classes 5 --min_clients 1 --num_rounds 1
```

### Step 3: Run the Layer Scaling Benchmark on Raspberry Pi
```bash
cd ~/research_project
python3 benchmark_automation/run_head_layer_benchmarks.py --device pi5 --runs 5 --server_address 172.20.10.3:8080
```
*(This automatically runs 0, 1, 2, 3, and 4 layers across all 4 configurations $\times$ 5 runs, logs each run, and computes $\mu \pm \sigma$).*

### Step 4: Plot Graph 1 (Publication High-Res PNG + PDF)
On the Pi (or Mac):
```bash
python3 benchmark_automation/plot_head_scaling.py --device pi5
```
This generates:
- `results_head_layers/graph1_head_scaling.png` (300 DPI high resolution)
- `results_head_layers/graph1_head_scaling.pdf` (vector graphic for LaTeX paper)
