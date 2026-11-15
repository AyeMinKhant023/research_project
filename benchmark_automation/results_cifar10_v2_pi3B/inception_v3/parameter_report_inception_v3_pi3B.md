# CIFAR-10 V2 InceptionV3 Parameter, Runtime & Energy Report (Baseline $L=0$, Raspberry Pi 3B)

This report summarizes the exact parameter counts, 40-round runtime, energy consumption (FNIRSI FNB58), and deterministic accuracy (`seed = 42`) recorded on **Raspberry Pi 3 Model B Rev 1.2** in `results_cifar10_v2_pi3B/inception_v3`.

## Experimental Configuration
- **Device:** Raspberry Pi 3 Model B Rev 1.2 + Coral USB Accelerator (USB 2.0)
- **Dataset:** CIFAR-10 (`num_classes = 10`, `2,000` local training samples, `250` held-out test samples)
- **Base Feature Extractor:** Quantized InceptionV3 (`299×299`), truncated at `AvgPool` (`feature_dim = 2048`)
  - CPU Model: `inception_v3_avg_pool.tflite`
  - EdgeTPU Model: `inception_v3_avg_pool_edgetpu.tflite`
- **Classification Head:** Linear Softmax Baseline ($L = 0$, `2048 → 10`)
- **Federated Training:** 40 rounds, 10 local epochs per round (`batch_size = 200`, `lr = 0.01`, `5` runs per setting, `seed = 42`)

---

## 1. Parameter Count Summary ($L = 0$)

| Hidden Layers ($L$) | Head Architecture | Base Extractor Params (Frozen) | Head Weights | Head Biases | Head Total Params (Trainable) | Head FP32 Size (MB) | Total Model Params (Base + Head) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `2048 → 10` | 21,768,354 | 20,480 | 10 | **20,490** | 0.08 MB | 21,788,844 |

---

## 2. Offline Extraction & Online Head Training Summary (5 Runs on Raspberry Pi 3B)

| Hardware Mode | Load Base (s) | Extract Features (s) | Train Head 40 Rds (s) | Extract Energy (mJ) | Train Head Energy (mJ) | Final Test Acc (Std) | Final Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU (4 cores)** | `0.43 ± 0.93` | `2,185.66 ± 137.62` | `71.33 ± 1.90` | `7,706,281.71 ± 179,879.36` | `189,810.79 ± 5,518.22` | **`0.7640`** (`0.0000`) | `0.6125` |
| **Coral EdgeTPU** | `4.06 ± 2.04` | **`969.78 ± 7.95`** | `72.53 ± 1.05` | **`2,058,024.92 ± 37,002.63`** | `199,613.63 ± 3,081.36` | **`0.7600`** (`0.0000`) | `0.6030` |
