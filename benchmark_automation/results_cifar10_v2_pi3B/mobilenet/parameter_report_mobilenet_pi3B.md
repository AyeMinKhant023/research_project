# CIFAR-10 V2 MobileNetV1 Parameter, Runtime & Energy Report (Baseline $L=0$, Raspberry Pi 3B)

This report summarizes the exact parameter counts, 40-round runtime, energy consumption (FNIRSI FNB58), and deterministic accuracy (`seed = 42`) recorded on **Raspberry Pi 3 Model B Rev 1.2** in `results_cifar10_v2_pi3B/mobilenet`.

## Experimental Configuration
- **Device:** Raspberry Pi 3 Model B Rev 1.2 + Coral USB Accelerator (USB 2.0)
- **Dataset:** CIFAR-10 (`num_classes = 10`, `2,000` local training samples, `250` held-out test samples)
- **Base Feature Extractor:** Quantized MobileNetV1 1.0 (`224×224`), truncated at `AvgPool_1a` (`feature_dim = 1024`)
  - CPU Model: `mobilenet_v1_1.0_224_quant_embedding_extractor.tflite`
  - EdgeTPU Model: `mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite`
- **Classification Head:** Linear Softmax Baseline ($L = 0$, `1024 → 10`)
- **Federated Training:** 40 rounds, 10 local epochs per round (`batch_size = 200`, `lr = 0.01`, `5` runs per setting, `seed = 42`)

---

## 1. Parameter Count Summary ($L = 0$)

| Hidden Layers ($L$) | Head Architecture | Base Extractor Params (Frozen) | Head Weights | Head Biases | Head Total Params (Trainable) | Head FP32 Size (MB) | Total Model Params (Base + Head) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `1024 → 10` | 3,196,032 | 10,240 | 10 | **10,250** | 0.04 MB | 3,206,282 |

---

## 2. Offline Extraction & Online Head Training Summary (5 Runs on Raspberry Pi 3B)

| Hardware Mode | Load Base (s) | Extract Features (s) | Train Head 40 Rds (s) | Extract Energy (mJ) | Train Head Energy (mJ) | Final Test Acc (Std) | Final Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU (4 cores)** | `0.16 ± 0.35` | `177.33 ± 8.68` | `40.42 ± 1.79` | `804,342.37 ± 14,153.32` | `111,433.51 ± 7,096.07` | **`0.6640`** (`0.0000`) | `0.9243` |
| **Coral EdgeTPU** | `3.54 ± 1.01` | **`29.89 ± 0.54`** | `38.07 ± 0.40` | **`68,698.95 ± 744.66`** | `100,123.57 ± 1,192.15` | **`0.6760`** (`0.0000`) | `0.9159` |
