# CIFAR-10 V2 InceptionV3 Parameter, Runtime & Energy Report (0, 3, 6, 9, 12 Hidden Layers)

This report summarizes the exact parameter counts for the **InceptionV3 Base Feature Extractor** (frozen) and the **Classification Head** (trainable across `0, 3, 6, 9, 12` hidden layers with `512` neurons per layer + ReLU), alongside the corresponding 40-round classification head training runtimes and energy consumption recorded on Raspberry Pi 5 in `results_cifar10_v2_pi5/inception_v3`.

## Experimental Configuration
- **Dataset:** CIFAR-10 (`num_classes = 10`, `2,000` local training samples, `250` test samples)
- **Base Feature Extractor:** Quantized InceptionV3 (`299×299`), truncated at `AvgPool` (`feature_dim = 2048`)
  - CPU Model: `inception_v3_avg_pool.tflite`
  - EdgeTPU Model: `inception_v3_avg_pool_edgetpu.tflite`
- **Classification Head:** Configurable MLP (`hidden_dim = 512` with ReLU activations + Softmax output)
- **Federated Training:** 40 rounds, 10 local epochs per round (`batch_size = 200`, `lr = 0.01`, `5` runs per setting)

---

## 1. Parameter Count Summary Table (`0 → 3 → 6 → 9 → 12` Hidden Layers)

| Hidden Layers ($L$) | Head Architecture | Base Extractor Params (Frozen) | Head Weights | Head Biases | Head Total Params (Trainable) | Head FP32 Size (MB) | Total Model Params (Base + Head) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `2048 → 10` | 21,768,354 | 20,480 | 10 | **20,490** | 0.08 MB | 21,788,844 |
| **3** | `2048 → [512×3] → 10` | 21,768,354 | 1,577,984 | 1,546 | **1,579,530** | 6.03 MB | 23,347,884 |
| **6** | `2048 → [512×6] → 10` | 21,768,354 | 2,364,416 | 3,082 | **2,367,498** | 9.03 MB | 24,135,852 |
| **9** | `2048 → [512×9] → 10` | 21,768,354 | 3,150,848 | 4,618 | **3,155,466** | 12.04 MB | 24,923,820 |
| **12** | `2048 → [512×12] → 10` | 21,768,354 | 3,937,280 | 6,154 | **3,943,434** | 15.04 MB | 25,711,788 |

---

## 2. Combined Parameter, Runtime & Energy Table (40 FL Rounds on Raspberry Pi 5)

| Hidden Layers ($L$) | Trainable Head Params | Head Runtime — CPU4 Base (Mean ± Std) | Head Runtime — EdgeTPU Base (Mean ± Std) | Head Energy — CPU4 Base (Mean ± Std) | Head Energy — EdgeTPU Base (Mean ± Std) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **20,490** | 8.44 ± 0.10 s | 8.51 ± 0.12 s | 49,483.88 ± 653.41 mJ | 44,669.28 ± 3,357.26 mJ |
| **3** | **1,579,530** | 213.83 ± 0.23 s | 214.06 ± 0.27 s | 1,352,734.89 ± 10,117.80 mJ | 1,334,850.07 ± 3,886.50 mJ |
| **6** | **2,367,498** | 341.75 ± 0.61 s | 346.29 ± 2.57 s | 2,157,390.20 ± 9,898.43 mJ | 2,295,103.82 ± 69,077.39 mJ |
| **9** | **3,155,466** | 469.59 ± 0.39 s | 471.21 ± 2.02 s | 2,965,812.46 ± 2,127.22 mJ | 3,075,514.38 ± 59,083.58 mJ |
| **12** | **3,943,434** | 597.52 ± 0.28 s | 597.98 ± 2.29 s | 3,797,158.00 ± 7,852.99 mJ | 3,846,135.76 ± 76,038.79 mJ |

---

## 3. Layer-by-Layer Parameter Calculation Details

### A. Base Feature Extractor (`21,768,354` parameters, Frozen)
- Consists of the quantized InceptionV3 convolutional stem and Inception modules truncated at the final global average pooling layer (`AvgPool`):
  - **Weights:** `21,751,136`
  - **Biases:** `17,218`
  - **Total Frozen Parameters:** `21,768,354`
- Outputs a `2048`-dimensional embedding vector per image.

### B. Classification Head (`MLPClassificationHead`, `hidden_dim = 512`, `num_classes = 10`)
For $L > 0$ hidden layers with width $H = 512$, input dimension $D = 2048$, and output classes $C = 10$:
$$\text{Params}(L) = \underbrace{(2048 \times 512 + 512)}_{\text{First hidden layer } (1,049,088)} + \underbrace{(L - 1) \times (512 \times 512 + 512)}_{\text{Subsequent hidden layers } ((L-1) \times 262,656)} + \underbrace{(512 \times 10 + 10)}_{\text{Output layer } (5,130)}$$
Each increment of $+3$ hidden layers ($3 \times 512$) adds exactly $3 \times 262,656 = \mathbf{787,968}$ trainable parameters ($786,432$ weights + $1,536$ biases):

- **0 Hidden Layers (`2048 → 10`):**
  - Output layer: $(2048 \times 10) + 10 = 20,480 + 10 =$ **20,490** parameters
- **3 Hidden Layers (`2048 → 512 → 512 → 512 → 10`):**
  - Hidden layer 1 (`2048 → 512`): $(2048 \times 512) + 512 = 1,048,576 + 512 = 1,049,088$
  - Hidden layers 2–3 ($2 \times$ `512 → 512`): $2 \times (262,144 + 512) = 525,312$
  - Output layer (`512 → 10`): $(512 \times 10) + 10 = 5,120 + 10 = 5,130$
  - Total: $1,049,088 + 525,312 + 5,130 =$ **1,579,530** parameters
- **6 Hidden Layers (`2048 → [512 × 6] → 10`):**
  - Adds 3 more `512 → 512` layers ($+787,968$):
  - Total: $1,579,530 + 787,968 =$ **2,367,498** parameters
- **9 Hidden Layers (`2048 → [512 × 9] → 10`):**
  - Adds 3 more `512 → 512` layers ($+787,968$):
  - Total: $2,367,498 + 787,968 =$ **3,155,466** parameters
- **12 Hidden Layers (`2048 → [512 × 12] → 10`):**
  - Adds 3 more `512 → 512` layers ($+787,968$):
  - Total: $3,155,466 + 787,968 =$ **3,943,434** parameters

