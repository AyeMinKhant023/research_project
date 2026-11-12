# CIFAR-10 V2 MobileNetV1 Parameter, Runtime & Energy Report (0, 3, 6, 9, 12 Hidden Layers)

This report summarizes the exact parameter counts for the **MobileNetV1 Base Feature Extractor** (frozen) and the **Classification Head** (trainable across `0, 3, 6, 9, 12` hidden layers with `512` neurons per layer + ReLU), alongside the corresponding 40-round classification head training runtimes and energy consumption recorded on Raspberry Pi 5 in `results_cifar10_v2_pi5/mobilenet`.

## Experimental Configuration
- **Dataset:** CIFAR-10 (`num_classes = 10`, `2,000` local training samples, `250` test samples)
- **Base Feature Extractor:** Quantized MobileNetV1 1.0 (`224×224`), truncated at `AvgPool_1a` (`feature_dim = 1024`)
  - CPU Model: `mobilenet_v1_1.0_224_quant_embedding_extractor.tflite`
  - EdgeTPU Model: `mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite`
- **Classification Head:** Configurable MLP (`hidden_dim = 512` with ReLU activations + Softmax output)
- **Federated Training:** 40 rounds, 10 local epochs per round (`batch_size = 200`, `lr = 0.01`, `5` runs per setting)

---

## 1. Parameter Count Summary Table (`0 → 3 → 6 → 9 → 12` Hidden Layers)

| Hidden Layers ($L$) | Head Architecture | Base Extractor Params (Frozen) | Head Weights | Head Biases | Head Total Params (Trainable) | Head FP32 Size (MB) | Total Model Params (Base + Head) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `1024 → 10` | 3,196,032 | 10,240 | 10 | **10,250** | 0.04 MB | 3,206,282 |
| **3** | `1024 → [512×3] → 10` | 3,196,032 | 1,053,696 | 1,546 | **1,055,242** | 4.03 MB | 4,251,274 |
| **6** | `1024 → [512×6] → 10` | 3,196,032 | 1,840,128 | 3,082 | **1,843,210** | 7.03 MB | 5,039,242 |
| **9** | `1024 → [512×9] → 10` | 3,196,032 | 2,626,560 | 4,618 | **2,631,178** | 10.04 MB | 5,827,210 |
| **12** | `1024 → [512×12] → 10` | 3,196,032 | 3,412,992 | 6,154 | **3,419,146** | 13.04 MB | 6,615,178 |

---

## 2. Combined Parameter, Runtime & Energy Table (40 FL Rounds on Raspberry Pi 5)

| Hidden Layers ($L$) | Trainable Head Params | Head Runtime — CPU4 Base (Mean ± Std) | Head Runtime — EdgeTPU Base (Mean ± Std) | Head Energy — CPU4 Base (Mean ± Std) | Head Energy — EdgeTPU Base (Mean ± Std) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **10,250** | 4.95 ± 0.12 s | 4.55 ± 0.03 s | 29,578.00 ± 1,247.08 mJ | 24,220.68 ± 431.60 mJ |
| **3** | **1,055,242** | 154.99 ± 0.29 s | 154.61 ± 0.41 s | 915,079.48 ± 8,636.16 mJ | 1,003,042.11 ± 6,034.90 mJ |
| **6** | **1,843,210** | 284.99 ± 0.25 s | 283.92 ± 0.27 s | 1,700,690.94 ± 11,443.43 mJ | 1,846,207.19 ± 10,060.03 mJ |
| **9** | **2,631,178** | 414.34 ± 0.81 s | 412.48 ± 0.19 s | 2,513,279.10 ± 15,220.07 mJ | 2,694,932.90 ± 9,536.22 mJ |
| **12** | **3,419,146** | 542.11 ± 2.08 s | 498.19 ± 89.02 s | 3,241,133.25 ± 82,737.28 mJ | 3,290,080.83 ± 265,846.47 mJ |

---

## 3. Layer-by-Layer Parameter Calculation Details

### A. Base Feature Extractor (`3,196,032` parameters, Frozen)
- Consists of `Conv2d_0` (standard $3 \times 3$ convolution) followed by 13 depthwise-separable convolution blocks (`Conv2d_1_depthwise/pointwise` through `Conv2d_13_depthwise/pointwise`) with BatchNorm folded into the convolution weights and biases:
  - **Weights:** `3,188,896`
  - **Biases:** `7,136`
  - **Total Frozen Parameters:** `3,196,032`
- Outputs a `1024`-dimensional embedding vector per image (`MobilenetV1/Logits/AvgPool_1a/AvgPool`, shape `[1, 1, 1, 1024]`).

### B. Classification Head (`MLPClassificationHead`, `hidden_dim = 512`, `num_classes = 10`)
For $L > 0$ hidden layers with width $H = 512$, input dimension $D = 1024$, and output classes $C = 10$:
$$\text{Params}(L) = \underbrace{(1024 \times 512 + 512)}_{\text{First hidden layer } (524,800)} + \underbrace{(L - 1) \times (512 \times 512 + 512)}_{\text{Subsequent hidden layers } ((L-1) \times 262,656)} + \underbrace{(512 \times 10 + 10)}_{\text{Output layer } (5,130)}$$
Each increment of $+3$ hidden layers ($3 \times 512$) adds exactly $3 \times 262,656 = \mathbf{787,968}$ trainable parameters ($786,432$ weights + $1,536$ biases):

- **0 Hidden Layers (`1024 → 10`):**
  - Output layer: $(1024 \times 10) + 10 = 10,240 + 10 =$ **10,250** parameters
- **3 Hidden Layers (`1024 → 512 → 512 → 512 → 10`):**
  - Hidden layer 1 (`1024 → 512`): $(1024 \times 512) + 512 = 524,288 + 512 = 524,800$
  - Hidden layers 2–3 ($2 \times$ `512 → 512`): $2 \times (262,144 + 512) = 525,312$
  - Output layer (`512 → 10`): $(512 \times 10) + 10 = 5,120 + 10 = 5,130$
  - Total: $524,800 + 525,312 + 5,130 =$ **1,055,242** parameters
- **6 Hidden Layers (`1024 → [512 × 6] → 10`):**
  - Adds 3 more `512 → 512` layers ($+787,968$):
  - Total: $1,055,242 + 787,968 =$ **1,843,210** parameters
- **9 Hidden Layers (`1024 → [512 × 9] → 10`):**
  - Adds 3 more `512 → 512` layers ($+787,968$):
  - Total: $1,843,210 + 787,968 =$ **2,631,178** parameters
- **12 Hidden Layers (`1024 → [512 × 12] → 10`):**
  - Adds 3 more `512 → 512` layers ($+787,968$):
  - Total: $2,631,178 + 787,968 =$ **3,419,146** parameters

