# CIFAR-10 Model Parameter & Runtime Report (0–4 Classification Head Layers)

This report summarizes the exact parameter counts for the **Base Feature Extractor** (frozen) and the **Classification Head** (trainable across 0–4 hidden layers), alongside the corresponding 40-round classification head training runtimes recorded on Raspberry Pi 5 in `results_cifar10`.

## Experimental Configuration
- **Dataset:** CIFAR-10 (`num_classes = 10`, `2,000` local training samples)
- **Base Feature Extractor:** Quantized MobileNetV1 1.0 (`224×224`), truncated at `AvgPool_1a` (`feature_dim = 1024`)
  - CPU Model: `mobilenet_v1_1.0_224_quant_embedding_extractor.tflite`
  - EdgeTPU Model: `mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite`
- **Classification Head:** Configurable MLP (`hidden_dim = 256` with ReLU activations + Softmax output)
- **Federated Training:** 40 rounds, 10 local epochs per round (`batch_size = 200`, `lr = 0.01`)

---

## 1. Summary Table (0–4 Hidden Layers on CIFAR-10)

| Hidden Layers | Head Architecture | Base Feature Extractor Params (Frozen) | Classification Head Params (Trainable) | Total Model Params (Base + Head) | Head Train Runtime — 40 Rounds (Mean ± Std) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | `1024 → 10` | 3,196,032 | **10,250** | 3,206,282 | 4.74 ± 0.04 s |
| **1** | `1024 → 256 → 10` | 3,196,032 | **264,970** | 3,461,002 | 21.75 ± 0.15 s |
| **2** | `1024 → 256 → 256 → 10` | 3,196,032 | **330,762** | 3,526,794 | 28.37 ± 0.52 s |
| **3** | `1024 → 256 → 256 → 256 → 10` | 3,196,032 | **396,554** | 3,592,586 | 34.33 ± 0.21 s |
| **4** | `1024 → 256 → 256 → 256 → 256 → 10` | 3,196,032 | **462,346** | 3,658,378 | 39.75 ± 0.25 s |

---

## 2. Parameter Calculation Details

### Base Feature Extractor (`3,196,032` parameters)
- Consists of `Conv2d_0` (standard 3×3 convolution) followed by 13 depthwise-separable convolution blocks (`Conv2d_1_depthwise/pointwise` through `Conv2d_13_depthwise/pointwise`) with BatchNorm folded into the convolution weights and biases:
  - **Weights:** `3,188,896`
  - **Biases:** `7,136`
  - **Total Frozen Parameters:** `3,196,032`
- Outputs a `1024`-dimensional embedding vector per image (`MobilenetV1/Logits/AvgPool_1a/AvgPool`, shape `[1, 1, 1, 1024]`).

### Classification Head (`MLPClassificationHead`)
- **0 Hidden Layers (`1024 → 10`):**
  - Output layer: $(1024 \times 10) + 10 = 10,240 + 10 =$ **10,250** parameters
- **1 Hidden Layer (`1024 → 256 → 10`):**
  - Hidden layer 1: $(1024 \times 256) + 256 = 262,144 + 256 = 262,400$
  - Output layer: $(256 \times 10) + 10 = 2,560 + 10 = 2,570$
  - Total: $262,400 + 2,570 =$ **264,970** parameters
- **2 Hidden Layers (`1024 → 256 → 256 → 10`):**
  - Hidden layer 1: $262,400$
  - Hidden layer 2: $(256 \times 256) + 256 = 65,536 + 256 = 65,792$
  - Output layer: $2,570$
  - Total: $262,400 + 65,792 + 2,570 =$ **330,762** parameters
- **3 Hidden Layers (`1024 → 256 → 256 → 256 → 10`):**
  - Adds a second `256 → 256` hidden layer ($+65,792$):
  - Total: $330,762 + 65,792 =$ **396,554** parameters
- **4 Hidden Layers (`1024 → 256 → 256 → 256 → 256 → 10`):**
  - Adds a third `256 → 256` hidden layer ($+65,792$):
  - Total: $396,554 + 65,792 =$ **462,346** parameters

---

## Reference Note (5-Class Dataset Comparison)
For experiments using 5 output classes (e.g., Flower dataset, `num_classes = 5`) with the same `1024`-dim MobileNetV1 extractor (`3,196,032` base parameters), the trainable classification head parameter counts for 0–4 hidden layers are:
- **0 Layers:** `5,125`
- **1 Layer:** `263,685`
- **2 Layers:** `329,477`
- **3 Layers:** `395,269`
- **4 Layers:** `461,061`
