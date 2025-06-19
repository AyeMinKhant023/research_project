# research_project

## Using `visualize.py` to Generate and Export HTML Files

### 1. Navigate to the `visualize.py` file location
```bash
cd ~/research_project/edgetpu/retrain-backprop
```

### 2. Generate HTML Files
```bash
python3 visualize.py mobilenet_v1_1.0_224_quant_embedding_extractor.tflite
cpu_model.html
python3 visualize.py mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite
tpu_model.html
```

### 3. Export HTML Files to Local Computer (if using Raspberry Pi via SSH)
On your **local computer**:
```bash
scp username@hostname:~/research_project/edgetpu/retrain-backprop/cpu_model.html .
scp username@hostname:~/research_project/edgetpu/retrain-backprop/tpu_model.html .
```
> Replace `username` and `hostname` with your Raspberry Pi's SSH username and IP
address.

### 4. View HTML Files
On your **local computer**:
```bash
open cpu_model.html
open tpu_model.html
```
> If you're on Windows, use `start` instead of `open`.
