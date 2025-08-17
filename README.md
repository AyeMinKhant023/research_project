# research_project

## Raspberry Pi Setup Guide
1. Follow the official Coral setup instructions:
https://coral.ai/docs/accelerator/get-started/#runtime-on-linux.
Then run the installation script provided here:
https://github.com/AyeMinKhant023/research_project/blob/main/PyEnv_Install.sh

2. Try installing Flower, i.e.,`pip install "flwr[simulation]"`
On Raspberry Pi, this may fail due to dependency issues.
If it fails, install a stable version without simulation:
`pip install flwr==1.15.2`.
Then manually install only the dependencies you need for simulation.

3. Install TensorFlow Lite Runtime: `pip install tflite-runtime`
This may conflict with flatbuffers on Raspberry Pi (sometimes installs a broken version)
Fix the issue manually:
`pip install --force-reinstall flatbuffers==25.1.24`.
After that, you may need to re-install TensorFlow
`pip install --force-reinstall tensorflow==2.14.0`.

4. For a known working combination of packages, see:
https://github.com/AyeMinKhant023/research_project/blob/main/requirements.txt

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
