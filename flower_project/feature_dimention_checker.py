import os
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import classify

model_path = os.path.expanduser('~/research_project/edgetpu/retrain-backprop/mobilenet_v1_1.0_224_quant_embedding_extractor.tflite')
interpreter = make_interpreter(model_path, device=':0')
interpreter.allocate_tensors()
feature_dim = classify.num_classes(interpreter)
print(f"Feature dimension: {feature_dim}")