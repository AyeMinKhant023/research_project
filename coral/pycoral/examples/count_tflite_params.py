import tensorflow as tf
import numpy as np

def count_tflite_parameters(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    total_params = 0
    tensor_details = interpreter.get_tensor_details()

    for tensor in tensor_details:
        # Skip intermediate tensors or inputs/outputs
        if "weights" in tensor['name'] or "bias" in tensor['name'] or tensor['name'].endswith('kernel') or tensor['name'].endswith('bias'):
            shape = tensor['shape']
            if np.prod(shape) > 0:
                total_params += np.prod(shape)

    print(f"Total parameters in {tflite_model_path}: {total_params:,}")

# Example usage
# TODO: replace cov2d_11_relu6.tflite with your own TFLite model
count_tflite_parameters("extractor1.tflite")
