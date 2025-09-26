import flwr as fl
import numpy as np
import os
import time
import psutil
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from softmax_regression import SoftmaxRegression
import contextlib
from typing import Dict, List, Tuple, Optional
from memory_profiler import profile

def log_phase_marker(phase_name, timestamp=None):
    """Log phase markers to the power data file"""
    import os
    import time

    if timestamp is None:
        timestamp = time.time()

    # Path to phases.txt in the target folder
    base_dir = os.path.dirname(__file__)  # directory of fl_client.py
    phase_file_path = os.path.join(base_dir, "..", "fnirsi-usb-power-data-logger", "phases.txt")
    phase_file_path = os.path.abspath(phase_file_path)  # make it absolute

    # Ensure directory exists
    os.makedirs(os.path.dirname(phase_file_path), exist_ok=True)

    # Check if this is the first phase marker (START_LOAD_MODEL) to overwrite the file
    if phase_name == "START_LOAD_MODEL":
        mode = "w"  # Overwrite the file for the first marker
    else:
        mode = "a"  # Append for all subsequent markers

    # Append to the file
    with open(phase_file_path, mode) as f:
        f.write(f"#phase {timestamp:.3f} {phase_name}\n")

    print(f"[PHASE MARKER] {phase_name} at timestamp {timestamp:.3f} (saved to {phase_file_path})")


@contextlib.contextmanager
def test_image(path):
    """Returns opened test image."""
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            yield image


def get_image_paths(data_dir):
    """Walks through data_dir and returns list of image paths and label map."""
    classes = None
    image_paths = []
    labels = []

    class_idx = 0
    for root, dirs, files in os.walk(data_dir):
        if root == data_dir:
            classes = dirs
        else:
            assert classes[class_idx] in root
            print('Reading dir: %s, which has %d images' % (root, len(files)))
            for img_name in files:
                image_paths.append(os.path.join(root, img_name))
                labels.append(class_idx)
            class_idx += 1
    
    return image_paths, labels, dict(zip(range(class_idx), classes))


def shuffle_and_split(image_paths, labels, val_percent=0.1, test_percent=0.1):
    """Shuffles and splits data into train, validation, and test sets."""
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    perm = np.random.permutation(image_paths.shape[0])
    image_paths = image_paths[perm]
    labels = labels[perm]

    num_total = image_paths.shape[0]
    num_val = int(num_total * val_percent)
    num_test = int(num_total * test_percent)
    num_train = num_total - num_val - num_test

    train_and_val_dataset = {}
    train_and_val_dataset['data_train'] = image_paths[0:num_train]
    train_and_val_dataset['labels_train'] = labels[0:num_train]
    train_and_val_dataset['data_val'] = image_paths[num_train:num_train + num_val]
    train_and_val_dataset['labels_val'] = labels[num_train:num_train + num_val]
    
    test_dataset = {}
    test_dataset['data_test'] = image_paths[num_train + num_val:]
    test_dataset['labels_test'] = labels[num_train + num_val:]
    
    return train_and_val_dataset, test_dataset

# Function to measure CPU and RAM usage
import psutil
import tracemalloc
import time
import threading

def log_resource_usage(label, func, *args, **kwargs):
    process = psutil.Process()
    cpu_percentages = []

    # Monitor CPU in background
    # def monitor_cpu():
    #     while not stop_event.is_set():
    #         cpu = psutil.cpu_percent(percpu=True)
    #         cpu_percentages.append(cpu)
    #         time.sleep(0.1)  # sample every 100ms

    stop_event = threading.Event()
    # monitor_thread = threading.Thread(target=monitor_cpu)
    # monitor_thread.start()

    # tracemalloc.start() # Memory measurement with tracemalloc # start
    mem_before = process.memory_info().rss # Memory measurement with psutil # before

    start_time = time.time()
    result = func(*args, **kwargs)
    mem_now = process.memory_info().rss # Memory measurement with psutil # now
    end_time = time.time()

    stop_event.set()
    # monitor_thread.join()
    mem_after = process.memory_info().rss # Memory measurement with psutil # after
    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop() # Memory measurement with tracemalloc # stop

    # Calculate average CPU per core
    # cpu_array = list(zip(*cpu_percentages))  # transpose
    # avg_cpu_per_core = [sum(core)/len(core) for core in cpu_array]
    # ram_usage_tracemalloc = peak / (1024 * 1024) # in MB
    ram_usage_psutil = (mem_after - mem_before) / (1024 * 1024)  # in MB
    ram_usage_psutil_absolute = mem_now / (1024 * 1024)  # in MB

    print(f"\n[RESOURCE] Usage for {label}")
    print(f"  Time: {end_time - start_time:.2f} sec")
    # print(f"  Peak RAM Usage (tracemalloc): {ram_usage_tracemalloc:.2f} MB")
    print(f"  Peak RAM Usage (psutil): {ram_usage_psutil:.2f} MB")
    print(f"  Absolute Peak RAM Usage (psutil): {ram_usage_psutil_absolute:.2f} MB")
    # print(f"  Avg CPU Usage Per Core: {[round(c, 1) for c in avg_cpu_per_core]}")
    
    return result


##########################################################################################
# # With Edge TPU Interpreter #
# def extract_embeddings(image_paths, interpreter):
#     """Uses model to process images as embeddings."""
#     input_size = common.input_size(interpreter)
#     feature_dim = classify.num_classes(interpreter)
#     embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
    
#     for idx, path in enumerate(image_paths): 
#         with test_image(path) as img:
#             common.set_input(interpreter, img.resize(input_size, Image.NEAREST))
#             interpreter.invoke()
#             embeddings[idx, :] = classify.get_scores(interpreter)

#     return embeddings

##########################################################################################

# With tflite Interpreter #
def extract_embeddings(image_paths, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_size = (input_shape[2], input_shape[1])  # width, height
    feature_dim = output_details[0]['shape'][-1]
    embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
    for idx, path in enumerate(image_paths):
        with test_image(path) as img:
            img_resized = img.resize(input_size, Image.NEAREST)
            img_array = np.array(img_resized).astype(np.uint8)
            img_array = np.expand_dims(img_array, axis=0)
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            embeddings[idx, :] = output.squeeze()
    return embeddings
##########################################################################################

# # For ResNet

# # Replace the extract_embeddings function in your fl_client.py with this:

# def extract_embeddings(image_paths, interpreter):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     input_shape = input_details[0]['shape']
#     input_size = (input_shape[2], input_shape[1])  # width, height
#     feature_dim = output_details[0]['shape'][-1]
    
#     # Check the expected input data type
#     input_dtype = input_details[0]['dtype']
#     print(f"Model expects input type: {input_dtype}")
    
#     embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
    
#     for idx, path in enumerate(image_paths):
#         with test_image(path) as img:
#             img_resized = img.resize(input_size, Image.NEAREST)
#             img_array = np.array(img_resized)
            
#             # Handle different input types
#             if input_dtype == np.float32:
#                 # For models expecting float32 (like ResNet)
#                 img_array = img_array.astype(np.float32) / 255.0  # Normalize to 0-1
#                 # Apply ImageNet normalization for ResNet (ensure float32)
#                 mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#                 std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#                 img_array = (img_array - mean) / std
#                 img_array = img_array.astype(np.float32)  # Ensure float32
#             elif input_dtype == np.uint8:
#                 # For quantized models expecting uint8 (like your old MobileNet)
#                 img_array = img_array.astype(np.uint8)
#             else:
#                 # Fallback: try float32 normalized
#                 img_array = img_array.astype(np.float32) / 255.0
            
#             img_array = np.expand_dims(img_array, axis=0)
            
#             interpreter.set_tensor(input_details[0]['index'], img_array)
#             interpreter.invoke()
#             output = interpreter.get_tensor(output_details[0]['index'])
#             embeddings[idx, :] = output.squeeze()
    
#     return embeddings
##########################################################################################

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, embedding_extractor_path: str, data_dir: str, num_classes: int):
        self.embedding_extractor_path = embedding_extractor_path
        self.data_dir = data_dir
        self.num_classes = num_classes

        # Log start of model loading
        log_phase_marker("START_LOAD_MODEL")

        ##########################################################################################
        # Initialize the backbone (embedding extractor) - this stays frozen

        # # For Edge TPU Interpreter #
        # from pycoral.utils.edgetpu import make_interpreter
        # start = time.time() #

        # def load_interpreter(model_path):
        #     interpreter = make_interpreter(model_path, device=':0')
        #     interpreter.allocate_tensors()
        #     return interpreter

        # self.interpreter = log_resource_usage(
        #     "Usage for Load Feature Extractor (EdgeTPU Interpreter)", load_interpreter, embedding_extractor_path)

        # end = time.time() #
        # print(f"[RUNTIME] Time for Load Feature Extractor (EdgeTPU Interpreter): {end - start:.2f} seconds") #

        ##########################################################################################

        # With tflite Interpreter #
        import tensorflow as tf
        start = time.time() #

        def load_tflite_interpreter(model_path):
            interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)  # The place to change threads: num_threads=1
            interpreter.allocate_tensors()
            return interpreter
        
        self.interpreter = log_resource_usage(
            "Usage for Load Feature Extractor (TFLite Interpreter)", load_tflite_interpreter, embedding_extractor_path)
        
        end = time.time() #
        print(f"[RUNTIME] Time for Load Feature Extractor (TFLite Interpreter): {end - start:.2f} seconds") #

        ##########################################################################################

        # Log end of model loading
        log_phase_marker("END_LOAD_MODEL")

        # Load and preprocess data
        self.load_data()
        
        # Initialize head (softmax regression) - this is what gets trained
        self.feature_dim = self.train_embeddings.shape[1]
        self.head = SoftmaxRegression(
            self.feature_dim, 
            self.num_classes, 
            weight_scale=5e-2, 
            reg=0.0
        )
        
        print(f"Client initialized with:")
        print(f"- Feature dimension: {self.feature_dim}")
        print(f"- Number of classes: {self.num_classes}")
        print(f"- Training samples: {len(self.train_embeddings)}")
        # print(f"- Validation samples: {len(self.val_embeddings)}")
    
    # @profile
    def load_data(self):
        """Load and preprocess local data."""
        print("Loading local data...")
        image_paths, labels, label_map = get_image_paths(self.data_dir)
        train_and_val_dataset, test_dataset = shuffle_and_split(image_paths, labels)

        # Log start of feature extraction
        log_phase_marker("START_EXTRACT_FEATURES")
        
        ##########################################################################################
        # Extract embeddings using the backbone (frozen feature extractor)
        # Training embeddings
        print("Extracting training embeddings...")
        start = time.time() #
        self.train_embeddings = log_resource_usage(
            "Usage for Extract Train Embeddings (Perform)", extract_embeddings, 
            train_and_val_dataset['data_train'], self.interpreter)
        end = time.time() #
        print(f"[RUNTIME] Time for Extract Train Embeddings (Perform): {end - start:.2f} seconds") #
        self.train_labels = train_and_val_dataset['labels_train']

        ##########################################################################################
        
        # # Validation embeddings
        # print("Extracting validation embeddings...")
        # start = time.time() #
        # self.val_embeddings = log_resource_usage(
        #     "Usage for Extract Validation Embeddings (Perform)", extract_embeddings,
        #     train_and_val_dataset['data_val'], self.interpreter)
        # end = time.time() #
        # print(f"[RUNTIME] Time for Extract Validation Embeddings (Perform): {end - start:.2f} seconds") #
        # self.val_labels = train_and_val_dataset['labels_val']

        ##########################################################################################

        # Log end of feature extraction
        log_phase_marker("END_EXTRACT_FEATURES")

        # Prepare dataset for training
        self.dataset = {
            'data_train': self.train_embeddings,
            'labels_train': self.train_labels,
            # 'data_val': self.val_embeddings,
            # 'labels_val': self.val_labels
            'data_val': np.array([]),  # Empty validation data
            'labels_val': np.array([])  # Empty validation labels
        }
    
    def get_parameters(self, config):
        """Return current head parameters."""
        # Return only the head parameters (weights and biases)
        return [self.head.W.flatten(), self.head.b.flatten()]
    
    def set_parameters(self, parameters):
        """Set head parameters received from server."""
        # Reshape parameters back to original shape
        W_flat, b_flat = parameters
        self.head.W = W_flat.reshape(self.feature_dim, self.num_classes)
        self.head.b = b_flat.reshape(self.num_classes)
    
    # @profile
    def fit(self, parameters, config):
        """Train the head locally using local data."""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Local training parameters
        learning_rate = config.get("learning_rate", 1e-2)
        batch_size = 200
        # batch_size = config.get("batch_size", 64)
        num_iter = 23 #config.get("num_iter", 2)  # Reduced for federated settingß

        print(f"NUM ITER={num_iter}")
        print(f"Starting local training with lr={learning_rate}, batch_size={batch_size}, num_iter={num_iter}")

        # Log start of head training
        log_phase_marker("START_TRAIN_HEAD")
        
        ##########################################################################################
        # Train the head (only the head parameters are updated)
        start = time.time() #
        log_resource_usage(
            "Usage for Train Classification Head", self.head.train_with_sgd,
            self.dataset, num_iter, learning_rate, batch_size=batch_size)
        end = time.time() #
        print(f"[RUNTIME] Time for  Train Classification Head: {end - start:.2f} seconds") #

        ##########################################################################################

        # Log end of head training
        log_phase_marker("END_TRAIN_HEAD")

        # Return updated parameters and training metrics
        return self.get_parameters(config), len(self.train_embeddings), {}
    
    ## Evaluate for validation
    # def evaluate(self, parameters, config):
    #     """Evaluate the head on local validation data."""
    #     # Set parameters from server
    #     self.set_parameters(parameters)
        
    #     # Make predictions on validation data
    #     predictions = self.head.forward(self.val_embeddings)
    #     accuracy = np.mean(np.argmax(predictions, axis=1) == self.val_labels)
        
    #     print(f"Local validation accuracy: {accuracy:.4f}")
        
    #     return float(accuracy), len(self.val_embeddings), {"accuracy": float(accuracy)}


def main():

    # Log overall start
    log_phase_marker("START_FL_CLIENT")

    global_start_time = time.time()

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_extractor_path", required=True, 
                       help="Path to embedding extractor tflite model")
    parser.add_argument("--data_dir", required=True, 
                       help="Directory containing local training data")
    parser.add_argument("--num_classes", type=int, required=True,
                       help="Number of classes in the dataset")
    parser.add_argument("--server_address", default="localhost:8080",
                       help="Server address (default: localhost:8080)")
    
    args = parser.parse_args()
    
    # Create federated client
    client = FederatedClient(
        args.embedding_extractor_path, 
        args.data_dir, 
        args.num_classes
    )

    # Log start of federated learning
    log_phase_marker("START_FEDERATED_LEARNING")
    
    # Start federated learning
    print(f"Starting federated learning client, connecting to {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

    # Log end of federated learning
    log_phase_marker("END_FEDERATED_LEARNING")

    global_end_time = time.time()

    # Log end
    log_phase_marker("END_FL_CLIENT")

    total_runtime = global_end_time - global_start_time
    print(f"[RUNTIME] Total Runtime (Full Client Run): {total_runtime:.2f} seconds")

    # Append it to the power log file
    with open("mydata.txt", "a") as f:
        f.write(f"#runtime {int(global_end_time)} {total_runtime:.2f}\n")

if __name__ == "__main__":
    main()