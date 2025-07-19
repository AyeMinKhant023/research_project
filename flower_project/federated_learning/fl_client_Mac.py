import flwr as fl
import numpy as np
import os
import time
from PIL import Image
import tensorflow as tf
from softmax_regression import SoftmaxRegression
import contextlib
from typing import Dict, List, Tuple, Optional


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


def extract_embeddings_tflite(image_paths, interpreter, input_size):
    """Uses TensorFlow Lite model to process images as embeddings."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Debug output shape
    print(f"Input details: {input_details[0]}")
    print(f"Output details: {output_details[0]}")
    
    # Get the feature dimension from output shape
    output_shape = output_details[0]['shape']
    print(f"Output shape: {output_shape}")
    
    # Handle different output shapes
    if len(output_shape) == 2:  # [batch_size, features]
        feature_dim = output_shape[1]
    elif len(output_shape) == 1:  # [features] (no batch dimension)
        feature_dim = output_shape[0]
    elif len(output_shape) == 4:  # [batch_size, height, width, features] - typical CNN output
        feature_dim = output_shape[3]
    else:
        # For any other shape, take the last dimension as features
        feature_dim = output_shape[-1]
    
    print(f"Feature dimension: {feature_dim}")
    embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
    
    # Check if model expects quantized input (UINT8)
    input_dtype = input_details[0]['dtype']
    is_quantized = input_dtype == np.uint8
    
    # Get quantization parameters if quantized
    if is_quantized:
        input_scale = input_details[0]['quantization_parameters']['scales'][0]
        input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
        print(f"Model expects quantized input (UINT8). Scale: {input_scale}, Zero point: {input_zero_point}")
    else:
        print(f"Model expects float input ({input_dtype})")
    
    for idx, path in enumerate(image_paths):
        with test_image(path) as img:
            # Resize image to input size
            img_resized = img.resize(input_size, Image.NEAREST)
            
            # Convert to numpy array
            img_array = np.array(img_resized)
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            
            # Handle RGB images - ensure 3 channels
            if img_array.shape[-1] == 3:
                pass  # Already RGB
            elif img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            
            # Prepare input based on model type
            if is_quantized:
                # For quantized models, keep values as UINT8 (0-255)
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            else:
                # For float models, normalize to [0, 1]
                img_array = img_array.astype(np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], img_array)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            # print(f"Raw output shape: {output_data.shape}")
            
            # Handle quantized output if needed
            if output_details[0]['dtype'] == np.uint8:
                # Dequantize output
                output_scale = output_details[0]['quantization_parameters']['scales'][0]
                output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            # Handle different output shapes
            if len(output_data.shape) == 2:  # [batch_size, features]
                embeddings[idx, :] = output_data[0]
            elif len(output_data.shape) == 1:  # [features] (no batch dimension)
                embeddings[idx, :] = output_data
            elif len(output_data.shape) == 4:  # [batch_size, height, width, features]
                # For 4D output like [1, 1, 1, 1024], squeeze and take the features
                embeddings[idx, :] = output_data.squeeze()
            else:
                # Flatten if needed
                embeddings[idx, :] = output_data.flatten()
            
            if idx == 0:  # Print debug info for first image only
                print(f"First embedding shape: {embeddings[idx, :].shape}")
                print(f"First few embedding values: {embeddings[idx, :5]}")

    return embeddings


def extract_embeddings_saved_model(image_paths, model, input_size):
    """Uses TensorFlow SavedModel to process images as embeddings."""
    embeddings_list = []
    
    for idx, path in enumerate(image_paths):
        with test_image(path) as img:
            # Resize image to input size
            img_resized = img.resize(input_size, Image.NEAREST)
            
            # Convert to numpy array and normalize
            img_array = np.array(img_resized).astype(np.float32)
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            
            # Handle RGB images - ensure 3 channels
            if img_array.shape[-1] == 3:
                pass  # Already RGB
            elif img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            
            # Normalize to [0, 1] if needed
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Convert to tensor
            img_tensor = tf.convert_to_tensor(img_array)
            
            # Run inference
            output = model(img_tensor)
            embeddings_list.append(output.numpy()[0])
    
    return np.array(embeddings_list)


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, embedding_extractor_path: str, data_dir: str, num_classes: int, 
                 input_size: Tuple[int, int] = (224, 224)):
        self.embedding_extractor_path = embedding_extractor_path
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Initialize the backbone (embedding extractor) - this stays frozen
        self.load_model()
        
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
        print(f"- Validation samples: {len(self.val_embeddings)}")
    
    def load_model(self):
        """Load the embedding extractor model."""
        # Expand user path and convert to absolute path
        expanded_path = os.path.expanduser(self.embedding_extractor_path)
        self.embedding_extractor_path = os.path.abspath(expanded_path)
        
        print(f"Loading model from: {self.embedding_extractor_path}")
        
        # Check if file exists
        if not os.path.exists(self.embedding_extractor_path):
            raise FileNotFoundError(f"Model file not found: {self.embedding_extractor_path}")
        
        # Check if it's a TensorFlow Lite model
        start = time.time() #
        if self.embedding_extractor_path.endswith('.tflite'):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=self.embedding_extractor_path)
                self.interpreter.allocate_tensors()
                self.model_type = 'tflite'
                print("Loaded TensorFlow Lite model")
            except Exception as e:
                print(f"Error loading TFLite model: {e}")
                raise
        else:
            # Assume it's a SavedModel
            try:
                self.model = tf.saved_model.load(self.embedding_extractor_path)
                self.model_type = 'saved_model'
                print("Loaded TensorFlow SavedModel")
            except Exception as e:
                print(f"Error loading SavedModel: {e}")
                raise
        end = time.time() #
        print(f"Time to load model {end - start:.2f} seconds") #
    
    def load_data(self):
        """Load and preprocess local data."""
        print("Loading local data...")
        image_paths, labels, label_map = get_image_paths(self.data_dir)
        train_and_val_dataset, test_dataset = shuffle_and_split(image_paths, labels)
        
        # Extract embeddings using the backbone (frozen feature extractor)
        # Training embeddings
        start = time.time() #
        print("Extracting training embeddings...")
        if self.model_type == 'tflite':
            self.train_embeddings = extract_embeddings_tflite(
                train_and_val_dataset['data_train'], self.interpreter, self.input_size)
        else:
            self.train_embeddings = extract_embeddings_saved_model(
                train_and_val_dataset['data_train'], self.model, self.input_size)
        end = time.time() #
        print(f"Time to extract training embeddings: {end - start:.2f} seconds") #
        
        self.train_labels = train_and_val_dataset['labels_train']
        
        # Validation embeddings
        start = time.time() #
        print("Extracting validation embeddings...")
        if self.model_type == 'tflite':
            self.val_embeddings = extract_embeddings_tflite(
                train_and_val_dataset['data_val'], self.interpreter, self.input_size)
        else:
            self.val_embeddings = extract_embeddings_saved_model(
                train_and_val_dataset['data_val'], self.model, self.input_size)
        end = time.time() #
        print(f"Time to extract validation embeddings: {end - start:.2f} seconds") #
        
        self.val_labels = train_and_val_dataset['labels_val']

        
        # Prepare dataset for training
        self.dataset = {
            'data_train': self.train_embeddings,
            'labels_train': self.train_labels,
            'data_val': self.val_embeddings,
            'labels_val': self.val_labels
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
    
    def fit(self, parameters, config):
        """Train the head locally using local data."""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Local training parameters
        learning_rate = config.get("learning_rate", 1e-2)
        batch_size = config.get("batch_size", 100)
        num_iter = config.get("num_iter", 100)  # Reduced for federated setting
        
        print(f"Starting local training with lr={learning_rate}, batch_size={batch_size}, num_iter={num_iter}")
        
        # Train the head (only the head parameters are updated)
        print("Training classification head ...")
        start = time.time() #
        self.head.train_with_sgd(
            self.dataset, 
            num_iter, 
            learning_rate, 
            batch_size=batch_size
        )
        end = time.time() #
        print(f"Time to train classification head: {end - start:.2f} seconds") #

        # Return updated parameters and training metrics
        return self.get_parameters(config), len(self.train_embeddings), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the head on local validation data."""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Make predictions on validation data
        predictions = self.head.forward(self.val_embeddings)
        accuracy = np.mean(np.argmax(predictions, axis=1) == self.val_labels)
        
        print(f"Local validation accuracy: {accuracy:.4f}")
        
        return float(accuracy), len(self.val_embeddings), {"accuracy": float(accuracy)}


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_extractor_path", required=True, 
                       help="Path to embedding extractor model (.tflite or SavedModel)")
    parser.add_argument("--data_dir", required=True, 
                       help="Directory containing local training data")
    parser.add_argument("--num_classes", type=int, required=True,
                       help="Number of classes in the dataset")
    parser.add_argument("--server_address", default="localhost:8080",
                       help="Server address (default: localhost:8080)")
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224],
                       help="Input image size (height width) (default: 224 224)")
    
    args = parser.parse_args()
    
    # Create federated client
    client = FederatedClient(
        args.embedding_extractor_path, 
        args.data_dir, 
        args.num_classes,
        input_size=tuple(args.input_size)
    )
    
    # Start federated learning
    print(f"Starting federated learning client, connecting to {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )


if __name__ == "__main__":
    main()