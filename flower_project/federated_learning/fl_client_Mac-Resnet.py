import flwr as fl
import numpy as np
import os
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
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


def extract_embeddings(image_paths, feature_extractor):
    """Uses ResNet50 model to process images as embeddings."""
    input_size = (224, 224)  # ResNet50 standard input size
    feature_dim = 2048  # ResNet50 with global average pooling outputs 2048 features
    embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
    
    # Process images in batches for efficiency (larger batches for Mac)
    batch_size = 32  # Larger batch size for Mac with more RAM
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load and preprocess batch of images
        for path in batch_paths:
            with test_image(path) as img:
                # Resize image to ResNet input size
                img_resized = img.resize(input_size, Image.LANCZOS)
                # Convert to RGB if needed
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                # Convert to numpy array
                img_array = np.array(img_resized).astype(np.float32)
                batch_images.append(img_array)
        
        # Stack into batch and preprocess
        batch_array = np.stack(batch_images)
        batch_preprocessed = preprocess_input(batch_array)
        
        # Extract features
        batch_features = feature_extractor.predict(batch_preprocessed, verbose=0)
        
        # Store embeddings
        end_idx = min(i + batch_size, len(image_paths))
        embeddings[i:end_idx] = batch_features[:end_idx-i]
        
        # Print progress
        print(f"Processed {end_idx}/{len(image_paths)} images")
    
    return embeddings


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, data_dir: str, num_classes: int, use_gpu: bool = False):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.use_gpu = use_gpu

        # Configure GPU/CPU usage - Force CPU for consistent comparison
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
        
        # Initialize ResNet50 feature extractor - this stays frozen
        self.load_model()
        
        # Load and preprocess data
        self.load_data()
        
        # Initialize head (softmax regression) - this is what gets trained
        self.feature_dim = self.train_embeddings.shape[1]
        print(f"Actual feature dimension from ResNet50: {self.feature_dim}")
        
        self.head = SoftmaxRegression(
            self.feature_dim, 
            self.num_classes, 
            weight_scale=5e-2, 
            reg=0.0
        )
        
        print(f"Client initialized with:")
        print(f"- Model type: ResNet50")
        print(f"- Feature dimension: {self.feature_dim}")
        print(f"- Number of classes: {self.num_classes}")
        print(f"- Training samples: {len(self.train_embeddings)}")
        print(f"- Validation samples: {len(self.val_embeddings)}")
    
    def load_model(self):
        """Load ResNet50 feature extractor."""
        print("Loading ResNet50 model...")
        
        start = time.time()
        
        # Load ResNet50 with ImageNet weights
        self.feature_extractor = ResNet50(
            weights='imagenet',
            include_top=False,  # Remove classification head
            pooling='avg'       # Global average pooling -> 2048 features
        )
        
        # Freeze all layers (no training)
        self.feature_extractor.trainable = False
        
        end = time.time()
        
        print(f"ResNet50 loaded with {len(self.feature_extractor.layers)} layers, all frozen")
        print(f"Feature output shape: {self.feature_extractor.output_shape}")  # Should be (None, 2048)
        print(f"Time to load ResNet50: {end - start:.2f} seconds")
    
    def load_data(self):
        """Load and preprocess local data."""
        print("Loading local data...")
        image_paths, labels, label_map = get_image_paths(self.data_dir)
        train_and_val_dataset, test_dataset = shuffle_and_split(image_paths, labels)
        
        # Extract embeddings using ResNet50 (frozen feature extractor)
        # Training embeddings
        start = time.time()
        print("Extracting training embeddings with ResNet50...")
        self.train_embeddings = extract_embeddings(
            train_and_val_dataset['data_train'], 
            self.feature_extractor
        )
        end = time.time()
        print(f"Time to extract training embeddings (ResNet50): {end - start:.2f} seconds")
        
        self.train_labels = train_and_val_dataset['labels_train']
        
        # Validation embeddings
        start = time.time()
        print("Extracting validation embeddings with ResNet50...")
        self.val_embeddings = extract_embeddings(
            train_and_val_dataset['data_val'], 
            self.feature_extractor
        )
        end = time.time()
        print(f"Time to extract validation embeddings (ResNet50): {end - start:.2f} seconds")
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
        print("Training classification head with ResNet50 features...")
        start = time.time()
        self.head.train_with_sgd(
            self.dataset, 
            num_iter, 
            learning_rate, 
            batch_size=batch_size
        )
        end = time.time()
        print(f"Time to train classification head: {end - start:.2f} seconds")

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
    parser.add_argument("--data_dir", required=True, 
                       help="Directory containing local training data")
    parser.add_argument("--num_classes", type=int, required=True,
                       help="Number of classes in the dataset")
    parser.add_argument("--server_address", default="localhost:8080",
                       help="Server address (default: localhost:8080)")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU if available (default: CPU only)")
    
    args = parser.parse_args()
    
    # Create federated client with ResNet50
    client = FederatedClient(
        args.data_dir, 
        args.num_classes,
        args.use_gpu
    )
    
    # Start federated learning
    print(f"Starting federated learning client with ResNet50, connecting to {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )


if __name__ == "__main__":
    main()