import flwr as fl
import numpy as np
import os
import time
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
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


def extract_embeddings(image_paths, interpreter):
    """Uses model to process images as embeddings."""
    input_size = common.input_size(interpreter)
    feature_dim = classify.num_classes(interpreter)
    embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
    
    for idx, path in enumerate(image_paths):
        with test_image(path) as img:
            common.set_input(interpreter, img.resize(input_size, Image.NEAREST))
            interpreter.invoke()
            embeddings[idx, :] = classify.get_scores(interpreter)

    return embeddings


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, embedding_extractor_path: str, data_dir: str, num_classes: int):
        self.embedding_extractor_path = embedding_extractor_path
        self.data_dir = data_dir
        self.num_classes = num_classes
        
        # Initialize the backbone (embedding extractor) - this stays frozen
        self.interpreter = make_interpreter(embedding_extractor_path, device=':0')
        self.interpreter.allocate_tensors()
        
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
    
    def load_data(self):
        """Load and preprocess local data."""
        print("Loading local data...")
        image_paths, labels, label_map = get_image_paths(self.data_dir)
        train_and_val_dataset, test_dataset = shuffle_and_split(image_paths, labels)
        
        # Extract embeddings using the backbone (frozen feature extractor)
        print("Extracting training embeddings...")
        self.train_embeddings = extract_embeddings(
            train_and_val_dataset['data_train'], self.interpreter)
        self.train_labels = train_and_val_dataset['labels_train']
        
        print("Extracting validation embeddings...")
        self.val_embeddings = extract_embeddings(
            train_and_val_dataset['data_val'], self.interpreter)
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
        self.head.train_with_sgd(
            self.dataset, 
            num_iter, 
            learning_rate, 
            batch_size=batch_size
        )
        
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
    
    # Start federated learning
    print(f"Starting federated learning client, connecting to {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )


if __name__ == "__main__":
    main()