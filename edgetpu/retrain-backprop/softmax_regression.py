import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
import tempfile
import os


class SoftmaxRegression:
    """A pure Python/TensorFlow implementation of softmax regression for TFLite compatibility.
    
    This implementation provides direct access to weights and biases, and can be
    easily converted to TensorFlow Lite format.
    """
    
    def __init__(self, feature_dim: int, num_classes: int, weight_scale: float = 0.01, reg: float = 0.0):
        """Initialize the softmax regression model.
        
        Args:
            feature_dim: The dimension of the input feature vector
            num_classes: The number of output classes
            weight_scale: Scale factor for weight initialization
            reg: L2 regularization strength
        """
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.weight_scale = weight_scale
        self.reg = reg
        
        # Initialize weights and biases
        self.weights = np.random.normal(0, weight_scale, (feature_dim, num_classes)).astype(np.float32)
        self.biases = np.zeros(num_classes, dtype=np.float32)
        
        # Keep track of training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current weights and biases.
        
        Returns:
            Tuple of (weights, biases) as numpy arrays
        """
        return self.weights.copy(), self.biases.copy()
    
    def set_weights(self, weights: np.ndarray, biases: np.ndarray):
        """Set the weights and biases.
        
        Args:
            weights: Weight matrix of shape (feature_dim, num_classes)
            biases: Bias vector of shape (num_classes,)
        """
        assert weights.shape == (self.feature_dim, self.num_classes)
        assert biases.shape == (self.num_classes,)
        self.weights = weights.astype(np.float32)
        self.biases = biases.astype(np.float32)
    
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities.
        
        Args:
            logits: Input logits of shape (batch_size, num_classes)
            
        Returns:
            Softmax probabilities of same shape
        """
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the model.
        
        Args:
            X: Input features of shape (batch_size, feature_dim)
            
        Returns:
            Softmax probabilities of shape (batch_size, num_classes)
        """
        logits = np.dot(X, self.weights) + self.biases
        return self.softmax(logits)
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss with L2 regularization.
        
        Args:
            X: Input features of shape (batch_size, feature_dim)
            y: True labels of shape (batch_size,)
            
        Returns:
            Average loss per sample
        """
        batch_size = X.shape[0]
        
        # Forward pass
        probs = self.forward(X)
        
        # Cross-entropy loss
        log_probs = np.log(probs + 1e-15)  # Add small epsilon to avoid log(0)
        loss = -np.sum(log_probs[np.arange(batch_size), y]) / batch_size
        
        # Add L2 regularization
        reg_loss = 0.5 * self.reg * np.sum(self.weights ** 2)
        
        return loss + reg_loss
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for weights and biases.
        
        Args:
            X: Input features of shape (batch_size, feature_dim)
            y: True labels of shape (batch_size,)
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        batch_size = X.shape[0]
        
        # Forward pass
        probs = self.forward(X)
        
        # Create one-hot encoding of labels
        y_onehot = np.zeros((batch_size, self.num_classes))
        y_onehot[np.arange(batch_size), y] = 1
        
        # Compute gradients
        dlogits = (probs - y_onehot) / batch_size
        dW = np.dot(X.T, dlogits) + self.reg * self.weights
        db = np.sum(dlogits, axis=0)
        
        return dW, db
    
    def get_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy on given data.
        
        Args:
            X: Input features of shape (batch_size, feature_dim)
            y: True labels of shape (batch_size,)
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        probs = self.forward(X)
        predictions = np.argmax(probs, axis=1)
        return np.mean(predictions == y)
    
    def train_with_sgd(self, data: Dict[str, np.ndarray], num_iter: int, 
                       learning_rate: float, batch_size: int = 100, print_every: int = 100):
        """Train the model using stochastic gradient descent.
        
        Args:
            data: Dictionary containing 'data_train', 'labels_train', 'data_val', 'labels_val'
            num_iter: Number of training iterations
            learning_rate: Learning rate for SGD
            batch_size: Batch size for mini-batch SGD
            print_every: Print stats every N iterations (0 to disable)
        """
        X_train = data['data_train']
        y_train = data['labels_train']
        X_val = data['data_val']
        y_val = data['labels_val']
        
        n_train = X_train.shape[0]
        
        for i in range(num_iter):
            # Create mini-batch
            batch_indices = np.random.choice(n_train, batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Compute gradients and update weights
            dW, db = self.compute_gradients(X_batch, y_batch)
            self.weights -= learning_rate * dW
            self.biases -= learning_rate * db
            
            # Print progress
            if print_every > 0 and (i + 1) % print_every == 0:
                train_loss = self.compute_loss(X_batch, y_batch)
                train_acc = self.get_accuracy(X_train, y_train)
                val_acc = self.get_accuracy(X_val, y_val)
                
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                
                print(f"Iteration {i+1}/{num_iter}: "
                      f"Loss = {train_loss:.4f}, "
                      f"Train Acc = {train_acc:.4f}, "
                      f"Val Acc = {val_acc:.4f}")
    
    def to_tflite_model(self) -> bytes:
        """Convert the trained model to TensorFlow Lite format.
        
        Returns:
            TFLite model as bytes
        """
        # Create a TensorFlow model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.num_classes,
                activation='softmax',
                input_shape=(self.feature_dim,),
                use_bias=True
            )
        ])
        
        # Set the trained weights
        model.layers[0].set_weights([self.weights, self.biases])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        return tflite_model
    
    def serialize_model(self, embedding_extractor_path: str) -> bytes:
        """Append the trained softmax layer to an existing embedding extractor.
        
        Args:
            embedding_extractor_path: Path to the embedding extractor .tflite file
            
        Returns:
            Combined model as bytes
        """
        # This is a simplified version - for full implementation you'd need
        # to properly combine the models using TensorFlow operations
        
        # For now, we'll create a standalone softmax model
        # In a full implementation, you'd load the embedding extractor,
        # append the softmax layer, and create a combined model
        
        return self.to_tflite_model()
    
    def save_weights(self, filepath: str):
        """Save weights and biases to a file.
        
        Args:
            filepath: Path where to save the weights
        """
        np.savez(filepath, weights=self.weights, biases=self.biases)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath: str):
        """Load weights and biases from a file.
        
        Args:
            filepath: Path to the saved weights file
        """
        data = np.load(filepath)
        self.weights = data['weights']
        self.biases = data['biases']
        print(f"Weights loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data for testing
    feature_dim = 128
    num_classes = 5
    n_samples = 1000
    
    # Generate random data
    np.random.seed(42)
    X = np.random.randn(n_samples, feature_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, n_samples)
    
    # Split data
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    
    data = {
        'data_train': X[:train_size],
        'labels_train': y[:train_size],
        'data_val': X[train_size:train_size+val_size],
        'labels_val': y[train_size:train_size+val_size]
    }
    
    # Create and train model
    model = SoftmaxRegression(feature_dim, num_classes, weight_scale=0.01, reg=0.001)
    
    print("Training softmax regression model...")
    model.train_with_sgd(data, num_iter=200, learning_rate=0.01, batch_size=64, print_every=50)
    
    # Get weights
    weights, biases = model.get_weights()
    print(f"\nTrained weights shape: {weights.shape}")
    print(f"Trained biases shape: {biases.shape}")
    
    # Test accuracy
    test_data = X[train_size+val_size:]
    test_labels = y[train_size+val_size:]
    test_acc = model.get_accuracy(test_data, test_labels)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Convert to TFLite
    tflite_model = model.to_tflite_model()
    print(f"TFLite model size: {len(tflite_model)} bytes")
    
    # Save weights
    model.save_weights("softmax_weights.npz")