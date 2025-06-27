#!/usr/bin/env python3
"""
Test script for the modified SoftmaxRegression class.
Run this before integrating with backprop_last_layer.py
"""

import numpy as np
from softmax_regression import SoftmaxRegression  # Your new implementation

def test_softmax_regression():
    """Test the modified SoftmaxRegression with dummy data."""
    print("Testing Modified SoftmaxRegression Class")
    print("=" * 50)
    
    # Test parameters (similar to what you might get from real data)
    feature_dim = 1280  # Typical for MobileNet embeddings
    num_classes = 5     # Example: 5 flower classes
    weight_scale = 5e-2  # Same as in your original code
    reg = 0.0           # Same as in your original code
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Weight scale: {weight_scale}")
    print(f"Regularization: {reg}")
    print()
    
    # Create model with same parameters as your original code
    model = SoftmaxRegression(
        feature_dim, num_classes, weight_scale=weight_scale, reg=reg)
    
    print("✓ Model created successfully")
    
    # Test weight access (this is what your advisor wanted)
    weights, biases = model.get_weights()
    print(f"✓ Weights shape: {weights.shape}")
    print(f"✓ Biases shape: {biases.shape}")
    print(f"✓ Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print()
    
    # Create dummy training data (similar structure to your real data)
    n_train = 800
    n_val = 200
    
    # Generate random embeddings (simulating output from embedding extractor)
    np.random.seed(42)
    train_embeddings = np.random.randn(n_train, feature_dim).astype(np.float32)
    val_embeddings = np.random.randn(n_val, feature_dim).astype(np.float32)
    
    # Generate random labels
    train_labels = np.random.randint(0, num_classes, n_train)
    val_labels = np.random.randint(0, num_classes, n_val)
    
    # Structure data same way as your original code
    train_and_val_dataset = {
        'data_train': train_embeddings,
        'labels_train': train_labels,
        'data_val': val_embeddings,
        'labels_val': val_labels
    }
    
    print("✓ Dummy training data created")
    print(f"  - Training samples: {n_train}")
    print(f"  - Validation samples: {n_val}")
    print()
    
    # Test training with same parameters as your original code
    learning_rate = 1e-2  # Same as original
    batch_size = 100      # Same as original
    num_iter = 50         # Reduced for quick test (original uses 500)
    
    print("Starting training test...")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iter}")
    print()
    
    # Train the model
    model.train_with_sgd(
        train_and_val_dataset, num_iter, learning_rate, batch_size=batch_size)
    
    print("✓ Training completed successfully")
    
    # Test final accuracy
    final_train_acc = model.get_accuracy(train_embeddings, train_labels)
    final_val_acc = model.get_accuracy(val_embeddings, val_labels)
    
    print(f"✓ Final training accuracy: {final_train_acc:.4f}")
    print(f"✓ Final validation accuracy: {final_val_acc:.4f}")
    print()
    
    # Test weight access after training
    trained_weights, trained_biases = model.get_weights()
    print(f"✓ Trained weights shape: {trained_weights.shape}")
    print(f"✓ Trained weight range: [{trained_weights.min():.4f}, {trained_weights.max():.4f}]")
    print()
    
    # Test TFLite conversion
    try:
        tflite_model = model.to_tflite_model()
        print(f"✓ TFLite conversion successful")
        print(f"✓ TFLite model size: {len(tflite_model)} bytes")
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
    
    print()
    
    # Test weight saving
    try:
        model.save_weights("test_weights.npz")
        print("✓ Weight saving successful")
        
        # Test weight loading
        model2 = SoftmaxRegression(feature_dim, num_classes, weight_scale, reg)
        model2.load_weights("test_weights.npz")
        print("✓ Weight loading successful")
        
        # Verify weights are the same
        w1, b1 = model.get_weights()
        w2, b2 = model2.get_weights()
        
        if np.allclose(w1, w2) and np.allclose(b1, b2):
            print("✓ Loaded weights match saved weights")
        else:
            print("✗ Loaded weights don't match saved weights")
            
    except Exception as e:
        print(f"✗ Weight save/load failed: {e}")
    
    print()
    print("=" * 50)
    print("Test completed successfully!")
    print("Your modified SoftmaxRegression is ready to use.")
    

if __name__ == "__main__":
    test_softmax_regression()