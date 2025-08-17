import tensorflow as tf
import numpy as np
import os

print("TensorFlow version:", tf.__version__)

def convert_resnet_simple():
    """Simple ResNet50 to TFLite converter for older TF versions"""
    
    print("\n🔄 Loading ResNet50...")
    model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    print(f"✅ Model loaded: {model.output_shape[-1]} features")
    
    # Try quantized conversion first
    try:
        print("\n🔄 Attempting quantized conversion...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Simple representative dataset
        def rep_data():
            for _ in range(20):  # Fewer samples for Pi
                yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]
        
        converter.representative_dataset = rep_data
        
        tflite_model = converter.convert()
        
        filename = 'resnet50_embedding_extractor_quantized.tflite'
        with open(filename, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ Quantized model saved: {filename} ({size_mb:.1f} MB)")
        return filename
        
    except Exception as e:
        print(f"❌ Quantized conversion failed: {e}")
        print("🔄 Trying non-quantized conversion...")
        
        # Fallback to simple conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        filename = 'resnet50_embedding_extractor.tflite'
        with open(filename, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ Non-quantized model saved: {filename} ({size_mb:.1f} MB)")
        return filename

def test_model(filename):
    """Test the converted model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=filename)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n🧪 Testing {filename}:")
        print(f"   Input: {input_details[0]['shape']}")
        print(f"   Output: {output_details[0]['shape']}")
        print(f"   Features: {output_details[0]['shape'][-1]}")
        
        # Test inference
        test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   Test output range: [{output.min():.3f}, {output.max():.3f}]")
        print("✅ Model works correctly!")
        
        return output_details[0]['shape'][-1]
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return None

if __name__ == "__main__":
    print("="*50)
    print("🚀 SIMPLE RESNET50 CONVERTER")
    print("="*50)
    
    filename = convert_resnet_simple()
    feature_dim = test_model(filename)
    
    if feature_dim:
        print(f"\n🎯 SUCCESS! Use these settings:")
        print(f"   Server: --feature_dim {feature_dim}")
        print(f"   Client: --embedding_extractor_path {filename}")
        
        if 'quantized' in filename:
            print(f"\n⚡ For Edge TPU:")
            print(f"   Run: edgetpu_compiler {filename}")
            print(f"   Use: --embedding_extractor_path {filename.replace('.tflite', '_edgetpu.tflite')}")
    
    print("="*50)