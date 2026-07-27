import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

# Import TFLite interpreter (lightweight tflite-runtime or full tensorflow package)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: Could not import 'tflite-runtime' or 'tensorflow'.")
        print("Please install tflite-runtime or tensorflow first:")
        print("    pip install tflite-runtime")
        print("    or")
        print("    pip install tensorflow")
        sys.exit(1)

def load_classes(classes_path):
    """Load classes configuration file."""
    if not os.path.exists(classes_path):
        print(f"Error: Classes file not found at {classes_path}")
        print("Make sure you provide the correct classes.json path.")
        sys.exit(1)
        
    try:
        with open(classes_path, 'r') as f:
            data = json.load(f)
        return data.get("idx_to_class", data.get("class_names", []))
    except Exception as e:
        print(f"Error reading classes file: {e}")
        sys.exit(1)

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess the input image."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
        
    try:
        # Load image
        img = Image.open(image_path)
        # Convert to RGB (handles RGBA or Grayscale)
        img = img.convert('RGB')
        # Resize to target input size
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        # Convert to numpy array and add batch dimension
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)

def run_inference(model_path, input_data):
    """Run inference on the TFLite model."""
    if not os.path.exists(model_path):
        print(f"Error: TFLite model not found at {model_path}")
        sys.exit(1)
        
    try:
        # Load the TFLite model and allocate tensors
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check expected shape and type
        expected_shape = input_details[0]['shape']
        expected_dtype = input_details[0]['dtype']
        
        # Point the data to the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(expected_dtype))

        # Run inference
        interpreter.invoke()

        # Extract predictions
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        return output_data
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run inference using the FP32 Waste Classifier TFLite Model.")
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument("--model", default=None, help="Path to the TFLite model file (defaults to ../models/waste_classifier_fp32.tflite relative to script).")
    parser.add_argument("--classes", default=None, help="Path to classes.json file (defaults to ../classes.json relative to script).")
    args = parser.parse_args()

    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if args.model else os.path.join(script_dir, "..", "models", "waste_classifier_fp32.tflite")
    classes_path = args.classes if args.classes else os.path.join(script_dir, "..", "classes.json")

    print("--- Waste Classifier Inference ---")
    print(f"Loading Model  : {os.path.abspath(model_path)}")
    print(f"Loading Classes: {os.path.abspath(classes_path)}")
    print(f"Input Image    : {args.image}")
    print("----------------------------------")

    # Load classes mapping
    classes = load_classes(classes_path)
    # Ensure classes is a dict with string keys, or list
    if isinstance(classes, list):
        idx_to_class = {str(i): name for i, name in enumerate(classes)}
    else:
        idx_to_class = classes

    # Load and preprocess image
    # EfficientNet-B0 default input resolution is 224x224
    input_data = preprocess_image(args.image, target_size=(224, 224))

    # Run prediction
    predictions = run_inference(model_path, input_data)

    # Output predictions
    print("\nClassification Probabilities:")
    predicted_idx = np.argmax(predictions)
    
    # Print sorted results by confidence
    sorted_indices = np.argsort(predictions)[::-1]
    for idx in sorted_indices:
        class_name = idx_to_class.get(str(idx), f"Class {idx}")
        prob = predictions[idx]
        # Format as percentage
        print(f"  {class_name:<15}: {prob * 100:>6.2f}%")

    predicted_class = idx_to_class.get(str(predicted_idx), f"Class {predicted_idx}")
    print("----------------------------------")
    print(f"PREDICTED CLASS: {predicted_class.upper()} (Confidence: {predictions[predicted_idx]*100:.2f}%)")
    print("----------------------------------")

if __name__ == "__main__":
    main()
