"""
Test TFLite model inference (Step 6 only).
Run this to validate TFLite model without re-training.

Assumes:
- models/plant_disease_mobilenetv2.keras exists
- models/plant_disease.tflite exists
"""

import numpy as np
import tensorflow as tf
import keras
from pathlib import Path

from config import TFLITE_SAVE_PATH, MODEL_SAVE_PATH


def test_tflite_inference(tflite_path: str = str(TFLITE_SAVE_PATH), model_path: str = str(MODEL_SAVE_PATH)):
    """
    Validate TFLite model by comparing inference with Keras model.

    Args:
        tflite_path: Path to .tflite model
        model_path: Path to Keras model
    """
    print(f"\n[STEP 6] Testing TFLite Model...")

    # Load Keras model (enable unsafe deserialization for old Lambda layer)
    keras.config.enable_unsafe_deserialization()
    keras_model = tf.keras.models.load_model(model_path)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[✓] TFLite model loaded successfully")
    print(f"    Input shape: {input_details[0]['shape']}")
    print(f"    Output shape: {output_details[0]['shape']}")

    # Create dummy test input
    test_input = np.random.randn(*input_details[0]['shape']).astype(np.float32)

    # Keras inference
    keras_output = keras_model(test_input, training=False).numpy()

    # TFLite inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # Compare (should be very similar)
    difference = np.abs(keras_output - tflite_output).max()
    print(f"[✓] Max output difference: {difference:.6f}")

    if difference < 0.01:
        print(f"[✓] TFLite inference matches Keras model!")
    else:
        print(f"[!] Warning: significant difference in outputs")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TFLITE MODEL VALIDATION (STEP 6 ONLY)")
    print("="*70)

    try:
        test_tflite_inference()
        print("\n" + "="*70)
        print("VALIDATION COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run: python src/evaluate.py")
        print("  2. Run: python src/realtime.py")
        print("  3. Run: streamlit run app/streamlit_app.py")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        import traceback
        traceback.print_exc()
