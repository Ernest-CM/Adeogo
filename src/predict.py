"""
Single-image inference module with module-level model caching.

Usage:
    from src.predict import predict, predict_batch, format_class_name

    result = predict("path/to/leaf.jpg")
    print(result)
    # {'class': 'Tomato___Early_blight', 'confidence': 0.95, ...}
"""

import json
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from PIL import Image

from config import (
    IMG_SIZE, MODELS_DIR, MODEL_SAVE_PATH, CLASS_NAMES_PATH,
    DISEASE_INFO
)


# Module-level cache
_model = None
_class_names = None


def load_model_cached():
    """Load model once and cache it."""
    global _model
    if _model is None:
        keras.config.enable_unsafe_deserialization()
        _model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))
    return _model


def load_class_names_cached():
    """Load class names once and cache them."""
    global _class_names
    if _class_names is None:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_dict = json.load(f)
            _class_names = [class_dict[str(i)] for i in range(len(class_dict))]
    return _class_names


def format_class_name(class_name: str) -> str:
    """
    Convert 'Tomato___Early_blight' to 'Tomato — Early Blight'.

    Args:
        class_name: Raw class name from model

    Returns:
        Formatted human-readable class name
    """
    parts = class_name.split('___')
    if len(parts) == 2:
        crop = parts[0]
        disease = parts[1].replace('_', ' ').title()
        return f"{crop} — {disease}"
    return class_name


def get_disease_info(class_name: str) -> dict:
    """
    Get disease information (description, symptoms, treatment).

    Args:
        class_name: Raw class name

    Returns:
        Dict with 'description', 'symptoms', 'treatment'
    """
    return DISEASE_INFO.get(class_name, {
        'description': 'No information available',
        'symptoms': [],
        'treatment': []
    })


def is_healthy(class_name: str) -> bool:
    """Check if prediction is a healthy plant."""
    return 'healthy' in class_name.lower()


def load_image(image_path) -> Image.Image:
    """
    Load image from file path, PIL Image, or numpy array.

    Args:
        image_path: Str path, Path object, PIL Image, or numpy array

    Returns:
        PIL Image (RGB)
    """
    if isinstance(image_path, (str, Path)):
        img = Image.open(image_path)
    elif isinstance(image_path, Image.Image):
        img = image_path
    elif isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path.astype('uint8'))
    else:
        raise TypeError(f"Unsupported type: {type(image_path)}")

    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return img


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess PIL image for model inference.

    Args:
        image: PIL Image

    Returns:
        Batch array (1, 224, 224, 3) in [0, 255] range
    """
    # Resize
    resized = image.resize(IMG_SIZE)

    # To array
    arr = np.array(resized, dtype=np.float32)

    # Keep in [0, 255] — model's Rescaling layer handles conversion to [-1, 1]
    # DO NOT normalize to [0, 1] here, as it breaks the Rescaling layer

    # Add batch dimension
    batch = np.expand_dims(arr, axis=0)

    return batch


def predict(image_input, return_top_k: int = 1) -> dict:
    """
    Run inference on single image.

    Args:
        image_input: File path (str/Path), PIL Image, or numpy array
        return_top_k: Return top K predictions (default 1)

    Returns:
        Dict with:
        - 'class': Predicted class name
        - 'class_formatted': Formatted name (e.g. 'Tomato — Early Blight')
        - 'confidence': Confidence score [0, 1]
        - 'is_healthy': Bool
        - 'all_probs': Array of all class probabilities
        - 'top_k': List of (class, prob) tuples
        - 'disease_info': Disease information dict
    """
    # Load model and class names
    model = load_model_cached()
    class_names = load_class_names_cached()

    # Load and preprocess image
    image = load_image(image_input)
    batch = preprocess_image(image)

    # Inference
    probs = model(batch, training=False).numpy()[0]
    class_idx = np.argmax(probs)
    confidence = probs[class_idx]

    class_name = class_names[class_idx]
    class_formatted = format_class_name(class_name)
    healthy = is_healthy(class_name)
    disease_info = get_disease_info(class_name)

    # Top-k
    top_k_idx = np.argsort(probs)[-return_top_k:][::-1]
    top_k = [(class_names[i], float(probs[i])) for i in top_k_idx]

    return {
        'class': class_name,
        'class_formatted': class_formatted,
        'confidence': float(confidence),
        'is_healthy': healthy,
        'all_probs': probs,
        'top_k': top_k,
        'disease_info': disease_info,
    }


def predict_batch(image_inputs, return_top_k: int = 1) -> list:
    """
    Run inference on multiple images.

    Args:
        image_inputs: List of image paths/PIL Images/arrays
        return_top_k: Return top K predictions per image

    Returns:
        List of result dicts (one per image)
    """
    return [predict(img, return_top_k) for img in image_inputs]


def batch_predict_fast(images: np.ndarray) -> np.ndarray:
    """
    Fast batch inference (raw numpy).

    Args:
        images: Batch array (N, 224, 224, 3) with values in [0, 1]

    Returns:
        Probabilities array (N, 15)
    """
    model = load_model_cached()
    return model(images, training=False).numpy()


if __name__ == "__main__":
    # Example usage
    print("Testing predict module...")

    # Create a dummy image for testing
    dummy_img = Image.new('RGB', IMG_SIZE, color=(100, 150, 100))

    result = predict(dummy_img)
    print(f"\nPrediction: {result['class_formatted']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Healthy: {result['is_healthy']}")
    print(f"\nTop 3 predictions:")
    for class_name, prob in result['top_k'][:3]:
        print(f"  - {format_class_name(class_name)}: {prob:.2%}")

    print("\n[✓] predict module working correctly!")
