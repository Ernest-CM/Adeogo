"""
Image augmentation pipeline for plant disease detection.
Augmentation applied to training set only (not validation/test).
From project document Section 3.3.1: rotation, flipping, zooming, brightness adjustment.
"""

import tensorflow as tf
import keras
from keras import layers

from config import (
    IMG_SIZE, ROTATION_FACTOR, ZOOM_FACTOR,
    BRIGHTNESS_FACTOR, CONTRAST_FACTOR
)


def build_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Create a Keras Sequential model with augmentation layers.
    Applied only during training via model.fit()'s training flag.

    Augmentations (from project document):
    - Random horizontal/vertical flip
    - Random rotation (±15°)
    - Random zoom (±10%)
    - Random brightness adjustment (±10%)
    - Random contrast adjustment (±10%)

    Returns:
        tf.keras.Sequential: Augmentation model
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),  # Horizontal flip (crops can be any orientation)
        layers.RandomRotation(factor=ROTATION_FACTOR),  # ±15 degrees
        layers.RandomZoom(height_factor=ZOOM_FACTOR, width_factor=ZOOM_FACTOR),  # ±10%
        layers.RandomBrightness(factor=BRIGHTNESS_FACTOR),  # ±10% brightness
        layers.RandomContrast(factor=CONTRAST_FACTOR),  # ±10% contrast
    ], name="augmentation_pipeline")


def build_preprocessing_model() -> tf.keras.Sequential:
    """
    Create preprocessing model (rescaling + normalization).
    This is applied to all sets (train/val/test).

    MobileNetV2 expects inputs in the range [-1, 1], not [0, 1].
    This model rescales from [0, 255] to [-1, 1].

    Returns:
        tf.keras.Sequential: Preprocessing model
    """
    # MobileNetV2 preprocessing: rescale to [-1, 1]
    # This is equivalent to tf.keras.applications.mobilenet_v2.preprocess_input
    return tf.keras.Sequential([
        layers.Rescaling(scale=2.0/255.0, offset=-1.0, name="rescaling"),
    ], name="preprocessing")


def get_data_pipeline(dataset,
                      augment: bool = False,
                      batch_size: int = 32,
                      shuffle_buffer: int = 1000) -> tf.data.Dataset:
    """
    Configure tf.data pipeline for optimal performance.

    Args:
        dataset: tf.data.Dataset from image_dataset_from_directory
        augment: Whether to apply augmentation (True for train, False for val/test)
        batch_size: Batch size (already applied by image_dataset_from_directory)
        shuffle_buffer: Buffer size for shuffling (train only)

    Returns:
        Optimized tf.data.Dataset
    """
    # Cache preprocessed tensors in memory (significantly speeds up training)
    dataset = dataset.cache()

    # Shuffle training set only (with prefetch for performance)
    if augment:
        dataset = dataset.shuffle(shuffle_buffer)

    # Prefetch for optimal pipeline performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing augmentation pipeline...")
    aug = build_augmentation_pipeline()
    print(aug)

    print("\nTesting preprocessing...")
    preproc = build_preprocessing_model()
    print(preproc)

    # Create a dummy image and test
    import numpy as np
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    dummy_batch = np.expand_dims(dummy_image, axis=0).astype(np.float32)

    print("\nOriginal image shape:", dummy_batch.shape)
    augmented = aug(dummy_batch, training=True)
    print("Augmented shape:", augmented.shape)

    print("\n[✓] Augmentation pipeline working!")
