"""
MobileNetV2 transfer learning model for plant disease classification.
Architecture from project document Chapter 4, Table 4.2.
"""

import tensorflow as tf
import keras
from keras import layers, models, optimizers

from config import (
    IMG_SIZE, NUM_CHANNELS, NUM_CLASSES,
    LEARNING_RATE, DROPOUT_RATE,
    FINE_TUNING_LEARNING_RATE
)
from augmentation import build_augmentation_pipeline, build_preprocessing_model


def build_mobilenetv2_model(num_classes: int = NUM_CLASSES,
                             img_size: tuple = IMG_SIZE,
                             learning_rate: float = LEARNING_RATE,
                             dropout_rate: float = DROPOUT_RATE,
                             freeze_base: bool = True) -> tf.keras.Model:
    """
    Build MobileNetV2 transfer learning model.

    Architecture (from project document Table 4.2):
    1. Input: (224, 224, 3) RGB images
    2. Augmentation (random flip, rotation, zoom, brightness)
    3. Preprocessing (rescale + normalize to [-1, 1])
    4. MobileNetV2 base (imagenet weights, no top classification layer)
    5. GlobalAveragePooling2D (spatial reduction)
    6. BatchNormalization
    7. Dense(256, relu) with dropout(dropout_rate)
    8. Dense(num_classes, softmax)

    Args:
        num_classes: Number of output classes (15 for solanaceous)
        img_size: Input image size (224, 224)
        learning_rate: Adam optimizer learning rate
        dropout_rate: Dropout rate (0.4)
        freeze_base: Whether to freeze MobileNetV2 base weights (True initially)

    Returns:
        Compiled tf.keras.Model
    """
    # --- Input layer ---
    inputs = layers.Input(shape=(img_size[0], img_size[1], NUM_CHANNELS),
                          name="input_images")

    # --- Augmentation (only active during training) ---
    x = build_augmentation_pipeline()(inputs, training=True)

    # --- Preprocessing ---
    x = build_preprocessing_model()(x)

    # --- MobileNetV2 base with ImageNet weights ---
    # MobileNetV2 is trained to recognize general visual features
    # We leverage these pre-trained weights for faster convergence and better accuracy
    mobilenetv2_base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], NUM_CHANNELS),
        include_top=False,  # Remove classification head
        weights='imagenet'  # Pre-trained ImageNet weights
    )

    # Freeze base weights if requested (no training of base during initial training)
    if freeze_base:
        mobilenetv2_base.trainable = False
    else:
        mobilenetv2_base.trainable = True

    # Pass through base
    x = mobilenetv2_base(x, training=False if freeze_base else True)

    # --- Feature aggregation ---
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # --- Dense classifier head ---
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dense(256, activation='relu', name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name=f"dropout_{dropout_rate}")(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name="predictions")(x)

    # --- Build model ---
    model = models.Model(inputs=inputs, outputs=outputs,
                         name="PlantDiseaseDetection_MobileNetV2")

    # --- Compile ---
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def unfreeze_top_layers(model: tf.keras.Model,
                        num_layers: int = 30,
                        new_lr: float = FINE_TUNING_LEARNING_RATE) -> tf.keras.Model:
    """
    Unfreeze top layers of MobileNetV2 for fine-tuning.
    Fine-tuning: train the top convolutional layers with very low learning rate
    to slightly adapt pre-trained features to this specific task.

    Args:
        model: Compiled model with frozen base
        num_layers: Number of layers to unfreeze from the top
        new_lr: Learning rate for fine-tuning (typically lower than initial)

    Returns:
        Recompiled model ready for fine-tuning
    """
    # Find the MobileNetV2 base layer
    mobilenetv2_layer = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            mobilenetv2_layer = layer
            break

    if mobilenetv2_layer is None:
        print("[!] Warning: Could not find MobileNetV2 layer, unfreezing last 30 layers of all model")
        # Fallback: unfreeze last num_layers of the entire model
        for layer in model.layers[-num_layers:]:
            layer.trainable = True
    else:
        # Unfreeze only the top num_layers of MobileNetV2
        mobilenetv2_layer.trainable = True
        for layer in mobilenetv2_layer.layers[:-num_layers]:
            layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=new_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a saved Keras model from disk.
    Handles both .keras and .h5 formats.

    Args:
        model_path: Path to model file

    Returns:
        Loaded model
    """
    return tf.keras.models.load_model(model_path)


def save_model(model: tf.keras.Model, save_path: str) -> None:
    """
    Save model to disk.

    Args:
        model: Model to save
        save_path: Path to save to (should end in .keras)
    """
    model.save(save_path)
    print(f"[✓] Model saved to: {save_path}")


if __name__ == "__main__":
    # Test model creation
    print("Creating MobileNetV2 model...")
    model = build_mobilenetv2_model(num_classes=NUM_CLASSES, freeze_base=True)

    print("\nModel summary:")
    model.summary()

    print(f"\n[✓] Model created successfully!")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Trainable parameters: {sum(p.numpy().size for p in model.trainable_weights):,}")
