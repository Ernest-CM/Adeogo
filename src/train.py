"""
Training pipeline for plant disease detection model.
Trains MobileNetV2 transfer learning model on solanaceous crop dataset.

Run: python src/train.py
"""

import os
import json
import tensorflow as tf
import keras
from pathlib import Path
from datetime import datetime

from config import (
    PROC_DATA_DIR, MODELS_DIR, IMG_SIZE, NUM_CLASSES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, DROPOUT_RATE,
    INITIAL_TRAINING_EPOCHS, FINE_TUNING_EPOCHS, FINE_TUNING_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    MODEL_SAVE_PATH, TFLITE_SAVE_PATH, CLASS_NAMES_PATH,
    TRAINING_LOG_PATH, BEST_MODEL_PATH
)
from model import build_mobilenetv2_model, unfreeze_top_layers
from augmentation import get_data_pipeline


def create_data_generators(proc_root: Path = PROC_DATA_DIR,
                           img_size: tuple = IMG_SIZE,
                           batch_size: int = BATCH_SIZE):
    """
    Create tf.data.Dataset generators for train/val/test sets.
    Uses tf.keras.utils.image_dataset_from_directory for automated image loading.

    Returns:
        (train_ds, val_ds, test_ds, class_names)
    """
    print("\n[STEP 1] Creating data pipelines...")

    # Training set with augmentation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        proc_root / "train",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    print(f"[✓] Train dataset: {len(train_ds)} batches")

    # Validation set (no augmentation)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        proc_root / "val",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    print(f"[✓] Validation dataset: {len(val_ds)} batches")

    # Test set (no augmentation)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        proc_root / "test",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    print(f"[✓] Test dataset: {len(test_ds)} batches")

    # Get class names from the training set
    class_names = train_ds.class_names
    print(f"[✓] Classes ({len(class_names)}): {', '.join(class_names[:3])}...")

    # Optimize pipelines
    print("\n[*] Optimizing data pipelines...")
    train_ds = get_data_pipeline(train_ds, augment=True, batch_size=batch_size)
    val_ds = get_data_pipeline(val_ds, augment=False, batch_size=batch_size)
    test_ds = get_data_pipeline(test_ds, augment=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds, class_names


def build_callbacks(models_dir: Path = MODELS_DIR) -> list:
    """
    Create Keras callbacks for training.
    Includes: checkpoint, early stopping, LR reduction, CSV logging.

    Returns:
        List of Keras callbacks
    """
    print("\n[STEP 2] Setting up training callbacks...")

    callbacks = [
        # Save best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_MODEL_PATH),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1,
            save_freq='epoch'
        ),

        # Stop training if validation loss doesn't improve
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate if validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),

        # Log training metrics to CSV
        tf.keras.callbacks.CSVLogger(
            str(TRAINING_LOG_PATH),
            separator=',',
            append=False
        ),

        # TensorBoard for visualization (optional)
        tf.keras.callbacks.TensorBoard(
            log_dir=str(models_dir / 'logs'),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        ),
    ]

    print(f"[✓] Callbacks configured:")
    for cb in callbacks:
        print(f"    - {cb.__class__.__name__}")

    return callbacks


def run_initial_training(model, train_ds, val_ds, callbacks, epochs: int = INITIAL_TRAINING_EPOCHS):
    """
    Initial training phase: frozen base, train only the classification head.

    Args:
        model: Compiled Keras model
        train_ds: Training dataset
        val_ds: Validation dataset
        callbacks: List of Keras callbacks
        epochs: Number of epochs

    Returns:
        History object
    """
    print(f"\n[STEP 3] Initial Training (Base Frozen, {epochs} epochs)...")
    print(f"[*] Learning rate: {LEARNING_RATE}")
    print(f"[*] Batch size: {BATCH_SIZE}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history


def run_fine_tuning(model, train_ds, val_ds, callbacks, epochs: int = FINE_TUNING_EPOCHS):
    """
    Fine-tuning phase: unfreeze top layers, train with lower learning rate.

    Args:
        model: Model from initial training
        train_ds: Training dataset
        val_ds: Validation dataset
        callbacks: List of Keras callbacks
        epochs: Number of fine-tuning epochs

    Returns:
        History object
    """
    print(f"\n[STEP 4] Fine-Tuning (Top Layers Unfrozen, {epochs} epochs)...")
    print(f"[*] Learning rate: {FINE_TUNING_LEARNING_RATE}")

    # Unfreeze top layers
    model = unfreeze_top_layers(
        model,
        num_layers=30,
        new_lr=FINE_TUNING_LEARNING_RATE
    )

    print(f"[✓] Top 30 layers unfrozen for fine-tuning")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history


def export_to_tflite(model, tflite_path: str = str(TFLITE_SAVE_PATH), quantize: bool = True) -> None:
    """
    Convert Keras model to TensorFlow Lite format.
    Includes dynamic range quantization for ~4x size reduction.

    Args:
        model: Trained Keras model
        tflite_path: Output path for .tflite file
        quantize: Whether to apply dynamic range quantization
    """
    print(f"\n[STEP 5] Exporting to TensorFlow Lite...")
    print(f"[*] Quantization: {'Yes (dynamic range)' if quantize else 'No'}")

    # Convert model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply dynamic range quantization
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    # Print sizes
    keras_size_mb = Path(MODEL_SAVE_PATH).stat().st_size / (1024 * 1024)
    tflite_size_mb = Path(tflite_path).stat().st_size / (1024 * 1024)
    compression_ratio = keras_size_mb / tflite_size_mb

    print(f"[✓] TFLite model exported successfully!")
    print(f"    Keras model:  {keras_size_mb:.1f} MB")
    print(f"    TFLite model: {tflite_size_mb:.1f} MB")
    print(f"    Compression:  {compression_ratio:.1f}x")


def test_tflite_inference(tflite_path: str = str(TFLITE_SAVE_PATH), model_path: str = str(MODEL_SAVE_PATH)):
    """
    Validate TFLite model by comparing inference with Keras model.

    Args:
        tflite_path: Path to .tflite model
        model_path: Path to Keras model
    """
    print(f"\n[STEP 6] Testing TFLite Model...")

    # Load Keras model
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
    import numpy as np
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


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("PLANT DISEASE DETECTION - MODEL TRAINING")
    print("="*70)
    print(f"Project: Landmark University - B.Sc. Computer Science")
    print(f"Author: Ekunjesu Adeogo (22CD009343)")
    print(f"Model: MobileNetV2 Transfer Learning")
    print("="*70)

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Create data pipelines
    train_ds, val_ds, test_ds, class_names = create_data_generators()

    # Step 2: Build model
    print(f"\n[*] Building MobileNetV2 model ({NUM_CLASSES} classes)...")
    model = build_mobilenetv2_model(
        num_classes=NUM_CLASSES,
        freeze_base=True
    )
    print(f"[✓] Model created")
    print(f"    Total parameters: {model.count_params():,}")

    # Step 3: Setup callbacks
    callbacks = build_callbacks()

    # Step 4: Initial training (frozen base)
    history_1 = run_initial_training(model, train_ds, val_ds, callbacks, INITIAL_TRAINING_EPOCHS)

    # Step 5: Fine-tuning (unfrozen top layers)
    history_2 = run_fine_tuning(model, train_ds, val_ds, callbacks, FINE_TUNING_EPOCHS)

    # Step 6: Save model
    print(f"\n[*] Saving model...")
    model.save(str(MODEL_SAVE_PATH))
    print(f"[✓] Model saved to: {MODEL_SAVE_PATH}")

    # Step 7: Save class names
    class_names_dict = {i: name for i, name in enumerate(class_names)}
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names_dict, f, indent=2)
    print(f"[✓] Class names saved to: {CLASS_NAMES_PATH}")

    # Step 8: Export to TFLite
    export_to_tflite(model, str(TFLITE_SAVE_PATH), quantize=True)

    # Step 9: Test TFLite
    test_tflite_inference()

    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Trained model: {MODEL_SAVE_PATH}")
    print(f"TFLite model:  {TFLITE_SAVE_PATH}")
    print(f"Class names:   {CLASS_NAMES_PATH}")
    print(f"Training log:  {TRAINING_LOG_PATH}")
    print("\nNext steps:")
    print("  1. Run: python src/evaluate.py")
    print("  2. Run: python src/realtime.py")
    print("  3. Run: streamlit run app/streamlit_app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        import traceback
        traceback.print_exc()
