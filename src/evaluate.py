"""
Evaluation pipeline: compute test metrics, confusion matrix, classification report.

Run: python src/evaluate.py
"""

import json
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROC_DATA_DIR, MODELS_DIR, IMG_SIZE, NUM_CLASSES,
    BATCH_SIZE, MODEL_SAVE_PATH, CLASS_NAMES_PATH
)


def load_test_dataset(test_root: Path = PROC_DATA_DIR / "test",
                     img_size: tuple = IMG_SIZE,
                     batch_size: int = BATCH_SIZE):
    """
    Load test dataset without augmentation.

    Returns:
        tf.data.Dataset, class_names
    """
    print(f"\n[*] Loading test dataset from {test_root}...")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_root,
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False  # No shuffle for evaluation
    )

    class_names = test_ds.class_names
    print(f"[✓] Test dataset loaded: {len(test_ds)} batches")
    print(f"[✓] Classes ({len(class_names)}): {', '.join(class_names[:3])}...")

    # Optimize pipeline
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return test_ds, class_names


def evaluate_model(model, test_ds, class_names: list):
    """
    Run inference on test set and compute metrics.

    Args:
        model: Compiled Keras model
        test_ds: Test dataset
        class_names: List of class names

    Returns:
        dict with metrics and predictions
    """
    print(f"\n[STEP 1] Running inference on test set...")

    y_true = []
    y_pred = []
    y_pred_probs = []

    # Iterate through batches
    for images, labels in test_ds:
        # Get predictions
        probs = model(images, training=False).numpy()
        preds = np.argmax(probs, axis=1)

        # Get true labels
        true_labels = np.argmax(labels.numpy(), axis=1)

        y_true.extend(true_labels)
        y_pred.extend(preds)
        y_pred_probs.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)

    print(f"[✓] Inference complete: {len(y_true)} test samples")

    # Compute metrics
    print(f"\n[STEP 2] Computing metrics...")

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n[✓] TEST METRICS:")
    print(f"    Accuracy:        {accuracy:.4f}")
    print(f"    Precision (macro): {precision_macro:.4f}")
    print(f"    Recall (macro):    {recall_macro:.4f}")
    print(f"    F1-Score (macro):  {f1_macro:.4f}")
    print(f"\n    Precision (weighted): {precision_weighted:.4f}")
    print(f"    Recall (weighted):    {recall_weighted:.4f}")
    print(f"    F1-Score (weighted):  {f1_weighted:.4f}")

    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
    }

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs,
        'metrics': metrics,
        'class_names': class_names
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """
    Plot and save confusion matrix.
    """
    print(f"\n[STEP 3] Generating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    # Normalize for visualization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'}, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"[✓] Confusion matrix saved to: {save_path}")
    plt.close()


def save_classification_report(y_true, y_pred, class_names, save_path: Path):
    """
    Generate and save detailed classification report.
    """
    print(f"\n[STEP 4] Generating classification report...")

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4, zero_division=0)

    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLASSIFICATION REPORT (Test Set)\n")
        f.write("="*80 + "\n\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")

    print(f"[✓] Classification report saved to: {save_path}")
    print(f"\n{report}")


def plot_per_class_metrics(y_true, y_pred, class_names, save_path: Path):
    """
    Plot per-class precision, recall, F1-score.
    """
    print(f"\n[STEP 5] Generating per-class metrics plot...")

    # Compute per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(class_names))
    width = 0.25

    ax.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
    ax.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics (Test Set)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"[✓] Per-class metrics plot saved to: {save_path}")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print("PLANT DISEASE DETECTION - MODEL EVALUATION")
    print("="*70)
    print(f"Model: {MODEL_SAVE_PATH}")
    print("="*70)

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load model
    print(f"\n[*] Loading model...")
    keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))
    print(f"[✓] Model loaded successfully")

    # Step 2: Load test dataset
    test_ds, class_names = load_test_dataset()

    # Step 3: Evaluate
    results = evaluate_model(model, test_ds, class_names)

    # Step 4: Save metrics
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"\n[✓] Metrics saved to: {metrics_path}")

    # Step 5: Plot confusion matrix
    plot_confusion_matrix(
        results['y_true'],
        results['y_pred'],
        results['class_names'],
        MODELS_DIR / "confusion_matrix.png"
    )

    # Step 6: Save classification report
    save_classification_report(
        results['y_true'],
        results['y_pred'],
        results['class_names'],
        MODELS_DIR / "classification_report.txt"
    )

    # Step 7: Plot per-class metrics
    plot_per_class_metrics(
        results['y_true'],
        results['y_pred'],
        results['class_names'],
        MODELS_DIR / "per_class_metrics.png"
    )

    # Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Metrics:              {metrics_path}")
    print(f"Confusion matrix:     {MODELS_DIR / 'confusion_matrix.png'}")
    print(f"Classification report: {MODELS_DIR / 'classification_report.txt'}")
    print(f"Per-class metrics:    {MODELS_DIR / 'per_class_metrics.png'}")
    print("\nNext steps:")
    print("  1. Run: python src/realtime.py (webcam detection)")
    print("  2. Run: streamlit run app/streamlit_app.py (web UI)")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        import traceback
        traceback.print_exc()
