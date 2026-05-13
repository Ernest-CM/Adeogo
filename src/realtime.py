"""
Real-time plant disease detection via webcam using OpenCV.

Run: python src/realtime.py

Controls:
  - 'q': Quit
  - 's': Save current frame to models/realtime_capture.jpg
"""

import json
import time
import numpy as np
import tensorflow as tf
import keras
import cv2
from pathlib import Path

from config import (
    IMG_SIZE, MODELS_DIR, MODEL_SAVE_PATH, CLASS_NAMES_PATH,
    DISEASE_INFO
)


class RealtimeDetector:
    """Real-time plant disease detector using webcam."""

    def __init__(self, model_path: str = str(MODEL_SAVE_PATH),
                 class_names_path: str = str(CLASS_NAMES_PATH),
                 img_size: tuple = IMG_SIZE):
        """
        Initialize detector with model and class names.

        Args:
            model_path: Path to trained Keras model
            class_names_path: Path to class names JSON
            img_size: Input image size (224, 224)
        """
        self.img_size = img_size
        self.model_path = model_path
        self.class_names_path = class_names_path

        # Load model
        print("[*] Loading model...")
        keras.config.enable_unsafe_deserialization()
        self.model = tf.keras.models.load_model(model_path)
        print("[✓] Model loaded")

        # Load class names
        print("[*] Loading class names...")
        with open(class_names_path, 'r') as f:
            class_dict = json.load(f)
            self.class_names = [class_dict[str(i)] for i in range(len(class_dict))]
        print(f"[✓] Classes loaded: {len(self.class_names)} classes")

        # Timing
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model inference.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Preprocessed image batch (1, 224, 224, 3)
        """
        # Resize
        resized = cv2.resize(frame, self.img_size)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Keep in [0, 255] — model's Rescaling layer handles conversion to [-1, 1]
        rgb_float = rgb.astype(np.float32)

        # Add batch dimension
        batch = np.expand_dims(rgb_float, axis=0)

        return batch

    def predict_frame(self, frame: np.ndarray) -> tuple:
        """
        Run inference on frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            (class_name, confidence, class_index)
        """
        # Preprocess
        batch = self.preprocess_frame(frame)

        # Inference
        probs = self.model(batch, training=False).numpy()[0]
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]

        class_name = self.class_names[class_idx]

        return class_name, float(confidence), int(class_idx)

    def format_class_name(self, class_name: str) -> str:
        """
        Convert 'Tomato___Early_blight' to 'Tomato — Early Blight'.
        """
        # Replace triple underscore with separator
        parts = class_name.split('___')
        if len(parts) == 2:
            crop = parts[0]
            disease = parts[1].replace('_', ' ').title()
            return f"{crop} — {disease}"
        return class_name

    def is_healthy(self, class_name: str) -> bool:
        """Check if prediction is healthy."""
        return 'healthy' in class_name.lower()

    def run(self, camera_idx: int = 0, predict_every: int = 5, display_fps: bool = True, confidence_threshold: float = 0.70):
        """
        Run real-time detection loop.

        Args:
            camera_idx: Webcam index (0 for default)
            predict_every: Run inference every N frames
            display_fps: Show FPS counter
            confidence_threshold: Min confidence to show prediction (0.70 = 70%). Below this shows "No leaf detected"
        """
        print(f"\n[*] Opening webcam (index={camera_idx})...")
        # Use DirectShow on Windows to avoid 30-second freeze
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("[✗] Error: Could not open webcam")
            print("[*] Try a different camera index (1, 2, etc)")
            return

        print("[✓] Webcam opened successfully")
        print("[*] Controls: Press 'q' to quit, 's' to save frame")
        print("[*] Starting inference loop...")

        frame_idx = 0
        inference_time = 0
        class_name = None
        confidence = 0.0
        class_idx = -1

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[✗] Error reading frame")
                    break

                frame_idx += 1
                display_frame = frame.copy()

                # Inference every N frames
                if frame_idx % predict_every == 0:
                    t0 = time.time()
                    class_name, confidence, class_idx = self.predict_frame(frame)
                    inference_time = (time.time() - t0) * 1000  # ms

                # Format class name (use previous prediction if not updated this frame)
                if class_name is None:
                    continue

                # Check confidence threshold — reject low-confidence predictions
                if confidence < confidence_threshold:
                    pred_text = "⚠️ No Leaf Detected"
                    conf_text = f"Confidence too low ({confidence:.2%})"
                    inf_text = f"Inference: {inference_time:.1f}ms"
                    color = (0, 165, 255)  # Orange warning
                else:
                    formatted_name = self.format_class_name(class_name)
                    is_healthy = self.is_healthy(class_name)
                    color = (0, 255, 0) if is_healthy else (0, 0, 255)  # Green or red
                    pred_text = f"{formatted_name}"
                    conf_text = f"Confidence: {confidence:.2%}"
                    inf_text = f"Inference: {inference_time:.1f}ms"

                # FPS
                if display_fps:
                    self.frame_count += 1
                    elapsed = time.time() - self.last_time
                    if elapsed >= 1.0:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.last_time = time.time()
                    fps_text = f"FPS: {self.fps:.1f}"
                else:
                    fps_text = None

                # Draw overlay
                h, w = display_frame.shape[:2]
                y_offset = 40

                # Background rectangle for readability
                cv2.rectangle(display_frame, (10, 10), (w - 10, 150), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (10, 10), (w - 10, 150), color, 2)

                # Text
                cv2.putText(display_frame, pred_text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(display_frame, conf_text, (20, y_offset + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(display_frame, inf_text, (20, y_offset + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if fps_text:
                    cv2.putText(display_frame, fps_text, (w - 150, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # Instructions
                cv2.putText(display_frame, "Press 'q' to quit, 's' to save", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Display
                cv2.imshow("Plant Disease Detection - Real-time", display_frame)

                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[*] Quit signal received")
                    break
                elif key == ord('s'):
                    save_path = MODELS_DIR / "realtime_capture.jpg"
                    cv2.imwrite(str(save_path), display_frame)
                    print(f"[✓] Frame saved to: {save_path}")

        except KeyboardInterrupt:
            print("[*] Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[✓] Webcam closed")


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("PLANT DISEASE DETECTION - REAL-TIME WEBCAM")
    print("="*70)
    print(f"Model: {MODEL_SAVE_PATH}")
    print("="*70)

    # Create detector
    detector = RealtimeDetector(
        model_path=str(MODEL_SAVE_PATH),
        class_names_path=str(CLASS_NAMES_PATH),
        img_size=IMG_SIZE
    )

    # Run detection loop
    print("\n[*] Starting real-time detection...")
    try:
        detector.run(camera_idx=0, predict_every=5, display_fps=True)
    except Exception as e:
        print(f"[✗] Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Real-time detection stopped")
    print("="*70)
    print("\nNext step:")
    print("  Run: streamlit run app/streamlit_app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
