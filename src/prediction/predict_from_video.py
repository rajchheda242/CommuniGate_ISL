"""
Predict a phrase from an MP4 video using the trained LSTM sequence model.

Usage (examples):
  python -m src.prediction.predict_from_video data/videos/sample_videos/phrase1.mp4
  python -m src.prediction.predict_from_video /path/to/video.mp4 --show
  python -m src.prediction.predict_from_video /path/to/video.mp4 --json

This script:
  - Reads the video frames
  - Extracts hand landmarks with MediaPipe Hands (126 features per frame)
  - Resamples the sequence to 90 frames
  - Scales features with the saved StandardScaler
  - Runs the trained LSTM model and prints the top prediction + confidence
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


# Model artifacts
MODEL_DIR = "models/saved"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "sequence_scaler.joblib")
MAPPING_PATH = os.path.join(MODEL_DIR, "phrase_mapping.json")

# Processing constants
TARGET_FRAMES = 90  # must match training
FEATURES_PER_FRAME = 126  # 2 hands * 21 landmarks * (x, y, z)


def _extract_landmarks_for_frame(results: mp.solutions.hands.Hands) -> np.ndarray:
    """Extract up to 2-hand landmarks from a single MediaPipe result as 126-dim vector.

    Pads with zeros when fewer than 2 hands, truncates when more.
    Returns a numpy array of shape (126,).
    """
    values = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                values.extend([lm.x, lm.y, lm.z])
            # Only keep up to two hands worth of landmarks
            if len(values) >= FEATURES_PER_FRAME:
                break
    # Pad or truncate to exactly 126
    if len(values) < FEATURES_PER_FRAME:
        values.extend([0.0] * (FEATURES_PER_FRAME - len(values)))
    else:
        values = values[:FEATURES_PER_FRAME]
    return np.array(values, dtype=np.float32)


def _normalize_sequence(sequence: np.ndarray, target_len: int = TARGET_FRAMES) -> np.ndarray:
    """Linearly resample a (n_frames, n_features) sequence to target_len frames."""
    n, f = sequence.shape
    if n == target_len:
        return sequence

    old_idx = np.linspace(0, n - 1, n, dtype=np.float32)
    new_idx = np.linspace(0, n - 1, target_len, dtype=np.float32)

    out = np.zeros((target_len, f), dtype=np.float32)
    for i in range(f):
        out[:, i] = np.interp(new_idx, old_idx, sequence[:, i])
    return out


def process_video_to_sequence(video_path: str, show: bool = False) -> np.ndarray:
    """Read an MP4 and return a normalized landmark sequence of shape (90, 126).

    If show=True, displays the video with drawn landmarks as it processes.
    Raises ValueError if the video can't be opened or no frames are read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    frames = []
    frames_with_hands = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Collect landmarks for this frame
            feats = _extract_landmarks_for_frame(results)
            frames.append(feats)

            # Optional display
            if show:
                if results.multi_hand_landmarks:
                    frames_with_hands += 1
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )
                cv2.imshow("Predict from video", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if not frames:
        raise ValueError("No frames read from video.")

    seq = np.stack(frames, axis=0)  # (n, 126)
    seq = _normalize_sequence(seq, TARGET_FRAMES)  # (90, 126)
    return seq


def load_artifacts():
    """Load trained model, scaler, and phrase mapping."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Mapping not found: {MAPPING_PATH}")

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(MAPPING_PATH, "r") as f:
        mapping_data = json.load(f)
        phrase_mapping = {int(k): v for k, v in mapping_data.items()}
    return model, scaler, phrase_mapping


def predict_from_sequence(seq: np.ndarray, model, scaler, mapping: dict):
    """Scale a (90,126) sequence and predict phrase + confidence."""
    if seq.shape != (TARGET_FRAMES, FEATURES_PER_FRAME):
        raise ValueError(f"Expected sequence shape {(TARGET_FRAMES, FEATURES_PER_FRAME)}, got {tuple(seq.shape)}")

    # Scale features frame-wise
    flat = seq.reshape(-1, FEATURES_PER_FRAME)
    flat_scaled = scaler.transform(flat)
    x = flat_scaled.reshape(1, TARGET_FRAMES, FEATURES_PER_FRAME)

    preds = model.predict(x, verbose=0)[0]
    cls = int(np.argmax(preds))
    conf = float(preds[cls])
    phrase = mapping.get(cls, "Unknown")
    return phrase, conf, preds.tolist()


def main():
    parser = argparse.ArgumentParser(description="Predict ISL phrase from an MP4 video")
    parser.add_argument("video", type=str, help="Path to an MP4/MOV/AVI video file")
    parser.add_argument("--show", action="store_true", help="Show video with landmarks while processing")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load artifacts
    model, scaler, mapping = load_artifacts()

    # Process video to a normalized sequence
    seq = process_video_to_sequence(video_path, show=args.show)

    # Predict
    phrase, conf, raw = predict_from_sequence(seq, model, scaler, mapping)

    if args.json:
        out = {
            "video": str(Path(video_path).resolve()),
            "phrase": phrase,
            "confidence": conf,
        }
        print(json.dumps(out, indent=2))
    else:
        print("\nPrediction Result")
        print("=" * 60)
        print(f"Video:      {video_path}")
        print(f"Predicted:  {phrase}")
        print(f"Confidence: {conf:.1%}")


if __name__ == "__main__":
    main()
