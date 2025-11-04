"""
Live sequence prediction module for LSTM model.
Loads trained LSTM model and predicts ISL phrases from webcam feed.
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
from collections import deque
from tensorflow.keras.models import load_model


MODEL_DIR = "models/saved"
SEQUENCE_LENGTH = 90  # Number of frames to collect (must match training data)
MIN_CONFIDENCE = 0.75  # Minimum confidence required to accept a prediction


class SequencePredictor:
    """Real-time gesture prediction using trained LSTM model."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.model = None
        self.scaler = None
        self.phrase_mapping = None
        
        # Sequence buffer for temporal analysis
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        self.load_model()
    
    def load_model(self):
        """Load trained LSTM model, scaler, and phrase mapping."""
        model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
        scaler_path = os.path.join(MODEL_DIR, "sequence_scaler.joblib")
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        
        print(f"Loading scaler from: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        print(f"Loading phrase mapping from: {mapping_path}")
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            # Convert string keys to integers
            self.phrase_mapping = {int(k): v for k, v in mapping_data.items()}
        
        print("✓ Model loaded successfully!")
        print(f"✓ Recognized phrases: {list(self.phrase_mapping.values())}")
        print(f"✓ Sequence length: {SEQUENCE_LENGTH} frames")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from Mediapipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def process_frame_landmarks(self, results):
        """Extract landmarks from current frame."""
        if not results.multi_hand_landmarks:
            # No hands detected - use zero padding
            return [0.0] * 126
        
        # Extract landmarks from detected hands
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            all_landmarks.extend(self.extract_landmarks(hand_landmarks))
        
        # Pad or truncate to 126 features (2 hands × 21 landmarks × 3 coords)
        while len(all_landmarks) < 126:
            all_landmarks.append(0.0)
        
        return all_landmarks[:126]
    
    def predict_sequence(self):
        """Predict phrase from buffered sequence."""
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0.0
        
        # Convert buffer to numpy array
        sequence = np.array(list(self.sequence_buffer))  # Shape: (60, 126)
        
        # Reshape for scaling: (60 * 126,)
        sequence_flat = sequence.reshape(-1, 126)
        
        # Scale features
        sequence_scaled = self.scaler.transform(sequence_flat)
        
        # Reshape back to sequence: (1, 60, 126)
        sequence_scaled = sequence_scaled.reshape(1, SEQUENCE_LENGTH, 126)
        
        # Predict
        predictions = self.model.predict(sequence_scaled, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        phrase = self.phrase_mapping.get(int(predicted_class), "Unknown")
        
        return phrase, confidence
    
    def run(self):
        """Start live prediction from webcam."""
        print("\n" + "="*70)
        print("Starting Live ISL Recognition")
        print("="*70)
        print("\nInstructions:")
        print("  1. Perform gestures in front of the camera")
        print("  2. Hold the gesture for 2-3 seconds")
        print("  3. Predictions will appear once sequence is complete")
        print("  4. Press 'q' to quit")
        print("  5. Press 'c' to clear sequence buffer")
        print("\n" + "="*70 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            current_phrase = "Collecting sequence..."
            confidence = 0.0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # IMPORTANT: Do NOT flip before landmark extraction to avoid
                # mirror-domain shift vs training. Process original frame.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                
                # Extract landmarks and add to buffer
                frame_landmarks = self.process_frame_landmarks(results)
                self.sequence_buffer.append(frame_landmarks)
                frame_count += 1
                
                # Predict when buffer is full
                if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                    pred_phrase, pred_conf = self.predict_sequence()
                    if pred_phrase is None:
                        current_phrase = "Processing..."
                        confidence = 0.0
                    else:
                        # Apply confidence gate to reduce false positives
                        if pred_conf >= MIN_CONFIDENCE:
                            current_phrase = pred_phrase
                            confidence = pred_conf
                        else:
                            current_phrase = "Low confidence — try again"
                            confidence = pred_conf
                
                # After processing and drawing, flip for user-friendly display only
                display_frame = cv2.flip(frame, 1)

                # Display UI
                # Background for text
                cv2.rectangle(display_frame, (10, 10), (display_frame.shape[1] - 10, 180), (0, 0, 0), -1)
                
                # Buffer status
                buffer_status = f'Frames: {len(self.sequence_buffer)}/{SEQUENCE_LENGTH}'
                cv2.putText(
                    display_frame, 
                    buffer_status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Predicted phrase
                phrase_text = f'Phrase: {current_phrase}'
                cv2.putText(
                    display_frame, 
                    phrase_text, 
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                
                # Confidence
                if confidence > 0:
                    # Gate low-confidence predictions to reduce false positives
                    accept = confidence >= MIN_CONFIDENCE
                    color = (0, 255, 0) if accept else (0, 255, 255)
                    conf_text = f'Confidence: {confidence:.1%}' + (" (low)" if not accept else "")
                    cv2.putText(
                        display_frame, 
                        conf_text, 
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        color, 
                        2
                    )
                
                # Instructions
                cv2.putText(
                    display_frame, 
                    'Press Q=Quit | C=Clear buffer', 
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (200, 200, 200), 
                    1
                )
                
                cv2.imshow('ISL Sequence Recognition', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.sequence_buffer.clear()
                    current_phrase = "Buffer cleared"
                    confidence = 0.0
                    print("Sequence buffer cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Prediction stopped.")


if __name__ == "__main__":
    try:
        predictor = SequencePredictor()
        predictor.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
