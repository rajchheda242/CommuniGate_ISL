"""
Smart ISL predictor with user-controlled recording.
Records gesture sequences only when user wants, then processes.
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
MIN_FRAMES = 60   # Minimum frames needed for prediction
MAX_FRAMES = 150  # Maximum frames to keep


class SmartPredictor:
    """User-controlled gesture prediction with better UX."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.model = None
        self.scaler = None
        self.phrase_mapping = None
        
        # Recording state
        self.is_recording = False
        self.recorded_sequence = []
        self.last_prediction = None
        self.last_confidence = 0.0
        
        # Hand detection tracking
        self.frames_without_hands = 0
        self.frames_with_hands = 0
        
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
            self.phrase_mapping = {int(k): v for k, v in mapping_data.items()}
        
        print("✓ Model loaded successfully!")
        print(f"✓ Recognized phrases:")
        for idx, phrase in self.phrase_mapping.items():
            print(f"  {idx + 1}. {phrase}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from Mediapipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def process_frame_landmarks(self, results):
        """Extract landmarks from current frame."""
        if not results.multi_hand_landmarks:
            return None  # Return None when no hands detected
        
        # Extract landmarks from detected hands
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            all_landmarks.extend(self.extract_landmarks(hand_landmarks))
        
        # Pad or truncate to 126 features
        while len(all_landmarks) < 126:
            all_landmarks.append(0.0)
        
        return all_landmarks[:126]
    
    def normalize_sequence(self, sequence):
        """Normalize sequence to 90 frames using interpolation."""
        current_length = len(sequence)
        target_length = 90
        
        if current_length == target_length:
            return sequence
        
        # Convert to numpy array
        sequence = np.array(sequence)
        
        # Create indices for interpolation
        old_indices = np.linspace(0, current_length - 1, current_length)
        new_indices = np.linspace(0, current_length - 1, target_length)
        
        # Interpolate each feature
        normalized = np.zeros((target_length, 126))
        for i in range(126):
            normalized[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
        
        return normalized.tolist()
    
    def predict_sequence(self):
        """Predict phrase from recorded sequence."""
        if len(self.recorded_sequence) < MIN_FRAMES:
            return "Need more frames", 0.0
        
        # Normalize to 90 frames
        sequence = self.normalize_sequence(self.recorded_sequence)
        sequence = np.array(sequence)  # Shape: (90, 126)
        
        # Reshape for scaling
        sequence_flat = sequence.reshape(-1, 126)
        
        # Scale features
        sequence_scaled = self.scaler.transform(sequence_flat)
        
        # Reshape for model: (1, 90, 126)
        sequence_scaled = sequence_scaled.reshape(1, 90, 126)
        
        # Predict
        predictions = self.model.predict(sequence_scaled, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        phrase = self.phrase_mapping.get(int(predicted_class), "Unknown")
        
        return phrase, confidence
    
    def start_recording(self):
        """Start recording a new sequence."""
        self.is_recording = True
        self.recorded_sequence = []
        self.last_prediction = None
        self.last_confidence = 0.0
        print("\n🔴 Recording started...")
    
    def stop_recording(self):
        """Stop recording and process the sequence."""
        self.is_recording = False
        print(f"⏹️  Recording stopped. Captured {len(self.recorded_sequence)} frames")
        
        if len(self.recorded_sequence) >= MIN_FRAMES:
            print("🤖 Processing...")
            self.last_prediction, self.last_confidence = self.predict_sequence()
            print(f"✓ Prediction: {self.last_prediction} ({self.last_confidence:.1%})")
        else:
            print(f"⚠️  Need at least {MIN_FRAMES} frames for prediction")
            self.last_prediction = f"Too short (only {len(self.recorded_sequence)} frames)"
            self.last_confidence = 0.0
    
    def clear_prediction(self):
        """Clear the current prediction."""
        self.last_prediction = None
        self.last_confidence = 0.0
        print("🗑️  Prediction cleared")
    
    def run(self):
        """Start live prediction interface."""
        print("\n" + "="*70)
        print("Smart ISL Recognition - User Controlled")
        print("="*70)
        print("\n📝 Instructions:")
        print("  SPACE = Start/Stop recording your gesture")
        print("  C = Clear last prediction")
        print("  Q = Quit")
        print("\n💡 Tips:")
        print("  - Press SPACE to start recording")
        print("  - Perform your gesture (take your time!)")
        print("  - Press SPACE again when done")
        print("  - App will process and show the prediction")
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
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Track hand presence
                hands_detected = results.multi_hand_landmarks is not None
                
                if hands_detected:
                    self.frames_with_hands += 1
                    self.frames_without_hands = 0
                else:
                    self.frames_without_hands += 1
                    self.frames_with_hands = 0
                
                # Draw hand landmarks
                if hands_detected:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                
                # Record frames when recording is active AND hands are detected
                if self.is_recording and hands_detected:
                    frame_landmarks = self.process_frame_landmarks(results)
                    if frame_landmarks is not None:
                        self.recorded_sequence.append(frame_landmarks)
                        # Limit maximum frames
                        if len(self.recorded_sequence) > MAX_FRAMES:
                            self.recorded_sequence.pop(0)
                
                # === UI OVERLAY ===
                overlay = frame.copy()
                
                # Status bar background
                if self.is_recording:
                    # Red background when recording
                    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 180), -1)
                else:
                    # Dark background when idle
                    cv2.rectangle(overlay, (0, 0), (w, 100), (40, 40, 40), -1)
                
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Status text
                if self.is_recording:
                    status_text = "🔴 RECORDING"
                    status_color = (255, 255, 255)
                    frame_count = f"Frames: {len(self.recorded_sequence)}"
                    
                    cv2.putText(frame, status_text, (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                    cv2.putText(frame, frame_count, (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Recording indicator (pulsing dot)
                    import time
                    if int(time.time() * 2) % 2:  # Blink every 0.5s
                        cv2.circle(frame, (w - 40, 40), 15, (0, 0, 255), -1)
                
                else:
                    status_text = "Ready - Press SPACE to record"
                    status_color = (200, 200, 200)
                    
                    cv2.putText(frame, status_text, (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                # Hand detection indicator
                if hands_detected:
                    cv2.circle(frame, (w - 40, 40), 15, (0, 255, 0), -1)
                    cv2.putText(frame, "Hands detected", (w - 180, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Result box (if prediction exists)
                if self.last_prediction:
                    result_y_start = h - 180
                    
                    # Result background
                    if self.last_confidence > 0.7:
                        bg_color = (0, 100, 0)  # Green
                    elif self.last_confidence > 0.5:
                        bg_color = (0, 100, 100)  # Yellow
                    else:
                        bg_color = (0, 0, 100)  # Red
                    
                    cv2.rectangle(frame, (10, result_y_start), (w - 10, h - 10), bg_color, -1)
                    cv2.rectangle(frame, (10, result_y_start), (w - 10, h - 10), (255, 255, 255), 2)
                    
                    # Prediction text
                    cv2.putText(frame, "Recognized:", (25, result_y_start + 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Phrase (wrapped if too long)
                    phrase_text = self.last_prediction
                    cv2.putText(frame, phrase_text, (25, result_y_start + 75), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                    # Confidence
                    if self.last_confidence > 0:
                        conf_text = f"Confidence: {self.last_confidence:.1%}"
                        cv2.putText(frame, conf_text, (25, result_y_start + 115), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Clear instruction
                    cv2.putText(frame, "Press C to clear", (25, result_y_start + 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Controls help (bottom)
                help_y = h - 35 if not self.last_prediction else 30
                cv2.putText(frame, "SPACE=Record | C=Clear | Q=Quit", 
                           (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Smart ISL Recognition', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space bar
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                
                elif key == ord('c') or key == ord('C'):
                    self.clear_prediction()
                
                elif key == ord('q') or key == ord('Q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Application closed.")


if __name__ == "__main__":
    try:
        predictor = SmartPredictor()
        predictor.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
