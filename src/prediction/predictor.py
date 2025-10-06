"""
Live gesture prediction module.
Loads trained model and predicts ISL phrases from webcam feed.
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import glob


MODEL_DIR = "models/saved"


class GesturePredictor:
    """Real-time gesture prediction using trained model."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.model = None
        self.scaler = None
        self.phrase_mapping = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model, scaler, and phrase mapping."""
        # Find the model file
        model_files = glob.glob(os.path.join(MODEL_DIR, "*_model.joblib"))
        
        if not model_files:
            raise FileNotFoundError(f"No model found in {MODEL_DIR}")
        
        model_path = model_files[0]
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.joblib")
        
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        
        print(f"Loading scaler from: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        print(f"Loading phrase mapping from: {mapping_path}")
        self.phrase_mapping = joblib.load(mapping_path)
        
        print("Model loaded successfully!")
        print(f"Recognized phrases: {list(self.phrase_mapping.values())}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from Mediapipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def predict(self, landmarks):
        """Predict phrase from landmarks."""
        # Pad or truncate to fixed size (126 features)
        while len(landmarks) < 126:
            landmarks.append(0.0)
        landmarks = landmarks[:126]
        
        # Reshape and scale
        X = np.array(landmarks).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 1.0
        
        phrase = self.phrase_mapping.get(int(prediction), "Unknown")
        
        return phrase, confidence
    
    def run(self):
        """Start live prediction from webcam."""
        print("\nStarting live prediction...")
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            current_phrase = "No gesture detected"
            confidence = 0.0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Draw landmarks and predict
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Extract landmarks and predict
                    all_landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        all_landmarks.extend(self.extract_landmarks(hand_landmarks))
                    
                    current_phrase, confidence = self.predict(all_landmarks)
                else:
                    current_phrase = "No gesture detected"
                    confidence = 0.0
                
                # Display prediction
                # Background for text
                cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 120), (0, 0, 0), -1)
                
                # Phrase
                cv2.putText(
                    frame, 
                    f'Phrase: {current_phrase}', 
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                
                # Confidence
                if confidence > 0:
                    cv2.putText(
                        frame, 
                        f'Confidence: {confidence:.2%}', 
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 0), 
                        2
                    )
                
                # Instructions
                cv2.putText(
                    frame, 
                    'Press Q to quit', 
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
                
                cv2.imshow('ISL Gesture Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Prediction stopped.")


if __name__ == "__main__":
    predictor = GesturePredictor()
    predictor.run()
