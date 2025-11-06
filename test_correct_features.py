#!/usr/bin/env python3
"""
Test the newly retrained model with correct 126-feature extraction
"""

import sys
import os
import time
import json
import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
from datetime import datetime
from collections import deque

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SimpleHandExtractor:
    """Simple hand landmark extractor for 126 features (original format)"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from Mediapipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def process_frame(self, frame):
        """Process a frame and extract 126 hand features"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Extract landmarks
        frame_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks for visual feedback
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                frame_landmarks.extend(self.extract_landmarks(hand_landmarks))
        
        # Pad or truncate to fixed size (2 hands * 21 landmarks * 3 coords = 126)
        while len(frame_landmarks) < 126:
            frame_landmarks.append(0.0)
        frame_landmarks = frame_landmarks[:126]
        
        return np.array(frame_landmarks), frame

def test_retrained_model():
    """Test the newly retrained model with 126 features."""
    print("🎯 Testing Model with Correct 126 Features")
    print("="*50)
    
    # Load the retrained LSTM model (126 features)
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"✓ Recognized phrases: {list(phrase_mapping.keys())}")
    print(f"✓ Sequence length: 60 frames")
    
    # Initialize hand extractor
    extractor = SimpleHandExtractor()
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    phrase_options = {
        0: "Hi my name is Reet",
        1: "How are you", 
        2: "I am from Delhi",
        3: "I like coffee",
        4: "What do you like"
    }
    
    print("\nAvailable phrases:")
    for key, phrase in phrase_options.items():
        print(f"  {key}: {phrase}")
    
    try:
        phrase_choice = int(input("\nWhich phrase will you perform? (0-4): "))
        if phrase_choice not in phrase_options:
            print("Invalid choice!")
            return
    except ValueError:
        print("Invalid input!")
        return
    
    selected_phrase = phrase_options[phrase_choice]
    print(f"\n👤 You will perform: {selected_phrase}")
    input("Press Enter when ready to start...")
    
    # Record sequence
    sequence = []
    predictions = []
    start_time = time.time()
    recording_duration = 10  # seconds
    
    print(f"\n🎬 Recording for {recording_duration} seconds... perform your gesture!")
    
    while time.time() - start_time < recording_duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Extract features (126 features)
        features, annotated_frame = extractor.process_frame(frame)
        sequence.append(features)
        
        # Keep sequence at 60 frames
        if len(sequence) > 60:
            sequence = sequence[-60:]
        
        # Make prediction if we have enough frames
        if len(sequence) == 60:
            # Prepare sequence for prediction
            sequence_array = np.array(sequence)
            
            # Flatten for scaler (60 * 126 = 7560)
            sequence_flat = sequence_array.reshape(1, -1)
            
            # Scale
            sequence_scaled = scaler.transform(sequence_flat)
            
            # Reshape back to (1, 60, 126)
            sequence_reshaped = sequence_scaled.reshape(1, 60, 126)
            
            # Predict
            prediction = model.predict(sequence_reshaped, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Only show predictions with reasonable confidence
            if confidence > 0.3:
                predicted_phrase = list(phrase_mapping.keys())[predicted_class]
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Predicted: {predicted_phrase} (confidence: {confidence:.3f})")
                predictions.append((elapsed, predicted_phrase, confidence))
        
        # Display frame
        elapsed = time.time() - start_time
        remaining = recording_duration - elapsed
        cv2.putText(
            annotated_frame,
            f"Recording: {remaining:.1f}s remaining",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            annotated_frame,
            f"Phrase: {selected_phrase}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow('ISL Recognition Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analyze results
    print(f"\n📊 RESULTS:")
    print("="*50)
    print(f"Expected: {selected_phrase}")
    
    if predictions:
        print("Predictions:")
        correct_count = 0
        for elapsed, predicted, confidence in predictions:
            is_correct = predicted == selected_phrase
            status = "✅" if is_correct else "❌"
            print(f"  {status} {elapsed:.1f}s: {predicted} ({confidence:.3f})")
            if is_correct:
                correct_count += 1
        
        accuracy = correct_count / len(predictions)
        print(f"\nAccuracy: {correct_count}/{len(predictions)} ({accuracy:.1%})")
        
        # Count prediction distribution
        prediction_counts = {}
        for _, predicted, _ in predictions:
            prediction_counts[predicted] = prediction_counts.get(predicted, 0) + 1
        
        print(f"\nPrediction distribution:")
        for phrase, count in prediction_counts.items():
            percentage = count / len(predictions) * 100
            print(f"  {phrase}: {count} times ({percentage:.1f}%)")
        
        if accuracy >= 0.8:
            print("🎉 EXCELLENT! Model is working well!")
        elif accuracy >= 0.6:
            print("✅ Good performance - model is learning!")
        else:
            print("⚠️  Still some issues, but improved from before")
    else:
        print("❌ No predictions made - check if gestures are visible")

if __name__ == "__main__":
    test_retrained_model()