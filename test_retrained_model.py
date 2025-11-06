#!/usr/bin/env python3
"""
Test the newly retrained model
"""

import sys
import os
import time
import json
from datetime import datetime
from collections import deque

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_retrained_model():
    """Test the newly retrained model."""
    print("🎯 Testing Newly Retrained Model")
    print("="*50)
    
    # We need to use the LSTM model now since we retrained that one
    print("Note: Using retrained LSTM model (not transformer)")
    
    # Import inference for proper preprocessing
    import cv2
    import numpy as np
    import tensorflow as tf
    import joblib
    import json
    from inference import HolisticInference
    
    # Load the retrained LSTM model
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    print("✅ Retrained LSTM model loaded")
    
    # Initialize HolisticInference for proper preprocessing
    inference = HolisticInference()
    
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
    for i, phrase in phrase_options.items():
        print(f"  {i}: {phrase}")
    
    gesture_choice = input("\nWhich phrase will you perform? (0-4): ")
    
    try:
        gesture_idx = int(gesture_choice)
        expected_phrase = phrase_options[gesture_idx]
        print(f"\n👤 You will perform: {expected_phrase}")
    except:
        expected_phrase = "unknown"
        print(f"\n👤 You will perform: unknown gesture")
    
    input("Press Enter when ready to start...")
    
    print("\n🎬 Recording for 10 seconds... perform your gesture!")
    
    sequence_buffer = deque(maxlen=60)
    predictions = []
    start_time = time.time()
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Process frame using HolisticInference (same as training)
        landmarks, _ = inference.process_frame(frame)
        sequence_buffer.append(landmarks)
        
        # Try prediction every 2 seconds
        elapsed = time.time() - start_time
        if len(sequence_buffer) == 60 and int(elapsed) % 2 == 0:
            # Prepare sequence for model
            sequence = np.array(list(sequence_buffer))
            sequence_flat = sequence.flatten().reshape(1, -1)
            sequence_scaled = scaler.transform(sequence_flat)
            sequence_lstm = sequence_scaled.reshape(1, 60, 1662)
            
            # Predict
            pred_probs = model.predict(sequence_lstm, verbose=0)
            predicted_class = np.argmax(pred_probs[0])
            confidence = np.max(pred_probs[0])
            
            phrase = phrase_mapping[str(predicted_class)]
            
            prediction_entry = {
                'time': elapsed,
                'predicted_phrase': phrase,
                'confidence': confidence,
                'expected_phrase': expected_phrase
            }
            predictions.append(prediction_entry)
            print(f"[{elapsed:.1f}s] Predicted: {phrase} (confidence: {confidence:.3f})")
        
        # Show frame
        cv2.imshow('Testing Retrained Model', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Results
    print("\n📊 RESULTS:")
    print("="*50)
    
    if not predictions:
        print("❌ No predictions made")
    else:
        print(f"Expected: {expected_phrase}")
        print("Predictions:")
        
        correct_predictions = 0
        for pred in predictions:
            is_correct = pred['predicted_phrase'] == pred['expected_phrase']
            correct_predictions += is_correct
            status = "✅" if is_correct else "❌"
            print(f"  {status} {pred['time']:.1f}s: {pred['predicted_phrase']} ({pred['confidence']:.3f})")
        
        accuracy = correct_predictions / len(predictions) * 100
        print(f"\nAccuracy: {correct_predictions}/{len(predictions)} ({accuracy:.1f}%)")
        
        # Show prediction distribution
        pred_counts = {}
        for pred in predictions:
            phrase = pred['predicted_phrase']
            pred_counts[phrase] = pred_counts.get(phrase, 0) + 1
        
        print(f"\nPrediction distribution:")
        for phrase, count in pred_counts.items():
            percentage = count / len(predictions) * 100
            print(f"  {phrase}: {count} times ({percentage:.1f}%)")
        
        if accuracy >= 80:
            print("🎉 EXCELLENT! Retraining was successful!")
        elif accuracy >= 60:
            print("👍 GOOD! Much better than before!")
        else:
            print("⚠️  Still some issues, but should be improved")

if __name__ == "__main__":
    test_retrained_model()