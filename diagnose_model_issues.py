#!/usr/bin/env python3
"""
Comprehensive model diagnosis to understand the prediction bias issue
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import json
import cv2
import mediapipe as mp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class SimpleHandExtractor:
    """Simple hand landmark extractor for 126 features (original format)"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
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
                frame_landmarks.extend(self.extract_landmarks(hand_landmarks))
        
        # Pad or truncate to fixed size (2 hands * 21 landmarks * 3 coords = 126)
        while len(frame_landmarks) < 126:
            frame_landmarks.append(0.0)
        frame_landmarks = frame_landmarks[:126]
        
        return np.array(frame_landmarks)

def diagnose_model_predictions():
    """Comprehensive diagnosis of model predictions"""
    
    print("🔍 COMPREHENSIVE MODEL DIAGNOSIS")
    print("=" * 60)
    
    # Load model and data
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    phrases = list(phrase_mapping.keys())
    print(f"📝 Phrases: {phrases}")
    
    # Test on ALL original training data
    print("\n📊 Testing on ALL Original Training Data:")
    print("-" * 50)
    
    data_dir = "data/sequences"
    
    # Collect all predictions per phrase
    phrase_predictions = {i: [] for i in range(5)}
    phrase_confidences = {i: [] for i in range(5)}
    
    for phrase_idx in range(5):
        phrase_dir = os.path.join(data_dir, f"phrase_{phrase_idx}")
        if not os.path.exists(phrase_dir):
            continue
            
        print(f"\n🎯 Testing phrase {phrase_idx}: {phrases[phrase_idx]}")
        
        files = [f for f in os.listdir(phrase_dir) if f.endswith('_seq.npy')]
        print(f"   Found {len(files)} samples")
        
        correct_predictions = 0
        total_predictions = 0
        
        for file in files:
            filepath = os.path.join(phrase_dir, file)
            sequence = np.load(filepath)
            
            # Ensure proper shape (60, 126)
            if sequence.shape[0] != 60:
                if sequence.shape[0] > 60:
                    sequence = sequence[:60]
                else:
                    last_frame = sequence[-1:] if len(sequence) > 0 else np.zeros((1, sequence.shape[1]))
                    padding = np.repeat(last_frame, 60 - sequence.shape[0], axis=0)
                    sequence = np.vstack([sequence, padding])
            
            # Flatten and scale
            sequence_flat = sequence.reshape(1, -1)
            sequence_scaled = scaler.transform(sequence_flat)
            sequence_reshaped = sequence_scaled.reshape(1, 60, 126)
            
            # Predict
            predictions = model.predict(sequence_reshaped, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            
            phrase_predictions[phrase_idx].append(predicted_idx)
            phrase_confidences[phrase_idx].append(confidence)
            
            if predicted_idx == phrase_idx:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_confidence = np.mean(phrase_confidences[phrase_idx]) if phrase_confidences[phrase_idx] else 0
        
        print(f"   📈 Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
        print(f"   📊 Avg Confidence: {avg_confidence:.3f}")
        
        # Show prediction distribution
        prediction_counts = {}
        for pred in phrase_predictions[phrase_idx]:
            pred_phrase = phrases[pred]
            prediction_counts[pred_phrase] = prediction_counts.get(pred_phrase, 0) + 1
        
        print(f"   📋 Predictions:")
        for pred_phrase, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(phrase_predictions[phrase_idx]) * 100
            print(f"      {pred_phrase}: {count} times ({percentage:.1f}%)")

def check_model_bias():
    """Check if model is biased towards specific classes"""
    
    print("\n🧠 MODEL BIAS ANALYSIS:")
    print("-" * 50)
    
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    
    # Check final layer weights
    final_layer = model.layers[-1]
    weights = final_layer.get_weights()
    
    if len(weights) > 0:
        bias = weights[1] if len(weights) > 1 else None
        print(f"Final layer bias: {bias}")
        
        if bias is not None:
            print("Bias analysis:")
            phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
            for i, phrase in enumerate(phrases):
                print(f"  {phrase}: {bias[i]:.4f}")
            
            # Check if heavily biased
            max_bias_idx = np.argmax(bias)
            min_bias_idx = np.argmin(bias)
            bias_range = bias[max_bias_idx] - bias[min_bias_idx]
            
            print(f"\nBias range: {bias_range:.4f}")
            print(f"Most biased towards: {phrases[max_bias_idx]}")
            print(f"Least biased towards: {phrases[min_bias_idx]}")
            
            if bias_range > 2.0:
                print("⚠️  HIGH BIAS DETECTED - Model heavily favors certain classes!")
                return True
    
    return False

def test_random_noise():
    """Test model on random noise to see default prediction"""
    
    print("\n🎲 RANDOM NOISE TEST:")
    print("-" * 50)
    
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    phrases = list(phrase_mapping.keys())
    
    # Generate random sequences
    noise_predictions = []
    
    for i in range(10):
        # Generate random sequence (60, 126)
        random_sequence = np.random.random((60, 126))
        
        # Process like real data
        sequence_flat = random_sequence.reshape(1, -1)
        sequence_scaled = scaler.transform(sequence_flat)
        sequence_reshaped = sequence_scaled.reshape(1, 60, 126)
        
        # Predict
        predictions = model.predict(sequence_reshaped, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        noise_predictions.append(predicted_idx)
        print(f"Random noise {i+1}: {phrases[predicted_idx]} (confidence: {confidence:.3f})")
    
    # Check if model defaults to specific class
    prediction_counts = {}
    for pred in noise_predictions:
        pred_phrase = phrases[pred]
        prediction_counts[pred_phrase] = prediction_counts.get(pred_phrase, 0) + 1
    
    print(f"\nRandom noise predictions:")
    for phrase, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(noise_predictions) * 100
        print(f"  {phrase}: {count}/10 times ({percentage:.0f}%)")
    
    # If one class dominates random predictions, model is biased
    max_count = max(prediction_counts.values())
    if max_count >= 7:  # 70% or more
        dominant_class = [k for k, v in prediction_counts.items() if v == max_count][0]
        print(f"⚠️  MODEL DEFAULTS TO: {dominant_class}")
        print("This indicates severe bias - model learned to always predict this class!")
        return True
    
    return False

def main():
    print("Starting comprehensive model diagnosis...")
    
    # Test 1: Training data performance
    diagnose_model_predictions()
    
    # Test 2: Check model bias
    bias_detected = check_model_bias()
    
    # Test 3: Random noise test
    noise_bias = test_random_noise()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS SUMMARY:")
    
    if bias_detected or noise_bias:
        print("❌ SERIOUS MODEL ISSUES DETECTED:")
        if bias_detected:
            print("  - High bias in final layer weights")
        if noise_bias:
            print("  - Model defaults to specific class on random input")
        print("\n💡 RECOMMENDED SOLUTIONS:")
        print("  1. Retrain with balanced class weights")
        print("  2. Use simpler model architecture") 
        print("  3. Add more regularization")
        print("  4. Check training data quality")
    else:
        print("✅ Model appears to be functioning normally")
        print("   Issue might be in real-time preprocessing pipeline")

if __name__ == "__main__":
    main()