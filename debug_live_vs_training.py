#!/usr/bin/env python3
"""
Debug Live vs Training Data Processing
Compares live camera sequences with training sequences to identify preprocessing differences
"""

import cv2
import numpy as np
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add current directory to path to import inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import HolisticInference

def collect_live_sequence(phrase_name, num_frames=60):
    """Collect a live sequence using the EXACT same process as training"""
    print(f"\n🎥 Collecting live sequence for: {phrase_name}")
    print("Press 's' to start recording, 'q' to quit")
    
    # Initialize HolisticInference (same as training)
    inference = HolisticInference()
    cap = cv2.VideoCapture(0)
    sequence = []
    recording = False
    
    try:
        while len(sequence) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Show recording status
            if recording:
                cv2.putText(frame, f'RECORDING: {len(sequence)}/{num_frames}', (15, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Process frame using HolisticInference (exactly like training)
                landmarks, _ = inference.process_frame(frame)  # Unpack tuple
                sequence.append(landmarks)
            else:
                cv2.putText(frame, f'Press "s" to start recording {phrase_name}', (15, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Live Sequence Collection', frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s') and not recording:
                recording = True
                sequence = []
                print(f"🔴 Recording started for {phrase_name}...")
            elif key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # HolisticInference doesn't have a close method
    
    if len(sequence) == num_frames:
        return np.array(sequence)
    else:
        print(f"⚠️ Only collected {len(sequence)} frames out of {num_frames}")
        return None

def load_training_sample(phrase_idx, sample_name="current_env_sample_1.npy"):
    """Load a training sample"""
    # First try the current environment sequences (new training data)
    current_env_path = f"data/sequences_current_env/phrase_{phrase_idx}/{sample_name}"
    if os.path.exists(current_env_path):
        return np.load(current_env_path)
    
    # Fallback to old sequences (but these have different dimensions)
    sample_path = f"data/sequences/phrase_{phrase_idx}/{sample_name}"
    if os.path.exists(sample_path):
        return np.load(sample_path)
    
    # If current_env sample doesn't exist, try any available sample
    phrase_dir = f"data/sequences/phrase_{phrase_idx}"
    if os.path.exists(phrase_dir):
        files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
        if files:
            sample_path = os.path.join(phrase_dir, files[0])
            print(f"Using alternative training sample: {files[0]}")
            return np.load(sample_path)
    
    return None

def compare_sequences(live_seq, training_seq, phrase_name):
    """Compare live and training sequences"""
    print(f"\n🔍 Comparing sequences for: {phrase_name}")
    print(f"Live sequence shape: {live_seq.shape}")
    print(f"Training sequence shape: {training_seq.shape}")
    
    # Basic statistics
    print(f"\nLive sequence stats:")
    print(f"  Mean: {np.mean(live_seq):.6f}")
    print(f"  Std: {np.std(live_seq):.6f}")
    print(f"  Min: {np.min(live_seq):.6f}")
    print(f"  Max: {np.max(live_seq):.6f}")
    
    print(f"\nTraining sequence stats:")
    print(f"  Mean: {np.mean(training_seq):.6f}")
    print(f"  Std: {np.std(training_seq):.6f}")
    print(f"  Min: {np.min(training_seq):.6f}")
    print(f"  Max: {np.max(training_seq):.6f}")
    
    # Difference analysis
    if live_seq.shape == training_seq.shape:
        diff = np.abs(live_seq - training_seq)
        print(f"\nDifference stats:")
        print(f"  Mean absolute difference: {np.mean(diff):.6f}")
        print(f"  Max absolute difference: {np.max(diff):.6f}")
        print(f"  Std of differences: {np.std(diff):.6f}")
        
        # Check for extreme differences
        large_diff_indices = np.where(diff > 0.1)[0]
        if len(large_diff_indices) > 0:
            print(f"  Found {len(large_diff_indices)} points with difference > 0.1")
            print(f"  First few large difference indices: {large_diff_indices[:10]}")
    
    return live_seq, training_seq

def test_with_model(sequence, scaler, model, phrase_mapping):
    """Test a sequence with the model"""
    print(f"Input sequence shape: {sequence.shape}")
    
    # Check if we need to reshape the sequence to match model expectations
    expected_frames = 60  # Updated to match model
    expected_features = 1662
    
    if sequence.shape[0] != expected_frames:
        if sequence.shape[0] > expected_frames:
            # Take first 60 frames
            sequence = sequence[:expected_frames]
            print(f"Truncated to {expected_frames} frames: {sequence.shape}")
        else:
            # Pad with zeros or repeat frames
            padding_needed = expected_frames - sequence.shape[0]
            padding = np.zeros((padding_needed, sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
            print(f"Padded to {expected_frames} frames: {sequence.shape}")
    
    if sequence.shape[1] != expected_features:
        print(f"⚠️ Feature dimension mismatch: got {sequence.shape[1]}, expected {expected_features}")
        return None, None, None
    
    # Scale the sequence
    scaled_sequence = scaler.transform(sequence.reshape(1, -1)).reshape(sequence.shape)
    
    # Predict
    prediction = model.predict(scaled_sequence.reshape(1, expected_frames, expected_features), verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    # Get phrase name
    phrase_name = phrase_mapping[str(predicted_class)]
    
    print(f"\nPrediction: {phrase_name} (confidence: {confidence:.3f})")
    print("All probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"  {i} ({phrase_mapping[str(i)]}): {prob:.3f}")
    
    return predicted_class, confidence, prediction[0]

def main():
    """Main debugging function"""
    print("🔧 Live vs Training Data Debugging")
    print("==================================")
    
    # Load model components
    try:
        from tensorflow import keras
        model = keras.models.load_model('models/saved/lstm_model.keras')
        scaler = joblib.load('models/saved/sequence_scaler.joblib')
        with open('models/saved/phrase_mapping.json', 'r') as f:
            phrase_mapping = json.load(f)
        print("✅ Model components loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test phrase (the one that's failing)
    test_phrase_idx = 1  # "How are you"
    test_phrase_name = phrase_mapping[str(test_phrase_idx)]
    
    print(f"\n🎯 Testing phrase: {test_phrase_name}")
    
    # Load training sample
    training_seq = load_training_sample(test_phrase_idx, "current_env_sample_1.npy")
    if training_seq is None:
        print("❌ Could not load training sample")
        return
    
    print(f"✅ Loaded training sample: {training_seq.shape}")
    
    # Test training sample with model
    print(f"\n🧪 Testing training sample with model:")
    test_with_model(training_seq, scaler, model, phrase_mapping)
    
    # Collect live sequence
    live_seq = collect_live_sequence(test_phrase_name)
    if live_seq is None:
        print("❌ Could not collect live sequence")
        return
    
    print(f"✅ Collected live sequence: {live_seq.shape}")
    
    # Test live sequence with model
    print(f"\n🧪 Testing live sequence with model:")
    test_with_model(live_seq, scaler, model, phrase_mapping)
    
    # Compare sequences
    compare_sequences(live_seq, training_seq, test_phrase_name)
    
    # Save sequences for further analysis
    np.save("debug_live_sequence.npy", live_seq)
    np.save("debug_training_sequence.npy", training_seq)
    print(f"\n💾 Sequences saved for further analysis")
    print("  - debug_live_sequence.npy")
    print("  - debug_training_sequence.npy")

if __name__ == "__main__":
    main()