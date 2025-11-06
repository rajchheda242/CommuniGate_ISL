#!/usr/bin/env python3
"""
Test the model on original training data to see if it's broken
"""

import numpy as np
import tensorflow as tf
import json
import os
from sklearn.preprocessing import StandardScaler
import joblib

def test_model_on_training_data():
    """Test if the model works on the original training data"""
    
    print("🧪 Testing model on original training data...")
    
    # Load model
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    print("✅ Model and scaler loaded")
    
    # Test a few samples from each phrase
    all_correct = 0
    all_total = 0
    
    for phrase_idx in range(5):
        phrase_name = phrase_mapping[str(phrase_idx)]
        print(f"\n📂 Testing phrase {phrase_idx}: {phrase_name}")
        
        # Get all files for this phrase
        data_dir = f"data/sequences/phrase_{phrase_idx}"
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Test first 5 files
        correct = 0
        total = 0
        
        for i, filename in enumerate(files[:5]):
            filepath = os.path.join(data_dir, filename)
            
            try:
                # Load sequence
                sequence = np.load(filepath)
                
                # Reshape and scale like in inference
                sequence_reshaped = sequence.reshape(1, -1)
                sequence_scaled = scaler.transform(sequence_reshaped)
                sequence_final = sequence_scaled.reshape(1, 60, -1)
                
                # Predict
                predictions = model.predict(sequence_final, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                is_correct = predicted_class == phrase_idx
                if is_correct:
                    correct += 1
                    status = "✅"
                else:
                    predicted_phrase = phrase_mapping[str(predicted_class)]
                    status = f"❌ -> {predicted_phrase}"
                
                total += 1
                all_correct += is_correct
                all_total += 1
                
                print(f"  {status} {filename}: confidence {confidence:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error with {filename}: {e}")
        
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"  Phrase {phrase_idx} accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    overall_accuracy = all_correct / all_total * 100 if all_total > 0 else 0
    print(f"\n🎯 Overall training data accuracy: {all_correct}/{all_total} ({overall_accuracy:.1f}%)")
    
    if overall_accuracy < 80:
        print("🚨 Model performs poorly even on training data!")
        print("This suggests the model wasn't trained properly.")
    else:
        print("✅ Model works well on training data.")
        print("The issue is likely with live data preprocessing or capture.")

if __name__ == "__main__":
    test_model_on_training_data()