#!/usr/bin/env python3
"""
Debug confidence issues with the retrained model
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_model_confidence():
    """Test model confidence on original training data"""
    
    print("🔍 Testing Model Confidence Issues")
    print("=" * 50)
    
    # Load model and data
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    phrases = list(phrase_mapping.keys())
    print(f"📝 Phrases: {phrases}")
    
    # Test on original training data
    data_dir = "data/sequences"
    
    print("\n📊 Testing on Original Training Data:")
    print("-" * 40)
    
    total_correct = 0
    total_samples = 0
    confidence_threshold = 0.7
    
    for phrase_idx, phrase in enumerate(phrases):
        phrase_dir = os.path.join(data_dir, f"phrase_{phrase_idx}")
        if not os.path.exists(phrase_dir):
            continue
            
        print(f"\n🎯 Testing phrase: {phrase}")
        
        phrase_correct = 0
        phrase_total = 0
        confidences = []
        
        # Test first 5 samples
        for file in sorted(os.listdir(phrase_dir))[:5]:
            if file.endswith('_seq.npy'):
                filepath = os.path.join(phrase_dir, file)
                sequence = np.load(filepath)
                
                # Ensure proper shape (60, 1662)
                if sequence.shape[0] != 60:
                    if sequence.shape[0] > 60:
                        sequence = sequence[:60]
                    else:
                        # Pad with last frame
                        last_frame = sequence[-1:] if len(sequence) > 0 else np.zeros((1, sequence.shape[1]))
                        padding = np.repeat(last_frame, 60 - sequence.shape[0], axis=0)
                        sequence = np.vstack([sequence, padding])
                
                # Flatten and scale
                sequence_flat = sequence.reshape(1, -1)
                sequence_scaled = scaler.transform(sequence_flat)
                sequence_reshaped = sequence_scaled.reshape(1, 60, 1662)
                
                # Predict
                predictions = model.predict(sequence_reshaped, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_idx]
                
                is_correct = predicted_idx == phrase_idx
                phrase_correct += is_correct
                phrase_total += 1
                confidences.append(confidence)
                
                status = "✅" if is_correct else "❌"
                conf_status = "🔥" if confidence >= confidence_threshold else "⚠️"
                
                print(f"  {status} {conf_status} {file}: {phrases[predicted_idx]} ({confidence:.3f})")
        
        if phrase_total > 0:
            accuracy = phrase_correct / phrase_total
            avg_confidence = np.mean(confidences)
            
            print(f"  📈 Accuracy: {phrase_correct}/{phrase_total} ({accuracy:.1%})")
            print(f"  📊 Avg Confidence: {avg_confidence:.3f}")
            
            total_correct += phrase_correct
            total_samples += phrase_total
    
    if total_samples > 0:
        overall_accuracy = total_correct / total_samples
        print(f"\n🎯 Overall Accuracy: {total_correct}/{total_samples} ({overall_accuracy:.1%})")
    
    # Test confidence distribution
    print(f"\n📊 Confidence Analysis:")
    print(f"   🔥 High confidence (≥{confidence_threshold}): Good predictions")
    print(f"   ⚠️  Low confidence (<{confidence_threshold}): Uncertain predictions")
    
    return total_correct, total_samples

def check_model_architecture():
    """Check if model architecture is reasonable"""
    
    print("\n🏗️ Model Architecture Analysis:")
    print("-" * 40)
    
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    model.summary()
    
    # Check if model is too complex (overfitting) or too simple (underfitting)
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")
    
    if total_params > 1000000:
        print("⚠️  Model might be too complex (risk of overfitting)")
    elif total_params < 10000:
        print("⚠️  Model might be too simple (risk of underfitting)")
    else:
        print("✅ Model complexity seems reasonable")

if __name__ == "__main__":
    correct, total = test_model_confidence()
    check_model_architecture()
    
    if correct / total < 0.8:
        print("\n⚠️  LOW PERFORMANCE DETECTED!")
        print("💡 Suggestions:")
        print("   1. Model might need more training on your specific data")
        print("   2. Multi-person data might have diluted performance")
        print("   3. Consider training a personalized model")
        print("   4. Check if preprocessing is consistent")
    else:
        print("\n✅ Performance looks reasonable!")