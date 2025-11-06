#!/usr/bin/env python3
"""
Debug the retrained model to see what's wrong
"""

import numpy as np
import tensorflow as tf
import joblib
import json
import os

def debug_retrained_model():
    """Debug the retrained model to understand the bias."""
    
    print("🔍 Debugging Retrained Model")
    print("="*50)
    
    # Load model
    model = tf.keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    print("✅ Model and components loaded")
    
    # Test the model on the newly collected training data
    print("\n🧪 Testing model on newly collected training data...")
    
    new_data_dir = "data/sequences_current_env"
    
    all_correct = 0
    all_total = 0
    
    for phrase_idx in range(5):
        phrase_dir = os.path.join(new_data_dir, f"phrase_{phrase_idx}")
        phrase_name = phrase_mapping[str(phrase_idx)]
        
        print(f"\n📂 Testing phrase {phrase_idx}: {phrase_name}")
        
        if not os.path.exists(phrase_dir):
            print(f"❌ Directory not found: {phrase_dir}")
            continue
        
        files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
        
        correct = 0
        total = 0
        
        for filename in files:
            filepath = os.path.join(phrase_dir, filename)
            
            try:
                # Load sequence
                sequence = np.load(filepath)
                
                # Process same as in inference
                sequence_flat = sequence.flatten().reshape(1, -1)
                sequence_scaled = scaler.transform(sequence_flat)
                sequence_lstm = sequence_scaled.reshape(1, 60, 1662)
                
                # Predict
                pred_probs = model.predict(sequence_lstm, verbose=0)
                predicted_class = np.argmax(pred_probs[0])
                confidence = np.max(pred_probs[0])
                
                # Show all class probabilities
                print(f"  {filename}:")
                print(f"    Predicted: {phrase_mapping[str(predicted_class)]} (confidence: {confidence:.3f})")
                print(f"    All probabilities:")
                for i, prob in enumerate(pred_probs[0]):
                    print(f"      {i} ({phrase_mapping[str(i)]}): {prob:.3f}")
                
                is_correct = predicted_class == phrase_idx
                if is_correct:
                    correct += 1
                
                total += 1
                all_correct += is_correct
                all_total += 1
                
            except Exception as e:
                print(f"  ❌ Error with {filename}: {e}")
        
        if total > 0:
            accuracy = correct / total * 100
            print(f"  Phrase {phrase_idx} accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    overall_accuracy = all_correct / all_total * 100 if all_total > 0 else 0
    print(f"\n🎯 Overall accuracy on new training data: {all_correct}/{all_total} ({overall_accuracy:.1f}%)")
    
    # Check if model weights are balanced
    print(f"\n🔬 Model Analysis:")
    print(f"Model has {len(model.layers)} layers")
    
    # Check final layer weights
    final_layer = model.layers[-1]
    if hasattr(final_layer, 'get_weights'):
        weights, biases = final_layer.get_weights()
        print(f"Final layer weights shape: {weights.shape}")
        print(f"Final layer biases: {biases}")
        
        # Check if biases are heavily skewed toward class 0
        max_bias_idx = np.argmax(biases)
        min_bias_idx = np.argmin(biases)
        
        print(f"Highest bias: Class {max_bias_idx} ({phrase_mapping[str(max_bias_idx)]}) = {biases[max_bias_idx]:.3f}")
        print(f"Lowest bias: Class {min_bias_idx} ({phrase_mapping[str(min_bias_idx)]}) = {biases[min_bias_idx]:.3f}")
        
        bias_range = np.max(biases) - np.min(biases)
        print(f"Bias range: {bias_range:.3f}")
        
        if bias_range > 2.0:
            print("🚨 Large bias range detected - model may be biased!")
        
        if max_bias_idx == 0:
            print("🚨 Class 0 has highest bias - explains 'Hi my name is Reet' bias!")

if __name__ == "__main__":
    debug_retrained_model()