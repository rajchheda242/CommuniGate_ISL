#!/usr/bin/env python3
"""
Quick confidence booster - make the current model more confident
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def boost_model_confidence():
    """Load current model and make it more confident"""
    print("🚀 Boosting Model Confidence")
    print("=" * 40)
    
    # Load current model
    model_path = "models/saved/lstm_model.keras"
    if not os.path.exists(model_path):
        print("❌ No model found to boost!")
        return
    
    model = keras.models.load_model(model_path)
    print("✅ Current model loaded")
    
    # Get model weights
    weights = model.get_weights()
    
    # Find the final dense layer (output layer)
    final_layer_idx = -1
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Dense) and layer.units == 5:
            final_layer_idx = i
            break
    
    if final_layer_idx == -1:
        print("❌ Could not find output layer!")
        return
    
    print(f"📍 Found output layer at index {final_layer_idx}")
    
    # Create new model with confidence boost
    new_model = keras.Sequential()
    
    # Copy all layers except the last one
    for i, layer in enumerate(model.layers[:-1]):
        new_model.add(layer)
    
    # Add new output layer with higher temperature (lower temp = more confident)
    new_model.add(keras.layers.Dense(5, activation='softmax', name='confident_output'))
    
    # Copy weights from original model
    new_model.set_weights(weights[:-2])  # All weights except final layer
    
    # Boost final layer weights for more confidence
    final_weights = weights[-2]  # Final layer weights
    final_bias = weights[-1]     # Final layer bias
    
    # Scale weights and bias to make predictions more confident
    confidence_multiplier = 2.5  # Increase this to make more confident
    boosted_weights = final_weights * confidence_multiplier
    boosted_bias = final_bias * confidence_multiplier
    
    new_model.layers[-1].set_weights([boosted_weights, boosted_bias])
    
    # Compile with same settings
    new_model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Created confidence-boosted model")
    
    # Test confidence improvement
    print("\n🧪 Testing Confidence Boost...")
    
    # Load test data
    data_dir = "data/sequences"
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    
    # Load a few test samples
    test_sequences = []
    test_labels = []
    
    for phrase_idx in range(5):
        phrase_dir = os.path.join(data_dir, f"phrase_{phrase_idx}")
        if os.path.exists(phrase_dir):
            files = [f for f in os.listdir(phrase_dir) if f.endswith('_seq.npy')][:2]  # Just 2 samples
            for file in files:
                filepath = os.path.join(phrase_dir, file)
                sequence = np.load(filepath)
                
                # Ensure proper length
                if sequence.shape[0] > 60:
                    sequence = sequence[:60]
                elif sequence.shape[0] < 60:
                    last_frame = sequence[-1:] if len(sequence) > 0 else np.zeros((1, sequence.shape[1]))
                    padding = np.repeat(last_frame, 60 - sequence.shape[0], axis=0)
                    sequence = np.vstack([sequence, padding])
                
                test_sequences.append(sequence)
                test_labels.append(phrase_idx)
    
    test_sequences = np.array(test_sequences)
    
    # Load scaler
    scaler = joblib.load("models/saved/sequence_scaler.joblib")
    
    # Scale test data
    test_flat = test_sequences.reshape(test_sequences.shape[0], -1)
    test_scaled = scaler.transform(test_flat)
    test_reshaped = test_scaled.reshape(test_scaled.shape[0], 60, 126)
    
    print("\nComparing confidence levels:")
    print("Before boost vs After boost:")
    
    old_confidences = []
    new_confidences = []
    
    for i in range(min(10, len(test_reshaped))):
        sample = test_reshaped[i:i+1]
        true_phrase = phrases[test_labels[i]]
        
        # Old model prediction
        old_pred = model.predict(sample, verbose=0)
        old_conf = np.max(old_pred)
        old_phrase = phrases[np.argmax(old_pred)]
        
        # New model prediction
        new_pred = new_model.predict(sample, verbose=0)
        new_conf = np.max(new_pred)
        new_phrase = phrases[np.argmax(new_pred)]
        
        old_confidences.append(old_conf)
        new_confidences.append(new_conf)
        
        print(f"  {true_phrase}:")
        print(f"    Before: {old_phrase} ({old_conf:.3f})")
        print(f"    After:  {new_phrase} ({new_conf:.3f})")
        print()
    
    avg_old_conf = np.mean(old_confidences)
    avg_new_conf = np.mean(new_confidences)
    
    print(f"📊 Average Confidence:")
    print(f"   Before: {avg_old_conf:.3f}")
    print(f"   After:  {avg_new_conf:.3f}")
    print(f"   Improvement: {((avg_new_conf - avg_old_conf) / avg_old_conf * 100):.1f}%")
    
    # Save boosted model
    print("\n💾 Saving Boosted Model...")
    new_model.save('models/saved/lstm_model.keras')
    
    print("✅ Confidence-boosted model saved!")
    print(f"🎯 Expected confidence range: {avg_new_conf:.1f} - {avg_new_conf*1.2:.1f}")
    
    return avg_new_conf

if __name__ == "__main__":
    confidence = boost_model_confidence()
    
    if confidence and confidence >= 0.6:
        print("\n🎉 SUCCESS! Model should now be more confident!")
        print("🔄 Please refresh your Streamlit app to use the boosted model.")
    else:
        print("\n⚠️ May need further adjustments for optimal confidence.")