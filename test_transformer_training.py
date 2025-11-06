#!/usr/bin/env python3
"""
Test the transformer model on its original training data
"""

import torch
import numpy as np
import os
import pickle
import json

def test_transformer_on_training_data():
    """Test if the transformer model works on its training data."""
    
    print("🧪 Testing Transformer on Original Training Data")
    print("="*60)
    
    # Load model
    model_path = "models/transformer/transformer_model.pth"
    scaler_path = "models/transformer/scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler not found: {scaler_path}")
        return
    
    # Import the model architecture
    from inference import TemporalTransformer
    
    # Load model
    device = torch.device('cpu')
    model = TemporalTransformer(
        feature_dim=1662,
        d_model=256,
        num_heads=4,
        num_layers=3,
        num_classes=5
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("✅ Model loaded")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Scaler loaded")
    
    # Load phrase mapping
    with open("models/transformer/phrase_mapping.json", 'r') as f:
        phrase_mapping = json.load(f)
    
    print("✅ Phrase mapping loaded")
    
    # Test on holistic training data
    all_correct = 0
    all_total = 0
    
    for phrase_idx in range(5):
        phrase_name = phrase_mapping[str(phrase_idx)]
        print(f"\n📂 Testing phrase {phrase_idx}: {phrase_name}")
        
        # Get training data for this phrase
        data_dir = f"data/sequences_holistic/phrase_{phrase_idx}"
        
        if not os.path.exists(data_dir):
            print(f"❌ No data found for phrase {phrase_idx}")
            continue
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        correct = 0
        total = 0
        
        # Test first 3 files
        for i, filename in enumerate(files[:3]):
            filepath = os.path.join(data_dir, filename)
            
            try:
                # Load sequence (should be 90 frames)
                sequence = np.load(filepath)
                
                # Take middle 60 frames like in inference
                if sequence.shape[0] == 90:
                    start_frame = 15
                    end_frame = 75
                    sequence_60 = sequence[start_frame:end_frame]
                elif sequence.shape[0] == 60:
                    sequence_60 = sequence
                else:
                    print(f"  ⚠️  Unexpected shape: {sequence.shape}")
                    continue
                
                # Flatten and scale
                sequence_flat = sequence_60.flatten()
                sequence_scaled = scaler.transform(sequence_flat.reshape(1, -1))
                
                # Reshape for model (1, 60, 1662)
                sequence_tensor = torch.FloatTensor(sequence_scaled.reshape(1, 60, 1662))
                
                # Predict
                with torch.no_grad():
                    outputs = model(sequence_tensor)
                    probabilities = torch.softmax(outputs, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = torch.max(probabilities).item()
                
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
    print(f"\n🎯 Overall accuracy on training data: {all_correct}/{all_total} ({overall_accuracy:.1f}%)")
    
    if overall_accuracy >= 80:
        print("✅ Transformer model works well on training data!")
        print("The issue is likely with live data preprocessing or capture.")
    else:
        print("🚨 Transformer model has issues even on training data!")
        print("Model may need retraining.")

if __name__ == "__main__":
    test_transformer_on_training_data()