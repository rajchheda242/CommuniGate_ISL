#!/usr/bin/env python3
"""
Test the inference pipeline step by step to identify where the problem occurs.
"""

import torch
import numpy as np
import pickle
import json
from collections import deque
import sys
import os

# Add the current directory to path to import inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_inference_pipeline():
    """Test each step of the inference pipeline."""
    print("🔍 Testing Inference Pipeline Step by Step")
    print("="*60)
    
    # 1. Load model components
    model_path = "models/transformer/transformer_model.pth"
    scaler_path = "models/transformer/scaler.pkl"
    phrase_path = "models/transformer/phrase_mapping.json"
    
    print("1. Loading model components...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"   ✅ Model loaded")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"   ✅ Scaler loaded")
    
    # Load phrases
    with open(phrase_path, 'r') as f:
        phrase_mapping = json.load(f)
    print(f"   ✅ Phrases loaded: {phrase_mapping}")
    
    # 2. Create model instance
    print("\n2. Creating model instance...")
    from inference import TemporalTransformer
    
    config = checkpoint['model_config']
    model = TemporalTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        max_seq_len=config['sequence_length']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✅ Model instance created and loaded")
    
    # 3. Test with different types of input
    print("\n3. Testing with different inputs...")
    
    # Test 1: All zeros
    print("\n   Test 1: All zeros input")
    zero_sequence = np.zeros((60, 1662))
    result = test_sequence(zero_sequence, model, scaler, phrase_mapping)
    print(f"   Result: {result}")
    
    # Test 2: All ones  
    print("\n   Test 2: All ones input")
    ones_sequence = np.ones((60, 1662))
    result = test_sequence(ones_sequence, model, scaler, phrase_mapping)
    print(f"   Result: {result}")
    
    # Test 3: Random normal
    print("\n   Test 3: Random normal input")
    random_sequence = np.random.normal(0, 1, (60, 1662))
    result = test_sequence(random_sequence, model, scaler, phrase_mapping)
    print(f"   Result: {result}")
    
    # Test 4: Random uniform
    print("\n   Test 4: Random uniform input")
    uniform_sequence = np.random.uniform(-1, 1, (60, 1662))
    result = test_sequence(uniform_sequence, model, scaler, phrase_mapping)
    print(f"   Result: {result}")
    
    # Test 5: Load actual training data
    print("\n   Test 5: Actual training data")
    try:
        # Load a real training sample
        sample_path = "data/sequences_holistic/phrase_0/take 4_seq.npy"
        if os.path.exists(sample_path):
            real_sequence = np.load(sample_path)
            print(f"   Loaded real sequence shape: {real_sequence.shape}")
            
            # Resample to 60 frames if needed
            if real_sequence.shape[0] == 90:
                # Simple resampling - take every 1.5th frame
                indices = np.linspace(0, 89, 60).astype(int)
                real_sequence = real_sequence[indices]
                print(f"   Resampled to: {real_sequence.shape}")
            
            result = test_sequence(real_sequence, model, scaler, phrase_mapping)
            print(f"   Result for 'Hi my name is Reet': {result}")
        else:
            print(f"   ❌ Training data not found at {sample_path}")
    except Exception as e:
        print(f"   ❌ Error loading training data: {e}")

def test_sequence(sequence, model, scaler, phrase_mapping):
    """Test a single sequence through the pipeline."""
    try:
        # Apply scaling
        sequence_flat = sequence.reshape(-1, 1662)
        sequence_scaled = scaler.transform(sequence_flat)
        sequence_scaled = sequence_scaled.reshape(1, 60, 1662)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_scaled)
        
        # Predict
        with torch.no_grad():
            outputs = model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        phrase = phrase_mapping.get(str(predicted_class), "Unknown")
        
        # Return detailed info
        return {
            'phrase': phrase,
            'confidence': confidence,
            'class': predicted_class,
            'all_probabilities': probabilities[0].tolist(),
            'raw_logits': outputs[0].tolist()
        }
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    test_inference_pipeline()