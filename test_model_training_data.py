#!/usr/bin/env python3
"""
Test if the transformer model can predict its own training data correctly
"""

import numpy as np
import torch
import pickle
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_on_training_samples():
    """Test the transformer model on multiple training samples."""
    
    print("🧪 Testing Transformer Model on Training Data")
    print("="*60)
    
    from inference import HolisticInference
    
    # Initialize inference
    try:
        inference = HolisticInference()
        print("✅ Inference system loaded")
    except Exception as e:
        print(f"❌ Failed to load inference: {e}")
        return
    
    # Test multiple samples from each phrase
    overall_correct = 0
    overall_total = 0
    
    for phrase_idx in range(5):
        print(f"\n📂 Testing phrase {phrase_idx}")
        
        # Get files for this phrase
        data_dir = f"data/sequences_holistic/phrase_{phrase_idx}"
        if not os.path.exists(data_dir):
            print(f"❌ No data found for {phrase_idx}")
            continue
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        files = files[:3]  # Test first 3 files
        
        phrase_correct = 0
        phrase_total = 0
        
        for filename in files:
            filepath = os.path.join(data_dir, filename)
            
            try:
                # Load sequence (90 frames)
                sequence = np.load(filepath)
                
                # Take middle 60 frames like inference does
                if sequence.shape[0] == 90:
                    start_frame = 15
                    end_frame = 75
                    sequence_60 = sequence[start_frame:end_frame]
                elif sequence.shape[0] == 60:
                    sequence_60 = sequence
                else:
                    print(f"  ⚠️  Unexpected shape: {sequence.shape}")
                    continue
                
                # Predict using the same method as live inference
                predicted_phrase, confidence = inference.predict_from_sequence(sequence_60)
                
                # Check if correct
                expected_phrase = inference.phrase_mapping[phrase_idx]
                is_correct = predicted_phrase == expected_phrase
                
                if is_correct:
                    phrase_correct += 1
                    status = "✅"
                else:
                    status = f"❌ -> {predicted_phrase}"
                
                phrase_total += 1
                overall_correct += is_correct
                overall_total += 1
                
                print(f"  {status} {filename}: {confidence:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error with {filename}: {e}")
        
        if phrase_total > 0:
            accuracy = phrase_correct / phrase_total * 100
            print(f"  Phrase {phrase_idx} accuracy: {phrase_correct}/{phrase_total} ({accuracy:.1f}%)")
        else:
            print(f"  No valid samples for phrase {phrase_idx}")
    
    if overall_total > 0:
        overall_accuracy = overall_correct / overall_total * 100
        print(f"\n🎯 Overall accuracy on training data: {overall_correct}/{overall_total} ({overall_accuracy:.1f}%)")
        
        if overall_accuracy >= 80:
            print("✅ Model works well on training data - issue is with live capture")
        elif overall_accuracy >= 50:
            print("⚠️  Model has moderate issues - may need some fixing")
        else:
            print("🚨 Model has serious issues - needs retraining or debugging")
    else:
        print("❌ No samples could be tested")

if __name__ == "__main__":
    test_model_on_training_samples()