"""
Demo script to test inference on preprocessed sequences.
"""

import torch
import numpy as np
import os
import glob
from inference import HolisticInference
import pickle

def test_inference_on_sequences():
    """Test inference on actual preprocessed sequences."""
    print("Testing inference on preprocessed sequences...")
    
    # Load inference model
    inference = HolisticInference()
    
    # Test on each phrase
    for phrase_idx in range(5):
        phrase_dir = f"data/sequences_holistic/phrase_{phrase_idx}"
        phrase_name = inference.phrase_mapping[phrase_idx]
        
        if not os.path.exists(phrase_dir):
            print(f"Warning: No data for phrase {phrase_idx}")
            continue
        
        # Get sequence files
        seq_files = glob.glob(os.path.join(phrase_dir, "*_seq.npy"))
        
        if not seq_files:
            print(f"Warning: No sequences for phrase {phrase_idx}")
            continue
        
        print(f"\\nTesting phrase {phrase_idx}: '{phrase_name}'")
        print(f"Available sequences: {len(seq_files)}")
        
        correct_predictions = 0
        total_predictions = 0
        
        # Test on a few sequences
        for seq_file in seq_files[:3]:  # Test first 3 sequences
            # Load sequence
            sequence = np.load(seq_file)
            
            # Apply same preprocessing as training
            sequence_flat = sequence.reshape(-1, 1662)
            sequence_scaled = inference.scaler.transform(sequence_flat)
            sequence_scaled = sequence_scaled.reshape(1, 90, 1662)
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence_scaled).to(inference.device)
            
            # Predict
            with torch.no_grad():
                outputs = inference.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            predicted_phrase = inference.phrase_mapping[predicted_class]
            
            print(f"  File: {os.path.basename(seq_file)}")
            print(f"    Predicted: '{predicted_phrase}' (class {predicted_class})")
            print(f"    Confidence: {confidence:.3f}")
            print(f"    Correct: {'✓' if predicted_class == phrase_idx else '✗'}")
            
            if predicted_class == phrase_idx:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"  Accuracy for this phrase: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    print("\\n✓ Inference testing complete!")

if __name__ == "__main__":
    test_inference_on_sequences()