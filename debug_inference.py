#!/usr/bin/env python3
"""
Debug version of inference script to analyze prediction confusion.
Shows all class probabilities to understand why the model confuses phrases.
"""

import sys
import os
import numpy as np
import torch
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import HolisticInference, SEQUENCE_LENGTH

class DebugISLPredictor(HolisticInference):
    """Debug version that shows detailed prediction analysis."""
    
    def predict_sequence(self):
        """Predict with detailed probability breakdown."""
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0.0
        
        # Prepare sequence
        sequence = np.array(self.sequence_buffer[-SEQUENCE_LENGTH:])
        sequence = self.resample_sequence_to_90(sequence)
        
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get all probabilities
            all_probs = probabilities[0].cpu().numpy()
            
            # Print detailed analysis
            print("\n" + "="*50)
            print("PREDICTION ANALYSIS:")
            print("="*50)
            
            # Show all probabilities
            for i, prob in enumerate(all_probs):
                phrase = self.phrase_mapping[str(i)]
                print(f"Class {i}: {phrase:<25} = {prob:.4f} ({prob*100:.1f}%)")
            
            # Get top prediction
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            
            # Show confusion analysis
            sorted_indices = np.argsort(all_probs)[::-1]
            print(f"\nTop 3 predictions:")
            for i, idx in enumerate(sorted_indices[:3]):
                phrase = self.phrase_mapping[str(idx)]
                print(f"  {i+1}. {phrase:<25} = {all_probs[idx]:.4f}")
            
            # Check for close predictions (potential confusion)
            top_prob = all_probs[sorted_indices[0]]
            second_prob = all_probs[sorted_indices[1]]
            margin = top_prob - second_prob
            
            print(f"\nConfidence margin: {margin:.4f}")
            if margin < 0.2:
                print("⚠️  LOW MARGIN - Potential confusion between top predictions!")
            
            phrase = self.phrase_mapping[str(predicted_class)]
            
            print("="*50)
            
            return phrase, confidence

def main():
    """Main function with debugging enabled."""
    print("🔍 DEBUG MODE: ISL Recognition with Detailed Analysis")
    print("This version shows all prediction probabilities to help diagnose confusion.")
    print("Press 'q' to quit, 'r' to reset sequence\n")
    
    try:
        predictor = DebugISLPredictor()
        predictor.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()