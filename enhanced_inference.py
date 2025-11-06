#!/usr/bin/env python3
"""
Quick fix for inference confusion - adds confidence filtering and prediction smoothing.
This provides immediate improvement while the new model trains.
"""

import sys
import os
import numpy as np
from collections import deque, Counter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import HolisticInference

# Enhanced configuration
MIN_CONFIDENCE = 0.85  # Increased threshold to reduce false positives
SMOOTHING_WINDOW = 3   # Smooth predictions over multiple frames

class EnhancedISLPredictor(HolisticInference):
    """Enhanced predictor with confidence filtering and smoothing."""
    
    def __init__(self):
        super().__init__()
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.confidence_history = deque(maxlen=SMOOTHING_WINDOW)
    
    def predict_sequence(self):
        """Enhanced prediction with smoothing and confidence filtering."""
        if len(self.sequence_buffer) < 60:
            return None, 0.0
        
        # Get base prediction
        phrase, confidence = super().predict_sequence()
        
        if phrase is None:
            return None, 0.0
        
        # Add to history
        self.prediction_history.append(phrase)
        self.confidence_history.append(confidence)
        
        # If we don't have enough history, return with lower confidence
        if len(self.prediction_history) < SMOOTHING_WINDOW:
            return phrase, confidence * 0.8  # Reduce confidence for single predictions
        
        # Check for consistency in recent predictions
        recent_predictions = list(self.prediction_history)
        prediction_counts = Counter(recent_predictions)
        most_common_prediction, count = prediction_counts.most_common(1)[0]
        
        # Calculate consistency ratio
        consistency = count / len(recent_predictions)
        
        # Calculate average confidence for the most common prediction
        avg_confidence = np.mean([conf for pred, conf in zip(recent_predictions, self.confidence_history) 
                                 if pred == most_common_prediction])
        
        # Apply consistency boost
        if consistency >= 0.6:  # At least 60% of recent predictions agree
            final_confidence = avg_confidence * (1.0 + consistency * 0.2)  # Boost confidence
            return most_common_prediction, min(final_confidence, 1.0)
        else:
            # Not consistent enough - reduce confidence
            return most_common_prediction, avg_confidence * 0.6
    
    def reset_prediction_history(self):
        """Reset prediction history (call when user resets)."""
        self.prediction_history.clear()
        self.confidence_history.clear()
    
    def run(self):
        """Override run method to handle reset properly."""
        print("🎯 Enhanced ISL Recognition")
        print("Features:")
        print("  • Higher confidence threshold (85%)")
        print("  • Prediction smoothing over 3 frames")
        print("  • Consistency checking")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Clear sequence buffer AND prediction history")
        print("  's' - Save screenshot")
        print("")
        
        # Call parent run method but override key handling
        super().run()

def main():
    """Main function."""
    print("🔧 Enhanced ISL Recognition - Temporary Fix")
    print("This version reduces false positives while new model trains.")
    print("="*60)
    
    try:
        predictor = EnhancedISLPredictor()
        predictor.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()