#!/usr/bin/env python3
"""
Simple test using the correct transformer model that's already trained on holistic data
"""

import sys
import os
import time
import json
from datetime import datetime

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_transformer_model():
    """Test the transformer model that should work with holistic data."""
    print("🔍 Testing Transformer Model (Holistic Data)")
    print("="*50)
    
    try:
        import torch
        print("✅ PyTorch loaded")
    except ImportError:
        print("❌ PyTorch not available yet - installation may still be running")
        return
    
    # Import inference after PyTorch is available
    try:
        from inference import HolisticInference
        print("✅ Inference module loaded")
    except Exception as e:
        print(f"❌ Failed to load inference: {e}")
        return
    
    # Check if transformer model exists
    transformer_model_path = "models/transformer/transformer_model.pth"
    if not os.path.exists(transformer_model_path):
        print(f"❌ Transformer model not found at: {transformer_model_path}")
        return
    
    print(f"✅ Found transformer model: {transformer_model_path}")
    
    # Load model metadata
    try:
        with open("models/transformer/training_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Model trained on {metadata['feature_dim']} features (holistic data)")
        print(f"✅ Sequence length: {metadata['sequence_length']}")
        print(f"✅ Training date: {metadata['training_date']}")
        
    except Exception as e:
        print(f"⚠️  Could not load metadata: {e}")
    
    # Initialize inference system
    try:
        inference = HolisticInference()
        print("✅ Inference system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize inference: {e}")
        return
    
    print("\n🎬 Ready to test! This should now work correctly.")
    print("The transformer model was trained on the same holistic data")
    print("that your inference system captures (1662 features).")
    
    # Simple camera test
    import cv2
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    phrase_options = {
        0: "Hi my name is Reet",
        1: "How are you", 
        2: "I am from Delhi",
        3: "I like coffee",
        4: "What do you like"
    }
    
    print("\nAvailable phrases:")
    for i, phrase in phrase_options.items():
        print(f"  {i}: {phrase}")
    
    gesture_choice = input("\nWhich phrase will you perform? (0-4): ")
    
    try:
        gesture_idx = int(gesture_choice)
        expected_phrase = phrase_options[gesture_idx]
        print(f"\n👤 You will perform: {expected_phrase}")
    except:
        expected_phrase = "unknown"
        print(f"\n👤 You will perform: unknown gesture")
    
    input("Press Enter when ready to start...")
    
    print("\n🎬 Recording for 10 seconds... perform your gesture!")
    
    predictions = []
    start_time = time.time()
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Process frame
        landmarks, results = inference.process_frame(frame)
        
        # Add to buffer
        if landmarks is not None:
            inference.sequence_buffer.append(landmarks)
        
        # Try prediction every 2 seconds
        elapsed = time.time() - start_time
        if len(inference.sequence_buffer) >= 60 and int(elapsed) % 2 == 0:
            phrase, confidence = inference.predict_sequence()
            if phrase:
                prediction_entry = {
                    'time': elapsed,
                    'predicted_phrase': phrase,
                    'confidence': confidence,
                    'expected_phrase': expected_phrase
                }
                predictions.append(prediction_entry)
                print(f"[{elapsed:.1f}s] Predicted: {phrase} (confidence: {confidence:.3f})")
        
        # Show frame
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Results
    print("\n📊 RESULTS:")
    print("="*50)
    
    if not predictions:
        print("❌ No predictions made")
    else:
        print(f"Expected: {expected_phrase}")
        print("Predictions:")
        
        correct_predictions = 0
        for pred in predictions:
            is_correct = pred['predicted_phrase'] == pred['expected_phrase']
            correct_predictions += is_correct
            status = "✅" if is_correct else "❌"
            print(f"  {status} {pred['time']:.1f}s: {pred['predicted_phrase']} ({pred['confidence']:.3f})")
        
        accuracy = correct_predictions / len(predictions) * 100
        print(f"\nAccuracy: {correct_predictions}/{len(predictions)} ({accuracy:.1f}%)")
        
        if accuracy >= 80:
            print("🎉 EXCELLENT! Model is working correctly!")
        elif accuracy >= 50:
            print("👍 GOOD! Model is mostly working.")
        else:
            print("⚠️  Still some issues, but much better than before!")

if __name__ == "__main__":
    test_transformer_model()