#!/usr/bin/env python3
"""
Simple troubleshooting tool - just capture what you're doing and what the model predicts.
"""

import sys
import os
import time
import json
from datetime import datetime

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simple_test():
    """Simple test to see what's going wrong."""
    print("🔍 Simple ISL Recognition Test")
    print("="*50)
    
    # Import inference after path setup
    try:
        from inference import HolisticInference
        print("✅ Inference module loaded")
    except Exception as e:
        print(f"❌ Failed to load inference: {e}")
        return
    
    # Load inference system
    try:
        inference = HolisticInference()
        print("✅ Inference system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize inference: {e}")
        return
    
    # Test with actual camera
    print("\n📹 Testing with camera...")
    print("Instructions:")
    print("1. I'll capture 10 seconds of data")
    print("2. During this time, perform ONE specific gesture")
    print("3. Tell me which gesture you're performing")
    print("4. I'll show you what the model predicts")
    
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
    
    input("Press Enter when ready to start capturing...")
    
    # Capture data
    import cv2
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("\n🎬 Recording... perform your gesture now!")
    
    predictions = []
    start_time = time.time()
    
    while time.time() - start_time < 10:  # 10 seconds
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
        
        # Show frame (optional)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n📊 RESULTS:")
    print("="*50)
    
    if not predictions:
        print("❌ No predictions made - buffer may not have filled")
        print("💡 Try moving more or checking camera")
    else:
        print(f"Expected: {expected_phrase}")
        print("Predictions:")
        
        correct_predictions = 0
        total_predictions = len(predictions)
        
        for pred in predictions:
            is_correct = pred['predicted_phrase'] == pred['expected_phrase']
            correct_predictions += is_correct
            status = "✅" if is_correct else "❌"
            
            print(f"  {status} {pred['time']:.1f}s: {pred['predicted_phrase']} ({pred['confidence']:.3f})")
        
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        print(f"\nAccuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simple_test_results_{timestamp}.json"
        
        test_results = {
            'timestamp': timestamp,
            'expected_phrase': expected_phrase,
            'predictions': predictions,
            'accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions
        }
        
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Analysis
        if accuracy < 50:
            print(f"\n🚨 LOW ACCURACY DETECTED!")
            print("Possible issues:")
            print("  - Model not trained properly on this gesture")
            print("  - Camera/lighting conditions different from training")
            print("  - Gesture execution different from training")
            print("  - Data preprocessing issues")
        
        # Show prediction distribution
        pred_counts = {}
        for pred in predictions:
            phrase = pred['predicted_phrase']
            pred_counts[phrase] = pred_counts.get(phrase, 0) + 1
        
        print(f"\nPrediction distribution:")
        for phrase, count in pred_counts.items():
            percentage = count / total_predictions * 100
            print(f"  {phrase}: {count} times ({percentage:.1f}%)")

if __name__ == "__main__":
    simple_test()