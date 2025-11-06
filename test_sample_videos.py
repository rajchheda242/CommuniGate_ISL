#!/usr/bin/env python3
"""
Test the model on sample videos with different clothes and angles
"""

import cv2
import numpy as np
import os
import sys
import time
from collections import deque

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_on_sample_videos():
    """Test the transformer model on sample videos."""
    
    print("🎬 Testing Model on Sample Videos")
    print("="*60)
    print("Testing on videos with different clothes and angles...")
    
    from inference import HolisticInference
    
    # Initialize inference
    try:
        inference = HolisticInference()
        print("✅ Inference system loaded")
    except Exception as e:
        print(f"❌ Failed to load inference: {e}")
        return
    
    video_dir = "data/videos/sample_videos"
    phrases = [
        "Hi my name is Reet",
        "How are you", 
        "I am from Delhi",
        "I like coffee",
        "What do you like"
    ]
    
    video_results = []
    
    for phrase_idx in range(5):
        video_file = os.path.join(video_dir, f"phrase{phrase_idx}.mp4")
        expected_phrase = phrases[phrase_idx]
        
        print(f"\n📹 Testing {video_file}")
        print(f"Expected: {expected_phrase}")
        
        if not os.path.exists(video_file):
            print(f"❌ Video not found: {video_file}")
            continue
        
        # Process video
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_file}")
            continue
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        print(f"Video info: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        # Clear sequence buffer
        inference.sequence_buffer.clear()
        
        predictions = []
        frame_num = 0
        landmarks_captured = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Process frame
            landmarks, results = inference.process_frame(frame)
            
            # Add to buffer if landmarks detected
            if landmarks is not None:
                inference.sequence_buffer.append(landmarks)
                landmarks_captured += 1
            
            # Try prediction every 30 frames (roughly every second)
            if len(inference.sequence_buffer) >= 60 and frame_num % 30 == 0:
                phrase, confidence = inference.predict_sequence()
                if phrase:
                    time_sec = frame_num / fps
                    prediction_entry = {
                        'time': time_sec,
                        'predicted_phrase': phrase,
                        'confidence': confidence,
                        'expected_phrase': expected_phrase
                    }
                    predictions.append(prediction_entry)
                    print(f"  [{time_sec:.1f}s] Predicted: {phrase} (confidence: {confidence:.3f})")
        
        cap.release()
        
        print(f"📊 Video processing complete:")
        print(f"  Frames processed: {frame_num}")
        print(f"  Landmarks captured: {landmarks_captured}")
        print(f"  Predictions made: {len(predictions)}")
        
        # Analyze predictions for this video
        if predictions:
            correct_predictions = sum(1 for p in predictions if p['predicted_phrase'] == p['expected_phrase'])
            accuracy = correct_predictions / len(predictions) * 100
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            # Get most common prediction
            pred_counts = {}
            for p in predictions:
                phrase = p['predicted_phrase']
                pred_counts[phrase] = pred_counts.get(phrase, 0) + 1
            
            most_common = max(pred_counts.items(), key=lambda x: x[1])
            
            result = {
                'video': video_file,
                'expected': expected_phrase,
                'predictions': predictions,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'most_common_prediction': most_common[0],
                'most_common_count': most_common[1],
                'total_predictions': len(predictions)
            }
            
            video_results.append(result)
            
            status = "✅" if accuracy >= 70 else "⚠️" if accuracy >= 40 else "❌"
            print(f"  {status} Accuracy: {correct_predictions}/{len(predictions)} ({accuracy:.1f}%)")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Most common prediction: {most_common[0]} ({most_common[1]}/{len(predictions)} times)")
        else:
            print("  ❌ No predictions made (insufficient landmarks)")
            result = {
                'video': video_file,
                'expected': expected_phrase,
                'predictions': [],
                'accuracy': 0,
                'avg_confidence': 0,
                'most_common_prediction': 'None',
                'most_common_count': 0,
                'total_predictions': 0
            }
            video_results.append(result)
    
    # Overall summary
    print(f"\n📊 OVERALL RESULTS:")
    print("="*60)
    
    if video_results:
        total_correct = 0
        total_predictions = 0
        successful_videos = 0
        
        for result in video_results:
            if result['total_predictions'] > 0:
                correct = sum(1 for p in result['predictions'] if p['predicted_phrase'] == p['expected_phrase'])
                total_correct += correct
                total_predictions += result['total_predictions']
                
                # Consider video successful if most common prediction is correct
                is_successful = result['most_common_prediction'] == result['expected']
                if is_successful:
                    successful_videos += 1
                
                status = "✅" if is_successful else "❌"
                print(f"{status} {os.path.basename(result['video'])}: {result['most_common_prediction']} (expected: {result['expected']})")
        
        if total_predictions > 0:
            overall_accuracy = total_correct / total_predictions * 100
            video_success_rate = successful_videos / len(video_results) * 100
            
            print(f"\nOverall Statistics:")
            print(f"  Per-prediction accuracy: {total_correct}/{total_predictions} ({overall_accuracy:.1f}%)")
            print(f"  Video-level success rate: {successful_videos}/{len(video_results)} ({video_success_rate:.1f}%)")
            
            if video_success_rate >= 80:
                print("🎉 EXCELLENT! Model works well on different clothes/angles")
                print("   The environment mismatch is likely lighting or gesture execution style.")
            elif video_success_rate >= 60:
                print("👍 GOOD! Model has some robustness to clothing/angle changes")
                print("   May need minor environment calibration.")
            elif video_success_rate >= 40:
                print("⚠️  MODERATE! Model struggles with clothing/angle variations")
                print("   Environment retraining recommended.")
            else:
                print("🚨 POOR! Model has significant issues with variations")
                print("   Definitely needs retraining with more diverse data.")
        else:
            print("❌ No predictions could be made from any video")
    
    return video_results

if __name__ == "__main__":
    test_model_on_sample_videos()