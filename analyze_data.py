#!/usr/bin/env python3
"""
Comprehensive data analysis to find why the model isn't working
"""

import cv2
import numpy as np
import os
import json
import time
import sys

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_data_mismatch():
    """Compare live camera data with training data."""
    
    print("🔍 Comprehensive Data Analysis")
    print("="*60)
    
    # Load training data sample
    training_file = "data/sequences_holistic/phrase_0/take 4_seq.npy"
    if not os.path.exists(training_file):
        # Try to find any file
        import glob
        files = glob.glob("data/sequences_holistic/phrase_0/*.npy")
        if files:
            training_file = files[0]
            print(f"Using: {training_file}")
        else:
            print(f"❌ No training files found in data/sequences_holistic/phrase_0/")
            return
    
    training_seq = np.load(training_file)
    print(f"✅ Training data shape: {training_seq.shape}")
    print(f"✅ Training data features per frame: {training_seq.shape[1]}")
    
    # Get middle 60 frames like inference does
    start_frame = 15
    end_frame = 75
    training_60 = training_seq[start_frame:end_frame]
    
    print(f"✅ Training 60-frame sequence: {training_60.shape}")
    
    # Now capture live data
    print("\n📹 Capturing live data for comparison...")
    print("Please perform 'Hi my name is Reet' gesture when ready.")
    
    from inference import HolisticInference
    
    try:
        inference = HolisticInference()
        print("✅ Inference system loaded")
    except Exception as e:
        print(f"❌ Failed to load inference: {e}")
        return
    
    # Capture live sequence
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    input("Press Enter to start capturing live data...")
    
    print("🎬 Capturing 60 frames of live data...")
    live_landmarks = []
    frame_count = 0
    
    while frame_count < 60:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process frame
        landmarks, results = inference.process_frame(frame)
        
        if landmarks is not None:
            live_landmarks.append(landmarks)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Captured {frame_count}/60 frames...")
        
        cv2.imshow('Capturing...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(live_landmarks) < 60:
        print(f"❌ Only captured {len(live_landmarks)} frames, need 60")
        return
    
    live_sequence = np.array(live_landmarks)
    print(f"✅ Live data shape: {live_sequence.shape}")
    
    # Compare statistics
    print("\n📊 STATISTICAL COMPARISON:")
    print("="*60)
    
    def analyze_sequence(seq, name):
        print(f"\n{name}:")
        print(f"  Shape: {seq.shape}")
        print(f"  Mean: {np.mean(seq):.6f}")
        print(f"  Std: {np.std(seq):.6f}")
        print(f"  Min: {np.min(seq):.6f}")
        print(f"  Max: {np.max(seq):.6f}")
        print(f"  Non-zero ratio: {np.count_nonzero(seq) / seq.size:.3f}")
        
        # Check for NaN or inf
        if np.any(np.isnan(seq)):
            print(f"  ⚠️  Contains NaN values!")
        if np.any(np.isinf(seq)):
            print(f"  ⚠️  Contains infinite values!")
        
        return {
            'mean': np.mean(seq),
            'std': np.std(seq),
            'min': np.min(seq),
            'max': np.max(seq),
            'nonzero_ratio': np.count_nonzero(seq) / seq.size
        }
    
    training_stats = analyze_sequence(training_60, "Training Data")
    live_stats = analyze_sequence(live_sequence, "Live Data")
    
    # Calculate differences
    print(f"\n📈 DIFFERENCES:")
    print("="*60)
    
    for stat in ['mean', 'std', 'min', 'max', 'nonzero_ratio']:
        diff = abs(training_stats[stat] - live_stats[stat])
        pct_diff = diff / (abs(training_stats[stat]) + 1e-8) * 100
        print(f"{stat:15}: {diff:.6f} ({pct_diff:.1f}% difference)")
    
    # Feature-by-feature analysis
    print(f"\n🔬 FEATURE-BY-FEATURE ANALYSIS:")
    print("="*60)
    
    frame_diffs = []
    for i in range(60):
        frame_diff = np.mean(np.abs(training_60[i] - live_sequence[i]))
        frame_diffs.append(frame_diff)
    
    avg_frame_diff = np.mean(frame_diffs)
    max_frame_diff = np.max(frame_diffs)
    min_frame_diff = np.min(frame_diffs)
    
    print(f"Average frame difference: {avg_frame_diff:.6f}")
    print(f"Max frame difference: {max_frame_diff:.6f}")
    print(f"Min frame difference: {min_frame_diff:.6f}")
    
    # Find most different features
    feature_diffs = []
    for feat_idx in range(training_60.shape[1]):
        training_feat = training_60[:, feat_idx]
        live_feat = live_sequence[:, feat_idx]
        feat_diff = np.mean(np.abs(training_feat - live_feat))
        feature_diffs.append(feat_diff)
    
    feature_diffs = np.array(feature_diffs)
    worst_features = np.argsort(feature_diffs)[-10:]  # Top 10 worst
    
    print(f"\nTop 10 most different features:")
    for i, feat_idx in enumerate(reversed(worst_features)):
        diff = feature_diffs[feat_idx]
        print(f"  {i+1}. Feature {feat_idx}: difference = {diff:.6f}")
    
    # Test with current model
    print(f"\n🤖 MODEL PREDICTION COMPARISON:")
    print("="*60)
    
    try:
        # Test both sequences
        training_phrase, training_conf = inference.predict_from_sequence(training_60)
        live_phrase, live_conf = inference.predict_from_sequence(live_sequence)
        
        print(f"Training sequence prediction: {training_phrase} (confidence: {training_conf:.3f})")
        print(f"Live sequence prediction: {live_phrase} (confidence: {live_conf:.3f})")
        
        if training_phrase == "Hi my name is Reet":
            print("✅ Model correctly predicts training data!")
            if live_phrase != "Hi my name is Reet":
                print("🚨 But fails on live data - preprocessing issue!")
        else:
            print("🚨 Model fails even on training data - model issue!")
    
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
    
    # Save analysis
    analysis_results = {
        'training_stats': training_stats,
        'live_stats': live_stats,
        'avg_frame_diff': float(avg_frame_diff),
        'max_frame_diff': float(max_frame_diff),
        'worst_features': [int(x) for x in worst_features[-5:]],
        'feature_differences': [float(x) for x in feature_diffs]
    }
    
    with open('data_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n💾 Analysis saved to: data_analysis_results.json")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("="*60)
    
    if avg_frame_diff > 0.1:
        print("🔴 Large difference between training and live data!")
        print("   - Check camera setup and lighting")
        print("   - Verify gesture execution matches training")
        print("   - Consider retraining with current setup")
    elif avg_frame_diff > 0.05:
        print("🟡 Moderate difference - may need calibration")
        print("   - Check preprocessing pipeline")
        print("   - Verify normalization is working correctly")
    else:
        print("🟢 Data looks similar - model or inference issue")
        print("   - Check model loading")
        print("   - Verify prediction pipeline")

if __name__ == "__main__":
    analyze_data_mismatch()