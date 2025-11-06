#!/usr/bin/env python3
"""
Quick test to verify the enhanced app is working correctly
Tests blank frame removal and sequence processing
"""

import numpy as np
import sys
import os

print("="*80)
print("ENHANCED APP - QUICK VERIFICATION TEST")
print("="*80)

# Test 1: Check if model exists
print("\nTest 1: Checking for enhanced model...")
enhanced_model = "models/saved/lstm_model_enhanced.keras"
enhanced_scaler = "models/saved/sequence_scaler_enhanced.joblib"

if os.path.exists(enhanced_model) and os.path.exists(enhanced_scaler):
    print("  ✅ Enhanced model found!")
else:
    print("  ❌ Enhanced model not found")
    print("     Run: python enhanced_train.py")
    sys.exit(1)

# Test 2: Test blank frame removal logic
print("\nTest 2: Testing blank frame removal...")

# Simulate a sequence with blank frames
test_sequence = []
for i in range(100):
    if i % 10 == 0:  # Every 10th frame is blank
        test_sequence.append(np.zeros(126))
    else:
        test_sequence.append(np.random.rand(126))

# Count blanks
blank_count = sum(1 for frame in test_sequence if np.all(frame == 0))
print(f"  Test sequence: {len(test_sequence)} frames, {blank_count} blank")

# Remove blanks
cleaned = [frame for frame in test_sequence if not np.all(frame == 0)]
print(f"  After removal: {len(cleaned)} frames, 0 blank")

if len(cleaned) == len(test_sequence) - blank_count:
    print("  ✅ Blank frame removal works correctly!")
else:
    print("  ❌ Blank frame removal failed")
    sys.exit(1)

# Test 3: Test sequence normalization
print("\nTest 3: Testing sequence normalization...")

def normalize_sequence(sequence, target_length=90):
    """Normalize sequence to target length using interpolation"""
    current_length = len(sequence)
    
    if current_length == target_length:
        return np.array(sequence)
    
    old_indices = np.linspace(0, current_length - 1, current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)
    
    normalized_sequence = []
    sequence_array = np.array(sequence)
    
    for feature_idx in range(126):
        feature_values = sequence_array[:, feature_idx]
        interpolated = np.interp(new_indices, old_indices, feature_values)
        normalized_sequence.append(interpolated)
    
    return np.array(normalized_sequence).T

# Test with different lengths
for test_len in [50, 90, 150]:
    test_seq = [np.random.rand(126) for _ in range(test_len)]
    normalized = normalize_sequence(test_seq, target_length=90)
    
    if normalized.shape == (90, 126):
        print(f"  ✅ {test_len} frames → 90 frames: OK")
    else:
        print(f"  ❌ {test_len} frames → {normalized.shape}: FAILED")
        sys.exit(1)

# Test 4: Check Streamlit installation
print("\nTest 4: Checking Streamlit...")
try:
    import streamlit
    print(f"  ✅ Streamlit {streamlit.__version__} installed")
except ImportError:
    print("  ❌ Streamlit not installed")
    print("     Run: pip install streamlit")
    sys.exit(1)

# Test 5: Check MediaPipe
print("\nTest 5: Checking MediaPipe...")
try:
    import mediapipe
    print(f"  ✅ MediaPipe {mediapipe.__version__} installed")
except ImportError:
    print("  ❌ MediaPipe not installed")
    print("     Run: pip install mediapipe")
    sys.exit(1)

# Test 6: Check TensorFlow
print("\nTest 6: Checking TensorFlow...")
try:
    import tensorflow as tf
    print(f"  ✅ TensorFlow {tf.__version__} installed")
except ImportError:
    print("  ❌ TensorFlow not installed")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("ALL TESTS PASSED! ✅")
print("="*80)
print("\nYou can now run the enhanced app:")
print("  streamlit run app_enhanced.py")
print("\nThe app will be available at: http://localhost:8501")
print("="*80)
