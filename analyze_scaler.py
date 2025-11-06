#!/usr/bin/env python3
"""
Fix the data preprocessing mismatch between training and inference.
"""

import numpy as np
import pickle
import os

def analyze_scaler():
    """Analyze the scaler used in training vs inference."""
    print("🔍 Analyzing Data Preprocessing")
    print("="*50)
    
    # Load the scaler
    scaler_path = "models/transformer/scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("Scaler information:")
    print(f"  Type: {type(scaler)}")
    print(f"  Mean shape: {scaler.mean_.shape}")
    print(f"  Scale shape: {scaler.scale_.shape}")
    print(f"  Mean range: {scaler.mean_.min():.3f} to {scaler.mean_.max():.3f}")
    print(f"  Scale range: {scaler.scale_.min():.3f} to {scaler.scale_.max():.3f}")
    
    # Test with different inputs
    print("\nTesting scaler with different inputs:")
    
    # Test 1: Training data
    training_data = np.load('data/sequences_holistic/phrase_0/take 4_seq.npy')
    training_flat = training_data.reshape(-1, 1662)
    training_scaled = scaler.transform(training_flat)
    print(f"\n1. Training data after scaling:")
    print(f"   Original: {training_flat.min():.3f} to {training_flat.max():.3f}, mean={training_flat.mean():.3f}")
    print(f"   Scaled:   {training_scaled.min():.3f} to {training_scaled.max():.3f}, mean={training_scaled.mean():.3f}")
    
    # Test 2: Zeros (like missing landmarks)
    zero_input = np.zeros((1, 1662))
    zero_scaled = scaler.transform(zero_input)
    print(f"\n2. All zeros after scaling:")
    print(f"   Original: {zero_input.min():.3f} to {zero_input.max():.3f}, mean={zero_input.mean():.3f}")
    print(f"   Scaled:   {zero_scaled.min():.3f} to {zero_scaled.max():.3f}, mean={zero_scaled.mean():.3f}")
    
    # Test 3: Live camera-like data (0-1 range)
    camera_like = np.random.uniform(0, 1, (1, 1662))
    camera_scaled = scaler.transform(camera_like)
    print(f"\n3. Camera-like data (0-1 range) after scaling:")
    print(f"   Original: {camera_like.min():.3f} to {camera_like.max():.3f}, mean={camera_like.mean():.3f}")
    print(f"   Scaled:   {camera_scaled.min():.3f} to {camera_scaled.max():.3f}, mean={camera_scaled.mean():.3f}")
    
    # Check if scaler was fitted properly
    print(f"\n4. Scaler statistics:")
    print(f"   Features with zero mean: {np.count_nonzero(scaler.mean_ == 0)}")
    print(f"   Features with scale=1: {np.count_nonzero(scaler.scale_ == 1.0)}")
    
    # Analyze problem areas
    zero_mean_indices = np.where(scaler.mean_ == 0)[0]
    if len(zero_mean_indices) > 0:
        print(f"\n⚠️  Warning: {len(zero_mean_indices)} features have zero mean!")
        print(f"   This suggests these features were always zero in training data.")
        
        # Check if these correspond to missing landmarks
        if len(zero_mean_indices) == 63:
            print("   → Likely missing hand landmarks (63 features)")
        elif len(zero_mean_indices) == 126:
            print("   → Likely missing both hand landmarks (126 features)")
        elif len(zero_mean_indices) == 1404:
            print("   → Likely missing face landmarks (1404 features)")

if __name__ == "__main__":
    analyze_scaler()