#!/usr/bin/env python3
"""
Create a robust preprocessor that normalizes live camera data to match training distribution.
"""

import numpy as np
import pickle
import torch
import json
from pathlib import Path

class RobustHolisticPreprocessor:
    """Robust preprocessor that handles coordinate system differences."""
    
    def __init__(self, model_dir="models/transformer"):
        self.model_dir = model_dir
        self.scaler = None
        self.training_stats = None
        self.load_components()
        
    def load_components(self):
        """Load scaler and calculate training data statistics."""
        # Load scaler
        scaler_path = Path(self.model_dir) / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Calculate training data statistics for reference
        self.calculate_training_stats()
    
    def calculate_training_stats(self):
        """Calculate statistics from training data for robust normalization."""
        print("Calculating training data statistics...")
        
        all_data = []
        
        # Load all training data
        for phrase_id in range(5):
            phrase_dir = f"data/sequences_holistic/phrase_{phrase_id}"
            if Path(phrase_dir).exists():
                for file_path in Path(phrase_dir).glob("*.npy"):
                    try:
                        data = np.load(file_path)
                        all_data.append(data)
                    except:
                        continue
        
        if all_data:
            combined_data = np.concatenate(all_data, axis=0)
            
            self.training_stats = {
                'mean': np.mean(combined_data, axis=0),
                'std': np.std(combined_data, axis=0),
                'min': np.min(combined_data, axis=0),
                'max': np.max(combined_data, axis=0),
                'q25': np.percentile(combined_data, 25, axis=0),
                'q75': np.percentile(combined_data, 75, axis=0)
            }
            
            print(f"✅ Training stats calculated from {combined_data.shape[0]} frames")
        else:
            print("❌ No training data found")
            
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to be in a reasonable range before scaling."""
        if landmarks is None:
            return None
            
        # Make a copy to avoid modifying original
        normalized = landmarks.copy()
        
        # 1. Clip extreme values (camera coordinates should be in [0, 1] mostly)
        # But allow some margin for coordinates outside frame
        normalized = np.clip(normalized, -2.0, 3.0)
        
        # 2. For zero values (missing landmarks), use training data mean for those features
        if self.training_stats is not None:
            zero_mask = (landmarks == 0)
            normalized[zero_mask] = self.training_stats['mean'][zero_mask]
        
        # 3. Apply robust scaling to bring camera data closer to training range
        # Identify non-zero landmarks and scale them appropriately
        non_zero_mask = (landmarks != 0)
        
        if np.any(non_zero_mask) and self.training_stats is not None:
            # Scale non-zero coordinates to match training data range
            for i in range(len(normalized)):
                if non_zero_mask[i]:
                    # Map [0, 1] camera coordinates to training data range for this feature
                    training_min = self.training_stats['min'][i]
                    training_max = self.training_stats['max'][i]
                    
                    if training_max > training_min:
                        # Scale from [0, 1] to [training_min, training_max]
                        normalized[i] = training_min + normalized[i] * (training_max - training_min)
        
        return normalized
    
    def preprocess_sequence(self, sequence):
        """Preprocess a sequence for model input."""
        if len(sequence) == 0:
            return None
            
        # Convert to numpy array
        sequence_array = np.array(sequence)
        
        # Normalize each frame
        normalized_sequence = []
        for frame in sequence_array:
            normalized_frame = self.normalize_landmarks(frame)
            if normalized_frame is not None:
                normalized_sequence.append(normalized_frame)
        
        if len(normalized_sequence) == 0:
            return None
            
        normalized_array = np.array(normalized_sequence)
        
        # Apply scaler
        sequence_flat = normalized_array.reshape(-1, 1662)
        sequence_scaled = self.scaler.transform(sequence_flat)
        sequence_scaled = sequence_scaled.reshape(len(normalized_sequence), 1662)
        
        return sequence_scaled

def test_robust_preprocessor():
    """Test the robust preprocessor."""
    print("🧪 Testing Robust Preprocessor")
    print("="*50)
    
    preprocessor = RobustHolisticPreprocessor()
    
    # Test with different input types
    
    # 1. All zeros (missing landmarks)
    zeros = np.zeros(1662)
    zeros_processed = preprocessor.normalize_landmarks(zeros)
    print(f"1. All zeros → Range: {zeros_processed.min():.3f} to {zeros_processed.max():.3f}")
    
    # 2. Camera-like coordinates (0-1 range)
    camera_coords = np.random.uniform(0, 1, 1662)
    camera_processed = preprocessor.normalize_landmarks(camera_coords)
    print(f"2. Camera coords → Range: {camera_processed.min():.3f} to {camera_processed.max():.3f}")
    
    # 3. Mixed (some zeros, some camera coords)
    mixed = np.random.uniform(0, 1, 1662)
    mixed[:126] = 0  # Missing pose landmarks
    mixed_processed = preprocessor.normalize_landmarks(mixed)
    print(f"3. Mixed data → Range: {mixed_processed.min():.3f} to {mixed_processed.max():.3f}")
    
    # 4. Test full sequence preprocessing
    test_sequence = [camera_coords for _ in range(60)]
    processed_sequence = preprocessor.preprocess_sequence(test_sequence)
    
    if processed_sequence is not None:
        print(f"4. Full sequence → Shape: {processed_sequence.shape}")
        print(f"   Range: {processed_sequence.min():.3f} to {processed_sequence.max():.3f}")
        print(f"   Mean: {processed_sequence.mean():.3f}")
    
    return preprocessor

if __name__ == "__main__":
    test_robust_preprocessor()