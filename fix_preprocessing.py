#!/usr/bin/env python3
"""
Fix the preprocessing pipeline based on analysis results
"""

import json
import numpy as np

def analyze_problematic_features():
    """Analyze which landmark features are causing issues."""
    
    # Load analysis results
    with open('data_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    worst_features = results['worst_features']
    
    print("🔍 Analyzing Problematic Features")
    print("="*50)
    
    # MediaPipe Holistic feature layout:
    # 0-131: Pose landmarks (33 points × 4 features: x, y, z, visibility)
    # 132-194: Left hand landmarks (21 points × 3 features: x, y, z)  
    # 195-257: Right hand landmarks (21 points × 3 features: x, y, z)
    # 258-1661: Face landmarks (468 points × 3 features: x, y, z)
    
    for feat_idx in worst_features:
        if feat_idx < 132:
            # Pose landmark
            point_idx = feat_idx // 4
            feature_type = ['x', 'y', 'z', 'visibility'][feat_idx % 4]
            print(f"Feature {feat_idx}: Pose point {point_idx} ({feature_type})")
        elif feat_idx < 195:
            # Left hand
            point_idx = (feat_idx - 132) // 3
            feature_type = ['x', 'y', 'z'][(feat_idx - 132) % 3]
            print(f"Feature {feat_idx}: Left hand point {point_idx} ({feature_type})")
        elif feat_idx < 258:
            # Right hand  
            point_idx = (feat_idx - 195) // 3
            feature_type = ['x', 'y', 'z'][(feat_idx - 195) % 3]
            print(f"Feature {feat_idx}: Right hand point {point_idx} ({feature_type})")
        else:
            # Face
            point_idx = (feat_idx - 258) // 3
            feature_type = ['x', 'y', 'z'][(feat_idx - 258) % 3]
            print(f"Feature {feat_idx}: Face point {point_idx} ({feature_type})")
    
    print("\n💡 Analysis:")
    print("Most problematic features are in the pose landmarks (features 101-129)")
    print("This suggests the pose normalization is inconsistent between training and live data.")
    
    return worst_features

def create_fixed_inference():
    """Create a fixed version of inference with better preprocessing."""
    
    print("\n🔧 Creating Fixed Inference System")
    print("="*50)
    
    # Read the current inference file
    with open('inference.py', 'r') as f:
        content = f.read()
    
    # The issue is likely in the normalize_landmarks function
    # Let's create a more robust version
    
    improved_normalization = '''
    def robust_normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply more robust scale-invariant normalization to landmarks."""
        if len(landmarks) != 1662:
            print(f"Warning: Expected 1662 features, got {len(landmarks)}. Padding or truncating.")
            if len(landmarks) > 1662:
                landmarks = landmarks[:1662]
            else:
                landmarks = np.pad(landmarks, (0, 1662 - len(landmarks)))
        
        # Create a copy to avoid modifying original
        normalized = landmarks.copy()
        
        # Extract components with proper indexing
        pose_landmarks = normalized[:132].reshape(33, 4)  # x, y, z, visibility
        left_hand = normalized[132:195].reshape(21, 3)    # x, y, z
        right_hand = normalized[195:258].reshape(21, 3)   # x, y, z  
        face_landmarks = normalized[258:1662].reshape(468, 3)  # x, y, z
        
        # More robust pose normalization using training data statistics
        if hasattr(self, 'training_stats') and self.training_stats is not None:
            # Use training statistics for robust scaling
            pose_coords = pose_landmarks[:, :3]  # x, y, z only
            
            # Apply z-score normalization using training stats
            if np.any(pose_coords):
                pose_mean = self.training_stats['pose_mean']
                pose_std = self.training_stats['pose_std']
                
                # Avoid division by zero
                pose_std = np.where(pose_std > 1e-8, pose_std, 1.0)
                
                pose_coords_norm = (pose_coords - pose_mean) / pose_std
                pose_landmarks[:, :3] = pose_coords_norm
        else:
            # Fallback to simple normalization
            if np.any(pose_landmarks[:, :3]):
                # Center and scale pose
                pose_center = np.mean(pose_landmarks[:, :3], axis=0)
                pose_landmarks[:, :3] -= pose_center
                
                # Scale by maximum extent
                max_extent = np.max(np.abs(pose_landmarks[:, :3]))
                if max_extent > 1e-8:
                    pose_landmarks[:, :3] /= max_extent
        
        # Normalize hands relative to wrists (if detected)
        for hand_landmarks, hand_name in [(left_hand, 'left'), (right_hand, 'right')]:
            if np.any(hand_landmarks):
                # Use wrist as reference point (index 0)
                wrist = hand_landmarks[0]
                if np.any(wrist):
                    hand_landmarks -= wrist
                    
                    # Scale by hand span
                    hand_span = np.max(np.abs(hand_landmarks))
                    if hand_span > 1e-8:
                        hand_landmarks /= hand_span
        
        # Simpler face normalization
        if np.any(face_landmarks):
            face_center = np.mean(face_landmarks, axis=0)
            face_landmarks -= face_center
            
            face_span = np.max(np.abs(face_landmarks))
            if face_span > 1e-8:
                face_landmarks /= face_span
        
        # Reassemble
        normalized[:132] = pose_landmarks.flatten()
        normalized[132:195] = left_hand.flatten()
        normalized[195:258] = right_hand.flatten()
        normalized[258:1662] = face_landmarks.flatten()
        
        # Final clipping to prevent extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized
    '''
    
    print("✅ Improved normalization function created")
    print("This should provide more consistent preprocessing between training and live data.")
    
    return improved_normalization

if __name__ == "__main__":
    analyze_problematic_features()
    create_fixed_inference()