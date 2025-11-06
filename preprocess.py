"""
MediaPipe Holistic landmark extraction and preprocessing for ISL gesture recognition.
Extracts pose, hands, and face landmarks with scale-invariant normalization.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from typing import Optional, Tuple, List
import json
from tqdm import tqdm


# Configuration
SEQUENCE_LENGTH = 60  # 60 frames for faster processing (updated from 90)
PHRASES = [
    "Hi my name is Reet",
    "How are you", 
    "I am from Delhi",
    "I like coffee",
    "What do you like"
]

# Data directories
VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/sequences_holistic"


class HolisticPreprocessor:
    """MediaPipe Holistic preprocessing for ISL gesture recognition."""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Initialize holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for i in range(len(PHRASES)):
            os.makedirs(os.path.join(OUTPUT_DIR, f"phrase_{i}"), exist_ok=True)
    
    def extract_pose_landmarks(self, results) -> np.ndarray:
        """Extract pose landmarks (33 points * 4 features = 132 features)."""
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([
                    landmark.x, landmark.y, landmark.z, landmark.visibility
                ])
            return np.array(landmarks)
        else:
            return np.zeros(33 * 4)  # 132 features
    
    def extract_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract hand landmarks (21 points * 3 features = 63 features per hand)."""
        if hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        else:
            return np.zeros(21 * 3)  # 63 features
    
    def extract_face_landmarks(self, results) -> np.ndarray:
        """Extract face landmarks (468 points * 3 features = 1404 features)."""
        if results.face_landmarks:
            landmarks = []
            for landmark in results.face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            # Ensure we have exactly 1404 features (468 * 3)
            landmarks = np.array(landmarks)
            if len(landmarks) > 1404:
                landmarks = landmarks[:1404]
            elif len(landmarks) < 1404:
                # Pad with zeros if we have fewer landmarks
                landmarks = np.pad(landmarks, (0, 1404 - len(landmarks)))
            return landmarks
        else:
            return np.zeros(468 * 3)  # 1404 features
    
    def extract_holistic_landmarks(self, results) -> np.ndarray:
        """Extract all holistic landmarks into a single feature vector."""
        # Extract individual components
        pose_landmarks = self.extract_pose_landmarks(results)
        left_hand_landmarks = self.extract_hand_landmarks(results.left_hand_landmarks)
        right_hand_landmarks = self.extract_hand_landmarks(results.right_hand_landmarks)
        face_landmarks = self.extract_face_landmarks(results)
        
        # Concatenate all landmarks
        # Total: 132 (pose) + 63 (left hand) + 63 (right hand) + 1404 (face) = 1662 features
        all_landmarks = np.concatenate([
            pose_landmarks,
            left_hand_landmarks,
            right_hand_landmarks,
            face_landmarks
        ])
        
        return all_landmarks
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply scale-invariant normalization to landmarks."""
        # Ensure we have the expected number of features
        if len(landmarks) != 1662:
            print(f"Warning: Expected 1662 features, got {len(landmarks)}. Padding or truncating.")
            if len(landmarks) > 1662:
                landmarks = landmarks[:1662]
            else:
                landmarks = np.pad(landmarks, (0, 1662 - len(landmarks)))
        
        # Reshape to (num_points, features_per_point)
        # Pose: 33 points * 4 features, Hands: 21 points * 3 features each, Face: 468 points * 3 features
        
        # Extract pose landmarks (first 132 features)
        pose_landmarks = landmarks[:132].reshape(33, 4)
        
        # Extract hand landmarks
        left_hand = landmarks[132:195].reshape(21, 3)
        right_hand = landmarks[195:258].reshape(21, 3)
        
        # Extract face landmarks  
        face_landmarks = landmarks[258:1662].reshape(468, 3)
        
        # Normalize pose relative to torso center (approximate)
        if np.any(pose_landmarks[:, :3]):  # Check if pose data exists
            # Use shoulders and hips to define torso center
            left_shoulder = pose_landmarks[11, :3]
            right_shoulder = pose_landmarks[12, :3]
            left_hip = pose_landmarks[23, :3]
            right_hip = pose_landmarks[24, :3]
            
            torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
            
            # Calculate torso scale (shoulder width)
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            
            if shoulder_width > 0:
                # Normalize pose landmarks
                pose_landmarks[:, :3] = (pose_landmarks[:, :3] - torso_center) / shoulder_width
        
        # Normalize hands relative to wrist position and scale
        def normalize_hand(hand_landmarks, wrist_idx=0):
            if np.any(hand_landmarks):
                wrist_pos = hand_landmarks[wrist_idx]
                # Calculate hand scale (distance from wrist to middle finger tip)
                middle_tip = hand_landmarks[12]  # Middle finger tip
                hand_scale = np.linalg.norm(middle_tip - wrist_pos)
                
                if hand_scale > 0:
                    hand_landmarks = (hand_landmarks - wrist_pos) / hand_scale
            
            return hand_landmarks
        
        left_hand = normalize_hand(left_hand)
        right_hand = normalize_hand(right_hand)
        
        # Normalize face landmarks relative to nose tip
        if np.any(face_landmarks):
            nose_tip = face_landmarks[1]  # Nose tip landmark
            
            # Calculate face scale (distance between eye corners)
            left_eye_corner = face_landmarks[33]
            right_eye_corner = face_landmarks[263]
            face_scale = np.linalg.norm(left_eye_corner - right_eye_corner)
            
            if face_scale > 0:
                face_landmarks = (face_landmarks - nose_tip) / face_scale
        
        # Flatten and concatenate normalized landmarks
        normalized = np.concatenate([
            pose_landmarks.flatten(),
            left_hand.flatten(),
            right_hand.flatten(),
            face_landmarks.flatten()
        ])
        
        return normalized
    
    def process_video(self, video_path: str) -> Optional[np.ndarray]:
        """Process a single video and extract landmark sequences."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        landmarks_sequence = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Holistic
            results = self.holistic.process(rgb_frame)
            
            # Extract landmarks
            frame_landmarks = self.extract_holistic_landmarks(results)
            
            # Apply normalization
            normalized_landmarks = self.normalize_landmarks(frame_landmarks)
            
            landmarks_sequence.append(normalized_landmarks)
            frame_count += 1
        
        cap.release()
        
        if len(landmarks_sequence) == 0:
            print(f"Warning: No landmarks extracted from {video_path}")
            return None
        
        # Convert to numpy array
        landmarks_sequence = np.array(landmarks_sequence)
        
        # Resample to fixed sequence length
        landmarks_sequence = self.resample_sequence(landmarks_sequence, SEQUENCE_LENGTH)
        
        return landmarks_sequence
    
    def resample_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Resample sequence to target length using interpolation."""
        if len(sequence) == target_length:
            return sequence
        
        # Create indices for resampling
        original_indices = np.linspace(0, len(sequence) - 1, len(sequence))
        target_indices = np.linspace(0, len(sequence) - 1, target_length)
        
        # Interpolate each feature dimension
        resampled = np.zeros((target_length, sequence.shape[1]))
        
        for feature_idx in range(sequence.shape[1]):
            resampled[:, feature_idx] = np.interp(
                target_indices, 
                original_indices, 
                sequence[:, feature_idx]
            )
        
        return resampled
    
    def process_all_videos(self):
        """Process all videos and save landmark sequences."""
        print("Starting holistic landmark extraction...")
        print(f"Target sequence length: {SEQUENCE_LENGTH} frames")
        print(f"Output directory: {OUTPUT_DIR}")
        
        total_processed = 0
        
        for phrase_idx in range(len(PHRASES)):
            phrase_name = PHRASES[phrase_idx]
            video_dir = os.path.join(VIDEO_DIR, f"phrase_{phrase_idx}")
            output_dir = os.path.join(OUTPUT_DIR, f"phrase_{phrase_idx}")
            
            if not os.path.exists(video_dir):
                print(f"Warning: Video directory not found: {video_dir}")
                continue
            
            # Find all video files
            video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + \
                         glob.glob(os.path.join(video_dir, "*.avi")) + \
                         glob.glob(os.path.join(video_dir, "*.mov"))
            
            if not video_files:
                print(f"Warning: No video files found in {video_dir}")
                continue
            
            print(f"\nProcessing phrase {phrase_idx}: '{phrase_name}'")
            print(f"Found {len(video_files)} videos")
            
            phrase_processed = 0
            
            for video_file in tqdm(video_files, desc=f"Phrase {phrase_idx}"):
                # Extract sequence
                sequence = self.process_video(video_file)
                
                if sequence is not None:
                    # Save sequence
                    video_name = os.path.splitext(os.path.basename(video_file))[0]
                    output_path = os.path.join(output_dir, f"{video_name}_seq.npy")
                    
                    np.save(output_path, sequence)
                    phrase_processed += 1
                    total_processed += 1
            
            print(f"Processed {phrase_processed} videos for phrase {phrase_idx}")
        
        print(f"\n✓ Preprocessing complete!")
        print(f"Total sequences processed: {total_processed}")
        print(f"Sequence shape: ({SEQUENCE_LENGTH}, 1662)")
        print(f"Saved to: {OUTPUT_DIR}")
        
        # Save metadata
        metadata = {
            "sequence_length": SEQUENCE_LENGTH,
            "feature_dimensions": 1662,
            "phrases": PHRASES,
            "total_sequences": total_processed,
            "feature_breakdown": {
                "pose": 132,  # 33 points * 4 features
                "left_hand": 63,  # 21 points * 3 features  
                "right_hand": 63,  # 21 points * 3 features
                "face": 1404  # 468 points * 3 features
            }
        }
        
        metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Main preprocessing pipeline."""
    preprocessor = HolisticPreprocessor()
    preprocessor.process_all_videos()


if __name__ == "__main__":
    main()