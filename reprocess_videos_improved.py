#!/usr/bin/env python3
"""
Reprocess existing videos with LOWER MediaPipe confidence thresholds.
This will detect hands more aggressively, reducing zero frames.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm

VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/sequences_reprocessed"  # New directory to compare
TARGET_FRAMES = 90

PHRASES = [
    "Hi my name is Reet",
    "How are you",
    "I am from Delhi",
    "I like coffee",
    "What do you like"
]

class ImprovedVideoProcessor:
    """Reprocess videos with lower detection thresholds."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_landmarks(self, hand_landmarks):
        """Extract x,y,z coordinates for all 21 landmarks."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def process_video(self, video_path, verbose=False):
        """Process video with LOWER confidence thresholds."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            if verbose:
                print(f"Error: Could not open {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if verbose:
            print(f"\nProcessing: {Path(video_path).name}")
            print(f"  FPS: {fps:.1f} | Total frames: {total_frames}")
        
        sequence_frames = []
        frame_count = 0
        frames_with_hands = 0
        
        # LOWER THRESHOLDS - more aggressive detection
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Lower from 0.5
            min_tracking_confidence=0.3,   # Lower from 0.5
            model_complexity=1              # Use more complex model
        ) as hands:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Extract landmarks
                frame_landmarks = []
                if results.multi_hand_landmarks:
                    frames_with_hands += 1
                    
                    # Extract all hand landmarks
                    for hand_landmarks in results.multi_hand_landmarks:
                        frame_landmarks.extend(self.extract_landmarks(hand_landmarks))
                
                # Pad or truncate to 126 features (2 hands × 21 landmarks × 3 coords)
                while len(frame_landmarks) < 126:
                    frame_landmarks.append(0.0)
                frame_landmarks = frame_landmarks[:126]
                
                sequence_frames.append(frame_landmarks)
                frame_count += 1
        
        cap.release()
        
        # Check detection rate
        hand_detection_rate = frames_with_hands / max(frame_count, 1)
        
        if verbose:
            print(f"  Frames processed: {frame_count}")
            print(f"  Hand detection rate: {hand_detection_rate:.1%}")
            
            if hand_detection_rate < 0.5:
                print(f"  ⚠️  WARNING: Still low detection - check video quality")
        
        if frame_count == 0:
            if verbose:
                print(f"  ❌ ERROR: No frames extracted!")
            return None
        
        # Normalize to TARGET_FRAMES
        sequence = np.array(sequence_frames)
        normalized_sequence = self.normalize_sequence_length(sequence, TARGET_FRAMES)
        
        # Calculate zero frame percentage
        zero_frames = np.all(normalized_sequence == 0, axis=1).sum()
        zero_pct = zero_frames / len(normalized_sequence) * 100
        
        if verbose:
            print(f"  ✓ Sequence shape: {normalized_sequence.shape}")
            print(f"  Zero frames: {zero_frames}/{len(normalized_sequence)} ({zero_pct:.1f}%)")
            
            if zero_pct < 10:
                print(f"  ✅ EXCELLENT quality!")
            elif zero_pct < 20:
                print(f"  ✅ GOOD quality")
            elif zero_pct < 40:
                print(f"  ⚠️  FAIR quality")
            else:
                print(f"  ⚠️  POOR quality - consider re-recording")
        
        return normalized_sequence
    
    def normalize_sequence_length(self, sequence, target_length):
        """Normalize sequence to target length using interpolation."""
        current_length = len(sequence)
        
        if current_length == target_length:
            return sequence
        
        # Use linear interpolation
        indices = np.linspace(0, current_length - 1, target_length)
        normalized = np.zeros((target_length, sequence.shape[1]))
        
        for i in range(sequence.shape[1]):
            normalized[:, i] = np.interp(indices, np.arange(current_length), sequence[:, i])
        
        return normalized
    
    def process_all_videos(self):
        """Process all videos with improved detection."""
        print("="*70)
        print("REPROCESSING VIDEOS WITH IMPROVED HAND DETECTION")
        print("="*70)
        print(f"\nSettings:")
        print(f"  min_detection_confidence: 0.3 (was 0.5)")
        print(f"  min_tracking_confidence: 0.3 (was 0.5)")
        print(f"  model_complexity: 1 (more accurate)")
        print(f"\nThis should reduce zero frames significantly!")
        print("="*70)
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for i in range(len(PHRASES)):
            os.makedirs(os.path.join(OUTPUT_DIR, f"phrase_{i}"), exist_ok=True)
        
        total_processed = 0
        total_failed = 0
        improvement_stats = []
        
        for phrase_idx in range(len(PHRASES)):
            video_dir = os.path.join(VIDEO_DIR, f"phrase_{phrase_idx}")
            
            if not os.path.exists(video_dir):
                print(f"\n⚠️  No videos found for phrase {phrase_idx}")
                continue
            
            video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
            
            if not video_files:
                print(f"\n⚠️  No MP4 files in {video_dir}")
                continue
            
            print(f"\n{'='*70}")
            print(f"Phrase {phrase_idx}: {PHRASES[phrase_idx]}")
            print(f"{'='*70}")
            print(f"Found {len(video_files)} videos\n")
            
            phrase_improvements = []
            
            for video_path in tqdm(video_files, desc=f"Processing phrase {phrase_idx}"):
                sequence = self.process_video(video_path, verbose=False)
                
                if sequence is not None:
                    # Save sequence
                    video_name = Path(video_path).stem
                    output_path = os.path.join(
                        OUTPUT_DIR,
                        f"phrase_{phrase_idx}",
                        f"{video_name}_seq.npy"
                    )
                    np.save(output_path, sequence)
                    
                    # Calculate improvement
                    zero_frames_new = np.all(sequence == 0, axis=1).sum()
                    
                    # Load old sequence if exists
                    old_path = os.path.join(
                        "data/sequences",
                        f"phrase_{phrase_idx}",
                        f"{video_name}_seq.npy"
                    )
                    if os.path.exists(old_path):
                        old_seq = np.load(old_path)
                        zero_frames_old = np.all(old_seq == 0, axis=1).sum()
                        improvement = zero_frames_old - zero_frames_new
                        phrase_improvements.append(improvement)
                    
                    total_processed += 1
                else:
                    total_failed += 1
            
            # Print phrase summary
            if phrase_improvements:
                avg_improvement = np.mean(phrase_improvements)
                print(f"\n✓ Processed {len(video_files)} videos")
                print(f"  Average zero frame reduction: {avg_improvement:.1f} frames per sequence")
                improvement_stats.extend(phrase_improvements)
        
        # Final summary
        print(f"\n{'='*70}")
        print("REPROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total videos processed: {total_processed}/{total_processed + total_failed}")
        print(f"Total failed: {total_failed}")
        
        if improvement_stats:
            avg_improvement = np.mean(improvement_stats)
            print(f"\n📊 Overall Improvement:")
            print(f"  Average zero frame reduction: {avg_improvement:.1f} frames")
            print(f"  Total reduction: {sum(improvement_stats)} zero frames eliminated")
            print(f"\nSequences saved to: {OUTPUT_DIR}/")
            print(f"\nNext steps:")
            print(f"  1. Run: python quick_data_quality_check.py")
            print(f"     (update DATA_DIR to '{OUTPUT_DIR}' in the script)")
            print(f"  2. If quality is good, replace old sequences:")
            print(f"     mv data/sequences data/sequences_old")
            print(f"     mv {OUTPUT_DIR} data/sequences")
            print(f"  3. Retrain model: python enhanced_train.py")

if __name__ == "__main__":
    processor = ImprovedVideoProcessor()
    processor.process_all_videos()
