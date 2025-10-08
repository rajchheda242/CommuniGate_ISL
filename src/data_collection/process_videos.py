"""
Process pre-recorded MP4 videos to extract hand landmark sequences.
Designed for batch processing of videos from multiple people.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm


# Configuration
VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/sequences"
TARGET_FRAMES = 60  # Normalize all sequences to 60 frames

PHRASES = [
    "Hi, my name is Madiha Siddiqui.",
    "I am a student.",
    "I enjoy running as a hobby.",
    "How are you doing today?"
]


class VideoProcessor:
    """Process MP4 videos to extract temporal hand landmark sequences."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for i in range(len(PHRASES)):
            os.makedirs(os.path.join(OUTPUT_DIR, f"phrase_{i}"), exist_ok=True)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from Mediapipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def process_video(self, video_path, save_preview=False):
        """
        Process a single video file to extract landmark sequence.
        
        Args:
            video_path: Path to the video file
            save_preview: If True, saves a preview video with landmarks drawn
            
        Returns:
            numpy array of shape (TARGET_FRAMES, 126) or None if processing failed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing: {Path(video_path).name}")
        print(f"  FPS: {fps:.1f} | Total frames: {total_frames}")
        
        sequence_frames = []
        frame_count = 0
        frames_with_hands = 0
        
        # Setup for preview video (optional)
        preview_writer = None
        if save_preview:
            preview_path = video_path.replace('.mp4', '_preview.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            preview_writer = cv2.VideoWriter(preview_path, fourcc, fps, 
                                            (frame_width, frame_height))
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
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
                    
                    # Draw landmarks for preview (optional)
                    if save_preview:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS
                            )
                    
                    # Extract all hand landmarks
                    for hand_landmarks in results.multi_hand_landmarks:
                        frame_landmarks.extend(self.extract_landmarks(hand_landmarks))
                
                # Pad or truncate to fixed size (2 hands * 21 landmarks * 3 coords = 126)
                while len(frame_landmarks) < 126:
                    frame_landmarks.append(0.0)
                frame_landmarks = frame_landmarks[:126]
                
                sequence_frames.append(frame_landmarks)
                
                # Save preview frame
                if save_preview and preview_writer:
                    preview_writer.write(frame)
                
                frame_count += 1
        
        cap.release()
        if preview_writer:
            preview_writer.release()
            print(f"  Preview saved: {preview_path}")
        
        # Check if we got useful data
        hand_detection_rate = frames_with_hands / max(frame_count, 1)
        print(f"  Frames processed: {frame_count}")
        print(f"  Hand detection rate: {hand_detection_rate:.1%}")
        
        if hand_detection_rate < 0.3:
            print(f"  ⚠️  WARNING: Low hand detection rate! Check video quality.")
        
        if frame_count == 0:
            print(f"  ❌ ERROR: No frames extracted!")
            return None
        
        # Normalize sequence length
        sequence = np.array(sequence_frames)
        normalized_sequence = self.normalize_sequence_length(sequence, TARGET_FRAMES)
        
        print(f"  ✓ Sequence shape: {normalized_sequence.shape}")
        
        return normalized_sequence
    
    def normalize_sequence_length(self, sequence, target_length):
        """
        Normalize sequence to target length using interpolation.
        
        Args:
            sequence: numpy array of shape (n_frames, n_features)
            target_length: desired number of frames
            
        Returns:
            numpy array of shape (target_length, n_features)
        """
        current_length = len(sequence)
        
        if current_length == target_length:
            return sequence
        
        # Use linear interpolation to resample
        indices = np.linspace(0, current_length - 1, target_length)
        normalized = np.zeros((target_length, sequence.shape[1]))
        
        for i in range(sequence.shape[1]):
            normalized[:, i] = np.interp(indices, np.arange(current_length), sequence[:, i])
        
        return normalized
    
    def process_all_videos(self, save_previews=False):
        """
        Process all videos in the VIDEO_DIR directory structure.
        
        Expected structure:
        data/videos/
            phrase_0/
                video1.mp4
                video2.mp4
            phrase_1/
                video1.mp4
                ...
        """
        print("="*70)
        print("VIDEO PROCESSING - Extract Landmark Sequences from MP4 Files")
        print("="*70)
        
        total_processed = 0
        total_failed = 0
        
        for phrase_idx in range(len(PHRASES)):
            phrase_dir = os.path.join(VIDEO_DIR, f"phrase_{phrase_idx}")
            
            if not os.path.exists(phrase_dir):
                print(f"\n⚠️  No directory found: {phrase_dir}")
                print(f"   Skipping phrase {phrase_idx}")
                continue
            
            # Find all video files
            video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(phrase_dir, ext)))
            
            if not video_files:
                print(f"\n⚠️  No video files found in {phrase_dir}")
                continue
            
            print(f"\n{'='*70}")
            print(f"Phrase {phrase_idx}: '{PHRASES[phrase_idx]}'")
            print(f"Found {len(video_files)} video(s)")
            print(f"{'='*70}")
            
            phrase_processed = 0
            phrase_failed = 0
            
            for video_idx, video_path in enumerate(tqdm(video_files, desc=f"Phrase {phrase_idx}")):
                sequence = self.process_video(video_path, save_preview=save_previews)
                
                if sequence is not None:
                    # Save sequence
                    video_name = Path(video_path).stem
                    output_path = os.path.join(
                        OUTPUT_DIR,
                        f"phrase_{phrase_idx}",
                        f"{video_name}_seq.npy"
                    )
                    np.save(output_path, sequence)
                    phrase_processed += 1
                    total_processed += 1
                else:
                    phrase_failed += 1
                    total_failed += 1
            
            print(f"\nPhrase {phrase_idx} Summary:")
            print(f"  ✓ Successfully processed: {phrase_processed}")
            print(f"  ✗ Failed: {phrase_failed}")
        
        # Final summary
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total videos processed: {total_processed}")
        print(f"Total failed: {total_failed}")
        print(f"\nSequences saved to: {OUTPUT_DIR}/")
        
        if total_processed > 0:
            print("\n✓ Ready for training!")
            print("\nNext steps:")
            print("  1. Review the sequences to ensure quality")
            print("  2. Run: python src/training/train_sequence_model.py")
        else:
            print("\n❌ No sequences were created. Please check:")
            print("  - Video file locations")
            print("  - Video quality and hand visibility")
            print("  - File formats (should be .mp4, .mov, or .avi)")
    
    def print_directory_structure(self):
        """Print expected directory structure for user guidance."""
        print("\nExpected Directory Structure:")
        print("="*70)
        print("data/videos/")
        print("  ├── phrase_0/  (Hi, my name is Madiha Siddiqui)")
        print("  │   ├── person1_video01.mp4")
        print("  │   ├── person1_video02.mp4")
        print("  │   ├── ...")
        print("  │   └── person5_video10.mp4  (50 videos total)")
        print("  ├── phrase_1/  (I am a student)")
        print("  │   └── ... (50 videos)")
        print("  ├── phrase_2/  (I enjoy running as a hobby)")
        print("  │   └── ... (50 videos)")
        print("  └── phrase_3/  (How are you doing today?)")
        print("      └── ... (50 videos)")
        print("="*70)


def main():
    processor = VideoProcessor()
    
    # Show expected structure
    processor.print_directory_structure()
    
    # Check if video directory exists
    if not os.path.exists(VIDEO_DIR):
        print(f"\n❌ Error: Video directory not found: {VIDEO_DIR}")
        print(f"\nCreating directory structure...")
        os.makedirs(VIDEO_DIR, exist_ok=True)
        for i in range(len(PHRASES)):
            os.makedirs(os.path.join(VIDEO_DIR, f"phrase_{i}"), exist_ok=True)
        print(f"✓ Directories created. Please add your MP4 videos and run again.")
        return
    
    # Ask about preview generation
    print("\nOptions:")
    print("  1. Process videos (extract sequences only)")
    print("  2. Process videos + generate preview videos with landmarks")
    
    choice = input("\nSelect option (1 or 2, default=1): ").strip() or "1"
    save_previews = (choice == "2")
    
    if save_previews:
        print("\n⚠️  Preview generation will take longer but helps verify quality")
    
    # Process all videos
    processor.process_all_videos(save_previews=save_previews)


if __name__ == "__main__":
    main()
