#!/usr/bin/env python3
"""
Ultra Memory-Efficient Video Processor
Processes one video at a time to avoid memory issues
"""

import cv2
import numpy as np
import os
import sys
import json
from pathlib import Path
import gc
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import HolisticInference

def process_single_video(video_path, phrase_idx, inference, target_frames=60):
    """Process a single video with minimal memory usage"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        landmarks_sequence = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            landmarks, _ = inference.process_frame(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
            
            frame_count += 1
            
            # Limit frames to avoid too long sequences
            if len(landmarks_sequence) >= target_frames * 2:  # Max 120 frames
                break
        
        cap.release()
        
        # Normalize to target frames
        if len(landmarks_sequence) < target_frames // 2:  # Need at least 30 frames
            print(f"⚠️ Too few frames in {video_path.name}: {len(landmarks_sequence)}")
            return None
        
        # Interpolate to exact target frames
        if len(landmarks_sequence) != target_frames:
            landmarks_array = np.array(landmarks_sequence)
            old_indices = np.linspace(0, len(landmarks_sequence) - 1, len(landmarks_sequence))
            new_indices = np.linspace(0, len(landmarks_sequence) - 1, target_frames)
            
            normalized = np.zeros((target_frames, 1662))
            for i in range(1662):
                normalized[:, i] = np.interp(new_indices, old_indices, landmarks_array[:, i])
            
            landmarks_sequence = normalized
        else:
            landmarks_sequence = np.array(landmarks_sequence)
        
        return landmarks_sequence
        
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()
        gc.collect()  # Force garbage collection

def main():
    """Main processing function - ultra memory efficient"""
    print("🔄 Ultra Memory-Efficient Video Processing")
    print("="*50)
    
    # Initialize inference once
    print("Initializing HolisticInference...")
    inference = HolisticInference()
    
    # Video directories
    video_base = Path("data/videos")
    output_base = Path("data/sequences_multi_person")
    output_base.mkdir(exist_ok=True)
    
    phrase_mapping = {
        "phrase_0": 0,  # Hi my name is Reet
        "phrase_1": 1,  # How are you
        "phrase_2": 2,  # I am from Delhi
        "phrase_3": 3,  # I like coffee
        "phrase_4": 4   # What do you like
    }
    
    total_processed = 0
    total_videos = 0
    
    # Count total videos first
    for phrase_dir in video_base.iterdir():
        if phrase_dir.is_dir() and phrase_dir.name in phrase_mapping:
            video_files = list(phrase_dir.glob("*.mp4"))
            total_videos += len(video_files)
    
    print(f"📊 Found {total_videos} videos to process")
    
    # Process each phrase directory
    for phrase_dir in video_base.iterdir():
        if not phrase_dir.is_dir() or phrase_dir.name not in phrase_mapping:
            continue
        
        phrase_idx = phrase_mapping[phrase_dir.name]
        phrase_name = ["Hi my name is Reet", "How are you", "I am from Delhi", 
                      "I like coffee", "What do you like"][phrase_idx]
        
        print(f"\n📁 Processing {phrase_dir.name}: {phrase_name}")
        
        # Create output directory
        output_dir = output_base / phrase_dir.name
        output_dir.mkdir(exist_ok=True)
        
        # Get video files
        video_files = list(phrase_dir.glob("*.mp4"))
        print(f"   Found {len(video_files)} videos")
        
        # Process each video individually
        for i, video_file in enumerate(video_files):
            print(f"   Processing {i+1}/{len(video_files)}: {video_file.name}")
            
            # Generate output filename
            output_file = output_dir / f"{video_file.stem}_seq.npy"
            
            # Skip if already processed
            if output_file.exists():
                print(f"   ✓ Already processed: {output_file.name}")
                total_processed += 1
                continue
            
            # Process video
            sequence = process_single_video(video_file, phrase_idx, inference)
            
            if sequence is not None:
                # Save immediately
                np.save(output_file, sequence)
                print(f"   ✅ Saved: {output_file.name} (shape: {sequence.shape})")
                total_processed += 1
            else:
                print(f"   ❌ Failed: {video_file.name}")
            
            # Force garbage collection after each video
            gc.collect()
            
            # Progress update
            if total_processed % 10 == 0:
                print(f"\n📈 Progress: {total_processed}/{total_videos} videos processed")
    
    print(f"\n🎉 Processing Complete!")
    print(f"✅ Successfully processed: {total_processed}/{total_videos} videos")
    
    # Summary
    print(f"\n📊 Summary:")
    for phrase_dir in output_base.iterdir():
        if phrase_dir.is_dir():
            seq_files = list(phrase_dir.glob("*.npy"))
            phrase_name = ["Hi my name is Reet", "How are you", "I am from Delhi", 
                          "I like coffee", "What do you like"][int(phrase_dir.name.split('_')[1])]
            print(f"   {phrase_name}: {len(seq_files)} sequences")

if __name__ == "__main__":
    main()