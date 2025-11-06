#!/usr/bin/env python3
"""
Optimized Multi-Person Video Processing with Multiprocessing
Processes existing video data efficiently with memory management
"""

import cv2
import numpy as np
import os
import sys
import joblib
import json
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial
import gc

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import HolisticInference

def process_single_video(video_path, target_frames=60):
    """Process a single video file and extract landmarks."""
    try:
        # Initialize HolisticInference for this process
        inference = HolisticInference()
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, f"Cannot open {video_path}"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None, f"No frames in {video_path}"
        
        landmarks_sequence = []
        frame_skip = max(1, total_frames // target_frames) if total_frames > target_frames else 1
        
        frame_idx = 0
        processed_frames = 0
        
        while processed_frames < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to get roughly target_frames
            if frame_idx % frame_skip == 0:
                try:
                    landmarks, _ = inference.process_frame(frame)
                    if landmarks is not None:
                        landmarks_sequence.append(landmarks)
                        processed_frames += 1
                except Exception as e:
                    print(f"Error processing frame {frame_idx} in {video_path}: {e}")
            
            frame_idx += 1
        
        cap.release()
        
        if len(landmarks_sequence) < target_frames:
            # Pad with last frame if needed
            if landmarks_sequence:
                last_frame = landmarks_sequence[-1]
                while len(landmarks_sequence) < target_frames:
                    landmarks_sequence.append(last_frame.copy())
            else:
                return None, f"No valid landmarks extracted from {video_path}"
        elif len(landmarks_sequence) > target_frames:
            # Truncate if too many
            landmarks_sequence = landmarks_sequence[:target_frames]
        
        sequence = np.array(landmarks_sequence)
        return sequence, None
        
    except Exception as e:
        return None, f"Error processing {video_path}: {e}"

def process_video_batch(video_paths_and_labels, process_id):
    """Process a batch of videos in a single process."""
    print(f"Process {process_id}: Processing {len(video_paths_and_labels)} videos")
    
    sequences = []
    labels = []
    errors = []
    
    for video_path, phrase_idx in video_paths_and_labels:
        sequence, error = process_single_video(video_path)
        
        if sequence is not None:
            sequences.append(sequence.flatten())  # Flatten for model
            labels.append(phrase_idx)
        else:
            errors.append(error)
    
    print(f"Process {process_id}: Completed {len(sequences)} videos successfully, {len(errors)} errors")
    return sequences, labels, errors

def collect_video_files():
    """Collect all video files organized by phrase."""
    video_files = {i: [] for i in range(5)}
    
    for phrase_idx in range(5):
        phrase_dir = Path(f"data/videos/phrase_{phrase_idx}")
        if phrase_dir.exists():
            for ext in ['*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV', '*.AVI']:
                video_files[phrase_idx].extend(phrase_dir.glob(ext))
    
    return video_files

def main():
    """Main processing function with multiprocessing."""
    print("🎯 Optimized Multi-Person Video Processing")
    print("="*50)
    
    # Get CPU count
    cpu_count = mp.cpu_count()
    max_workers = min(cpu_count - 1, 8)  # Leave one core free, max 8 processes
    print(f"💻 Using {max_workers} processes (CPU count: {cpu_count})")
    
    # Collect video files
    print("📁 Collecting video files...")
    video_files = collect_video_files()
    
    total_videos = sum(len(files) for files in video_files.values())
    print(f"📊 Found {total_videos} videos:")
    for phrase_idx, files in video_files.items():
        phrase_names = [
            "Hi my name is Reet",
            "How are you", 
            "I am from Delhi",
            "I like coffee",
            "What do you like"
        ]
        print(f"  Phrase {phrase_idx} ({phrase_names[phrase_idx]}): {len(files)} videos")
    
    if total_videos == 0:
        print("❌ No video files found")
        return
    
    # Prepare work batches
    print("\n📦 Preparing work batches...")
    all_tasks = []
    for phrase_idx, files in video_files.items():
        for file_path in files:
            all_tasks.append((file_path, phrase_idx))
    
    # Split tasks into batches for each process
    batch_size = max(1, len(all_tasks) // max_workers)
    batches = []
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i + batch_size]
        batches.append(batch)
    
    print(f"📦 Created {len(batches)} batches with ~{batch_size} videos each")
    
    # Process in parallel
    print(f"\n🚀 Starting parallel processing...")
    start_time = time.time()
    
    all_sequences = []
    all_labels = []
    all_errors = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {}
        for i, batch in enumerate(batches):
            future = executor.submit(process_video_batch, batch, i)
            future_to_batch[future] = i
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                sequences, labels, errors = future.result()
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                all_errors.extend(errors)
                
                completed += 1
                progress = (completed / len(batches)) * 100
                print(f"✅ Batch {batch_id} completed ({completed}/{len(batches)}, {progress:.1f}%)")
                
            except Exception as e:
                print(f"❌ Batch {batch_id} failed: {e}")
    
    processing_time = time.time() - start_time
    print(f"\n⏱️ Processing completed in {processing_time:.1f} seconds")
    
    if all_errors:
        print(f"⚠️ {len(all_errors)} errors occurred:")
        for error in all_errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(all_errors) > 5:
            print(f"  ... and {len(all_errors) - 5} more")
    
    if not all_sequences:
        print("❌ No sequences processed successfully")
        return
    
    print(f"\n✅ Successfully processed {len(all_sequences)} sequences")
    
    # Convert to numpy arrays
    print("🔄 Converting to training format...")
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    print(f"📊 Final dataset shape: {X.shape}")
    for phrase_idx in range(5):
        count = np.sum(y == phrase_idx)
        phrase_names = [
            "Hi my name is Reet",
            "How are you", 
            "I am from Delhi",
            "I like coffee",
            "What do you like"
        ]
        print(f"  Phrase {phrase_idx} ({phrase_names[phrase_idx]}): {count} sequences")
    
    # Save processed data
    print("\n💾 Saving processed data...")
    os.makedirs("data/processed_multi_person", exist_ok=True)
    
    np.save("data/processed_multi_person/X_multi_person.npy", X)
    np.save("data/processed_multi_person/y_multi_person.npy", y)
    
    # Also save metadata
    metadata = {
        "total_sequences": len(all_sequences),
        "total_videos_processed": total_videos - len(all_errors),
        "total_errors": len(all_errors),
        "processing_time_seconds": processing_time,
        "target_frames": 60,
        "features_per_frame": 1662,
        "phrase_distribution": {i: int(np.sum(y == i)) for i in range(5)}
    }
    
    with open("data/processed_multi_person/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Data saved to data/processed_multi_person/")
    print(f"📁 Files created:")
    print(f"  - X_multi_person.npy: {X.shape}")
    print(f"  - y_multi_person.npy: {y.shape}")
    print(f"  - metadata.json")
    
    print(f"\n🎉 Processing complete! Ready for retraining.")
    print(f"Next step: Run the retraining script to use this multi-person data")

if __name__ == "__main__":
    main()