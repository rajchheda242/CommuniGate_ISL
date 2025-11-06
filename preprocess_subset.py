"""
Process a subset of videos for faster testing.
"""

from preprocess import HolisticPreprocessor
import os
import glob
import numpy as np
from tqdm import tqdm

# Process only first 10 videos per phrase for testing
MAX_VIDEOS_PER_PHRASE = 10

def process_subset():
    processor = HolisticPreprocessor()
    
    print(f"Processing subset: {MAX_VIDEOS_PER_PHRASE} videos per phrase")
    
    for phrase_idx in range(5):
        phrase_name = processor.PHRASES[phrase_idx] if hasattr(processor, 'PHRASES') else f"Phrase {phrase_idx}"
        video_dir = os.path.join("data/videos", f"phrase_{phrase_idx}")
        output_dir = os.path.join("data/sequences_holistic", f"phrase_{phrase_idx}")
        
        if not os.path.exists(video_dir):
            print(f"Warning: Video directory not found: {video_dir}")
            continue
        
        # Find all video files
        video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + \
                     glob.glob(os.path.join(video_dir, "*.avi")) + \
                     glob.glob(os.path.join(video_dir, "*.mov")) + \
                     glob.glob(os.path.join(video_dir, "*.MOV"))
        
        # Take only first MAX_VIDEOS_PER_PHRASE videos
        video_files = video_files[:MAX_VIDEOS_PER_PHRASE]
        
        if not video_files:
            print(f"Warning: No video files found in {video_dir}")
            continue
        
        print(f"\\nProcessing phrase {phrase_idx}: '{phrase_name}'")
        print(f"Processing {len(video_files)} videos")
        
        phrase_processed = 0
        
        for video_file in tqdm(video_files, desc=f"Phrase {phrase_idx}"):
            # Extract sequence
            sequence = processor.process_video(video_file)
            
            if sequence is not None:
                # Save sequence
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                output_path = os.path.join(output_dir, f"{video_name}_seq.npy")
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                np.save(output_path, sequence)
                phrase_processed += 1
        
        print(f"Processed {phrase_processed} videos for phrase {phrase_idx}")
    
    print(f"\\n✓ Subset preprocessing complete!")

if __name__ == "__main__":
    process_subset()