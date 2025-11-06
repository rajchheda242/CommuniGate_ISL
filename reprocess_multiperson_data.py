#!/usr/bin/env python3
"""
Reprocess all video data with HolisticInference for multi-person training
"""

import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import HolisticInference

def process_video_to_sequence(video_path, target_frames=60):
    """Process a video file into a sequence using HolisticInference"""
    print(f"Processing: {video_path}")
    
    # Initialize inference
    inference = HolisticInference()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Video info: {total_frames} frames, {fps:.1f} FPS")
    
    # Extract landmarks from all frames
    landmarks_sequence = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with HolisticInference
        try:
            landmarks, _ = inference.process_frame(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
        except Exception as e:
            print(f"  Warning: Frame {frame_count} failed: {e}")
            
        frame_count += 1
        
        # Show progress for long videos
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    if len(landmarks_sequence) < target_frames:
        print(f"  ❌ Too few landmarks: {len(landmarks_sequence)} < {target_frames}")
        return None
    
    # Normalize to target frames using interpolation
    if len(landmarks_sequence) != target_frames:
        print(f"  Normalizing {len(landmarks_sequence)} frames to {target_frames}")
        landmarks_array = np.array(landmarks_sequence)
        
        # Interpolate to target frames
        old_indices = np.linspace(0, len(landmarks_sequence) - 1, len(landmarks_sequence))
        new_indices = np.linspace(0, len(landmarks_sequence) - 1, target_frames)
        
        normalized_sequence = np.zeros((target_frames, 1662))
        for i in range(1662):
            normalized_sequence[:, i] = np.interp(new_indices, old_indices, landmarks_array[:, i])
        
        landmarks_sequence = normalized_sequence
    else:
        landmarks_sequence = np.array(landmarks_sequence)
    
    print(f"  ✅ Generated sequence: {landmarks_sequence.shape}")
    return landmarks_sequence

def reprocess_all_videos():
    """Reprocess all videos with HolisticInference"""
    print("🔄 Reprocessing all videos with HolisticInference")
    print("="*60)
    
    video_dir = "data/videos"
    output_dir = "data/sequences_holistic"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    phrases = [
        "Hi my name is Reet",
        "How are you", 
        "I am from Delhi",
        "I like coffee",
        "What do you like"
    ]
    
    total_processed = 0
    
    for phrase_idx in range(5):
        phrase_name = phrases[phrase_idx]
        phrase_video_dir = os.path.join(video_dir, f"phrase_{phrase_idx}")
        phrase_output_dir = os.path.join(output_dir, f"phrase_{phrase_idx}")
        
        print(f"\n📁 Processing Phrase {phrase_idx}: {phrase_name}")
        print(f"Input: {phrase_video_dir}")
        print(f"Output: {phrase_output_dir}")
        
        if not os.path.exists(phrase_video_dir):
            print(f"❌ Video directory not found: {phrase_video_dir}")
            continue
        
        # Create output directory
        os.makedirs(phrase_output_dir, exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(phrase_video_dir).glob(f"*{ext}"))
            video_files.extend(Path(phrase_video_dir).glob(f"*{ext.upper()}"))
        
        print(f"Found {len(video_files)} videos")
        
        processed_count = 0
        for video_file in video_files:
            video_path = str(video_file)
            
            # Generate output filename
            video_name = video_file.stem
            output_filename = f"{video_name}_holistic_seq.npy"
            output_path = os.path.join(phrase_output_dir, output_filename)
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"  ⏭️  Skipping (already exists): {output_filename}")
                processed_count += 1
                continue
            
            # Process video
            sequence = process_video_to_sequence(video_path)
            
            if sequence is not None:
                # Save sequence
                np.save(output_path, sequence)
                print(f"  ✅ Saved: {output_filename}")
                processed_count += 1
            else:
                print(f"  ❌ Failed: {video_name}")
        
        print(f"Phrase {phrase_idx} complete: {processed_count}/{len(video_files)} videos processed")
        total_processed += processed_count
    
    print(f"\n🎉 Reprocessing complete!")
    print(f"Total sequences generated: {total_processed}")
    print(f"Output directory: {output_dir}")

def retrain_with_holistic_data():
    """Retrain model with all holistic sequences"""
    print(f"\n🏋️ Retraining model with holistic data")
    print("="*50)
    
    from tensorflow import keras
    import joblib
    import json
    
    # Load existing model and scaler
    model = keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    # Collect all training data
    all_sequences = []
    all_labels = []
    
    data_sources = [
        ("data/sequences_holistic", "multi-person holistic", 1),  # Original weight
        ("data/sequences_current_env", "current environment", 2)  # Slight boost for current env
    ]
    
    for data_dir, desc, weight in data_sources:
        if not os.path.exists(data_dir):
            print(f"⚠️  Skipping {desc}: {data_dir} not found")
            continue
            
        print(f"\n📂 Loading {desc} data...")
        
        for phrase_idx in range(5):
            phrase_dir = os.path.join(data_dir, f"phrase_{phrase_idx}")
            if not os.path.exists(phrase_dir):
                continue
                
            files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
            loaded_count = 0
            
            for file in files:
                filepath = os.path.join(phrase_dir, file)
                try:
                    sequence = np.load(filepath)
                    if sequence.shape == (60, 1662):
                        # Add multiple copies for weighting
                        for _ in range(weight):
                            all_sequences.append(sequence.flatten())
                            all_labels.append(phrase_idx)
                        loaded_count += 1
                    else:
                        print(f"  ⚠️  Skipping {file}: wrong shape {sequence.shape}")
                except Exception as e:
                    print(f"  ❌ Failed to load {file}: {e}")
            
            print(f"  Phrase {phrase_idx}: {loaded_count} files × {weight} weight = {loaded_count * weight} samples")
    
    if not all_sequences:
        print("❌ No training data found")
        return
    
    # Prepare training data
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    # Convert to categorical
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y, num_classes=5)
    
    print(f"\n📊 Final training data: {X.shape[0]} samples")
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    for i in range(5):
        count = np.sum(y == i)
        print(f"  Phrase {i} ({phrases[i]}): {count} samples")
    
    # Scale data (X is already flattened)
    X_scaled = scaler.transform(X)
    X_lstm = X_scaled.reshape(X_scaled.shape[0], 60, 1662)
    
    # Train model
    print("\n🏋️ Training model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_lstm, y_categorical,
        epochs=15,
        batch_size=16,
        verbose=1,
        validation_split=0.2,
        shuffle=True
    )
    
    # Save updated model
    model.save('models/saved/lstm_model.keras')
    print("✅ Model updated and saved")
    
    # Test accuracy
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"📈 Final accuracy: {final_accuracy:.3f}")
    print(f"📈 Final validation accuracy: {final_val_accuracy:.3f}")

def main():
    """Main function"""
    print("🔄 Multi-Person Model Retraining")
    print("="*60)
    print("This will:")
    print("1. Reprocess all videos with HolisticInference")
    print("2. Retrain model with multi-person data")
    print("3. Create a generalized model for all users")
    print("="*60)
    
    choice = input("\nProceed? (y/n): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Reprocess videos
    reprocess_all_videos()
    
    # Step 2: Retrain model
    retrain_with_holistic_data()
    
    print("\n🎉 Multi-person retraining complete!")
    print("The model should now work better for different people.")

if __name__ == "__main__":
    main()