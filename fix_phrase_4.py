#!/usr/bin/env python3
"""
Quick fix for "What do you like" phrase confusion
Collect additional training data and retrain
"""

import cv2
import numpy as np
import os
import sys
import joblib
import json
from tensorflow import keras

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import HolisticInference

def collect_additional_samples(phrase_idx=4, phrase_name="What do you like", num_samples=10):
    """Collect additional samples for the problematic phrase"""
    print(f"🎯 Collecting Additional Training Data")
    print(f"Phrase: {phrase_name}")
    print(f"We'll collect {num_samples} samples to improve recognition")
    print("="*50)
    
    # Create directory
    phrase_dir = f"data/sequences_current_env/phrase_{phrase_idx}"
    os.makedirs(phrase_dir, exist_ok=True)
    
    # Initialize inference
    inference = HolisticInference()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return []
    
    collected_sequences = []
    
    for sample_idx in range(num_samples):
        print(f"\n📹 Sample {sample_idx+1}/{num_samples}")
        print(f"Perform: {phrase_name}")
        print("Press 's' to start recording, 'q' to quit")
        
        landmarks_sequence = []
        recording = False
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            if recording:
                # Process frame using HolisticInference
                landmarks, _ = inference.process_frame(frame)
                landmarks_sequence.append(landmarks)
                frame_count += 1
                
                cv2.putText(frame, f'RECORDING {sample_idx+1}/{num_samples}: {frame_count}/60', (15, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if frame_count >= 60:
                    break
            else:
                cv2.putText(frame, f'Sample {sample_idx+1}/{num_samples} - Press "s" to start', (15, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Additional Training Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not recording:
                recording = True
                landmarks_sequence = []
                frame_count = 0
                print(f"🔴 Recording sample {sample_idx+1}...")
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return collected_sequences
        
        if len(landmarks_sequence) == 60:
            # Save sequence
            sequence = np.array(landmarks_sequence)
            filename = f"additional_sample_{sample_idx+1}.npy"
            filepath = os.path.join(phrase_dir, filename)
            np.save(filepath, sequence)
            collected_sequences.append(sequence)
            print(f"✅ Saved {filename}")
        else:
            print(f"❌ Only captured {len(landmarks_sequence)} frames, need 60")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return collected_sequences

def quick_retrain_with_additional_data():
    """Retrain the model with additional data for phrase 4"""
    print(f"\n🔄 Quick Retraining with Additional Data")
    print("="*50)
    
    # Load existing model and scaler
    model = keras.models.load_model('models/saved/lstm_model.keras')
    scaler = joblib.load('models/saved/sequence_scaler.joblib')
    
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    # Collect all training data
    all_sequences = []
    all_labels = []
    
    phrases = [
        "Hi my name is Reet",
        "How are you", 
        "I am from Delhi",
        "I like coffee",
        "What do you like"
    ]
    
    # Load existing current_env data
    for phrase_idx in range(5):
        phrase_dir = f"data/sequences_current_env/phrase_{phrase_idx}"
        if os.path.exists(phrase_dir):
            files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
            weight = 10 if phrase_idx == 4 else 3  # Extra weight for phrase 4
            
            for file in files:
                filepath = os.path.join(phrase_dir, file)
                sequence = np.load(filepath)
                if sequence.shape == (60, 1662):
                    # Add multiple copies for extra weight
                    for _ in range(weight):
                        all_sequences.append(sequence.flatten())
                        all_labels.append(phrase_idx)
                    print(f"✅ Loaded {file} with weight {weight}")
    
    if not all_sequences:
        print("❌ No training data found")
        return
    
    # Prepare training data
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    # Convert to categorical
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y, num_classes=5)
    
    print(f"📊 Training data: {X.shape[0]} samples")
    for i in range(5):
        count = np.sum(y == i)
        print(f"  Phrase {i} ({phrases[i]}): {count} samples")
    
    # Scale data
    X_scaled = scaler.transform(X)
    X_lstm = X_scaled.reshape(X_scaled.shape[0], 60, 1662)
    
    # Quick training (just a few epochs to adapt)
    print("\n🏋️ Fine-tuning model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_lstm, y_categorical,
        epochs=10,
        batch_size=16,
        verbose=1,
        validation_split=0.2
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
    print("🔧 Fixing 'What do you like' Recognition")
    print("="*50)
    
    # Collect additional data
    collected = collect_additional_samples()
    
    if len(collected) > 0:
        print(f"✅ Collected {len(collected)} additional samples")
        
        # Retrain with additional data
        quick_retrain_with_additional_data()
        
        print("\n🎉 Retraining complete!")
        print("Please test 'What do you like' again")
    else:
        print("❌ No additional samples collected")

if __name__ == "__main__":
    main()