#!/usr/bin/env python3
"""
Quick data collection and retraining in your current environment
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def collect_new_training_data():
    """Collect a small amount of training data in your current environment."""
    
    print("🎬 Quick Data Collection for Environment Matching")
    print("="*60)
    print("We'll collect just 5 samples per phrase in your current setup.")
    print("This should be enough to adapt the model to your environment.")
    
    from inference import HolisticInference
    
    # Initialize inference for landmark extraction
    inference = HolisticInference()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    phrases = [
        "Hi my name is Reet",
        "How are you", 
        "I am from Delhi",
        "I like coffee",
        "What do you like"
    ]
    
    # Create directory for new data
    new_data_dir = "data/sequences_current_env"
    os.makedirs(new_data_dir, exist_ok=True)
    
    all_sequences = []
    all_labels = []
    
    for phrase_idx, phrase in enumerate(phrases):
        phrase_dir = os.path.join(new_data_dir, f"phrase_{phrase_idx}")
        os.makedirs(phrase_dir, exist_ok=True)
        
        print(f"\n📂 Collecting data for phrase {phrase_idx}: {phrase}")
        print("We'll collect 5 samples. Each sample is 60 frames.")
        
        for sample_idx in range(5):
            print(f"\n🎬 Sample {sample_idx + 1}/5")
            print(f"Perform: {phrase}")
            input("Press Enter when ready...")
            
            # Collect 60 frames
            landmarks_sequence = []
            frame_count = 0
            
            print("Recording... perform your gesture now!")
            start_time = time.time()
            
            while frame_count < 60:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame
                landmarks, results = inference.process_frame(frame)
                
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        print(f"  Captured {frame_count}/60 frames...")
                
                # Show frame
                cv2.imshow('Collecting Training Data', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if len(landmarks_sequence) == 60:
                # Save sequence
                sequence = np.array(landmarks_sequence)
                filename = f"current_env_sample_{sample_idx+1}.npy"
                filepath = os.path.join(phrase_dir, filename)
                np.save(filepath, sequence)
                
                # Add to training data
                all_sequences.append(sequence.flatten())  # Flatten for model
                all_labels.append(phrase_idx)
                
                print(f"✅ Saved {filename}")
            else:
                print(f"❌ Only captured {len(landmarks_sequence)} frames, need 60")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Data collection complete!")
    print(f"Collected {len(all_sequences)} sequences")
    
    return np.array(all_sequences), np.array(all_labels)

def quick_retrain(X_new, y_new):
    """Quickly retrain the model with new data."""
    
    print("\n🚀 Quick Retraining with Current Environment Data")
    print("="*60)
    
    # Load existing training data
    print("Loading existing training data...")
    
    X_existing = []
    y_existing = []
    
    for phrase_idx in range(5):
        data_dir = f"data/sequences_holistic/phrase_{phrase_idx}"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
            
            for filename in files:
                filepath = os.path.join(data_dir, filename)
                sequence = np.load(filepath)
                
                # Take middle 60 frames
                if sequence.shape[0] == 90:
                    sequence = sequence[15:75]
                
                X_existing.append(sequence.flatten())
                y_existing.append(phrase_idx)
    
    X_existing = np.array(X_existing)
    y_existing = np.array(y_existing)
    
    print(f"✅ Loaded {len(X_existing)} existing sequences")
    print(f"✅ Collected {len(X_new)} new sequences")
    
    # Combine datasets (weight new data more heavily)
    X_combined = np.vstack([X_existing, X_new, X_new, X_new])  # 3x weight for new data
    y_combined = np.hstack([y_existing, y_new, y_new, y_new])
    
    print(f"✅ Combined dataset: {len(X_combined)} sequences")
    
    # Train using the LSTM approach (simpler and faster)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
    import joblib
    import json
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 60, 1662)
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 60, 1662)
    
    print(f"✅ Training shape: {X_train_lstm.shape}")
    
    # Build model
    model = Sequential([
        Input(shape=(60, 1662)),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
        Bidirectional(LSTM(32, dropout=0.3)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("🚀 Training model...")
    
    # Train (fewer epochs since we're fine-tuning)
    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_test_lstm, y_test),
        epochs=20,
        batch_size=8,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test, verbose=0)
    print(f"\n🎯 Test Accuracy: {test_accuracy:.3f}")
    
    # Save model (overwrite the old LSTM model)
    model_dir = "models/saved"
    model.save(os.path.join(model_dir, "lstm_model.keras"))
    joblib.dump(scaler, os.path.join(model_dir, "sequence_scaler.joblib"))
    
    phrases = [
        "Hi my name is Reet",
        "How are you",
        "I am from Delhi", 
        "I like coffee",
        "What do you like"
    ]
    
    phrase_mapping = {str(i): phrase for i, phrase in enumerate(phrases)}
    with open(os.path.join(model_dir, "phrase_mapping.json"), 'w') as f:
        json.dump(phrase_mapping, f, indent=2)
    
    print("\n🎉 Retraining complete!")
    print("The model should now work much better in your current environment.")
    
    return model, scaler

if __name__ == "__main__":
    print("This will collect new training data in your current environment")
    print("and retrain the model to work better with your setup.")
    
    choice = input("Continue? (y/n): ")
    if choice.lower() == 'y':
        X_new, y_new = collect_new_training_data()
        if len(X_new) > 0:
            quick_retrain(X_new, y_new)
        else:
            print("No data collected.")
    else:
        print("Operation cancelled.")