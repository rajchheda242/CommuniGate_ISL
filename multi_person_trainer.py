#!/usr/bin/env python3
"""
Optimized Multi-Person Model Retraining
Uses efficiently processed multi-person data for better generalization
"""

import numpy as np
import json
import os
import sys
import joblib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

def load_multi_person_data():
    """Load the efficiently processed multi-person data."""
    print("📂 Loading multi-person training data...")
    
    data_dir = "data/processed_multi_person"
    if not os.path.exists(data_dir):
        print(f"❌ Multi-person data not found in {data_dir}")
        print("Please run optimized_video_processor.py first")
        return None, None, None
    
    # Load data
    X = np.load(f"{data_dir}/X_multi_person.npy")
    y = np.load(f"{data_dir}/y_multi_person.npy")
    
    with open(f"{data_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded {X.shape[0]} sequences from {metadata['total_videos_processed']} videos")
    print(f"📊 Dataset distribution:")
    
    phrase_names = [
        "Hi my name is Reet",
        "How are you", 
        "I am from Delhi",
        "I like coffee",
        "What do you like"
    ]
    
    for phrase_idx in range(5):
        count = metadata['phrase_distribution'][str(phrase_idx)]
        print(f"  Phrase {phrase_idx} ({phrase_names[phrase_idx]}): {count} sequences")
    
    return X, y, metadata

def combine_with_existing_data(X_multi, y_multi):
    """Combine multi-person data with existing current environment data."""
    print("\n🔄 Combining with existing current environment data...")
    
    # Load existing current environment data
    existing_sequences = []
    existing_labels = []
    
    for phrase_idx in range(5):
        phrase_dir = f"data/sequences_current_env/phrase_{phrase_idx}"
        if os.path.exists(phrase_dir):
            files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
            for file in files:
                filepath = os.path.join(phrase_dir, file)
                sequence = np.load(filepath)
                if sequence.shape == (60, 1662):
                    # Add multiple copies for balance (but less than before)
                    for _ in range(2):  # Reduced weight
                        existing_sequences.append(sequence.flatten())
                        existing_labels.append(phrase_idx)
    
    if existing_sequences:
        X_existing = np.array(existing_sequences)
        y_existing = np.array(existing_labels)
        
        # Combine datasets
        X_combined = np.vstack([X_multi, X_existing])
        y_combined = np.hstack([y_multi, y_existing])
        
        print(f"✅ Combined {X_multi.shape[0]} multi-person + {X_existing.shape[0]} current env = {X_combined.shape[0]} total")
        return X_combined, y_combined
    else:
        print("⚠️ No existing current environment data found, using only multi-person data")
        return X_multi, y_multi

def create_improved_model(input_shape, num_classes=5):
    """Create an improved LSTM model for better generalization."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # More robust LSTM layers with dropout for generalization
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        # Dense layers with dropout
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_multi_person_model():
    """Train model on multi-person data for better generalization."""
    print("🏋️ Multi-Person Model Training")
    print("="*50)
    
    # Load data
    X_multi, y_multi, metadata = load_multi_person_data()
    if X_multi is None:
        return
    
    # Combine with existing data
    X, y = combine_with_existing_data(X_multi, y_multi)
    
    # Convert to categorical
    y_categorical = to_categorical(y, num_classes=5)
    
    print(f"\n📊 Final training data: {X.shape[0]} sequences")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train: {X_train.shape[0]} sequences, Test: {X_test.shape[0]} sequences")
    
    # Scale data
    print("\n⚖️ Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 60, 1662)
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 60, 1662)
    
    # Create model
    print("\n🧠 Creating improved model...")
    model = create_improved_model((60, 1662))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📋 Model summary:")
    model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train model
    print("\n🚀 Starting training...")
    start_time = time.time()
    
    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_test_lstm, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n⏱️ Training completed in {training_time:.1f} seconds")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    train_loss, train_acc = model.evaluate(X_train_lstm, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_lstm, y_test, verbose=0)
    
    print(f"📊 Final Results:")
    print(f"  Training accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"  Training loss: {train_loss:.3f}")
    print(f"  Test loss: {test_loss:.3f}")
    
    # Save model and components
    print("\n💾 Saving model...")
    model.save('models/saved/lstm_model.keras')
    joblib.dump(scaler, 'models/saved/sequence_scaler.joblib')
    
    # Save phrase mapping
    phrase_mapping = {
        str(i): phrase for i, phrase in enumerate([
            "Hi my name is Reet",
            "How are you", 
            "I am from Delhi",
            "I like coffee",
            "What do you like"
        ])
    }
    
    with open('models/saved/phrase_mapping.json', 'w') as f:
        json.dump(phrase_mapping, f, indent=2)
    
    # Save training metadata
    training_metadata = {
        "model_type": "LSTM_Multi_Person",
        "training_sequences": X.shape[0],
        "multi_person_sequences": X_multi.shape[0],
        "training_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "training_time_seconds": training_time,
        "epochs_trained": len(history.history['loss']),
        "features_per_frame": 1662,
        "sequence_length": 60
    }
    
    with open('models/saved/training_metadata.json', 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    print("✅ Model saved successfully!")
    print("\n🎉 Multi-person training complete!")
    print(f"The model should now work better for different people")
    
    if test_acc > 0.85:
        print("🎯 Excellent accuracy! Model should generalize well")
    elif test_acc > 0.75:
        print("👍 Good accuracy! Some improvement expected")
    else:
        print("⚠️ Model accuracy could be better. Consider collecting more diverse data")

if __name__ == "__main__":
    train_multi_person_model()

def collect_multi_person_data():
    """Collect training data from multiple people"""
    print("🌍 Multi-Person ISL Training Data Collection")
    print("="*60)
    print("This will help the model work for different people!")
    print()
    
    # Get person info
    person_name = input("Enter person's name (e.g., John, Sarah, etc.): ").strip()
    if not person_name:
        person_name = "person"
    
    person_name = person_name.replace(" ", "_").lower()
    
    print(f"👤 Collecting data for: {person_name}")
    print()
    
    phrases = [
        "Hi my name is Reet",
        "How are you", 
        "I am from Delhi",
        "I like coffee",
        "What do you like"
    ]
    
    # Create directory for this person
    person_dir = f"data/multi_person_sequences/{person_name}"
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize inference
    inference = HolisticInference()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return []
    
    all_collected_data = []
    samples_per_phrase = 8  # Collect more samples for better diversity
    
    for phrase_idx, phrase in enumerate(phrases):
        print(f"\n📝 Phrase {phrase_idx + 1}/5: {phrase}")
        print(f"We'll collect {samples_per_phrase} samples of this phrase")
        print("-" * 50)
        
        phrase_dir = os.path.join(person_dir, f"phrase_{phrase_idx}")
        os.makedirs(phrase_dir, exist_ok=True)
        
        for sample_idx in range(samples_per_phrase):
            print(f"\n📹 Sample {sample_idx + 1}/{samples_per_phrase}")
            print(f"Perform: '{phrase}'")
            print("Press 's' to start recording, 'q' to quit, 'n' to skip this sample")
            
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
                    
                    cv2.putText(frame, f'RECORDING {phrase_idx+1}/5 - {sample_idx+1}/{samples_per_phrase}: {frame_count}/60', (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'{phrase}', (15, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if frame_count >= 60:
                        break
                else:
                    cv2.putText(frame, f'Person: {person_name} | Phrase {phrase_idx+1}/5 | Sample {sample_idx+1}/{samples_per_phrase}', (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Press "s" to start: {phrase}', (15, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'Press "n" to skip, "q" to quit', (15, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)
                
                cv2.imshow('Multi-Person Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and not recording:
                    recording = True
                    landmarks_sequence = []
                    frame_count = 0
                    print(f"🔴 Recording sample {sample_idx+1} for '{phrase}'...")
                elif key == ord('n') and not recording:
                    print(f"⏭️ Skipping sample {sample_idx+1}")
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return all_collected_data
            
            if len(landmarks_sequence) == 60:
                # Save sequence
                sequence = np.array(landmarks_sequence)
                filename = f"{person_name}_sample_{sample_idx+1}.npy"
                filepath = os.path.join(phrase_dir, filename)
                np.save(filepath, sequence)
                
                # Add to collected data
                all_collected_data.append({
                    'person': person_name,
                    'phrase_idx': phrase_idx,
                    'phrase': phrase,
                    'sequence': sequence,
                    'file': filepath
                })
                
                print(f"✅ Saved {filename}")
            else:
                print(f"❌ Only captured {len(landmarks_sequence)} frames, need 60")
                sample_idx -= 1  # Retry this sample
    
    cap.release()
    cv2.destroyAllWindows()
    
    return all_collected_data

def retrain_with_multi_person_data():
    """Retrain model with data from multiple people"""
    print(f"\n🔄 Retraining with Multi-Person Data")
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
    
    # Load original current_env data (your data)
    print("Loading original training data...")
    for phrase_idx in range(5):
        phrase_dir = f"data/sequences_current_env/phrase_{phrase_idx}"
        if os.path.exists(phrase_dir):
            files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
            for file in files:
                filepath = os.path.join(phrase_dir, file)
                sequence = np.load(filepath)
                if sequence.shape == (60, 1662):
                    # Add original data with weight 1
                    all_sequences.append(sequence.flatten())
                    all_labels.append(phrase_idx)
                    print(f"✅ Loaded original: {file}")
    
    # Load multi-person data
    print("\nLoading multi-person data...")
    multi_person_dir = "data/multi_person_sequences"
    if os.path.exists(multi_person_dir):
        person_count = 0
        for person_name in os.listdir(multi_person_dir):
            person_path = os.path.join(multi_person_dir, person_name)
            if os.path.isdir(person_path):
                person_count += 1
                print(f"Loading data for person: {person_name}")
                
                for phrase_idx in range(5):
                    phrase_dir = os.path.join(person_path, f"phrase_{phrase_idx}")
                    if os.path.exists(phrase_dir):
                        files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
                        for file in files:
                            filepath = os.path.join(phrase_dir, file)
                            sequence = np.load(filepath)
                            if sequence.shape == (60, 1662):
                                # Add multi-person data with weight 2 (higher importance)
                                for _ in range(2):
                                    all_sequences.append(sequence.flatten())
                                    all_labels.append(phrase_idx)
                                print(f"✅ Loaded multi-person: {file}")
        
        print(f"📊 Loaded data from {person_count} additional people")
    
    if not all_sequences:
        print("❌ No training data found")
        return
    
    # Prepare training data
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    # Convert to categorical
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y, num_classes=5)
    
    print(f"\n📊 Total training data: {X.shape[0]} samples")
    for i in range(5):
        count = np.sum(y == i)
        print(f"  Phrase {i} ({phrases[i]}): {count} samples")
    
    # Scale data
    X_scaled = scaler.transform(X)
    X_lstm = X_scaled.reshape(X_scaled.shape[0], 60, 1662)
    
    # Enhanced training for better generalization
    print("\n🏋️ Training model for multi-person generalization...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add data augmentation and regularization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5)
    ]
    
    history = model.fit(
        X_lstm, y_categorical,
        epochs=30,
        batch_size=16,
        verbose=1,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    # Save updated model
    model.save('models/saved/lstm_model.keras')
    print("✅ Multi-person model saved")
    
    # Test accuracy
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"📈 Final accuracy: {final_accuracy:.3f}")
    print(f"📈 Final validation accuracy: {final_val_accuracy:.3f}")

def main():
    """Main function"""
    print("🌍 Multi-Person ISL Model Training")
    print("="*60)
    print("This tool helps collect data from multiple people to improve")
    print("model generalization so it works for anyone, not just one person!")
    print()
    
    choice = input("Choose option:\n1. Collect data from new person\n2. Retrain with all collected data\n3. Both\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n🎥 Starting data collection...")
        collected = collect_multi_person_data()
        print(f"\n✅ Collected {len(collected)} samples")
    
    if choice in ['2', '3']:
        print("\n🏋️ Starting multi-person retraining...")
        retrain_with_multi_person_data()
        print("\n🎉 Multi-person training complete!")
        print("The model should now work better for different people!")

if __name__ == "__main__":
    main()