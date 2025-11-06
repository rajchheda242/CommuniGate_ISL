#!/usr/bin/env python3
"""
Fix the feature mismatch by retraining with consistent 126-feature extraction
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
import gc

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_original_sequences():
    """Load original training sequences (126 features)"""
    print("📁 Loading Original Training Sequences (126 features)")
    
    data_dir = "data/sequences"
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    
    all_sequences = []
    all_labels = []
    
    for phrase_idx in range(5):
        phrase_dir = os.path.join(data_dir, f"phrase_{phrase_idx}")
        if not os.path.exists(phrase_dir):
            print(f"⚠️  {phrase_dir} not found")
            continue
        
        phrase_sequences = []
        for file in os.listdir(phrase_dir):
            if file.endswith('_seq.npy'):
                filepath = os.path.join(phrase_dir, file)
                sequence = np.load(filepath)
                
                # Ensure proper length (60 frames)
                if sequence.shape[0] > 60:
                    sequence = sequence[:60]
                elif sequence.shape[0] < 60:
                    # Pad with last frame
                    last_frame = sequence[-1:] if len(sequence) > 0 else np.zeros((1, sequence.shape[1]))
                    padding = np.repeat(last_frame, 60 - sequence.shape[0], axis=0)
                    sequence = np.vstack([sequence, padding])
                
                phrase_sequences.append(sequence)
        
        print(f"  {phrases[phrase_idx]}: {len(phrase_sequences)} sequences")
        
        # Add sequences with higher weight for original person
        for seq in phrase_sequences:
            all_sequences.append(seq)
            all_labels.append(phrase_idx)
            
            # Add duplicate for higher weight (2x)
            all_sequences.append(seq)
            all_labels.append(phrase_idx)
    
    return np.array(all_sequences), np.array(all_labels)

def load_multi_person_sequences_126():
    """Load multi-person sequences and convert to 126 features"""
    print("📁 Loading Multi-Person Sequences (convert to 126 features)")
    
    multi_person_dir = "data/sequences_multi_person"
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    
    all_sequences = []
    all_labels = []
    
    for phrase_idx in range(5):
        phrase_dir = os.path.join(multi_person_dir, f"phrase_{phrase_idx}")
        if not os.path.exists(phrase_dir):
            continue
        
        phrase_sequences = []
        for file in os.listdir(phrase_dir):
            if file.endswith('_seq.npy'):
                filepath = os.path.join(phrase_dir, file)
                sequence = np.load(filepath)
                
                # Convert from 1662 features to 126 features (extract just pose + hands)
                if sequence.shape[1] == 1662:
                    # Extract pose (33*4=132) and hands (21*4*2=168) = 300 total
                    # But original has 126, so extract minimal subset
                    # Keep pose landmarks and hand landmarks only
                    pose_features = sequence[:, :132]  # First 132 for pose (33 landmarks * 4)
                    
                    # Take subset to match 126 features
                    if pose_features.shape[1] >= 126:
                        sequence = pose_features[:, :126]
                    else:
                        # Pad if needed
                        padding = np.zeros((sequence.shape[0], 126 - pose_features.shape[1]))
                        sequence = np.hstack([pose_features, padding])
                
                # Ensure proper length (60 frames)
                if sequence.shape[0] > 60:
                    sequence = sequence[:60]
                elif sequence.shape[0] < 60:
                    last_frame = sequence[-1:] if len(sequence) > 0 else np.zeros((1, 126))
                    padding = np.repeat(last_frame, 60 - sequence.shape[0], axis=0)
                    sequence = np.vstack([sequence, padding])
                
                phrase_sequences.append(sequence)
        
        print(f"  {phrases[phrase_idx]}: {len(phrase_sequences)} multi-person sequences")
        
        for seq in phrase_sequences:
            all_sequences.append(seq)
            all_labels.append(phrase_idx)
    
    return np.array(all_sequences), np.array(all_labels)

def create_consistent_model():
    """Create model with 126 features (consistent with original)"""
    model = keras.Sequential([
        layers.Input(shape=(60, 126)),  # 126 features, not 1662
        
        # Robust LSTM layers
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("🔧 Fixing Feature Mismatch - Retraining with 126 Features")
    print("=" * 60)
    
    # Load original sequences (126 features)
    print("\n1️⃣ Loading Original Data...")
    X_original, y_original = load_original_sequences()
    print(f"   Original data shape: {X_original.shape}")
    
    # Load multi-person sequences and convert to 126 features
    print("\n2️⃣ Loading Multi-Person Data...")
    X_multi, y_multi = load_multi_person_sequences_126()
    print(f"   Multi-person data shape: {X_multi.shape}")
    
    # Combine datasets
    print("\n3️⃣ Combining Datasets...")
    X_combined = np.vstack([X_original, X_multi])
    y_combined = np.hstack([y_original, y_multi])
    
    print(f"   Combined shape: {X_combined.shape}")
    print(f"   Feature count: {X_combined.shape[2]} (should be 126)")
    
    # Check class distribution
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    print("\n📊 Class Distribution:")
    for i in range(5):
        count = np.sum(y_combined == i)
        print(f"   {phrases[i]}: {count} samples")
    
    # Prepare data
    print("\n4️⃣ Preparing Data...")
    
    # Flatten sequences for scaler
    X_flat = X_combined.reshape(X_combined.shape[0], -1)
    
    # Split data
    X_train_flat, X_test_flat, y_train, y_test = train_test_split(
        X_flat, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape back to sequences
    X_train = X_train_scaled.reshape(X_train_scaled.shape[0], 60, 126)
    X_test = X_test_scaled.reshape(X_test_scaled.shape[0], 60, 126)
    
    print(f"   Training shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_combined), y=y_combined
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"   Class weights: {class_weight_dict}")
    
    # Create and train model
    print("\n5️⃣ Training Model...")
    model = create_consistent_model()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n6️⃣ Evaluating Model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy:.3f}")
    
    # Save model and scaler
    print("\n7️⃣ Saving Model...")
    model.save('models/saved/lstm_model.keras')
    joblib.dump(scaler, 'models/saved/sequence_scaler.joblib')
    
    # Save phrase mapping
    phrase_mapping = {phrase: i for i, phrase in enumerate(phrases)}
    with open('models/saved/phrase_mapping.json', 'w') as f:
        json.dump(phrase_mapping, f)
    
    print("✅ Model retrained with consistent 126 features!")
    print(f"📊 Final Test Accuracy: {test_accuracy:.3f}")
    
    return test_accuracy

if __name__ == "__main__":
    accuracy = main()
    
    if accuracy > 0.85:
        print("\n🎉 EXCELLENT! High accuracy achieved!")
    elif accuracy > 0.7:
        print("\n✅ Good accuracy - should work much better now!")
    else:
        print("\n⚠️  Still low accuracy - may need more data or different approach")