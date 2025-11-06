#!/usr/bin/env python3
"""
Create a hybrid model that prioritizes your data but includes some multi-person data
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

def load_original_sequences_with_high_weight():
    """Load original training sequences with very high weighting"""
    print("📁 Loading Original Training Sequences (your data - high priority)")
    
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
        
        # Add sequences with VERY high weight (5x)
        for seq in phrase_sequences:
            for _ in range(5):  # 5x weight for your data
                all_sequences.append(seq)
                all_labels.append(phrase_idx)
    
    return np.array(all_sequences), np.array(all_labels)

def load_minimal_multi_person_sequences():
    """Load a minimal amount of multi-person data for generalization"""
    print("📁 Loading Minimal Multi-Person Data (for generalization only)")
    
    multi_person_dir = "data/sequences_multi_person"
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    
    all_sequences = []
    all_labels = []
    
    for phrase_idx in range(5):
        phrase_dir = os.path.join(multi_person_dir, f"phrase_{phrase_idx}")
        if not os.path.exists(phrase_dir):
            continue
        
        phrase_sequences = []
        files = [f for f in os.listdir(phrase_dir) if f.endswith('_seq.npy')]
        
        # Take only 2-3 samples per phrase to avoid diluting your data
        max_samples = min(3, len(files))
        
        for file in files[:max_samples]:
            filepath = os.path.join(phrase_dir, file)
            sequence = np.load(filepath)
            
            # Convert from 1662 features to 126 features
            if sequence.shape[1] == 1662:
                pose_features = sequence[:, :132]  # First 132 for pose
                if pose_features.shape[1] >= 126:
                    sequence = pose_features[:, :126]
                else:
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
        
        print(f"  {phrases[phrase_idx]}: {len(phrase_sequences)} minimal multi-person samples")
        
        # Add with minimal weight (1x only)
        for seq in phrase_sequences:
            all_sequences.append(seq)
            all_labels.append(phrase_idx)
    
    return np.array(all_sequences), np.array(all_labels)

def create_your_optimized_model():
    """Create model optimized for your gestures"""
    model = keras.Sequential([
        layers.Input(shape=(60, 126)),
        
        # Optimized for your specific patterns
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.BatchNormalization(),
        
        layers.LSTM(32, return_sequences=True, dropout=0.2),
        layers.BatchNormalization(),
        
        layers.LSTM(16, dropout=0.2),
        layers.BatchNormalization(),
        
        # Dense layers optimized for your data
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        # Output
        layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0005),  # Lower learning rate for stability
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("🎯 Creating Hybrid Model - Prioritizing Your Data")
    print("=" * 60)
    
    # Load your data with high weight
    print("\n1️⃣ Loading Your Data (high priority)...")
    X_yours, y_yours = load_original_sequences_with_high_weight()
    print(f"   Your data shape: {X_yours.shape}")
    
    # Load minimal multi-person data
    print("\n2️⃣ Loading Minimal Multi-Person Data...")
    X_multi, y_multi = load_minimal_multi_person_sequences()
    print(f"   Multi-person data shape: {X_multi.shape}")
    
    # Combine with your data heavily weighted
    print("\n3️⃣ Combining Datasets (your data prioritized)...")
    if X_multi.shape[0] > 0:
        X_combined = np.vstack([X_yours, X_multi])
        y_combined = np.hstack([y_yours, y_multi])
    else:
        X_combined = X_yours
        y_combined = y_yours
    
    print(f"   Combined shape: {X_combined.shape}")
    print(f"   Your data samples: {X_yours.shape[0]}")
    print(f"   Multi-person samples: {X_multi.shape[0] if X_multi.shape[0] > 0 else 0}")
    
    # Check class distribution
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    print("\n📊 Class Distribution (heavily weighted for your data):")
    for i in range(5):
        count = np.sum(y_combined == i)
        your_count = np.sum(y_yours == i)
        multi_count = np.sum(y_multi == i) if len(y_multi) > 0 else 0
        print(f"   {phrases[i]}: {count} total (yours: {your_count}, multi: {multi_count})")
    
    # Prepare data
    print("\n4️⃣ Preparing Data...")
    
    # Flatten sequences for scaler
    X_flat = X_combined.reshape(X_combined.shape[0], -1)
    
    # Split data ensuring your data is in training set
    X_train_flat, X_test_flat, y_train, y_test = train_test_split(
        X_flat, y_combined, test_size=0.15, random_state=42, stratify=y_combined
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
    
    # Create and train model
    print("\n5️⃣ Training Your-Optimized Model...")
    model = create_your_optimized_model()
    
    # Callbacks for your data optimization
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.7, patience=7, min_lr=1e-6
        )
    ]
    
    # Train with focus on your patterns
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=16,  # Smaller batch size for better learning
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n6️⃣ Evaluating Model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy:.3f}")
    
    # Test on your original data specifically
    print("\n7️⃣ Testing on Your Original Data...")
    X_yours_flat = X_yours.reshape(X_yours.shape[0], -1)
    X_yours_scaled = scaler.transform(X_yours_flat)
    X_yours_reshaped = X_yours_scaled.reshape(X_yours.shape[0], 60, 126)
    
    yours_predictions = model.predict(X_yours_reshaped, verbose=0)
    yours_predicted_classes = np.argmax(yours_predictions, axis=1)
    yours_accuracy = np.mean(yours_predicted_classes == y_yours)
    
    print(f"   Accuracy on YOUR data: {yours_accuracy:.3f}")
    
    # Save model and scaler
    print("\n8️⃣ Saving Your-Optimized Model...")
    model.save('models/saved/lstm_model.keras')
    joblib.dump(scaler, 'models/saved/sequence_scaler.joblib')
    
    # Save phrase mapping
    phrase_mapping = {phrase: i for i, phrase in enumerate(phrases)}
    with open('models/saved/phrase_mapping.json', 'w') as f:
        json.dump(phrase_mapping, f)
    
    print("✅ Your-optimized model trained!")
    print(f"📊 Test Accuracy: {test_accuracy:.3f}")
    print(f"🎯 Your Data Accuracy: {yours_accuracy:.3f}")
    
    return test_accuracy, yours_accuracy

if __name__ == "__main__":
    test_acc, yours_acc = main()
    
    if yours_acc >= 0.95:
        print("\n🎉 EXCELLENT! Your gestures work perfectly!")
    elif yours_acc >= 0.85:
        print("\n✅ Very good performance on your gestures!")
    else:
        print("\n⚠️  Still need improvement for your specific gestures")
    
    if test_acc >= 0.85:
        print("🌟 Good generalization to other people too!")
    else:
        print("📝 Focused on your gestures - may need more diverse data for others")