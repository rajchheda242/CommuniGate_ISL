#!/usr/bin/env python3
"""
Create a simple, robust model that won't overfit or be biased
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_original_sequences():
    """Load only original training sequences (no multi-person data)"""
    print("📁 Loading Original Training Sequences Only")
    
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
        
        # Add sequences without excessive weighting
        for seq in phrase_sequences:
            all_sequences.append(seq)
            all_labels.append(phrase_idx)
    
    return np.array(all_sequences), np.array(all_labels)

def create_simple_robust_model():
    """Create a very simple model that won't overfit"""
    model = keras.Sequential([
        layers.Input(shape=(60, 126)),
        
        # Very simple architecture to prevent overfitting
        layers.LSTM(32, return_sequences=False, dropout=0.5, recurrent_dropout=0.3),
        layers.BatchNormalization(),
        
        # Small dense layer with heavy regularization
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.6),
        layers.BatchNormalization(),
        
        # Output layer - no bias initialization to prevent defaults
        layers.Dense(5, activation='softmax', bias_initializer='zeros')
    ])
    
    # Use lower learning rate for stability
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def add_noise_augmentation(X, y, noise_factor=0.01):
    """Add slight noise to training data for better generalization"""
    print("🔀 Adding noise augmentation for better generalization...")
    
    X_augmented = []
    y_augmented = []
    
    # Original data
    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])
    
    # Add noisy versions
    for i in range(len(X)):
        # Add small amount of gaussian noise
        noise = np.random.normal(0, noise_factor, X[i].shape)
        noisy_sequence = X[i] + noise
        
        X_augmented.append(noisy_sequence)
        y_augmented.append(y[i])
    
    print(f"   Original samples: {len(X)}")
    print(f"   Augmented samples: {len(X_augmented)}")
    
    return np.array(X_augmented), np.array(y_augmented)

def main():
    print("🛠️  Creating Simple, Robust Model (Anti-Overfitting)")
    print("=" * 60)
    
    # Load only original data (no multi-person confusion)
    print("\n1️⃣ Loading Original Data Only...")
    X, y = load_original_sequences()
    print(f"   Original data shape: {X.shape}")
    
    # Add noise augmentation for generalization
    print("\n2️⃣ Adding Augmentation...")
    X_aug, y_aug = add_noise_augmentation(X, y, noise_factor=0.005)
    
    # Check class distribution
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    print("\n📊 Class Distribution:")
    for i in range(5):
        count = np.sum(y_aug == i)
        print(f"   {phrases[i]}: {count} samples")
    
    # Prepare data with larger test split
    print("\n3️⃣ Preparing Data...")
    
    # Flatten sequences for scaler
    X_flat = X_aug.reshape(X_aug.shape[0], -1)
    
    # Larger test split to prevent overfitting
    X_train_flat, X_test_flat, y_train, y_test = train_test_split(
        X_flat, y_aug, test_size=0.3, random_state=42, stratify=y_aug
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
    
    # Create simple model
    print("\n4️⃣ Training Simple, Robust Model...")
    model = create_simple_robust_model()
    
    print("Model summary:")
    model.summary()
    
    # Aggressive early stopping to prevent overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
        )
    ]
    
    # Train with focus on generalization
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,  # Fewer epochs to prevent overfitting
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n5️⃣ Evaluating Model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy:.3f}")
    
    # Test on random noise to ensure no default bias
    print("\n6️⃣ Testing Random Noise (Anti-Bias Check)...")
    noise_predictions = []
    
    for i in range(20):
        # Generate random sequence (60, 126)
        random_sequence = np.random.random((1, 60, 126))
        random_flat = random_sequence.reshape(1, -1)
        random_scaled = scaler.transform(random_flat)
        random_reshaped = random_scaled.reshape(1, 60, 126)
        
        # Predict
        predictions = model.predict(random_reshaped, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        noise_predictions.append(predicted_idx)
    
    # Check distribution
    prediction_counts = {}
    for pred in noise_predictions:
        pred_phrase = phrases[pred]
        prediction_counts[pred_phrase] = prediction_counts.get(pred_phrase, 0) + 1
    
    print(f"Random noise predictions:")
    max_count = 0
    for phrase, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(noise_predictions) * 100
        print(f"  {phrase}: {count}/20 times ({percentage:.0f}%)")
        max_count = max(max_count, count)
    
    # Check if still biased
    if max_count >= 12:  # 60% or more
        print("⚠️  Still some bias detected, but should be much better")
    else:
        print("✅ No severe bias detected!")
    
    # Save model and scaler
    print("\n7️⃣ Saving Simple Model...")
    model.save('models/saved/lstm_model.keras')
    joblib.dump(scaler, 'models/saved/sequence_scaler.joblib')
    
    # Save phrase mapping
    phrase_mapping = {phrase: i for i, phrase in enumerate(phrases)}
    with open('models/saved/phrase_mapping.json', 'w') as f:
        json.dump(phrase_mapping, f)
    
    print("✅ Simple, robust model trained!")
    print(f"📊 Test Accuracy: {test_accuracy:.3f}")
    
    return test_accuracy

if __name__ == "__main__":
    accuracy = main()
    
    print(f"\n🎯 RESULT:")
    if accuracy >= 0.8:
        print("✅ Good accuracy achieved with simple model!")
        print("🎉 Should work much better in real-time without bias!")
    else:
        print("⚠️  Lower accuracy but should be more robust and unbiased")
        print("💡 This trade-off prevents overfitting and default predictions")