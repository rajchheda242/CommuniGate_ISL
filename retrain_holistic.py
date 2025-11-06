#!/usr/bin/env python3
"""
Quick fix: Retrain the model using holistic data to match what inference.py captures
"""

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras import callbacks
import joblib
import json

# Use holistic data to match inference.py
DATA_DIR = "data/sequences_holistic" 
MODEL_DIR = "models/saved"
PHRASES = [
    "Hi my name is Reet",
    "How are you",
    "I am from Delhi",
    "I like coffee",
    "What do you like"
]

def retrain_holistic_model():
    """Retrain model using holistic data to match inference system."""
    
    print("🔄 Retraining model with holistic data...")
    print(f"Using data from: {DATA_DIR}")
    
    # Load holistic sequence data
    X = []
    y = []
    
    for phrase_idx in range(len(PHRASES)):
        phrase_dir = os.path.join(DATA_DIR, f"phrase_{phrase_idx}")
        
        if not os.path.exists(phrase_dir):
            print(f"❌ No holistic data found for phrase {phrase_idx}: {PHRASES[phrase_idx]}")
            continue
            
        sequence_files = glob.glob(os.path.join(phrase_dir, "*.npy"))
        print(f"📂 Phrase {phrase_idx} ({PHRASES[phrase_idx]}): {len(sequence_files)} sequences")
        
        for seq_file in sequence_files:
            try:
                sequence = np.load(seq_file)
                
                # Reshape to match what inference.py expects
                if sequence.shape[0] == 90:  # 90 frames
                    # Take middle 60 frames to match inference
                    start_frame = 15
                    end_frame = 75
                    sequence_60 = sequence[start_frame:end_frame]
                elif sequence.shape[0] == 60:
                    sequence_60 = sequence
                else:
                    print(f"⚠️  Unexpected sequence length: {sequence.shape[0]} in {seq_file}")
                    continue
                
                # Flatten for scaling (60 frames * 1662 features = 99720)
                sequence_flat = sequence_60.flatten()
                
                X.append(sequence_flat)
                y.append(phrase_idx)
                
            except Exception as e:
                print(f"❌ Error loading {seq_file}: {e}")
    
    if len(X) == 0:
        print("❌ No training data loaded!")
        return
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} sequences")
    print(f"✅ Feature shape: {X.shape[1]} (should be 60 * 1662 = 99720)")
    print(f"✅ Classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM (samples, timesteps, features)
    # We have 60 timesteps and 1662 features per timestep
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 60, 1662)
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 60, 1662)
    
    print(f"✅ Training shape: {X_train_lstm.shape}")
    print(f"✅ Test shape: {X_test_lstm.shape}")
    
    # Build model
    model = Sequential([
        Input(shape=(60, 1662)),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
        Bidirectional(LSTM(64, dropout=0.3)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(PHRASES), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model architecture:")
    model.summary()
    
    # Training callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Train model
    print("🚀 Starting training...")
    
    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_test_lstm, y_test),
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test, verbose=0)
    print(f"\n🎯 Test Accuracy: {test_accuracy:.3f}")
    
    # Save everything
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
    print(f"✅ Model saved to {MODEL_DIR}/lstm_model.keras")
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "sequence_scaler.joblib"))
    print(f"✅ Scaler saved to {MODEL_DIR}/sequence_scaler.joblib")
    
    # Save phrase mapping
    phrase_mapping = {str(i): phrase for i, phrase in enumerate(PHRASES)}
    with open(os.path.join(MODEL_DIR, "phrase_mapping.json"), 'w') as f:
        json.dump(phrase_mapping, f, indent=2)
    print(f"✅ Phrase mapping saved to {MODEL_DIR}/phrase_mapping.json")
    
    print("\n🎉 Model retrained successfully!")
    print("Now the model should work with your inference system.")
    
    return model, scaler, test_accuracy

if __name__ == "__main__":
    retrain_holistic_model()