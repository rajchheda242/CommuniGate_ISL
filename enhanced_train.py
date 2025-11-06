#!/usr/bin/env python3
"""
Enhanced training script with better model architecture and data augmentation.
Use this for training when you have 100+ sequences per phrase.
"""

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input,
    BatchNormalization
)
from tensorflow.keras import callbacks
import joblib
import json

DATA_DIR = "data/sequences"
MODEL_DIR = "models/saved"
PHRASES = [
    "Hi my name is Reet",
    "How are you",
    "I am from Delhi",
    "I like coffee",
    "What do you like"
]

class EnhancedTrainer:
    """Enhanced trainer with better architecture and augmentation."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def augment_sequence(self, sequence):
        """Apply data augmentation to a sequence."""
        aug_seq = sequence.copy()
        
        # Random noise injection (camera jitter)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.005, sequence.shape)
            aug_seq += noise
        
        # Random temporal scaling (faster/slower signing)
        if np.random.rand() > 0.5:
            speed_factor = np.random.uniform(0.9, 1.1)
            original_len = len(sequence)
            new_len = int(original_len * speed_factor)
            # Resample
            indices = np.linspace(0, original_len - 1, new_len)
            resampled = np.zeros((new_len, sequence.shape[1]))
            for i in range(sequence.shape[1]):
                resampled[:, i] = np.interp(indices, np.arange(original_len), sequence[:, i])
            # Resize back to original length
            indices = np.linspace(0, new_len - 1, original_len)
            for i in range(sequence.shape[1]):
                aug_seq[:, i] = np.interp(indices, np.arange(new_len), resampled[:, i])
        
        return aug_seq
    
    def load_sequences(self, augment=True):
        """Load sequences with optional augmentation."""
        X = []
        y = []
        
        print("Loading sequence data...")
        
        for phrase_idx in range(len(PHRASES)):
            phrase_dir = os.path.join(DATA_DIR, f"phrase_{phrase_idx}")
            
            if not os.path.exists(phrase_dir):
                print(f"Warning: No data found for phrase {phrase_idx}")
                continue
            
            sequence_files = glob.glob(os.path.join(phrase_dir, "*_seq.npy"))
            
            original_count = 0
            augmented_count = 0
            
            for seq_file in sequence_files:
                sequence = np.load(seq_file)
                
                # Add original
                X.append(sequence)
                y.append(phrase_idx)
                original_count += 1
                
                # Add augmented versions
                if augment:
                    for _ in range(2):  # 2x augmentation
                        aug_seq = self.augment_sequence(sequence)
                        X.append(aug_seq)
                        y.append(phrase_idx)
                        augmented_count += 1
            
            print(f"Phrase {phrase_idx}: {original_count} original + {augmented_count} augmented = {original_count + augmented_count} total - '{PHRASES[phrase_idx]}'")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nTotal sequences: {len(X)}")
        print(f"Sequence shape: {X.shape}")
        
        return X, y
    
    def build_enhanced_model(self, sequence_length, n_features, n_classes):
        """Build enhanced LSTM model."""
        model = Sequential([
            Input(shape=(sequence_length, n_features)),
            
            # Deeper bidirectional LSTM layers
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.4),
            
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.4),
            
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            
            # Stronger dense layers with batch normalization
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(64, activation='relu'),
            Dropout(0.4),
            
            # Output layer
            Dense(n_classes, activation='softmax')
        ])
        
        # Use AdamW optimizer for better generalization
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def normalize_sequences(self, X_train, X_val, X_test):
        """Normalize sequences."""
        n_train, n_frames, n_features = X_train.shape
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]
        
        X_train_flat = X_train.reshape(-1, n_features)
        X_val_flat = X_val.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)
        
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_val_scaled = self.scaler.transform(X_val_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        X_train_scaled = X_train_scaled.reshape(n_train, n_frames, n_features)
        X_val_scaled = X_val_scaled.reshape(n_val, n_frames, n_features)
        X_test_scaled = X_test_scaled.reshape(n_test, n_frames, n_features)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train(self):
        """Main training pipeline."""
        print("="*70)
        print("ENHANCED LSTM TRAINING")
        print("="*70)
        
        # Load data with augmentation
        X, y = self.load_sequences(augment=True)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} sequences")
        print(f"  Validation: {len(X_val)} sequences")
        print(f"  Test: {len(X_test)} sequences")
        
        # Normalize
        print("\nNormalizing sequences...")
        X_train, X_val, X_test = self.normalize_sequences(X_train, X_val, X_test)
        
        # Build model
        _, sequence_length, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        
        print(f"\nBuilding enhanced model...")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Features: {n_features}")
        print(f"  Classes: {n_classes}")
        
        self.model = self.build_enhanced_model(sequence_length, n_features, n_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train
        print(f"\n{'='*70}")
        print("Training Model...")
        print(f"{'='*70}\n")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=16,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        print(f"\n{'='*70}")
        print("Evaluating Model...")
        print(f"{'='*70}")
        
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nResults:")
        print(f"  Training:   Loss={train_loss:.4f}, Accuracy={train_acc:.1%}")
        print(f"  Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.1%}")
        print(f"  Test:       Loss={test_loss:.4f}, Accuracy={test_acc:.1%}")
        
        # Save model
        print(f"\n{'='*70}")
        print("Saving Model...")
        print(f"{'='*70}")
        
        model_path = os.path.join(MODEL_DIR, "lstm_model_enhanced.keras")
        scaler_path = os.path.join(MODEL_DIR, "sequence_scaler_enhanced.joblib")
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        phrase_mapping = {phrase: idx for idx, phrase in enumerate(PHRASES)}
        with open(mapping_path, 'w') as f:
            json.dump(phrase_mapping, f, indent=2)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
        print(f"✓ Mapping saved to: {mapping_path}")
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nTest Accuracy: {test_acc:.1%}")
        
        if test_acc > 0.85:
            print("🎉 Excellent performance!")
        elif test_acc > 0.70:
            print("✅ Good performance - should work well")
        else:
            print("⚠️  Low performance - collect more diverse data")

if __name__ == "__main__":
    trainer = EnhancedTrainer()
    trainer.train()
