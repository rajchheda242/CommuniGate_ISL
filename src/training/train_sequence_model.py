"""
LSTM-based model training for temporal ISL gesture sequences.
Trains a neural network on sequence data for multi-word phrase recognition.
"""

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import Keras via TensorFlow (recommended for TF 2.x)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
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


class SequenceModelTrainer:
    """Trains LSTM model on temporal gesture sequences."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def load_sequences(self):
        """Load all sequence data from directories."""
        X = []
        y = []
        
        print("Loading sequence data...")
        
        for phrase_idx in range(len(PHRASES)):
            phrase_dir = os.path.join(DATA_DIR, f"phrase_{phrase_idx}")
            
            if not os.path.exists(phrase_dir):
                print(f"Warning: No data found for phrase {phrase_idx}")
                continue
            
            sequence_files = glob.glob(os.path.join(phrase_dir, "*_seq.npy"))
            
            for seq_file in sequence_files:
                sequence = np.load(seq_file)
                X.append(sequence)
                y.append(phrase_idx)
            
            print(f"Phrase {phrase_idx}: Loaded {len(sequence_files)} sequences - '{PHRASES[phrase_idx]}'")
        
        if len(X) == 0:
            raise ValueError(f"No sequence data found in {DATA_DIR}")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nTotal sequences loaded: {len(X)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X, y
    
    def normalize_sequences(self, X_train, X_test):
        """Normalize features across all sequences."""
        # Reshape for scaling: (num_samples * num_frames, num_features)
        n_samples_train, n_frames, n_features = X_train.shape
        n_samples_test = X_test.shape[0]
        
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        # Reshape back to sequences
        X_train_scaled = X_train_scaled.reshape(n_samples_train, n_frames, n_features)
        X_test_scaled = X_test_scaled.reshape(n_samples_test, n_frames, n_features)
        
        return X_train_scaled, X_test_scaled
    
    def build_lstm_model(self, sequence_length, n_features, n_classes):
        """Build LSTM model for sequence classification."""
        model = Sequential([
            # Input layer
            Input(shape=(sequence_length, n_features)),
            
            # First LSTM layer (return sequences for next LSTM)
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            
            # Second LSTM layer
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model."""
        _, sequence_length, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        
        print(f"\n{'='*70}")
        print("Building LSTM Model...")
        print(f"{'='*70}")
        print(f"Input shape: (sequence_length={sequence_length}, features={n_features})")
        print(f"Number of classes: {n_classes}")
        
        self.model = self.build_lstm_model(sequence_length, n_features, n_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        print(f"\n{'='*70}")
        print("Training Model...")
        print(f"{'='*70}\n")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=8,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        print(f"\n{'='*70}")
        print("Model Evaluation")
        print(f"{'='*70}")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2%}")
        
        # Per-class accuracy
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        
        print("\nPer-Phrase Accuracy:")
        for phrase_idx in range(len(PHRASES)):
            mask = y_test == phrase_idx
            if mask.sum() > 0:
                phrase_accuracy = (y_pred[mask] == y_test[mask]).mean()
                print(f"  Phrase {phrase_idx} ('{PHRASES[phrase_idx][:30]}...'): {phrase_accuracy:.2%}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return accuracy
    
    def save_model(self):
        """Save trained model and metadata."""
        # Save Keras model
        model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
        self.model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save scaler (use .pkl extension for compatibility with app)
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save phrase mapping (inverted format: phrase -> id for compatibility)
        mapping = {phrase: i for i, phrase in enumerate(PHRASES)}
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Phrase mapping saved to: {mapping_path}")
    
    def run(self):
        """Main training pipeline."""
        print("="*70)
        print("LSTM SEQUENCE MODEL TRAINING")
        print("="*70)
        
        # Load data
        X, y = self.load_sequences()
        
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
        X_train_scaled, X_val_scaled = self.normalize_sequences(X_train, X_val)
        X_train_scaled, X_test_scaled = self.normalize_sequences(X_train, X_test)
        
        # Train
        history = self.train(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate
        accuracy = self.evaluate(X_test_scaled, y_test)
        
        # Save
        self.save_model()
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Final Test Accuracy: {accuracy:.2%}")
        print("\nNext steps:")
        print("  Run the app: streamlit run app_enhanced.py")


if __name__ == "__main__":
    trainer = SequenceModelTrainer()
    trainer.run()
