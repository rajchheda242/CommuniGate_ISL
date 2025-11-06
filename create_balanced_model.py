#!/usr/bin/env python3
"""
Create a balanced model - good confidence but not overfitted
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
    """Load only original training sequences"""
    print("📁 Loading Original Training Sequences")
    
    data_dir = "data/sequences"
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    
    all_sequences = []
    all_labels = []
    
    for phrase_idx in range(5):
        phrase_dir = os.path.join(data_dir, f"phrase_{phrase_idx}")
        if not os.path.exists(phrase_dir):
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
                    last_frame = sequence[-1:] if len(sequence) > 0 else np.zeros((1, sequence.shape[1]))
                    padding = np.repeat(last_frame, 60 - sequence.shape[0], axis=0)
                    sequence = np.vstack([sequence, padding])
                
                phrase_sequences.append(sequence)
        
        print(f"  {phrases[phrase_idx]}: {len(phrase_sequences)} sequences")
        
        for seq in phrase_sequences:
            all_sequences.append(seq)
            all_labels.append(phrase_idx)
    
    return np.array(all_sequences), np.array(all_labels)

def create_balanced_model():
    """Create a balanced model - not too simple, not too complex"""
    model = keras.Sequential([
        layers.Input(shape=(60, 126)),
        
        # Balanced architecture
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        # Medium dense layer with moderate regularization
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        
        # Output layer 
        layers.Dense(5, activation='softmax')
    ])
    
    # Balanced learning rate
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def add_moderate_augmentation(X, y, noise_factor=0.01):
    """Add moderate augmentation for good generalization"""
    print("🔀 Adding moderate augmentation...")
    
    X_augmented = []
    y_augmented = []
    
    # Original data (weight it 2x for stability)
    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # Add copy for stability
        X_augmented.append(X[i])
        y_augmented.append(y[i])
    
    # Add moderate noise versions
    for i in range(len(X)):
        noise = np.random.normal(0, noise_factor, X[i].shape)
        noisy_sequence = X[i] + noise
        
        X_augmented.append(noisy_sequence)
        y_augmented.append(y[i])
    
    print(f"   Original samples: {len(X)}")
    print(f"   Augmented samples: {len(X_augmented)}")
    
    return np.array(X_augmented), np.array(y_augmented)

def main():
    print("⚖️  Creating Balanced Model (Good Confidence + No Overfitting)")
    print("=" * 65)
    
    # Load data
    print("\n1️⃣ Loading Data...")
    X, y = load_original_sequences()
    
    # Add moderate augmentation
    print("\n2️⃣ Adding Moderate Augmentation...")
    X_aug, y_aug = add_moderate_augmentation(X, y, noise_factor=0.008)
    
    # Check class distribution
    phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
    print("\n📊 Class Distribution:")
    for i in range(5):
        count = np.sum(y_aug == i)
        print(f"   {phrases[i]}: {count} samples")
    
    # Prepare data with smaller test split for more training data
    print("\n3️⃣ Preparing Data...")
    
    # Flatten sequences for scaler
    X_flat = X_aug.reshape(X_aug.shape[0], -1)
    
    # Smaller test split to give model more training data
    X_train_flat, X_test_flat, y_train, y_test = train_test_split(
        X_flat, y_aug, test_size=0.2, random_state=42, stratify=y_aug
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
    
    # Create balanced model
    print("\n4️⃣ Training Balanced Model...")
    model = create_balanced_model()
    
    print("Model summary:")
    model.summary()
    
    # Moderate early stopping - not too aggressive
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        )
    ]
    
    # Train with balanced approach
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=80,  # More epochs but with early stopping
        batch_size=16,  # Smaller batch size for stability
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n5️⃣ Evaluating Model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy:.3f}")
    
    # Test confidence on training samples
    print("\n6️⃣ Testing Confidence on Known Good Samples...")
    sample_indices = np.random.choice(len(X_train), 10, replace=False)
    confidences = []
    
    for idx in sample_indices:
        sample = X_train[idx:idx+1]  # Keep batch dimension
        predictions = model.predict(sample, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        actual_phrase = phrases[y_train[idx]]
        predicted_phrase = phrases[predicted_idx]
        
        confidences.append(confidence)
        correct = "✅" if predicted_idx == y_train[idx] else "❌"
        print(f"  {correct} {actual_phrase} -> {predicted_phrase} (conf: {confidence:.3f})")
    
    avg_confidence = np.mean(confidences)
    print(f"\n📊 Average confidence on good samples: {avg_confidence:.3f}")
    
    # Test on random noise (bias check)
    print("\n7️⃣ Testing Random Noise (Bias Check)...")
    noise_predictions = []
    
    for i in range(10):
        random_sequence = np.random.random((1, 60, 126))
        random_flat = random_sequence.reshape(1, -1)
        random_scaled = scaler.transform(random_flat)
        random_reshaped = random_scaled.reshape(1, 60, 126)
        
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
        print(f"  {phrase}: {count}/10 times ({percentage:.0f}%)")
        max_count = max(max_count, count)
    
    bias_ok = max_count <= 6  # 60% or less is acceptable
    confidence_ok = avg_confidence >= 0.7  # 70% or higher
    accuracy_ok = test_accuracy >= 0.85  # 85% or higher
    
    # Save model
    print("\n8️⃣ Saving Balanced Model...")
    model.save('models/saved/lstm_model.keras')
    joblib.dump(scaler, 'models/saved/sequence_scaler.joblib')
    
    # Save phrase mapping
    phrase_mapping = {phrase: i for i, phrase in enumerate(phrases)}
    with open('models/saved/phrase_mapping.json', 'w') as f:
        json.dump(phrase_mapping, f)
    
    print("✅ Balanced model trained!")
    print(f"📊 Test Accuracy: {test_accuracy:.3f}")
    print(f"🎯 Average Confidence: {avg_confidence:.3f}")
    
    # Overall assessment
    print(f"\n🎯 ASSESSMENT:")
    print(f"   Accuracy: {'✅' if accuracy_ok else '⚠️'} {test_accuracy:.3f} ({'Good' if accuracy_ok else 'Needs improvement'})")
    print(f"   Confidence: {'✅' if confidence_ok else '⚠️'} {avg_confidence:.3f} ({'Good' if confidence_ok else 'Too low'})")
    print(f"   Bias: {'✅' if bias_ok else '⚠️'} Max {max_count}/10 ({'Good' if bias_ok else 'Still biased'})")
    
    if accuracy_ok and confidence_ok and bias_ok:
        print("🎉 EXCELLENT! This model should work well in real-time!")
    elif confidence_ok and accuracy_ok:
        print("👍 GOOD! Should work well with acceptable bias.")
    else:
        print("⚠️  Needs improvement but better than before.")
    
    return test_accuracy, avg_confidence

if __name__ == "__main__":
    accuracy, confidence = main()