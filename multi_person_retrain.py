#!/usr/bin/env python3
"""
Multi-Person Model Retraining
Combines your original data with multi-person data for better generalization
"""

import numpy as np
import os
import json
import joblib
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import gc

def load_all_training_data():
    """Load both original and multi-person training data"""
    print("📂 Loading Training Data")
    print("="*50)
    
    all_sequences = []
    all_labels = []
    
    phrases = [
        "Hi my name is Reet",
        "How are you", 
        "I am from Delhi",
        "I like coffee",
        "What do you like"
    ]
    
    # 1. Load original current environment data (your data)
    print("Loading your original training data...")
    current_env_dir = "data/sequences_current_env"
    if os.path.exists(current_env_dir):
        for phrase_idx in range(5):
            phrase_dir = os.path.join(current_env_dir, f"phrase_{phrase_idx}")
            if os.path.exists(phrase_dir):
                files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
                for file in files:
                    filepath = os.path.join(phrase_dir, file)
                    sequence = np.load(filepath)
                    if sequence.shape == (60, 1662):
                        # Weight your data 2x to maintain good performance for you
                        for _ in range(2):
                            all_sequences.append(sequence.flatten())
                            all_labels.append(phrase_idx)
                print(f"  ✅ {phrases[phrase_idx]}: {len(files)} × 2 = {len(files)*2} samples")
    
    # 2. Load multi-person data
    print("\nLoading multi-person training data...")
    multi_person_dir = "data/sequences_multi_person"
    if os.path.exists(multi_person_dir):
        for phrase_idx in range(5):
            phrase_dir = os.path.join(multi_person_dir, f"phrase_{phrase_idx}")
            if os.path.exists(phrase_dir):
                files = [f for f in os.listdir(phrase_dir) if f.endswith('.npy')]
                for file in files:
                    filepath = os.path.join(phrase_dir, file)
                    sequence = np.load(filepath)
                    if sequence.shape == (60, 1662):
                        all_sequences.append(sequence.flatten())
                        all_labels.append(phrase_idx)
                print(f"  ✅ {phrases[phrase_idx]}: {len(files)} samples")
    
    # 3. Load additional samples for phrase 4 (if they exist)
    additional_dir = "data/sequences_current_env/phrase_4"
    if os.path.exists(additional_dir):
        additional_files = [f for f in os.listdir(additional_dir) if f.startswith('additional_') and f.endswith('.npy')]
        if additional_files:
            print(f"\nLoading additional phrase 4 samples...")
            for file in additional_files:
                filepath = os.path.join(additional_dir, file)
                sequence = np.load(filepath)
                if sequence.shape == (60, 1662):
                    all_sequences.append(sequence.flatten())
                    all_labels.append(4)
            print(f"  ✅ Additional 'What do you like': {len(additional_files)} samples")
    
    if not all_sequences:
        raise ValueError("No training data found!")
    
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    print(f"\n📊 Total Training Data: {X.shape[0]} samples")
    for i in range(5):
        count = np.sum(y == i)
        print(f"  {phrases[i]}: {count} samples")
    
    return X, y

def create_improved_model():
    """Create an improved LSTM model with better generalization"""
    model = keras.Sequential([
        layers.Input(shape=(60, 1662)),
        
        # More robust feature extraction
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        # Regularized dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(5, activation='softmax')
    ])
    
    return model

def main():
    """Main retraining function"""
    print("🚀 Multi-Person Model Retraining")
    print("="*50)
    
    # Load all data
    X, y = load_all_training_data()
    
    # Create and fit scaler
    print("\n⚖️ Scaling Data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM
    X_lstm = X_scaled.reshape(X_scaled.shape[0], 60, 1662)
    
    # Convert to categorical
    y_categorical = keras.utils.to_categorical(y, num_classes=5)
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_categorical, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Compute class weights for balanced training
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y), 
        y=y
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"\n⚖️ Class weights: {class_weight_dict}")
    
    # Create model
    print("\n🏗️ Creating Improved Model...")
    model = create_improved_model()
    
    # Compile with appropriate settings for generalization
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks for better training
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
            min_lr=0.0001
        )
    ]
    
    # Train model
    print("\n🏋️ Training Multi-Person Model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n📈 Evaluating Model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Loss: {test_loss:.3f}")
    
    # Save model and components
    print("\n💾 Saving Model...")
    model.save('models/saved/lstm_model.keras')
    joblib.dump(scaler, 'models/saved/sequence_scaler.joblib')
    
    # Save updated phrase mapping
    phrase_mapping = {
        "0": "Hi my name is Reet",
        "1": "How are you",
        "2": "I am from Delhi",
        "3": "I like coffee",
        "4": "What do you like"
    }
    
    with open('models/saved/phrase_mapping.json', 'w') as f:
        json.dump(phrase_mapping, f, indent=2)
    
    print("✅ Multi-person model saved successfully!")
    print("\n🎯 Model should now work better for different people!")
    
    # Training summary
    final_train_acc = max(history.history['accuracy'])
    final_val_acc = max(history.history['val_accuracy'])
    
    print(f"\n📊 Training Summary:")
    print(f"   Best Training Accuracy: {final_train_acc:.3f}")
    print(f"   Best Validation Accuracy: {final_val_acc:.3f}")
    print(f"   Final Test Accuracy: {test_accuracy:.3f}")
    
    # Clean up memory
    del X, y, X_scaled, X_lstm, X_train, X_test, y_train, y_test
    gc.collect()

if __name__ == "__main__":
    main()