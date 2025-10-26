"""
Test script for evaluating the trained LSTM model.
Loads test data and evaluates model performance.
"""

import numpy as np
import os
import glob
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tensorflow.keras.models import load_model


DATA_DIR = "data/sequences"
MODEL_DIR = "models/saved"
PHRASES = [
    "Hi my name is Reet",
    "How are you",
    "I am from Delhi",
    "I like coffee",
    "What do you like"
]


def load_test_data():
    """Load all sequence data for testing."""
    X = []
    y = []
    
    print("Loading sequence data for testing...")
    
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
        
        print(f"  Phrase {phrase_idx}: {len(sequence_files)} sequences - '{PHRASES[phrase_idx]}'")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTotal sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    
    return X, y


def normalize_data(X, scaler_path):
    """Normalize data using saved scaler."""
    scaler = joblib.load(scaler_path)
    
    n_samples, n_frames, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_frames, n_features)
    
    return X_scaled


def evaluate_model():
    """Evaluate the trained model."""
    print("="*70)
    print("LSTM Model Evaluation")
    print("="*70)
    
    # Load model and scaler
    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    scaler_path = os.path.join(MODEL_DIR, "sequence_scaler.joblib")
    mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print("\nLoading model and scaler...")
    model = load_model(model_path)
    
    with open(mapping_path, 'r') as f:
        phrase_mapping = json.load(f)
    
    # Load and prepare data
    X, y = load_test_data()
    
    # Split data (use same random state as training for consistency)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTest set size: {len(X_test)} sequences")
    
    # Normalize test data
    X_test_scaled = normalize_data(X_test, scaler_path)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\n{'='*70}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"{'='*70}")
    
    # Predictions
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print("="*70)
    target_names = [phrase_mapping[str(i)] for i in range(len(PHRASES))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("="*70)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print formatted confusion matrix
    print("\n" + " "*15 + "Predicted")
    print(" "*10 + "".join([f"P{i:2d} " for i in range(len(PHRASES))]))
    print(" "*10 + "-"*40)
    
    for i, row in enumerate(cm):
        print(f"Actual P{i}  |  " + "".join([f"{val:3d} " for val in row]))
    
    print("\nPhrase Legend:")
    for i, phrase in enumerate(PHRASES):
        print(f"  P{i}: {phrase}")
    
    # Per-class accuracy
    print("\n" + "="*70)
    print("Per-Class Accuracy:")
    print("="*70)
    
    for i in range(len(PHRASES)):
        class_mask = y_test == i
        class_accuracy = np.mean(y_pred[class_mask] == y_test[class_mask])
        print(f"  {PHRASES[i]:<40} {class_accuracy:.2%}")
    
    # Confidence analysis
    print("\n" + "="*70)
    print("Confidence Analysis:")
    print("="*70)
    
    confidences = np.max(y_pred_probs, axis=1)
    print(f"  Average confidence: {np.mean(confidences):.2%}")
    print(f"  Min confidence: {np.min(confidences):.2%}")
    print(f"  Max confidence: {np.max(confidences):.2%}")
    
    # Prediction confidence distribution
    high_conf = np.sum(confidences > 0.8)
    med_conf = np.sum((confidences > 0.5) & (confidences <= 0.8))
    low_conf = np.sum(confidences <= 0.5)
    
    print(f"\n  High confidence (>80%): {high_conf} ({high_conf/len(confidences):.1%})")
    print(f"  Medium confidence (50-80%): {med_conf} ({med_conf/len(confidences):.1%})")
    print(f"  Low confidence (<50%): {low_conf} ({low_conf/len(confidences):.1%})")
    
    print("\n" + "="*70)
    print("✓ Evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
