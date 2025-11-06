#!/usr/bin/env python3
"""
Test the newly trained enhanced model
Tests on both training data and live camera input
"""

import numpy as np
import tensorflow as tf
import joblib
import json
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import mediapipe as mp

# Paths
MODEL_PATH = 'models/saved/lstm_model_enhanced.keras'
SCALER_PATH = 'models/saved/sequence_scaler_enhanced.joblib'
PHRASE_MAPPING_PATH = 'models/saved/phrase_mapping.json'
DATA_DIR = 'data/sequences'

def test_on_training_data():
    """Test model on training sequences"""
    print("\n" + "="*80)
    print("TEST 1: MODEL ACCURACY ON TRAINING DATA")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(PHRASE_MAPPING_PATH, 'r') as f:
        phrase_mapping = json.load(f)
    
    reverse_mapping = {v: k for k, v in phrase_mapping.items()}
    
    # Load all sequences
    all_sequences = []
    all_labels = []
    
    for phrase_idx in range(5):
        phrase_dir = os.path.join(DATA_DIR, f"phrase_{phrase_idx}")
        files = glob.glob(os.path.join(phrase_dir, "*_seq.npy"))
        
        for file in files:
            seq = np.load(file)
            all_sequences.append(seq)
            all_labels.append(phrase_idx)
    
    print(f"Loaded {len(all_sequences)} sequences")
    
    # Normalize
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    n_samples = X.shape[0]
    n_frames = X.shape[1]
    n_features = X.shape[2]
    
    # Scale frame-by-frame (same as enhanced_train.py)
    X_flat = X.reshape(-1, n_features)
    X_normalized = scaler.transform(X_flat)
    X_normalized = X_normalized.reshape(n_samples, n_frames, n_features)
    
    # Predict
    print("\nMaking predictions...")
    predictions = model.predict(X_normalized, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == y)
    
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY: {accuracy*100:.1f}%")
    print(f"{'='*80}")
    
    # Per-phrase accuracy
    print("\nPer-Phrase Accuracy:")
    print("-"*80)
    for phrase_idx in range(5):
        mask = y == phrase_idx
        phrase_acc = np.mean(predicted_labels[mask] == y[mask])
        phrase_name = reverse_mapping[phrase_idx]
        correct = np.sum(predicted_labels[mask] == y[mask])
        total = np.sum(mask)
        print(f"Phrase {phrase_idx} ('{phrase_name}'): {phrase_acc*100:.1f}% ({correct}/{total} correct)")
    
    # Confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    cm = confusion_matrix(y, predicted_labels)
    
    print("\n           ", end="")
    for i in range(5):
        print(f"P{i:1d}  ", end="")
    print()
    print("           " + "-"*25)
    
    for i in range(5):
        print(f"Actual P{i}: ", end="")
        for j in range(5):
            print(f"{cm[i][j]:3d} ", end="")
        print()
    
    # Show misclassifications
    print("\n" + "="*80)
    print("MISCLASSIFICATIONS")
    print("="*80)
    misclassified = np.where(predicted_labels != y)[0]
    
    if len(misclassified) == 0:
        print("🎉 No misclassifications! Perfect accuracy!")
    else:
        print(f"Total misclassified: {len(misclassified)}/{len(y)} ({len(misclassified)/len(y)*100:.1f}%)\n")
        
        # Group by actual phrase
        for phrase_idx in range(5):
            phrase_misclas = []
            for idx in misclassified:
                if y[idx] == phrase_idx:
                    phrase_misclas.append((idx, predicted_labels[idx]))
            
            if phrase_misclas:
                print(f"Phrase {phrase_idx} ('{reverse_mapping[phrase_idx]}'):")
                for idx, pred in phrase_misclas[:5]:  # Show first 5
                    print(f"  - Predicted as: Phrase {pred} ('{reverse_mapping[pred]}')")
                if len(phrase_misclas) > 5:
                    print(f"  ... and {len(phrase_misclas)-5} more")
                print()
    
    return accuracy


def test_live_camera():
    """Test model with live camera input"""
    print("\n" + "="*80)
    print("TEST 2: LIVE CAMERA TESTING")
    print("="*80)
    print("\nInstructions:")
    print("- Perform any of the 5 phrases")
    print("- Model will predict in real-time")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset and try again")
    print("\nPhrases:")
    print("  0: Hi my name is Reet")
    print("  1: How are you")
    print("  2: I am from Delhi")
    print("  3: I like coffee")
    print("  4: What do you like")
    
    input("\nPress ENTER to start camera...")
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(PHRASE_MAPPING_PATH, 'r') as f:
        phrase_mapping = json.load(f)
    
    reverse_mapping = {v: k for k, v in phrase_mapping.items()}
    
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    cap = cv2.VideoCapture(0)
    sequence = []
    predictions_log = []
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Extract landmarks
            frame_data = np.zeros(126)
            hands_detected = False
            
            if results.multi_hand_landmarks:
                hands_detected = True
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if idx < 2:
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        start_idx = idx * 63
                        frame_data[start_idx:start_idx + 63] = landmarks
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Add to sequence
            sequence.append(frame_data)
            
            # Keep sequence at 90 frames
            if len(sequence) > 90:
                sequence.pop(0)
            
            # Predict when we have enough frames
            if len(sequence) == 90 and hands_detected:
                seq_array = np.array([sequence])
                # Scale frame-by-frame
                seq_flat = seq_array.reshape(-1, 126)
                seq_normalized = scaler.transform(seq_flat)
                seq_normalized = seq_normalized.reshape(1, 90, 126)
                
                prediction = model.predict(seq_normalized, verbose=0)[0]
                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class] * 100
                
                predictions_log.append({
                    'class': predicted_class,
                    'confidence': confidence,
                    'phrase': reverse_mapping[predicted_class]
                })
                
                # Keep last 10 predictions for stability
                if len(predictions_log) > 10:
                    predictions_log.pop(0)
                
                # Stable prediction (most common in last 5)
                if len(predictions_log) >= 5:
                    recent_classes = [p['class'] for p in predictions_log[-5:]]
                    most_common = max(set(recent_classes), key=recent_classes.count)
                    
                    if recent_classes.count(most_common) >= 3:  # Stable
                        stable_phrase = reverse_mapping[most_common]
                        avg_conf = np.mean([p['confidence'] for p in predictions_log[-5:] 
                                          if p['class'] == most_common])
                        
                        cv2.putText(frame, f"Prediction: {stable_phrase}", (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {avg_conf:.1f}%", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display info
            cv2.putText(frame, f"Frames: {len(sequence)}/90", (10, frame.shape[0] - 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Live Testing', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                sequence = []
                predictions_log = []
                print("Reset sequence")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Live testing complete!")


def main():
    print("\n" + "="*80)
    print("ENHANCED MODEL TESTING")
    print("="*80)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("   Run: python enhanced_train.py")
        return
    
    # Test 1: Training data accuracy
    accuracy = test_on_training_data()
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Training accuracy: {accuracy*100:.1f}%")
    
    if accuracy >= 0.90:
        print("🎉 EXCELLENT! Model is performing great!")
    elif accuracy >= 0.80:
        print("✅ GOOD! Model is working well.")
    elif accuracy >= 0.70:
        print("⚠️  FAIR - Model could use more training data.")
    else:
        print("❌ POOR - Model needs improvement.")
    
    # Test 2: Live camera
    print("\n" + "="*80)
    response = input("\nTest with live camera? (yes/no): ").strip().lower()
    
    if response == 'yes':
        test_live_camera()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  - If accuracy is good (>80%), you can use the model in production")
    print("  - If accuracy is low, record more training data")
    print("  - To use in the app: python src/ui/app.py")
    print("="*80)


if __name__ == "__main__":
    main()
