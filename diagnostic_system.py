#!/usr/bin/env python3
"""
Comprehensive diagnostic system for ISL recognition troubleshooting.
This will log everything and help us identify the exact issue.
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import pickle
import os
from collections import deque
import time
from datetime import datetime
import sys

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ISLDiagnosticSystem:
    """Comprehensive diagnostic system to troubleshoot ISL recognition issues."""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"diagnostic_logs/{self.session_id}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize components
        self.setup_mediapipe()
        self.load_model_components()
        self.setup_logging()
        
        # Data collection
        self.sequence_buffer = deque(maxlen=60)
        self.frame_count = 0
        self.prediction_log = []
        
    def setup_mediapipe(self):
        """Initialize MediaPipe holistic."""
        mp_holistic = mp.solutions.holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def load_model_components(self):
        """Load trained model, scaler, and phrase mapping."""
        print("Loading model components...")
        
        # Load model
        model_path = "models/transformer/transformer_model.pth"
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load model architecture
        from inference import TemporalTransformer
        config = checkpoint['model_config']
        self.model = TemporalTransformer(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            max_seq_len=config['sequence_length']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        with open("models/transformer/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load phrase mapping
        with open("models/transformer/phrase_mapping.json", 'r') as f:
            phrase_data = json.load(f)
            self.phrase_mapping = {int(k): v for k, v in phrase_data.items()}
            
        print("✅ Model components loaded successfully")
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        self.log_file = open(f"{self.log_dir}/session_log.txt", 'w')
        self.log(f"Diagnostic Session Started: {self.session_id}")
        self.log(f"Model: Transformer with {sum(p.numel() for p in self.model.parameters())} parameters")
        self.log(f"Phrases: {self.phrase_mapping}")
        
    def log(self, message):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_file.write(log_entry + "\n")
        self.log_file.flush()
        
    def extract_holistic_landmarks(self, results):
        """Extract holistic landmarks - same as inference.py but with logging."""
        landmarks = []
        
        # Pose landmarks (33 points × 4 = 132 features)
        pose_detected = results.pose_landmarks is not None
        if pose_detected:
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            landmarks.extend([0.0] * 132)
        
        # Left hand landmarks (21 points × 3 = 63 features)
        left_hand_detected = results.left_hand_landmarks is not None
        if left_hand_detected:
            for landmark in results.left_hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Right hand landmarks (21 points × 3 = 63 features)
        right_hand_detected = results.right_hand_landmarks is not None
        if right_hand_detected:
            for landmark in results.right_hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Face landmarks (468 points × 3 = 1404 features)
        face_detected = results.face_landmarks is not None
        if face_detected:
            face_coords = []
            for landmark in results.face_landmarks.landmark:
                face_coords.extend([landmark.x, landmark.y, landmark.z])
            
            # Ensure exactly 1404 features
            if len(face_coords) >= 1404:
                landmarks.extend(face_coords[:1404])
            else:
                landmarks.extend(face_coords + [0.0] * (1404 - len(face_coords)))
        else:
            landmarks.extend([0.0] * 1404)
        
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Log detection status
        detection_status = {
            'pose': pose_detected,
            'left_hand': left_hand_detected,
            'right_hand': right_hand_detected,
            'face': face_detected,
            'total_features': len(landmarks_array)
        }
        
        return landmarks_array, detection_status
    
    def analyze_landmarks(self, landmarks, detection_status):
        """Analyze landmark quality and characteristics."""
        analysis = {
            'shape': landmarks.shape,
            'range': [float(landmarks.min()), float(landmarks.max())],
            'mean': float(landmarks.mean()),
            'std': float(landmarks.std()),
            'zeros_count': int(np.count_nonzero(landmarks == 0)),
            'non_zeros_count': int(np.count_nonzero(landmarks)),
            'detection_status': detection_status
        }
        
        return analysis
    
    def predict_with_logging(self, sequence):
        """Make prediction with detailed logging."""
        if len(sequence) < 60:
            return None, 0.0, {"error": "Insufficient frames"}
        
        # Convert to numpy array
        sequence_array = np.array(sequence)
        
        # Log input characteristics
        input_analysis = {
            'input_shape': sequence_array.shape,
            'input_range': [float(sequence_array.min()), float(sequence_array.max())],
            'input_mean': float(sequence_array.mean()),
            'input_std': float(sequence_array.std()),
            'input_zeros': int(np.count_nonzero(sequence_array == 0))
        }
        
        # Apply scaling
        sequence_flat = sequence_array.reshape(-1, 1662)
        sequence_scaled = self.scaler.transform(sequence_flat)
        sequence_scaled = sequence_scaled.reshape(1, 60, 1662)
        
        # Log scaled characteristics
        scaled_analysis = {
            'scaled_shape': sequence_scaled.shape,
            'scaled_range': [float(sequence_scaled.min()), float(sequence_scaled.max())],
            'scaled_mean': float(sequence_scaled.mean()),
            'scaled_std': float(sequence_scaled.std())
        }
        
        # Convert to tensor and predict
        sequence_tensor = torch.FloatTensor(sequence_scaled)
        
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        phrase = self.phrase_mapping.get(predicted_class, "Unknown")
        
        # Log prediction details
        prediction_analysis = {
            'raw_logits': outputs[0].tolist(),
            'probabilities': probabilities[0].tolist(),
            'predicted_class': predicted_class,
            'predicted_phrase': phrase,
            'confidence': confidence,
            'all_class_probs': {
                f"Class {i} ({self.phrase_mapping[i]})": float(probabilities[0][i]) 
                for i in range(len(self.phrase_mapping))
            }
        }
        
        return phrase, confidence, {
            'input_analysis': input_analysis,
            'scaled_analysis': scaled_analysis,
            'prediction_analysis': prediction_analysis
        }
    
    def save_landmarks_sample(self, landmarks, label, frame_count):
        """Save landmarks sample for comparison with training data."""
        sample_data = {
            'landmarks': landmarks.tolist(),
            'label': label,
            'frame_count': frame_count,
            'timestamp': datetime.now().isoformat(),
            'shape': landmarks.shape,
            'stats': {
                'min': float(landmarks.min()),
                'max': float(landmarks.max()),
                'mean': float(landmarks.mean()),
                'std': float(landmarks.std())
            }
        }
        
        filename = f"{self.log_dir}/landmarks_sample_{frame_count}_{label.replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(sample_data, f, indent=2)
            
        self.log(f"Saved landmarks sample: {filename}")
    
    def run_diagnostic(self):
        """Run interactive diagnostic session."""
        print(f"\n🔍 ISL Recognition Diagnostic System")
        print(f"Session ID: {self.session_id}")
        print(f"Logs will be saved to: {self.log_dir}")
        print("="*60)
        print("Instructions:")
        print("1. Press number keys (0-4) when performing specific phrases:")
        print("   0 - Hi my name is Reet")
        print("   1 - How are you") 
        print("   2 - I am from Delhi")
        print("   3 - I like coffee")
        print("   4 - What do you like")
        print("2. Press 's' to save current landmarks sample")
        print("3. Press 'p' to force a prediction")
        print("4. Press 'q' to quit")
        print("="*60)
        
        self.log("Diagnostic session started")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log("ERROR: Cannot open camera")
            return
        
        current_label = "unknown"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_frame)
            
            # Extract landmarks
            landmarks, detection_status = self.extract_holistic_landmarks(results)
            
            # Analyze landmarks
            analysis = self.analyze_landmarks(landmarks, detection_status)
            
            # Add to sequence buffer
            self.sequence_buffer.append(landmarks)
            
            # Draw landmarks
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            if results.face_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
            
            # Display info
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Buffer: {len(self.sequence_buffer)}/60", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {current_label}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Detection status
            status_text = f"P:{detection_status['pose']} LH:{detection_status['left_hand']} RH:{detection_status['right_hand']} F:{detection_status['face']}"
            cv2.putText(frame, status_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('ISL Diagnostic', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
                phrase_idx = key - ord('0')
                current_label = self.phrase_mapping[phrase_idx]
                self.log(f"User performing: {current_label}")
                
                # Save landmarks sample
                self.save_landmarks_sample(landmarks, current_label, self.frame_count)
                
            elif key == ord('s'):
                self.save_landmarks_sample(landmarks, current_label, self.frame_count)
                
            elif key == ord('p') and len(self.sequence_buffer) >= 60:
                # Force prediction
                phrase, confidence, detailed_analysis = self.predict_with_logging(list(self.sequence_buffer))
                
                prediction_entry = {
                    'frame_count': self.frame_count,
                    'expected_label': current_label,
                    'predicted_phrase': phrase,
                    'confidence': confidence,
                    'detailed_analysis': detailed_analysis,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.prediction_log.append(prediction_entry)
                
                self.log(f"PREDICTION - Expected: {current_label}, Got: {phrase}, Confidence: {confidence:.3f}")
                
                # Show prediction on screen
                pred_text = f"Expected: {current_label}"
                cv2.putText(frame, pred_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                pred_text2 = f"Got: {phrase} ({confidence:.3f})"
                cv2.putText(frame, pred_text2, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final prediction log
        with open(f"{self.log_dir}/prediction_log.json", 'w') as f:
            json.dump(self.prediction_log, f, indent=2)
            
        self.log("Diagnostic session completed")
        self.log_file.close()
        
        print(f"\n📊 Session Complete! Check logs in: {self.log_dir}")

if __name__ == "__main__":
    diagnostic = ISLDiagnosticSystem()
    diagnostic.run_diagnostic()
