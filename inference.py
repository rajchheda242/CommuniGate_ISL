"""
Real-time inference for ISL gesture recognition using Temporal Transformer.
Provides webcam demo with MediaPipe Holistic and trained Transformer model.
"""

import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import json
import pickle
import os
import math
from collections import deque
from typing import Optional, Tuple
import time


# Configuration
MODEL_DIR = "models/transformer"
SEQUENCE_LENGTH = 60  # Reduced from 90 frames for faster response
FEATURE_DIM = 1662
MIN_CONFIDENCE = 0.7  # Minimum confidence threshold

# Model hyperparameters (must match training)
D_MODEL = 256  # Updated to match training
NUM_HEADS = 4  # Updated to match training
NUM_LAYERS = 3  # Updated to match training
NUM_CLASSES = 5
DROPOUT = 0.1


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TemporalTransformer(nn.Module):
    """Temporal Transformer for ISL gesture sequence classification."""
    
    def __init__(self, input_dim, d_model, num_heads, num_layers, 
                 num_classes, dropout=0.1, max_seq_len=100):
        super(TemporalTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=1024,  # Updated to match training
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Temperature scaling for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward pass."""
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model)  # Scale by sqrt(d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoding
        encoded = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling across time dimension
        pooled = torch.mean(encoded, dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        
        # Apply temperature scaling for calibration
        calibrated_logits = logits / self.temperature
        
        return calibrated_logits


class HolisticInference:
    """Real-time ISL inference using MediaPipe Holistic and Transformer."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load preprocessing components
        self.scaler = None
        self.phrase_mapping = None
        self.training_stats = None
        
        # Sequence buffer for temporal analysis
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # Load trained model
        self.load_model()
        
        # Calculate training statistics for robust normalization
        self.calculate_training_stats()
    
    def load_model(self):
        """Load trained Transformer model and preprocessing."""
        print("Loading trained Transformer model...")
        
        # Load model
        model_path = os.path.join(MODEL_DIR, "transformer_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with saved config
        config = checkpoint['model_config']
        self.model = TemporalTransformer(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            max_seq_len=config['sequence_length']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load temperature scaling parameter
        self.model.temperature.data = torch.tensor([checkpoint['temperature']])
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load phrase mapping
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
        with open(mapping_path, 'r') as f:
            phrase_data = json.load(f)
            self.phrase_mapping = {int(k): v for k, v in phrase_data.items()}
        
        print("✓ Model loaded successfully!")
        print(f"✓ Recognized phrases: {list(self.phrase_mapping.values())}")
        print(f"✓ Sequence length: {SEQUENCE_LENGTH} frames")
    
    def calculate_training_stats(self):
        """Calculate training data statistics for robust normalization."""
        print("Calculating training data statistics for robust preprocessing...")
        
        all_data = []
        
        # Load all training data
        for phrase_id in range(5):
            phrase_dir = f"data/sequences_holistic/phrase_{phrase_id}"
            if os.path.exists(phrase_dir):
                import glob
                for file_path in glob.glob(os.path.join(phrase_dir, "*.npy")):
                    try:
                        data = np.load(file_path)
                        all_data.append(data)
                    except:
                        continue
        
        if all_data:
            combined_data = np.concatenate(all_data, axis=0)
            
            self.training_stats = {
                'mean': np.mean(combined_data, axis=0),
                'std': np.std(combined_data, axis=0),
                'min': np.min(combined_data, axis=0),
                'max': np.max(combined_data, axis=0)
            }
            
            print(f"✓ Training stats calculated from {combined_data.shape[0]} frames")
        else:
            print("⚠️  No training data found for robust normalization")
            self.training_stats = None
    
    def robust_normalize_landmarks(self, landmarks):
        """Apply robust normalization to landmarks before scaling."""
        if landmarks is None or self.training_stats is None:
            return landmarks
            
        # Make a copy to avoid modifying original
        normalized = landmarks.copy()
        
        # 1. Clip extreme values (camera coordinates should be roughly in [0, 1])
        normalized = np.clip(normalized, -2.0, 3.0)
        
        # 2. For zero values (missing landmarks), use training data mean
        zero_mask = (landmarks == 0)
        if np.any(zero_mask):
            normalized[zero_mask] = self.training_stats['mean'][zero_mask]
        
        return normalized
    
    def extract_pose_landmarks(self, results) -> np.ndarray:
        """Extract pose landmarks (33 points * 4 features = 132 features)."""
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([
                    landmark.x, landmark.y, landmark.z, landmark.visibility
                ])
            return np.array(landmarks)
        else:
            return np.zeros(33 * 4)  # 132 features
    
    def extract_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract hand landmarks (21 points * 3 features = 63 features per hand)."""
        if hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        else:
            return np.zeros(21 * 3)  # 63 features
    
    def extract_face_landmarks(self, results) -> np.ndarray:
        """Extract face landmarks (468 points * 3 features = 1404 features)."""
        if results.face_landmarks:
            landmarks = []
            for landmark in results.face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            # Ensure we have exactly 1404 features (468 * 3)
            landmarks = np.array(landmarks)
            if len(landmarks) > 1404:
                landmarks = landmarks[:1404]
            elif len(landmarks) < 1404:
                # Pad with zeros if we have fewer landmarks
                landmarks = np.pad(landmarks, (0, 1404 - len(landmarks)))
            return landmarks
        else:
            return np.zeros(468 * 3)  # 1404 features
    
    def extract_holistic_landmarks(self, results) -> np.ndarray:
        """Extract all holistic landmarks into a single feature vector."""
        # Extract individual components
        pose_landmarks = self.extract_pose_landmarks(results)
        left_hand_landmarks = self.extract_hand_landmarks(results.left_hand_landmarks)
        right_hand_landmarks = self.extract_hand_landmarks(results.right_hand_landmarks)
        face_landmarks = self.extract_face_landmarks(results)
        
        # Concatenate all landmarks
        # Total: 132 (pose) + 63 (left hand) + 63 (right hand) + 1404 (face) = 1662 features
        all_landmarks = np.concatenate([
            pose_landmarks,
            left_hand_landmarks,
            right_hand_landmarks,
            face_landmarks
        ])
        
        return all_landmarks
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply robust scale-invariant normalization to landmarks."""
        if len(landmarks) != 1662:
            print(f"Warning: Expected 1662 features, got {len(landmarks)}. Padding or truncating.")
            if len(landmarks) > 1662:
                landmarks = landmarks[:1662]
            else:
                landmarks = np.pad(landmarks, (0, 1662 - len(landmarks)))
        
        # Create a copy to avoid modifying original
        normalized = landmarks.copy()
        
        # Extract components with proper indexing
        pose_landmarks = normalized[:132].reshape(33, 4)  # x, y, z, visibility
        left_hand = normalized[132:195].reshape(21, 3)    # x, y, z
        right_hand = normalized[195:258].reshape(21, 3)   # x, y, z  
        face_landmarks = normalized[258:1662].reshape(468, 3)  # x, y, z
        
        # More robust pose normalization
        if np.any(pose_landmarks[:, :3]):  # Check if pose data exists
            # Center pose landmarks relative to torso center (more stable reference)
            left_shoulder = pose_landmarks[11, :3]
            right_shoulder = pose_landmarks[12, :3] 
            left_hip = pose_landmarks[23, :3]
            right_hip = pose_landmarks[24, :3]
            
            # Use torso center for translation
            torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
            pose_landmarks[:, :3] -= torso_center
            
            # Use shoulder width for consistent scale
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            if shoulder_width > 1e-8:
                pose_landmarks[:, :3] /= shoulder_width
            
            # Additional clipping for problematic pose points (28-32)
            pose_landmarks[28:33, :3] = np.clip(pose_landmarks[28:33, :3], -2.0, 2.0)
        
        # Normalize hands relative to wrists (if detected)
        for hand_landmarks, hand_name in [(left_hand, 'left'), (right_hand, 'right')]:
            if np.any(hand_landmarks):
                # Use wrist as reference point (index 0)
                wrist = hand_landmarks[0]
                if np.any(wrist):
                    hand_landmarks -= wrist
                    
                    # Scale by hand span
                    hand_span = np.max(np.abs(hand_landmarks))
                    if hand_span > 1e-8:
                        hand_landmarks /= hand_span
        
        # Simpler face normalization
        if np.any(face_landmarks):
            face_center = np.mean(face_landmarks, axis=0)
            face_landmarks -= face_center
            
            face_span = np.max(np.abs(face_landmarks))
            if face_span > 1e-8:
                face_landmarks /= face_span
        
        # Reassemble
        normalized[:132] = pose_landmarks.flatten()
        normalized[132:195] = left_hand.flatten()
        normalized[195:258] = right_hand.flatten()
        normalized[258:1662] = face_landmarks.flatten()
        
        # Final clipping to prevent extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized
    
    def process_frame(self, frame) -> np.ndarray:
        """Process single frame and extract normalized landmarks."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Holistic
        results = self.holistic.process(rgb_frame)
        
        # Extract landmarks
        frame_landmarks = self.extract_holistic_landmarks(results)
        
        # Apply normalization
        normalized_landmarks = self.normalize_landmarks(frame_landmarks)
        
        return normalized_landmarks, results
    
    def predict_sequence(self) -> Tuple[Optional[str], float]:
        """Predict phrase from buffered sequence."""
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0.0
        
        # Convert buffer to numpy array
        sequence = np.array(list(self.sequence_buffer))  # Shape: (60, 1662)
        
        return self.predict_from_sequence(sequence)
    
    def predict_from_sequence(self, sequence: np.ndarray) -> Tuple[Optional[str], float]:
        """Predict phrase from any given sequence."""
        if sequence.shape[0] != SEQUENCE_LENGTH:
            return None, 0.0
        
        # Apply scaling (same as training)
        sequence_flat = sequence.reshape(-1, FEATURE_DIM)
        sequence_scaled = self.scaler.transform(sequence_flat)
        sequence_scaled = sequence_scaled.reshape(1, SEQUENCE_LENGTH, FEATURE_DIM)  # Model expects 60 frames
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        phrase = self.phrase_mapping.get(predicted_class, "Unknown")
        
        return phrase, confidence
    
    def resample_sequence_to_90(self, sequence: np.ndarray) -> np.ndarray:
        """Resample 60-frame sequence to 90 frames for model compatibility."""
        if len(sequence) == 90:
            return sequence
        
        # Create indices for resampling
        original_indices = np.linspace(0, len(sequence) - 1, len(sequence))
        target_indices = np.linspace(0, len(sequence) - 1, 90)
        
        # Interpolate each feature dimension
        resampled = np.zeros((90, sequence.shape[1]))
        
        for feature_idx in range(sequence.shape[1]):
            resampled[:, feature_idx] = np.interp(
                target_indices, 
                original_indices, 
                sequence[:, feature_idx]
            )
        
        return resampled
    
    def draw_landmarks(self, frame, results):
        """Draw MediaPipe landmarks on frame."""
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks (corrected for mirrored display)
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw face landmarks (simplified)
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
    
    def draw_ui(self, frame, phrase, confidence, buffer_size):
        """Draw user interface elements on frame."""
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(
            frame,
            'ISL Transformer Recognition',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Current prediction
        if phrase and confidence >= MIN_CONFIDENCE:
            color = (0, 255, 0)  # Green
            display_text = f'Phrase: "{phrase}"'
        elif phrase:
            color = (0, 255, 255)  # Yellow
            display_text = f'Low confidence: "{phrase}"'
        else:
            color = (255, 255, 255)  # White
            display_text = 'Collecting sequence...'
        
        cv2.putText(
            frame,
            display_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Confidence
        if phrase:
            cv2.putText(
                frame,
                f'Confidence: {confidence:.3f}',
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Buffer status with progress bar
        buffer_text = f'Buffer: {buffer_size}/{SEQUENCE_LENGTH}'
        cv2.putText(
            frame,
            buffer_text,
            (width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Progress bar for buffer
        bar_width = 180
        bar_height = 10
        bar_x = width - 200
        bar_y = 40
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar
        if buffer_size > 0:
            progress_width = int((buffer_size / SEQUENCE_LENGTH) * bar_width)
            bar_color = (0, 255, 0) if buffer_size >= SEQUENCE_LENGTH else (0, 255, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), bar_color, -1)
        
        # Mirror note
        cv2.putText(
            frame,
            'Mirror view for natural interaction',
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'c' to clear buffer",
            "Press 's' to save screenshot"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (width - 250, height - 80 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
    
    def run(self):
        """Start real-time inference."""
        print("Starting real-time ISL recognition...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Clear sequence buffer")
        print("  's' - Save screenshot")
        print()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        current_phrase = None
        confidence = 0.0
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame (don't flip for landmark extraction)
                frame_landmarks, results = self.process_frame(frame)
                
                # Add to sequence buffer
                self.sequence_buffer.append(frame_landmarks)
                frame_count += 1
                
                # Predict when buffer is full
                if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                    current_phrase, confidence = self.predict_sequence()
                
                # Flip frame for display (mirror effect for user)
                display_frame = cv2.flip(frame, 1)
                
                # Draw landmarks (on flipped frame for display)
                if results:
                    self.draw_landmarks(display_frame, results)
                
                # Draw UI
                self.draw_ui(
                    display_frame, 
                    current_phrase, 
                    confidence, 
                    len(self.sequence_buffer)
                )
                
                # FPS calculation
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    current_fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                # Draw FPS
                cv2.putText(
                    display_frame,
                    f'FPS: {current_fps:.1f}',
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Show frame
                cv2.imshow('ISL Transformer Recognition', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.sequence_buffer.clear()
                    current_phrase = None
                    confidence = 0.0
                    print("Sequence buffer cleared")
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshot_{timestamp}.png"
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Real-time recognition stopped")


def main():
    """Main inference function."""
    try:
        inference = HolisticInference()
        inference.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()