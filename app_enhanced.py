"""
Enhanced Streamlit UI for ISL Recognition
Features:
- Manual Start/Stop recording with visual feedback
- Automatic blank frame removal
- Uses enhanced model with 100% accuracy
- Clean prediction after recording stops
- No constant prediction noise
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import json
import os
from PIL import Image
import time
from tensorflow.keras.models import load_model
from datetime import datetime
import mediapipe as mp

# Try to import text-to-speech (optional)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


MODEL_DIR = "models/saved"
SEQUENCE_LENGTH = 90  # Model expects 90 frames
FEATURES_PER_FRAME = 126  # 2 hands × 21 landmarks × 3 coords
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from Mediapipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def process_frame(self, frame):
        """
        Process a frame and extract hand landmarks
        Returns: (landmarks_array, annotated_frame, hands_detected)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Initialize landmarks array (126 features: 2 hands × 21 landmarks × 3 coords)
        frame_landmarks = np.zeros(FEATURES_PER_FRAME)
        hands_detected = False
        
        if results.multi_hand_landmarks:
            hands_detected = True
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx < 2:  # Only process first 2 hands
                    landmarks = self.extract_landmarks(hand_landmarks)
                    start_idx = idx * 63
                    frame_landmarks[start_idx:start_idx + 63] = landmarks
                
                # Draw landmarks for visual feedback
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame_landmarks, frame, hands_detected


class ISLRecognitionApp:
    """Enhanced Streamlit application for ISL recognition"""
    
    def __init__(self):
        self.extractor = HandLandmarkExtractor()
        self.model = None
        self.scaler = None
        self.phrase_mapping = None
        self.tts_engine = None
        
        # Initialize session state
        self.init_session_state()
        
        # Load model
        self.load_model_and_scaler()
        
        # Initialize TTS
        if TTS_AVAILABLE:
            self.init_tts()
    
    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'recorded_sequence' not in st.session_state:
            st.session_state.recorded_sequence = []
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
        if 'last_confidence' not in st.session_state:
            st.session_state.last_confidence = 0.0
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
    
    def load_model_and_scaler(self):
        """Load the trained model, scaler, and phrase mapping"""
        model_path = os.path.join(MODEL_DIR, "lstm_model_enhanced.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
        
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please retrain the model: python src/training/train_sequence_model.py")
            return
        
        if not os.path.exists(scaler_path):
            st.error(f"❌ Scaler file not found: {scaler_path}")
            st.info("Please retrain the model: python src/training/train_sequence_model.py")
            return
        
        if not os.path.exists(mapping_path):
            st.error(f"❌ Phrase mapping not found: {mapping_path}")
            st.info("Please retrain the model: python src/training/train_sequence_model.py")
            return
        
        try:
            # Load model with compatibility settings
            self.model = load_model(model_path, compile=False, safe_mode=False)
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            st.success(f"✅ Model loaded successfully from {model_path}")
            
        except (ValueError, OSError) as e:
            error_msg = str(e)
            if "expected" in error_msg.lower() and "variables" in error_msg.lower():
                st.error("❌ Model compatibility error!")
                st.error(f"Details: {error_msg}")
                st.warning("The model was trained on a different system. Please retrain on THIS computer:")
                st.code("python src/training/train_sequence_model.py")
                return
            else:
                st.error(f"❌ Error loading model: {error_msg}")
                raise
        
        try:
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            st.success(f"✅ Scaler loaded successfully")
            
            # Load phrase mapping and invert it (file has phrase->id, we need id->phrase)
            with open(mapping_path, 'r') as f:
                phrase_to_id = json.load(f)
                # Invert the mapping: id -> phrase
                self.phrase_mapping = {v: k for k, v in phrase_to_id.items()}
            st.success(f"✅ Phrase mapping loaded: {len(self.phrase_mapping)} phrases")
                
        except Exception as e:
            st.error(f"❌ Error loading scaler/mapping: {e}")
            raise
    
    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        except:
            self.tts_engine = None
    
    def speak(self, text):
        """Speak the predicted phrase"""
        if self.tts_engine and TTS_AVAILABLE:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass
    
    def normalize_sequence(self, sequence):
        """
        Normalize sequence to SEQUENCE_LENGTH frames
        Uses interpolation for smooth normalization
        """
        current_length = len(sequence)
        
        if current_length == SEQUENCE_LENGTH:
            return np.array(sequence)
        
        # Create indices for interpolation
        old_indices = np.linspace(0, current_length - 1, current_length)
        new_indices = np.linspace(0, current_length - 1, SEQUENCE_LENGTH)
        
        # Interpolate each feature
        normalized_sequence = []
        sequence_array = np.array(sequence)
        
        for feature_idx in range(FEATURES_PER_FRAME):
            feature_values = sequence_array[:, feature_idx]
            interpolated = np.interp(new_indices, old_indices, feature_values)
            normalized_sequence.append(interpolated)
        
        # Transpose to get (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
        return np.array(normalized_sequence).T
    
    def remove_blank_frames(self, sequence):
        """
        Remove frames with no hand detection (all zeros)
        Returns cleaned sequence with only frames containing hand data
        """
        cleaned_sequence = []
        
        for frame in sequence:
            # Check if frame has any non-zero values (hands detected)
            if not np.all(frame == 0):
                cleaned_sequence.append(frame)
        
        return cleaned_sequence
    
    def predict_sequence(self, sequence):
        """
        Predict the phrase from a recorded sequence
        Returns: (predicted_phrase, confidence, all_confidences)
        """
        if self.model is None or len(sequence) == 0:
            return None, 0.0, None
        
        # Remove blank frames
        cleaned_sequence = self.remove_blank_frames(sequence)
        
        if len(cleaned_sequence) < 30:  # Need at least 30 frames with hands
            return None, 0.0, None
        
        # Normalize to SEQUENCE_LENGTH frames
        normalized_sequence = self.normalize_sequence(cleaned_sequence)
        
        # Scale frame-by-frame (same as training)
        sequence_flat = normalized_sequence.reshape(-1, FEATURES_PER_FRAME)
        sequence_scaled = self.scaler.transform(sequence_flat)
        sequence_scaled = sequence_scaled.reshape(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
        
        # Predict
        prediction = self.model.predict(sequence_scaled, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class] * 100
        
        predicted_phrase = self.phrase_mapping.get(predicted_class, "Unknown")
        
        return predicted_phrase, confidence, prediction
    
    def start_recording(self):
        """Start recording a new sequence"""
        st.session_state.is_recording = True
        st.session_state.recorded_sequence = []
        st.session_state.last_prediction = None
        st.session_state.last_confidence = 0.0
    
    def stop_recording(self):
        """Stop recording and predict"""
        st.session_state.is_recording = False
        
        if len(st.session_state.recorded_sequence) > 0:
            # Predict the recorded sequence
            phrase, confidence, all_confidences = self.predict_sequence(
                st.session_state.recorded_sequence
            )
            
            if phrase:
                st.session_state.last_prediction = phrase
                st.session_state.last_confidence = confidence
                
                # Add to history
                st.session_state.prediction_history.append({
                    'phrase': phrase,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'frames': len(st.session_state.recorded_sequence),
                    'cleaned_frames': len(self.remove_blank_frames(st.session_state.recorded_sequence))
                })
                
                # Speak the prediction
                if TTS_AVAILABLE:
                    self.speak(phrase)
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="ISL Recognition",
            page_icon="🤟",
            layout="wide"
        )
        
        st.title("🤟 Indian Sign Language Recognition")
        st.markdown("### Enhanced Model - Manual Recording Control")
        
        # Sidebar
        with st.sidebar:
            st.header("ℹ️ Instructions")
            st.markdown("""
            **How to use:**
            1. Click **Start Recording** 
            2. Perform the ISL phrase
            3. Click **Stop & Predict**
            4. View prediction results
            
            **Supported Phrases:**
            - Hi my name is Reet
            - How are you
            - I am from Delhi
            - I like coffee
            - What do you like
            
            **Tips:**
            - Keep hands visible in frame
            - Perform gesture at natural speed
            - Good lighting helps accuracy
            - Plain background is best
            """)
            
            st.markdown("---")
            
            # Settings
            st.header("⚙️ Settings")
            enable_tts = st.checkbox("Enable Text-to-Speech", value=TTS_AVAILABLE)
            show_debug = st.checkbox("Show Debug Info", value=False)
            
            st.markdown("---")
            
            # Model info
            st.header("📊 Model Info")
            if self.model:
                st.success("✅ Model Loaded")
                st.info(f"Sequence Length: {SEQUENCE_LENGTH} frames")
                st.info(f"Features: {FEATURES_PER_FRAME} per frame")
        
        # Main content - two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Camera Feed")
            
            # Camera placeholder
            camera_placeholder = st.empty()
            
            # Control buttons
            button_col1, button_col2, button_col3 = st.columns(3)
            
            with button_col1:
                if st.button("🎬 Start Recording", type="primary", disabled=st.session_state.is_recording):
                    self.start_recording()
                    st.rerun()
            
            with button_col2:
                if st.button("⏹️ Stop & Predict", type="secondary", disabled=not st.session_state.is_recording):
                    self.stop_recording()
                    st.rerun()
            
            with button_col3:
                if st.button("🔄 Clear History"):
                    st.session_state.prediction_history = []
                    st.session_state.last_prediction = None
                    st.rerun()
            
            # Recording status
            if st.session_state.is_recording:
                frames_recorded = len(st.session_state.recorded_sequence)
                cleaned_frames = len(self.remove_blank_frames(st.session_state.recorded_sequence))
                
                st.error(f"🔴 **RECORDING** - {frames_recorded} frames ({cleaned_frames} with hands detected)")
                st.progress(min(frames_recorded / 150, 1.0))  # Show progress up to 150 frames
            else:
                st.info("⚪ Ready - Click 'Start Recording' to begin")
        
        with col2:
            st.subheader("🎯 Prediction")
            
            # Current prediction
            if st.session_state.last_prediction:
                st.success(f"**Phrase:** {st.session_state.last_prediction}")
                st.metric("Confidence", f"{st.session_state.last_confidence:.1f}%")
                
                # Confidence indicator
                if st.session_state.last_confidence >= 90:
                    st.success("🎉 Excellent confidence!")
                elif st.session_state.last_confidence >= 70:
                    st.info("👍 Good confidence")
                else:
                    st.warning("⚠️ Low confidence - try again")
            else:
                st.info("No prediction yet")
            
            # Prediction history
            if st.session_state.prediction_history:
                st.markdown("---")
                st.subheader("📜 Recent Predictions")
                
                for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):
                    with st.expander(f"{pred['timestamp']} - {pred['phrase']}", expanded=(i==0)):
                        st.write(f"**Phrase:** {pred['phrase']}")
                        st.write(f"**Confidence:** {pred['confidence']:.1f}%")
                        st.write(f"**Total Frames:** {pred['frames']}")
                        st.write(f"**Valid Frames:** {pred['cleaned_frames']}")
        
        # Camera feed with optimized frame processing
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for performance
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Cap at 30 FPS
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process frame
                landmarks, annotated_frame, hands_detected = self.extractor.process_frame(frame)
                
                # Record if active
                if st.session_state.is_recording:
                    st.session_state.recorded_sequence.append(landmarks)
                    
                    # Add recording indicator
                    cv2.circle(annotated_frame, (30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "REC", (60, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show frame count
                    frames_recorded = len(st.session_state.recorded_sequence)
                    cv2.putText(annotated_frame, f"Frames: {frames_recorded}", (60, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show hands detected status
                if hands_detected:
                    cv2.putText(annotated_frame, "Hands: Detected", (10, annotated_frame.shape[0] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, "Hands: Not Detected", (10, annotated_frame.shape[0] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Convert to RGB for display
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display - only update every 2 frames to reduce lag
                frame_count += 1
                if frame_count % 2 == 0:
                    camera_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Shorter delay for smoother video
                time.sleep(0.01)  # ~100 FPS processing, but only display ~15 FPS
                
        finally:
            cap.release()


if __name__ == "__main__":
    app = ISLRecognitionApp()
    app.run()
