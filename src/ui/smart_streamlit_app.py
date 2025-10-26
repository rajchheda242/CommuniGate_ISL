"""
Streamlit UI for ISL Recognition - User Controlled Recording.
Better UX: User decides when to start/stop recording.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
from PIL import Image
import threading
from tensorflow.keras.models import load_model

# Try to import text-to-speech (optional)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


MODEL_DIR = "models/saved"
MIN_FRAMES = 60
MAX_FRAMES = 150
TARGET_FRAMES = 90


class SmartStreamlitApp:
    """Streamlit application with user-controlled recording."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.model = None
        self.scaler = None
        self.phrase_mapping = None
        self.tts_engine = None
        
        # Recording state
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'recorded_frames' not in st.session_state:
            st.session_state.recorded_frames = []
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
        if 'last_confidence' not in st.session_state:
            st.session_state.last_confidence = 0.0
        
        self.load_model()
        if TTS_AVAILABLE:
            self.init_tts()
    
    def load_model(self):
        """Load trained LSTM model, scaler, and mapping."""
        model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
        scaler_path = os.path.join(MODEL_DIR, "sequence_scaler.joblib")
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found: {model_path}")
            return
        
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            self.phrase_mapping = {int(k): v for k, v in mapping_data.items()}
    
    def init_tts(self):
        """Initialize text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        except:
            self.tts_engine = None
    
    def speak(self, text):
        """Speak the given text in a separate thread."""
        if self.tts_engine and text:
            def _speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except:
                    pass
            threading.Thread(target=_speak, daemon=True).start()
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def process_frame_landmarks(self, results):
        """Extract landmarks from current frame."""
        if not results.multi_hand_landmarks:
            return None
        
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            all_landmarks.extend(self.extract_landmarks(hand_landmarks))
        
        while len(all_landmarks) < 126:
            all_landmarks.append(0.0)
        
        return all_landmarks[:126]
    
    def normalize_sequence(self, sequence):
        """Normalize sequence to TARGET_FRAMES using interpolation."""
        current_length = len(sequence)
        
        if current_length == TARGET_FRAMES:
            return sequence
        
        sequence = np.array(sequence)
        old_indices = np.linspace(0, current_length - 1, current_length)
        new_indices = np.linspace(0, current_length - 1, TARGET_FRAMES)
        
        normalized = np.zeros((TARGET_FRAMES, 126))
        for i in range(126):
            normalized[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
        
        return normalized.tolist()
    
    def predict_sequence(self, sequence):
        """Predict phrase from recorded sequence."""
        if len(sequence) < MIN_FRAMES:
            return f"Too short ({len(sequence)} frames)", 0.0
        
        # Normalize to TARGET_FRAMES
        normalized_seq = self.normalize_sequence(sequence)
        normalized_seq = np.array(normalized_seq)
        
        # Reshape for scaling
        seq_flat = normalized_seq.reshape(-1, 126)
        seq_scaled = self.scaler.transform(seq_flat)
        seq_scaled = seq_scaled.reshape(1, TARGET_FRAMES, 126)
        
        # Predict
        predictions = self.model.predict(seq_scaled, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        phrase = self.phrase_mapping.get(int(predicted_class), "Unknown")
        
        return phrase, confidence
    
    def process_frame(self, frame, hands):
        """Process a single frame."""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks
        hands_detected = False
        if results.multi_hand_landmarks:
            hands_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Record if recording is active and hands detected
        if st.session_state.is_recording and hands_detected:
            frame_landmarks = self.process_frame_landmarks(results)
            if frame_landmarks is not None:
                st.session_state.recorded_frames.append(frame_landmarks)
                # Limit max frames
                if len(st.session_state.recorded_frames) > MAX_FRAMES:
                    st.session_state.recorded_frames.pop(0)
        
        return frame, hands_detected


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="CommuniGate ISL",
        page_icon="🤟",
        layout="wide"
    )
    
    st.title("🤟 CommuniGate ISL - Smart Recognition")
    st.markdown("### User-Controlled Recording")
    
    # Initialize app
    app = SmartStreamlitApp()
    
    if app.model is None:
        st.error("❌ Model not loaded. Please train the model first.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    if TTS_AVAILABLE:
        enable_tts = st.sidebar.checkbox("🔊 Text-to-Speech", value=False)
    else:
        st.sidebar.info("💡 Install pyttsx3 for TTS")
        enable_tts = False
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.5, 0.05
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Recognized Phrases")
    for idx, phrase in app.phrase_mapping.items():
        st.sidebar.markdown(f"{idx + 1}. {phrase}")
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **How to use:**
    1. Click "Start Recording" 
    2. Perform your gesture
    3. Click "Stop Recording"
    4. See the prediction!
    
    **Tips:**
    - Take your time!
    - Need {MIN_FRAMES}+ frames
    - Clear hands visibility
    """)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🎬 Recording Controls")
        
        # Recording status
        if st.session_state.is_recording:
            st.error("🔴 **RECORDING IN PROGRESS**")
            frame_count = len(st.session_state.recorded_frames)
            st.metric("Frames Captured", frame_count)
            
            if frame_count < MIN_FRAMES:
                st.warning(f"Need {MIN_FRAMES - frame_count} more frames")
            else:
                st.success(f"✓ Ready to process!")
        else:
            st.info("⚪ Ready to record")
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if not st.session_state.is_recording:
                if st.button("▶️ Start", use_container_width=True, type="primary"):
                    st.session_state.is_recording = True
                    st.session_state.recorded_frames = []
                    st.session_state.last_prediction = None
                    st.rerun()
            else:
                if st.button("⏹️ Stop", use_container_width=True, type="secondary"):
                    st.session_state.is_recording = False
                    
                    # Process the recording
                    if len(st.session_state.recorded_frames) >= MIN_FRAMES:
                        with st.spinner("🤖 Processing..."):
                            phrase, conf = app.predict_sequence(
                                st.session_state.recorded_frames
                            )
                            st.session_state.last_prediction = phrase
                            st.session_state.last_confidence = conf
                            
                            if enable_tts and conf > confidence_threshold:
                                app.speak(phrase)
                    else:
                        st.session_state.last_prediction = "Too short"
                        st.session_state.last_confidence = 0.0
                    
                    st.rerun()
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.recorded_frames = []
                st.session_state.last_prediction = None
                st.session_state.last_confidence = 0.0
                st.rerun()
        
        with col_btn3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.is_recording = False
                st.session_state.recorded_frames = []
                st.session_state.last_prediction = None
                st.session_state.last_confidence = 0.0
                st.rerun()
        
        # Prediction results
        st.markdown("---")
        st.subheader("🎯 Recognition Result")
        
        if st.session_state.last_prediction:
            phrase = st.session_state.last_prediction
            conf = st.session_state.last_confidence
            
            if conf > confidence_threshold:
                st.success(f"**{phrase}**")
                st.metric("Confidence", f"{conf:.1%}")
            elif conf > 0:
                st.warning(f"**{phrase}**")
                st.metric("Confidence", f"{conf:.1%}", 
                         delta="Low confidence")
            else:
                st.error(phrase)
        else:
            st.info("No prediction yet")
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        # Camera placeholder
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Start camera toggle
        run_camera = st.checkbox("📷 Enable Camera", value=False)
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot access camera")
                return
            
            with app.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                
                # Process frames
                while run_camera:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("❌ Failed to read from camera")
                        break
                    
                    # Process frame
                    processed_frame, hands_detected = app.process_frame(frame, hands)
                    
                    # Display video
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                    
                    # Status
                    if hands_detected:
                        status_placeholder.success("✓ Hands detected")
                    else:
                        status_placeholder.warning("⚠️ No hands detected")
                    
                    # Check if we should continue
                    if not st.session_state.get('run_camera', True):
                        break
            
            cap.release()
        else:
            st.info("👆 Enable camera to start")


if __name__ == "__main__":
    main()
