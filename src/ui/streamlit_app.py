"""
Streamlit UI for ISL Sequence Recognition - User Controlled Version.
Provides a web-based interface with manual recording control.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
from collections import deque
from PIL import Image
import threading
import time
from tensorflow.keras.models import load_model

# Try to import text-to-speech (optional)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


MODEL_DIR = "models/saved"
MIN_FRAMES = 60   # Minimum frames needed
MAX_FRAMES = 150  # Maximum frames to keep
TARGET_FRAMES = 90  # Model expects 90 frames


class StreamlitSequenceApp:
    """Streamlit application for sequence-based gesture recognition."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.model = None
        self.scaler = None
        self.phrase_mapping = None
        self.tts_engine = None
        
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        self.load_model()
        if TTS_AVAILABLE:
            self.init_tts()
    
    def load_model(self):
        """Load trained LSTM model, scaler, and mapping."""
        model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
        scaler_path = os.path.join(MODEL_DIR, "sequence_scaler.joblib")
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
        
        if not os.path.exists(model_path):
            st.error(f"Model not found: {model_path}")
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
            st.warning("Text-to-speech initialization failed")
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
            return [0.0] * 126
        
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            all_landmarks.extend(self.extract_landmarks(hand_landmarks))
        
        while len(all_landmarks) < 126:
            all_landmarks.append(0.0)
        
        return all_landmarks[:126]
    
    def predict_sequence(self):
        """Predict phrase from buffered sequence."""
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0.0
        
        sequence = np.array(list(self.sequence_buffer))
        sequence_flat = sequence.reshape(-1, 126)
        sequence_scaled = self.scaler.transform(sequence_flat)
        sequence_scaled = sequence_scaled.reshape(1, SEQUENCE_LENGTH, 126)
        
        predictions = self.model.predict(sequence_scaled, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        phrase = self.phrase_mapping.get(int(predicted_class), "Unknown")
        
        return phrase, confidence
    
    def process_frame(self, frame, hands, enable_tts, confidence_threshold):
        """Process a single frame and return annotated image with prediction."""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Add frame to sequence buffer
        frame_landmarks = self.process_frame_landmarks(results)
        self.sequence_buffer.append(frame_landmarks)
        
        # Predict when buffer is full
        current_phrase = "Collecting frames..."
        confidence = 0.0
        
        if len(self.sequence_buffer) == SEQUENCE_LENGTH:
            current_phrase, confidence = self.predict_sequence()
            if current_phrase and confidence > confidence_threshold and enable_tts:
                self.speak(current_phrase)
        
        return frame, current_phrase, confidence, len(self.sequence_buffer)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="CommuniGate ISL",
        page_icon="🤟",
        layout="wide"
    )
    
    st.title("🤟 CommuniGate ISL - Indian Sign Language Recognition")
    st.markdown("### Real-time Sequence-based ISL Phrase Recognition")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    if TTS_AVAILABLE:
        enable_tts = st.sidebar.checkbox("🔊 Enable Text-to-Speech", value=False)
    else:
        st.sidebar.info("💡 Install pyttsx3 for text-to-speech")
        enable_tts = False
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.5, 0.05,
        help="Minimum confidence to show prediction"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Recognized Phrases")
    
    # Initialize app
    app = StreamlitSequenceApp()
    
    if app.model is None:
        st.error("❌ Model not loaded. Please train the model first.")
        st.info("Run: `python src/training/train_sequence_model.py`")
        return
    
    # Display phrases
    for idx, phrase in app.phrase_mapping.items():
        st.sidebar.markdown(f"{idx + 1}. {phrase}")
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Sequence Length: {SEQUENCE_LENGTH} frames")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("🎯 Recognition Output")
        buffer_placeholder = st.empty()
        phrase_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        if st.button("🔄 Clear Buffer"):
            app.sequence_buffer.clear()
            st.success("Buffer cleared!")
    
    # Camera controls
    run = st.checkbox("▶️ Start Camera", value=False)
    
    if run:
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
            
            while run:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                # Process frame
                processed_frame, phrase, confidence, buffer_size = app.process_frame(
                    frame, hands, enable_tts, confidence_threshold
                )
                
                # Display video
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Display predictions
                buffer_placeholder.metric(
                    "Frames Collected", 
                    f"{buffer_size}/{SEQUENCE_LENGTH}"
                )
                
                if phrase and confidence > 0:
                    phrase_placeholder.success(f"**{phrase}**")
                    
                    if confidence >= confidence_threshold:
                        confidence_placeholder.metric(
                            "Confidence", 
                            f"{confidence:.1%}",
                            delta="High" if confidence > 0.8 else "Medium"
                        )
                    else:
                        confidence_placeholder.warning(
                            f"Low confidence: {confidence:.1%}"
                        )
                else:
                    phrase_placeholder.info(phrase if phrase else "Collecting frames...")
        
        cap.release()
    else:
        st.info("👆 Check the box above to start the camera")


if __name__ == "__main__":
    main()
