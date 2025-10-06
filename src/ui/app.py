"""
Streamlit UI for ISL Gesture Recognition.
Provides a web-based interface with live camera feed and predictions.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import glob
from PIL import Image
import pyttsx3
import threading


MODEL_DIR = "models/saved"


class StreamlitGestureApp:
    """Streamlit application for gesture recognition."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.model = None
        self.scaler = None
        self.phrase_mapping = None
        self.tts_engine = None
        
        self.load_model()
        self.init_tts()
    
    def load_model(self):
        """Load trained model and scaler."""
        model_files = glob.glob(os.path.join(MODEL_DIR, "*_model.joblib"))
        
        if not model_files:
            st.error(f"No model found in {MODEL_DIR}. Please train the model first.")
            return
        
        model_path = model_files[0]
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.joblib")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.phrase_mapping = joblib.load(mapping_path)
    
    def init_tts(self):
        """Initialize text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        except:
            st.warning("Text-to-speech not available on this system")
            self.tts_engine = None
    
    def speak(self, text):
        """Speak the given text in a separate thread."""
        if self.tts_engine and text != "No gesture detected":
            def _speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            threading.Thread(target=_speak, daemon=True).start()
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def predict(self, landmarks):
        """Predict phrase from landmarks."""
        while len(landmarks) < 126:
            landmarks.append(0.0)
        landmarks = landmarks[:126]
        
        X = np.array(landmarks).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 1.0
        
        phrase = self.phrase_mapping.get(int(prediction), "Unknown")
        
        return phrase, confidence
    
    def process_frame(self, frame, hands, enable_tts):
        """Process a single frame and return annotated image with prediction."""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_phrase = "No gesture detected"
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            all_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                all_landmarks.extend(self.extract_landmarks(hand_landmarks))
            
            current_phrase, confidence = self.predict(all_landmarks)
            
            if enable_tts and confidence > 0.7:
                self.speak(current_phrase)
        
        return frame, current_phrase, confidence


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="CommuniGate ISL",
        page_icon="👋",
        layout="wide"
    )
    
    st.title("🤟 CommuniGate ISL - Indian Sign Language Recognition")
    st.markdown("### Real-time ISL Phrase Recognition")
    
    # Sidebar
    st.sidebar.header("Settings")
    enable_tts = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Recognized Phrases")
    st.sidebar.markdown("""
    1. Hi, my name is Madiha Siddiqui.
    2. I am a student.
    3. I enjoy running as a hobby.
    4. How are you doing today?
    """)
    
    # Initialize app
    app = StreamlitGestureApp()
    
    if app.model is None:
        st.error("Model not loaded. Please train the model first using `train_model.py`")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("Recognition Output")
        phrase_placeholder = st.empty()
        confidence_placeholder = st.empty()
        status_placeholder = st.empty()
    
    # Camera controls
    run = st.checkbox("Start Camera", value=False)
    
    if run:
        cap = cv2.VideoCapture(0)
        
        with app.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            while run:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                # Process frame
                processed_frame, phrase, confidence = app.process_frame(
                    frame, hands, enable_tts
                )
                
                # Display video
                video_placeholder.image(
                    processed_frame,
                    channels="BGR",
                    use_container_width=True
                )
                
                # Display predictions
                if confidence >= confidence_threshold:
                    phrase_placeholder.success(f"**Detected Phrase:**\n\n*{phrase}*")
                    confidence_placeholder.info(f"**Confidence:** {confidence:.2%}")
                    status_placeholder.empty()
                else:
                    phrase_placeholder.warning("**No gesture detected**")
                    confidence_placeholder.empty()
                    status_placeholder.info("Show your gesture to the camera")
        
        cap.release()
    else:
        st.info("Click 'Start Camera' to begin recognition")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>CommuniGate ISL - Academic Project | Powered by Mediapipe & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
