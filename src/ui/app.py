"""
Streamlit UI for ISL Recognition - Smart User-Controlled Recording.
Provides a web-based interface with manual recording control for better UX.
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
import time
from tensorflow.keras.models import load_model
from datetime import datetime

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
DEBUG = True  # Set to False to silence debug logs


def debug_log(msg: str):
    if DEBUG:
        try:
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[UI DEBUG {ts}] {msg}")
        except Exception:
            pass


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
        
        # Recording state - stored in session_state for persistence
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'recorded_frames' not in st.session_state:
            st.session_state.recorded_frames = []
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
        if 'last_confidence' not in st.session_state:
            st.session_state.last_confidence = 0.0
        # Camera state to avoid double-start UX
        if 'camera_toggle' not in st.session_state:
            st.session_state.camera_toggle = False
        # Frame counter (derived from recorded frames but cached for UI)
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
        
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
            st.info("Please train the model first: `python src/training/train_sequence_model.py`")
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
                # Update frame counter in session state for consistent UI
                st.session_state.frame_count = len(st.session_state.recorded_frames)
        
        return frame, hands_detected


def main():
    """Main Streamlit application."""
    # Note: app expiry removed to allow continued usage

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
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.5, 0.05,
        help="Minimum confidence to show prediction"
    )
    require_confidence = st.sidebar.toggle(
        "Require confidence (ask to redo if low)",
        value=True,
        help="If enabled, the app will NOT show a phrase when confidence is below the threshold and will ask you to redo your gesture."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Recognized Phrases")
    for idx, phrase in app.phrase_mapping.items():
        st.sidebar.markdown(f"{idx + 1}. {phrase}")
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **How to use:**
    1. Click "▶️ Start Recording" 
    2. Perform your gesture
    3. Click "⏹️ Stop Recording"
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
        
        # Dynamic placeholders for right panel so we can update from the camera loop
        status_banner_ph = st.empty()
        frame_counter_ph = st.empty()
        guidance_ph = st.empty()

        # Initial draw based on current state
        if st.session_state.is_recording:
            status_banner_ph.error("🔴 **RECORDING IN PROGRESS**")
            frame_counter_ph.metric("Frames Captured", st.session_state.frame_count)
            if st.session_state.frame_count < MIN_FRAMES:
                guidance_ph.warning(f"Need {MIN_FRAMES - st.session_state.frame_count} more frames")
            else:
                guidance_ph.success("✓ Ready to process!")
        else:
            status_banner_ph.info("⚪ Ready to record")
            frame_counter_ph.empty()
            guidance_ph.empty()
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if not st.session_state.is_recording:
                if st.button("▶️ Start", use_container_width=True, type="primary", key="start_btn"):
                    # Start recording and ensure camera is enabled to avoid double-click confusion
                    st.session_state.is_recording = True
                    st.session_state.camera_toggle = True
                    st.session_state.recorded_frames = []
                    st.session_state.frame_count = 0
                    st.session_state.last_prediction = None
                    st.session_state.last_confidence = 0.0
                    debug_log("Start pressed -> is_recording=True, camera_toggle=True, frame_count reset")
                    # Immediately rerun to reflect Start state and begin camera loop without needing a second click
                    st.rerun()
            else:
                if st.button("⏹️ Stop", use_container_width=True, type="secondary", key="stop_btn"):
                    st.session_state.is_recording = False
                    
                    # Process the recording
                    if len(st.session_state.recorded_frames) >= MIN_FRAMES:
                        with st.spinner("🤖 Processing..."):
                            phrase, conf = app.predict_sequence(
                                st.session_state.recorded_frames
                            )
                            st.session_state.last_prediction = phrase
                            st.session_state.last_confidence = conf
                            debug_log(f"Stop pressed -> processed frames={len(st.session_state.recorded_frames)}, phrase={phrase}, conf={conf:.3f}")
                    else:
                        st.session_state.last_prediction = "Too short"
                        st.session_state.last_confidence = 0.0
                        debug_log(f"Stop pressed -> too short, frames={len(st.session_state.recorded_frames)}")

                    # After stopping, exit camera loop to refresh the whole UI
                    # so the Start button is shown immediately for the next take.
                    st.session_state.camera_toggle = False
                    debug_log("Stop pressed -> camera_toggle=False, triggering rerun")
                    st.rerun()
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
                st.session_state.recorded_frames = []
                st.session_state.frame_count = 0
                st.session_state.last_prediction = None
                st.session_state.last_confidence = 0.0
        
        with col_btn3:
            if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
                st.session_state.is_recording = False
                st.session_state.recorded_frames = []
                st.session_state.frame_count = 0
                st.session_state.last_prediction = None
                st.session_state.last_confidence = 0.0
        
        # Prediction results
        st.markdown("---")
        st.subheader("🎯 Recognition Result")
        
        if st.session_state.last_prediction:
            phrase = st.session_state.last_prediction
            conf = st.session_state.last_confidence
            
            # If confidence requirement is enabled and confidence is low, ask the user to redo
            if require_confidence and conf > 0 and conf < confidence_threshold:
                st.warning("Confidence is too low — please redo your gesture.")
                st.metric("Confidence", f"{conf:.1%}", delta="Low")
                # Offer a quick retry action
                if st.button("🔁 Try again", key="retry_btn", use_container_width=True):
                    st.session_state.recorded_frames = []
                    st.session_state.frame_count = 0
                    st.session_state.last_prediction = None
                    st.session_state.last_confidence = 0.0
                    st.session_state.is_recording = False
                    # Also exit camera loop to refresh full UI and show Start again
                    st.session_state.camera_toggle = False
                    debug_log("Try again -> cleared state, camera_toggle=False, triggering rerun")
                    st.rerun()
            else:
                if conf > confidence_threshold:
                    st.success(f"**{phrase}**")
                    st.metric("Confidence", f"{conf:.1%}")
                    
                    # Speaker button to hear the prediction
                    if TTS_AVAILABLE:
                        if st.button("🔊 Speak", key="speak_btn", use_container_width=True):
                            app.speak(phrase)
                    else:
                        st.caption("💡 Install pyttsx3 for text-to-speech")
                        
                elif conf > 0:
                    # Confidence requirement disabled: still show best guess but mark as low
                    st.warning(f"**{phrase}**")
                    st.metric("Confidence", f"{conf:.1%}", 
                             delta="Low confidence")
                    
                    if TTS_AVAILABLE:
                        if st.button("🔊 Speak", key="speak_low_btn", use_container_width=True):
                            app.speak(phrase)
                else:
                    st.error(phrase)
        else:
            st.info("No prediction yet")
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        # Camera placeholder
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Start camera toggle (auto-enabled when you press Start)
        run_camera = st.checkbox("📷 Enable Camera", value=st.session_state.get("camera_toggle", False), key="camera_toggle")
        
        if run_camera:
            debug_log("Camera loop starting (camera_toggle=True)")
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
                
                # Process frames continuously
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
                    
                    # Update right panel (status + counter) in real-time
                    if st.session_state.is_recording:
                        status_banner_ph.error("🔴 **RECORDING IN PROGRESS**")
                        frame_counter_ph.metric("Frames Captured", st.session_state.frame_count)
                        if st.session_state.frame_count < MIN_FRAMES:
                            guidance_ph.warning(f"Need {MIN_FRAMES - st.session_state.frame_count} more frames")
                        else:
                            guidance_ph.success("✓ Ready to process!")
                    else:
                        status_banner_ph.info("⚪ Ready to record")
                        frame_counter_ph.empty()
                        guidance_ph.empty()
                    
                    # Status message
                    if hands_detected:
                        status_placeholder.success("✓ Hands detected")
                    else:
                        status_placeholder.warning("⚠️ No hands detected")
                    
                    # Check if camera should continue (check session state)
                    if not st.session_state.get('camera_toggle', True):
                        break
            
            cap.release()
            debug_log("Camera loop stopped (camera released)")
        else:
            st.info("👆 Enable camera to start")


if __name__ == "__main__":
    main()
