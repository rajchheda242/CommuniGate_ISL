"""
Sequence-based data collection for ISL gesture recognition.
Captures temporal sequences of hand landmarks for multi-word phrases.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import time


# Fixed phrases to collect
PHRASES = [
    "Hi, my name is Madiha Siddiqui.",
    "I am a student.",
    "I enjoy running as a hobby.",
    "How are you doing today?"
]

SEQUENCES_PER_PHRASE = 40  # Number of sequence samples per phrase
SEQUENCE_LENGTH = 60  # Number of frames per sequence (~2 seconds at 30fps)
COUNTDOWN_SECONDS = 3

DATA_DIR = "data/sequences"


class SequenceCollector:
    """Collects temporal sequences of hand landmarks for gesture recognition."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create data directories
        os.makedirs(DATA_DIR, exist_ok=True)
        for i in range(len(PHRASES)):
            os.makedirs(os.path.join(DATA_DIR, f"phrase_{i}"), exist_ok=True)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from Mediapipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def show_countdown(self, cap, phrase_text, sequence_num):
        """Display countdown before recording."""
        for i in range(COUNTDOWN_SECONDS, 0, -1):
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Show countdown
            cv2.putText(
                frame, 
                f'GET READY: {i}', 
                (frame.shape[1]//2 - 100, frame.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (0, 255, 255), 
                3
            )
            cv2.putText(
                frame, 
                f'Phrase: "{phrase_text}"', 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            cv2.putText(
                frame, 
                f'Sequence: {sequence_num + 1}/{SEQUENCES_PER_PHRASE}', 
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            cv2.imshow('Sequence Collection', frame)
            cv2.waitKey(1000)  # Wait 1 second
    
    def record_sequence(self, cap, hands, phrase_text, sequence_num):
        """Record a sequence of frames for one gesture performance."""
        sequence_frames = []
        frame_count = 0
        
        # Show "GO!" signal
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame, 
                'GO! START SIGNING!', 
                (frame.shape[1]//2 - 200, frame.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (0, 255, 0), 
                3
            )
            cv2.imshow('Sequence Collection', frame)
            cv2.waitKey(500)
        
        # Record sequence
        while frame_count < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Extract landmarks
            frame_landmarks = []
            if results.multi_hand_landmarks:
                # Draw landmarks for visual feedback
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    frame_landmarks.extend(self.extract_landmarks(hand_landmarks))
            
            # Pad or truncate to fixed size (2 hands * 21 landmarks * 3 coords = 126)
            while len(frame_landmarks) < 126:
                frame_landmarks.append(0.0)
            frame_landmarks = frame_landmarks[:126]
            
            sequence_frames.append(frame_landmarks)
            
            # Display recording status
            progress = int((frame_count / SEQUENCE_LENGTH) * 100)
            cv2.putText(
                frame, 
                'RECORDING...', 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            cv2.rectangle(frame, (10, 50), (10 + progress * 6, 80), (0, 255, 0), -1)
            cv2.putText(
                frame, 
                f'{progress}%', 
                (10 + progress * 6 + 10, 73),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            cv2.imshow('Sequence Collection', frame)
            cv2.waitKey(1)
            
            frame_count += 1
        
        return np.array(sequence_frames)
    
    def collect_phrase_sequences(self, phrase_idx, phrase_text):
        """Collect sequences for a specific phrase."""
        print(f"\n{'='*70}")
        print(f"Collecting sequences for Phrase {phrase_idx + 1}:")
        print(f'"{phrase_text}"')
        print(f"{'='*70}")
        print(f"Target: {SEQUENCES_PER_PHRASE} sequences")
        print(f"Each sequence: {SEQUENCE_LENGTH} frames (~2 seconds)")
        print("\nInstructions:")
        print("- Wait for countdown (3, 2, 1, GO!)")
        print("- Perform the ENTIRE phrase when you see 'GO!'")
        print("- Sign each word in sequence naturally")
        print("- Press 's' to skip this phrase")
        print("- Press 'q' to quit\n")
        
        input("Press ENTER when ready to start...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set 30 fps
        sequences_collected = 0
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            while sequences_collected < SEQUENCES_PER_PHRASE:
                # Show preview and wait for ready
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame, 
                    f'Sequence {sequences_collected + 1}/{SEQUENCES_PER_PHRASE}', 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 255, 255), 
                    2
                )
                cv2.putText(
                    frame, 
                    'Press SPACE when ready (or S to skip, Q to quit)', 
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 0), 
                    2
                )
                
                cv2.imshow('Sequence Collection', frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):
                    # Start recording
                    self.show_countdown(cap, phrase_text, sequences_collected)
                    sequence_data = self.record_sequence(cap, hands, phrase_text, sequences_collected)
                    
                    # Save sequence
                    filename = os.path.join(
                        DATA_DIR, 
                        f"phrase_{phrase_idx}", 
                        f"sequence_{sequences_collected:03d}.npy"
                    )
                    np.save(filename, sequence_data)
                    
                    sequences_collected += 1
                    print(f"✓ Sequence {sequences_collected} saved! ({sequence_data.shape})")
                    
                    # Brief pause
                    time.sleep(1)
                    
                elif key == ord('s'):
                    print(f"Skipping phrase {phrase_idx + 1}")
                    break
                    
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Collected {sequences_collected} sequences for phrase {phrase_idx + 1}")
        return True
    
    def run(self):
        """Main collection loop."""
        print("="*70)
        print("ISL TEMPORAL SEQUENCE DATA COLLECTION")
        print("="*70)
        print("\nThis will collect temporal sequences (not single frames)")
        print("Each sequence captures the full phrase being signed over time\n")
        
        for idx, phrase in enumerate(PHRASES):
            if not self.collect_phrase_sequences(idx, phrase):
                break
        
        print("\n" + "="*70)
        print("DATA COLLECTION COMPLETE!")
        print("="*70)
        print(f"\nSequences saved in: {DATA_DIR}/")
        print("\nNext steps:")
        print("1. Run: python src/training/train_sequence_model.py")
        print("2. Then: streamlit run src/ui/app_sequence.py")


if __name__ == "__main__":
    collector = SequenceCollector()
    collector.run()
