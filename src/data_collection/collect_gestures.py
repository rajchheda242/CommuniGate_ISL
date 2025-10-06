"""
Data collection script for ISL gesture recognition.
Captures hand landmarks for 4 fixed phrases and saves them to CSV.
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from datetime import datetime


# Fixed phrases to collect
PHRASES = [
    "Hi, my name is Madiha Siddiqui.",
    "I am a student.",
    "I enjoy running as a hobby.",
    "How are you doing today?"
]

SAMPLES_PER_PHRASE = 50
DATA_DIR = "data/processed"


class GestureCollector:
    """Collects hand landmark data for gesture recognition."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.data = []
        
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from Mediapipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def collect_phrase_data(self, phrase_idx, phrase_text):
        """Collect samples for a specific phrase."""
        print(f"\n{'='*60}")
        print(f"Collecting data for Phrase {phrase_idx + 1}:")
        print(f'"{phrase_text}"')
        print(f"{'='*60}")
        print(f"Target: {SAMPLES_PER_PHRASE} samples")
        print("\nInstructions:")
        print("- Position your hand(s) for the gesture")
        print("- Press SPACE to capture a sample")
        print("- Press 'q' to finish early")
        print("- Press 's' to skip this phrase\n")
        
        cap = cv2.VideoCapture(0)
        samples_collected = 0
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            while samples_collected < SAMPLES_PER_PHRASE:
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                
                # Display information
                cv2.putText(
                    frame, 
                    f'Phrase {phrase_idx + 1}: "{phrase_text}"', 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                cv2.putText(
                    frame, 
                    f'Samples: {samples_collected}/{SAMPLES_PER_PHRASE}', 
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                cv2.putText(
                    frame, 
                    'SPACE: Capture | Q: Quit | S: Skip', 
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 0), 
                    2
                )
                
                cv2.imshow('Gesture Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Capture sample on SPACE
                if key == ord(' '):
                    if results.multi_hand_landmarks:
                        # Extract landmarks from all detected hands
                        all_landmarks = []
                        for hand_landmarks in results.multi_hand_landmarks:
                            all_landmarks.extend(self.extract_landmarks(hand_landmarks))
                        
                        # Pad or truncate to fixed size (2 hands * 21 landmarks * 3 coords = 126)
                        while len(all_landmarks) < 126:
                            all_landmarks.append(0.0)
                        all_landmarks = all_landmarks[:126]
                        
                        # Store data
                        self.data.append({
                            'phrase_id': phrase_idx,
                            'phrase_text': phrase_text,
                            'landmarks': all_landmarks,
                            'timestamp': datetime.now().isoformat()
                        })
                        samples_collected += 1
                        print(f"Sample {samples_collected} captured!")
                    else:
                        print("No hand detected! Please show your hand(s) to the camera.")
                
                # Skip phrase
                elif key == ord('s'):
                    print(f"Skipping phrase {phrase_idx + 1}")
                    break
                
                # Quit
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {samples_collected} samples for phrase {phrase_idx + 1}")
        return True
    
    def save_data(self):
        """Save collected data to CSV file."""
        if not self.data:
            print("No data collected!")
            return
        
        # Prepare dataframe
        rows = []
        for item in self.data:
            row = {
                'phrase_id': item['phrase_id'],
                'phrase_text': item['phrase_text'],
                'timestamp': item['timestamp']
            }
            # Add landmark columns
            for i, value in enumerate(item['landmarks']):
                row[f'landmark_{i}'] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(DATA_DIR, f"gesture_data_{timestamp}.csv")
        df.to_csv(filename, index=False)
        
        print(f"\n{'='*60}")
        print(f"Data saved to: {filename}")
        print(f"Total samples: {len(self.data)}")
        print(f"{'='*60}")
    
    def run(self):
        """Main collection loop."""
        print("ISL Gesture Data Collection")
        print("="*60)
        
        for idx, phrase in enumerate(PHRASES):
            if not self.collect_phrase_data(idx, phrase):
                break
        
        self.save_data()


if __name__ == "__main__":
    collector = GestureCollector()
    collector.run()
