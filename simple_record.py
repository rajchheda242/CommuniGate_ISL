#!/usr/bin/env python3
"""
Simple recording script that starts immediately (no interactive prompts)
Just for phrases 1-4 (skip phrase 0)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

PHRASES = {
    1: "How are you",
    2: "I am from Delhi", 
    3: "I like coffee",
    4: "What do you like"
}

SEQUENCES_NEEDED = {
    1: 16,
    2: 20,
    3: 18,
    4: 19
}

SEQUENCE_LENGTH = 90
DATA_DIR = "data/sequences"

class SimpleRecorder:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def record_phrase(self, phrase_idx, phrase_text, num_sequences):
        """Record sequences for one phrase"""
        
        print(f"\n{'='*70}")
        print(f"PHRASE {phrase_idx}: '{phrase_text}'")
        print(f"{'='*70}")
        print(f"Need to record: {num_sequences} sequences")
        print(f"\nGet ready... Camera starting in 3 seconds...")
        time.sleep(3)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        sequences_collected = 0
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            while sequences_collected < num_sequences:
                print(f"\n{'='*70}")
                print(f"Sequence {sequences_collected + 1}/{num_sequences}")
                print(f"{'='*70}")
                
                # Countdown
                for i in range(3, 0, -1):
                    print(f"Get ready... {i}")
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, f"Ready in {i}...", (50, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Phrase: {phrase_text}", (50, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow('ISL Recording', frame)
                        cv2.waitKey(1000)
                
                print("🎬 GO! Perform the gesture now!")
                
                # Record sequence
                sequence = []
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    
                    # Extract landmarks
                    frame_data = np.zeros(126)  # 2 hands × 21 landmarks × 3 coords
                    
                    if results.multi_hand_landmarks:
                        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            if idx < 2:  # Only process first 2 hands
                                landmarks = self.extract_landmarks(hand_landmarks)
                                start_idx = idx * 63
                                frame_data[start_idx:start_idx + 63] = landmarks
                            
                            # Draw landmarks
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                    
                    sequence.append(frame_data)
                    
                    # Show progress
                    progress = int((frame_num / SEQUENCE_LENGTH) * 100)
                    cv2.putText(frame, f"Recording: {progress}%", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Phrase: {phrase_text}", (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Seq {sequences_collected + 1}/{num_sequences}", (50, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow('ISL Recording', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return sequences_collected
                
                # Save sequence
                sequence_array = np.array(sequence)
                
                # Count zero frames
                zero_frames = np.all(sequence_array == 0, axis=1).sum()
                zero_pct = zero_frames / SEQUENCE_LENGTH * 100
                
                # Find next available filename
                phrase_dir = os.path.join(DATA_DIR, f"phrase_{phrase_idx}")
                existing_files = os.listdir(phrase_dir)
                take_nums = []
                for f in existing_files:
                    if f.startswith('take') or f.startswith('Take'):
                        try:
                            num = int(f.split('_')[0].replace('take', '').replace('Take', '').strip())
                            take_nums.append(num)
                        except:
                            pass
                
                next_take = max(take_nums) + 1 if take_nums else 1
                filename = os.path.join(phrase_dir, f"take{next_take}_seq.npy")
                
                np.save(filename, sequence_array)
                
                print(f"✅ Saved: {os.path.basename(filename)}")
                print(f"   Quality: {zero_frames} zero frames ({zero_pct:.1f}%)")
                
                if zero_pct > 20:
                    print(f"   ⚠️  High zero frames - make sure hands stay visible!")
                elif zero_pct < 10:
                    print(f"   🎉 Excellent quality!")
                
                sequences_collected += 1
                
                # Short pause before next sequence
                time.sleep(1)
        
        cap.release()
        cv2.destroyAllWindows()
        return sequences_collected
    
    def run(self):
        """Record all needed sequences"""
        print("\n" + "="*70)
        print("ISL RECORDING SESSION")
        print("="*70)
        print("\nYou will record:")
        for phrase_idx, count in SEQUENCES_NEEDED.items():
            print(f"  Phrase {phrase_idx}: '{PHRASES[phrase_idx]}' - {count} sequences")
        
        print(f"\nTotal: {sum(SEQUENCES_NEEDED.values())} sequences")
        print(f"Time estimate: ~{sum(SEQUENCES_NEEDED.values())} minutes")
        
        print("\n" + "="*70)
        print("TIPS:")
        print("="*70)
        print("- Keep hands in frame throughout gesture")
        print("- Perform at natural speed")
        print("- Wait for countdown (3, 2, 1, GO!)")
        print("- Press 'q' during recording to quit")
        print("="*70)
        
        input("\nPress ENTER to start recording...")
        
        total_recorded = 0
        
        for phrase_idx in sorted(PHRASES.keys()):
            phrase_text = PHRASES[phrase_idx]
            num_needed = SEQUENCES_NEEDED[phrase_idx]
            
            recorded = self.record_phrase(phrase_idx, phrase_text, num_needed)
            total_recorded += recorded
            
            if recorded < num_needed:
                print(f"\n⚠️  Only recorded {recorded}/{num_needed} sequences")
                break
        
        print("\n" + "="*70)
        print("RECORDING COMPLETE!")
        print("="*70)
        print(f"Total sequences recorded: {total_recorded}/{sum(SEQUENCES_NEEDED.values())}")
        print("\nNext step: Train the model")
        print("  python enhanced_train.py")
        print("="*70)

if __name__ == "__main__":
    recorder = SimpleRecorder()
    recorder.run()
