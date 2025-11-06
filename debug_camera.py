#!/usr/bin/env python3
"""
Debug the live camera input to see what landmarks are being captured.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_camera_input():
    """Debug what the camera is actually capturing."""
    print("🔍 Debugging Live Camera Input")
    print("="*50)
    print("This will show you what landmarks the camera is detecting.")
    print("Press 'q' to quit, 's' to save current landmarks")
    print("")
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    sequence_buffer = deque(maxlen=60)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        # Extract landmarks
        landmarks = extract_holistic_landmarks(results)
        
        if landmarks is not None:
            sequence_buffer.append(landmarks)
            
            # Analyze the landmarks
            if frame_count % 30 == 0:  # Every 30 frames (1 second)
                analyze_landmarks(landmarks, len(sequence_buffer))
        
        # Draw landmarks on frame
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        
        # Show buffer status
        cv2.putText(frame, f"Buffer: {len(sequence_buffer)}/60", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if landmarks is not None:
            cv2.putText(frame, f"Features: {landmarks.shape[0]}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No landmarks detected", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Camera Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and landmarks is not None:
            # Save current landmarks for analysis
            np.save(f'debug_landmarks_{frame_count}.npy', landmarks)
            print(f"💾 Saved landmarks to debug_landmarks_{frame_count}.npy")
            print(f"   Shape: {landmarks.shape}")
            print(f"   Range: {landmarks.min():.3f} to {landmarks.max():.3f}")
            print(f"   Mean: {landmarks.mean():.3f}, Std: {landmarks.std():.3f}")
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

def extract_holistic_landmarks(results):
    """Extract holistic landmarks (same as inference.py)."""
    landmarks = []
    
    # Pose landmarks (33 points × 4 = 132 features)
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        landmarks.extend([0.0] * 132)
    
    # Left hand landmarks (21 points × 3 = 63 features)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Right hand landmarks (21 points × 3 = 63 features)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Face landmarks (468 points × 3 = 1404 features)
    if results.face_landmarks:
        face_coords = []
        for landmark in results.face_landmarks.landmark:
            face_coords.extend([landmark.x, landmark.y, landmark.z])
        
        # Ensure we have exactly 1404 features
        if len(face_coords) >= 1404:
            landmarks.extend(face_coords[:1404])
        else:
            landmarks.extend(face_coords + [0.0] * (1404 - len(face_coords)))
    else:
        landmarks.extend([0.0] * 1404)
    
    landmarks_array = np.array(landmarks, dtype=np.float32)
    
    # Should be 1662 features total
    expected_size = 132 + 63 + 63 + 1404  # = 1662
    if len(landmarks_array) != expected_size:
        print(f"⚠️  Landmark size mismatch: {len(landmarks_array)} != {expected_size}")
        return None
    
    return landmarks_array

def analyze_landmarks(landmarks, buffer_size):
    """Analyze the current landmarks."""
    print(f"\n📊 Frame Analysis (Buffer: {buffer_size}/60)")
    print(f"   Shape: {landmarks.shape}")
    print(f"   Range: {landmarks.min():.3f} to {landmarks.max():.3f}")
    print(f"   Mean: {landmarks.mean():.3f}")
    print(f"   Std: {landmarks.std():.3f}")
    print(f"   Zeros: {np.count_nonzero(landmarks == 0)} / {len(landmarks)}")
    print(f"   Non-zeros: {np.count_nonzero(landmarks)} / {len(landmarks)}")

if __name__ == "__main__":
    debug_camera_input()