"""
Test script to verify Mediapipe hand detection functionality.
Displays webcam feed with hand landmarks overlaid.
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import sys


def test_hand_detection():
    """Test Mediapipe hand landmark detection."""
    print("Testing Mediapipe hand detection...")
    print("Show your hand(s) to the camera")
    print("Press 'q' to quit")
    
    # Initialize Mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    # Initialize hand detection
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Display number of detected hands
                num_hands = len(results.multi_hand_landmarks)
                cv2.putText(
                    frame, 
                    f'Hands detected: {num_hands}', 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
            
            # Display the frame
            cv2.imshow('Mediapipe Hand Detection - Press Q to quit', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Mediapipe test completed!")


if __name__ == "__main__":
    test_hand_detection()
