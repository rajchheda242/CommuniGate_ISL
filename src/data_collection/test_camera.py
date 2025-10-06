"""
Test script to verify webcam functionality with OpenCV.
This script opens the webcam and displays the video feed.
Press 'q' to quit.
"""

import cv2
import sys


def test_camera():
    """Test if the webcam is working properly."""
    print("Testing camera...")
    print("Press 'q' to quit")
    
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    print("Camera opened successfully!")
    print("Camera resolution: {}x{}".format(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Display the frame
        cv2.imshow('Camera Test - Press Q to quit', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed!")


if __name__ == "__main__":
    test_camera()
