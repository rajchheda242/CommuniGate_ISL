"""
Test camera access and inference with better error handling.
"""

import cv2
import sys
import time
from inference import HolisticInference

def test_camera():
    """Test camera access."""
    print("Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        print("   Please check:")
        print("   - Camera is connected")
        print("   - No other applications are using the camera")
        print("   - Camera permissions are enabled")
        return False
    
    print("✓ Camera opened successfully")
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print(f"✓ Frame captured successfully: {frame.shape}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✓ Camera properties set")
    else:
        print("❌ Could not capture frame from camera")
        cap.release()
        return False
    
    cap.release()
    return True

def test_inference_demo():
    """Test inference system with automatic exit."""
    print("\\nInitializing inference system...")
    
    try:
        inference = HolisticInference()
        print("✓ Inference system ready")
        
        print("\\nStarting camera demo (will auto-exit after 30 seconds)...")
        print("Controls:")
        print("  'q' - Quit immediately")
        print("  'c' - Clear sequence buffer")
        print("  ESC - Exit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start_time = time.time()
        max_duration = 30  # seconds
        frame_count = 0
        
        print("\\n🎥 Camera demo started...")
        
        while True:
            # Check timeout
            if time.time() - start_time > max_duration:
                print(f"\\n⏰ Auto-exit after {max_duration} seconds")
                break
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Process frame (simplified version)
            frame_landmarks, results = inference.process_frame(frame)
            inference.sequence_buffer.append(frame_landmarks)
            
            # Predict when buffer is full
            current_phrase = None
            confidence = 0.0
            
            if len(inference.sequence_buffer) == 60:  # SEQUENCE_LENGTH updated to 60
                current_phrase, confidence = inference.predict_sequence()
            
            # Flip frame for display
            display_frame = cv2.flip(frame, 1)
            
            # Draw landmarks if available
            if results:
                inference.draw_landmarks(display_frame, results)
            
            # Draw UI
            inference.draw_ui(
                display_frame, 
                current_phrase, 
                confidence, 
                len(inference.sequence_buffer)
            )
            
            # Show frame
            cv2.imshow('ISL Transformer Recognition (Auto-exit in 30s)', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\\n🛑 User requested exit")
                break
            elif key == ord('c'):
                inference.sequence_buffer.clear()
                print("Sequence buffer cleared")
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                remaining = max_duration - elapsed
                print(f"Demo running... {remaining:.0f}s remaining")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\\n✓ Camera demo completed successfully")
        
    except Exception as e:
        print(f"❌ Error during inference demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main test function."""
    print("="*60)
    print("ISL Transformer Inference - System Test")
    print("="*60)
    
    # Test camera
    if not test_camera():
        print("\\n❌ Camera test failed. Cannot proceed with inference demo.")
        sys.exit(1)
    
    # Test inference
    if test_inference_demo():
        print("\\n✅ All tests passed successfully!")
    else:
        print("\\n❌ Inference demo failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()