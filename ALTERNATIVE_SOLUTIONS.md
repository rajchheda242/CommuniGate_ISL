# 🔧 Alternative Solutions (If Fragment Doesn't Work)

## If You Get Error: "st.fragment not found"

The `st.fragment` feature was added in **Streamlit 1.33.0** (April 2024).

### Check Your Streamlit Version
```bash
streamlit --version
```

If you have version < 1.33.0, here are alternative solutions:

---

## **Solution 1: Upgrade Streamlit (Recommended)**

```bash
pip install --upgrade streamlit
```

Then restart your app:
```bash
streamlit run app_enhanced.py
```

---

## **Solution 2: Use Alternative Implementation**

If you can't upgrade, replace the camera feed section with this:

### Find this code (around line 442):
```python
# Create persistent camera using fragment for continuous updates
@st.fragment(run_every=0.033)  # ~30 FPS refresh rate
def camera_feed():
    # ... camera code ...

# Run the camera feed fragment
camera_feed()
```

### Replace with:
```python
# Create persistent camera - manual refresh approach
def camera_feed():
    # Initialize camera in session state if not already done
    if 'camera' not in st.session_state or st.session_state.camera is None:
        with st.spinner("🎥 Starting camera..."):
            st.session_state.camera = cv2.VideoCapture(0)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.camera.set(cv2.CAP_PROP_FPS, 30)
            st.session_state.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cap = st.session_state.camera
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.flip(frame, 1)
        extractor = self.get_extractor()
        landmarks, annotated_frame, hands_detected = extractor.process_frame(frame)
        
        if st.session_state.is_recording:
            st.session_state.recorded_sequence.append(landmarks)
            cv2.circle(annotated_frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(annotated_frame, "REC", (60, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frames_recorded = len(st.session_state.recorded_sequence)
            cv2.putText(annotated_frame, f"Frames: {frames_recorded}", (60, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if hands_detected:
            cv2.putText(annotated_frame, "Hands: Detected", (10, annotated_frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "Hands: Not Detected", (10, annotated_frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # Manual refresh with shorter sleep
        time.sleep(0.01)
        st.rerun()
    else:
        st.error("Failed to access camera")
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None

# Run the camera feed
camera_feed()
```

**Note:** This solution still uses `st.rerun()` but the camera persists in session state, so:
- ✅ Camera doesn't restart
- ✅ No 3-5 second delay
- ⚠️ Page still refreshes (but much faster)
- ⚠️ May still have minor scroll issues

---

## **Solution 3: JavaScript-based WebRTC (Advanced)**

Use `streamlit-webrtc` for the best performance:

```bash
pip install streamlit-webrtc
```

This requires more code changes but provides the smoothest experience.

---

## **Recommended Approach**

1. **Try upgrading Streamlit first** - easiest solution
2. If can't upgrade, use Solution 2 (still much better than original)
3. For production apps, consider streamlit-webrtc

---

## Check What's Installed

```bash
pip list | grep streamlit
```

Should show:
```
streamlit                1.33.0 or higher
```

If not, upgrade:
```bash
pip install --upgrade streamlit
```

---

## Still Having Issues?

Create a minimal test file to check fragment support:

**test_fragment.py:**
```python
import streamlit as st
import time

st.title("Fragment Test")

@st.fragment(run_every=1.0)
def test_fragment():
    st.write(f"Current time: {time.time()}")

test_fragment()
```

Run it:
```bash
streamlit run test_fragment.py
```

- ✅ Works? → Your main app should work
- ❌ Error? → Use Solution 2 above
