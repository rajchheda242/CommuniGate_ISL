# Indian Sign Language (ISL) MVP – Project Roadmap

## 🎯 Goal
Build a **desktop MVP application** (Windows/Mac) that recognizes **4 fixed ISL phrases** using temporal sequence recognition with LSTM neural networks. The system processes video sequences to understand multi-word phrases where each word is signed sequentially.

---

## 🛠 Tech Stack
- **Programming Language**: Python 3.11 (required for Mediapipe)
- **Libraries**:
  - [OpenCV](https://opencv.org/) → Video processing and frame extraction
  - [Mediapipe](https://developers.google.com/mediapipe) → Hand landmark detection
  - [TensorFlow/Keras](https://www.tensorflow.org/) → LSTM neural network
  - [Streamlit](https://streamlit.io/) → User interface
  - [pyttsx3](https://pypi.org/project/pyttsx3/) → Text-to-speech (optional)
  - [tqdm](https://tqdm.github.io/) → Progress bars
- **Packaging**: PyInstaller / auto-py-to-exe → `.exe` (Windows) / `.app` (Mac)

---

## 📋 Deliverables
1. Video processing pipeline for MP4 files ✅
2. Hand landmark sequence extraction using Mediapipe ✅
3. LSTM-based temporal sequence classifier ✅
4. Streamlit UI with live webcam recognition
5. Optional: Text-to-speech integration
6. Packaged desktop application (.exe / .app)
7. Usage documentation ✅

---

## 🗂 Fixed Phrases
1. "Hi, my name is Madiha Siddiqui."
2. "I am a student."
3. "I enjoy running as a hobby."
4. "How are you doing today."

---

## 📊 Data Collection Strategy

**Approach**: Pre-recorded MP4 videos from multiple contributors

**Setup**:
- 5 people recording
- Each person performs each phrase 10 times
- Total: 200 videos (50 per phrase)

**Benefits**:
- Better generalization across different people
- Consistent quality control
- Can re-use and augment later
- No need for live data collection sessions  

---

## 🗺️ Development Phases

### **Phase 1: Setup & Infrastructure** ✅ COMPLETE
- [x] Install Python 3.11 and dependencies
- [x] Create project structure
- [x] Set up version control
- [x] Install OpenCV, Mediapipe, TensorFlow
- [x] Test camera and hand detection

### **Phase 2: Video Collection & Processing** 🔄 IN PROGRESS
- [ ] Coordinate with 5 contributors
- [ ] Record 200 MP4 videos (50 per phrase)
  - Each contributor: 10 videos per phrase
  - Duration: 2-4 seconds each
  - Quality: Good lighting, clear hand visibility
- [x] Create video processing script
- [ ] Process videos to extract sequences
- [ ] Verify data quality

**Timeline**: 1-2 days for recording, 1-2 hours for processing

### **Phase 3: Model Development** 📋 READY
- [x] Design LSTM architecture
- [x] Implement sequence normalization
- [x] Create training pipeline
- [ ] Train model on processed sequences
- [ ] Evaluate performance (target: >85% accuracy)
- [ ] Fine-tune hyperparameters if needed

**Timeline**: 2-3 hours

### **Phase 4: Live Recognition System**
- [ ] Build Streamlit UI
- [ ] Integrate webcam capture
- [ ] Implement real-time sequence buffering
- [ ] Connect trained model for live prediction
- [ ] Display recognized phrases
- [ ] Add confidence scores

**Timeline**: 1-2 days

### **Phase 5: Enhancement & Polish**
- [ ] Add text-to-speech integration
- [ ] Improve UI/UX design
- [ ] Add visual feedback (landmark overlay)
- [ ] Implement smoothing for predictions
- [ ] Create demo mode

**Timeline**: 1 day

### **Phase 6: Packaging & Deployment**
- [ ] Test on fresh machine
- [ ] Package with PyInstaller
- [ ] Create `.exe` for Windows
- [ ] Create `.app` for macOS
- [ ] Write user documentation
- [ ] Prepare demo presentation

**Timeline**: 1-2 days  

---

## ✅ Current Status

**What's Complete:**
- ✅ Development environment setup
- ✅ Video processing script (`process_videos.py`)
- ✅ LSTM model architecture (`train_sequence_model.py`)
- ✅ Data collection infrastructure
- ✅ Documentation

**Next Immediate Steps:**
1. **Collect Videos**: Get 200 MP4 files from 5 contributors
2. **Process Videos**: Run `python src/data_collection/process_videos.py`
3. **Train Model**: Run `python src/training/train_sequence_model.py`
4. **Build UI**: Create Streamlit interface for live recognition

---

## 📁 Data Structure

```
data/
├── videos/              # Raw MP4 files (to be collected)
│   ├── phrase_0/        # 50 videos: person1-5 × 10 takes each
│   ├── phrase_1/        # 50 videos
│   ├── phrase_2/        # 50 videos
│   └── phrase_3/        # 50 videos
└── sequences/           # Processed landmark sequences (auto-generated)
    ├── phrase_0/        # 50 .npy files (60 frames × 126 features each)
    ├── phrase_1/        # 50 .npy files
    ├── phrase_2/        # 50 .npy files
    └── phrase_3/        # 50 .npy files
```

---

## 🎯 Success Metrics

**Model Performance:**
- Training accuracy: >90%
- Validation accuracy: >85%
- Test accuracy: >80%
- Per-phrase accuracy: >75% for each

**User Experience:**
- Recognition latency: <500ms
- Smooth real-time prediction
- Clear visual feedback
- Intuitive interface

---

## 📖 Documentation

- **README.md** - Project overview and quick start
- **ROADMAP.md** - This file, development plan
- **VIDEO_PROCESSING_GUIDE.md** - Detailed guide for processing MP4 files

---

## ✅ Final Output
- Standalone desktop app with:  
  - Live webcam input  
  - Real-time LSTM-based sequence recognition
  - Recognition of 4 multi-word ISL phrases  
  - Text display of recognized phrase  
  - Optional spoken output  
  - Confidence scores
- Complete documentation
- Trained model with >80% accuracy
- Presentation-ready demo

---

## 🚀 Getting Started

**For contributors collecting videos:**
See `VIDEO_PROCESSING_GUIDE.md` for recording guidelines

**For developers:**
1. Install dependencies: `pip install -r requirements.txt && pip install mediapipe`
2. Test setup: `python src/data_collection/test_mediapipe.py`
3. Process videos: `python src/data_collection/process_videos.py`
4. Train model: `python src/training/train_sequence_model.py`
5. Run app: `streamlit run src/ui/app.py`
