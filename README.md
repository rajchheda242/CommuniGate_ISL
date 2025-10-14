# CommuniGate_ISL

**Indian Sign Language Recognition System** - A desktop MVP application for recognizing fixed ISL phrases using temporal sequence recognition.

## 📖 Overview

CommuniGate_ISL is an academic project that uses **temporal sequence recognition** with LSTM neural networks to recognize **4 fixed Indian Sign Language phrases**. The system processes video sequences to understand multi-word phrases where each word has its own sign performed over time.

## 🎯 Recognized Phrases

1. "Hi my name is Reet"
2. "How are you"
3. "I am from Delhi"
4. "I like coffee"
5. "What do you like"

## 🛠 Tech Stack

- **Python 3.11** (Required for Mediapipe compatibility)
- **OpenCV** - Video processing and frame extraction
- **Mediapipe** - Hand landmark detection
- **TensorFlow/Keras** - LSTM neural network for sequence classification
- **Streamlit** - User interface
- **pyttsx3** - Text-to-speech (optional)
- **PyInstaller** - Desktop application packaging

## 🧠 How It Works

Unlike static gesture recognition, this system uses **temporal sequence analysis**:

1. **Video Input**: 2-3 second video clips of ISL phrases
2. **Frame Processing**: Extract hand landmarks from each frame (~60 frames)
3. **Sequence Formation**: Create temporal sequences (60 frames × 126 features)
4. **LSTM Classification**: Bidirectional LSTM learns temporal patterns
5. **Phrase Prediction**: Outputs recognized phrase with confidence

This approach properly handles **multi-word phrases** where each word is signed sequentially.

## 📁 Project Structure

```
CommuniGate_ISL/
├── data/
│   ├── videos/           # Raw MP4 video files
│   │   ├── phrase_0/     # Videos for phrase 0
│   │   ├── phrase_1/     # Videos for phrase 1
│   │   ├── phrase_2/     # Videos for phrase 2
│   │   └── phrase_3/     # Videos for phrase 3
│   └── sequences/        # Processed landmark sequences
│       ├── phrase_0/     # .npy files with landmark sequences
│       ├── phrase_1/
│       ├── phrase_2/
│       └── phrase_3/
├── models/
│   └── saved/            # Trained LSTM models
├── src/
│   ├── data_collection/
│   │   ├── test_camera.py           # Test webcam
│   │   ├── test_mediapipe.py        # Test hand detection
│   │   ├── collect_sequences.py     # Live webcam collection
│   │   └── process_videos.py        # Process pre-recorded MP4s ⭐ NEW
│   ├── training/
│   │   └── train_sequence_model.py  # LSTM training
│   ├── prediction/
│   │   └── predictor.py
│   └── ui/
│       └── app.py
├── requirements.txt
├── ROADMAP.md
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.11** (Required for Mediapipe compatibility)
- pip (Python package manager)
- macOS, Windows, or Linux
- **Pre-recorded MP4 videos** OR webcam for live data collection

### Installation

#### 1. Install Python 3.11
```bash
# macOS (Homebrew)
brew install python@3.11

# or use pyenv
brew install pyenv
pyenv install 3.11.9
```

#### 2. Setup Project
```bash
git clone <repository-url>
cd CommuniGate_ISL

# Create virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install mediapipe
```

#### 4. Verify Installation
```bash
python src/data_collection/test_camera.py
python src/data_collection/test_mediapipe.py
```

## 📊 Workflow

### Option A: Using Pre-recorded MP4 Videos (Recommended for Multiple Contributors)

**Perfect for your use case with 5 people recording 10 videos each!**

#### 1. Organize Video Files
```bash
# Create directory structure
mkdir -p data/videos/phrase_{0,1,2,3}

# Place videos in appropriate folders:
# data/videos/phrase_0/person1_take1.mp4
# data/videos/phrase_0/person1_take2.mp4
# ... (10 videos per person, 5 people = 50 videos per phrase)
```

#### 2. Process Videos to Extract Sequences
```bash
python src/data_collection/process_videos.py
```
This will:
- Read all MP4 files from `data/videos/`
- Extract hand landmarks from each frame
- Save as sequence arrays in `data/sequences/`

#### 3. Train LSTM Model
```bash
python src/training/train_sequence_model.py
```

#### 4. Run Live Recognition
```bash
streamlit run src/ui/app.py
```

---

### Option B: Live Webcam Collection

If you want to collect data directly:
```bash
python src/data_collection/collect_sequences.py
```

---

## 🎥 Video Recording Guidelines

For best results when recording the MP4 videos:

- **Duration**: 2-4 seconds per video
- **Format**: MP4, MOV, or AVI
- **Frame Rate**: 30 fps (standard)
- **Lighting**: Good, even lighting
- **Background**: Plain background preferred
- **Camera Position**: Front-facing, waist-up view
- **Hand Visibility**: Both hands clearly visible
- **Signing**: Perform all signs in the phrase naturally and sequentially

**Example naming convention:**
```
phrase_0/person1_video01.mp4
phrase_0/person1_video02.mp4
...
phrase_0/person5_video10.mp4
```

## 📖 Documentation

- **ROADMAP.md** - Development phases and timeline
- **VIDEO_PROCESSING_GUIDE.md** - Detailed guide for processing MP4 files

## 🧪 Testing

```bash
# Test camera
python src/data_collection/test_camera.py

# Test mediapipe
python src/data_collection/test_mediapipe.py
```

## 📄 License

This project is created for academic purposes.

## 👥 Authors

- Madiha Siddiqui

## 🙏 Acknowledgments

- Mediapipe team for hand landmark detection
- OpenCV community
- Streamlit framework

---

**Note**: This is an MVP for demonstration and pitching purposes, not intended for production use.
