# CommuniGate_ISL

**Indian Sign Language Recognition System** - A desktop MVP application for recognizing fixed ISL phrases using computer vision.

## 📖 Overview

CommuniGate_ISL is an academic project that uses computer vision and machine learning to recognize **4 fixed Indian Sign Language phrases** through webcam input. The application detects hand landmarks using Mediapipe and classifies gestures to output corresponding text with optional text-to-speech.

## 🎯 Recognized Phrases

1. "Hi, my name is Madiha Siddiqui."
2. "I am a student."
3. "I enjoy running as a hobby."
4. "How are you doing today?"

## 🛠 Tech Stack

- **Python 3.10+**
- **OpenCV** - Webcam input and image processing
- **Mediapipe** - Hand landmark detection
- **scikit-learn** - Gesture classification (KNN/SVM)
- **Streamlit** - User interface
- **pyttsx3** - Text-to-speech (optional)
- **PyInstaller** - Desktop application packaging

## 📁 Project Structure

```
CommuniGate_ISL/
├── .github/
│   └── copilot-instructions.md
├── data/
│   ├── raw/              # Raw gesture recordings
│   └── processed/        # Processed landmark data (CSV)
├── models/
│   └── saved/            # Trained models
├── src/
│   ├── data_collection/  # Scripts for capturing gestures
│   ├── training/         # Model training scripts
│   ├── prediction/       # Live prediction logic
│   └── ui/               # Streamlit interface
├── tests/                # Unit tests
├── .gitignore
├── requirements.txt
├── ROADMAP.md
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.11** (Required for Mediapipe compatibility)
- Webcam/camera device
- pip (Python package manager)
- macOS, Windows, or Linux

### Installation

⚠️ **Important:** This project requires Python 3.11 due to Mediapipe compatibility.

#### Quick Setup (macOS):
```bash
# Run the automated setup script
chmod +x setup.sh
./setup.sh
```

#### Manual Setup:

1. **Install Python 3.11** (if not already installed):
```bash
# macOS (Homebrew)
brew install python@3.11

# or use pyenv
brew install pyenv
pyenv install 3.11.9
```

2. **Clone and setup the project**:
```bash
git clone <repository-url>
cd CommuniGate_ISL
```

3. **Create virtual environment with Python 3.11**:
```bash
# Remove existing .venv if it exists
rm -rf .venv

# Create new venv
python3.11 -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

4. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install mediapipe  # Install separately
```

📖 **For detailed installation instructions, see [INSTALL.md](INSTALL.md)**

### Usage

#### 1. Test Webcam Setup
```bash
python src/data_collection/test_camera.py
```

#### 2. Collect Training Data
```bash
python src/data_collection/collect_gestures.py
```

#### 3. Train the Model
```bash
python src/training/train_model.py
```

#### 4. Run the Application
```bash
streamlit run src/ui/app.py
```

## 📊 Development Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development phases and timeline.

## 🧪 Testing

Run tests using:
```bash
pytest tests/
```

## 📦 Building Desktop Application

### Windows (.exe)
```bash
pyinstaller --onefile --windowed src/ui/app.py
```

### macOS (.app)
```bash
pyinstaller --onefile --windowed --osx-bundle-identifier com.communigate.isl src/ui/app.py
```

## 🤝 Contributing

This is an academic project. For suggestions or improvements, please open an issue or submit a pull request.

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
