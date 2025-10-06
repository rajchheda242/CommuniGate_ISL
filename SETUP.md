# CommuniGate_ISL - Setup Guide

## 🎉 Project Successfully Created!

Your **CommuniGate_ISL** project has been set up with a complete Python structure for Indian Sign Language recognition.

---

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
│   │   ├── test_camera.py
│   │   ├── test_mediapipe.py
│   │   └── collect_gestures.py
│   ├── training/         # Model training scripts
│   │   └── train_model.py
│   ├── prediction/       # Live prediction logic
│   │   └── predictor.py
│   └── ui/               # Streamlit interface
│       └── app.py
├── tests/                # Unit tests
├── .gitignore
├── requirements.txt
├── ROADMAP.md
├── README.md
└── SETUP.md (this file)
```

---

## ⚙️ Environment Setup

### Python Environment
✅ **Virtual environment created**: `.venv`  
✅ **Python version**: 3.13.2  
✅ **Dependencies installed** (except mediapipe - see note below)

### Important Note about Mediapipe
⚠️ **Mediapipe** is not currently available for Python 3.13 via pip. You have two options:

**Option 1: Use Python 3.11 (Recommended)**
```bash
# Install Python 3.11
brew install python@3.11  # macOS with Homebrew

# Create new venv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install mediapipe
```

**Option 2: Build from source (Advanced)**
Follow instructions at: https://github.com/google/mediapipe

---

## 🚀 Quick Start

### 1. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Install Mediapipe (if using Python 3.11)
```bash
pip install mediapipe
```

### 3. Test Your Camera
```bash
python src/data_collection/test_camera.py
```

### 4. Test Mediapipe Hand Detection
```bash
python src/data_collection/test_mediapipe.py
```

### 5. Collect Gesture Data
```bash
python src/data_collection/collect_gestures.py
```

### 6. Train the Model
```bash
python src/training/train_model.py
```

### 7. Run Live Prediction (Terminal)
```bash
python src/prediction/predictor.py
```

### 8. Launch Streamlit UI
```bash
streamlit run src/ui/app.py
```

---

## 📝 Development Workflow

### Phase 1: Setup & Testing (Current)
- [x] Project structure created
- [x] Dependencies installed (except mediapipe)
- [ ] Install mediapipe with Python 3.11
- [ ] Test camera functionality
- [ ] Test hand landmark detection

### Phase 2: Data Collection
1. Run `test_camera.py` to verify webcam
2. Run `test_mediapipe.py` to verify hand detection
3. Run `collect_gestures.py` to capture training data
4. Collect 30-50 samples per phrase

### Phase 3: Model Training
1. Run `train_model.py` to train classifier
2. Model will be saved to `models/saved/`
3. Review accuracy and classification report

### Phase 4: Live Prediction
1. Test with `predictor.py` (terminal version)
2. Launch `app.py` (Streamlit UI)
3. Fine-tune model if needed

### Phase 5: Packaging
```bash
# Windows
pyinstaller --onefile --windowed src/ui/app.py

# macOS
pyinstaller --onefile --windowed --osx-bundle-identifier com.communigate.isl src/ui/app.py
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

---

## 📚 Key Files

- **ROADMAP.md** - Detailed project roadmap with weekly milestones
- **README.md** - Project documentation and usage guide
- **requirements.txt** - Python dependencies
- **.gitignore** - Git ignore patterns

---

## 🔧 Troubleshooting

### Camera Access Issues
- Ensure your terminal/IDE has camera permissions
- On macOS: System Settings > Privacy & Security > Camera

### Import Errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Mediapipe Not Working
- Use Python 3.11 instead of 3.13
- Check https://github.com/google/mediapipe for compatibility

---

## 📖 Next Steps

1. **Install mediapipe** using Python 3.11
2. **Test camera and hand detection** with provided scripts
3. **Collect training data** for 4 phrases
4. **Train model** and evaluate accuracy
5. **Build UI** and test live recognition
6. **Package application** for distribution

---

## 💡 Tips

- Start with good lighting for data collection
- Keep hand gestures consistent across samples
- Test model regularly during development
- Use Streamlit for quick UI prototyping
- Document any changes to the workflow

---

## 📞 Support

For issues or questions:
- Check the ROADMAP.md for detailed steps
- Review README.md for usage examples
- Refer to individual script docstrings

---

**Happy Coding! 🎉**
