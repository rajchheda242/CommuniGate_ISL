# 🍎 CommuniGate ISL - Mac Setup Guide

Quick setup guide for running the Indian Sign Language Recognition app on macOS.

---

## Prerequisites

- **macOS** (tested on macOS 10.15+)
- **Python 3.11 or 3.12** (recommended)
- **Webcam** for gesture recognition

---

## Step 1: Install Python (if needed)

Check if Python is installed:
```bash
python3 --version
```

If not installed or version is wrong, install using Homebrew:
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12
brew install python@3.12
```

---

## Step 2: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/rajchheda242/CommuniGate_ISL.git

# Navigate to project directory
cd CommuniGate_ISL
```

---

## Step 3: Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

---

## Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Note:** This installs TensorFlow CPU version which works on all Macs (Intel and Apple Silicon).

---

## Step 5: Train the Model

You need to train the model on your Mac to ensure compatibility:

```bash
# Make sure you're in the project directory
python src/training/train_sequence_model.py
```

**This will take 5-10 minutes.** You should see:
- Loading sequences from data
- Training progress with epochs
- Final accuracy metrics
- "TRAINING COMPLETE!" message

---

## Step 6: Run the Application

### Option A: Using the launch script (easiest)
```bash
# Make the script executable (first time only)
chmod +x launch.sh

# Run the app
./launch.sh
```

### Option B: Direct command
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the app
streamlit run app_enhanced.py
```

The app will open automatically in your default browser at `http://localhost:8501`

---

## Using the App

1. **Allow camera access** when prompted by your browser
2. Click **"Start Recording"** button
3. **Perform one of the supported ISL phrases:**
   - Hi my name is Reet
   - How are you
   - I am from Delhi
   - I like coffee
   - What do you like
4. Click **"Stop & Predict"** when done
5. View the **prediction and confidence** on the right panel

---

## Troubleshooting

### Camera not working
- Grant camera permissions in **System Preferences > Security & Privacy > Camera**
- Make sure no other app is using the camera
- Try restarting the app

### "Model compatibility error"
```bash
# Delete old model files
rm models/saved/lstm_model.keras
rm models/saved/scaler.pkl

# Retrain on your Mac
python src/training/train_sequence_model.py
```

### Dependencies installation fails
```bash
# Make sure you're using the right Python version
python3 --version  # Should be 3.11 or 3.12

# Try installing packages one by one
pip install tensorflow-cpu==2.15.0
pip install streamlit mediapipe opencv-python
pip install joblib scikit-learn
```

### Port already in use
```bash
# Kill the process using port 8501
lsof -ti:8501 | xargs kill -9

# Or run on a different port
streamlit run app_enhanced.py --server.port 8502
```

---

## Quick Commands Reference

```bash
# Activate virtual environment
source venv/bin/activate

# Deactivate virtual environment
deactivate

# Update from GitHub
git pull origin main

# Retrain model (if needed)
python src/training/train_sequence_model.py

# Run the app
streamlit run app_enhanced.py

# Run with custom port
streamlit run app_enhanced.py --server.port 8502
```

---

## Performance Tips for Mac

### For Apple Silicon Macs (M1/M2/M3)
The tensorflow-cpu package works but isn't optimized. For better performance:
```bash
# Optional: Install TensorFlow Metal plugin
pip install tensorflow-metal
```

### For Intel Macs
The setup works out of the box with good performance.

### General optimization
- Close other camera apps
- Use good lighting for better hand detection
- Keep background plain for best results

---

## Recording New Gestures (Optional)

If you want to add your own gestures:

1. **Record gesture samples:**
   ```bash
   python simple_record.py
   ```

2. **Follow the prompts** to record 20-30 samples per gesture

3. **Retrain the model:**
   ```bash
   python src/training/train_sequence_model.py
   ```

---

## Project Structure

```
CommuniGate_ISL/
├── app_enhanced.py          # Main application (USE THIS)
├── src/
│   ├── training/
│   │   └── train_sequence_model.py  # Training script
│   └── ui/
│       └── app.py           # Older version (don't use)
├── models/saved/            # Trained models
├── data/sequences/          # Training data
├── requirements.txt         # Python dependencies
└── launch.sh               # Quick launch script
```

---

## Need Help?

- Check the main [README.md](README.md) for more details
- Review [RECORDING_GUIDE.md](RECORDING_GUIDE.md) for recording tips
- See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for cloud deployment

---

## Notes

- **Mac-trained models won't work on Windows** (and vice versa) due to TensorFlow binary differences
- Each computer needs its own trained model
- The training data (in `data/sequences/`) works across all platforms
- Don't commit trained model files to Git if working across multiple computers

---

**Enjoy using CommuniGate ISL! 🤟**
