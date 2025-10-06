# 🚨 NEXT STEPS - CommuniGate_ISL Setup

## Current Status: Python 3.11 Required

Your project is set up, but **Mediapipe requires Python 3.11** (currently using Python 3.13).

---

## ✅ Quick Fix - Run the Setup Script

I've created an automated setup script for you:

```bash
./setup.sh
```

This will:
1. ✅ Install Python 3.11 (via Homebrew)
2. ✅ Create a new virtual environment with Python 3.11
3. ✅ Install all dependencies including Mediapipe
4. ✅ Verify the installation

---

## 📋 Manual Setup (if script fails)

### Step 1: Install Python 3.11

```bash
# Using Homebrew (recommended)
brew install python@3.11

# OR using pyenv
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9
```

### Step 2: Recreate Virtual Environment

```bash
# Remove current venv
rm -rf .venv

# Create new venv with Python 3.11
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install mediapipe
```

### Step 4: Verify Installation

```bash
python --version  # Should show 3.11.x
python -c "import mediapipe; print('✓ Mediapipe installed')"
python -c "import cv2; print('✓ OpenCV installed')"
```

---

## 🎯 After Setup - Start Development

Once Python 3.11 is installed:

### 1. Test Camera
```bash
python src/data_collection/test_camera.py
```

### 2. Test Hand Detection
```bash
python src/data_collection/test_mediapipe.py
```

### 3. Collect Gesture Data
```bash
python src/data_collection/collect_gestures.py
```

### 4. Train Model
```bash
python src/training/train_model.py
```

### 5. Launch Streamlit UI
```bash
streamlit run src/ui/app.py
```

---

## 📚 Documentation

- **INSTALL.md** - Detailed installation guide
- **SETUP.md** - Complete setup and usage guide
- **ROADMAP.md** - Project roadmap and milestones
- **README.md** - Project overview

---

## 🐛 Troubleshooting

**"python3.11: command not found"**
```bash
brew install python@3.11
```

**"No module named mediapipe"**
```bash
pip install mediapipe
```

**Camera not working**
- Check: System Settings > Privacy & Security > Camera
- Enable camera access for Terminal

---

## 🎉 Ready to Code!

Once setup is complete, follow the **ROADMAP.md** for your 4-week development plan.

**Good luck with your ISL Recognition project!** 🤟
