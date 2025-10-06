# CommuniGate_ISL - Installation Guide

## ⚠️ Important: Python Version Requirement

**Mediapipe requires Python 3.11 or lower.** This project currently uses Python 3.13, which is not yet supported by Mediapipe.

---

## 🔧 Solution: Install Python 3.11

### For macOS (Homebrew):

```bash
# Install Python 3.11
brew install python@3.11

# Remove current virtual environment
rm -rf .venv

# Create new virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install mediapipe
```

### For macOS (pyenv):

```bash
# Install pyenv if not already installed
brew install pyenv

# Install Python 3.11
pyenv install 3.11.9

# Set Python 3.11 for this project
cd /Users/rajchheda/coding/ISL
pyenv local 3.11.9

# Remove current virtual environment
rm -rf .venv

# Create new virtual environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install mediapipe
```

### For Windows:

```bash
# Download Python 3.11 from python.org
# Install it, then:

# Remove current virtual environment
rmdir /s .venv

# Create new virtual environment with Python 3.11
py -3.11 -m venv .venv

# Activate the environment
.venv\Scripts\activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install mediapipe
```

---

## ✅ Verification

After installing Python 3.11 and dependencies, verify the installation:

```bash
# Activate environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Check Python version (should be 3.11.x)
python --version

# Check mediapipe installation
python -c "import mediapipe; print('Mediapipe version:', mediapipe.__version__)"

# Check OpenCV installation
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

---

## 🚀 Quick Setup Script (macOS)

Save this as `setup.sh` and run it:

```bash
#!/bin/bash

echo "Setting up CommuniGate_ISL with Python 3.11..."

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 not found. Installing via Homebrew..."
    brew install python@3.11
fi

# Remove old venv
echo "Removing old virtual environment..."
rm -rf .venv

# Create new venv with Python 3.11
echo "Creating virtual environment with Python 3.11..."
python3.11 -m venv .venv

# Activate venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install mediapipe

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python src/data_collection/test_camera.py"
```

Make it executable and run:
```bash
chmod +x setup.sh
./setup.sh
```

---

## 🐛 Troubleshooting

### "mediapipe not found" error
- Ensure you're using Python 3.11 or 3.10 (not 3.12 or 3.13)
- Check: `python --version` should show 3.11.x or 3.10.x

### "No module named 'cv2'" error
- Make sure virtual environment is activated
- Reinstall: `pip install opencv-python`

### Camera permission issues (macOS)
- Go to System Settings > Privacy & Security > Camera
- Enable camera access for Terminal or your IDE

### Import errors
```bash
# Reinstall all dependencies
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
pip install mediapipe
```

---

## 📦 Alternative: Docker Setup (Advanced)

If you prefer using Docker to avoid Python version issues:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install mediapipe

COPY . .

CMD ["streamlit", "run", "src/ui/app.py"]
```

Build and run:
```bash
docker build -t communigate-isl .
docker run -p 8501:8501 communigate-isl
```

---

## 📝 Summary

1. **Install Python 3.11** using Homebrew or pyenv
2. **Recreate virtual environment** with Python 3.11
3. **Install dependencies** including mediapipe
4. **Verify installation** with test scripts
5. **Start developing!**

For detailed usage instructions, see **SETUP.md**.
