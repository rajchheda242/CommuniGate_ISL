# 🚀 CommuniGate ISL - Deployment Guide

## 📊 **Deployment Options Comparison**

| Method | Best For | Pros | Cons | Setup Time |
|--------|----------|------|------|------------|
| **Streamlit Cloud** ☁️ | Quick demos, sharing | Free, easy, auto-updates | Requires internet, camera permission issues | 15 min |
| **Local Executable** 💻 | Offline demos, presentations | Works offline, no camera issues | Large file size, OS-specific | 30 min |
| **Docker + Cloud** 🐳 | Production, scaling | Professional, scalable | Complex, needs server | 1-2 hours |
| **Heroku/Railway** 🚂 | Web hosting | Easy deploy, good for web | Camera issues, costly | 30 min |

---

## ⭐ **RECOMMENDED: Local Demo (Streamlit + Python)**

**Best for your demo scenario** - Works offline, full camera access, easy to set up.

### Quick Setup:
```bash
# On demo machine
git clone <your-repo>
cd CommuniGate_ISL
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/ui/app.py
```

**Advantages:**
- ✅ Full camera access (no browser permission issues)
- ✅ Works without internet
- ✅ Easy to troubleshoot
- ✅ Best performance

---

## 🎨 **Adding Your Logo**

### Step 1: Prepare Logo Files

Create two versions:
1. **Small Icon** (`icon.png`) - 64x64px or 128x128px for browser tab
2. **Large Logo** (`logo.png`) - 200x200px to 500x500px for header

### Step 2: Add to Assets Folder

```bash
# Place your logo files here:
src/ui/assets/logo.png    # Main logo (displays in app header)
src/ui/assets/icon.png    # Small icon (browser tab)
```

### Step 3: Logo Usage (Already Implemented!)

The app now automatically:
- Uses `icon.png` for browser tab if available
- Displays `logo.png` in header if available
- Falls back to emoji 🤟 if no custom logo

---

## 💻 **Option 1: Create Standalone Executable (PyInstaller)**

**Best for:** Offline demos at events, sharing with non-technical users

### Installation:
```bash
source .venv/bin/activate
pip install pyinstaller
```

### Create Build Script:

Save as `build_app.py`:
```python
import PyInstaller.__main__
import os
import shutil

# Clean previous builds
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('build'):
    shutil.rmtree('build')

PyInstaller.__main__.run([
    'src/ui/app.py',
    '--name=CommuniGateISL',
    '--onefile',  # Single executable
    '--windowed',  # No console window (optional)
    '--add-data=models:models',
    '--add-data=src:src',
    '--hidden-import=streamlit',
    '--hidden-import=tensorflow',
    '--hidden-import=mediapipe',
    '--hidden-import=cv2',
    '--collect-all=streamlit',
    '--collect-all=mediapipe',
    '--icon=src/ui/assets/icon.png',  # Optional: Add icon
])
```

### Build:
```bash
python build_app.py
```

### Run:
```bash
# The executable will be in dist/
./dist/CommuniGateISL
```

### ⚠️ Limitations:
- Large file size (500MB - 1GB+)
- Needs to be built separately for each OS
- Streamlit in executable can be tricky

---

## ☁️ **Option 2: Streamlit Cloud (Free)**

**Best for:** Quick sharing, web access, no installation needed

### Steps:

1. **Push to GitHub:**
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repo
   - Set main file: `src/ui/app.py`
   - Click Deploy!

3. **Configure (create `.streamlit/config.toml`):**
```toml
[server]
maxUploadSize = 200
enableCORS = true
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### ⚠️ Camera Issues:
- Browser camera access requires HTTPS ✅ (Streamlit Cloud has this)
- User must grant camera permission
- Some corporate networks block camera access

---

## 🐳 **Option 3: Docker Container**

**Best for:** Consistent environment, professional deployment

### Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0"]
```

### Build & Run:
```bash
docker build -t communigate-isl .
docker run -p 8501:8501 communigate-isl
```

---

## 🌐 **Option 4: Cloud Platforms**

### Heroku
```bash
# Install Heroku CLI, then:
heroku create communigate-isl
git push heroku main
```

### Railway.app
1. Connect GitHub repo
2. Auto-deploys on push
3. Free tier available

### AWS/GCP/Azure
- More complex but scalable
- Use VM + Docker approach

---

## 🎯 **RECOMMENDATION FOR YOUR DEMO**

### **Best Choice: Local Streamlit App** ✅

**Why:**
1. ✅ Full camera control (no browser permission issues)
2. ✅ Works offline (no internet dependency)
3. ✅ Fast and responsive
4. ✅ Easy to debug on-site
5. ✅ Professional appearance

### **Setup Checklist:**

**Before Demo Day:**
- [ ] Test on demo machine
- [ ] Install Python 3.9+
- [ ] Clone repo and install dependencies
- [ ] Add your logo to `src/ui/assets/`
- [ ] Test camera access
- [ ] Create shortcut/script to launch app

**Demo Day:**
1. Connect camera
2. Open terminal
3. Run: `streamlit run src/ui/app.py`
4. Browser opens automatically
5. Demo! 🎉

### **Backup Plan:**
- Keep a USB with:
  - Complete project folder
  - Python installer
  - Demo video (in case of technical issues)

---

## 🚀 **Quick Launch Scripts**

### macOS/Linux (`launch.sh`):
```bash
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run src/ui/app.py
```

### Windows (`launch.bat`):
```batch
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run src/ui/app.py
```

Make executable:
```bash
chmod +x launch.sh  # macOS/Linux
```

---

## 📝 **Final Notes**

1. **Logo Tips:**
   - Use PNG with transparency
   - Keep file size < 500KB
   - Square aspect ratio works best

2. **Demo Tips:**
   - Arrive early to test setup
   - Have backup recordings ready
   - Keep terminal visible for debugging
   - Test all gestures before demo

3. **Performance:**
   - Close unnecessary apps
   - Ensure good lighting
   - Position camera at chest level
   - Clean background helps

---

## 🆘 **Troubleshooting**

### Camera not working:
```bash
# Check camera access
ls /dev/video*  # Linux/macOS
# Grant camera permissions in System Preferences (macOS)
```

### Streamlit won't start:
```bash
# Kill existing instances
pkill -9 -f streamlit
# Try again
streamlit run src/ui/app.py
```

### Model not loading:
```bash
# Verify model exists
ls -la models/saved/
# Retrain if needed
python src/training/train_sequence_model.py
```

---

**Need help? Check README.md or create an issue!** 🚀
