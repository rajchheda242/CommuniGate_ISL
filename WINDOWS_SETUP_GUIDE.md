# 🪟 Windows Setup Guide - CommuniGate ISL
## Complete Step-by-Step Instructions for Windows Laptop

---

## 📋 **PART 1: Initial Setup (One-Time)**

### Step 1: Install Python 3.9 or 3.10

1. **Download Python:**
   - Go to: https://www.python.org/downloads/
   - Download **Python 3.10.x** (recommended) or **Python 3.9.x**
   - ⚠️ **IMPORTANT:** During installation, check "Add Python to PATH"

2. **Verify Installation:**
   - Open **Command Prompt** (search "cmd" in Start menu)
   - Run these commands:
   ```cmd
   python --version
   pip --version
   ```
   - Should show Python 3.9 or 3.10

---

### Step 2: Install Git (to download the project)

1. **Download Git:**
   - Go to: https://git-scm.com/download/win
   - Download and install (use default options)

2. **Verify Installation:**
   ```cmd
   git --version
   ```

---

### Step 3: Download the Project

**Option A: Using Git (Recommended)**
```cmd
cd C:\Users\%USERNAME%\Desktop
git clone https://github.com/rajchheda242/CommuniGate_ISL.git
cd CommuniGate_ISL
```

**Option B: Download ZIP**
1. Go to GitHub repository
2. Click "Code" → "Download ZIP"
3. Extract to `C:\Users\YOUR_NAME\Desktop\CommuniGate_ISL`
4. Open Command Prompt:
```cmd
cd C:\Users\%USERNAME%\Desktop\CommuniGate_ISL
```

---

### Step 4: Create Virtual Environment & Install Dependencies

**Copy-paste these commands ONE BY ONE:**

```cmd
REM Create virtual environment
python -m venv .venv

REM Activate virtual environment
.venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install all dependencies
pip install -r requirements.txt

REM Install PyInstaller for creating exe
pip install pyinstaller

REM Verify installation
pip list
```

**Expected output:** Should see tensorflow, streamlit, mediapipe, opencv-python, etc.

---

## 🎯 **PART 2: Test the App First**

**Before creating EXE, test that everything works:**

```cmd
REM Make sure you're in the project directory
cd C:\Users\%USERNAME%\Desktop\CommuniGate_ISL

REM Activate virtual environment
.venv\Scripts\activate.bat

REM Run the app
streamlit run src\ui\app.py
```

**What should happen:**
- Browser opens automatically
- App shows "CommuniGate ISL - Smart Recognition"
- You see your logo in the header
- Camera works when you click "Start Recording"

**If it works → Great! Proceed to create EXE**
**If it doesn't work → Fix issues first before creating EXE**

---

## 📦 **PART 3: Create Standalone EXE**

### Method 1: Using the Build Script (Easiest)

**Create the build script:**

1. **Create file:** `build_windows_exe.py` in the project root
2. **Paste this code:**

```python
"""
Windows EXE Builder for CommuniGate ISL
Run this script to create a standalone executable
"""
import PyInstaller.__main__
import os
import shutil
import sys

print("=" * 60)
print("  CommuniGate ISL - Windows EXE Builder")
print("=" * 60)
print()

# Clean previous builds
print("🧹 Cleaning previous builds...")
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('build'):
    shutil.rmtree('build')

# Check if model exists
if not os.path.exists('models/saved/lstm_model.keras'):
    print("⚠️  WARNING: Model not found!")
    print("   You need to train the model first.")
    response = input("   Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

print("📦 Building executable...")
print("   This may take 5-10 minutes...")
print()

# Build configuration
PyInstaller.__main__.run([
    'src/ui/app.py',                    # Main file
    '--name=CommuniGateISL',            # EXE name
    '--onefile',                        # Single file
    '--noconsole',                      # No console window
    '--add-data=models;models',         # Include models
    '--add-data=src;src',               # Include source
    '--add-data=.streamlit;.streamlit', # Include config
    '--hidden-import=streamlit',
    '--hidden-import=streamlit.web.cli',
    '--hidden-import=streamlit.runtime.scriptrunner.magic_funcs',
    '--hidden-import=tensorflow',
    '--hidden-import=mediapipe',
    '--hidden-import=cv2',
    '--hidden-import=sklearn',
    '--hidden-import=joblib',
    '--collect-all=streamlit',
    '--collect-all=mediapipe',
    '--collect-all=cv2',
    '--icon=src/ui/assets/icon.png',    # App icon
])

print()
print("=" * 60)
print("✅ BUILD COMPLETE!")
print("=" * 60)
print()
print("📍 Your executable is here:")
print(f"   {os.path.abspath('dist/CommuniGateISL.exe')}")
print()
print("📝 Next steps:")
print("   1. Test the EXE by double-clicking it")
print("   2. Copy to USB drive for your demo")
print("   3. Bring the entire 'dist' folder, not just the EXE")
print()
```

**Run the builder:**
```cmd
REM Make sure virtual environment is activated
.venv\Scripts\activate.bat

REM Run the builder
python build_windows_exe.py
```

---

### Method 2: Manual PyInstaller (Alternative)

**If the script doesn't work, use this command:**

```cmd
.venv\Scripts\activate.bat

pyinstaller --name=CommuniGateISL ^
  --onefile ^
  --noconsole ^
  --add-data="models;models" ^
  --add-data="src;src" ^
  --add-data=".streamlit;.streamlit" ^
  --hidden-import=streamlit ^
  --hidden-import=tensorflow ^
  --hidden-import=mediapipe ^
  --hidden-import=cv2 ^
  --collect-all=streamlit ^
  --collect-all=mediapipe ^
  --icon=src/ui/assets/icon.png ^
  src/ui/app.py
```

---

## ⚠️ **IMPORTANT: EXE Limitations & Alternative**

### 🚨 **Problem with Streamlit EXE:**
Creating a working Streamlit app as EXE is **very tricky** and often fails because:
- Streamlit needs to run a web server
- Large file size (1-2 GB)
- May not work on all Windows machines
- Camera access can be problematic

### ✅ **RECOMMENDED SOLUTION: Portable Python Package**

**This is MUCH MORE RELIABLE for demos!**

Instead of a single EXE, create a **portable folder** that works on any Windows machine:

```cmd
REM Create portable package directory
mkdir CommuniGate_ISL_Portable
cd CommuniGate_ISL_Portable

REM Copy your project
xcopy /E /I C:\Users\%USERNAME%\Desktop\CommuniGate_ISL\* .

REM Create a simple launcher
echo @echo off > LAUNCH_APP.bat
echo cd /d "%~dp0" >> LAUNCH_APP.bat
echo python -m venv .venv >> LAUNCH_APP.bat
echo call .venv\Scripts\activate.bat >> LAUNCH_APP.bat
echo pip install -r requirements.txt >> LAUNCH_APP.bat
echo streamlit run src\ui\app.py >> LAUNCH_APP.bat
echo pause >> LAUNCH_APP.bat
```

**Create a better launcher script:**

Save as `LAUNCH_APP.bat` in the project root:

```batch
@echo off
REM ============================================
REM   CommuniGate ISL - Quick Launch
REM ============================================

echo.
echo Starting CommuniGate ISL...
echo.

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.9 or 3.10 from python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    
    echo Installing dependencies...
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

REM Check if model exists
if not exist "models\saved\lstm_model.keras" (
    echo.
    echo WARNING: Model not found!
    echo The app may not work without a trained model.
    echo.
)

echo.
echo ✅ Starting CommuniGate ISL...
echo    The app will open in your browser.
echo    If not, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo ============================================
echo.

REM Kill any existing Streamlit
taskkill /F /IM streamlit.exe 2>nul

REM Launch app
streamlit run src\ui\app.py

echo.
echo App stopped.
pause
```

---

## 📁 **PART 4: Package for Demo**

### **Option A: USB Drive Package (RECOMMENDED)**

**What to copy:**
1. Entire project folder
2. Python installer (as backup)
3. Quick start instructions

**Steps:**
```cmd
REM Create demo package
mkdir C:\Demo_Package
xcopy /E /I C:\Users\%USERNAME%\Desktop\CommuniGate_ISL C:\Demo_Package\CommuniGate_ISL

REM Copy to USB drive (replace F: with your USB drive letter)
xcopy /E /I C:\Demo_Package\CommuniGate_ISL F:\CommuniGate_ISL
```

**On demo machine:**
1. Copy folder from USB to Desktop
2. Double-click `LAUNCH_APP.bat`
3. Wait for setup (first time only)
4. App launches!

---

### **Option B: Create Installer with Inno Setup**

**Download Inno Setup:**
- https://jrsoftware.org/isinfo.php

**Create installer script:** `installer.iss`

```ini
[Setup]
AppName=CommuniGate ISL
AppVersion=1.0
DefaultDirName={pf}\CommuniGate_ISL
DefaultGroupName=CommuniGate ISL
OutputBaseFilename=CommuniGate_ISL_Setup
Compression=lzma2
SolidCompression=yes

[Files]
Source: "C:\Users\YOUR_NAME\Desktop\CommuniGate_ISL\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\CommuniGate ISL"; Filename: "{app}\LAUNCH_APP.bat"
Name: "{commondesktop}\CommuniGate ISL"; Filename: "{app}\LAUNCH_APP.bat"
```

---

## 🎯 **RECOMMENDED FOR YOUR DEMO**

### **Best Approach: Portable Installation**

1. **On your Windows laptop:**
   ```cmd
   cd C:\Users\%USERNAME%\Desktop
   git clone https://github.com/rajchheda242/CommuniGate_ISL.git
   cd CommuniGate_ISL
   python -m venv .venv
   .venv\Scripts\activate.bat
   pip install -r requirements.txt
   streamlit run src\ui\app.py
   ```

2. **Test everything works**

3. **Copy entire folder to USB drive**

4. **On demo machine:**
   - Copy folder from USB
   - Double-click `launch.bat`
   - Demo!

---

## ✅ **Quick Copy-Paste Commands Summary**

**Complete setup (copy all at once):**

```cmd
cd C:\Users\%USERNAME%\Desktop
git clone https://github.com/rajchheda242/CommuniGate_ISL.git
cd CommuniGate_ISL
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run src\ui\app.py
```

---

## 🆘 **Troubleshooting**

### Python not found
```cmd
REM Install from python.org, make sure to check "Add to PATH"
```

### Pip install fails
```cmd
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

### Camera not working
- Grant camera permissions in Windows Settings
- Check if antivirus is blocking camera access

### Streamlit won't start
```cmd
taskkill /F /IM streamlit.exe
streamlit run src\ui\app.py
```

---

## 📞 **Need Help?**

If you encounter issues:
1. Check error messages carefully
2. Make sure Python 3.9 or 3.10 is installed
3. Verify all files are present
4. Try running from Command Prompt, not PowerShell

**Good luck with your demo! 🚀**
