@echo off
REM ============================================================
REM   CommuniGate ISL - Quick Launch for Windows
REM   Double-click this file to start the application
REM ============================================================

title CommuniGate ISL Launcher

echo.
echo ============================================================
echo   CommuniGate ISL - Indian Sign Language Recognition
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo.
    echo Please install Python 3.9 or 3.10 from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if virtual environment exists
if not exist ".venv\" (
    echo.
    echo 📦 First time setup - Creating virtual environment...
    python -m venv .venv
    
    echo.
    echo 📥 Installing dependencies (this may take 2-5 minutes)...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    echo.
    echo ✅ Setup complete!
) else (
    call .venv\Scripts\activate.bat
)

REM Check if model exists
if not exist "models\saved\lstm_model.keras" (
    echo.
    echo ⚠️  WARNING: Trained model not found!
    echo    Path: models\saved\lstm_model.keras
    echo.
    echo    The app may not work properly without a trained model.
    echo    You may need to train it first or copy it from another location.
    echo.
    echo Press any key to continue anyway, or Ctrl+C to cancel...
    pause >nul
)

echo.
echo ============================================================
echo   Starting CommuniGate ISL Application
echo ============================================================
echo.
echo 🎥 The app will open in your web browser automatically
echo    If it doesn't, manually go to: http://localhost:8501
echo.
echo 📱 Make sure your camera is connected and working
echo.
echo ⏹️  To stop the app: Close this window or press Ctrl+C
echo ============================================================
echo.

REM Kill any existing Streamlit instances
taskkill /F /IM streamlit.exe 2>nul >nul

REM Launch Streamlit
streamlit run src\ui\app.py

echo.
echo ============================================================
echo   CommuniGate ISL stopped
echo ============================================================
echo.
pause
