@echo off
REM ============================================================
REM   CommuniGate ISL - Portable Launcher
REM   NO PYTHON INSTALLATION REQUIRED!
REM   Works on any Windows 10/11 computer
REM ============================================================

title CommuniGate ISL - Portable Edition

color 0A
echo.
echo ============================================================
echo   CommuniGate ISL - Indian Sign Language Recognition
echo   PORTABLE VERSION - No installation required!
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Find WinPython installation (looks for any WPy64-* folder)
set WINPYTHON_DIR=
for /d %%i in (WPy64-*) do set WINPYTHON_DIR=%%i

if "%WINPYTHON_DIR%"=="" (
    echo.
    echo ERROR: WinPython not found!
    echo.
    echo This portable package requires WinPython.
    echo Please make sure the WPy64-* folder exists here.
    echo.
    echo Download WinPython from: https://winpython.github.io/
    echo Then extract it to this folder.
    echo.
    pause
    exit /b 1
)

echo [OK] Found portable Python: %WINPYTHON_DIR%
echo.

REM Find Python executable inside WinPython
set PYTHON_EXE=
for /d %%i in (%WINPYTHON_DIR%\python-*) do set PYTHON_EXE=%%i\python.exe

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found!
    echo Expected location: %WINPYTHON_DIR%\python-*\python.exe
    echo.
    pause
    exit /b 1
)

echo [OK] Python executable found
"%PYTHON_EXE%" --version
echo.

REM Check if project folder exists
if not exist "CommuniGate_ISL\" (
    echo ERROR: CommuniGate_ISL folder not found!
    echo Please make sure the project folder is in the same location as this script.
    echo.
    pause
    exit /b 1
)

REM Navigate to project
cd CommuniGate_ISL

REM Install dependencies (first time only)
if not exist ".portable_setup_complete" (
    echo ============================================================
    echo   FIRST TIME SETUP
    echo   Installing dependencies... (3-5 minutes)
    echo ============================================================
    echo.
    
    "%PYTHON_EXE%" -m pip install --upgrade pip --quiet
    
    echo Installing required packages...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies!
        echo Please check your internet connection and try again.
        echo.
        pause
        exit /b 1
    )
    
    REM Create marker file
    echo Setup completed on %date% %time% > .portable_setup_complete
    
    echo.
    echo [OK] Installation complete!
    echo.
) else (
    echo [OK] Dependencies already installed
    echo.
)

REM Check for model
if not exist "models\saved\lstm_model.keras" (
    echo.
    echo ============================================================
    echo   WARNING: Trained model not found!
    echo ============================================================
    echo.
    echo The model file is missing at:
    echo   models\saved\lstm_model.keras
    echo.
    echo The app may not work properly without it.
    echo Make sure to include the trained model files!
    echo.
    echo Press any key to continue anyway, or Ctrl+C to cancel...
    pause >nul
    echo.
)

echo ============================================================
echo   Starting CommuniGate ISL Application
echo ============================================================
echo.
echo [*] The app will open in your web browser automatically
echo     If it doesn't open, go to: http://localhost:8501
echo.
echo [*] Make sure your webcam is connected and working
echo.
echo [*] To stop the app: Close this window or press Ctrl+C
echo.
echo ============================================================
echo.

REM Kill any existing Streamlit instances
taskkill /F /IM streamlit.exe 2>nul >nul

REM Launch Streamlit app
"%PYTHON_EXE%" -m streamlit run src\ui\app.py

echo.
echo ============================================================
echo   CommuniGate ISL stopped
echo ============================================================
echo.
pause
