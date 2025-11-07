"""
Windows EXE Builder for CommuniGate ISL
Run this script to create a standalone executable

Note: Creating Streamlit as EXE is challenging. 
For demos, the portable launcher (LAUNCH_APP.bat) is more reliable.
"""
import PyInstaller.__main__
import os
import shutil
import sys

print("=" * 70)
print("  CommuniGate ISL - Windows EXE Builder")
print("=" * 70)
print()
print("⚠️  WARNING: Streamlit EXEs can be unreliable!")
print("   Consider using the portable launcher (LAUNCH_APP.bat) instead.")
print("   It's more reliable for demos and smaller in size.")
print()
response = input("Continue with EXE build anyway? (y/n): ")
if response.lower() != 'y':
    print("✅ Using portable launcher is the recommended approach!")
    sys.exit(0)

# Clean previous builds
print("\n🧹 Cleaning previous builds...")
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('build'):
    shutil.rmtree('build')
if os.path.exists('CommuniGateISL.spec'):
    os.remove('CommuniGateISL.spec')

# Check if model exists
if not os.path.exists('models/saved/lstm_model.keras'):
    print("\n⚠️  WARNING: Model not found at models/saved/lstm_model.keras")
    print("   The EXE will not work without a trained model.")
    response = input("   Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

print("\n📦 Building executable...")
print("   This will take 5-15 minutes...")
print("   File size will be 500MB - 1.5GB")
print()

# Build configuration
try:
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
        '--hidden-import=streamlit.runtime',
        '--hidden-import=streamlit.runtime.scriptrunner',
        '--hidden-import=tensorflow',
        '--hidden-import=mediapipe',
        '--hidden-import=cv2',
        '--hidden-import=sklearn',
        '--hidden-import=joblib',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--collect-all=streamlit',
        '--collect-all=mediapipe',
        '--collect-all=altair',
        '--icon=src/ui/assets/icon.png',    # App icon
    ])
    
    print()
    print("=" * 70)
    print("✅ BUILD COMPLETE!")
    print("=" * 70)
    print()
    print("📍 Your executable is here:")
    exe_path = os.path.abspath('dist/CommuniGateISL.exe')
    print(f"   {exe_path}")
    print()
    
    # Check file size
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.1f} MB")
        print()
    
    print("⚠️  IMPORTANT NOTES:")
    print("   1. Test the EXE thoroughly before your demo")
    print("   2. The EXE may be flagged by antivirus (false positive)")
    print("   3. Camera permissions may need to be granted")
    print("   4. First launch may be slow (30-60 seconds)")
    print()
    print("📝 To run:")
    print("   - Double-click CommuniGateISL.exe")
    print("   - Wait for browser to open")
    print("   - Grant camera permissions if asked")
    print()
    print("🔄 Backup plan:")
    print("   Keep LAUNCH_APP.bat as a backup!")
    print()
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ BUILD FAILED!")
    print("=" * 70)
    print(f"\nError: {e}")
    print()
    print("💡 Try using the portable launcher instead:")
    print("   Just use LAUNCH_APP.bat - it's more reliable!")
    print()
    sys.exit(1)
