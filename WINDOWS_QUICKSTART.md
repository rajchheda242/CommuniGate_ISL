# 🚀 CommuniGate ISL - Quick Start for Windows

## 🎯 For Demo/Presentation - FASTEST SETUP

### **What You Need:**
1. ✅ Windows 10/11 laptop
2. ✅ Python 3.9 or 3.10 installed
3. ✅ Working webcam
4. ✅ Internet (first time only)

---

## 📦 **Option 1: Quick Launch (RECOMMENDED)**

### If you already have the project folder:

1. **Open the project folder**
2. **Double-click:** `launch.bat`
3. **Wait** (first time takes 2-5 minutes to install)
4. **Browser opens** with the app
5. **Start your demo!** 🎉

**That's it!** No other steps needed.

---

## 🆕 **Option 2: Fresh Install from GitHub**

### If starting from scratch:

**Step 1: Install Python** (if not installed)
- Download: https://www.python.org/downloads/
- Install Python 3.10.x
- ⚠️ **CHECK "Add Python to PATH"** during installation!

**Step 2: Open Command Prompt**
- Press `Win + R`
- Type `cmd` and press Enter

**Step 3: Copy-paste ALL these commands:**
```cmd
cd %USERPROFILE%\Desktop
git clone https://github.com/rajchheda242/CommuniGate_ISL.git
cd CommuniGate_ISL
launch.bat
```

**Done!** The app will start automatically.

---

## 📁 **Option 3: USB Drive Package**

### If you have the project on a USB drive:

1. **Copy folder** from USB to Desktop
2. **Open folder**
3. **Double-click:** `launch.bat`
4. **Wait for setup** (first time only)
5. **App launches!**

---

## 🎨 **Logo is Already Configured!**

Your logo automatically appears in:
- ✅ Browser tab (favicon)
- ✅ App header (top of page)

No additional setup needed!

---

## ⚡ **Troubleshooting**

### "Python is not recognized"
→ Install Python and check "Add to PATH"
→ Restart computer after installing

### "Model not found"
→ You need the trained model files in `models/saved/`
→ Copy from your Mac or train on Windows

### Camera not working
→ Grant camera permissions in Windows Settings
→ Check if other apps can use the camera
→ Try a different browser if webcam not detected

### Streamlit won't start
→ Open Command Prompt
→ Run: `taskkill /F /IM streamlit.exe`
→ Try again with `launch.bat`

---

## 📝 **For Your Demo**

### **Before the Presentation:**

1. ✅ Test the app 1 day before
2. ✅ Check camera works
3. ✅ Test all gestures
4. ✅ Keep a backup video recording
5. ✅ Arrive early to set up

### **Demo Day Checklist:**

- [ ] Laptop fully charged
- [ ] Camera connected and tested
- [ ] Good lighting
- [ ] Clean background
- [ ] App tested and working
- [ ] Backup plan ready (video recording)

---

## 🎯 **Quick Commands Reference**

### To start the app:
```cmd
cd path\to\CommuniGate_ISL
launch.bat
```

### To stop the app:
- Close the browser window
- Close the Command Prompt window
- Or press `Ctrl + C` in the terminal

### To restart:
```cmd
launch.bat
```

---

## 📞 **Need Help During Demo?**

### Quick fixes:

**App won't start:**
```cmd
taskkill /F /IM streamlit.exe
launch.bat
```

**Camera frozen:**
- Refresh browser (F5)
- Or restart app

**Low confidence predictions:**
- Better lighting
- Slower, clearer gestures
- Position camera at chest level

---

## 🎓 **For Judges/Evaluators**

This is a real-time Indian Sign Language recognition system that:
- Recognizes hand gestures using AI/ML
- Provides instant translation to text
- Works completely offline (after first setup)
- Uses computer vision and deep learning

**Tech Stack:**
- TensorFlow/Keras (LSTM model)
- MediaPipe (hand tracking)
- Streamlit (web interface)
- OpenCV (video processing)

---

## ✅ **Success Indicators**

You know everything is working when:
1. ✅ Browser opens with the app
2. ✅ Logo appears in header
3. ✅ Camera view shows up
4. ✅ Hand landmarks are drawn in real-time
5. ✅ Gestures are recognized with confidence scores

---

## 🚀 **You're All Set!**

Just run `launch.bat` and you're ready to demo!

**Good luck with your presentation!** 🎉
