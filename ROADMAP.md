# Indian Sign Language (ISL) MVP – Project Roadmap

## 🎯 Goal
Build a **desktop MVP application** (Windows/Mac) that recognizes **4 fixed ISL phrases** using a webcam, and outputs the corresponding text (with optional text-to-speech).  
This is an **academic project** for pitching purposes, not a production system.

---

## 🛠 Tech Stack
- **Programming Language**: Python 3.10+
- **Libraries**:
  - [OpenCV](https://opencv.org/) → Webcam input and image processing
  - [Mediapipe](https://developers.google.com/mediapipe) → Hand landmark detection
  - [scikit-learn](https://scikit-learn.org/) → Lightweight classifier (KNN/SVM)
  - [Streamlit](https://streamlit.io/) → Basic UI
  - [pyttsx3](https://pypi.org/project/pyttsx3/) → Text-to-speech (optional)
- **Packaging**: PyInstaller / auto-py-to-exe → `.exe` (Windows) / `.app` (Mac)

---

## 📋 Deliverables
1. Webcam integration with OpenCV  
2. Hand landmark detection using Mediapipe  
3. Training small classifier on 4 gestures  
4. Streamlit-based UI to show output  
5. Optional: Text-to-speech integration  
6. Packaged desktop application (.exe / .app)  
7. Usage documentation  

---

## 🗂 Fixed Phrases
1. "Hi, my name is Madiha Siddiqui."  
2. "I am a student."  
3. "I enjoy running as a hobby."  
4. "How are you doing today?"  

---

## 🗺️ Roadmap & Steps

### **Phase 1: Setup (Week 1)**
- [ ] Install dependencies (`opencv-python`, `mediapipe`, `scikit-learn`, `streamlit`, `pyttsx3`)  
- [ ] Create Git repository and roadmap file  
- [ ] Build simple webcam test script with OpenCV  
- [ ] Integrate Mediapipe to display hand landmarks  

### **Phase 2: Data Collection (Week 1–2)**
- [ ] Define gesture for each phrase  
- [ ] Capture ~30–50 samples per phrase (landmark coordinates)  
- [ ] Save data in `.csv` format (label + landmarks)  

### **Phase 3: Model Training (Week 2)**
- [ ] Normalize landmark data (scale/position invariant)  
- [ ] Train a simple classifier (KNN or SVM)  
- [ ] Test model offline with sample data  
- [ ] Validate recognition for 4 phrases  

### **Phase 4: Live Prediction Integration (Week 3)**
- [ ] Connect trained model with live webcam feed  
- [ ] Display predicted phrase in terminal/logs  
- [ ] Fine-tune for accuracy/stability  

### **Phase 5: UI Development (Week 3)**
- [ ] Build basic Streamlit interface  
- [ ] Show live camera feed in UI  
- [ ] Display recognized phrase as text in real time  
- [ ] Add text-to-speech option (toggle)  

### **Phase 6: Packaging & Finalization (Week 4)**
- [ ] Package into `.exe` (Windows) and `.app` (Mac) using PyInstaller  
- [ ] Test on fresh machine without Python installed  
- [ ] Add basic usage guide (README.md)  

---

## ✅ Final Output
- Standalone desktop app with:  
  - Live webcam input  
  - Recognition of 4 phrases  
  - Text display of recognized phrase  
  - Optional spoken output  
- Documentation (README + roadmap)  

---
