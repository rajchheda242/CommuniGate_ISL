# Quick Start Guide - MP4 Video Processing Workflow

## 🎬 For Your Use Case: 5 People, 10 Videos Each

### Overview
You have 5 people who will each record 10 videos for each of the 4 phrases.
- **Total videos**: 200 (5 people × 10 recordings × 4 phrases)
- **Per phrase**: 50 videos
- **Format**: MP4 files

---

## 📋 Workflow Steps

### Step 1: Organize Videos 📁

Place your MP4 files in this structure:

```
data/videos/
├── phrase_0/    (Hi, my name is Madiha Siddiqui)
│   ├── person1_video01.mp4
│   ├── person1_video02.mp4
│   ├── ...
│   ├── person1_video10.mp4
│   ├── person2_video01.mp4
│   ├── ...
│   └── person5_video10.mp4    [50 videos total]
│
├── phrase_1/    (I am a student)
│   └── [50 videos]
│
├── phrase_2/    (I enjoy running as a hobby)
│   └── [50 videos]
│
└── phrase_3/    (How are you doing today?)
    └── [50 videos]
```

**Note**: Naming is flexible! Use any descriptive names.

---

### Step 2: Install Required Package 📦

```bash
pip install tqdm
```

---

### Step 3: Process All Videos 🔧

```bash
python src/data_collection/process_videos.py
```

**What it does:**
- Reads all MP4 files from `data/videos/`
- Extracts hand landmarks from each frame
- Normalizes to 60-frame sequences
- Saves as `.npy` files in `data/sequences/`

**Time estimate**: ~20-30 minutes for 200 videos

---

### Step 4: Verify Output ✅

Check that sequences were created:

```bash
# Count sequences per phrase (should be ~50 each)
ls data/sequences/phrase_0/*.npy | wc -l
ls data/sequences/phrase_1/*.npy | wc -l
ls data/sequences/phrase_2/*.npy | wc -l
ls data/sequences/phrase_3/*.npy | wc -l
```

Expected: ~50 files in each folder

---

### Step 5: Train the Model 🤖

```bash
python src/training/train_sequence_model.py
```

**What it does:**
- Loads all 200 sequences
- Splits into train/validation/test sets
- Trains LSTM neural network
- Evaluates accuracy
- Saves trained model to `models/saved/`

**Time estimate**: ~15-30 minutes

**Expected accuracy**: 80-95% with 50 samples per phrase

---

### Step 6: Run Live Recognition 🎥

```bash
streamlit run src/ui/app.py
```

Opens a web interface where you can:
- See live webcam feed
- Perform ISL phrases
- Get real-time recognition
- See predicted phrases

---

## 📸 Video Recording Guidelines for Contributors

### Technical Specs
- **Duration**: 2-4 seconds
- **Format**: MP4 (preferred), MOV, or AVI
- **Frame rate**: 30 fps (standard)
- **Resolution**: 720p or higher

### Setup
- **Camera**: Front-facing, waist-up view
- **Distance**: 2-3 feet from camera
- **Lighting**: Bright, even lighting (avoid backlighting)
- **Background**: Plain/solid color preferred
- **Hands**: Both hands fully visible throughout

### Performance
- Perform all words in the phrase sequentially
- Natural signing pace (not rushed)
- Keep hands in frame the entire time
- Face the camera
- Clear, deliberate movements

### Quality Checklist
Before accepting a video:
- [ ] 2-4 seconds long
- [ ] Both hands visible entire time
- [ ] Good lighting
- [ ] No blur
- [ ] Complete phrase performed
- [ ] Hands don't go off-screen

---

## 🎯 Expected Results

With your dataset:
- **200 total videos** (50 per phrase)
- **5 different people** = good diversity
- **10 samples per person** = good coverage

You should achieve:
- ✅ Training accuracy: 90-98%
- ✅ Validation accuracy: 85-95%
- ✅ Test accuracy: 80-90%
- ✅ Real-time recognition that works well

---

## ⚡ Quick Command Reference

```bash
# 1. Create directories
mkdir -p data/videos/phrase_{0,1,2,3}

# 2. Place your MP4s in the folders (manually)

# 3. Install dependencies
pip install tqdm tensorflow

# 4. Process videos
python src/data_collection/process_videos.py

# 5. Train model
python src/training/train_sequence_model.py

# 6. Run app
streamlit run src/ui/app.py
```

---

## 🔍 Troubleshooting

**"No video files found"**
- Check files are in correct folders: `data/videos/phrase_X/`
- Verify file extensions: `.mp4`, `.MP4`, `.mov`, `.avi`

**"Low hand detection rate"**
- Video quality issue
- Re-record with better lighting
- Ensure hands stay in frame

**"Import error: tqdm"**
- Run: `pip install tqdm`

**"Import error: tensorflow"**
- Run: `pip install tensorflow`

---

## 📞 Support

For detailed information:
- **VIDEO_PROCESSING_GUIDE.md** - Complete video processing guide
- **README.md** - Project overview
- **ROADMAP.md** - Development plan

---

**Ready to process? Place your MP4s and run:**
```bash
python src/data_collection/process_videos.py
```
