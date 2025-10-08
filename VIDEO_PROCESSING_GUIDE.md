# Video Processing Guide - CommuniGate ISL

## 📋 Overview

This guide explains how to process pre-recorded MP4 videos from multiple contributors to create training data for the ISL recognition system.

---

## 🎯 Your Scenario

**Setup:**
- 5 people will record videos
- Each person performs each phrase 10 times
- Total: 5 people × 10 recordings × 4 phrases = **200 videos**

**Data distribution per phrase:**
- Phrase 0: 50 videos
- Phrase 1: 50 videos
- Phrase 2: 50 videos
- Phrase 3: 50 videos

This is **excellent** for model generalization across different people!

---

## 📁 Step 1: Organize Your Videos

### Directory Structure

Create the following folder structure:

```
CommuniGate_ISL/
└── data/
    └── videos/
        ├── phrase_0/
        │   ├── person1_take01.mp4
        │   ├── person1_take02.mp4
        │   ├── ...
        │   ├── person1_take10.mp4
        │   ├── person2_take01.mp4
        │   ├── ...
        │   └── person5_take10.mp4  (50 videos total)
        ├── phrase_1/
        │   └── (50 videos)
        ├── phrase_2/
        │   └── (50 videos)
        └── phrase_3/
            └── (50 videos)
```

### Create Directories

```bash
cd CommuniGate_ISL
mkdir -p data/videos/phrase_{0,1,2,3}
```

### Naming Convention (Recommended)

```
personX_takeYY.mp4
```

Examples:
- `person1_take01.mp4` - Person 1, first recording
- `person1_take02.mp4` - Person 1, second recording
- `person3_take07.mp4` - Person 3, seventh recording
- `person5_take10.mp4` - Person 5, tenth recording

**Alternative naming is fine too:**
- `john_video1.mp4`
- `sarah_attempt03.mp4`
- Any descriptive name works!

---

## 🎥 Step 2: Video Recording Guidelines

### Technical Requirements

- **Format**: MP4, MOV, or AVI
- **Duration**: 2-4 seconds per video
- **Frame Rate**: 30 fps (standard camera setting)
- **Resolution**: 720p or higher
- **File size**: Typically 5-20 MB per video

### Recording Setup

**Camera Position:**
- Front-facing camera
- Waist-up or chest-up view
- Both hands fully visible throughout
- Distance: ~2-3 feet from camera

**Environment:**
- Good, even lighting (avoid backlighting)
- Plain background (solid color preferred)
- Minimize shadows on hands
- Stable camera (not handheld shaking)

**Performance:**
- Perform all signs in the phrase sequentially
- Natural signing pace (not too fast, not too slow)
- Clear hand movements
- Keep hands in frame throughout
- Face the camera

### Quality Checklist

Before accepting a video, verify:
- [ ] Both hands are visible for entire duration
- [ ] Hands don't go off-screen
- [ ] Good lighting on hands
- [ ] No blur or motion artifacts
- [ ] Person performs complete phrase
- [ ] Video is 2-4 seconds long

---

## 🔧 Step 3: Process Videos

### Install Dependencies

```bash
pip install tqdm
```

### Run Processing Script

```bash
python src/data_collection/process_videos.py
```

### What This Does

The script will:
1. Read all MP4 files from `data/videos/phrase_X/`
2. Extract frames from each video
3. Detect hand landmarks using Mediapipe
4. Normalize all sequences to 60 frames
5. Save processed sequences as `.npy` files in `data/sequences/`

### Processing Options

**Option 1: Basic Processing (Faster)**
```
Select option (1 or 2): 1
```
- Extracts sequences only
- No preview videos created
- Faster processing

**Option 2: With Preview Videos (Recommended for first time)**
```
Select option (1 or 2): 2
```
- Extracts sequences
- Creates preview videos with landmarks drawn
- Helps verify quality
- Slower but useful for debugging

### Output

After processing, you'll have:

```
data/
├── videos/              # Original MP4s
│   └── phrase_0/
│       ├── person1_take01.mp4
│       └── person1_take01_preview.mp4  (if option 2)
└── sequences/           # Processed landmarks
    └── phrase_0/
        └── person1_take01_seq.npy
```

Each `.npy` file contains:
- Shape: `(60, 126)`
- 60 frames normalized from original video
- 126 features = 2 hands × 21 landmarks × 3 coordinates (x, y, z)

---

## 📊 Step 4: Verify Data Quality

### Check Sequence Counts

```bash
# Count sequences per phrase
ls data/sequences/phrase_0/*.npy | wc -l  # Should be ~50
ls data/sequences/phrase_1/*.npy | wc -l  # Should be ~50
ls data/sequences/phrase_2/*.npy | wc -l  # Should be ~50
ls data/sequences/phrase_3/*.npy | wc -l  # Should be ~50
```

### Review Processing Logs

Look for warnings in the output:
- **Low hand detection rate**: Video quality issue or hands not visible
- **Processing failed**: Corrupted video file or wrong format

### Check Preview Videos (if generated)

Open the `*_preview.mp4` files to verify:
- Landmarks are correctly detected on hands
- Tracking is stable throughout video
- No major detection gaps

---

## 🚀 Step 5: Train the Model

Once all videos are processed:

```bash
python src/training/train_sequence_model.py
```

Expected output:
- Loading ~200 sequences (50 per phrase)
- Training/validation/test split: 70%/15%/15%
- LSTM model training with progress bars
- Final accuracy metrics

With 50 samples per phrase from 5 different people, you should achieve:
- **Training accuracy**: 90-98%
- **Validation accuracy**: 85-95%
- **Test accuracy**: 80-90%

---

## 🎬 Step 6: Run the Application

```bash
streamlit run src/ui/app.py
```

The UI will:
- Access your webcam
- Detect hand landmarks in real-time
- Predict which phrase you're signing
- Display the recognized phrase

---

## ⚠️ Troubleshooting

### "No video files found"

**Problem**: Videos not in correct location

**Solution**:
```bash
# Check directory structure
ls -R data/videos/

# Should show:
# data/videos/phrase_0/person1_take01.mp4
# data/videos/phrase_1/person1_take01.mp4
# etc.
```

### "Low hand detection rate"

**Problem**: Hands not visible in video

**Solutions**:
- Re-record with better lighting
- Ensure hands stay in frame
- Check camera focus
- Avoid busy backgrounds

### "Import Error: No module named 'tqdm'"

**Solution**:
```bash
pip install tqdm
```

### "Video file won't open"

**Problem**: Unsupported format or corrupted file

**Solutions**:
- Convert to MP4: `ffmpeg -i input.mov -c copy output.mp4`
- Re-record the video
- Check file isn't corrupted

### "Sequence shape mismatch"

**Problem**: Processing created inconsistent shapes

**Solution**:
- Delete sequences: `rm -rf data/sequences/*`
- Re-run processing script
- Check all videos are valid

---

## 📈 Expected Timeline

| Task | Time Estimate |
|------|---------------|
| Record 200 videos (5 people) | 2-3 hours |
| Organize files | 15-30 minutes |
| Process videos | 20-30 minutes |
| Train model | 15-30 minutes |
| Test & deploy | 10-20 minutes |
| **Total** | **~4 hours** |

---

## ✅ Quality Assurance Checklist

Before training:
- [ ] All 4 phrase folders exist in `data/videos/`
- [ ] Each folder has ~50 MP4 files
- [ ] Total of ~200 video files
- [ ] All videos are 2-4 seconds long
- [ ] Sample videos show clear hand visibility
- [ ] Processing completed without major errors
- [ ] ~200 `.npy` files created in `data/sequences/`
- [ ] Spot-check: previews show accurate landmark tracking

---

## 📞 Next Steps After This Guide

1. **Collect/receive the 200 MP4 videos from your 5 contributors**
2. **Organize them** into the folder structure
3. **Run the processing script**
4. **Train the model**
5. **Test live recognition**
6. **Package as desktop app** (optional)

---

## 💡 Tips for Success

1. **Consistency**: All contributors should sign at similar pace
2. **Diversity**: Different hand sizes/skin tones helps generalization
3. **Variations**: Each take should have slight natural variations
4. **Quality over quantity**: 40 good videos > 50 poor quality videos
5. **Test early**: Process a few videos first to verify quality before collecting all 200

---

**Ready to process your videos? Run:**
```bash
python src/data_collection/process_videos.py
```
