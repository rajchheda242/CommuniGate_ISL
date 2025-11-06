# 📹 Recording Setup Checklist

## Quick Summary
Your 2 people need to record **20 sequences each** for **5 phrases** = **200 total videos**

---

## 🎯 Setup Requirements (Critical!)

### Background
- ✅ **Plain solid wall** (white, beige, gray, or light color)
- ❌ **NO busy patterns, posters, or cluttered backgrounds**
- ❌ **NO dark backgrounds** (reduces hand contrast)

### Lighting
- ✅ **Face a window** (natural light is best)
- ✅ **Room lights ON** (ceiling lights + lamps)
- ✅ **Light should fall on hands**, not create shadows
- ❌ **NO backlighting** (don't stand with window behind you)

### Camera Position
- ✅ **Stable mount** (tripod or phone stand)
- ✅ **Chest height** (camera at mid-torso level)
- ✅ **3-4 feet away** (one large step back)
- ✅ **Horizontal orientation** (landscape mode)
- ✅ **Frame includes: head to waist**

### Clothing
- ✅ **Solid colors** (any color is fine)
- ✅ **Long sleeves or short sleeves** (both work)
- ❌ **NO busy patterns or stripes on shirt**
- ❌ **NO jewelry on hands** (rings/bracelets can confuse detector)

---

## 🧪 Test Before Full Recording

### Step 1: Setup Test
```bash
./test_recording_setup.sh
```

This will:
1. Record 3 test sequences
2. Check quality automatically
3. Tell you if setup is good enough

### Step 2: Quality Check
- **Target**: < 15% zero frames (excellent)
- **Acceptable**: < 20% zero frames (good)
- **Need adjustment**: > 25% zero frames

If quality is poor, adjust:
1. Move closer to plain wall
2. Add more lighting
3. Make sure hands stay visible

---

## 📝 Recording Session Plan

### Per Person (2 hours each)
1. **Person A** records 20 sequences per phrase × 5 phrases = 100 videos
2. **Person B** records 20 sequences per phrase × 5 phrases = 100 videos

### The 5 Phrases
1. "Hi my name is Reet"
2. "How are you"
3. "I am from Delhi"
4. "I like coffee"
5. "What do you like"

### Tips for Recording
- Record at **natural speed** (not too fast, not too slow)
- Keep hands **in frame** throughout
- Perform gesture clearly
- Take **2-3 second break** between sequences
- Can rest after each phrase

---

## 🚀 Full Workflow

### Before Recording
```bash
# 1. Run setup test
./test_recording_setup.sh

# 2. If quality is good, proceed to full recording
```

### During Recording
```bash
# Person A records (phrases 0-4)
source .venv/bin/activate
python src/data_collection/collect_sequences.py
```

The script will:
- Show phrase on screen
- Detect hands via webcam
- Save each sequence automatically
- You just perform gesture 20 times per phrase

### After Recording (Both People Done)
```bash
# Check how many videos were recorded
ls -la data/videos/*/
```

Expected:
- **phrase_0/**: ~27 videos (7 old + 20 new from each person = 47 total)
- **phrase_1/**: ~40 videos
- **phrase_2/**: ~40 videos
- **phrase_3/**: ~40 videos
- **phrase_4/**: ~40 videos

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Setup + test | 15 mins |
| Person A records | 2 hours |
| Person B records | 2 hours |
| **Total** | **~4.5 hours** |

---

## ✅ Success Criteria

After recording, each phrase should have:
- **~80-90 total sequences** (old reprocessed + new recorded)
- **< 15% zero frames** on new recordings
- **Good variety** (2 different people)

This should give you **85-90% model accuracy** 🎯

---

## 🆘 Troubleshooting

### "Webcam not detected"
- Check camera permissions in System Preferences
- Try plugging/unplugging webcam
- Restart computer

### "High zero frame percentage (> 25%)"
1. **Lighting**: Add more light sources
2. **Background**: Move to plainer wall
3. **Distance**: Try 3 feet from camera
4. **Hands**: Keep hands in center of frame

### "Videos not saving"
- Check `data/videos/phrase_X/` directories exist
- Make sure you're in activated virtual environment
- Check disk space

---

## 📞 Questions?

Run the test first, see what quality you get, then we can adjust!

**Next command to run:**
```bash
./test_recording_setup.sh
```
