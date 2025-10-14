# Sample Video Review Guide

## 📍 Where to Put Sample Videos

Place your sample videos here:
```
data/videos/sample_videos/
```

You can use any naming convention, for example:
- `phrase0_sample.mp4` (for phrase "Hi my name is Reet")
- `phrase1_test.mp4` (for phrase "How are you")
- `person1_phrase2.mp4` (for phrase "I am from Delhi")
- `test_video.mp4`
- etc.

## 🔍 How to Analyze Sample Videos

### Step 1: Place Sample Videos
```bash
# Copy your sample videos to the analysis folder
cp /path/to/your/sample.mp4 data/videos/sample_videos/
```

### Step 2: Run Analysis Script
```bash
python src/data_collection/analyze_samples.py
```

This will provide detailed feedback on:
- ✓ Video duration (should be 2-4 seconds)
- ✓ Hand detection quality (>80% of frames)
- ✓ Lighting conditions (brightness levels)
- ✓ Hand visibility (both hands when needed)
- ✓ Frame composition (hands not going off-screen)
- ✓ Overall quality score (0-100)

### Step 3: Review Output

The script will tell you:
- **Quality Score**: EXCELLENT (80+) / GOOD (60-80) / ACCEPTABLE (40-60) / NEEDS IMPROVEMENT (<40)
- **Issues Found**: Specific problems with the video
- **Suggestions**: How to improve for final dataset

---

## ✅ What We're Looking For

### Ideal Sample Video Characteristics:

**Duration:**
- ✅ 2-4 seconds long
- ❌ Too short (<2s): Phrase might be rushed
- ❌ Too long (>5s): Unnecessary pauses

**Hand Detection:**
- ✅ Hands visible in >80% of frames
- ✅ Both hands visible when phrase requires it
- ❌ Hands frequently disappear
- ❌ Only one hand visible when both needed

**Lighting:**
- ✅ Brightness: 80-200 (on 0-255 scale)
- ✅ Even lighting on hands
- ❌ Too dark (<80): Hard to see details
- ❌ Too bright (>200): Overexposed, washed out

**Composition:**
- ✅ Hands stay centered in frame
- ✅ Waist-up or chest-up view
- ✅ Plain background
- ❌ Hands touching frame edges
- ❌ Hands going off-screen
- ❌ Busy/distracting background

**Performance:**
- ✅ Natural signing pace
- ✅ Clear, deliberate movements
- ✅ Complete phrase performed
- ❌ Rushed or too slow
- ❌ Unclear hand shapes

---

## 📊 Updated Project Info

### New Phrases (5 total):

| # | Phrase | Words |
|---|--------|-------|
| 0 | "Hi my name is Reet" | 5 words |
| 1 | "How are you" | 3 words |
| 2 | "I am from Delhi" | 4 words |
| 3 | "I like coffee" | 3 words |
| 4 | "What do you like" | 4 words |

### Updated Data Requirements:

- **Total videos needed**: 250 (was 200)
- **Per phrase**: 50 videos
- **5 people × 10 recordings × 5 phrases** = 250 videos

---

## 🎬 Sample Video Workflow

### 1. Record Sample Videos
Record 1-2 sample videos for each phrase (5-10 samples total)

### 2. Place in Analysis Folder
```bash
cp sample*.mp4 data/videos/sample_videos/
```

### 3. Analyze Quality
```bash
python src/data_collection/analyze_samples.py
```

### 4. Review Feedback
Read the output carefully:
- Check quality scores
- Note any issues
- Follow suggestions

### 5. Improve Setup
Based on feedback:
- Adjust lighting
- Reposition camera
- Improve background
- Adjust signing speed

### 6. Test Again
Record new samples with improvements and re-analyze

### 7. Once Satisfied (Score >70)
Proceed with full dataset recording!

---

## 💡 Common Issues & Solutions

### Issue: "Low hand detection rate"
**Solution:**
- Ensure hands are clearly visible
- Improve lighting specifically on hands
- Avoid wearing clothes similar to background color
- Don't move hands too fast (motion blur)

### Issue: "Video too dark"
**Solution:**
- Record near window during daytime
- Add artificial lighting (desk lamps)
- Increase camera exposure if possible
- Avoid backlighting (light behind person)

### Issue: "Hands frequently off-screen"
**Solution:**
- Step back from camera
- Use wider camera angle
- Keep hands more centered
- Reduce gesture size slightly

### Issue: "Video too short/long"
**Solution:**
- Practice phrase timing
- Count in your head (2-3 seconds)
- Don't rush or pause unnecessarily
- Trim video if needed

---

## 📋 Pre-Recording Checklist

Before recording final dataset, ensure samples show:
- [ ] Quality score >70/100
- [ ] Duration consistently 2-4 seconds
- [ ] Hand detection >80%
- [ ] Good lighting (brightness 80-200)
- [ ] Hands stay in frame
- [ ] Clear, smooth hand movements
- [ ] Plain, uncluttered background
- [ ] Stable camera (no shaking)

---

## 🚀 Next Steps After Sample Review

1. **Analyze your samples**: Run the analysis script
2. **Address issues**: Fix any problems found
3. **Test improvements**: Record new samples
4. **Iterate**: Keep improving until quality >70
5. **Share guidelines**: Send updated instructions to contributors
6. **Record dataset**: Collect all 250 videos
7. **Process videos**: Run `process_videos.py`
8. **Train model**: Run `train_sequence_model.py`

---

## 📞 Quick Commands

```bash
# Place sample videos
cp /path/to/samples/*.mp4 data/videos/sample_videos/

# Analyze samples
python src/data_collection/analyze_samples.py

# If satisfied, proceed to organize final videos:
# Phrase 0 → data/videos/phrase_0/
# Phrase 1 → data/videos/phrase_1/
# Phrase 2 → data/videos/phrase_2/
# Phrase 3 → data/videos/phrase_3/
# Phrase 4 → data/videos/phrase_4/

# Process all videos
python src/data_collection/process_videos.py

# Train model
python src/training/train_sequence_model.py
```

---

**Ready to analyze? Place your sample videos in `data/videos/sample_videos/` and run:**
```bash
python src/data_collection/analyze_samples.py
```
