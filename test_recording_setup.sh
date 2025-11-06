#!/bin/bash
# Quick Recording Setup Test Script
# Run this to test if your setup is good before recording all 200 videos

echo "========================================================================"
echo "RECORDING SETUP TEST"
echo "========================================================================"
echo ""
echo "This will help you test your recording setup before the full session."
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment is active"
else
    echo "⚠ Activating virtual environment..."
    source .venv/bin/activate
fi

echo ""
echo "========================================================================"
echo "STEP 1: Camera & Background Check"
echo "========================================================================"
echo ""
echo "Before recording, ensure:"
echo "  ✓ Plain wall background (white/beige/gray)"
echo "  ✓ Camera stable at chest height"
echo "  ✓ Good lighting (window + room lights)"
echo "  ✓ 3-4 feet from camera"
echo ""
read -p "Setup ready? Press ENTER to record 3 TEST videos..."

echo ""
echo "Recording 3 test sequences for Phrase 0..."
echo "Press 'q' after 3 sequences to quit"
echo ""
python3 << 'EOF'
import subprocess
import sys

# Just open the collection script
# User will manually record 3 sequences and press 'q'
subprocess.run([sys.executable, "src/data_collection/collect_sequences.py"])
EOF

echo ""
echo "========================================================================"
echo "STEP 2: Quality Check"
echo "========================================================================"
echo ""
echo "Checking quality of test recordings..."
echo ""

python3 << 'EOF'
import numpy as np
import glob
import os

phrase_dir = "data/sequences/phrase_0"
files = sorted(glob.glob(os.path.join(phrase_dir, "*_seq.npy")))

# Get last 3 files (the test recordings)
test_files = files[-3:] if len(files) >= 3 else files

print(f"Found {len(test_files)} test sequences\n")

if not test_files:
    print("❌ No sequences found! Recording may have failed.")
    print("   Check that webcam is working.")
    exit(1)

zero_frame_counts = []
for f in test_files:
    seq = np.load(f)
    zero_frames = np.all(seq == 0, axis=1).sum()
    zero_pct = zero_frames / len(seq) * 100
    
    filename = os.path.basename(f)
    print(f"📁 {filename}")
    print(f"   Zero frames: {zero_frames}/90 ({zero_pct:.1f}%)")
    
    if zero_pct < 10:
        print(f"   ✅ EXCELLENT quality!")
    elif zero_pct < 20:
        print(f"   ✅ GOOD quality")
    elif zero_pct < 30:
        print(f"   ⚠️  FAIR quality - could improve lighting/background")
    else:
        print(f"   ❌ POOR quality - adjust setup before continuing")
    
    print()
    zero_frame_counts.append(zero_pct)

avg_zero = np.mean(zero_frame_counts)
print("="*70)
print(f"Average zero frames: {avg_zero:.1f}%")
print("="*70)
print()

if avg_zero < 15:
    print("🎉 EXCELLENT! Your setup is perfect!")
    print("   → Proceed with full recording (20 sequences per person)")
elif avg_zero < 25:
    print("✅ GOOD! This will work.")
    print("   → You can proceed, or adjust for even better quality")
else:
    print("⚠️  NEEDS IMPROVEMENT")
    print()
    print("Recommendations:")
    print("  1. Move to plainer background (solid wall)")
    print("  2. Add more light (face window or turn on more lights)")
    print("  3. Make sure hands don't drop out of frame")
    print("  4. Record 3 more test videos with adjustments")
    print()
    print("   → Fix setup, then run this test again")

EOF

echo ""
echo "========================================================================"
echo "NEXT STEPS"
echo "========================================================================"
echo ""
echo "If quality is GOOD (< 20% zero frames):"
echo "  → Run full recording session:"
echo "     python src/data_collection/collect_sequences.py"
echo ""
echo "If quality is POOR (> 25% zero frames):"
echo "  → Adjust setup and run this test again:"
echo "     bash test_recording_setup.sh"
echo ""
echo "Full recording session (per person):"
echo "  - 20 sequences per phrase"
echo "  - 5 phrases total"
echo "  - ~2 hours per person"
echo ""
