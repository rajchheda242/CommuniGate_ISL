#!/usr/bin/env python3
"""
Recording Helper - Shows exactly what to record
"""

print("="*80)
print("🎬 RECORDING SESSION GUIDE")
print("="*80)
print()
print("You need to record 73 NEW sequences total")
print()
print("SKIP Phrase 0 - Already have 46 good sequences! ✅")
print()

phrases = {
    1: {"name": "How are you", "need": 16},
    2: {"name": "I am from Delhi", "need": 20},
    3: {"name": "I like coffee", "need": 18},
    4: {"name": "What do you like", "need": 19}
}

print("-"*80)
print("Recording Breakdown:")
print("-"*80)

total = 0
for phrase_num, info in phrases.items():
    print(f"Phrase {phrase_num}: '{info['name']}' - {info['need']} sequences")
    total += info['need']

print(f"\nTOTAL: {total} sequences")
print()

print("="*80)
print("⏱️  TIME ESTIMATE")
print("="*80)
print()
print("Option A: One person records all")
print(f"  • Total sequences: {total}")
print(f"  • Time per sequence: ~60 seconds (including breaks)")
print(f"  • Total time: ~{total} minutes = {total/60:.1f} hours")
print()
print("Option B: Two people split")
print(f"  • Person 1: {total//2} sequences (~{(total//2)/60:.1f} hours)")
print(f"  • Person 2: {total - total//2} sequences (~{(total - total//2)/60:.1f} hours)")
print()

print("="*80)
print("📋 RECORDING CHECKLIST")
print("="*80)
print()
print("Before starting:")
print("  1. ✅ Plain wall background")
print("  2. ✅ Good lighting (face window + room lights)")
print("  3. ✅ Camera at chest height, 3-4 feet away")
print("  4. ✅ Solid color shirt (no patterns)")
print("  5. ✅ Test setup first: ./test_recording_setup.sh")
print()
print("During recording:")
print("  • Perform gesture at natural speed")
print("  • Keep hands in frame")
print("  • Take 2-3 second break between sequences")
print("  • Press 'q' to quit when done")
print()

print("="*80)
print("🚀 READY TO START?")
print("="*80)
print()
print("Run these commands:")
print()
print("  # 1. Test setup quality")
print("  ./test_recording_setup.sh")
print()
print("  # 2. If test shows < 20% zero frames, start recording!")
print("  python src/data_collection/collect_sequences.py")
print()
print("="*80)
