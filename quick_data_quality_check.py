#!/usr/bin/env python3
"""
Quick data quality checker for ISL sequences.
Identifies problematic sequences that should be re-recorded.
"""

import numpy as np
import glob
import os
from pathlib import Path

DATA_DIR = "data/sequences"
PHRASES = [
    "Hi my name is Reet",
    "How are you",
    "I am from Delhi",
    "I like coffee",
    "What do you like"
]

def check_sequence_quality(sequence_path):
    """Check a single sequence for quality issues."""
    seq = np.load(sequence_path)
    issues = []
    
    # Check for zero frames (no hands detected)
    zero_frames = np.all(seq == 0, axis=1).sum()
    zero_ratio = zero_frames / len(seq)
    
    if zero_ratio > 0.15:
        issues.append(f"Too many zero frames: {zero_ratio:.1%}")
    
    # Check for suspicious low variance (static hands)
    frame_variance = np.var(seq, axis=0).mean()
    if frame_variance < 0.001:
        issues.append(f"Very low variance: {frame_variance:.6f} (possibly static)")
    
    # Check for outliers
    if seq.max() > 2.0 or seq.min() < -0.5:
        issues.append(f"Unusual values: min={seq.min():.3f}, max={seq.max():.3f}")
    
    return {
        'zero_frames': zero_frames,
        'zero_ratio': zero_ratio,
        'variance': frame_variance,
        'min': seq.min(),
        'max': seq.max(),
        'mean': seq.mean(),
        'issues': issues
    }

def main():
    print("="*80)
    print("ISL SEQUENCE DATA QUALITY REPORT")
    print("="*80)
    
    total_sequences = 0
    total_bad = 0
    
    for phrase_idx in range(len(PHRASES)):
        phrase_dir = os.path.join(DATA_DIR, f"phrase_{phrase_idx}")
        
        if not os.path.exists(phrase_dir):
            print(f"\n❌ Phrase {phrase_idx}: Directory not found!")
            continue
        
        sequence_files = sorted(glob.glob(os.path.join(phrase_dir, "*_seq.npy")))
        
        if not sequence_files:
            print(f"\n❌ Phrase {phrase_idx}: No sequences found!")
            continue
        
        print(f"\n{'='*80}")
        print(f"Phrase {phrase_idx}: {PHRASES[phrase_idx]}")
        print(f"{'='*80}")
        print(f"Total sequences: {len(sequence_files)}")
        
        # Analyze each sequence
        bad_sequences = []
        stats = {
            'zero_frames': [],
            'variance': [],
            'min': [],
            'max': []
        }
        
        for seq_file in sequence_files:
            total_sequences += 1
            quality = check_sequence_quality(seq_file)
            
            stats['zero_frames'].append(quality['zero_frames'])
            stats['variance'].append(quality['variance'])
            stats['min'].append(quality['min'])
            stats['max'].append(quality['max'])
            
            if quality['issues']:
                bad_sequences.append((Path(seq_file).name, quality))
                total_bad += 1
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"  Avg zero frames: {np.mean(stats['zero_frames']):.1f} / 90")
        print(f"  Avg variance: {np.mean(stats['variance']):.6f}")
        print(f"  Value range: [{np.min(stats['min']):.3f}, {np.max(stats['max']):.3f}]")
        
        # Print bad sequences
        if bad_sequences:
            print(f"\n⚠️  Found {len(bad_sequences)} problematic sequences:")
            for filename, quality in bad_sequences:
                print(f"\n  📁 {filename}")
                for issue in quality['issues']:
                    print(f"     - {issue}")
                print(f"     Stats: {quality['zero_frames']} zero frames, "
                      f"variance={quality['variance']:.6f}")
        else:
            print(f"\n✅ All sequences look good!")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total sequences checked: {total_sequences}")
    print(f"Problematic sequences: {total_bad} ({total_bad/max(total_sequences, 1)*100:.1f}%)")
    
    if total_bad > 0:
        print(f"\n💡 Recommendation: Re-record the {total_bad} problematic sequences")
        print("   or delete them before training to improve model quality.")
    else:
        print("\n✅ Dataset quality looks good!")
    
    # Data balance check
    print(f"\n{'='*80}")
    print("DATA BALANCE CHECK")
    print(f"{'='*80}")
    
    counts = []
    for phrase_idx in range(len(PHRASES)):
        phrase_dir = os.path.join(DATA_DIR, f"phrase_{phrase_idx}")
        if os.path.exists(phrase_dir):
            count = len(glob.glob(os.path.join(phrase_dir, "*_seq.npy")))
            counts.append(count)
            print(f"Phrase {phrase_idx}: {count:3d} sequences - {PHRASES[phrase_idx]}")
        else:
            counts.append(0)
            print(f"Phrase {phrase_idx}:   0 sequences - {PHRASES[phrase_idx]} ❌")
    
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        imbalance = (max_count - min_count) / max(max_count, 1) * 100
        
        print(f"\nBalance: min={min_count}, max={max_count}, imbalance={imbalance:.1f}%")
        
        if imbalance > 20:
            print(f"⚠️  WARNING: Dataset is imbalanced!")
            print(f"   Recommendation: Record more sequences for classes with <{max_count} samples")
        else:
            print("✅ Dataset is reasonably balanced")
        
        if max_count < 100:
            print(f"\n⚠️  WARNING: Only {max_count} sequences per class")
            print(f"   Recommendation: Collect at least 100 sequences per phrase for robust model")

if __name__ == "__main__":
    main()
