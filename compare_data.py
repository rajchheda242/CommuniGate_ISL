#!/usr/bin/env python3
"""
Compare live captured data with training data to identify differences.
"""

import numpy as np
import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def compare_with_training_data(log_dir):
    """Compare captured diagnostic data with original training data."""
    print(f"🔍 Comparing diagnostic data with training data")
    print("="*60)
    
    # Load training data
    training_stats = {}
    
    for phrase_id in range(5):
        phrase_dir = f"data/sequences_holistic/phrase_{phrase_id}"
        if os.path.exists(phrase_dir):
            all_samples = []
            for file_path in glob.glob(os.path.join(phrase_dir, "*.npy")):
                try:
                    data = np.load(file_path)
                    all_samples.append(data)
                except:
                    continue
            
            if all_samples:
                combined = np.concatenate(all_samples, axis=0)
                training_stats[phrase_id] = {
                    'samples': len(all_samples),
                    'total_frames': combined.shape[0],
                    'mean': combined.mean(),
                    'std': combined.std(),
                    'min': combined.min(),
                    'max': combined.max(),
                    'shape': combined.shape
                }
    
    print("Training Data Summary:")
    for phrase_id, stats in training_stats.items():
        phrase_mapping = {
            0: "Hi my name is Reet",
            1: "How are you", 
            2: "I am from Delhi",
            3: "I like coffee",
            4: "What do you like"
        }
        print(f"  Phrase {phrase_id} ({phrase_mapping[phrase_id]}):")
        print(f"    Samples: {stats['samples']}, Frames: {stats['total_frames']}")
        print(f"    Range: {stats['min']:.3f} to {stats['max']:.3f}")
        print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # Load diagnostic data
    diagnostic_files = list(Path(log_dir).glob("landmarks_sample_*.json"))
    
    if not diagnostic_files:
        print("\n❌ No diagnostic landmark samples found!")
        print("Run the diagnostic system first with 'python diagnostic_system.py'")
        return
    
    print(f"\nFound {len(diagnostic_files)} diagnostic samples")
    
    # Analyze each diagnostic sample
    comparisons = []
    
    for diag_file in diagnostic_files:
        with open(diag_file, 'r') as f:
            diag_data = json.load(f)
        
        landmarks = np.array(diag_data['landmarks'])
        label = diag_data['label']
        
        # Find corresponding training phrase
        phrase_mapping_reverse = {
            "Hi my name is Reet": 0,
            "How are you": 1,
            "I am from Delhi": 2,
            "I like coffee": 3,
            "What do you like": 4,
            "unknown": -1
        }
        
        phrase_id = phrase_mapping_reverse.get(label, -1)
        
        comparison = {
            'file': diag_file.name,
            'label': label,
            'phrase_id': phrase_id,
            'diagnostic_stats': diag_data['stats'],
            'landmarks_shape': landmarks.shape
        }
        
        if phrase_id >= 0 and phrase_id in training_stats:
            training_stat = training_stats[phrase_id]
            
            # Calculate differences
            comparison['differences'] = {
                'mean_diff': abs(diag_data['stats']['mean'] - training_stat['mean']),
                'std_diff': abs(diag_data['stats']['std'] - training_stat['std']),
                'range_diff_min': abs(diag_data['stats']['min'] - training_stat['min']),
                'range_diff_max': abs(diag_data['stats']['max'] - training_stat['max']),
                'scale_factor': training_stat['std'] / diag_data['stats']['std'] if diag_data['stats']['std'] > 0 else float('inf')
            }
            
            print(f"\n📊 {diag_file.name}:")
            print(f"  Label: {label}")
            print(f"  Diagnostic: mean={diag_data['stats']['mean']:.3f}, std={diag_data['stats']['std']:.3f}")
            print(f"  Training:   mean={training_stat['mean']:.3f}, std={training_stat['std']:.3f}")
            print(f"  Differences:")
            print(f"    Mean diff: {comparison['differences']['mean_diff']:.3f}")
            print(f"    Std diff: {comparison['differences']['std_diff']:.3f}")
            print(f"    Scale factor: {comparison['differences']['scale_factor']:.3f}")
            
            # Flag significant differences
            if comparison['differences']['mean_diff'] > 0.5:
                print(f"    ⚠️  LARGE MEAN DIFFERENCE!")
            if comparison['differences']['scale_factor'] > 2 or comparison['differences']['scale_factor'] < 0.5:
                print(f"    ⚠️  LARGE SCALE DIFFERENCE!")
        
        comparisons.append(comparison)
    
    # Generate summary
    print(f"\n📋 SUMMARY:")
    
    # Check for consistent issues
    large_mean_diffs = [c for c in comparisons if c.get('differences', {}).get('mean_diff', 0) > 0.5]
    large_scale_diffs = [c for c in comparisons if c.get('differences', {}).get('scale_factor', 1) > 2 or c.get('differences', {}).get('scale_factor', 1) < 0.5]
    
    if large_mean_diffs:
        print(f"  ⚠️  {len(large_mean_diffs)} samples have large mean differences")
    
    if large_scale_diffs:
        print(f"  ⚠️  {len(large_scale_diffs)} samples have large scale differences")
    
    if not large_mean_diffs and not large_scale_diffs:
        print(f"  ✅ Data distributions look similar")
    else:
        print(f"\n💡 RECOMMENDATIONS:")
        if large_mean_diffs:
            print(f"  - Check coordinate system alignment")
            print(f"  - Verify MediaPipe preprocessing")
        if large_scale_diffs:
            print(f"  - Check normalization/scaling")
            print(f"  - Verify data preprocessing pipeline")
    
    # Save comparison report
    report_path = f"{log_dir}/training_comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'training_stats': {str(k): v for k, v in training_stats.items()},
            'diagnostic_comparisons': comparisons,
            'summary': {
                'total_diagnostic_samples': len(comparisons),
                'large_mean_differences': len(large_mean_diffs),
                'large_scale_differences': len(large_scale_diffs)
            }
        }, f, indent=2)
    
    print(f"\n📄 Full comparison report saved to: {report_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # Find most recent diagnostic log
        if os.path.exists("diagnostic_logs"):
            log_dirs = [d for d in os.listdir("diagnostic_logs") if os.path.isdir(f"diagnostic_logs/{d}")]
            if log_dirs:
                log_dir = f"diagnostic_logs/{sorted(log_dirs)[-1]}"
            else:
                print("No diagnostic logs found. Run diagnostic_system.py first.")
                sys.exit(1)
        else:
            print("No diagnostic_logs directory found. Run diagnostic_system.py first.")
            sys.exit(1)
    
    compare_with_training_data(log_dir)