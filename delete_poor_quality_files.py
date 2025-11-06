#!/usr/bin/env python3
"""
Delete poor quality sequence files (>25% zero frames)
Creates backup before deletion
"""

import numpy as np
import glob
import os
import shutil
from datetime import datetime

def backup_and_delete():
    """Identify poor quality files, backup, and delete them"""
    
    print("="*80)
    print("DELETING POOR QUALITY FILES")
    print("="*80)
    
    # Create backup directory
    backup_dir = f"data/sequences_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    print(f"\n✅ Created backup directory: {backup_dir}\n")
    
    total_deleted = 0
    
    for phrase_idx in range(5):
        phrase_dir = f"data/sequences/phrase_{phrase_idx}"
        backup_phrase_dir = os.path.join(backup_dir, f"phrase_{phrase_idx}")
        os.makedirs(backup_phrase_dir, exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(phrase_dir, "*_seq.npy")))
        
        deleted_count = 0
        files_to_delete = []
        
        # Identify files to delete (>25% zero frames)
        for f in files:
            seq = np.load(f)
            zero_frames = np.all(seq == 0, axis=1).sum()
            zero_pct = zero_frames / 90 * 100
            
            if zero_pct > 25:  # Delete threshold
                files_to_delete.append((f, zero_frames, zero_pct))
        
        if files_to_delete:
            print(f"Phrase {phrase_idx}: Deleting {len(files_to_delete)} files")
            print("-" * 80)
            
            for filepath, zero_frames, zero_pct in files_to_delete:
                filename = os.path.basename(filepath)
                
                # Backup
                backup_path = os.path.join(backup_phrase_dir, filename)
                shutil.copy2(filepath, backup_path)
                
                # Delete
                os.remove(filepath)
                
                print(f"  ❌ Deleted: {filename} ({zero_frames} zero frames, {zero_pct:.1f}%)")
                deleted_count += 1
                total_deleted += 1
            
            print(f"  ✅ Deleted {deleted_count} files from phrase {phrase_idx}")
            print()
    
    print("="*80)
    print(f"DELETION COMPLETE")
    print("="*80)
    print(f"Total files deleted: {total_deleted}")
    print(f"Backup location: {backup_dir}")
    print()
    print("If you need to restore, copy files from backup directory back to data/sequences/")
    print("="*80)

if __name__ == "__main__":
    # Confirmation
    print("\n⚠️  WARNING: This will DELETE 96 poor quality files!")
    print("   (Backup will be created first)")
    print()
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response == 'yes':
        backup_and_delete()
        print("\n✅ Done! Run quality check again to verify:")
        print("   python quick_data_quality_check.py")
    else:
        print("\n❌ Cancelled. No files were deleted.")
