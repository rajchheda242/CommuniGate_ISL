#!/usr/bin/env python3
"""
Analyze training data to understand potential gesture confusion.
This script examines the features of different phrases to identify similarities.
"""

import numpy as np
import json
import os
from pathlib import Path

def load_phrase_data():
    """Load all phrase data and analyze similarities."""
    
    # Load phrase mapping
    with open('models/saved/phrase_mapping.json', 'r') as f:
        phrase_mapping = json.load(f)
    
    print("📋 Phrase Analysis:")
    print("==================")
    for i, phrase in phrase_mapping.items():
        print(f"Class {i}: {phrase}")
    
    # Check if we have training data
    data_dirs = []
    for phrase_id in phrase_mapping.keys():
        phrase_dir = f"data/sequences/phrase_{phrase_id}"
        if os.path.exists(phrase_dir):
            data_dirs.append((phrase_id, phrase_dir, phrase_mapping[phrase_id]))
    
    if not data_dirs:
        print("\n❌ No training data found in data/sequences/")
        return
    
    print(f"\n📁 Found training data for {len(data_dirs)} phrases:")
    
    phrase_stats = {}
    
    for phrase_id, phrase_dir, phrase_text in data_dirs:
        files = list(Path(phrase_dir).glob("*.npy"))
        if files:
            print(f"\n🎯 Analyzing: {phrase_text}")
            print(f"   📁 {phrase_dir}")
            print(f"   📄 {len(files)} samples")
            
            # Load a sample to check dimensions
            sample = np.load(files[0])
            print(f"   📏 Shape: {sample.shape}")
            
            # Load all samples for this phrase
            all_samples = []
            for file in files[:5]:  # Analyze first 5 samples
                try:
                    data = np.load(file)
                    if data.shape[1] == 1662:  # Expected feature count
                        all_samples.append(data)
                    else:
                        print(f"   ⚠️  Skipping {file.name}: wrong shape {data.shape}")
                except Exception as e:
                    print(f"   ❌ Error loading {file.name}: {e}")
            
            if all_samples:
                # Calculate statistics
                all_data = np.concatenate(all_samples, axis=0)
                mean_features = np.mean(all_data, axis=0)
                std_features = np.std(all_data, axis=0)
                
                phrase_stats[phrase_id] = {
                    'phrase': phrase_text,
                    'samples': len(all_samples),
                    'total_frames': all_data.shape[0],
                    'mean_features': mean_features,
                    'std_features': std_features
                }
                
                print(f"   📊 Statistics:")
                print(f"      • Total frames analyzed: {all_data.shape[0]}")
                print(f"      • Mean feature range: {mean_features.min():.3f} to {mean_features.max():.3f}")
                print(f"      • Std feature range: {std_features.min():.3f} to {std_features.max():.3f}")
    
    # Analyze similarities between phrases
    if len(phrase_stats) >= 2:
        print(f"\n🔍 SIMILARITY ANALYSIS:")
        print("========================")
        
        phrase_ids = list(phrase_stats.keys())
        similarities = {}
        
        for i, id1 in enumerate(phrase_ids):
            for j, id2 in enumerate(phrase_ids):
                if i < j:  # Avoid duplicates
                    feat1 = phrase_stats[id1]['mean_features']
                    feat2 = phrase_stats[id2]['mean_features']
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(feat1, feat2)
                    norm1 = np.linalg.norm(feat1)
                    norm2 = np.linalg.norm(feat2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                        similarities[(id1, id2)] = similarity
                        
                        phrase1 = phrase_stats[id1]['phrase']
                        phrase2 = phrase_stats[id2]['phrase']
                        
                        print(f"\n'{phrase1}' vs '{phrase2}':")
                        print(f"   Cosine similarity: {similarity:.4f}")
                        
                        if similarity > 0.95:
                            print("   🔴 HIGH SIMILARITY - Potential confusion!")
                        elif similarity > 0.90:
                            print("   🟡 MODERATE SIMILARITY - May cause confusion")
                        else:
                            print("   🟢 Low similarity - Should be distinguishable")
        
        # Find most similar pair
        if similarities:
            most_similar = max(similarities.items(), key=lambda x: x[1])
            (id1, id2), max_sim = most_similar
            
            print(f"\n🎯 MOST CONFUSING PAIR:")
            print(f"   '{phrase_stats[id1]['phrase']}' vs '{phrase_stats[id2]['phrase']}'")
            print(f"   Similarity: {max_sim:.4f}")
            
            if max_sim > 0.90:
                print(f"\n💡 RECOMMENDATIONS:")
                print(f"   1. Collect more diverse training data for these phrases")
                print(f"   2. Focus on distinctive hand/pose differences")
                print(f"   3. Consider data augmentation to increase variation")
                print(f"   4. Review gesture execution for clear differences")

def main():
    """Main analysis function."""
    print("🔍 ISL Training Data Analysis")
    print("="*50)
    
    if not os.path.exists('models/saved/phrase_mapping.json'):
        print("❌ No phrase mapping found. Train the model first!")
        return
    
    load_phrase_data()
    
    print(f"\n💡 ABOUT THE CONFUSION:")
    print("If 'My name is Reet' is being predicted as 'I am from Delhi',")
    print("this suggests these gestures have similar patterns in the training data.")
    print("\nPossible causes:")
    print("• Similar hand movements or positions")
    print("• Similar body pose sequences") 
    print("• Insufficient training data variation")
    print("• Overlapping gesture components")

if __name__ == "__main__":
    main()