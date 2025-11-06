#!/usr/bin/env python3
"""
Fix the model configuration to match the actual training setup.
"""

import torch
import os

def fix_model_config():
    """Fix the saved model configuration."""
    model_path = "models/transformer/transformer_model.pth"
    
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return
    
    print("🔧 Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("Original config:", checkpoint['model_config'])
    
    # Fix the sequence length
    checkpoint['model_config']['sequence_length'] = 60  # Correct training length
    
    print("Updated config:", checkpoint['model_config'])
    
    # Save the corrected model
    torch.save(checkpoint, model_path)
    print("✅ Model configuration fixed!")
    
    # Also update the metadata file
    import json
    metadata_path = "models/transformer/training_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['sequence_length'] = 60
        metadata['note'] = "Configuration corrected: actual training used 60-frame sequences"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Metadata file updated!")

if __name__ == "__main__":
    fix_model_config()