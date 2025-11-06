#!/usr/bin/env python3
"""
Quick diagnostic to understand why the model always predicts "I am from Delhi"
"""

import torch
import numpy as np
import os

def diagnose_model():
    """Diagnose the model predictions."""
    print("🔍 Model Diagnosis - Why always 'I am from Delhi'?")
    print("="*60)
    
    # Load the model
    model_path = "models/transformer/transformer_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("✅ Model loaded successfully")
    
    # Check model weights in the final classifier
    model_config = checkpoint['model_config']
    print(f"Model config: {model_config}")
    
    # Look at the final classification layer weights
    state_dict = checkpoint['model_state_dict']
    
    # Find classifier weights
    classifier_weight = None
    classifier_bias = None
    
    for key, value in state_dict.items():
        if 'classifier' in key and 'weight' in key:
            classifier_weight = value
            print(f"Classifier weight shape: {value.shape}")
        elif 'classifier' in key and 'bias' in key:
            classifier_bias = value
            print(f"Classifier bias: {value}")
    
    if classifier_bias is not None:
        print(f"\nClassifier bias values:")
        for i, bias in enumerate(classifier_bias):
            print(f"  Class {i}: {bias:.4f}")
        
        # Check if one class is heavily biased
        max_bias_idx = torch.argmax(classifier_bias).item()
        min_bias_idx = torch.argmin(classifier_bias).item()
        
        print(f"\nBias analysis:")
        print(f"  Highest bias: Class {max_bias_idx} = {classifier_bias[max_bias_idx]:.4f}")
        print(f"  Lowest bias: Class {min_bias_idx} = {classifier_bias[min_bias_idx]:.4f}")
        
        if max_bias_idx == 2:  # "I am from Delhi" is class 2
            print(f"  🔴 PROBLEM: Class 2 ('I am from Delhi') has highest bias!")
            print(f"     This explains why it's always predicted.")
    
    # Check training metadata
    metadata_path = "models/transformer/training_metadata.json"
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\nTraining metadata:")
        print(f"  Final accuracy: {metadata.get('final_test_accuracy', 'N/A')}")
        print(f"  Training date: {metadata.get('training_date', 'N/A')}")
        print(f"  Epochs completed: {metadata.get('hyperparameters', {}).get('epochs', 'N/A')}")
    
    # Test with dummy input
    print(f"\nTesting with random input...")
    dummy_input = torch.randn(1, 60, 1662)  # Random 60-frame sequence
    
    # Simple forward pass simulation (just the final layer)
    if classifier_weight is not None and classifier_bias is not None:
        # Simulate what the model would output with random features
        dummy_features = torch.randn(1, classifier_weight.shape[1])  # Random features
        logits = torch.mm(dummy_features, classifier_weight.t()) + classifier_bias
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        print(f"Random input prediction:")
        print(f"  Logits: {logits[0]}")
        print(f"  Probabilities: {probabilities[0]}")
        print(f"  Predicted class: {predicted_class}")
        
        phrases = ["Hi my name is Reet", "How are you", "I am from Delhi", "I like coffee", "What do you like"]
        print(f"  Predicted phrase: '{phrases[predicted_class]}'")

if __name__ == "__main__":
    diagnose_model()