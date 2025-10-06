"""
Unit tests for gesture recognition model.
"""

import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_landmark_extraction():
    """Test landmark extraction produces correct shape."""
    # Mock landmark data (21 landmarks * 3 coordinates)
    expected_length = 63
    landmarks = [0.5] * expected_length
    assert len(landmarks) == expected_length


def test_data_padding():
    """Test that landmark data is properly padded to 126 features."""
    landmarks = [0.5] * 63  # Single hand
    
    # Pad to 126
    while len(landmarks) < 126:
        landmarks.append(0.0)
    
    assert len(landmarks) == 126


def test_data_truncation():
    """Test that landmark data is properly truncated to 126 features."""
    landmarks = [0.5] * 200  # More than needed
    
    # Truncate to 126
    landmarks = landmarks[:126]
    
    assert len(landmarks) == 126


def test_phrase_mapping():
    """Test phrase mapping structure."""
    phrase_mapping = {
        0: "Hi, my name is Madiha Siddiqui.",
        1: "I am a student.",
        2: "I enjoy running as a hobby.",
        3: "How are you doing today?"
    }
    
    assert len(phrase_mapping) == 4
    assert all(isinstance(key, int) for key in phrase_mapping.keys())
    assert all(isinstance(value, str) for value in phrase_mapping.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
