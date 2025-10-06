#!/bin/bash

# CommuniGate_ISL Setup Script for macOS
# This script sets up the project with Python 3.11

set -e  # Exit on error

echo "================================================"
echo "  CommuniGate_ISL - Setup Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.11 is installed
echo "Checking for Python 3.11..."
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION=$(python3.11 --version)
    echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3.11 not found${NC}"
    echo ""
    echo "Installing Python 3.11 via Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}✗ Homebrew not found${NC}"
        echo "Please install Homebrew first: https://brew.sh"
        exit 1
    fi
    
    brew install python@3.11
    echo -e "${GREEN}✓ Python 3.11 installed${NC}"
fi

echo ""

# Remove old virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
    echo -e "${GREEN}✓ Old environment removed${NC}"
fi

echo ""

# Create new virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
python3.11 -m venv .venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓ Environment activated${NC}"

echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

echo ""

# Install dependencies
echo "Installing project dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo ""

# Install mediapipe
echo "Installing mediapipe..."
pip install mediapipe --quiet
echo -e "${GREEN}✓ mediapipe installed${NC}"

echo ""
echo "================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "================================================"
echo ""

# Verify installation
echo "Verifying installation..."
PYTHON_VER=$(python --version)
MEDIAPIPE_VER=$(python -c "import mediapipe; print(mediapipe.__version__)" 2>/dev/null || echo "Error")
OPENCV_VER=$(python -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Error")

echo ""
echo "Installed versions:"
echo "  Python: $PYTHON_VER"
echo "  Mediapipe: $MEDIAPIPE_VER"
echo "  OpenCV: $OPENCV_VER"

echo ""
echo "================================================"
echo "  Next Steps"
echo "================================================"
echo ""
echo "1. Activate the environment:"
echo "   ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo "2. Test your camera:"
echo "   ${YELLOW}python src/data_collection/test_camera.py${NC}"
echo ""
echo "3. Test hand detection:"
echo "   ${YELLOW}python src/data_collection/test_mediapipe.py${NC}"
echo ""
echo "4. Read the documentation:"
echo "   ${YELLOW}cat SETUP.md${NC}"
echo ""
echo "================================================"
echo ""
