#!/bin/bash
# CommuniGate ISL - Quick Launch Script for macOS/Linux
# Double-click this file to start the app

echo "🚀 Starting CommuniGate ISL..."
echo ""

# Get script directory
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    
    echo "📥 Installing dependencies..."
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Check if model exists
if [ ! -f "models/saved/lstm_model.keras" ]; then
    echo "⚠️  Model not found. You may need to train the model first."
    echo "   Run: python src/training/train_sequence_model.py"
    echo ""
fi

echo "✅ Environment ready!"
echo "🎥 Starting CommuniGate ISL application..."
echo ""
echo "📍 The app will open in your browser automatically."
echo "   If not, go to: http://localhost:8501"
echo ""
echo "⏹️  Press Ctrl+C to stop the app"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Kill any existing Streamlit instances
pkill -f "streamlit run" 2>/dev/null

# Launch Streamlit
streamlit run src/ui/app.py

echo ""
echo "👋 CommuniGate ISL stopped."
