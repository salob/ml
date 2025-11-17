#!/bin/bash
# Quick start script for the web demo

# Detect Python executable
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "Error: Python not found!"
    exit 1
fi

echo "Starting IMDB Sentiment Analysis Web Demo..."
echo ""

# Check if models exist
if [ ! -d "models" ]; then
    echo "Error: models/ directory not found!"
    echo ""
    echo "Please train your models first:"
    echo "  $PYTHON IMDB/imdb_logreg.py"
    echo "  $PYTHON IMDB/imdb_cnn.py"
    echo "  $PYTHON IMDB/imdb_transformer.py"
    echo ""
    echo "Or run all experiments:"
    echo "  $PYTHON scripts/run_multiple_experiments.py"
    exit 1
fi

# Check if gradio is installed
if ! $PYTHON -c "import gradio" 2>/dev/null; then
    echo "Gradio not installed. Installing..."
    uv pip install gradio
fi

echo "Starting web server..."
echo ""
echo "The demo will open in your browser at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo ""

$PYTHON scripts/web_demo.py
