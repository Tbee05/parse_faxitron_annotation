#!/bin/bash

# Faxitron Final Detector Installation Script
# This script sets up a conda environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Installing Faxitron Final Detector..."
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install conda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
ENV_NAME="faxitron"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n $ENV_NAME
    else
        echo "Using existing environment..."
    fi
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "🔧 Creating conda environment '$ENV_NAME' with Python 3.9..."
    conda create -n $ENV_NAME python=3.9 -y
fi

# Activate environment
echo "🔌 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test installation
echo "🧪 Testing installation..."
python -c "
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
print('✅ All dependencies imported successfully!')
"

echo ""
echo "🎉 Installation complete!"
echo "========================================"
echo "To use the detector:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Run the detector: python faxitron_final_detector.py --help"
echo ""
echo "Examples:"
echo "  # Process single image"
echo "  python faxitron_final_detector.py -i /path/to/image.jpg"
echo ""
echo "  # Process folder of images"
echo "  python faxitron_final_detector.py -f /path/to/folder"
echo ""
echo "  # Run demo mode"
echo "  python faxitron_final_detector.py"
