#!/bin/bash

set -e

echo "Setting up llama-cpp-server..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
pip install Flask Flask-CORS python-dotenv

# Check for CUDA support
echo ""
echo "Select installation type:"
echo "1) CPU only"
echo "2) NVIDIA GPU (CUDA)"
echo "3) Apple Silicon (Metal)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Installing llama-cpp-python (CPU only)..."
        pip install llama-cpp-python==0.2.85
        ;;
    2)
        echo "Installing llama-cpp-python with CUDA support..."
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.2.85 --force-reinstall --no-cache-dir
        ;;
    3)
        echo "Installing llama-cpp-python with Metal support..."
        CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python==0.2.85 --force-reinstall --no-cache-dir
        ;;
    *)
        echo "Invalid choice, installing CPU version..."
        pip install llama-cpp-python==0.2.85
        ;;
esac

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the server:"
echo "  python server.py"
