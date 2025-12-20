#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing llama-cpp-server ==="

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Create virtual environment if needed
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
else
    echo "Virtual environment already exists"
fi

# Upgrade pip
echo "Upgrading pip..."
"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
"$SCRIPT_DIR/venv/bin/pip" install Flask Flask-CORS python-dotenv

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
        "$SCRIPT_DIR/venv/bin/pip" install llama-cpp-python==0.2.85
        ;;
    2)
        echo "Installing llama-cpp-python with CUDA support..."
        CMAKE_ARGS="-DGGML_CUDA=on" "$SCRIPT_DIR/venv/bin/pip" install llama-cpp-python==0.2.85 --force-reinstall --no-cache-dir
        ;;
    3)
        echo "Installing llama-cpp-python with Metal support..."
        CMAKE_ARGS="-DGGML_METAL=on" "$SCRIPT_DIR/venv/bin/pip" install llama-cpp-python==0.2.85 --force-reinstall --no-cache-dir
        ;;
    *)
        echo "Invalid choice, installing CPU version..."
        "$SCRIPT_DIR/venv/bin/pip" install llama-cpp-python==0.2.85
        ;;
esac

# Install systemd user service (symlink)
echo "Installing systemd user service..."
mkdir -p ~/.config/systemd/user
ln -sf "$SCRIPT_DIR/systemd/llama-cpp-server.service" ~/.config/systemd/user/
systemctl --user daemon-reload

echo ""
echo "=== Installation complete ==="
echo ""
echo "Start with:      systemctl --user start llama-cpp-server"
echo "Enable on login: systemctl --user enable llama-cpp-server"
echo "View logs:       journalctl --user -u llama-cpp-server -f"
