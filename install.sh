#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing llama-cpp-server ==="

# Check dependencies
for cmd in python3 cmake git; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is required"
        exit 1
    fi
done

# Clone or update llama.cpp
if [ ! -d "$SCRIPT_DIR/vendor/llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    mkdir -p "$SCRIPT_DIR/vendor"
    git clone git@github.com:ggml-org/llama.cpp.git "$SCRIPT_DIR/vendor/llama.cpp"
else
    echo "Updating llama.cpp..."
    git -C "$SCRIPT_DIR/vendor/llama.cpp" pull
fi

# Build type selection
echo ""
echo "Select build type:"
echo "1) CPU only"
echo "2) NVIDIA GPU (CUDA)"
echo "3) Apple Silicon (Metal)"
read -p "Enter choice [1-3]: " choice

CMAKE_ARGS=""
case $choice in
    1)
        echo "Building for CPU..."
        CMAKE_ARGS=""
        ;;
    2)
        echo "Building with CUDA support..."
        CMAKE_ARGS="-DGGML_CUDA=on"
        ;;
    3)
        echo "Building with Metal support..."
        CMAKE_ARGS="-DGGML_METAL=on"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Build llama-server
echo "Building llama-server..."
cd "$SCRIPT_DIR/vendor/llama.cpp"
cmake -B build $CMAKE_ARGS
cmake --build build --target llama-server -j$(nproc)

# Install binary
mkdir -p "$SCRIPT_DIR/bin"
cp build/bin/llama-server "$SCRIPT_DIR/bin/llama-server"
echo "Binary installed to bin/llama-server"
cd "$SCRIPT_DIR"

# Install systemd user service
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
