#!/bin/bash
set -e

echo "=== Uninstalling llama-cpp-server ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="llama-cpp-server.service"

# Stop and disable user service
echo "Stopping service..."
systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true

# Remove systemd symlink
echo "Removing systemd service..."
rm -f ~/.config/systemd/user/"$SERVICE_NAME"
systemctl --user daemon-reload

# Ask about virtual environment
read -p "Remove Python virtual environment (venv)? (y/N): " remove_venv
if [[ "$remove_venv" =~ ^[Yy]$ ]]; then
    if [ -d "$SCRIPT_DIR/venv" ]; then
        echo "Removing virtual environment..."
        rm -rf "$SCRIPT_DIR/venv"
    fi
fi

# Ask about config
read -p "Remove configuration file (config.json)? (y/N): " remove_config
if [[ "$remove_config" =~ ^[Yy]$ ]]; then
    if [ -f "$SCRIPT_DIR/config.json" ]; then
        echo "Removing configuration..."
        rm -f "$SCRIPT_DIR/config.json"
    fi
fi

echo ""
echo "=== Uninstallation complete ==="
echo "Repository remains at: $SCRIPT_DIR"
