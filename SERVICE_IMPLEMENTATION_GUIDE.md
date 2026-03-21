# Service Implementation Guide

This guide describes how to structure a service so it can be managed by the Service Admin Dashboard.

---

## Core Principles

All services **must** follow these principles:

1. **Run from repository** - Services run directly from their git repository. No copying files to `~/.local/share`, `/opt`, or other locations.
2. **Symlinks only** - System integrations (systemd units, desktop entries) use symlinks pointing to repository files.
3. **User-level by default** - Use `systemctl --user` unless the service genuinely requires root privileges.
4. **Idempotent scripts** - Install works as reinstall/update. Uninstall cleanly removes integrations without breaking anything.

---

## Required Files

Every service **must** have these files in its root directory:

| File | Purpose |
|------|---------|
| `service.json` | Service manifest - metadata, type, file paths |
| `README.md` | Human-readable documentation |
| `install.sh` | Installation/update script (must be executable) |
| `uninstall.sh` | Clean removal script (must be executable) |

---

## Optional Files

These files enhance dashboard functionality:

| File | Purpose |
|------|---------|
| `AGENT.md` | AI assistant context for troubleshooting (recommended) |
| `systemd/*.service` | Systemd unit file |
| `systemd/*.timer` | Systemd timer file (for scheduled services) |

---

## Service Types

### 1. `user-service` (Preferred)

- Managed via `systemctl --user`
- Service file symlinked to `~/.config/systemd/user/`
- No root privileges required
- **Use this unless you have a specific reason for system-level**

### 2. `system-service`

- Managed via `sudo systemctl`
- Service file symlinked to `/etc/systemd/system/`
- Requires root/sudo for start/stop/restart
- **Only use when the service genuinely needs root** (e.g., binding to privileged ports, accessing system resources)

### 3. `application`

- Not a daemon - CLI tool or GUI application
- No systemd integration
- Install/uninstall handle shortcuts, PATH registration, etc.

---

## File Specifications

### `service.json` - Service Manifest

```json
{
  "name": "string (required) - Display name",
  "version": "string (required) - Semantic version",
  "description": "string (required) - Brief description",
  "type": "string (required) - user-service | system-service | application",

  "systemd": {
    "unit": "string - Relative path to .service file",
    "timer": "string | null - Relative path to .timer file",
    "service_name": "string - Unit name for systemctl (e.g., 'myapp.service')"
  },

  "network": {
    "port": "number | null - Primary port the service listens on",
    "health": "string | null - Health check endpoint (e.g., '/health')"
  },

  "paths": {
    "config": "string | null - Path to config file (supports ~)",
    "logs": "string - 'journalctl' or path to log file",
    "install": "string (required) - Relative path to install script",
    "uninstall": "string (required) - Relative path to uninstall script"
  },

  "ui": {
    "icon": "string - Emoji or icon identifier",
    "category": "string - Category for grouping (e.g., 'AI', 'Backup', 'Media')"
  }
}
```

#### Example: User Service

```json
{
  "name": "AutoBackUp",
  "version": "2.0.0",
  "description": "Automatic backup utility with scheduling",
  "type": "user-service",

  "systemd": {
    "unit": "systemd/autobackup.service",
    "timer": "systemd/autobackup.timer",
    "service_name": "autobackup.service"
  },

  "network": {
    "port": null,
    "health": null
  },

  "paths": {
    "config": "~/.config/autobackup/config.json",
    "logs": "journalctl",
    "install": "install.sh",
    "uninstall": "uninstall.sh"
  },

  "ui": {
    "icon": "💾",
    "category": "Backup"
  }
}
```

#### Example: System Service (when root is required)

```json
{
  "name": "llama-cpp-server",
  "version": "1.0.0",
  "description": "OpenAI-compatible LLM inference server",
  "type": "system-service",

  "systemd": {
    "unit": "systemd/llama-cpp-server.service",
    "timer": null,
    "service_name": "llama-cpp-server.service"
  },

  "network": {
    "port": 8080,
    "health": "/health"
  },

  "paths": {
    "config": "config.json",
    "logs": "journalctl",
    "install": "install.sh",
    "uninstall": "uninstall.sh"
  },

  "ui": {
    "icon": "🤖",
    "category": "AI"
  }
}
```

#### Example: Application

```json
{
  "name": "CaptiX",
  "version": "0.4.0",
  "description": "Screenshot and screen recording tool for Linux X11",
  "type": "application",

  "systemd": null,

  "network": {
    "port": null,
    "health": null
  },

  "paths": {
    "config": "~/.config/captix/config.json",
    "logs": null,
    "install": "install.sh",
    "uninstall": "uninstall.sh"
  },

  "ui": {
    "icon": "📸",
    "category": "Media"
  }
}
```

---

### `install.sh` - Installation Script

**Requirements:**
- Must be executable (`chmod +x install.sh`)
- Must be idempotent (running twice = same result as running once)
- Must work as an update (re-running updates/reinstalls cleanly)
- Must use symlinks for all system integrations
- Must NOT copy files outside the repository

#### Template: User Service

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing ServiceName ==="

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Create virtual environment if needed
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# Install/update dependencies
echo "Installing dependencies..."
"$SCRIPT_DIR/venv/bin/pip" install -q -r "$SCRIPT_DIR/requirements.txt"

# Symlink systemd service (user-level)
echo "Installing systemd user service..."
mkdir -p ~/.config/systemd/user
ln -sf "$SCRIPT_DIR/systemd/myservice.service" ~/.config/systemd/user/
systemctl --user daemon-reload

echo "=== Installation complete ==="
echo "Start with: systemctl --user start myservice"
echo "Enable on login: systemctl --user enable myservice"
```

#### Template: System Service (when root required)

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing ServiceName ==="

# Check for root
if [ "$EUID" -ne 0 ]; then
    echo "This service requires root. Re-running with sudo..."
    exec sudo "$0" "$@"
fi

# Install dependencies...

# Symlink systemd service (system-level)
echo "Installing systemd service..."
ln -sf "$SCRIPT_DIR/systemd/myservice.service" /etc/systemd/system/
systemctl daemon-reload

echo "=== Installation complete ==="
echo "Start with: sudo systemctl start myservice"
```

#### Template: Application

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing AppName ==="

# Install dependencies...

# Symlink desktop entry (if GUI app)
if [ -f "$SCRIPT_DIR/myapp.desktop" ]; then
    mkdir -p ~/.local/share/applications
    ln -sf "$SCRIPT_DIR/myapp.desktop" ~/.local/share/applications/
    update-desktop-database ~/.local/share/applications 2>/dev/null || true
fi

# Symlink to PATH (if CLI tool)
mkdir -p ~/.local/bin
ln -sf "$SCRIPT_DIR/myapp" ~/.local/bin/

echo "=== Installation complete ==="
```

---

### `uninstall.sh` - Uninstallation Script

**Requirements:**
- Must be executable (`chmod +x uninstall.sh`)
- Must remove all system integrations (symlinks, registrations)
- Must NOT delete the repository itself
- Must NOT break other services or system functionality
- Should be safe to run even if not fully installed

#### Template: User Service

```bash
#!/bin/bash
set -e

echo "=== Uninstalling ServiceName ==="

# Stop and disable service
systemctl --user stop myservice.service 2>/dev/null || true
systemctl --user disable myservice.service 2>/dev/null || true

# Remove systemd symlink
rm -f ~/.config/systemd/user/myservice.service
systemctl --user daemon-reload

# Optionally remove config (ask user)
if [ -d ~/.config/myservice ]; then
    read -p "Remove configuration files in ~/.config/myservice? (y/N): " remove_config
    if [[ "$remove_config" =~ ^[Yy]$ ]]; then
        rm -rf ~/.config/myservice
        echo "Configuration removed."
    fi
fi

echo "=== Uninstallation complete ==="
echo "Repository remains at: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

#### Template: System Service

```bash
#!/bin/bash
set -e

echo "=== Uninstalling ServiceName ==="

# Check for root
if [ "$EUID" -ne 0 ]; then
    echo "This requires root. Re-running with sudo..."
    exec sudo "$0" "$@"
fi

# Stop and disable service
systemctl stop myservice.service 2>/dev/null || true
systemctl disable myservice.service 2>/dev/null || true

# Remove systemd symlink
rm -f /etc/systemd/system/myservice.service
systemctl daemon-reload

echo "=== Uninstallation complete ==="
```

#### Template: Application

```bash
#!/bin/bash
set -e

echo "=== Uninstalling AppName ==="

# Remove desktop entry symlink
rm -f ~/.local/share/applications/myapp.desktop
update-desktop-database ~/.local/share/applications 2>/dev/null || true

# Remove PATH symlink
rm -f ~/.local/bin/myapp

# Optionally remove config
if [ -d ~/.config/myapp ]; then
    read -p "Remove configuration files? (y/N): " remove_config
    if [[ "$remove_config" =~ ^[Yy]$ ]]; then
        rm -rf ~/.config/myapp
    fi
fi

echo "=== Uninstallation complete ==="
```

---

### `AGENT.md` - AI Assistant Context (Recommended)

This file provides context to the AI assistant for service-specific troubleshooting.

```markdown
# ServiceName - Agent Context

## Overview
Brief description of what the service does.

## Quick Commands

```bash
# Start/stop/restart
systemctl --user start myservice
systemctl --user stop myservice
systemctl --user restart myservice

# View logs
journalctl --user -u myservice -f

# Check status
systemctl --user status myservice
```

## Configuration
- Config file: `~/.config/myservice/config.json`
- Key settings and their effects

## Troubleshooting

### Service won't start
1. Check logs: `journalctl --user -u myservice -n 50`
2. Verify config syntax
3. Check port availability: `lsof -i :PORT`

### Common issues
- Issue 1: cause and solution
- Issue 2: cause and solution
```

---

## Directory Structure

### Standard Layout

```
my-service/
├── service.json          # Required: manifest
├── README.md             # Required: documentation
├── install.sh            # Required: installation
├── uninstall.sh          # Required: uninstallation
├── AGENT.md              # Recommended: AI context
├── systemd/
│   ├── my-service.service
│   └── my-service.timer  # Optional: for scheduled tasks
├── src/
│   └── ...
└── requirements.txt
```

---

## Validation Checklist

Before adding a service to the dashboard:

### Required
- [ ] `service.json` exists with all required fields
- [ ] `README.md` exists
- [ ] `install.sh` exists and is executable
- [ ] `uninstall.sh` exists and is executable
- [ ] Install script uses symlinks (not copies)
- [ ] Install script is idempotent
- [ ] Uninstall script removes all system integrations
- [ ] Service runs from repository (no files copied elsewhere)

### For Systemd Services
- [ ] Service file exists at `systemd.unit` path
- [ ] Service type is `user-service` unless root is genuinely required
- [ ] Install creates symlink in correct systemd directory

### Recommended
- [ ] `AGENT.md` exists with troubleshooting information
- [ ] Health endpoint if network service

---

## Migration Guide

To migrate an existing service:

### Step 1: Ensure repository-based execution
- Remove any install steps that copy files to `/opt`, `~/.local/share`, etc.
- All code runs from the repository directory

### Step 2: Convert copies to symlinks
- Change `cp` to `ln -sf` for systemd units, desktop entries, etc.
- Update paths to point to repository files

### Step 3: Create/update uninstall.sh
- Must remove all symlinks created by install
- Must NOT delete repository or break other services

### Step 4: Prefer user-level
- If service doesn't need root, change to `user-service`
- Use `systemctl --user` instead of `sudo systemctl`

### Step 5: Test
```bash
# Clean test
./uninstall.sh
./install.sh
# Verify service works

# Update test
./install.sh  # Should work without uninstall first
```
