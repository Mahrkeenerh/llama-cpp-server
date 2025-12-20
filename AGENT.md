# llama-cpp-server - Agent Context

## Overview

llama-cpp-server is an OpenAI-compatible LLM inference server built on llama.cpp. It provides a REST API compatible with OpenAI's chat completions format, with features including lazy model loading, automatic model unloading after idle timeout, streaming responses, and GPU acceleration support.

## Architecture

### Main Components

- **server.py** - Flask-based REST API server providing OpenAI-compatible endpoints
- **model_manager.py** - Handles model loading, unloading, and lifecycle management
- **tasks.py** - Background task for monitoring idle models and auto-unloading
- **config.json** - Server and model manager configuration

### Key Features

- Lazy loading: Models are only loaded when first requested
- Auto-unload: Models unload after configurable idle timeout (default 5 minutes)
- Multi-model: Discovers all .gguf files in configured models directory
- GPU offloading: Configurable number of layers to offload to GPU

### External Dependencies

- llama-cpp-python (Python bindings for llama.cpp)
- Flask, Flask-CORS
- GGUF model files (stored separately in models_directory)

## Quick Commands

### Service Management

```bash
# Start service
systemctl --user start llama-cpp-server

# Stop service
systemctl --user stop llama-cpp-server

# Restart service
systemctl --user restart llama-cpp-server

# Check status
systemctl --user status llama-cpp-server

# Enable on login
systemctl --user enable llama-cpp-server

# Disable on login
systemctl --user disable llama-cpp-server
```

### View Logs

```bash
# Follow logs in real-time
journalctl --user -u llama-cpp-server -f

# View last 100 lines
journalctl --user -u llama-cpp-server -n 100

# View logs since boot
journalctl --user -u llama-cpp-server -b

# View logs from specific time
journalctl --user -u llama-cpp-server --since "1 hour ago"
```

### Check Port Usage

```bash
# Check if port 8080 is in use
sudo ss -tlnp | grep :8080

# Find process using port 8080
sudo lsof -i :8080
```

## Configuration

### File Location

`/home/samo/Data/Programs/AI/llama-cpp-server/config.json`

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `server.host` | Bind address | 127.0.0.1 |
| `server.port` | Server port | 8080 |
| `server.cors_origins` | Allowed CORS origins | ["http://localhost:5000"] |
| `model_manager.models_directory` | Path to .gguf model files | /mnt/DataShare/Models/LLM |
| `model_manager.idle_timeout` | Seconds before auto-unload | 300 |
| `model_manager.check_interval` | Seconds between idle checks | 60 |
| `model_manager.n_ctx` | Context window size | 16384 |
| `model_manager.n_gpu_layers` | GPU layers (-1=all, 0=none) | -1 |
| `model_manager.n_threads` | CPU threads | 8 |
| `model_manager.default_model` | Default model filename | Qwen3-8B-Q8_0.gguf |

### Example Configuration

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "cors_origins": ["http://localhost:5000"],
    "log_level": "INFO"
  },
  "model_manager": {
    "models_directory": "/mnt/DataShare/Models/LLM",
    "idle_timeout": 300,
    "check_interval": 60,
    "n_ctx": 16384,
    "n_gpu_layers": -1,
    "n_threads": 8,
    "default_model": "Qwen3-8B-Q8_0.gguf"
  }
}
```

### After Changing Config

```bash
# Restart to apply changes
systemctl --user restart llama-cpp-server

# Or hot-reload via API (rediscovers models, updates settings)
curl -X POST http://localhost:8080/admin/reload
```

## Troubleshooting

### Service won't start

**Symptoms:** `systemctl --user status` shows "failed" or "activating"

**Diagnostic Steps:**

1. Check logs for errors:
   ```bash
   journalctl --user -u llama-cpp-server -n 50
   ```

2. Verify config.json is valid JSON:
   ```bash
   python3 -c "import json; json.load(open('/home/samo/Data/Programs/AI/llama-cpp-server/config.json'))"
   ```

3. Check if port 8080 is already in use:
   ```bash
   sudo ss -tlnp | grep :8080
   ```

4. Verify virtual environment exists:
   ```bash
   ls -la /home/samo/Data/Programs/AI/llama-cpp-server/venv/
   ```

5. Test running manually:
   ```bash
   cd /home/samo/Data/Programs/AI/llama-cpp-server
   source venv/bin/activate
   python server.py
   ```

**Common Causes:**

- Invalid JSON in config.json (syntax error, trailing comma)
- Port 8080 already in use by another service
- Virtual environment missing or corrupted
- Models directory doesn't exist or is inaccessible
- Permission issues on service files

### Port already in use

**Symptoms:** "Address already in use" error in logs

**Solution:**

```bash
# Find what's using port 8080
sudo lsof -i :8080

# Kill the conflicting process (replace PID)
sudo kill PID

# Or change port in config.json and restart
```

### Model fails to load

**Symptoms:** 404 or 500 errors when making API requests

**Diagnostic Steps:**

1. Check if model file exists:
   ```bash
   ls -la /mnt/DataShare/Models/LLM/
   ```

2. Verify model name matches (case-sensitive, can omit .gguf extension):
   ```bash
   curl http://localhost:8080/v1/models
   ```

3. Check logs for specific error:
   ```bash
   journalctl --user -u llama-cpp-server -n 50 | grep -i error
   ```

**Common Causes:**

- Model file doesn't exist in models_directory
- Insufficient RAM/VRAM to load model
- Corrupted .gguf file
- models_directory path incorrect in config

### Out of memory (OOM)

**Symptoms:** Service crashes, "out of memory" in logs, system becomes unresponsive

**Solutions:**

1. Reduce context window:
   ```json
   "n_ctx": 8192
   ```

2. Reduce GPU layers (offload less to VRAM):
   ```json
   "n_gpu_layers": 20
   ```

3. Use a smaller/more quantized model

4. Unload unused models:
   ```bash
   curl -X POST http://localhost:8080/admin/unload -H "Content-Type: application/json" -d '{"model": "all"}'
   ```

### High memory usage

**Symptoms:** Service consuming excessive RAM even when idle

**Diagnostic:**

```bash
# Check memory usage
ps aux | grep server.py

# Check which models are loaded
curl http://localhost:8080/health
```

**Solution:**

Models stay loaded until idle_timeout expires. To free memory immediately:

```bash
curl -X POST http://localhost:8080/admin/unload -H "Content-Type: application/json" -d '{"model": "all"}'
```

Or reduce idle_timeout in config.json.

### Service keeps restarting

**Symptoms:** Service restarts in a loop

**Diagnostic:**

```bash
# View restart count
systemctl --user show llama-cpp-server --property=NRestarts

# Check for crash reason
journalctl --user -u llama-cpp-server --since "10 minutes ago"
```

**Common Causes:**

- Repeated OOM crashes
- Invalid configuration causing startup failure
- Model loading crash

### Slow responses

**Symptoms:** API responses take very long

**Possible Causes & Solutions:**

1. **Model not using GPU:**
   - Check n_gpu_layers is set to -1 (all layers)
   - Verify CUDA version of llama-cpp-python is installed

2. **Context too large:**
   - Reduce n_ctx in config

3. **First request loads model:**
   - This is expected behavior (lazy loading)
   - Subsequent requests will be faster

4. **CPU-only inference:**
   - Reinstall with CUDA support:
     ```bash
     source venv/bin/activate
     CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
     ```

### CORS errors in browser

**Symptoms:** Browser console shows CORS errors

**Solution:**

Add your frontend's origin to config.json:

```json
"cors_origins": ["http://localhost:5000", "http://localhost:3000"]
```

Then restart the service.

## Health Checks

### HTTP Health Endpoint

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "ok",
  "loaded_models": ["Qwen3-8B-Q8_0"],
  "uptime": 3600
}
```

### Process Check

```bash
# Check if service is active
systemctl --user is-active llama-cpp-server

# Check if process is running
pgrep -f "python server.py"
```

### API Test

```bash
# List available models
curl http://localhost:8080/v1/models

# Simple chat test
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-8B-Q8_0", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 50}'
```

## Dependencies

### System Packages

- `python3` (3.8+) - Python runtime
- `python3-venv` - Virtual environment support
- NVIDIA drivers + CUDA (optional, for GPU acceleration)

### Python Packages

- `Flask` - Web framework
- `Flask-CORS` - CORS support
- `llama-cpp-python` - llama.cpp Python bindings
- `python-dotenv` - Environment variable support

### Runtime Requirements

- Python 3.8+
- systemd (for service management)
- Sufficient RAM (8GB+ recommended, depends on model size)
- NVIDIA GPU with CUDA (optional, for GPU acceleration)

## File Locations

| File | Path |
|------|------|
| Service root | /home/samo/Data/Programs/AI/llama-cpp-server |
| Configuration | /home/samo/Data/Programs/AI/llama-cpp-server/config.json |
| Systemd unit | ~/.config/systemd/user/llama-cpp-server.service (symlink) |
| Virtual env | /home/samo/Data/Programs/AI/llama-cpp-server/venv |
| Models | /mnt/DataShare/Models/LLM (configurable) |
| Logs | journalctl --user -u llama-cpp-server |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with loaded models and uptime |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/admin/unload` | POST | Manually unload model(s) |
| `/admin/reload` | POST | Hot-reload configuration |

### Chat Completions Request Format

```json
{
  "model": "Qwen3-8B-Q8_0",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false
}
```

## Maintenance

### Backup Configuration

```bash
cp /home/samo/Data/Programs/AI/llama-cpp-server/config.json \
   /home/samo/Data/Programs/AI/llama-cpp-server/config.json.bak
```

### Update Service

```bash
cd /home/samo/Data/Programs/AI/llama-cpp-server
git pull
./install.sh
systemctl --user restart llama-cpp-server
```

### Reinstall with GPU Support

```bash
cd /home/samo/Data/Programs/AI/llama-cpp-server
source venv/bin/activate
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
systemctl --user restart llama-cpp-server
```

### Add New Models

1. Copy .gguf files to models_directory
2. Reload configuration:
   ```bash
   curl -X POST http://localhost:8080/admin/reload
   ```
3. Verify model appears:
   ```bash
   curl http://localhost:8080/v1/models
   ```
