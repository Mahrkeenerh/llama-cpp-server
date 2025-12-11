# llama-cpp-server

OpenAI-compatible inference server using llama.cpp with automatic model management.

## Features

- OpenAI-compatible API (`/v1/chat/completions`)
- Lazy model loading + auto-unload after 5 min idle
- Streaming and non-streaming responses
- GPU acceleration support

## Setup

```bash
./setup.sh
```

Choose installation type (CPU/CUDA/Metal) when prompted.

## Configuration

Edit `config.json`:
- `models_directory`: Path to .gguf files
- `models`: Model registry with settings
- `idle_timeout`: Seconds before auto-unload (default: 300)

## Usage

**Development:**
```bash
source venv/bin/activate
python server.py
```

**Production (systemd):**
```bash
sudo cp llama-cpp-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llama-cpp-server
sudo systemctl start llama-cpp-server
sudo systemctl status llama-cpp-server
```

View logs:
```bash
sudo journalctl -u llama-cpp-server -f
```

## API Examples

**Chat:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-14b-q6_K", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Health:**
```bash
curl http://localhost:8080/health
```

**Unload:**
```bash
curl -X POST http://localhost:8080/admin/unload -H "Content-Type: application/json" -d '{"model": "all"}'
```

## OpenAI Client Integration

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8080/v1', api_key='not-needed')
response = client.chat.completions.create(
    model="qwen3-14b-q6_K",
    messages=[{"role": "user", "content": "Hello"}]
)
```
