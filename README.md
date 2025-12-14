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

**Server settings:**
- `host`: Server bind address (default: 127.0.0.1)
- `port`: Server port (default: 8080)
- `cors_origins`: Allowed CORS origins

**Model Manager settings:**
- `models_directory`: Path to directory containing .gguf files (auto-discovers all models)
- `idle_timeout`: Seconds before auto-unload (default: 300)
- `check_interval`: Seconds between idle checks (default: 60)
- `n_ctx`: Context window size (default: 40960)
- `n_gpu_layers`: GPU layers to offload (0=CPU only, -1=all layers)
- `n_threads`: CPU threads for inference
- `default_model`: Default model filename (e.g., "Qwen3-14B-Q6_K.gguf")

**Note:** All .gguf files in `models_directory` are automatically discovered. Temperature is set via API calls, not config.

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

Restart after config changes:
```bash
sudo systemctl restart llama-cpp-server
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
  -d '{"model": "Qwen3-14B-Q6_K", "messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}'
```

**Health:**
```bash
curl http://localhost:8080/health
```

**Unload:**
```bash
curl -X POST http://localhost:8080/admin/unload -H "Content-Type: application/json" -d '{"model": "all"}'
```

**Reload config (rediscovers models in directory):**
```bash
curl -X POST http://localhost:8080/admin/reload
```

**List models:**
```bash
curl http://localhost:8080/v1/models
```

## OpenAI Client Integration

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8080/v1', api_key='not-needed')
response = client.chat.completions.create(
    model="Qwen3-14B-Q6_K",  # or "Qwen3-8B-Q6_K", "Qwen3-8B-Q8_0"
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```
