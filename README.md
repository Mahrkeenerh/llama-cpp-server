# llama-cpp-server

OpenAI-compatible inference server using native llama-server from llama.cpp with automatic model management.

## Features

- Full OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
- Lazy model loading + auto-unload after idle timeout
- Concurrent requests with continuous batching
- Flash attention enabled by default
- GPU acceleration (CUDA / Metal)
- Model aliases with per-model configuration
- CORS proxy for browser-based clients

## Setup

```bash
./install.sh
```

Choose build type (CPU/CUDA/Metal) when prompted. This clones llama.cpp and builds the `llama-server` binary.

## Configuration

Edit `config.json`:

**Server settings:**
- `host`: Bind address (default: 127.0.0.1)
- `port`: Public port (default: 8080)

**Model Manager settings:**
- `models_directory`: Path to directory containing .gguf files (auto-discovers all models)
- `idle_timeout`: Seconds before auto-unload (default: 300)
- `n_ctx`: Default context window size (default: 16384)
- `n_gpu_layers`: GPU layers to offload (-1=all, 0=CPU only)
- `n_threads`: CPU threads for inference
- `override_tensor`: Tensor override pattern for MoE models (e.g., `.ffn_.*_exps.=CPU`)
- `offload_kqv`: KV cache placement (true=GPU, false=CPU)

**Per-model overrides** (`model_settings`):
- Override any default per model name
- Add `"file"` field to create aliases pointing to the same .gguf with different settings

Config changes take effect on service restart.

## Usage

**Development:**
```bash
python3 launcher.py
```

**Production (systemd user service):**
```bash
systemctl --user start llama-cpp-server
systemctl --user restart llama-cpp-server
journalctl --user -u llama-cpp-server -f
```

## API Examples

**Chat:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-8B-Q8_0", "messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}'
```

**Health:**
```bash
curl http://localhost:8080/health
```

**List models:**
```bash
curl http://localhost:8080/v1/models
```

**Load/unload models:**
```bash
curl -X POST http://localhost:8080/models/load -d '{"model": "Qwen3-8B-Q8_0"}'
curl -X POST http://localhost:8080/models/unload -d '{"model": "Qwen3-8B-Q8_0"}'
```

## OpenAI Client Integration

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8080/v1', api_key='not-needed')
response = client.chat.completions.create(
    model="Qwen3-8B-Q8_0",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

## Updating

To get support for new model architectures:

```bash
./install.sh
systemctl --user restart llama-cpp-server
```
