# llama-cpp-server - Agent Context

## Overview

llama-cpp-server is an OpenAI-compatible LLM inference server using the native llama-server binary from llama.cpp. A thin Python launcher reads config.json, generates model presets, and runs a CORS reverse proxy in front of llama-server.

## Architecture

### Components

- **launcher.py** — Reads config.json, generates models.preset INI, starts llama-server (port 8081) and CORS proxy (port 8080)
- **bin/llama-server** — Native llama.cpp binary handling all inference and OpenAI API endpoints
- **config.json** — Server and model configuration (single source of truth)
- **models.preset** — Generated at launch from config.json (consumed by llama-server)

### Key Features

- Lazy loading: Models load on first request (`--models-autoload`)
- Auto-unload: Models unload after idle timeout (`--sleep-idle-seconds`)
- Multi-model: Discovers all .gguf files via `--models-dir`
- Concurrent requests: Parallel slots with continuous batching
- Flash attention enabled by default
- CORS proxy for browser-based clients

### External Dependencies

- Python 3 (stdlib only, no pip packages)
- llama.cpp (built from source, stored in vendor/)
- GGUF model files (stored separately in models_directory)
- cmake (for building from source)
- CUDA toolkit (for GPU acceleration)

## Quick Commands

### Service Management

```bash
systemctl --user start llama-cpp-server
systemctl --user stop llama-cpp-server
systemctl --user restart llama-cpp-server
systemctl --user status llama-cpp-server
systemctl --user enable llama-cpp-server
```

### View Logs

```bash
journalctl --user -u llama-cpp-server -f
journalctl --user -u llama-cpp-server -n 100
journalctl --user -u llama-cpp-server --since "1 hour ago"
```

### Check Port Usage

```bash
ss -tlnp | grep -E ':808[01]'
```

## Configuration

### File Location

`config.json` in the project root.

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `server.host` | Bind address | 127.0.0.1 |
| `server.port` | Public proxy port | 8080 |
| `model_manager.models_directory` | Path to .gguf files | /mnt/DataShare/Models/LLM |
| `model_manager.idle_timeout` | Seconds before auto-unload | 300 |
| `model_manager.n_ctx` | Default context window | 16384 |
| `model_manager.n_gpu_layers` | GPU layers (-1=all) | -1 |
| `model_manager.n_threads` | CPU threads | 8 |
| `model_manager.override_tensor` | Tensor override pattern | null |
| `model_manager.offload_kqv` | KV cache on GPU | true |
| `model_manager.parallel` | Concurrent slots (np) | 1 |
| `model_settings.<name>` | Per-model overrides | {} |

### Per-model override fields (in `model_settings.<name>`)

| Field | Preset key | Purpose |
|-------|-----------|---------|
| `file` | `model` | Alias another .gguf |
| `n_ctx` | `c` | Context size |
| `n_gpu_layers` | `n-gpu-layers` | GPU layer count |
| `n_threads` | `t` | CPU threads |
| `override_tensor` | `override-tensor` | Pattern-based tensor placement |
| `offload_kqv: false` | `no-kv-offload = true` | KV cache in CPU RAM |
| `n_cpu_moe` | `n-cpu-moe` | MoE: first N layers' experts on CPU |
| `cache_type_k` | `cache-type-k` | `q8_0` halves K cache memory |
| `cache_type_v` | `cache-type-v` | `q8_0` halves V cache memory |
| `parallel` | `np` | Concurrent slots |
| `fit` | `fit` | Auto-offload to fit in VRAM (MoE) |

### After Changing Config

```bash
systemctl --user restart llama-cpp-server
```

Config changes take effect on restart (launcher regenerates models.preset).

## Troubleshooting

### Service won't start

1. Check logs:
   ```bash
   journalctl --user -u llama-cpp-server -n 50
   ```

2. Verify binary exists:
   ```bash
   ls -la bin/llama-server
   ```
   If missing, run `./install.sh` to build it.

3. Verify config.json is valid:
   ```bash
   python3 -c "import json; json.load(open('config.json'))"
   ```

4. Check if ports 8080/8081 are in use:
   ```bash
   ss -tlnp | grep -E ':808[01]'
   ```

5. Test running manually:
   ```bash
   python3 launcher.py
   ```

### Model fails to load

1. Check model file exists:
   ```bash
   ls -la /mnt/DataShare/Models/LLM/
   ```

2. List available models:
   ```bash
   curl http://localhost:8080/v1/models
   ```

3. Check for architecture support — if a model uses a new architecture, rebuild:
   ```bash
   ./install.sh
   ```

### Out of memory (OOM)

- Reduce context: set `n_ctx` lower in model_settings
- Enable `cache_type_k`/`cache_type_v: "q8_0"` to halve KV cache memory
- For MoE models, set `fit: true` so llama-server auto-offloads experts to CPU
- For MoE models, set `n_cpu_moe: N` to pin the offload split (first N layers' experts on CPU)
- Reduce parallel slots: `parallel: 1` minimizes recurrent-state memory
- Reduce GPU layers manually: set `n_gpu_layers` to a number instead of -1
- Use a smaller quantization

### CORS errors

The CORS proxy allows all origins by default. If you still see CORS errors, check that requests go to port 8080 (proxy), not 8081 (llama-server directly).

### Updating llama.cpp

To get support for new model architectures:
```bash
./install.sh   # pulls latest llama.cpp and rebuilds
systemctl --user restart llama-cpp-server
```

## Health Checks

```bash
# Health endpoint
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Chat test
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b-Q8_0", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 50}'
```

## File Locations

| File | Path |
|------|------|
| Service root | /home/samo/Data/Programs/AI/llama-cpp-server |
| Configuration | config.json |
| Binary | bin/llama-server |
| Systemd unit | ~/.config/systemd/user/llama-cpp-server.service (symlink) |
| Models | /mnt/DataShare/Models/LLM |
| Logs | journalctl --user -u llama-cpp-server |

## API Endpoints

All provided by llama-server natively:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Embeddings |
| `/tokenize` | POST | Tokenization |
| `/detokenize` | POST | Detokenization |
| `/models/load` | POST | Load a model |
| `/models/unload` | POST | Unload a model |
| `/slots` | GET | View active slots |
| `/metrics` | GET | Prometheus metrics |
