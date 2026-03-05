# llama-cpp-server

OpenAI-compatible LLM inference server built on llama.cpp with automatic model lifecycle management (lazy loading, idle unloading, subprocess isolation for GPU memory).

## Tech Stack

- Python 3, Flask 3.0.3, Flask-CORS 4.0.1
- llama-cpp-python 0.2.85 (llama.cpp bindings)
- systemd user service for production

## Commands

```bash
# Dev
source venv/bin/activate && python server.py

# Production
systemctl --user start llama-cpp-server
systemctl --user restart llama-cpp-server
journalctl --user -u llama-cpp-server -f

# Install (prompts for CPU/CUDA/Metal)
./install.sh

# Test API
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_NAME","messages":[{"role":"user","content":"Test"}]}'
```

## Directory Structure

- `server.py` — Flask app, all REST API endpoints, entry point (`main()`)
- `model_proxy.py` — `ModelProxyManager` / `ModelProxy`: model lifecycle, subprocess IPC
- `model_worker.py` — `ModelWorker`: subprocess owning the Llama instance
- `ipc_protocol.py` — IPC message definitions (dataclasses: Request, Response, Command)
- `tasks.py` — Background idle monitor thread
- `model_manager.py` — **Legacy, unused** — do not modify
- `config.json` — All configuration (server, model defaults, per-model overrides)
- `systemd/llama-cpp-server.service` — systemd user service unit
- `install.sh` / `uninstall.sh` — Setup and teardown scripts

## Architecture

- **Subprocess isolation**: Each model runs in its own subprocess (`multiprocessing.set_start_method('spawn')`) to isolate CUDA memory
- **Request flow**: Flask endpoint → `ModelProxyManager.get_model()` → IPC pipe → worker subprocess → response back
- **Model discovery**: Scans `models_directory` for `.gguf` files; names are filenames without extension
- **Model aliases**: `model_settings` entries with a `"file"` field create virtual models pointing to the same `.gguf` with different settings
- **Idle management**: Background thread checks `last_used` timestamps, shuts down subprocess after `idle_timeout`
- **Streaming**: SSE via subprocess yielding CHUNK responses; holds lock during entire generation
- **Chat templates**: Case-insensitive model name matching with Qwen fallback

## API Endpoints

- `GET /health` — Health check
- `GET /v1/models` — List available models
- `POST /v1/chat/completions` — Chat completions (OpenAI-compatible, streaming supported)
- `POST /v1/tokenize` — Tokenization (uses loaded model or fallback vocab-only tokenizer)
- `POST /v1/stop` — Stop active generation
- `POST /admin/unload` — Unload model(s)
- `POST /admin/reload` — Hot-reload config and rediscover models

## Configuration

All via `config.json`. No environment variables at runtime.
- `model_manager` section: defaults for all models (n_ctx, n_gpu_layers, n_threads, etc.)
- `model_settings` section: per-model overrides keyed by model name (optional)
- `override_tensor`: MoE expert offload pattern (e.g., `.ffn_.*_exps.=CPU`)
- `offload_kqv`: KV cache placement (true=GPU VRAM, false=CPU RAM)
- Models directory: `/mnt/DataShare/Models/LLM`

## Git Conventions

- Conventional commits: `feat:`, `fix:`, `refactor:` prefix
- Sentence case after colon, concise descriptions

## Gotchas

- `model_manager.py` is legacy code — production uses `ModelProxyManager` from `model_proxy.py`
- `multiprocessing.set_start_method('spawn')` must be called before any CUDA imports (server.py line 11)
- Worker subprocess crash causes `EOFError` in main process (no auto-restart)
- Streaming requests hold the proxy lock for the entire generation duration
- No test suite exists
