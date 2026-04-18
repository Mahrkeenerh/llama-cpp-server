# llama-cpp-server

OpenAI-compatible LLM inference server, running native `llama-server` from llama.cpp with lazy model loading and automatic VRAM fitting.

## Setup

```bash
./install.sh
```

## Run

```bash
systemctl --user start llama-cpp-server
```

API is at `http://localhost:8080/v1` (OpenAI-compatible).

## Models

Benchmarked on RTX 4070 Ti Super (16 GB), ~15 GB budget for the loaded model.

| Model | Size | Context | tg t/s | Best for |
|---|--:|--:|--:|---|
| gemma-4-E4B-it-Q6_K | 5.9 GB | 128k | 99 | small & fast |
| Qwen3.5-9B-Q6_K | 6.9 GB | 256k | 68 | 256k dense |
| **gpt-oss-20b-Q8_0** | 11.3 GB | 48k | **171** | fastest all-purpose |
| gpt-oss-20b-Q8_0-128k | — | 128k | 92 | same model, max ctx |
| gemma-4-26B-A4B-it-Q6_K | 21.3 GB | 128k | 31 | Google big MoE |
| **Qwen3.6-35B-A3B-UD-Q4_K_XL** | 22.4 GB | 256k | **51** | flagship |
| Qwen3.6-35B-A3B-UD-Q5_K_XL | 26.6 GB | 128k | 37 | quality-max |

## Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="gpt-oss-20b-Q8_0",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Configuration

Edit `config.json`, then `systemctl --user restart llama-cpp-server`. Models auto-discovered from `models_directory`; per-model overrides live under `model_settings`. See `AGENT.md` for field reference.
