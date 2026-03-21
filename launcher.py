#!/usr/bin/env python3
"""
Launcher for llama-server with CORS proxy.

Reads config.json, generates a models.preset INI file for llama-server,
starts llama-server on an internal port, and runs a CORS reverse proxy
on the public port.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
PRESET_PATH = os.path.join(SCRIPT_DIR, "models.preset")
BINARY_PATH = os.path.join(SCRIPT_DIR, "bin", "llama-server")

# Internal port for llama-server; proxy listens on the public port
INTERNAL_PORT = 8081


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def generate_preset(config):
    """Generate models.preset INI content from config.json."""
    mm = config["model_manager"]
    models_dir = mm["models_directory"]
    settings = config.get("model_settings", {})

    lines = ["version = 1", ""]

    # Global defaults
    lines.append("[*]")
    lines.append(f"c = {mm.get('n_ctx', 16384)}")

    ngl = mm.get("n_gpu_layers", -1)
    lines.append(f"n-gpu-layers = {ngl}")
    lines.append(f"t = {mm.get('n_threads', 8)}")
    lines.append("flash-attn = on")

    if not mm.get("offload_kqv", True):
        lines.append("no-kv-offload = true")

    if mm.get("override_tensor"):
        lines.append(f"override-tensor = {mm['override_tensor']}")

    lines.append("")

    # Per-model settings
    for name, opts in settings.items():
        lines.append(f"[{name}]")

        # Alias: model with a "file" field points to a specific GGUF
        if "file" in opts:
            filename = opts["file"]
            if not os.path.isabs(filename):
                filename = os.path.join(models_dir, filename)
            lines.append(f"model = {filename}")

        if "n_ctx" in opts:
            lines.append(f"c = {opts['n_ctx']}")

        if "n_gpu_layers" in opts:
            lines.append(f"n-gpu-layers = {opts['n_gpu_layers']}")

        if "n_threads" in opts:
            lines.append(f"t = {opts['n_threads']}")

        if "override_tensor" in opts:
            lines.append(f"override-tensor = {opts['override_tensor']}")

        if "offload_kqv" in opts:
            if not opts["offload_kqv"]:
                lines.append("no-kv-offload = true")

        lines.append("")

    return "\n".join(lines)


def write_preset(config):
    content = generate_preset(config)
    with open(PRESET_PATH, "w") as f:
        f.write(content)


def build_llama_server_cmd(config):
    server = config["server"]
    mm = config["model_manager"]

    cmd = [
        BINARY_PATH,
        "--host", server.get("host", "127.0.0.1"),
        "--port", str(INTERNAL_PORT),
        "--models-dir", mm["models_directory"],
        "--models-preset", PRESET_PATH,
        "--sleep-idle-seconds", str(mm.get("idle_timeout", 300)),
        "--models-max", "1",
        "--slot-prompt-similarity", "0.5",
    ]

    return cmd


class CORSProxyHandler(BaseHTTPRequestHandler):
    """Reverse proxy that adds CORS headers to llama-server responses."""

    upstream = f"http://127.0.0.1:{INTERNAL_PORT}"
    _current_model = None
    _model_lock = threading.Lock()

    def log_message(self, format, *args):
        # Suppress default access logs (llama-server logs its own)
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _ensure_model_unloaded(self, requested_model):
        """Pre-unload the current model if switching to a different one."""
        with self._model_lock:
            if self._current_model and self._current_model != requested_model:
                try:
                    unload_req = urllib.request.Request(
                        f"{self.upstream}/models/unload",
                        data=json.dumps({"model": self._current_model}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(unload_req, timeout=30) as resp:
                        resp.read()
                except Exception:
                    pass
                # Wait for VRAM to be freed
                import time
                for _ in range(30):
                    try:
                        models_req = urllib.request.Request(f"{self.upstream}/v1/models")
                        with urllib.request.urlopen(models_req, timeout=5) as resp:
                            data = json.loads(resp.read())
                            for m in data.get("data", []):
                                if m["id"] == self._current_model:
                                    if m.get("status", {}).get("value") == "unloaded":
                                        CORSProxyHandler._current_model = None
                                        return
                    except Exception:
                        pass
                    time.sleep(1)
                CORSProxyHandler._current_model = None

    def _proxy(self):
        url = f"{self.upstream}{self.path}"

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Pre-unload if switching models (for endpoints that trigger model loading)
        if body and self.command == "POST" and any(p in self.path for p in ["/chat/completions", "/completions", "/tokenize"]):
            try:
                req_json = json.loads(body)
                requested_model = req_json.get("model")
                if requested_model:
                    self._ensure_model_unloaded(requested_model)
            except (json.JSONDecodeError, KeyError):
                pass

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
            method=self.command,
        )

        try:
            with urllib.request.urlopen(req) as resp:
                # Track loaded model on successful model-loading requests
                if resp.status == 200 and body and self.command == "POST":
                    try:
                        req_json = json.loads(body)
                        if req_json.get("model"):
                            CORSProxyHandler._current_model = req_json["model"]
                    except (json.JSONDecodeError, KeyError):
                        pass

                self.send_response(resp.status)
                # Forward headers, add CORS
                for key, value in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self._cors_headers()
                self.end_headers()

                # Stream response body
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except BrokenPipeError:
                        break

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = e.read()
            if body:
                self.wfile.write(body)

        except (urllib.error.URLError, ConnectionRefusedError):
            self.send_response(503)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": {"code": 503, "message": "llama-server not ready", "type": "unavailable_error"}
            }).encode())


def main():
    config = load_config()

    # Validate binary exists
    if not os.path.isfile(BINARY_PATH):
        print(f"Error: llama-server binary not found at {BINARY_PATH}", file=sys.stderr)
        print("Run install.sh to build it.", file=sys.stderr)
        sys.exit(1)

    # Generate preset file from config
    write_preset(config)

    # Build command
    cmd = build_llama_server_cmd(config)
    print(f"Starting llama-server: {' '.join(cmd)}", flush=True)

    # Start llama-server as child process
    llama_proc = subprocess.Popen(cmd)

    # Forward signals to child
    def shutdown(signum, frame):
        llama_proc.terminate()
        llama_proc.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Start CORS proxy on the public port
    public_port = config["server"].get("port", 8080)
    host = config["server"].get("host", "127.0.0.1")

    proxy = ThreadingHTTPServer((host, public_port), CORSProxyHandler)
    print(f"CORS proxy listening on {host}:{public_port} -> 127.0.0.1:{INTERNAL_PORT}", flush=True)

    try:
        proxy.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proxy.server_close()
        llama_proc.terminate()
        try:
            llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_proc.kill()


if __name__ == "__main__":
    main()
