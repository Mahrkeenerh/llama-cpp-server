"""Model proxy - main process interface to worker subprocesses."""
import os
import time
import threading
import multiprocessing
import logging
from typing import Generator, Dict, Optional

from ipc_protocol import Command, ResponseType, Request, Response
from model_worker import worker_main

logger = logging.getLogger(__name__)


class ModelProxy:
    """Main process proxy to a model worker subprocess."""

    def __init__(self, model_name: str, model_path: str, config: dict):
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.process: Optional[multiprocessing.Process] = None
        self.conn = None
        self.lock = threading.Lock()
        self.last_used = 0
        self.stop_event: Optional[multiprocessing.Event] = None

    def start(self):
        """Start the subprocess and load the model."""
        if self.process is not None and self.process.is_alive():
            return

        logger.info(f"Starting subprocess for model: {self.model_name}")

        parent_conn, child_conn = multiprocessing.Pipe()
        self.conn = parent_conn
        self.stop_event = multiprocessing.Event()

        self.process = multiprocessing.Process(
            target=worker_main,
            args=(child_conn, self.model_name, self.config, self.stop_event),
            daemon=False
        )
        self.process.start()
        child_conn.close()

        self._send_load_command()
        self.last_used = time.time()

    def _send_load_command(self):
        """Send LOAD command to subprocess."""
        # Get default settings from model_manager
        n_ctx = self.config["model_manager"]["n_ctx"]
        n_gpu_layers = self.config["model_manager"]["n_gpu_layers"]
        n_threads = self.config["model_manager"].get("n_threads", 8)

        # Check for per-model settings
        model_settings = self.config.get("model_settings", {}).get(self.model_name, {})
        if "n_ctx" in model_settings:
            n_ctx = model_settings["n_ctx"]
        if "n_gpu_layers" in model_settings:
            n_gpu_layers = model_settings["n_gpu_layers"]
        if "n_threads" in model_settings:
            n_threads = model_settings["n_threads"]

        request = Request(
            command=Command.LOAD,
            payload={
                "model_path": self.model_path,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "n_threads": n_threads
            }
        )
        self.conn.send(request)

        if self.conn.poll(timeout=120):
            response = self.conn.recv()
            if response.type == ResponseType.ERROR:
                raise RuntimeError(f"Failed to load model: {response.payload.get('error')}")
            logger.info(f"Model loaded: {response.payload}")
        else:
            raise TimeoutError("Model load timeout")

    def is_alive(self) -> bool:
        """Check if subprocess is running."""
        return self.process is not None and self.process.is_alive()

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> dict:
        """Non-streaming generation."""
        with self.lock:
            if not self.is_alive():
                raise RuntimeError("Subprocess not running")

            self.last_used = time.time()

            request = Request(
                command=Command.GENERATE,
                payload={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            self.conn.send(request)

            if self.conn.poll(timeout=300):
                response = self.conn.recv()
                if response.type == ResponseType.ERROR:
                    raise RuntimeError(response.payload.get("error", "Unknown error"))
                return response.payload
            else:
                raise TimeoutError("Generation timeout")

    def generate_stream(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> Generator[dict, None, None]:
        """Streaming generation - yields chunk payloads."""
        with self.lock:
            if not self.is_alive():
                raise RuntimeError("Subprocess not running")

            # Clear any previous stop signal
            self.clear_stop()
            self.last_used = time.time()

            request = Request(
                command=Command.GENERATE_STREAM,
                payload={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            self.conn.send(request)

            while True:
                try:
                    if self.conn.poll(timeout=60):
                        response = self.conn.recv()

                        if response.type == ResponseType.CHUNK:
                            yield response.payload
                        elif response.type == ResponseType.DONE:
                            break
                        elif response.type == ResponseType.ERROR:
                            raise RuntimeError(response.payload.get("error", "Unknown error"))
                    else:
                        raise TimeoutError("Streaming timeout")
                except EOFError:
                    raise RuntimeError("Model worker crashed unexpectedly (likely out of memory - try a smaller model or quantization)")

    def tokenize(self, text: str, add_bos: bool = False) -> dict:
        """Tokenize text and return token count."""
        with self.lock:
            if not self.is_alive():
                raise RuntimeError("Subprocess not running")

            self.last_used = time.time()

            request = Request(
                command=Command.TOKENIZE,
                payload={
                    "text": text,
                    "add_bos": add_bos
                }
            )
            self.conn.send(request)

            if self.conn.poll(timeout=30):
                response = self.conn.recv()
                if response.type == ResponseType.ERROR:
                    raise RuntimeError(response.payload.get("error", "Unknown error"))
                return response.payload
            else:
                raise TimeoutError("Tokenize timeout")

    def stop_generation(self):
        """Signal the worker to stop current generation."""
        if self.stop_event:
            self.stop_event.set()
            logger.info(f"Stop signal sent to model: {self.model_name}")

    def clear_stop(self):
        """Clear the stop signal for new generation."""
        if self.stop_event:
            self.stop_event.clear()

    def shutdown(self):
        """Gracefully shutdown subprocess - releases CUDA memory."""
        if self.process is None:
            return

        logger.info(f"Shutting down subprocess for model: {self.model_name}")

        if self.process.is_alive() and self.conn:
            try:
                request = Request(command=Command.SHUTDOWN)
                self.conn.send(request)
                self.process.join(timeout=5)
            except Exception as e:
                logger.warning(f"Error during graceful shutdown: {e}")

        if self.process.is_alive():
            logger.warning("Subprocess did not exit gracefully, terminating")
            self.process.terminate()
            self.process.join(timeout=2)

        if self.process.is_alive():
            logger.warning("Subprocess did not terminate, killing")
            self.process.kill()
            self.process.join(timeout=1)

        self.process = None
        self.conn = None
        logger.info(f"Subprocess shutdown complete for model: {self.model_name}")


class ModelProxyManager:
    """Manages model proxies - only one subprocess at a time."""

    def __init__(self, config: dict):
        self.config = config
        self.models: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.active_proxy: Optional[ModelProxy] = None
        self.active_model: Optional[str] = None
        self.default_model: Optional[str] = None
        self.loading = False  # True while a model is being loaded

        self._discover_models()

    def _discover_models(self):
        """Discover all .gguf files in the models directory."""
        models_dir = self.config["model_manager"]["models_directory"]

        if not os.path.exists(models_dir):
            logger.error(f"Models directory does not exist: {models_dir}")
            return

        gguf_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]

        if not gguf_files:
            logger.warning(f"No .gguf files found in {models_dir}")
            return

        for filename in sorted(gguf_files):
            model_name = filename.replace('.gguf', '')
            self.models[model_name] = {
                "filename": filename,
                "path": os.path.join(models_dir, filename)
            }

        default_filename = self.config["model_manager"].get("default_model")
        if default_filename:
            self.default_model = default_filename.replace('.gguf', '')
        else:
            self.default_model = list(self.models.keys())[0] if self.models else None

        logger.info(f"Discovered {len(gguf_files)} models: {', '.join(sorted(gguf_files))}")
        if self.default_model:
            logger.info(f"Default model: {self.default_model}")

    def get_model(self, model_name: str = None) -> ModelProxy:
        """Get proxy for model, switching subprocess if needed."""
        if model_name is None:
            model_name = self.default_model

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry")

        with self.lock:
            if self.active_model == model_name and self.active_proxy and self.active_proxy.is_alive():
                self.active_proxy.last_used = time.time()
                return self.active_proxy

            if self.active_proxy is not None:
                logger.info(f"Switching from {self.active_model} to {model_name}")
                self.active_proxy.shutdown()

            model_info = self.models[model_name]
            self.active_proxy = ModelProxy(
                model_name=model_name,
                model_path=model_info["path"],
                config=self.config
            )
            self.loading = True
            try:
                self.active_proxy.start()
                self.active_model = model_name
            finally:
                self.loading = False

            return self.active_proxy

    def unload_model(self, model_name: str = None) -> bool:
        """Unload active model (kill subprocess)."""
        with self.lock:
            if model_name and model_name != self.active_model:
                logger.info(f"Model '{model_name}' is not loaded")
                return False

            if self.active_proxy is not None:
                self.active_proxy.shutdown()
                self.active_proxy = None
                self.active_model = None
                return True

            return False

    def stop_generation(self) -> bool:
        """Stop current generation on active model."""
        with self.lock:
            if self.active_proxy is not None:
                self.active_proxy.stop_generation()
                return True
            return False

    def unload_all_models(self) -> int:
        """Unload active model."""
        if self.unload_model():
            return 1
        return 0

    def unload_idle_models(self, timeout: float) -> list:
        """Unload model if idle longer than timeout."""
        with self.lock:
            if self.active_proxy is None:
                return []

            idle_time = time.time() - self.active_proxy.last_used
            if idle_time > timeout:
                model_name = self.active_model
                logger.info(f"Auto-unloading '{model_name}' (idle for {idle_time:.1f}s)")
                self.active_proxy.shutdown()
                self.active_proxy = None
                self.active_model = None
                return [model_name]

            return []

    def list_models(self) -> list:
        """Return list of available models with their status."""
        models_list = []
        for model_name in self.models:
            models_list.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "loaded": model_name == self.active_model
            })
        return models_list

    def get_model_status(self) -> dict:
        """Return status of all models."""
        status = {}
        for model_name, model_info in self.models.items():
            is_loaded = model_name == self.active_model
            status[model_name] = {
                "loaded": is_loaded,
                "last_used": self.active_proxy.last_used if is_loaded and self.active_proxy else 0,
                "filename": model_info["filename"]
            }
        return status

    def update_config(self, new_config: dict):
        """Update configuration and rediscover models."""
        with self.lock:
            if self.active_proxy:
                self.active_proxy.shutdown()
                self.active_proxy = None
                self.active_model = None

            self.config = new_config
            self.models.clear()
            self._discover_models()
