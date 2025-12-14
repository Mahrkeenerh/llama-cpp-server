import os
import time
import threading
import gc
import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, unloading, and lifecycle."""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.locks = {}
        self.global_lock = threading.Lock()

        # Discover models from directory
        self._discover_models()

        logger.info(f"ModelManager initialized with {len(self.models)} models")
        if self.default_model:
            logger.info(f"Default model: {self.default_model}")

    def _discover_models(self):
        """Discover all .gguf files in the models directory."""
        models_dir = self.config["model_manager"]["models_directory"]

        if not os.path.exists(models_dir):
            logger.error(f"Models directory does not exist: {models_dir}")
            self.default_model = None
            return

        # Find all .gguf files
        gguf_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]

        if not gguf_files:
            logger.warning(f"No .gguf files found in {models_dir}")
            self.default_model = None
            return

        # Create model entries
        for filename in sorted(gguf_files):
            model_name = filename.replace('.gguf', '')
            if model_name not in self.models:
                self.models[model_name] = {
                    "instance": None,
                    "last_used": 0,
                    "filename": filename
                }
                self.locks[model_name] = threading.Lock()

        # Set default model
        default_filename = self.config["model_manager"].get("default_model")
        if default_filename:
            self.default_model = default_filename.replace('.gguf', '')
        else:
            self.default_model = list(self.models.keys())[0] if self.models else None

        logger.info(f"Discovered {len(gguf_files)} models: {', '.join(sorted(gguf_files))}")

    def get_model(self, model_name=None):
        """Get model instance, loading it if necessary."""
        if model_name is None:
            model_name = self.default_model

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry")

        with self.locks[model_name]:
            model_data = self.models[model_name]

            # Lazy load if not loaded
            if model_data["instance"] is None:
                logger.info(f"Loading model: {model_name}")
                model_path = os.path.join(
                    self.config["model_manager"]["models_directory"],
                    model_data["filename"]
                )

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                # Use global settings from config
                start_time = time.time()
                try:
                    model_data["instance"] = Llama(
                        model_path=model_path,
                        n_ctx=self.config["model_manager"]["n_ctx"],
                        n_gpu_layers=self.config["model_manager"]["n_gpu_layers"],
                        n_threads=self.config["model_manager"].get("n_threads", 8),
                        verbose=False
                    )
                    load_time = time.time() - start_time
                    logger.info(f"Model '{model_name}' loaded in {load_time:.2f}s")
                except Exception as e:
                    logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
                    logger.error(f"Model config: n_ctx={self.config['model_manager']['n_ctx']}, n_gpu_layers={self.config['model_manager']['n_gpu_layers']}, n_threads={self.config['model_manager'].get('n_threads', 8)}")
                    raise ValueError(f"Failed to create llama_context: {e}")

            # Update last used timestamp
            model_data["last_used"] = time.time()
            return model_data["instance"]

    def unload_model(self, model_name):
        """Explicitly unload a model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry")

        with self.locks[model_name]:
            if self.models[model_name]["instance"] is not None:
                logger.info(f"Unloading model: {model_name}")
                self.models[model_name]["instance"] = None
                gc.collect()
                logger.info(f"Model '{model_name}' unloaded")
                return True
            else:
                logger.info(f"Model '{model_name}' is not loaded")
                return False

    def unload_all_models(self):
        """Unload all loaded models."""
        count = 0
        for model_name in list(self.models.keys()):
            if self.unload_model(model_name):
                count += 1
        logger.info(f"Unloaded {count} models")
        return count

    def unload_idle_models(self, timeout):
        """Unload models that have been idle for longer than timeout."""
        current_time = time.time()
        unloaded = []

        for model_name in list(self.models.keys()):
            with self.locks[model_name]:
                model_data = self.models[model_name]
                if model_data["instance"] is not None:
                    idle_time = current_time - model_data["last_used"]
                    if idle_time > timeout:
                        logger.info(f"Auto-unloading '{model_name}' (idle for {idle_time:.1f}s)")
                        model_data["instance"] = None
                        gc.collect()
                        unloaded.append(model_name)

        return unloaded

    def list_models(self):
        """Return list of available models with their status."""
        models_list = []
        for model_name, model_data in self.models.items():
            models_list.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "loaded": model_data["instance"] is not None
            })
        return models_list

    def get_model_status(self):
        """Return status of all models."""
        status = {}
        for model_name, model_data in self.models.items():
            status[model_name] = {
                "loaded": model_data["instance"] is not None,
                "last_used": model_data["last_used"],
                "filename": model_data["filename"]
            }
        return status

    def update_config(self, new_config):
        """Update configuration and rediscover models from directory."""
        with self.global_lock:
            # Store current models
            current_models = set(self.models.keys())

            # Update config
            self.config = new_config

            # Rediscover models from directory
            models_dir = self.config["model_manager"]["models_directory"]
            if not os.path.exists(models_dir):
                logger.error(f"Models directory does not exist: {models_dir}")
                return

            # Find all .gguf files
            gguf_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
            new_models = set(f.replace('.gguf', '') for f in gguf_files)

            # Remove models that are no longer in directory
            for model_name in current_models - new_models:
                logger.info(f"Removing model from registry (file deleted): {model_name}")
                self.unload_model(model_name)
                del self.models[model_name]
                del self.locks[model_name]

            # Add new models found in directory
            for model_name in new_models - current_models:
                filename = model_name + '.gguf'
                logger.info(f"Adding new model to registry: {filename}")
                self.models[model_name] = {
                    "instance": None,
                    "last_used": 0,
                    "filename": filename
                }
                self.locks[model_name] = threading.Lock()

            # Update default model
            default_filename = self.config["model_manager"].get("default_model")
            if default_filename:
                self.default_model = default_filename.replace('.gguf', '')
            else:
                self.default_model = list(self.models.keys())[0] if self.models else None

            logger.info(f"Configuration updated. Active models: {len(self.models)}")
            if self.default_model:
                logger.info(f"Default model: {self.default_model}")
            logger.info(f"Discovered models: {', '.join(sorted(f + '.gguf' for f in self.models.keys()))}")
