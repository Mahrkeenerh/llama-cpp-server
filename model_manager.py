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

        # Initialize model registry from config
        for model_name, model_config in config["models"].items():
            self.models[model_name] = {
                "instance": None,
                "last_used": 0,
                "config": model_config
            }
            self.locks[model_name] = threading.Lock()

        # Determine default model
        self.default_model = None
        for model_name, model_config in config["models"].items():
            if model_config.get("default", False):
                self.default_model = model_name
                break

        if not self.default_model and self.models:
            self.default_model = list(self.models.keys())[0]

        logger.info(f"ModelManager initialized with {len(self.models)} models")
        if self.default_model:
            logger.info(f"Default model: {self.default_model}")

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
                    model_data["config"]["file"]
                )

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                start_time = time.time()
                model_data["instance"] = Llama(
                    model_path=model_path,
                    n_ctx=model_data["config"]["n_ctx"],
                    n_gpu_layers=model_data["config"]["n_gpu_layers"],
                    n_threads=model_data["config"].get("n_threads", 8),
                    verbose=False
                )
                load_time = time.time() - start_time
                logger.info(f"Model '{model_name}' loaded in {load_time:.2f}s")

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
                "config": model_data["config"]
            }
        return status
