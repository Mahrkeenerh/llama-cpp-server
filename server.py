#!/usr/bin/env python3
import os
import time
import json
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from model_proxy import ModelProxyManager
from tasks import start_idle_monitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Filter to suppress health check request logs
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        # Suppress werkzeug logs for /health endpoint
        return '/health' not in record.getMessage()


# Apply filter to werkzeug logger to reduce log spam
logging.getLogger('werkzeug').addFilter(HealthCheckFilter())

# Initialize Flask app
app = Flask(__name__)

# Global variables
model_manager = None
config = None
start_time = time.time()


# Chat templates for different models
CHAT_TEMPLATES = {
    "qwen3-14b-q6_K": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n"
    },
    "qwen2.5:14b-instruct-q6_K": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n"
    },
    "llama-3-8b-instruct": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n"
    }
}


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def format_chat_prompt(messages, model_name):
    """Convert OpenAI messages format to model-specific prompt."""
    template = CHAT_TEMPLATES.get(model_name)

    if not template:
        # Fallback to basic template
        template = CHAT_TEMPLATES["qwen3-14b-q6_K"]

    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            prompt += template["system"].format(content=content)
        elif role == "user":
            prompt += template["user"].format(content=content)
        elif role == "assistant":
            prompt += template["assistant"].format(content=content)

    # Add assistant start token
    prompt += template["assistant_start"]

    return prompt


def format_openai_response(response, model_name):
    """Format llama.cpp response to OpenAI format."""
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response["choices"][0]["text"]
            },
            "finish_reason": response["choices"][0].get("finish_reason", "stop")
        }],
        "usage": {
            "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
        }
    }


def generate_stream(proxy, prompt, temperature, max_tokens, model_name):
    """Generate streaming response via subprocess."""
    try:
        for chunk_data in proxy.generate_stream(prompt, temperature, max_tokens):
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": chunk_data["text"]
                    },
                    "finish_reason": chunk_data.get("finish_reason")
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        error_chunk = {"error": str(e)}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.json
        model_name = data.get("model", model_manager.default_model)
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 2048)

        if not messages:
            return jsonify({"error": "messages field is required"}), 400

        # Get model proxy (spawns subprocess if needed)
        try:
            proxy = model_manager.get_model(model_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except (RuntimeError, TimeoutError) as e:
            return jsonify({"error": str(e)}), 500

        # Convert messages to prompt
        prompt = format_chat_prompt(messages, model_name)

        if stream:
            return Response(
                stream_with_context(generate_stream(proxy, prompt, temperature, max_tokens, model_name)),
                content_type='text/event-stream'
            )
        else:
            result = proxy.generate(prompt, temperature=temperature, max_tokens=max_tokens)
            response = {
                "choices": [{"text": result["text"], "finish_reason": result.get("finish_reason", "stop")}],
                "usage": result.get("usage", {})
            }
            return jsonify(format_openai_response(response, model_name))

    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models."""
    try:
        models = model_manager.list_models()
        return jsonify({
            "object": "list",
            "data": models
        })
    except Exception as e:
        logger.error(f"Error in list_models: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        loaded_models = [
            name for name, status in model_manager.get_model_status().items()
            if status["loaded"]
        ]

        return jsonify({
            "status": "ok",
            "loaded_models": loaded_models,
            "uptime": int(time.time() - start_time)
        })
    except Exception as e:
        logger.error(f"Error in health: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/admin/unload', methods=['POST'])
def admin_unload():
    """Manually unload a model or all models."""
    try:
        data = request.json
        model_name = data.get("model")

        if not model_name:
            return jsonify({"error": "model field is required"}), 400

        if model_name == "all":
            count = model_manager.unload_all_models()
            return jsonify({
                "status": "success",
                "message": f"Unloaded {count} models"
            })
        else:
            try:
                model_manager.unload_model(model_name)
                return jsonify({
                    "status": "success",
                    "message": f"Model '{model_name}' unloaded"
                })
            except ValueError as e:
                return jsonify({"error": str(e)}), 404

    except Exception as e:
        logger.error(f"Error in admin_unload: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/admin/reload', methods=['POST'])
def admin_reload():
    """Reload configuration without restarting the server."""
    try:
        global config, model_manager

        # Load new configuration
        new_config = load_config("config.json")

        # Update global config
        config = new_config

        # Update model manager with new config
        model_manager.update_config(new_config)

        logger.info("Configuration reloaded successfully")
        return jsonify({
            "status": "success",
            "message": "Configuration reloaded successfully",
            "config": new_config
        })

    except Exception as e:
        logger.error(f"Error in admin_reload: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def main():
    """Initialize and run the server."""
    global model_manager, config

    # Load configuration
    config = load_config("config.json")

    # Setup CORS
    CORS(app, origins=config["server"]["cors_origins"])

    # Initialize model manager (subprocess-based)
    model_manager = ModelProxyManager(config)

    # Start background tasks
    start_idle_monitor(
        model_manager,
        config["model_manager"]["check_interval"],
        config["model_manager"]["idle_timeout"]
    )

    # Run server
    logger.info(f"Starting server on {config['server']['host']}:{config['server']['port']}")
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        threaded=True
    )


if __name__ == '__main__':
    main()
