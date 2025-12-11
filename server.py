#!/usr/bin/env python3
import os
import time
import json
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from model_manager import ModelManager
from tasks import start_idle_monitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


def generate_stream(llm, prompt, temperature, max_tokens):
    """Generate streaming response."""
    stream = llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    for output in stream:
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "local",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": output["choices"][0]["text"]
                },
                "finish_reason": output["choices"][0].get("finish_reason")
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


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

        # Get model (loads if needed)
        try:
            llm = model_manager.get_model(model_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 500

        # Convert messages to prompt
        prompt = format_chat_prompt(messages, model_name)

        if stream:
            return Response(
                stream_with_context(generate_stream(llm, prompt, temperature, max_tokens)),
                content_type='text/event-stream'
            )
        else:
            response = llm(prompt, temperature=temperature, max_tokens=max_tokens)
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


def main():
    """Initialize and run the server."""
    global model_manager, config

    # Load configuration
    config = load_config("config.json")

    # Setup CORS
    CORS(app, origins=config["server"]["cors_origins"])

    # Initialize model manager
    model_manager = ModelManager(config)

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
