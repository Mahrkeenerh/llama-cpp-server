"""Model worker subprocess - owns Llama instance and CUDA memory."""
import os
import sys
import logging
from llama_cpp import Llama
from ipc_protocol import Command, ResponseType, Request, Response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - worker - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelWorker:
    """Runs in subprocess, owns Llama instance and CUDA memory."""

    def __init__(self, pipe_conn, model_name: str, config: dict):
        self.conn = pipe_conn
        self.model_name = model_name
        self.config = config
        self.llm = None

    def run(self):
        """Main loop - listen for commands, execute, respond."""
        logger.info(f"Worker started for model: {self.model_name}")

        while True:
            try:
                request = self.conn.recv()
                self._handle_request(request)

                if request.command == Command.SHUTDOWN:
                    logger.info("Shutdown command received, exiting")
                    break

            except EOFError:
                logger.info("Pipe closed, exiting")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                try:
                    self.conn.send(Response(
                        id=getattr(request, 'id', 'unknown'),
                        type=ResponseType.ERROR,
                        payload={"error": str(e)}
                    ))
                except:
                    pass

        logger.info("Worker exiting")

    def _handle_request(self, request: Request):
        """Route request to appropriate handler."""
        try:
            if request.command == Command.LOAD:
                self._handle_load(request)
            elif request.command == Command.GENERATE:
                self._handle_generate(request)
            elif request.command == Command.GENERATE_STREAM:
                self._handle_stream(request)
            elif request.command == Command.STATUS:
                self._handle_status(request)
            elif request.command == Command.SHUTDOWN:
                self._handle_shutdown(request)
            elif request.command == Command.HEARTBEAT:
                self._handle_heartbeat(request)
            else:
                self.conn.send(Response(
                    id=request.id,
                    type=ResponseType.ERROR,
                    payload={"error": f"Unknown command: {request.command}"}
                ))
        except Exception as e:
            logger.error(f"Error handling {request.command}: {e}", exc_info=True)
            self.conn.send(Response(
                id=request.id,
                type=ResponseType.ERROR,
                payload={"error": str(e)}
            ))

    def _handle_load(self, request: Request):
        """Load the model into memory."""
        if self.llm is not None:
            self.conn.send(Response(
                id=request.id,
                type=ResponseType.RESULT,
                payload={"status": "already_loaded", "model": self.model_name}
            ))
            return

        model_path = request.payload["model_path"]
        n_ctx = request.payload.get("n_ctx", 4096)
        n_gpu_layers = request.payload.get("n_gpu_layers", -1)
        n_threads = request.payload.get("n_threads", 8)

        logger.info(f"Loading model: {model_path}")
        logger.info(f"Config: n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}, n_threads={n_threads}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False
        )

        logger.info(f"Model loaded: {self.model_name}")
        self.conn.send(Response(
            id=request.id,
            type=ResponseType.RESULT,
            payload={"status": "loaded", "model": self.model_name}
        ))

    def _handle_generate(self, request: Request):
        """Non-streaming generation."""
        if self.llm is None:
            self.conn.send(Response(
                id=request.id,
                type=ResponseType.ERROR,
                payload={"error": "Model not loaded"}
            ))
            return

        prompt = request.payload["prompt"]
        temperature = request.payload.get("temperature", 0.7)
        max_tokens = request.payload.get("max_tokens", 2048)

        response = self.llm(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        self.conn.send(Response(
            id=request.id,
            type=ResponseType.RESULT,
            payload={
                "text": response["choices"][0]["text"],
                "finish_reason": response["choices"][0].get("finish_reason", "stop"),
                "usage": response.get("usage", {})
            }
        ))

    def _handle_stream(self, request: Request):
        """Streaming generation - sends multiple CHUNK responses."""
        if self.llm is None:
            self.conn.send(Response(
                id=request.id,
                type=ResponseType.ERROR,
                payload={"error": "Model not loaded"}
            ))
            return

        prompt = request.payload["prompt"]
        temperature = request.payload.get("temperature", 0.7)
        max_tokens = request.payload.get("max_tokens", 2048)

        stream = self.llm(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        for output in stream:
            text = output["choices"][0]["text"]
            finish_reason = output["choices"][0].get("finish_reason")

            self.conn.send(Response(
                id=request.id,
                type=ResponseType.CHUNK,
                payload={"text": text, "finish_reason": finish_reason}
            ))

        self.conn.send(Response(
            id=request.id,
            type=ResponseType.DONE,
            payload={"finish_reason": "stop"}
        ))

    def _handle_status(self, request: Request):
        """Return worker status."""
        self.conn.send(Response(
            id=request.id,
            type=ResponseType.RESULT,
            payload={
                "model": self.model_name,
                "loaded": self.llm is not None,
                "pid": os.getpid()
            }
        ))

    def _handle_shutdown(self, request: Request):
        """Graceful shutdown."""
        logger.info("Shutting down worker")
        self.llm = None
        self.conn.send(Response(
            id=request.id,
            type=ResponseType.RESULT,
            payload={"status": "shutdown"}
        ))

    def _handle_heartbeat(self, request: Request):
        """Respond to heartbeat check."""
        self.conn.send(Response(
            id=request.id,
            type=ResponseType.RESULT,
            payload={"status": "alive", "pid": os.getpid()}
        ))


def worker_main(pipe_conn, model_name: str, config: dict):
    """Entry point called by multiprocessing.Process."""
    worker = ModelWorker(pipe_conn, model_name, config)
    worker.run()
