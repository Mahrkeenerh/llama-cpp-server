"""IPC protocol definitions for subprocess communication."""
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid


class Command(Enum):
    """Commands sent from main process to worker subprocess."""
    LOAD = "load"
    GENERATE = "generate"
    GENERATE_STREAM = "generate_stream"
    TOKENIZE = "tokenize"
    STATUS = "status"
    SHUTDOWN = "shutdown"
    HEARTBEAT = "heartbeat"


class ResponseType(Enum):
    """Response types sent from worker subprocess to main process."""
    RESULT = "result"
    CHUNK = "chunk"
    DONE = "done"
    ERROR = "error"


@dataclass
class Request:
    """Request message sent to worker subprocess."""
    command: Command
    payload: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Response:
    """Response message sent from worker subprocess."""
    id: str
    type: ResponseType
    payload: dict = field(default_factory=dict)
