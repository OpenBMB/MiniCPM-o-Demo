"""Runtime event primitives.

These are worker-internal events.  They are deliberately smaller than the public
API protocol and can be translated to legacy WebSocket payloads, OpenAI-style
Realtime events, recording entries, or metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Protocol


RuntimeChannel = Literal[
    "session",
    "input.audio",
    "input.video",
    "output.text",
    "output.audio",
    "output.duplex_result",
    "response",
    "model.state",
    "metrics",
    "backend",
    "error",
]

RuntimeControlType = Literal[
    "session.pause",
    "session.resume",
    "session.close",
    "response.cancel",
    "legacy.interrupt",
]


@dataclass
class RuntimeEvent:
    channel: RuntimeChannel
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeControl:
    type: RuntimeControlType
    payload: Dict[str, Any] = field(default_factory=dict)


class OutputSink(Protocol):
    async def emit(self, event: RuntimeEvent) -> None:
        ...
