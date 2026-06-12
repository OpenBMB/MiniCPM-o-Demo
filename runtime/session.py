"""Protocol-shaped runtime sessions."""

from __future__ import annotations

from typing import Any, Dict, Optional

from runtime.backend_client import RemoteBackendSession
from runtime.events import RuntimeEvent


def backend_event_to_runtime_event(event: Dict[str, Any]) -> RuntimeEvent:
    """Keep runtime close to backend protocol while preserving channel hints."""

    event_type = str(event.get("type") or "")
    payload = {"event": event, **event}

    if event_type.startswith("session."):
        return RuntimeEvent(channel="session", payload=payload)

    if event_type == "response.done":
        return RuntimeEvent(channel="response", payload=payload)

    if event_type == "response.output.delta":
        kind = event.get("kind")
        if kind == "text":
            return RuntimeEvent(channel="output.text", payload=payload)
        if kind == "audio":
            return RuntimeEvent(channel="output.audio", payload=payload)
        if kind == "listen":
            return RuntimeEvent(channel="model.state", payload=payload)

    return RuntimeEvent(channel="backend", payload=payload)


class BackendRuntimeSession:
    """Runtime facade over a remote backend session.

    This class intentionally does not implement model semantics.  It exists to
    make the worker/gateway side talk in init/push/pull/close primitives while
    BackendClient owns the network transport.
    """

    def __init__(
        self,
        *,
        backend_base_url: str,
        mode: str,
        session_id: Optional[str] = None,
    ) -> None:
        self.backend = RemoteBackendSession(
            base_url=backend_base_url,
            mode=mode,
            session_id=session_id,
        )

    @property
    def session_id(self) -> Optional[str]:
        return self.backend.session_id

    async def init(self, params: Optional[Dict[str, Any]] = None) -> RuntimeEvent:
        event = await self.backend.init(params)
        return backend_event_to_runtime_event(event)

    async def push(self, input_payload: Dict[str, Any]) -> None:
        await self.backend.push(input_payload)

    async def pull(self) -> RuntimeEvent:
        event = await self.backend.pull()
        return backend_event_to_runtime_event(event)

    async def unary(self, method: str, payload: Optional[Dict[str, Any]] = None) -> RuntimeEvent:
        event = await self.backend.unary(method, payload)
        return RuntimeEvent(channel="session", payload={"event": event, **event})

    async def aclose(self) -> None:
        await self.backend.aclose()
