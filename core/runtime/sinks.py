"""Runtime output sinks."""

from __future__ import annotations

from typing import Iterable

from core.runtime.events import OutputSink, RuntimeEvent
from core.runtime.legacy_duplex import legacy_result_payload_from_event


class CompositeSink:
    """Fan out runtime events to multiple sinks in order."""

    def __init__(self, sinks: Iterable[OutputSink]):
        self._sinks = list(sinks)

    async def emit(self, event: RuntimeEvent) -> None:
        for sink in self._sinks:
            await sink.emit(event)


class LegacyDuplexWebSocketSink:
    """Emit runtime duplex events using the legacy /ws/duplex payload shape."""

    def __init__(self, ws):
        self.ws = ws

    async def emit(self, event: RuntimeEvent) -> None:
        await self.ws.send_json(legacy_result_payload_from_event(event))

