import asyncio

from core.runtime.events import RuntimeEvent
from core.runtime.sinks import CompositeSink, LegacyDuplexWebSocketSink


class _FakeWs:
    def __init__(self):
        self.messages = []

    async def send_json(self, msg):
        self.messages.append(msg)


class _CollectSink:
    def __init__(self):
        self.events = []

    async def emit(self, event):
        self.events.append(event)


def test_legacy_duplex_websocket_sink_sends_legacy_result_payload():
    async def _run():
        ws = _FakeWs()
        sink = LegacyDuplexWebSocketSink(ws)

        await sink.emit(RuntimeEvent(
            channel="output.duplex_result",
            payload={"result_dict": {"is_listen": False, "text": "hi"}},
        ))

        assert ws.messages == [{"type": "result", "is_listen": False, "text": "hi"}]

    asyncio.run(_run())


def test_composite_sink_fans_out_events_in_order():
    async def _run():
        a = _CollectSink()
        b = _CollectSink()
        sink = CompositeSink([a, b])
        event = RuntimeEvent(channel="metrics", payload={"x": 1})

        await sink.emit(event)

        assert a.events == [event]
        assert b.events == [event]

    asyncio.run(_run())

