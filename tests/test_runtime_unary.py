"""Unit tests for the generic runtime unary primitive."""

import asyncio

import pytest

from runtime.session import BackendRuntimeSession


class _FakeBackend:
    """Stand-in for RemoteBackendSession capturing unary dispatch."""

    def __init__(self):
        self.session_id = "sess_test"
        self.calls = []

    async def unary(self, method, payload=None):
        self.calls.append((method, payload))
        if method in {"close", "backend.close", "session.close"}:
            return {"type": "session.closed", "session_id": self.session_id,
                    "reason": (payload or {}).get("reason", "client_closed")}
        raise ValueError(f"unsupported backend unary method: {method}")


def _runtime_with_fake():
    runtime = BackendRuntimeSession.__new__(BackendRuntimeSession)
    runtime.backend = _FakeBackend()
    return runtime


def test_unary_close_dispatches_and_wraps_event():
    runtime = _runtime_with_fake()
    event = asyncio.run(runtime.unary("close", {"reason": "turn_done"}))

    assert runtime.backend.calls == [("close", {"reason": "turn_done"})]
    assert event.channel == "session"
    assert event.payload["type"] == "session.closed"
    assert event.payload["reason"] == "turn_done"


def test_unary_unknown_method_raises():
    runtime = _runtime_with_fake()
    with pytest.raises(ValueError):
        asyncio.run(runtime.unary("does_not_exist", {}))


def test_backend_runtime_session_has_no_standalone_close():
    # close is now folded into unary("close"); the dedicated method must be gone.
    assert "close" not in BackendRuntimeSession.__dict__
