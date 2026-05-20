import asyncio

from core.runtime.manager import RuntimeManager
from tests.test_duplex_runtime import _FakeWorker
from core.runtime.backends import WorkerDuplexBackendAdapter


def test_runtime_manager_creates_and_closes_duplex_runtime():
    async def _run():
        manager = RuntimeManager()
        worker = _FakeWorker()
        runtime = manager.create_duplex("s1", WorkerDuplexBackendAdapter(worker))

        assert manager.get_duplex("s1") is runtime

        await manager.close_duplex("s1")

        assert worker.calls[-2:] == ["stop", "cleanup"]

    asyncio.run(_run())


def test_runtime_manager_rejects_duplicate_session_id():
    manager = RuntimeManager()
    worker = _FakeWorker()
    manager.create_duplex("s1", WorkerDuplexBackendAdapter(worker))

    try:
        manager.create_duplex("s1", WorkerDuplexBackendAdapter(worker))
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected duplicate session id to fail")


def test_runtime_manager_can_forget_already_closed_runtime():
    manager = RuntimeManager()
    worker = _FakeWorker()
    manager.create_duplex("s1", WorkerDuplexBackendAdapter(worker))

    manager.forget_duplex("s1")

    assert worker.calls == []

