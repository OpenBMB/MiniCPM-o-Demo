import asyncio

from gateway_modules.worker_pool import WorkerPool
from gateway_modules.models import GatewayWorkerStatus


def _make_pool(num_workers: int = 2) -> WorkerPool:
    pool = WorkerPool(
        worker_addresses=[f"localhost:{22400 + i}" for i in range(num_workers)],
        max_queue_size=10,
    )
    for worker in pool.workers.values():
        worker.status = GatewayWorkerStatus.IDLE
    return pool


def test_immediate_assignment_respects_worker_capabilities():
    async def _run():
        pool = _make_pool(2)
        workers = list(pool.workers.values())
        workers[0].capabilities = ["chat"]
        workers[1].capabilities = ["omni_duplex"]

        _, future = pool.enqueue("omni_duplex")

        assert future.done()
        assert future.result() == workers[1]

    asyncio.run(_run())


def test_unsupported_request_waits_even_when_other_worker_idle():
    async def _run():
        pool = _make_pool(1)
        worker = list(pool.workers.values())[0]
        worker.capabilities = ["chat"]

        ticket, future = pool.enqueue("audio_duplex")

        assert not future.done()
        assert ticket.position == 1
        assert pool.queue_length == 1

    asyncio.run(_run())


def test_dispatch_skips_unsupported_head_of_line_request():
    async def _run():
        pool = _make_pool(1)
        worker = list(pool.workers.values())[0]
        worker.capabilities = ["chat"]
        worker.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat")

        audio_ticket, audio_future = pool.enqueue("audio_duplex")
        chat_ticket, chat_future = pool.enqueue("chat")

        pool.release_worker(worker, "chat", 1.0)

        assert not audio_future.done()
        assert audio_ticket.position == 1
        assert chat_future.done()
        assert chat_future.result() == worker
        assert chat_ticket.position == 0
        assert pool.queue_length == 1

    asyncio.run(_run())


def test_dynamic_registration_dispatches_waiting_request():
    async def _run():
        pool = WorkerPool(worker_addresses=[], max_queue_size=10)

        async def fake_refresh(worker):
            worker.status = GatewayWorkerStatus.IDLE
            worker.capabilities = ["chat"]

        pool._refresh_worker_status = fake_refresh

        ticket, future = pool.enqueue("chat")
        assert not future.done()
        assert ticket.position == 1

        worker = await pool.register_worker(
            worker_id="worker-a",
            endpoint="127.0.0.1:22400",
            gpu_group="gpu-0",
        )

        assert future.done()
        assert future.result() == worker
        assert ticket.position == 0
        assert pool.queue_length == 0

    asyncio.run(_run())
