"""队列引擎单元测试

无需 GPU，纯逻辑测试。覆盖：
- FIFO 顺序保证
- 容量限制
- 取消机制
- 位置追踪
- ETA 估算
- Worker 释放触发调度
- 并发安全
- 即时分配（空队列 + 空闲 Worker）
- Streaming LRU 缓存命中路由
- EMA 动态 ETA
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Optional

from gateway_modules.models import GatewayWorkerStatus, EtaConfig
from gateway_modules.worker_pool import WorkerPool, WorkerConnection


# ============ Fixtures ============

def make_pool(
    num_workers: int = 3,
    max_queue_size: int = 1000,
    all_idle: bool = True,
) -> WorkerPool:
    """创建测试用 WorkerPool（不启动健康检查）"""
    addresses = [f"localhost:{22400 + i}" for i in range(num_workers)]
    pool = WorkerPool(
        worker_addresses=addresses,
        max_queue_size=max_queue_size,
        eta_config=EtaConfig(eta_chat_s=10.0, eta_half_duplex_s=15.0, eta_omni_duplex_s=90.0),
    )
    if all_idle:
        for w in pool.workers.values():
            w.status = GatewayWorkerStatus.IDLE
    return pool


def make_workers_busy(pool: WorkerPool, count: Optional[int] = None,
                      request_type: str = "omni_duplex") -> None:
    """将指定数量的 Worker 标记为忙碌"""
    n = count or len(pool.workers)
    for i, w in enumerate(pool.workers.values()):
        if i >= n:
            break
        w.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, request_type)


# ============ 1. FIFO 顺序 ============

class TestFIFO:
    """验证排队顺序是严格 FIFO"""

    @pytest.mark.asyncio
    async def test_fifo_order_3_requests(self) -> None:
        """3 个请求入队，按 FIFO 顺序分配"""
        pool = make_pool(num_workers=1, all_idle=True)
        # 先占满唯一 Worker
        make_workers_busy(pool, count=1)

        t1, f1 = pool.enqueue("chat")
        t2, f2 = pool.enqueue("half_duplex_audio")
        t3, f3 = pool.enqueue("omni_duplex")

        assert t1.position == 1
        assert t2.position == 2
        assert t3.position == 3

        # 释放 Worker → 队头 t1 应该获得
        w = list(pool.workers.values())[0]
        pool.release_worker(w, "omni_duplex", 5.0)

        assert f1.done()
        assert not f2.done()
        assert not f3.done()
        assert f1.result() == w

    @pytest.mark.asyncio
    async def test_fifo_sequential_dispatch(self) -> None:
        """依次释放 Worker，依次分配"""
        pool = make_pool(num_workers=1, all_idle=True)
        make_workers_busy(pool, count=1)

        t1, f1 = pool.enqueue("chat")
        t2, f2 = pool.enqueue("chat")

        w = list(pool.workers.values())[0]

        # 释放 → t1 获得
        pool.release_worker(w, "chat", 2.0)
        assert f1.done()
        assert not f2.done()

        # 标记忙后释放 → t2 获得
        w.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat")
        pool.release_worker(w, "chat", 2.0)
        assert f2.done()


# ============ 2. 容量限制 ============

class TestCapacity:
    """验证队列容量限制"""

    @pytest.mark.asyncio
    async def test_queue_full_reject(self) -> None:
        """队列满后拒绝新请求"""
        pool = make_pool(num_workers=1, max_queue_size=3, all_idle=True)
        make_workers_busy(pool, count=1)

        pool.enqueue("chat")
        pool.enqueue("chat")
        pool.enqueue("chat")

        with pytest.raises(WorkerPool.QueueFullError):
            pool.enqueue("chat")

    @pytest.mark.asyncio
    async def test_queue_length_tracking(self) -> None:
        """队列长度正确追踪"""
        pool = make_pool(num_workers=1, max_queue_size=10, all_idle=True)
        make_workers_busy(pool, count=1)

        assert pool.queue_length == 0
        pool.enqueue("chat")
        assert pool.queue_length == 1
        t2, _ = pool.enqueue("chat")
        assert pool.queue_length == 2

        pool.cancel(t2.ticket_id)
        assert pool.queue_length == 1


# ============ 3. 取消 ============

class TestCancel:
    """验证取消机制"""

    @pytest.mark.asyncio
    async def test_cancel_middle_item(self) -> None:
        """取消中间项，后面的项位置前移"""
        pool = make_pool(num_workers=1, all_idle=True)
        make_workers_busy(pool, count=1)

        t1, f1 = pool.enqueue("chat")
        t2, f2 = pool.enqueue("chat")
        t3, f3 = pool.enqueue("chat")

        assert t1.position == 1
        assert t2.position == 2
        assert t3.position == 3

        # 取消 t2
        ok = pool.cancel(t2.ticket_id)
        assert ok
        assert f2.cancelled()

        # t3 位置前移到 2
        assert t3.position == 2
        assert pool.queue_length == 2

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self) -> None:
        """取消不存在的 ticket 返回 False"""
        pool = make_pool(num_workers=1)
        assert pool.cancel("nonexistent") is False

    @pytest.mark.asyncio
    async def test_cancel_dispatches_next(self) -> None:
        """取消后如果有空闲 Worker，尝试调度下一个"""
        pool = make_pool(num_workers=2, all_idle=True)
        # 占满 2 个 Worker
        make_workers_busy(pool, count=2)

        t1, f1 = pool.enqueue("chat")
        t2, f2 = pool.enqueue("chat")

        # 释放 1 个 Worker → t1 获得
        w0 = list(pool.workers.values())[0]
        pool.release_worker(w0, "chat", 1.0)
        assert f1.done()

        # t2 还在排队，取消 t2 不会有什么问题
        pool.cancel(t2.ticket_id)
        assert f2.cancelled()
        assert pool.queue_length == 0


# ============ 4. 位置追踪 ============

class TestPositionTracking:
    """验证位置在队列变化时正确更新"""

    @pytest.mark.asyncio
    async def test_positions_after_dispatch(self) -> None:
        """分配后剩余项位置前移"""
        pool = make_pool(num_workers=1, all_idle=True)
        make_workers_busy(pool, count=1)

        t1, f1 = pool.enqueue("chat")
        t2, f2 = pool.enqueue("chat")
        t3, f3 = pool.enqueue("chat")

        assert t1.position == 1
        assert t2.position == 2
        assert t3.position == 3

        # 释放 → t1 分配，t2 和 t3 前移
        w = list(pool.workers.values())[0]
        pool.release_worker(w, "chat", 1.0)

        assert f1.done()
        assert t2.position == 1
        assert t3.position == 2


# ============ 5. ETA 估算 ============

class TestETA:
    """验证等待时间估算"""

    @pytest.mark.asyncio
    async def test_eta_basic(self) -> None:
        """基本 ETA 估算"""
        pool = make_pool(num_workers=1, all_idle=True)
        make_workers_busy(pool, count=1, request_type="omni_duplex")

        t1, _ = pool.enqueue("chat")
        # ETA 应该 > 0（前面有 duplex 在运行）
        assert t1.estimated_wait_s > 0

    @pytest.mark.asyncio
    async def test_eta_increases_with_position(self) -> None:
        """位置越靠后，ETA 越大"""
        pool = make_pool(num_workers=1, all_idle=True)
        make_workers_busy(pool, count=1, request_type="chat")

        t1, _ = pool.enqueue("chat")
        t2, _ = pool.enqueue("chat")

        assert t2.estimated_wait_s >= t1.estimated_wait_s

    @pytest.mark.asyncio
    async def test_ema_updates(self) -> None:
        """EMA 在请求完成后更新"""
        pool = make_pool(num_workers=1, all_idle=True)

        # 初始 ETA = 基准值
        assert pool.eta_tracker.get_eta("chat") == 10.0

        # 记录几个实际值
        pool.eta_tracker.record_duration("chat", 5.0)
        pool.eta_tracker.record_duration("chat", 5.0)
        pool.eta_tracker.record_duration("chat", 5.0)

        # 3 个样本后 EMA 生效，应接近 5.0
        ema = pool.eta_tracker.get_eta("chat")
        assert ema < 10.0  # 比基准值小
        assert ema > 4.0   # 但不会太小

    @pytest.mark.asyncio
    async def test_ema_not_active_below_min_samples(self) -> None:
        """EMA 样本不足时使用基准值"""
        pool = make_pool(num_workers=1, all_idle=True)

        pool.eta_tracker.record_duration("chat", 5.0)
        pool.eta_tracker.record_duration("chat", 5.0)

        # 只有 2 个样本（min=3），仍然用基准值
        assert pool.eta_tracker.get_eta("chat") == 10.0


# ============ 6. Worker 释放触发调度 ============

class TestDispatch:
    """验证 Worker 释放后自动调度"""

    @pytest.mark.asyncio
    async def test_release_triggers_dispatch(self) -> None:
        """释放 Worker 后，队头获得 Worker"""
        pool = make_pool(num_workers=2, all_idle=True)
        make_workers_busy(pool, count=2)

        t1, f1 = pool.enqueue("chat")
        assert not f1.done()

        w0 = list(pool.workers.values())[0]
        pool.release_worker(w0, "chat", 1.0)
        assert f1.done()
        assert f1.result() == w0

    @pytest.mark.asyncio
    async def test_multiple_release_multiple_dispatch(self) -> None:
        """释放多个 Worker，调度多个请求"""
        pool = make_pool(num_workers=2, all_idle=True)
        make_workers_busy(pool, count=2)

        t1, f1 = pool.enqueue("chat")
        t2, f2 = pool.enqueue("chat")

        workers = list(pool.workers.values())
        pool.release_worker(workers[0], "chat", 1.0)
        pool.release_worker(workers[1], "chat", 1.0)

        assert f1.done()
        assert f2.done()


# ============ 7. 即时分配 ============

class TestImmediateAssign:
    """验证有空闲 Worker 时立即分配（不入队）"""

    @pytest.mark.asyncio
    async def test_immediate_when_idle(self) -> None:
        """有空闲 Worker 时，Future 立即 resolve"""
        pool = make_pool(num_workers=2, all_idle=True)

        t, f = pool.enqueue("chat")
        assert f.done()
        assert f.result() is not None
        assert t.position == 0  # 0 表示已分配
        assert pool.queue_length == 0

    @pytest.mark.asyncio
    async def test_immediate_does_not_enter_queue(self) -> None:
        """立即分配的请求不进入队列"""
        pool = make_pool(num_workers=3, all_idle=True)

        pool.enqueue("chat")
        pool.enqueue("half_duplex_audio")
        pool.enqueue("omni_duplex")

        assert pool.queue_length == 0


# ============ 8. Streaming LRU 缓存命中路由 ============

class TestStreamingLRU:
    """验证 Streaming 的 LRU 缓存命中路由"""

    @pytest.mark.asyncio
    async def test_cache_hit(self) -> None:
        """缓存匹配时命中"""
        pool = make_pool(num_workers=2, all_idle=True)

        w0 = list(pool.workers.values())[0]
        w0.cached_hash = "hash_A"
        w0.last_cache_used_at = datetime.now()

        t, f = pool.enqueue("streaming", history_hash="hash_A")
        assert f.done()
        assert f.result() == w0

    @pytest.mark.asyncio
    async def test_prefer_no_cache_worker(self) -> None:
        """无缓存匹配时，优先选 cached_hash=None 的 Worker"""
        pool = make_pool(num_workers=2, all_idle=True)

        w0, w1 = list(pool.workers.values())
        w0.cached_hash = "hash_A"
        w0.last_cache_used_at = datetime.now()
        w1.cached_hash = None  # 无缓存

        t, f = pool.enqueue("streaming", history_hash="hash_B")
        assert f.done()
        assert f.result() == w1  # 优先无缓存

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """所有 Worker 都有缓存时，淘汰最旧的"""
        pool = make_pool(num_workers=2, all_idle=True)

        w0, w1 = list(pool.workers.values())
        w0.cached_hash = "hash_A"
        w0.last_cache_used_at = datetime.now() - timedelta(minutes=5)  # 旧
        w1.cached_hash = "hash_B"
        w1.last_cache_used_at = datetime.now()  # 新

        t, f = pool.enqueue("streaming", history_hash="hash_C")
        assert f.done()
        assert f.result() == w0  # 淘汰最旧的

    @pytest.mark.asyncio
    async def test_cache_hit_refreshes_time(self) -> None:
        """缓存命中后，应该由调用者刷新 last_cache_used_at"""
        pool = make_pool(num_workers=2, all_idle=True)

        w0, w1 = list(pool.workers.values())
        old_time = datetime.now() - timedelta(minutes=10)
        w0.cached_hash = "hash_A"
        w0.last_cache_used_at = old_time
        w1.cached_hash = "hash_B"
        w1.last_cache_used_at = datetime.now()

        t, f = pool.enqueue("streaming", history_hash="hash_A")
        assert f.done()
        assert f.result() == w0

        # 模拟 gateway 刷新时间并释放 Worker
        w0.last_cache_used_at = datetime.now()
        pool.release_worker(w0, duration_s=1.0)

        # 现在 w0 是最新的，如果有新请求不匹配，应该淘汰 w1（更旧）
        t2, f2 = pool.enqueue("streaming", history_hash="hash_C")
        assert f2.done()
        assert f2.result() == w1  # 淘汰 w1（更旧）

    @pytest.mark.asyncio
    async def test_2_worker_2_user_scenario(self) -> None:
        """2 Worker 2 User 场景：各命中各自缓存"""
        pool = make_pool(num_workers=2, all_idle=True)

        w0, w1 = list(pool.workers.values())

        # User A 第1轮 → cache miss → 选 w0（无缓存）
        t_a1, f_a1 = pool.enqueue("streaming", history_hash="hash_empty")
        assert f_a1.done()
        assigned_a = f_a1.result()
        # 模拟完成：设置缓存并释放 Worker
        assigned_a.cached_hash = "hash_A1"
        assigned_a.last_cache_used_at = datetime.now()
        pool.release_worker(assigned_a, duration_s=1.0)

        # User B 第1轮 → cache miss → 选 w1（无缓存）
        t_b1, f_b1 = pool.enqueue("streaming", history_hash="hash_empty_b")
        assert f_b1.done()
        assigned_b = f_b1.result()
        assert assigned_b != assigned_a  # 不同 Worker
        # 模拟完成：设置缓存并释放 Worker
        assigned_b.cached_hash = "hash_B1"
        assigned_b.last_cache_used_at = datetime.now()
        pool.release_worker(assigned_b, duration_s=1.0)

        # User A 第2轮 → cache HIT
        t_a2, f_a2 = pool.enqueue("streaming", history_hash="hash_A1")
        assert f_a2.done()
        assert f_a2.result() == assigned_a  # 命中同一个 Worker
        pool.release_worker(assigned_a, duration_s=1.0)

        # User B 第2轮 → cache HIT
        t_b2, f_b2 = pool.enqueue("streaming", history_hash="hash_B1")
        assert f_b2.done()
        assert f_b2.result() == assigned_b  # 命中同一个 Worker


# ============ 9. 并发安全 ============

class TestConcurrency:
    """验证多协程并发操作的安全性"""

    @pytest.mark.asyncio
    async def test_concurrent_enqueue(self) -> None:
        """多个协程同时入队，位置唯一且正确"""
        pool = make_pool(num_workers=1, max_queue_size=100, all_idle=True)
        make_workers_busy(pool, count=1)

        async def enqueue_one(idx: int) -> int:
            t, _ = pool.enqueue("chat")
            return t.position

        tasks = [enqueue_one(i) for i in range(20)]
        positions = await asyncio.gather(*tasks)

        # 所有位置应该唯一
        assert len(set(positions)) == 20
        assert pool.queue_length == 20

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_and_release(self) -> None:
        """同时入队和释放，不会丢失请求"""
        pool = make_pool(num_workers=3, max_queue_size=100, all_idle=True)
        make_workers_busy(pool, count=3)

        completed: list = []

        async def enqueue_and_wait(idx: int) -> None:
            t, f = pool.enqueue("chat")
            worker = await f
            if worker:
                completed.append(idx)
                # 模拟处理
                await asyncio.sleep(0.01)
                worker.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat")
                pool.release_worker(worker, "chat", 0.01)

        # 先释放所有 Worker 让调度开始
        async def release_all() -> None:
            await asyncio.sleep(0.05)
            for w in pool.workers.values():
                if w.is_busy:
                    pool.release_worker(w, "chat", 1.0)

        # 10 个请求 + 释放
        tasks = [enqueue_and_wait(i) for i in range(10)]
        tasks.append(release_all())

        await asyncio.gather(*tasks)

        # 所有请求应该都完成
        assert len(completed) == 10


# ============ 10. 队列状态查询 ============

class TestQueueStatus:
    """验证队列状态快照"""

    @pytest.mark.asyncio
    async def test_get_queue_status(self) -> None:
        """获取队列状态"""
        pool = make_pool(num_workers=1, all_idle=True)
        make_workers_busy(pool, count=1, request_type="omni_duplex")

        pool.enqueue("chat")
        pool.enqueue("half_duplex_audio")

        status = pool.get_queue_status()
        assert status.queue_length == 2
        assert status.max_queue_size == 1000
        assert len(status.items) == 2
        assert status.items[0].request_type == "chat"
        assert status.items[1].request_type == "half_duplex_audio"

    @pytest.mark.asyncio
    async def test_running_tasks_info(self) -> None:
        """运行中任务信息"""
        pool = make_pool(num_workers=2, all_idle=True)
        make_workers_busy(pool, count=1, request_type="omni_duplex")

        tasks = pool._get_running_tasks()
        assert len(tasks) == 1
        assert tasks[0].request_type == "omni_duplex"
        assert tasks[0].elapsed_s >= 0

    @pytest.mark.asyncio
    async def test_get_ticket(self) -> None:
        """获取指定 ticket"""
        pool = make_pool(num_workers=1, all_idle=True)
        make_workers_busy(pool, count=1)

        t1, _ = pool.enqueue("chat")
        found = pool.get_ticket(t1.ticket_id)
        assert found is not None
        assert found.ticket_id == t1.ticket_id

        # 不存在的
        assert pool.get_ticket("nonexistent") is None


# ============ 11. Chat/Duplex 路由（非 Streaming） ============

class TestNonStreamingRouting:
    """验证 Chat/Duplex 的路由策略"""

    @pytest.mark.asyncio
    async def test_prefer_no_cache_worker(self) -> None:
        """Chat/Duplex 优先选无缓存 Worker"""
        pool = make_pool(num_workers=2, all_idle=True)

        w0, w1 = list(pool.workers.values())
        w0.cached_hash = "hash_A"
        w0.last_cache_used_at = datetime.now()
        w1.cached_hash = None

        t, f = pool.enqueue("chat")
        assert f.done()
        assert f.result() == w1  # 无缓存优先

    @pytest.mark.asyncio
    async def test_lru_when_all_cached(self) -> None:
        """所有 Worker 都有缓存时，Chat 也用 LRU"""
        pool = make_pool(num_workers=2, all_idle=True)

        w0, w1 = list(pool.workers.values())
        w0.cached_hash = "hash_A"
        w0.last_cache_used_at = datetime.now() - timedelta(hours=1)
        w1.cached_hash = "hash_B"
        w1.last_cache_used_at = datetime.now()

        t, f = pool.enqueue("omni_duplex")
        assert f.done()
        assert f.result() == w0  # 淘汰最旧缓存


# ============ 12. ETA Config ============

class TestEtaConfig:
    """验证 ETA 配置更新"""

    @pytest.mark.asyncio
    async def test_update_eta_config(self) -> None:
        """更新 ETA 基准值"""
        pool = make_pool(num_workers=1, all_idle=True)

        assert pool.eta_tracker.get_eta("chat") == 10.0

        pool.eta_tracker.update_config(EtaConfig(
            eta_chat_s=20.0,
            eta_half_duplex_s=30.0,
            eta_omni_duplex_s=120.0,
        ))

        assert pool.eta_tracker.get_eta("chat") == 20.0
        assert pool.eta_tracker.get_eta("half_duplex_audio") == 30.0
        assert pool.eta_tracker.get_eta("omni_duplex") == 120.0

    @pytest.mark.asyncio
    async def test_eta_status(self) -> None:
        """ETA 状态包含配置和 EMA 数据"""
        pool = make_pool(num_workers=1, all_idle=True)

        pool.eta_tracker.record_duration("chat", 5.0)

        status = pool.eta_tracker.get_status()
        assert status.config.eta_chat_s == 10.0
        assert status.ema_chat_s == 5.0
        assert status.ema_chat_samples == 1
        assert status.ema_half_duplex_samples == 0


# ============ 13. ETA 时间精度验证 ============

class TestEtaAccuracy:
    """验证 ETA 在各种时间点上的准确性（回归测试）"""

    @pytest.mark.asyncio
    async def test_eta_decreases_as_task_progresses(self) -> None:
        """场景：1 Worker, eta_chat=15s
        t=0s Worker 接手任务 A
        t=5s 新请求 B 入队 → 预期 ETA ≈ 10s（非 15s）
        """
        pool = make_pool(num_workers=1, all_idle=True)
        pool.eta_tracker.update_config(
            EtaConfig(eta_chat_s=15.0, eta_half_duplex_s=15.0, eta_omni_duplex_s=90.0)
        )

        # t=0s: Worker 接手任务 A
        ticket_a, future_a = pool.enqueue("chat")
        worker_a = future_a.result()  # 立即分配
        assert worker_a is not None

        # 模拟 5s 后：把 task_started_at 回拨 5s
        worker_a.task_started_at = datetime.now() - timedelta(seconds=5)

        # t=5s: 新请求 B 入队
        ticket_b, future_b = pool.enqueue("chat")
        assert not future_b.done()  # B 在排队

        # 检查 B 的 ETA：应该 ≈ 10s（15 - 5），而不是被 floor 到 15s
        assert ticket_b.estimated_wait_s is not None
        assert 8.0 <= ticket_b.estimated_wait_s <= 12.0, \
            f"ETA should be ~10s, got {ticket_b.estimated_wait_s}"

    @pytest.mark.asyncio
    async def test_eta_floor_on_overrun(self) -> None:
        """任务超时（elapsed > eta）时兜底 15s，不显示 0 或负数"""
        pool = make_pool(num_workers=1, all_idle=True)
        pool.eta_tracker.update_config(
            EtaConfig(eta_chat_s=10.0, eta_half_duplex_s=15.0, eta_omni_duplex_s=90.0)
        )

        # Worker 接手任务
        ticket_a, future_a = pool.enqueue("chat")
        worker_a = future_a.result()
        assert worker_a is not None

        # 模拟任务已跑了 20s（eta=10s，已超时）
        worker_a.task_started_at = datetime.now() - timedelta(seconds=20)

        # 新请求入队
        ticket_b, future_b = pool.enqueue("chat")
        assert not future_b.done()

        # ETA 应该兜底到 15s
        assert ticket_b.estimated_wait_s is not None
        assert 14.0 <= ticket_b.estimated_wait_s <= 16.0, \
            f"ETA floor should be ~15s on overrun, got {ticket_b.estimated_wait_s}"

    @pytest.mark.asyncio
    async def test_eta_multi_worker_staggered(self) -> None:
        """2 Worker 交错繁忙，3 请求排队，ETA 递增"""
        pool = make_pool(num_workers=2, all_idle=True)
        pool.eta_tracker.update_config(
            EtaConfig(eta_chat_s=20.0, eta_half_duplex_s=15.0, eta_omni_duplex_s=90.0)
        )

        # Worker 0 接手 A（已跑 5s，剩余 15s）
        ticket_a, future_a = pool.enqueue("chat")
        worker_a = future_a.result()
        worker_a.task_started_at = datetime.now() - timedelta(seconds=5)

        # Worker 1 接手 B（已跑 10s，剩余 10s）
        ticket_b, future_b = pool.enqueue("chat")
        worker_b = future_b.result()
        worker_b.task_started_at = datetime.now() - timedelta(seconds=10)

        # 排队请求 C, D, E
        ticket_c, _ = pool.enqueue("chat")
        ticket_d, _ = pool.enqueue("chat")
        ticket_e, _ = pool.enqueue("chat")

        # C 应该等最快释放的 Worker（剩余 10s）
        assert ticket_c.estimated_wait_s is not None
        assert 8.0 <= ticket_c.estimated_wait_s <= 12.0, \
            f"C should wait ~10s (fastest worker), got {ticket_c.estimated_wait_s}"

        # D 应该等第二快释放的 Worker（剩余 15s）
        assert ticket_d.estimated_wait_s is not None
        assert 13.0 <= ticket_d.estimated_wait_s <= 17.0, \
            f"D should wait ~15s (second worker), got {ticket_d.estimated_wait_s}"

        # E 需要第二轮：堆模拟精确计算
        # C→pop(10,w1),push(30). D→pop(15,w0),push(35). E→pop(30,w1). ETA=30s
        assert ticket_e.estimated_wait_s is not None
        assert 28.0 <= ticket_e.estimated_wait_s <= 32.0, \
            f"E should wait ~30s (heap simulation), got {ticket_e.estimated_wait_s}"

    @pytest.mark.asyncio
    async def test_running_tasks_remaining_decreases(self) -> None:
        """_get_running_tasks 返回的 estimated_remaining_s 随时间递减"""
        pool = make_pool(num_workers=1, all_idle=True)
        pool.eta_tracker.update_config(
            EtaConfig(eta_chat_s=15.0, eta_half_duplex_s=15.0, eta_omni_duplex_s=90.0)
        )

        ticket_a, future_a = pool.enqueue("chat")
        worker_a = future_a.result()
        assert worker_a is not None

        # 模拟已跑 5s
        worker_a.task_started_at = datetime.now() - timedelta(seconds=5)

        tasks = pool._get_running_tasks()
        assert len(tasks) == 1
        # remaining ≈ 10s（15 - 5）
        assert 8.0 <= tasks[0].estimated_remaining_s <= 12.0, \
            f"Remaining should be ~10s, got {tasks[0].estimated_remaining_s}"
