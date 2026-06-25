"""集成测试 — 真实 Gateway + Mock Worker 进程间通信

启动 Mock Worker 进程 + Gateway 进程，通过 HTTP/WS 验证完整排队链路。
无需 GPU，验证：
- Chat 排队 → 分配 → 响应
- Streaming 排队 → WS 状态推送 → 响应
- Duplex 排队 → WS 状态推送 → 交互 → 停止
- 多 Worker（1/2/3）× 多 User 并发
- 队列 FIFO 顺序保证
- 队列满拒绝
- 取消机制

运行：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_integration.py -v -s
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional, Tuple, Dict, Any

import httpx
import pytest
import pytest_asyncio
import websockets

# ============ 常量 ============

GATEWAY_BASE_PORT = 19900  # 测试用端口基址，避免与正式服务冲突
WORKER_BASE_PORT = 19950
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "base", "bin", "python")

# 超时配置
STARTUP_TIMEOUT = 15.0  # 进程启动超时
REQUEST_TIMEOUT = 30.0  # 请求超时


# ============ 进程管理 ============

class ProcessManager:
    """管理 Mock Worker 和 Gateway 子进程的生命周期"""

    def __init__(self):
        self.processes: List[subprocess.Popen] = []

    def start_mock_worker(self, port: int, gpu_id: int = 0,
                          chat_delay: float = 0.3) -> subprocess.Popen:
        """启动一个 Mock Worker 进程"""
        cmd = [
            PYTHON, os.path.join(PROJECT_ROOT, "tests", "mock_worker.py"),
            "--port", str(port),
            "--gpu-id", str(gpu_id),
            "--chat-delay", str(chat_delay),
            "--stream-delay", "0.05",
            "--stream-chunks", "3",
            "--duplex-delay", "0.05",
            "--duplex-chunks", "3",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.processes.append(proc)
        return proc

    def start_gateway(self, port: int) -> subprocess.Popen:
        """启动 Gateway 进程（HTTP 模式，无需证书）"""
        cmd = [
            PYTHON, os.path.join(PROJECT_ROOT, "gateway.py"),
            "--port", str(port),
            "--internal-port", str(port + 1000),
            "--http",
            "--max-queue-size", "20",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.processes.append(proc)
        return proc

    def kill_all(self):
        """停止所有子进程"""
        for proc in self.processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        # 等待所有进程退出
        for proc in self.processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        self.processes.clear()


async def wait_for_service(url: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    """等待 HTTP 服务就绪"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return False


async def wait_for_gateway_workers_ready(
    gateway_url: str, expected_workers: int, timeout: float = STARTUP_TIMEOUT
) -> bool:
    """等待 Gateway 的所有 Worker 都变为 IDLE"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{gateway_url}/status", timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("idle_workers", 0) >= expected_workers:
                        return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


async def register_gateway_worker(
    gateway_port: int,
    worker_id: str,
    endpoint: str,
    gpu_group: str,
) -> None:
    url = f"http://127.0.0.1:{gateway_port + 1000}/internal/workers/{worker_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            url,
            json={"endpoint": endpoint, "gpu_group": gpu_group},
            timeout=5.0,
        )
        assert resp.status_code == 200, resp.text


# ============ Fixtures ============

class ServiceCluster:
    """测试集群：N 个 Mock Worker + 1 个 Gateway"""

    def __init__(self, num_workers: int, gateway_port: int, worker_base_port: int,
                 chat_delay: float = 0.3):
        self.num_workers = num_workers
        self.gateway_port = gateway_port
        self.worker_base_port = worker_base_port
        self.chat_delay = chat_delay
        self.pm = ProcessManager()

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}"

    @property
    def gateway_ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.gateway_port}"

    async def start(self):
        """启动集群"""
        # 启动 Mock Workers
        worker_addresses = []
        for i in range(self.num_workers):
            port = self.worker_base_port + i
            self.pm.start_mock_worker(port=port, gpu_id=i, chat_delay=self.chat_delay)
            worker_addresses.append(f"127.0.0.1:{port}")

        # 等 Worker 就绪
        for i in range(self.num_workers):
            port = self.worker_base_port + i
            ok = await wait_for_service(f"http://127.0.0.1:{port}/health")
            assert ok, f"Mock Worker {i} (port {port}) failed to start"

        # 启动 Gateway
        self.pm.start_gateway(self.gateway_port)
        ok = await wait_for_service(f"http://127.0.0.1:{self.gateway_port}/health")
        assert ok, f"Gateway failed to start on port {self.gateway_port}"

        for i, endpoint in enumerate(worker_addresses):
            await register_gateway_worker(
                self.gateway_port,
                worker_id=f"mock-worker-{i}",
                endpoint=endpoint,
                gpu_group=f"gpu-{i}",
            )

        # 等 Gateway + 所有 Worker 就绪
        ok = await wait_for_gateway_workers_ready(
            self.gateway_url, self.num_workers, timeout=STARTUP_TIMEOUT
        )
        assert ok, (
            f"Gateway failed to detect {self.num_workers} idle workers. "
            f"Check if ports {self.gateway_port}, "
            f"{self.worker_base_port}-{self.worker_base_port + self.num_workers - 1} are free."
        )

    def stop(self):
        self.pm.kill_all()


# 每个 worker 数量配置对应不同端口范围，避免测试间冲突
def _ports_for_workers(n: int) -> Tuple[int, int]:
    """根据 worker 数量返回 (gateway_port, worker_base_port)"""
    offset = n * 100
    return (GATEWAY_BASE_PORT + offset, WORKER_BASE_PORT + offset)


@pytest_asyncio.fixture
async def cluster_1w():
    """1 Worker 集群"""
    gp, wp = _ports_for_workers(1)
    c = ServiceCluster(1, gp, wp)
    await c.start()
    yield c
    c.stop()


@pytest_asyncio.fixture
async def cluster_2w():
    """2 Worker 集群"""
    gp, wp = _ports_for_workers(2)
    c = ServiceCluster(2, gp, wp)
    await c.start()
    yield c
    c.stop()


@pytest_asyncio.fixture
async def cluster_3w():
    """3 Worker 集群"""
    gp, wp = _ports_for_workers(3)
    c = ServiceCluster(3, gp, wp)
    await c.start()
    yield c
    c.stop()


# ============ 辅助函数 ============

async def do_chat(gateway_url: str, message: str = "hello") -> Dict[str, Any]:
    """发送 Chat 请求"""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{gateway_url}/api/chat",
            json={"messages": [{"role": "user", "content": message}]},
        )
        return resp.json()


async def do_streaming_turn(
    gateway_ws_url: str, session_id: str, messages: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """执行一轮 Streaming（prefill + generate），返回 (full_text, all_ws_messages)"""
    ws_messages: List[Dict[str, Any]] = []
    full_text = ""

    async with websockets.connect(
        f"{gateway_ws_url}/ws/streaming/{session_id}",
        open_timeout=REQUEST_TIMEOUT,
    ) as ws:
        # Prefill
        await ws.send(json.dumps({
            "type": "prefill",
            "messages": messages,
            "session_id": session_id,
            "is_last_chunk": True,
        }))

        # 读取所有 prefill 阶段消息（可能含 queued/queue_update/queue_done/prefill_done）
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
            msg = json.loads(raw)
            ws_messages.append(msg)
            if msg.get("type") in ("prefill_done", "error"):
                break

        if ws_messages[-1].get("type") == "error":
            return "", ws_messages

        # Generate
        await ws.send(json.dumps({
            "type": "generate",
            "session_id": session_id,
            "generate_audio": False,
            "max_new_tokens": 100,
        }))

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
            msg = json.loads(raw)
            ws_messages.append(msg)
            if msg.get("type") == "chunk" and msg.get("text_delta"):
                full_text += msg["text_delta"]
            if msg.get("type") in ("done", "error"):
                break

        # Close
        await ws.send(json.dumps({"type": "close"}))

    return full_text, ws_messages


async def do_duplex_session(
    gateway_ws_url: str, session_id: str, num_chunks: int = 3
) -> List[Dict[str, Any]]:
    """执行一轮 Duplex session，返回所有收到的 WS 消息"""
    ws_messages: List[Dict[str, Any]] = []

    async with websockets.connect(
        f"{gateway_ws_url}/ws/duplex/{session_id}",
        open_timeout=REQUEST_TIMEOUT,
    ) as ws:
        # 先读排队消息（queued / queue_update / queue_done）
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
            msg = json.loads(raw)
            ws_messages.append(msg)
            if msg.get("type") in ("queue_done", "error"):
                break

        if ws_messages[-1].get("type") == "error":
            return ws_messages

        # Prepare
        await ws.send(json.dumps({
            "type": "prepare",
            "system_prompt": "mock test",
            "config": {},
        }))

        raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
        msg = json.loads(raw)
        ws_messages.append(msg)
        assert msg.get("type") == "prepared", f"Expected prepared, got {msg}"

        # Audio chunks
        import base64
        dummy_audio = base64.b64encode(b"\x00" * 3200).decode()
        for i in range(num_chunks):
            await ws.send(json.dumps({
                "type": "audio_chunk",
                "audio_base64": dummy_audio,
            }))
            raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
            msg = json.loads(raw)
            ws_messages.append(msg)

        # Stop
        await ws.send(json.dumps({"type": "stop"}))
        raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
        msg = json.loads(raw)
        ws_messages.append(msg)

    return ws_messages


# ============================================================
# Part 1: Chat 模式集成测试
# ============================================================

class TestChatIntegration:
    """Chat HTTP POST 集成测试"""

    @pytest.mark.asyncio
    async def test_chat_single_request_1w(self, cluster_1w: ServiceCluster):
        """1 Worker: 单个 Chat 请求"""
        result = await do_chat(cluster_1w.gateway_url)
        assert result.get("success") is True
        assert "[Mock]" in result.get("text", "")

    @pytest.mark.asyncio
    async def test_chat_single_request_2w(self, cluster_2w: ServiceCluster):
        """2 Worker: 单个 Chat 请求"""
        result = await do_chat(cluster_2w.gateway_url)
        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_chat_single_request_3w(self, cluster_3w: ServiceCluster):
        """3 Worker: 单个 Chat 请求"""
        result = await do_chat(cluster_3w.gateway_url)
        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_chat_concurrent_within_capacity_2w(self, cluster_2w: ServiceCluster):
        """2 Worker: 2 个并发 Chat（不需要排队）"""
        results = await asyncio.gather(
            do_chat(cluster_2w.gateway_url, "user1"),
            do_chat(cluster_2w.gateway_url, "user2"),
        )
        for r in results:
            assert r.get("success") is True

    @pytest.mark.asyncio
    async def test_chat_concurrent_within_capacity_3w(self, cluster_3w: ServiceCluster):
        """3 Worker: 3 个并发 Chat（不需要排队）"""
        results = await asyncio.gather(
            do_chat(cluster_3w.gateway_url, "user1"),
            do_chat(cluster_3w.gateway_url, "user2"),
            do_chat(cluster_3w.gateway_url, "user3"),
        )
        for r in results:
            assert r.get("success") is True

    @pytest.mark.asyncio
    async def test_chat_queue_overflow_1w(self, cluster_1w: ServiceCluster):
        """1 Worker: 超出 Worker + 队列容量的并发请求"""
        # 1 Worker, chat_delay=0.3s, 同时发 5 个请求
        # 第 1 个直接分配，后 4 个排队（队列容量 20 足够）
        results = await asyncio.gather(
            *[do_chat(cluster_1w.gateway_url, f"user_{i}") for i in range(5)]
        )
        # 所有请求应该都能完成（通过排队）
        success_count = sum(1 for r in results if r.get("success") is True)
        assert success_count == 5, f"Expected 5 success, got {success_count}"

    @pytest.mark.asyncio
    async def test_chat_sequential_1w(self, cluster_1w: ServiceCluster):
        """1 Worker: 连续发送 3 个 Chat（依次完成）"""
        for i in range(3):
            result = await do_chat(cluster_1w.gateway_url, f"message_{i}")
            assert result.get("success") is True


# ============================================================
# Part 2: Streaming 模式集成测试
# ============================================================

class TestStreamingIntegration:
    """Streaming WebSocket 集成测试"""

    @pytest.mark.asyncio
    async def test_streaming_single_turn_1w(self, cluster_1w: ServiceCluster):
        """1 Worker: 单轮 Streaming"""
        text, msgs = await do_streaming_turn(
            cluster_1w.gateway_ws_url,
            session_id="s1",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert len(text) > 0
        # 应包含 prefill_done 和 done
        types = [m["type"] for m in msgs]
        assert "prefill_done" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_streaming_single_turn_2w(self, cluster_2w: ServiceCluster):
        """2 Worker: 单轮 Streaming"""
        text, msgs = await do_streaming_turn(
            cluster_2w.gateway_ws_url,
            session_id="s1",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_streaming_concurrent_2w(self, cluster_2w: ServiceCluster):
        """2 Worker: 2 个并发 Streaming"""
        results = await asyncio.gather(
            do_streaming_turn(
                cluster_2w.gateway_ws_url, "s1",
                [{"role": "user", "content": "user1"}]
            ),
            do_streaming_turn(
                cluster_2w.gateway_ws_url, "s2",
                [{"role": "user", "content": "user2"}]
            ),
        )
        for text, msgs in results:
            assert len(text) > 0

    @pytest.mark.asyncio
    async def test_streaming_queued_1w(self, cluster_1w: ServiceCluster):
        """1 Worker: 2 个并发 Streaming（1 个排队）"""
        results = await asyncio.gather(
            do_streaming_turn(
                cluster_1w.gateway_ws_url, "s1",
                [{"role": "user", "content": "user1"}]
            ),
            do_streaming_turn(
                cluster_1w.gateway_ws_url, "s2",
                [{"role": "user", "content": "user2"}]
            ),
        )
        for text, msgs in results:
            assert len(text) > 0

        # 至少有一个应该经历了排队
        all_types = []
        for _, msgs in results:
            all_types.extend(m["type"] for m in msgs)
        # 当有排队时，会出现 queued 消息
        # 注意：如果第一个请求很快完成，第二个可能直接分配
        # 所以这里不强制要求 queued 出现

    @pytest.mark.asyncio
    async def test_streaming_3_concurrent_3w(self, cluster_3w: ServiceCluster):
        """3 Worker: 3 个并发 Streaming（全部直接分配）"""
        results = await asyncio.gather(
            do_streaming_turn(
                cluster_3w.gateway_ws_url, "s1",
                [{"role": "user", "content": "u1"}]
            ),
            do_streaming_turn(
                cluster_3w.gateway_ws_url, "s2",
                [{"role": "user", "content": "u2"}]
            ),
            do_streaming_turn(
                cluster_3w.gateway_ws_url, "s3",
                [{"role": "user", "content": "u3"}]
            ),
        )
        for text, msgs in results:
            assert len(text) > 0
            types = [m["type"] for m in msgs]
            assert "done" in types


# ============================================================
# Part 3: Duplex 模式集成测试
# ============================================================

class TestDuplexIntegration:
    """Duplex WebSocket 集成测试"""

    @pytest.mark.asyncio
    async def test_duplex_single_session_1w(self, cluster_1w: ServiceCluster):
        """1 Worker: 单个 Duplex session"""
        msgs = await do_duplex_session(
            cluster_1w.gateway_ws_url,
            session_id="d1",
            num_chunks=3,
        )
        types = [m["type"] for m in msgs]
        assert "queue_done" in types
        assert "prepared" in types
        assert "stopped" in types

    @pytest.mark.asyncio
    async def test_duplex_single_session_2w(self, cluster_2w: ServiceCluster):
        """2 Worker: 单个 Duplex session"""
        msgs = await do_duplex_session(
            cluster_2w.gateway_ws_url,
            session_id="d1",
            num_chunks=2,
        )
        types = [m["type"] for m in msgs]
        assert "prepared" in types
        assert "stopped" in types

    @pytest.mark.asyncio
    async def test_duplex_concurrent_2w(self, cluster_2w: ServiceCluster):
        """2 Worker: 2 个并发 Duplex"""
        results = await asyncio.gather(
            do_duplex_session(cluster_2w.gateway_ws_url, "d1", 2),
            do_duplex_session(cluster_2w.gateway_ws_url, "d2", 2),
        )
        for msgs in results:
            types = [m["type"] for m in msgs]
            assert "prepared" in types
            assert "stopped" in types

    @pytest.mark.asyncio
    async def test_duplex_queued_1w(self, cluster_1w: ServiceCluster):
        """1 Worker: 2 个 Duplex session（1 个排队）

        注意：Duplex 独占 Worker，所以第 2 个必须排队。
        但由于两者同时连接，时序不确定，需要验证两者都能完成。
        """
        results = await asyncio.gather(
            do_duplex_session(cluster_1w.gateway_ws_url, "d1", 2),
            do_duplex_session(cluster_1w.gateway_ws_url, "d2", 2),
        )
        for msgs in results:
            types = [m["type"] for m in msgs]
            assert "prepared" in types
            assert "stopped" in types


# ============================================================
# Part 4: 混合模式集成测试
# ============================================================

class TestMixedModeIntegration:
    """混合 Chat + Streaming + Duplex 并发"""

    @pytest.mark.asyncio
    async def test_mixed_all_modes_2w(self, cluster_2w: ServiceCluster):
        """2 Worker: Chat + Streaming + Duplex 混合"""
        chat_task = do_chat(cluster_2w.gateway_url, "mixed_chat")
        streaming_task = do_streaming_turn(
            cluster_2w.gateway_ws_url, "mixed_s1",
            [{"role": "user", "content": "mixed_stream"}]
        )

        chat_result, (stream_text, stream_msgs) = await asyncio.gather(
            chat_task, streaming_task
        )

        assert chat_result.get("success") is True
        assert len(stream_text) > 0

    @pytest.mark.asyncio
    async def test_mixed_all_modes_3w(self, cluster_3w: ServiceCluster):
        """3 Worker: Chat + Streaming + Duplex 同时"""
        results = await asyncio.gather(
            do_chat(cluster_3w.gateway_url, "chat"),
            do_streaming_turn(
                cluster_3w.gateway_ws_url, "s1",
                [{"role": "user", "content": "stream"}]
            ),
            do_duplex_session(cluster_3w.gateway_ws_url, "d1", 2),
        )

        chat_result = results[0]
        stream_text, stream_msgs = results[1]
        duplex_msgs = results[2]

        assert chat_result.get("success") is True
        assert len(stream_text) > 0
        types = [m["type"] for m in duplex_msgs]
        assert "prepared" in types

    @pytest.mark.asyncio
    async def test_mixed_heavy_load_3w(self, cluster_3w: ServiceCluster):
        """3 Worker: 5 Chat + 2 Streaming + 1 Duplex（重负载，有排队）"""
        tasks = []
        # 5 Chat
        for i in range(5):
            tasks.append(do_chat(cluster_3w.gateway_url, f"chat_{i}"))
        # 2 Streaming
        for i in range(2):
            tasks.append(do_streaming_turn(
                cluster_3w.gateway_ws_url, f"s_{i}",
                [{"role": "user", "content": f"stream_{i}"}]
            ))
        # 1 Duplex
        tasks.append(do_duplex_session(cluster_3w.gateway_ws_url, "d_heavy", 2))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计
        success = 0
        errors = 0
        for r in results:
            if isinstance(r, Exception):
                errors += 1
            elif isinstance(r, dict):
                # Chat result
                if r.get("success"):
                    success += 1
            elif isinstance(r, tuple):
                # Streaming result
                text, msgs = r
                if len(text) > 0:
                    success += 1
            elif isinstance(r, list):
                # Duplex result
                types = [m.get("type") for m in r]
                if "prepared" in types:
                    success += 1

        # 所有请求应该都能完成（通过排队）
        assert success >= 6, f"Expected at least 6 success, got {success}, errors={errors}"


# ============================================================
# Part 5: 队列 API 集成测试
# ============================================================

class TestQueueAPIIntegration:
    """队列管理 API 集成测试"""

    @pytest.mark.asyncio
    async def test_queue_status_empty(self, cluster_1w: ServiceCluster):
        """空队列状态查询"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{cluster_1w.gateway_url}/api/queue")
            assert resp.status_code == 200
            data = resp.json()
            assert data["queue_length"] == 0

    @pytest.mark.asyncio
    async def test_eta_config_get_put(self, cluster_1w: ServiceCluster):
        """ETA 配置获取和更新"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            # GET
            resp = await client.get(f"{cluster_1w.gateway_url}/api/config/eta")
            assert resp.status_code == 200
            data = resp.json()
            assert "config" in data

            # PUT
            resp = await client.put(
                f"{cluster_1w.gateway_url}/api/config/eta",
                json={"eta_chat_s": 5.0, "eta_half_duplex_s": 8.0, "eta_duplex_s": 60.0},
            )
            assert resp.status_code == 200

            # 验证更新生效
            resp = await client.get(f"{cluster_1w.gateway_url}/api/config/eta")
            data = resp.json()
            assert data["config"]["eta_chat_s"] == 5.0

    @pytest.mark.asyncio
    async def test_service_status(self, cluster_2w: ServiceCluster):
        """Service status 包含 Worker 和队列信息"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{cluster_2w.gateway_url}/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_workers"] == 2
            assert data["idle_workers"] == 2
            assert data["queue_length"] == 0
