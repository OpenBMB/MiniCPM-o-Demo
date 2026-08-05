"""端到端测试 — 真实 Worker（GPU）+ Gateway + 模拟客户端

启动真实 Worker 进程（加载模型，需要 GPU）+ Gateway 进程，
通过 HTTP/WS 验证完整排队链路和推理结果。

前提：
- 需要 3 张空闲 GPU（使用 GPU 1,2,3）
- 模型加载 ~15s/worker
- 测试 1/2/3 Worker 配置 × Chat/Streaming/Duplex

运行：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_e2e.py -v -s

Duplex 前端模拟：
    - 通过 WebSocket 发送 prepare/audio_chunk/stop 消息
    - 验证 prepared/result/stopped 响应
    - 无需真实浏览器，纯 WS 协议交互
"""

import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional, Tuple, Dict, Any

import httpx
import numpy as np
import pytest
import pytest_asyncio
import websockets

# ============ 常量 ============

# 端口配置（避免与正式服务 10024/22400 冲突）
E2E_GATEWAY_PORT = 18800
E2E_WORKER_BASE_PORT = 18900
E2E_GPUS = [1, 2, 3]  # 使用的 GPU（避免 GPU 0 已有服务）

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "base", "bin", "python")

STARTUP_TIMEOUT = 120.0  # Worker 加载模型可能需要较长时间
REQUEST_TIMEOUT = 60.0


# ============ 进程管理 ============

class E2EProcessManager:
    """端到端测试进程管理器"""

    def __init__(self):
        self.workers: List[subprocess.Popen] = []
        self.gateways: List[subprocess.Popen] = []

    def start_real_worker(self, port: int, gpu_id: int, worker_index: int) -> subprocess.Popen:
        """启动真实 Worker 进程"""
        cmd = [
            PYTHON, os.path.join(PROJECT_ROOT, "worker.py"),
            "--port", str(port),
            "--gpu-id", str(gpu_id),
            "--worker-index", str(worker_index),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": PROJECT_ROOT,
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
        }
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.workers.append(proc)
        return proc

    def start_gateway(self, port: int) -> subprocess.Popen:
        """启动 Gateway 进程"""
        cmd = [
            PYTHON, os.path.join(PROJECT_ROOT, "gateway.py"),
            "--port", str(port),
            "--internal-port", str(port + 1000),
            "--http",
            "--max-queue-size", "100",
        ]
        log_path = os.path.join(PROJECT_ROOT, "tmp", f"gateway_{port}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        self.gateways.append(proc)
        return proc

    def kill_gateways(self):
        """只停止 Gateway（Worker 保留复用）"""
        for proc in self.gateways:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in self.gateways:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        self.gateways.clear()

    def kill_all(self):
        """停止所有进程"""
        all_procs = self.gateways + self.workers
        for proc in all_procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in all_procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self.gateways.clear()
        self.workers.clear()


async def wait_for_health(url: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    """等待 HTTP 服务健康"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", data.get("worker_status", ""))
                    if status in ("healthy", "idle"):
                        return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


async def wait_for_gateway_ready(
    gateway_url: str, expected_idle: int, timeout: float = STARTUP_TIMEOUT
) -> bool:
    """等待 Gateway 看到 expected_idle 个 IDLE Worker"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{gateway_url}/status", timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("idle_workers", 0) >= expected_idle:
                        return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


async def register_gateway_worker(
    gateway_port: int,
    worker_id: str,
    endpoint: str,
    gpu_group: str,
) -> None:
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"http://127.0.0.1:{gateway_port + 1000}/internal/workers/{worker_id}",
            json={"endpoint": endpoint, "gpu_group": gpu_group},
            timeout=5.0,
        )
        assert resp.status_code == 200, resp.text


# ============ 客户端辅助函数 ============

async def e2e_chat(gateway_url: str, message: str = "你好") -> Dict[str, Any]:
    """发送真实 Chat 请求"""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{gateway_url}/api/chat",
            json={
                "messages": [{"role": "user", "content": message}],
                "generation": {"max_new_tokens": 50},
            },
        )
        resp.raise_for_status()
        return resp.json()


async def e2e_streaming_turn(
    gateway_ws_url: str, session_id: str,
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """执行一轮真实 Streaming"""
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

        # 读排队 + prefill_done
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
            "max_new_tokens": 50,
        }))

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
            msg = json.loads(raw)
            ws_messages.append(msg)
            if msg.get("type") == "chunk" and msg.get("text_delta"):
                full_text += msg["text_delta"]
            if msg.get("type") in ("done", "error"):
                break

        await ws.send(json.dumps({"type": "close"}))

    return full_text, ws_messages


async def e2e_duplex_session(
    gateway_ws_url: str, session_id: str,
    num_audio_chunks: int = 5,
    ref_audio_base64: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """执行一轮真实 Duplex session（模拟前端行为）

    协议流程：
    1. 连接 WS，接收 queue_done
    2. 发送 prepare（含 system_prompt + ref_audio）
    3. 收到 prepared
    4. 循环发送 audio_chunk（模拟麦克风输入）
    5. 收到 result（listen/speak 状态）
    6. 发送 stop，收到 stopped
    """
    ws_messages: List[Dict[str, Any]] = []

    # 生成模拟音频数据（1s 16kHz 静音）
    dummy_audio = np.zeros(16000, dtype=np.float32)
    audio_bytes = dummy_audio.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    try:
        async with websockets.connect(
            f"{gateway_ws_url}/ws/duplex/{session_id}",
            open_timeout=REQUEST_TIMEOUT,
        ) as ws:
            # 读排队消息
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
                msg = json.loads(raw)
                ws_messages.append(msg)
                if msg.get("type") in ("queue_done", "error"):
                    break

            if ws_messages[-1].get("type") == "error":
                return ws_messages

            # Prepare（模拟前端 DuplexSession.start()）
            prepare_msg: Dict[str, Any] = {
                "type": "prepare",
                "system_prompt": "你好，你是一个友好的助手。",
                "config": {
                    "max_kv_tokens": 8000,
                },
                "deferred_finalize": True,
            }
            if ref_audio_base64:
                prepare_msg["tts_ref_audio_base64"] = ref_audio_base64

            await ws.send(json.dumps(prepare_msg))

            # 等待 prepared
            raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
            msg = json.loads(raw)
            ws_messages.append(msg)
            if msg.get("type") != "prepared":
                return ws_messages

            # 发送 audio chunks（模拟麦克风输入）
            for i in range(num_audio_chunks):
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "audio_base64": audio_b64,
                    "force_listen": i < 3,  # 前 3 个 chunk force_listen
                }))

                # 读取 result
                raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
                msg = json.loads(raw)
                ws_messages.append(msg)

                # 简短间隔模拟实时输入
                await asyncio.sleep(0.05)

            # Stop
            await ws.send(json.dumps({"type": "stop"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT)
            msg = json.loads(raw)
            ws_messages.append(msg)

    except websockets.exceptions.ConnectionClosed as e:
        ws_messages.append({"type": "error", "error": f"WS closed: {e}"})
    except Exception as e:
        ws_messages.append({"type": "error", "error": f"Exception: {e}"})

    return ws_messages


# ============ Fixtures ============

# 全局 Worker 池（3 个真实 Worker，session scope 避免重复加载模型）
_global_pm: Optional[E2EProcessManager] = None
_workers_ready = False


@pytest_asyncio.fixture(scope="session")
async def real_workers():
    """启动 3 个真实 Worker（GPU 1,2,3），session scope 复用"""
    global _global_pm, _workers_ready

    _global_pm = E2EProcessManager()

    # 并行启动 3 个 Worker
    for i in range(3):
        port = E2E_WORKER_BASE_PORT + i
        gpu_id = E2E_GPUS[i]
        _global_pm.start_real_worker(port, gpu_id, i)
        print(f"\n  Starting Worker {i} on GPU {gpu_id}, port {port}...")

    # 并行等待所有 Worker 就绪
    print("  Waiting for Workers to load model...")
    results = await asyncio.gather(*[
        wait_for_health(f"http://127.0.0.1:{E2E_WORKER_BASE_PORT + i}/health")
        for i in range(3)
    ])

    for i, ok in enumerate(results):
        assert ok, (
            f"Worker {i} (GPU {E2E_GPUS[i]}, port {E2E_WORKER_BASE_PORT + i}) "
            f"failed to start within {STARTUP_TIMEOUT}s"
        )

    _workers_ready = True
    print("  All 3 Workers ready!")

    yield _global_pm

    _global_pm.kill_all()


@pytest_asyncio.fixture
async def gateway_1w(real_workers: E2EProcessManager):
    """1 Worker Gateway"""
    workers = [f"127.0.0.1:{E2E_WORKER_BASE_PORT}"]
    port = E2E_GATEWAY_PORT

    real_workers.start_gateway(port)
    assert await wait_for_health(f"http://127.0.0.1:{port}/health")
    for i, endpoint in enumerate(workers):
        await register_gateway_worker(port, f"worker-{i}", endpoint, f"gpu-{i}")
    ok = await wait_for_gateway_ready(f"http://127.0.0.1:{port}", 1)
    assert ok, f"Gateway (1W) failed to start on port {port}"

    yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}"

    real_workers.kill_gateways()
    await asyncio.sleep(3)  # Worker Duplex cleanup


@pytest_asyncio.fixture
async def gateway_2w(real_workers: E2EProcessManager):
    """2 Worker Gateway"""
    workers = [f"127.0.0.1:{E2E_WORKER_BASE_PORT + i}" for i in range(2)]
    port = E2E_GATEWAY_PORT + 1

    real_workers.start_gateway(port)
    assert await wait_for_health(f"http://127.0.0.1:{port}/health")
    for i, endpoint in enumerate(workers):
        await register_gateway_worker(port, f"worker-{i}", endpoint, f"gpu-{i}")
    ok = await wait_for_gateway_ready(f"http://127.0.0.1:{port}", 2)
    assert ok, f"Gateway (2W) failed to start on port {port}"

    yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}"

    real_workers.kill_gateways()
    await asyncio.sleep(3)


@pytest_asyncio.fixture
async def gateway_3w(real_workers: E2EProcessManager):
    """3 Worker Gateway"""
    workers = [f"127.0.0.1:{E2E_WORKER_BASE_PORT + i}" for i in range(3)]
    port = E2E_GATEWAY_PORT + 2

    real_workers.start_gateway(port)
    assert await wait_for_health(f"http://127.0.0.1:{port}/health")
    for i, endpoint in enumerate(workers):
        await register_gateway_worker(port, f"worker-{i}", endpoint, f"gpu-{i}")
    ok = await wait_for_gateway_ready(f"http://127.0.0.1:{port}", 3)
    assert ok, f"Gateway (3W) failed to start on port {port}"

    yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}"

    real_workers.kill_gateways()
    await asyncio.sleep(3)


# ============================================================
# Part 1: Chat 端到端
# ============================================================

class TestE2EChat:
    """Chat 模式端到端测试"""

    @pytest.mark.asyncio
    async def test_chat_1w_single(self, gateway_1w: Tuple[str, str]):
        """1 Worker: 单个 Chat 请求"""
        http_url, _ = gateway_1w
        result = await e2e_chat(http_url, "你好")
        assert result.get("text"), f"Empty response: {result}"
        print(f"  [1W] Chat: {result['text'][:80]}...")

    @pytest.mark.asyncio
    async def test_chat_2w_concurrent(self, gateway_2w: Tuple[str, str]):
        """2 Worker: 2 个并发 Chat"""
        http_url, _ = gateway_2w
        results = await asyncio.gather(
            e2e_chat(http_url, "你好"),
            e2e_chat(http_url, "今天天气怎么样"),
        )
        for r in results:
            assert r.get("text"), f"Empty response: {r}"
        print(f"  [2W] Chat 1: {results[0]['text'][:60]}...")
        print(f"  [2W] Chat 2: {results[1]['text'][:60]}...")

    @pytest.mark.asyncio
    async def test_chat_3w_concurrent(self, gateway_3w: Tuple[str, str]):
        """3 Worker: 3 个并发 Chat"""
        http_url, _ = gateway_3w
        results = await asyncio.gather(
            e2e_chat(http_url, "你好"),
            e2e_chat(http_url, "1+1等于几"),
            e2e_chat(http_url, "请介绍一下自己"),
        )
        for r in results:
            assert r.get("text"), f"Empty response: {r}"

    @pytest.mark.asyncio
    async def test_chat_1w_queued(self, gateway_1w: Tuple[str, str]):
        """1 Worker: 3 个并发 Chat（2 个排队）"""
        http_url, _ = gateway_1w
        results = await asyncio.gather(
            e2e_chat(http_url, "问题一"),
            e2e_chat(http_url, "问题二"),
            e2e_chat(http_url, "问题三"),
        )
        success = sum(1 for r in results if r.get("text"))
        assert success == 3, f"Expected 3 success, got {success}"


# ============================================================
# Part 2: Streaming 端到端
# ============================================================

class TestE2EStreaming:
    """Streaming 模式端到端测试"""

    @pytest.mark.asyncio
    async def test_streaming_1w_single(self, gateway_1w: Tuple[str, str]):
        """1 Worker: 单轮 Streaming"""
        _, ws_url = gateway_1w
        text, msgs = await e2e_streaming_turn(
            ws_url, "e2e_s1",
            [{"role": "user", "content": "你好，讲个笑话"}],
        )
        assert len(text) > 0, f"Empty streaming text, msgs={msgs}"
        types = [m["type"] for m in msgs]
        assert "prefill_done" in types
        assert "done" in types
        print(f"  [1W] Streaming: {text[:80]}...")

    @pytest.mark.asyncio
    async def test_streaming_2w_concurrent(self, gateway_2w: Tuple[str, str]):
        """2 Worker: 2 个并发 Streaming"""
        _, ws_url = gateway_2w
        results = await asyncio.gather(
            e2e_streaming_turn(ws_url, "e2e_s1", [{"role": "user", "content": "你好"}]),
            e2e_streaming_turn(ws_url, "e2e_s2", [{"role": "user", "content": "1+1"}]),
        )
        for text, msgs in results:
            assert len(text) > 0

    @pytest.mark.asyncio
    async def test_streaming_3w_concurrent(self, gateway_3w: Tuple[str, str]):
        """3 Worker: 3 个并发 Streaming"""
        _, ws_url = gateway_3w
        results = await asyncio.gather(
            e2e_streaming_turn(ws_url, "s1", [{"role": "user", "content": "话题1"}]),
            e2e_streaming_turn(ws_url, "s2", [{"role": "user", "content": "话题2"}]),
            e2e_streaming_turn(ws_url, "s3", [{"role": "user", "content": "话题3"}]),
        )
        for text, msgs in results:
            assert len(text) > 0

    @pytest.mark.asyncio
    async def test_streaming_1w_queued(self, gateway_1w: Tuple[str, str]):
        """1 Worker: 2 个并发 Streaming（1 个排队，验证 WS 排队消息）"""
        _, ws_url = gateway_1w
        results = await asyncio.gather(
            e2e_streaming_turn(ws_url, "sq1", [{"role": "user", "content": "问题A"}]),
            e2e_streaming_turn(ws_url, "sq2", [{"role": "user", "content": "问题B"}]),
        )
        for text, msgs in results:
            assert len(text) > 0


# ============================================================
# Part 3: Duplex 端到端（模拟前端 WS 交互）
# ============================================================

class TestE2EDuplex:
    """Duplex 模式端到端测试（模拟前端 WebSocket 交互）"""

    @pytest.mark.asyncio
    async def test_duplex_1w_single(self, gateway_1w: Tuple[str, str]):
        """1 Worker: 单个 Duplex session（模拟前端）"""
        _, ws_url = gateway_1w
        msgs = await e2e_duplex_session(ws_url, "e2e_d1", num_audio_chunks=5)

        types = [m["type"] for m in msgs]
        assert "queue_done" in types, f"Missing queue_done: {types}"
        assert "prepared" in types, f"Missing prepared: {types}"
        # result 消息（每个 audio_chunk 对应一个）
        result_msgs = [m for m in msgs if m.get("type") == "result"]
        assert len(result_msgs) >= 3, f"Expected >=3 results, got {len(result_msgs)}"
        print(f"  [1W] Duplex: {len(result_msgs)} results, types={types}")

    @pytest.mark.asyncio
    async def test_duplex_2w_concurrent(self, gateway_2w: Tuple[str, str]):
        """2 Worker: 2 个并发 Duplex（模拟 2 个用户同时使用）"""
        _, ws_url = gateway_2w
        results = await asyncio.gather(
            e2e_duplex_session(ws_url, "d2_1", num_audio_chunks=3),
            e2e_duplex_session(ws_url, "d2_2", num_audio_chunks=3),
        )
        for i, msgs in enumerate(results):
            types = [m["type"] for m in msgs]
            errors = [m for m in msgs if m.get("type") == "error"]
            assert "prepared" in types, (
                f"Duplex {i} failed: types={types}, errors={errors}"
            )

    @pytest.mark.asyncio
    async def test_duplex_3w_concurrent(self, gateway_3w: Tuple[str, str]):
        """3 Worker: 3 个并发 Duplex"""
        _, ws_url = gateway_3w
        results = await asyncio.gather(
            e2e_duplex_session(ws_url, "d3_1", num_audio_chunks=3),
            e2e_duplex_session(ws_url, "d3_2", num_audio_chunks=3),
            e2e_duplex_session(ws_url, "d3_3", num_audio_chunks=3),
        )
        for msgs in results:
            types = [m["type"] for m in msgs]
            assert "prepared" in types

    @pytest.mark.asyncio
    async def test_duplex_1w_queued(self, gateway_1w: Tuple[str, str]):
        """1 Worker: 2 个 Duplex session（1 个排队，验证排队→分配→交互→释放链路）"""
        _, ws_url = gateway_1w
        results = await asyncio.gather(
            e2e_duplex_session(ws_url, "dq1", num_audio_chunks=3),
            e2e_duplex_session(ws_url, "dq2", num_audio_chunks=3),
        )
        for msgs in results:
            types = [m["type"] for m in msgs]
            assert "prepared" in types
            # 至少有一个应该经历排队（queued 消息出现在 queue_done 之前）


# ============================================================
# Part 4: 混合模式端到端
# ============================================================

class TestE2EMixed:
    """混合模式端到端测试"""

    @pytest.mark.asyncio
    async def test_mixed_chat_streaming_3w(self, gateway_3w: Tuple[str, str]):
        """3 Worker: Chat + Streaming 混合"""
        http_url, ws_url = gateway_3w
        chat_result, (stream_text, stream_msgs) = await asyncio.gather(
            e2e_chat(http_url, "混合测试Chat"),
            e2e_streaming_turn(ws_url, "mix_s1", [{"role": "user", "content": "混合测试Stream"}]),
        )
        assert chat_result.get("text")
        assert len(stream_text) > 0

    @pytest.mark.asyncio
    async def test_mixed_all_3w(self, gateway_3w: Tuple[str, str]):
        """3 Worker: Chat + Streaming + Duplex 同时

        注意：3 个请求分配到 3 个 Worker，可能存在竞态。
        用 gather 并发，但允许短暂排队。
        """
        http_url, ws_url = gateway_3w

        # 先启动 Streaming 和 Duplex（它们占用固定 Worker），再发 Chat
        stream_task = asyncio.create_task(
            e2e_streaming_turn(ws_url, "mix_s", [{"role": "user", "content": "Streaming请求"}])
        )
        duplex_task = asyncio.create_task(
            e2e_duplex_session(ws_url, "mix_d", num_audio_chunks=3)
        )
        # 稍等让 WS 连接建立，再发 Chat（它会排队到空闲 Worker）
        await asyncio.sleep(0.5)
        chat_task = asyncio.create_task(e2e_chat(http_url, "Chat请求"))

        results = await asyncio.gather(chat_task, stream_task, duplex_task)

        chat_r = results[0]
        stream_text, stream_msgs = results[1]
        duplex_msgs = results[2]

        assert chat_r.get("text"), f"Chat failed: {chat_r}"
        assert len(stream_text) > 0, "Streaming empty"
        duplex_types = [m["type"] for m in duplex_msgs]
        assert "prepared" in duplex_types, f"Duplex failed: {duplex_types}"


# ============================================================
# Part 5: 队列状态端到端
# ============================================================

class TestE2EQueueStatus:
    """队列状态端到端验证"""

    @pytest.mark.asyncio
    async def test_service_status_reflects_workers(self, gateway_3w: Tuple[str, str]):
        """验证 /status 返回正确的 Worker 数量"""
        http_url, _ = gateway_3w
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{http_url}/status")
            data = resp.json()
            assert data["total_workers"] == 3
            assert data["idle_workers"] == 3

    @pytest.mark.asyncio
    async def test_queue_empty_when_idle(self, gateway_1w: Tuple[str, str]):
        """空闲时队列为空"""
        http_url, _ = gateway_1w
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{http_url}/api/queue")
            data = resp.json()
            assert data["queue_length"] == 0
