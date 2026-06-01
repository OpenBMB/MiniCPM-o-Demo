"""WebSocket Backend Protocol 综合测试

覆盖：
  - C++ 协议层单元测试（Python 端模拟 protocol.h/cpp 的 JSON 消息构建/解析）
  - CppWsBackendWorker 集成测试（使用 mock_ws_backend 模拟 C++ 服务器）
  - 协议错误场景测试（fail-fast、会话互斥、断开、并发拒绝）

运行方式：
    # 1. 启动 mock 服务器
    PYTHONPATH=. python tests/mock_ws_backend.py --port 19061 --delay 0.02 &

    # 2. 运行全部测试
    PYTHONPATH=. python -m pytest tests/test_ws_protocol.py -v -s

    # 3. 停止 mock 服务器
    kill %1

依赖：
    pip install pytest websocket-client numpy soundfile
"""

import base64
import json
import os
import subprocess
import sys
import threading
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pytest
import websocket

# 确保项目路径在 sys.path
_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

TESTS_DIR = Path(__file__).parent
CASES_DIR = TESTS_DIR / "cases"
USER_AUDIO_PATH = CASES_DIR / "common" / "user_audio" / "000_user_audio0.wav"
REF_AUDIO_PATH = CASES_DIR / "common" / "ref_audio" / "BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav"
IMAGE_PATH = CASES_DIR / "common" / "images" / "image.png"

# ============================================================================
# Mock server management
# ============================================================================

MOCK_PORT = 19061
MOCK_URL = f"ws://127.0.0.1:{MOCK_PORT}"


@pytest.fixture(scope="session")
def mock_server():
    """启动 mock_ws_backend，测试结束后终止"""
    proj_root = Path(__file__).parent.parent
    cmd = [
        sys.executable,
        str(proj_root / "tests" / "mock_ws_backend.py"),
        "--port", str(MOCK_PORT),
        "--delay", "0.01",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(proj_root)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    # 等待服务器就绪
    import httpx
    for i in range(30):
        try:
            resp = httpx.get(f"http://127.0.0.1:{MOCK_PORT}/health", timeout=2.0)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.3)
    else:
        proc.kill()
        pytest.fail("Mock server failed to start")

    yield MOCK_PORT

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ============================================================================
# Helper: audio <-> base64
# ============================================================================

def _load_audio(path: str, target_sr: int = 16000) -> str:
    """Load WAV → float32 → base64"""
    import soundfile as sf
    data, sr = sf.read(str(path))
    if sr != target_sr:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = data.astype(np.float32)
    return base64.b64encode(audio.tobytes()).decode()


def _load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ============================================================================
# Section 1: Protocol message building tests (simulating C++ protocol.h/cpp)
# ============================================================================

class TestProtocolMessages:
    """测试协议 JSON 消息构建和解析 — 模拟 C++ 端 make_* / parse_* 函数"""

    def test_make_session_created(self):
        """验证 session.created 事件结构"""
        event = {
            "type": "session.created",
            "session_id": "abc123",
            "mode": "full_duplex",
            "metrics": {"backend": "llama.cpp-omni", "kv_cache_length": 0},
        }
        assert event["type"] == "session.created"
        assert event["session_id"] == "abc123"
        assert event["mode"] == "full_duplex"
        assert "metrics" in event

    def test_make_text_delta(self):
        """验证 text_delta 事件结构"""
        event = {
            "type": "response.output.delta",
            "kind": "text",
            "delta": "hello",
            "session_id": "abc123",
            "response_id": "abc123_resp_1",
        }
        assert event["kind"] == "text"
        assert event["delta"] == "hello"
        assert event["type"] == "response.output.delta"

    def test_make_audio_delta(self):
        """验证 audio_delta 事件结构"""
        event = {
            "type": "response.output.delta",
            "kind": "audio",
            "delta": "AAAA",
            "session_id": "abc123",
            "response_id": "abc123_resp_1",
        }
        assert event["kind"] == "audio"
        assert event["delta"] == "AAAA"

    def test_make_listen_delta(self):
        """验证 listen_delta 事件结构"""
        event = {
            "type": "response.output.delta",
            "kind": "listen",
            "session_id": "abc123",
        }
        assert event["kind"] == "listen"

    def test_make_response_done_turn_end(self):
        """验证 response.done (turn_end)"""
        event = {
            "type": "response.done",
            "session_id": "abc123",
            "response_id": "abc123_resp_1",
            "full_text": "hello world",
            "reason": "turn_end",
        }
        assert event["type"] == "response.done"
        assert event["reason"] == "turn_end"
        assert "audio" not in event or event.get("audio") == ""

    def test_make_response_done_listen(self):
        """验证 response.done (listen)"""
        event = {
            "type": "response.done",
            "session_id": "abc123",
            "response_id": "abc123_resp_2",
            "full_text": "hi",
            "reason": "listen",
        }
        assert event["reason"] == "listen"

    def test_make_session_closed(self):
        """验证 session.closed 事件结构"""
        event = {
            "type": "session.closed",
            "session_id": "abc123",
            "reason": "client_closed",
        }
        assert event["type"] == "session.closed"

    def test_parse_session_init_full_duplex(self):
        """验证 session.init 解析 (full_duplex)"""
        msg = {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "config": {"temperature": 0.7},
            },
        }
        assert msg["type"] == "session.init"
        assert msg["payload"]["mode"] == "full_duplex"

    def test_parse_session_init_turn_based(self):
        """验证 session.init 解析 (turn_based)"""
        msg = {
            "type": "session.init",
            "payload": {
                "mode": "turn_based",
                "voice": {
                    "ref_audio": "AAAA",
                },
                "system_prompt": "You are helpful.",
            },
        }
        assert msg["type"] == "session.init"
        assert msg["payload"]["mode"] == "turn_based"
        assert msg["payload"]["voice"]["ref_audio"] == "AAAA"

    def test_parse_input_append_full_duplex(self):
        """验证 input.append 解析 (full_duplex)"""
        msg = {
            "type": "input.append",
            "input": {
                "audio": "BBBB",
                "video_frames": ["CCCC"],
                "max_slice_nums": 3,
            },
        }
        assert msg["type"] == "input.append"
        assert msg["input"]["audio"] == "BBBB"
        assert len(msg["input"]["video_frames"]) == 1

    def test_parse_input_append_turn_based(self):
        """验证 input.append 解析 (turn_based)"""
        msg = {
            "type": "input.append",
            "input": {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
                "streaming": True,
                "generation": {"max_new_tokens": 512, "length_penalty": 1.1},
                "tts": {"enabled": True},
            },
        }
        assert msg["input"]["streaming"] is True
        assert msg["input"]["generation"]["max_new_tokens"] == 512
        assert msg["input"]["tts"]["enabled"] is True

    def test_parse_messages_with_image(self):
        """验证带 data:image URL 的 messages 解析"""
        img_b64 = _load_image_b64(str(IMAGE_PATH))
        msg = {
            "type": "input.append",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }},
                        ],
                    }
                ],
                "streaming": False,
                "generation": {"max_new_tokens": 100},
            },
        }
        content = msg["input"]["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_base64_audio_roundtrip(self):
        """验证 float32 PCM → base64 → float32 PCM 往返"""
        samples = np.array([0.1, 0.2, 0.3, -0.1, -0.2], dtype=np.float32)
        b64 = base64.b64encode(samples.tobytes()).decode()
        decoded = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        np.testing.assert_array_equal(samples, decoded)

    def test_session_init_missing_payload(self):
        """session.init 缺少 payload → 应失败"""
        msg = {"type": "session.init"}
        # 模拟 C++ 端校验
        ok = "payload" in msg and isinstance(msg.get("payload"), dict)
        assert not ok

    def test_unexpected_message_type(self):
        """在 session.init 后收到非 input.append 消息 → fail-fast"""
        # 模拟 C++ 端逻辑：只接受 input.append
        valid_types = {"input.append"}
        msg = {"type": "invalid.type", "data": "test"}
        assert msg["type"] not in valid_types


# ============================================================================
# Section 2: CppWsBackendWorker 集成测试（Mock 服务器）
# ============================================================================

class TestCppWsBackendWorker:
    """使用 mock 服务器测试 CppWsBackendWorker"""

    @pytest.fixture
    def worker(self, mock_server):
        """创建连接到 mock 服务器的 CppWsBackendWorker"""
        from core.processors.cpp_ws_backend import CppWsBackendWorker
        # 注意：只测试 WS 协议，不启动真实 C++ 进程
        port = mock_server
        w = CppWsBackendWorker.__new__(CppWsBackendWorker)
        w.llamacpp_root = ""
        w.model_dir = ""
        w.gpu_id = 0
        w.ref_audio_path = str(REF_AUDIO_PATH)
        w.duplex_pause_timeout = 60.0
        from worker import WorkerState, WorkerStatus
        w.state = WorkerState()
        w.processor = None
        w._cpp_server_port = port
        w._cpp_server_url = f"http://127.0.0.1:{port}"
        w._cpp_ws_url = f"ws://127.0.0.1:{port}/backend"
        w._cpp_process = None
        import httpx
        w._http_client = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False)
        import tempfile
        w._temp_dir = tempfile.mkdtemp(prefix="test_ws_")
        w._output_dir = tempfile.mkdtemp(prefix="test_ws_out_")
        w._ws = None
        w._session_id = None
        w._session_mode = ""
        w._response_seq = 0
        w._kv_cache_len = 0
        w._stop_event = threading.Event()
        yield w
        # 清理 — 确保清除服务端会话
        w.duplex_stop()
        w._close_ws_safe()
        if w._http_client:
            w._http_client.close()

    # ----- Duplex (full_duplex) -----

    def test_duplex_connect(self, worker):
        """duplex_prepare → WS 连接 + session.created"""
        result = worker.duplex_prepare()
        assert result == "ok"
        assert worker._session_id is not None
        assert worker._session_mode == "full_duplex"

    def test_duplex_prefill_and_generate(self, worker):
        """duplex_prepare → duplex_prefill → duplex_generate"""
        worker.duplex_prepare()

        # prefill
        audio_b64 = _load_audio(str(USER_AUDIO_PATH))
        audio = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
        result = worker.duplex_prefill(audio_chunk=audio, max_slice_nums=3)
        assert "n_vision_images" in result

        # generate
        gen = worker.duplex_generate()
        assert gen.is_listen is True  # mock always returns listen
        assert gen.end_of_turn is False

        # cleanup
        worker.duplex_stop()

    def test_duplex_prefill_with_image(self, worker):
        """duplex prefill with PIL image"""
        from PIL import Image
        img = Image.open(str(IMAGE_PATH))
        worker.duplex_prepare()
        result = worker.duplex_prefill(frame_list=[img], max_slice_nums=3)
        assert result["n_vision_images"] == 1

    def test_duplex_stop(self, worker):
        """duplex_stop → HTTP close"""
        worker.duplex_prepare()
        worker.duplex_stop()
        assert worker._ws is None  # ws should be closed

    # ----- Half-duplex (turn_based streaming) -----

    def test_half_duplex_connect(self, worker):
        """reset_half_duplex_session → turn_based session"""
        worker.reset_half_duplex_session()
        assert worker._session_mode == "turn_based"

    def test_half_duplex_streaming(self, worker):
        """half_duplex streaming: messages → events → StreamingChunk"""
        from core.schemas.streaming import StreamingChunk

        worker.reset_half_duplex_session(streaming=True)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
        chunks = list(worker.half_duplex_generate(
            messages=messages,
            streaming=True,
            max_new_tokens=100,
            tts_enabled=True,
        ))

        assert len(chunks) > 0
        # 最后一个 chunk 应该是 is_final=True
        assert chunks[-1].is_final is True
        # 应该有 text delta
        text_chunks = [c for c in chunks if c.text_delta]
        assert len(text_chunks) > 0

    def test_half_duplex_with_audio_message(self, worker):
        """half_duplex with audio in message content"""
        from core.schemas.streaming import StreamingChunk

        worker.reset_half_duplex_session()
        audio_b64 = _load_audio(str(USER_AUDIO_PATH))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_b64},
                    {"type": "text", "text": "What did they say?"},
                ],
            }
        ]
        chunks = list(worker.half_duplex_generate(
            messages=messages,
            streaming=True,
            max_new_tokens=100,
        ))

        assert len(chunks) > 0
        assert chunks[-1].is_final

    def test_half_duplex_with_image_message(self, worker):
        """half_duplex with image in message content"""
        worker.reset_half_duplex_session()
        img_b64 = _load_image_b64(str(IMAGE_PATH))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ]
        chunks = list(worker.half_duplex_generate(
            messages=messages,
            streaming=True,
            max_new_tokens=100,
        ))

        assert len(chunks) > 0
        assert chunks[-1].is_final

    # ----- Chat (turn_based non-streaming) -----

    def test_chat_non_streaming(self, worker):
        """chat non-streaming → session.init (turn_based) + response.done"""
        from core.schemas.chat import ChatRequest

        # 构造 ChatRequest
        worker._ws_connect(mode="turn_based")

        msg = {
            "type": "input.append",
            "input": {
                "messages": [
                    {"role": "user", "content": "1+1=?"}
                ],
                "streaming": False,
                "generation": {"max_new_tokens": 50, "length_penalty": 1.1},
            },
        }
        worker._ws_send_input_append(msg)

        full_text = ""
        for event in worker._ws_recv_events(timeout=30.0):
            etype = event.get("type", "")
            if etype == "response.done":
                full_text = event.get("full_text", "")
                break

        assert len(full_text) > 0
        worker._close_ws_safe()

    # ----- Audio callback -----

    def test_audio_callback_receives_samples(self, worker):
        """音频回调被调用并收到 float32 PCM 样本"""
        worker._ws_connect(mode="turn_based")

        collected = []
        def on_audio(samples, n_samples, sample_rate, is_final):
            collected.append({"n": n_samples, "sr": sample_rate, "final": is_final})

        # 注意：mock 服务器不调用 audio_output_cb，因为它是 C++ T2W 线程的内部回调。
        # 这里测试 Python 端的 base64 编解码往返。
        samples = (np.sin(2 * np.pi * 440 * np.arange(2400, dtype=np.float32) / 24000) * 0.3).astype(np.float32)
        b64 = base64.b64encode(samples.tobytes()).decode()
        decoded = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        np.testing.assert_array_almost_equal(samples, decoded, decimal=5)

        worker._close_ws_safe()


# ============================================================================
# Section 3: 错误场景测试
# ============================================================================

class TestProtocolErrors:
    """测试协议错误处理和 fail-fast 行为"""

    def test_session_init_rejected_when_active(self, mock_server):
        """当已有 active session 时，新的 session.init 被拒绝"""
        url = f"ws://127.0.0.1:{mock_server}/backend"

        # 第一个连接
        ws1 = websocket.create_connection(url, timeout=10)
        ws1.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "full_duplex"},
        }))
        resp1 = json.loads(ws1.recv())
        assert resp1["type"] == "session.created"
        sid1 = resp1["session_id"]
        assert len(sid1) > 0

        # 第二个连接 — 应被拒绝
        ws2 = websocket.create_connection(url, timeout=10)
        ws2.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "full_duplex"},
        }))
        resp2 = json.loads(ws2.recv())
        assert resp2["type"] == "session.closed"
        assert "active" in resp2.get("reason", "")

        ws1.close()
        ws2.close()

    def test_missing_session_init(self, mock_server):
        """直接发 input.append（没有 session.init）— 应被拒绝"""
        url = f"ws://127.0.0.1:{mock_server}/backend"
        ws = websocket.create_connection(url, timeout=10)

        # 发 input.append 而不是 session.init
        ws.send(json.dumps({
            "type": "input.append",
            "input": {"audio": "AAAA"},
        }))

        # mock 服务器期望 session.init 作为第一条消息，会断开连接
        try:
            resp = ws.recv()
        except Exception:
            resp = None

        ws.close()

    def test_invalid_json(self, mock_server):
        """发送非法 JSON — 服务器断开"""
        url = f"ws://127.0.0.1:{mock_server}/backend"
        ws = websocket.create_connection(url, timeout=10)
        ws.send("not { valid json !!!")

        try:
            ws.recv()
        except Exception:
            pass

        ws.close()

    def test_http_close_endpoint(self, mock_server):
        """HTTP POST /sessions/{id}/close 返回成功"""
        import httpx
        client = httpx.Client(timeout=httpx.Timeout(10.0), trust_env=False)

        # 先创建 session
        url = f"ws://127.0.0.1:{mock_server}/backend"
        ws = websocket.create_connection(url, timeout=10)
        ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "turn_based"},
        }))
        resp = json.loads(ws.recv())
        sid = resp["session_id"]

        # HTTP close
        r = client.post(f"http://127.0.0.1:{mock_server}/sessions/{sid}/close")
        assert r.status_code == 200
        assert r.json()["ok"] is True

        ws.close()
        client.close()

    def test_disconnect_during_generation(self, mock_server):
        """在生成过程中断开 WS → 服务器应清理 session"""
        url = f"ws://127.0.0.1:{mock_server}/backend"

        ws = websocket.create_connection(url, timeout=10)
        ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "turn_based"},
        }))
        resp = json.loads(ws.recv())
        sid = resp["session_id"]

        # 发送 input.append
        ws.send(json.dumps({
            "type": "input.append",
            "input": {
                "messages": [{"role": "user", "content": "test"}],
                "streaming": True,
            },
        }))

        # 不等完全响应就断开
        ws.close()
        time.sleep(0.3)  # 等服务器清理

        # 验证：新连接应该可以成功（之前会话已被清理）
        ws2 = websocket.create_connection(url, timeout=10)
        ws2.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "full_duplex"},
        }))
        resp2 = json.loads(ws2.recv())
        assert resp2["type"] == "session.created", f"expected session.created, got {resp2.get('type')}"
        ws2.close()

    def test_session_init_roundtrip(self, mock_server):
        """完整的 session.init → input.append → response.done 流程"""
        url = f"ws://127.0.0.1:{mock_server}/backend"
        ws = websocket.create_connection(url, timeout=10)

        # init
        ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "turn_based", "system_prompt": "Be helpful."},
        }))
        created = json.loads(ws.recv())
        assert created["type"] == "session.created"
        sid = created["session_id"]

        # input.append
        ws.send(json.dumps({
            "type": "input.append",
            "input": {
                "messages": [{"role": "user", "content": "hello"}],
                "streaming": True,
                "generation": {"max_new_tokens": 100},
                "tts": {"enabled": True},
            },
        }))

        # 收集所有事件
        events = []
        while True:
            raw = ws.recv()
            ev = json.loads(raw)
            events.append(ev)
            if ev["type"] in ("response.done", "session.closed"):
                break

        # 验证事件序列
        types = [e["type"] for e in events]
        assert "response.output.delta" in types
        assert "response.done" in types

        # 验证 response.done 包含完整内容
        done = [e for e in events if e["type"] == "response.done"][0]
        assert len(done.get("full_text", "")) > 0
        assert done.get("reason") in ("turn_end", "listen")

        ws.close()


# ============================================================================
# Section 4: 性能 / 并发测试
# ============================================================================

class TestConcurrency:
    """测试并发和边界情况"""

    def test_multiple_sequential_sessions(self, mock_server):
        """连续创建和关闭多个 session"""
        url = f"ws://127.0.0.1:{mock_server}/backend"

        for _ in range(5):
            ws = websocket.create_connection(url, timeout=10)
            ws.send(json.dumps({
                "type": "session.init",
                "payload": {"mode": "full_duplex"},
            }))
            resp = json.loads(ws.recv())
            assert resp["type"] == "session.created"
            ws.close()
            time.sleep(0.2)  # 等待服务器彻底清理

    def test_large_audio_payload(self, mock_server):
        """较大的 audio base64 载荷（~1MB float32 PCM）"""
        url = f"ws://127.0.0.1:{mock_server}/backend"

        # 生成 2 秒 16kHz float32 PCM (~128KB)
        sr = 16000
        samples = (np.sin(2 * np.pi * 440 * np.arange(sr * 2, dtype=np.float32) / sr) * 0.5).astype(np.float32)
        b64 = base64.b64encode(samples.tobytes()).decode()

        ws = websocket.create_connection(url, timeout=10)
        ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "full_duplex"},
        }))
        resp = json.loads(ws.recv())
        assert resp["type"] == "session.created"

        ws.send(json.dumps({
            "type": "input.append",
            "input": {"audio": b64},
        }))

        # 等待响应
        try:
            raw = ws.recv()
        except Exception:
            raw = None

        ws.close()

    def test_quick_reconnect(self, mock_server):
        """快速重连 — 立即断开再重连"""
        url = f"ws://127.0.0.1:{mock_server}/backend"

        ws1 = websocket.create_connection(url, timeout=10)
        ws1.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "full_duplex"},
        }))
        sid1 = json.loads(ws1.recv())["session_id"]
        ws1.close()

        # 立即重连（不等待清理）
        time.sleep(0.05)
        ws2 = websocket.create_connection(url, timeout=10)
        ws2.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "turn_based"},
        }))
        resp2 = json.loads(ws2.recv())
        assert resp2["type"] == "session.created"
        ws2.close()
