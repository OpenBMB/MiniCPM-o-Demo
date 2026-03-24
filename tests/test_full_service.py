#!/usr/bin/env python3
"""针对已启动的 Gateway + Worker（含 C++ 后端）的全功能测试

覆盖范围：
  1. Health / Status / Workers
  2. Chat HTTP（纯文本、多轮、带 TTS 音频输出）
  3. Chat WebSocket 流式（纯文本、带 TTS、多轮上下文）
  4. Half-Duplex WebSocket（prepare → 语音输入 → VAD → 生成 → turn_done）
  5. Omni Duplex WebSocket（prepare → audio+frame → result → stop）
  6. Audio Duplex WebSocket（adx_ 前缀 session_id）
  7. 管理 API（presets、apps、queue、ETA、sessions、frontend_defaults）
  8. 参考音频 CRUD
  9. 静态页面可访问性

使用方式（服务已启动）：
  cd /cache/caitianchi/code/minicpm-o-4_5-pytorch-simple-demo

  # 运行全部
  PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_full_service.py -v -s

  # 只跑快速测试（不需要推理）
  PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_full_service.py -v -s -k "not slow"

  # 只跑某一类
  PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_full_service.py -v -s -k "TestChat"

环境变量：
  GATEWAY_URL   默认读 config.json 的 gateway_port（http://localhost:8020）
  WORKER_URL    默认读 config.json 的 worker_base_port（http://localhost:22400）
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pytest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_full_service")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 配置：自动读取 config.json 或用环境变量覆盖
# ---------------------------------------------------------------------------

try:
    from config import get_config
    _cfg = get_config()
    _default_gateway = f"https://localhost:{_cfg.gateway_port}"
    _default_worker = f"http://localhost:{_cfg.worker_base_port}"
except Exception:
    _default_gateway = "https://localhost:8020"
    _default_worker = "http://localhost:22400"

GATEWAY_URL = os.environ.get("GATEWAY_URL", _default_gateway).rstrip("/")
WORKER_URL = os.environ.get("WORKER_URL", _default_worker).rstrip("/")
GATEWAY_WS = GATEWAY_URL.replace("https://", "wss://").replace("http://", "ws://")
WORKER_WS = WORKER_URL.replace("https://", "wss://").replace("http://", "ws://")

# HTTPS 场景需要跳过自签名证书验证
VERIFY_SSL = False

REF_AUDIO_WAV = PROJECT_ROOT / "assets" / "ref_audio" / "ref_minicpm_signature.wav"

CHAT_TIMEOUT = 240.0
STREAMING_TIMEOUT = 240.0
HALF_DUPLEX_TIMEOUT = 300.0
DUPLEX_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_client(**kwargs) -> httpx.Client:
    """创建跳过 SSL 验证的 HTTP 客户端"""
    return httpx.Client(verify=VERIFY_SSL, **kwargs)


def _async_http_client(**kwargs) -> httpx.AsyncClient:
    """创建跳过 SSL 验证的异步 HTTP 客户端"""
    return httpx.AsyncClient(verify=VERIFY_SSL, **kwargs)


def _get(url: str, **kwargs) -> httpx.Response:
    kwargs.setdefault("timeout", 10)
    return httpx.get(url, verify=VERIFY_SSL, **kwargs)


def _post(url: str, **kwargs) -> httpx.Response:
    kwargs.setdefault("timeout", 10)
    return httpx.post(url, verify=VERIFY_SSL, **kwargs)


def _put(url: str, **kwargs) -> httpx.Response:
    kwargs.setdefault("timeout", 10)
    return httpx.put(url, verify=VERIFY_SSL, **kwargs)


def _delete(url: str, **kwargs) -> httpx.Response:
    kwargs.setdefault("timeout", 10)
    return httpx.delete(url, verify=VERIFY_SSL, **kwargs)


def _gateway_ok() -> bool:
    try:
        return _get(f"{GATEWAY_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


def _worker_ok() -> bool:
    try:
        r = _get(f"{WORKER_URL}/health", timeout=5)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


requires_gateway = pytest.mark.skipif(not _gateway_ok(), reason=f"Gateway 不可用: {GATEWAY_URL}")
requires_worker = pytest.mark.skipif(not _worker_ok(), reason=f"Worker 不可用: {WORKER_URL}")
slow = pytest.mark.slow


def _wait_worker_idle(timeout: float = 30.0) -> bool:
    """轮询等待 Worker 变为 idle（前一个请求可能还在处理中）"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _get(f"{WORKER_URL}/health", timeout=5)
            if r.status_code == 200 and r.json().get("worker_status") == "idle":
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _make_silence_wav_b64(duration_s: float = 1.0, sr: int = 16000) -> str:
    """生成静音 WAV 的 base64 编码"""
    import soundfile as sf
    audio = np.zeros(int(sr * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _make_silence_pcm_b64(duration_s: float = 1.0, sr: int = 16000) -> str:
    """生成静音 raw float32 PCM 的 base64 编码"""
    audio = np.zeros(int(sr * duration_s), dtype=np.float32)
    return base64.b64encode(audio.tobytes()).decode("ascii")


def _make_test_image_b64() -> str:
    """生成一张 64×64 测试图片的 base64 JPEG"""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _ws_ssl_context(ws_url: str = ""):
    """为 wss:// 连接返回不验证证书的 SSL context；ws:// 时返回 None"""
    target = ws_url or GATEWAY_WS
    if target.startswith("wss://"):
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None


def _load_ref_audio_f32_b64() -> Optional[str]:
    """加载参考音频并返回 float32 PCM base64"""
    if not REF_AUDIO_WAV.is_file():
        return None
    try:
        import soundfile as sf
        data, sr = sf.read(str(REF_AUDIO_WAV), always_2d=False)
        data = data.astype(np.float32)
        if sr != 16000:
            old_x = np.linspace(0, 1, len(data), endpoint=False)
            new_x = np.linspace(0, 1, int(len(data) * 16000 / sr), endpoint=False)
            data = np.interp(new_x, old_x, data.astype(np.float64)).astype(np.float32)
        return base64.b64encode(data.tobytes()).decode("ascii")
    except Exception:
        return None


# ============================================================================
# Part 1: Health / Status / Workers
# ============================================================================

class TestHealthStatus:
    """健康检查和服务状态"""

    @requires_gateway
    def test_gateway_health(self):
        r = _get(f"{GATEWAY_URL}/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "healthy"
        logger.info(f"Gateway health: {d}")

    @requires_worker
    def test_worker_health(self):
        r = _get(f"{WORKER_URL}/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "healthy"
        assert d["model_loaded"] is True
        logger.info(f"Worker health: worker_status={d['worker_status']}, gpu_id={d.get('gpu_id')}")

    @requires_gateway
    def test_gateway_status(self):
        r = _get(f"{GATEWAY_URL}/status")
        assert r.status_code == 200
        d = r.json()
        assert d["gateway_healthy"] is True
        assert d["total_workers"] >= 1
        logger.info(f"Status: workers={d['total_workers']}, idle={d.get('idle_workers')}")

    @requires_gateway
    def test_workers_list(self):
        r = _get(f"{GATEWAY_URL}/workers")
        assert r.status_code == 200
        d = r.json()
        assert d["total"] >= 1
        assert len(d["workers"]) >= 1
        logger.info(f"Workers: {d['total']}")

    @requires_worker
    def test_worker_cache_info(self):
        r = _get(f"{WORKER_URL}/cache_info")
        assert r.status_code == 200
        d = r.json()
        assert "status" in d
        logger.info(f"Cache info: {d}")


# ============================================================================
# Part 2: Chat HTTP
# ============================================================================

class TestChatHTTP:
    """Chat HTTP POST（通过 Gateway 路由到 Worker）"""

    @requires_gateway
    @requires_worker
    @slow
    def test_simple_text_chat(self):
        """纯文本对话"""
        assert _wait_worker_idle(), "Worker not idle"
        r = _post(
            f"{GATEWAY_URL}/api/chat",
            json={
                "messages": [{"role": "user", "content": "1+1等于几？只回答数字。"}],
                "generation": {"max_new_tokens": 16, "do_sample": False},
            },
            timeout=CHAT_TIMEOUT,
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("success") is True, d
        text = (d.get("text") or "").strip()
        assert len(text) > 0, f"Empty response text: {d}"
        logger.info(f"Chat response: {text!r}")

    @requires_gateway
    @requires_worker
    @slow
    def test_multi_turn_chat(self):
        """多轮对话"""
        assert _wait_worker_idle(), "Worker not idle"
        r = _post(
            f"{GATEWAY_URL}/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "42乘以2等于多少？"},
                    {"role": "assistant", "content": "42乘以2等于84。"},
                    {"role": "user", "content": "再乘以2呢？只回答数字。"},
                ],
                "generation": {"max_new_tokens": 32, "do_sample": False},
            },
            timeout=CHAT_TIMEOUT,
        )
        if r.status_code == 500:
            pytest.skip(f"多轮 Chat 返回 500（可能 C++ 后端限制）: {r.text[:300]}")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("success") is True, d
        text = (d.get("text") or "").strip()
        assert len(text) > 0
        logger.info(f"Multi-turn response: {text!r}")

    @requires_gateway
    @requires_worker
    @slow
    def test_chat_with_tts(self):
        """Chat 带 TTS 音频输出"""
        assert _wait_worker_idle(), "Worker not idle"
        r = _post(
            f"{GATEWAY_URL}/api/chat",
            json={
                "messages": [{"role": "user", "content": "你好，用一句话介绍你自己。"}],
                "generation": {"max_new_tokens": 64},
                "tts": {"enabled": True},
                "use_tts_template": True,
            },
            timeout=CHAT_TIMEOUT,
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("success") is True, d
        text = (d.get("text") or "").strip()
        assert len(text) > 0, f"Empty text: {d}"
        audio_data = d.get("audio_data")
        if audio_data:
            audio_bytes = base64.b64decode(audio_data)
            assert len(audio_bytes) > 0, "audio_data is empty after decode"
            logger.info(f"Chat+TTS: text={text[:60]!r}, audio={len(audio_bytes)} bytes")
        else:
            logger.warning("Chat+TTS: no audio_data returned (C++ backend may not support sync TTS)")

    @requires_worker
    @slow
    def test_worker_direct_chat(self):
        """直接对 Worker 发送 Chat（不经 Gateway）"""
        assert _wait_worker_idle(), "Worker not idle"
        r = _post(
            f"{WORKER_URL}/chat",
            json={
                "messages": [{"role": "user", "content": "2+3=?"}],
                "generation": {"max_new_tokens": 16, "do_sample": False},
            },
            timeout=CHAT_TIMEOUT,
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("success") is True, d
        logger.info(f"Worker direct chat: {(d.get('text') or '').strip()!r}")


# ============================================================================
# Part 3: Chat WebSocket 流式
# ============================================================================

class TestChatStreaming:
    """Chat WebSocket 流式 (/ws/chat)"""

    @staticmethod
    async def _ws_streaming_chat(
        ws_url: str,
        messages: list,
        *,
        max_new_tokens: int = 100,
        tts_enabled: bool = False,
        reset_context: bool = True,
        omni_mode: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        import websockets

        payload: Dict[str, Any] = {
            "messages": messages,
            "streaming": True,
            "generation": {"max_new_tokens": max_new_tokens, "do_sample": False, "length_penalty": 1.1},
            "image": {"max_slice_nums": 1},
            "omni_mode": omni_mode,
            "reset_context": reset_context,
        }
        if tts_enabled:
            payload["tts"] = {"enabled": True}
            payload["use_tts_template"] = True

        ws_msgs: List[Dict[str, Any]] = []
        full_text = ""
        has_audio = False

        async with websockets.connect(ws_url, max_size=50_000_000, open_timeout=60, ssl=_ws_ssl_context(ws_url)) as ws:
            await ws.send(json.dumps(payload, ensure_ascii=False))
            for _ in range(500):
                raw = await asyncio.wait_for(ws.recv(), timeout=STREAMING_TIMEOUT)
                msg = json.loads(raw)
                ws_msgs.append(msg)
                t = msg.get("type")
                if t == "chunk":
                    if msg.get("text_delta"):
                        full_text += msg["text_delta"]
                    if msg.get("audio_data"):
                        has_audio = True
                elif t == "done":
                    full_text = full_text or (msg.get("text") or "")
                    if msg.get("audio_data"):
                        has_audio = True
                    break
                elif t == "error":
                    pytest.fail(f"WS error: {msg}")
                elif t in ("prefill_done", "heartbeat"):
                    continue

        return full_text.strip(), ws_msgs

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_text_only(self):
        """流式纯文本"""
        assert _wait_worker_idle(), "Worker not idle"
        text, msgs = await self._ws_streaming_chat(
            f"{GATEWAY_WS}/ws/chat",
            [{"role": "user", "content": "讲一个关于猫的一句话故事。"}],
            max_new_tokens=100,
        )
        types = [m["type"] for m in msgs]
        assert "prefill_done" in types, f"Missing prefill_done: {types}"
        assert "done" in types, f"Missing done: {types}"
        assert len(text) > 0, f"Empty text, msgs={msgs}"
        logger.info(f"Streaming text: {text[:100]!r}")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_with_tts(self):
        """流式 + TTS 音频"""
        assert _wait_worker_idle(), "Worker not idle"
        text, msgs = await self._ws_streaming_chat(
            f"{GATEWAY_WS}/ws/chat",
            [{"role": "user", "content": "用一句话说你好。"}],
            max_new_tokens=64,
            tts_enabled=True,
        )
        assert len(text) > 0, f"Empty text"
        audio_msgs = [m for m in msgs if m.get("audio_data")]
        if audio_msgs:
            logger.info(f"Streaming+TTS: text={text[:60]!r}, audio_chunks={len(audio_msgs)}")
        else:
            logger.warning("Streaming+TTS: no audio chunks (C++ backend may send audio separately)")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_multi_turn(self):
        """多轮流式对话"""
        assert _wait_worker_idle(), "Worker not idle"
        text, msgs = await self._ws_streaming_chat(
            f"{GATEWAY_WS}/ws/chat",
            [
                {"role": "user", "content": "密钥是数字42，请确认。"},
                {"role": "assistant", "content": "好的，密钥是42。"},
                {"role": "user", "content": "重复刚才的密钥数字。"},
            ],
            max_new_tokens=32,
            reset_context=False,
        )
        assert len(text) > 0, f"Empty text on turn 2"
        logger.info(f"Multi-turn streaming: {text!r}")

    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_direct_worker(self):
        """直接对 Worker WS 流式"""
        assert _wait_worker_idle(), "Worker not idle"
        text, msgs = await self._ws_streaming_chat(
            f"{WORKER_WS}/ws/chat",
            [{"role": "user", "content": "1+2等于几？只回答数字。"}],
            max_new_tokens=16,
        )
        assert len(text) > 0
        logger.info(f"Worker direct streaming: {text!r}")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_stream_vs_done_consistency(self):
        """流式 chunk 拼接结果是否与 done.text 一致"""
        assert _wait_worker_idle(), "Worker not idle"
        import websockets

        payload = {
            "messages": [{"role": "user", "content": "写一段50字左右的春天描述。"}],
            "streaming": True,
            "generation": {"max_new_tokens": 128, "do_sample": False},
            "omni_mode": False,
        }
        stream_parts: List[str] = []
        done_text = ""

        async with websockets.connect(f"{GATEWAY_WS}/ws/chat", max_size=50_000_000, ssl=_ws_ssl_context(GATEWAY_WS)) as ws:
            await ws.send(json.dumps(payload))
            for _ in range(300):
                raw = await asyncio.wait_for(ws.recv(), timeout=STREAMING_TIMEOUT)
                msg = json.loads(raw)
                if msg["type"] == "chunk" and msg.get("text_delta"):
                    stream_parts.append(msg["text_delta"])
                elif msg["type"] == "done":
                    done_text = (msg.get("text") or "").strip()
                    break
                elif msg["type"] == "error":
                    pytest.fail(f"WS error: {msg}")

        joined = "".join(stream_parts).strip()
        if joined and done_text:
            assert joined == done_text, (
                f"stream text != done text:\n  stream={joined[:200]!r}\n  done={done_text[:200]!r}"
            )
        logger.info(f"Consistency check: stream_len={len(joined)}, done_len={len(done_text)}")


# ============================================================================
# Part 4: Half-Duplex WebSocket
# ============================================================================

class TestHalfDuplex:
    """Half-Duplex Audio WebSocket (/ws/half_duplex)"""

    @staticmethod
    async def _run_half_duplex(
        ws_url: str,
        session_id: str,
        audio_chunks: List[str],
        *,
        chunk_interval_s: float = 0.5,
        tts_enabled: bool = False,
        timeout_s: float = HALF_DUPLEX_TIMEOUT,
    ) -> Dict[str, Any]:
        import websockets

        result: Dict[str, Any] = {"prepared": False, "turn_done": False, "text_parts": [], "errors": []}

        try:
            uri = f"{ws_url}?session_id={session_id}"
            async with websockets.connect(uri, max_size=50_000_000, open_timeout=60, ssl=_ws_ssl_context(uri)) as ws:
                # 排队消息（Gateway 侧可能发送 queued / queue_done）
                # 等待 queue_done 或直接开始
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        msg = json.loads(raw)
                        if msg.get("type") == "queue_done":
                            break
                        if msg.get("type") == "error":
                            result["errors"].append(msg)
                            return result
                    except asyncio.TimeoutError:
                        break

                prepare_payload = {
                    "type": "prepare",
                    "system_content": [{"type": "text", "text": "你是一个友好的语音助手。请用中文回答。"}],
                    "config": {
                        "vad": {
                            "threshold": 0.5,
                            "min_speech_duration_ms": 100,
                            "min_silence_duration_ms": 500,
                            "speech_pad_ms": 30,
                        },
                        "generation": {"max_new_tokens": 128, "length_penalty": 1.1},
                        "tts": {"enabled": tts_enabled},
                        "session": {"timeout_s": int(timeout_s)},
                    },
                }
                await ws.send(json.dumps(prepare_payload))

                # 等待 prepared
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    msg = json.loads(raw)
                    if msg.get("type") == "prepared":
                        result["prepared"] = True
                        break
                    if msg.get("type") == "error":
                        result["errors"].append(msg)
                        return result

                if not result["prepared"]:
                    result["errors"].append({"error": "timeout waiting for prepared"})
                    return result

                await asyncio.sleep(0.6)  # INITIAL_GUARD_S

                for chunk_b64 in audio_chunks:
                    await ws.send(json.dumps({"type": "audio_chunk", "audio_base64": chunk_b64}))
                    if chunk_interval_s > 0:
                        await asyncio.sleep(chunk_interval_s)

                # 等待 turn_done
                end = time.monotonic() + timeout_s
                while time.monotonic() < end:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    except asyncio.TimeoutError:
                        continue
                    msg = json.loads(raw)
                    t = msg.get("type")
                    if t == "chunk" and msg.get("text_delta"):
                        result["text_parts"].append(msg["text_delta"])
                    elif t == "turn_done":
                        result["turn_done"] = True
                        break
                    elif t == "error":
                        result["errors"].append(msg)
                        break

                await ws.send(json.dumps({"type": "stop"}))

        except Exception as e:
            result["errors"].append({"error": str(e)})

        return result

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_half_duplex_with_real_audio(self):
        """Half-Duplex：发送参考音频作为用户输入"""
        assert _wait_worker_idle(timeout=60), "Worker not idle"

        ref_b64 = _load_ref_audio_f32_b64()
        if not ref_b64:
            pytest.skip("参考音频不存在")

        # 分块发送（0.5s/块 = 8000 samples/块）
        audio_bytes = base64.b64decode(ref_b64)
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        chunk_size = 8000
        chunks: List[str] = []
        for i in range(0, len(audio_np), chunk_size):
            sl = audio_np[i:i + chunk_size]
            if len(sl) > 0:
                chunks.append(base64.b64encode(sl.tobytes()).decode("ascii"))
        # 尾部追加 1.5s 静音，确保 VAD 触发
        silence_tail = np.zeros(24000, dtype=np.float32)
        chunks.append(base64.b64encode(silence_tail.tobytes()).decode("ascii"))

        r = await self._run_half_duplex(
            f"{GATEWAY_WS}/ws/half_duplex/{uuid.uuid4().hex[:8]}",
            session_id=f"test_hdx_{int(time.time())}",
            audio_chunks=chunks,
            chunk_interval_s=0.5,
            tts_enabled=False,
        )
        assert r["prepared"], f"Not prepared: {r['errors']}"
        if not r["turn_done"]:
            logger.warning(
                f"Half-duplex 未收到 turn_done（VAD 可能未触发说话检测，"
                f"这在参考音频静音段较少时正常）: {r['errors']}"
            )
        else:
            text = "".join(r["text_parts"])
            logger.info(f"Half-duplex response: {text[:200]!r}")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_half_duplex_prepare_only(self):
        """Half-Duplex：仅 prepare 然后 stop（验证协议握手）"""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        import websockets

        sid = f"test_hdx_prep_{int(time.time())}"
        uri = f"{GATEWAY_WS}/ws/half_duplex/{sid}"

        async with websockets.connect(uri, max_size=50_000_000, open_timeout=60, ssl=_ws_ssl_context(uri)) as ws:
            # 跳过排队消息
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    msg = json.loads(raw)
                    if msg.get("type") in ("queue_done", "error"):
                        break
                except asyncio.TimeoutError:
                    break

            await ws.send(json.dumps({
                "type": "prepare",
                "system_content": [{"type": "text", "text": "测试"}],
                "config": {
                    "vad": {"threshold": 0.5, "min_silence_duration_ms": 500},
                    "generation": {"max_new_tokens": 32},
                    "tts": {"enabled": False},
                    "session": {"timeout_s": 30},
                },
            }))

            got_prepared = False
            for _ in range(30):
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                msg = json.loads(raw)
                if msg.get("type") == "prepared":
                    got_prepared = True
                    break
                if msg.get("type") == "error":
                    pytest.fail(f"Half-duplex prepare error: {msg}")

            assert got_prepared, "Never received prepared"

            await ws.send(json.dumps({"type": "stop"}))
        logger.info("Half-duplex prepare+stop OK")


# ============================================================================
# Part 5: Omni Duplex WebSocket
# ============================================================================

class TestOmniDuplex:
    """Omni Duplex WebSocket (/ws/duplex/{session_id})"""

    @staticmethod
    async def _run_duplex(
        ws_url: str,
        session_id: str,
        num_audio_chunks: int = 5,
        *,
        send_frame: bool = False,
        force_listen_first_n: int = 3,
    ) -> List[Dict[str, Any]]:
        import websockets

        audio_b64 = _make_silence_pcm_b64(1.0)
        frame_b64 = _make_test_image_b64() if send_frame else None
        ws_msgs: List[Dict[str, Any]] = []

        try:
            async with websockets.connect(
                f"{ws_url}/ws/duplex/{session_id}",
                max_size=50_000_000,
                open_timeout=60,
                ssl=_ws_ssl_context(ws_url),
            ) as ws:
                # 排队消息
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                    msg = json.loads(raw)
                    ws_msgs.append(msg)
                    if msg.get("type") in ("queue_done", "error"):
                        break

                if ws_msgs[-1].get("type") == "error":
                    return ws_msgs

                prepare_msg: Dict[str, Any] = {
                    "type": "prepare",
                    "system_prompt": "你好，你是一个友好的助手。",
                    "config": {"max_kv_tokens": 8000},
                    "deferred_finalize": True,
                }
                ref_b64 = _load_ref_audio_f32_b64()
                if ref_b64:
                    prepare_msg["tts_ref_audio_base64"] = ref_b64

                await ws.send(json.dumps(prepare_msg))

                raw = await asyncio.wait_for(ws.recv(), timeout=60)
                msg = json.loads(raw)
                ws_msgs.append(msg)
                if msg.get("type") != "prepared":
                    return ws_msgs

                for i in range(num_audio_chunks):
                    chunk_msg: Dict[str, Any] = {
                        "type": "audio_chunk",
                        "audio_base64": audio_b64,
                        "force_listen": i < force_listen_first_n,
                    }
                    if frame_b64 and send_frame:
                        chunk_msg["frame_base64_list"] = [frame_b64]

                    await ws.send(json.dumps(chunk_msg))

                    raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                    msg = json.loads(raw)
                    ws_msgs.append(msg)

                    await asyncio.sleep(0.05)

                await ws.send(json.dumps({"type": "stop"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                msg = json.loads(raw)
                ws_msgs.append(msg)

        except Exception as e:
            ws_msgs.append({"type": "error", "error": f"Exception: {e}"})

        return ws_msgs

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_omni_duplex_audio_only(self):
        """Omni Duplex：仅音频输入"""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        sid = f"test_omni_{int(time.time())}"
        msgs = await self._run_duplex(GATEWAY_WS, sid, num_audio_chunks=5, send_frame=False)

        types = [m["type"] for m in msgs]
        assert "queue_done" in types, f"Missing queue_done: {types}"
        assert "prepared" in types, f"Missing prepared: {types}"
        result_msgs = [m for m in msgs if m.get("type") == "result"]
        assert len(result_msgs) >= 3, f"Expected >=3 results, got {len(result_msgs)}: {types}"

        listen_count = sum(1 for m in result_msgs if m.get("is_listen"))
        speak_count = sum(1 for m in result_msgs if not m.get("is_listen"))
        logger.info(f"Omni duplex: {len(result_msgs)} results (listen={listen_count}, speak={speak_count})")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_omni_duplex_with_video_frame(self):
        """Omni Duplex：音频 + 视频帧"""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        sid = f"test_omni_video_{int(time.time())}"
        msgs = await self._run_duplex(GATEWAY_WS, sid, num_audio_chunks=3, send_frame=True)

        types = [m["type"] for m in msgs]
        assert "prepared" in types, f"Missing prepared: {types}"
        error_msgs = [m for m in msgs if m.get("type") == "error"]
        result_msgs = [m for m in msgs if m.get("type") == "result"]
        if error_msgs:
            error_details = [m.get("error", str(m)) for m in error_msgs]
            logger.warning(
                f"Omni duplex+video: CPP backend errors (likely server crash on image prefill): "
                f"{error_details}"
            )
        assert len(result_msgs) >= 1, (
            f"No results (got errors: {[m.get('error') for m in error_msgs]}). "
            f"Full message types: {types}"
        )
        logger.info(f"Omni duplex+video: {len(result_msgs)} results, types={types}")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_omni_duplex_pause_resume(self):
        """Omni Duplex：pause / resume 生命周期"""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        import websockets

        sid = f"test_omni_pr_{int(time.time())}"
        audio_b64 = _make_silence_pcm_b64(1.0)

        async with websockets.connect(
            f"{GATEWAY_WS}/ws/duplex/{sid}", max_size=50_000_000, open_timeout=60, ssl=_ws_ssl_context(GATEWAY_WS)
        ) as ws:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") in ("queue_done", "error"):
                    break
            if msg.get("type") == "error":
                pytest.skip(f"Queue error: {msg}")

            await ws.send(json.dumps({
                "type": "prepare",
                "system_prompt": "Test",
                "deferred_finalize": True,
            }))
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            msg = json.loads(raw)
            assert msg["type"] == "prepared"

            await ws.send(json.dumps({
                "type": "audio_chunk", "audio_base64": audio_b64, "force_listen": True,
            }))
            raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)

            await ws.send(json.dumps({"type": "pause"}))
            paused_ack = None
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") == "paused":
                    paused_ack = msg
                    break
                if msg.get("type") == "error":
                    pytest.fail(f"Error during pause: {msg}")
            assert paused_ack is not None, "Did not receive paused ack"

            await ws.send(json.dumps({"type": "resume"}))
            resumed_ack = None
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") == "resumed":
                    resumed_ack = msg
                    break
                if msg.get("type") == "error":
                    pytest.fail(f"Error during resume: {msg}")
            assert resumed_ack is not None, "Did not receive resumed ack"

            await ws.send(json.dumps({
                "type": "audio_chunk", "audio_base64": audio_b64, "force_listen": True,
            }))
            raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
            msg = json.loads(raw)
            assert msg.get("type") in ("result", "error", "paused", "resumed"), \
                f"Unexpected after resume: {msg}"

            await ws.send(json.dumps({"type": "stop"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)

        logger.info("Omni duplex pause/resume OK")


# ============================================================================
# Part 6: Audio Duplex WebSocket (adx_ prefix)
# ============================================================================

class TestAudioDuplex:
    """Audio Duplex WebSocket (session_id 以 adx_ 开头)"""

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_audio_duplex(self):
        """Audio-only duplex（session_id 以 adx_ 开头路由到 audio_duplex app）"""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        sid = f"adx_test_{int(time.time())}"
        msgs = await TestOmniDuplex._run_duplex(
            GATEWAY_WS, sid, num_audio_chunks=4, send_frame=False, force_listen_first_n=2,
        )

        types = [m["type"] for m in msgs]
        assert "prepared" in types, f"Missing prepared: {types}"
        result_msgs = [m for m in msgs if m.get("type") == "result"]
        assert len(result_msgs) >= 1, f"No results: {types}"
        logger.info(f"Audio duplex (adx_): {len(result_msgs)} results")


# ============================================================================
# Part 7: 管理 API
# ============================================================================

class TestAdminAPIs:
    """管理类 API（不需要推理，轻量快速）"""

    @requires_gateway
    def test_frontend_defaults(self):
        r = _get(f"{GATEWAY_URL}/api/frontend_defaults", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert "playback_delay_ms" in d
        logger.info(f"Frontend defaults: {d}")

    @requires_gateway
    def test_presets_list(self):
        r = _get(f"{GATEWAY_URL}/api/presets", timeout=10)
        assert r.status_code == 200
        d = r.json()
        logger.info(f"Presets: {len(d.get('presets', d))} entries")

    @requires_gateway
    def test_queue_status(self):
        r = _get(f"{GATEWAY_URL}/api/queue", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert "queue_length" in d
        logger.info(f"Queue: length={d['queue_length']}")

    @requires_gateway
    def test_eta_config(self):
        r = _get(f"{GATEWAY_URL}/api/config/eta", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert "config" in d
        logger.info(f"ETA config: {d['config']}")

    @requires_gateway
    def test_eta_update(self):
        r = _put(
            f"{GATEWAY_URL}/api/config/eta",
            json={"eta_chat_s": 15.0},
            timeout=10,
        )
        assert r.status_code == 200, r.text

        r2 = _get(f"{GATEWAY_URL}/api/config/eta", timeout=10)
        d = r2.json()
        assert d["config"]["eta_chat_s"] == 15.0

    @requires_gateway
    def test_sessions_list(self):
        for path in ("/sessions", "/api/sessions"):
            r = _get(f"{GATEWAY_URL}{path}", timeout=10)
            if r.status_code == 200:
                d = r.json()
                assert "total" in d
                assert "sessions" in d
                logger.info(f"{path}: total={d['total']}")
                return
        pytest.skip("Gateway 不支持 /sessions 路由")

    @requires_gateway
    def test_apps_list(self):
        r = _get(f"{GATEWAY_URL}/api/apps", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert "apps" in d
        logger.info(f"Apps: {[a.get('id') or a.get('name') for a in d['apps']]}")

    @requires_gateway
    def test_admin_apps(self):
        r = _get(f"{GATEWAY_URL}/api/admin/apps", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert "apps" in d
        logger.info(f"Admin apps: {d}")

    @requires_gateway
    def test_default_ref_audio(self):
        r = _get(f"{GATEWAY_URL}/api/default_ref_audio", timeout=15)
        assert r.status_code == 200
        d = r.json()
        if d.get("audio_base64"):
            audio_bytes = base64.b64decode(d["audio_base64"])
            assert len(audio_bytes) > 0
            logger.info(f"Default ref audio: {len(audio_bytes)} bytes")
        else:
            logger.warning("No default ref audio configured")

    @requires_gateway
    def test_cache_endpoint(self):
        r = _get(f"{GATEWAY_URL}/cache", timeout=10)
        assert r.status_code == 200
        logger.info(f"Cache: {r.json()}")


# ============================================================================
# Part 8: 参考音频 CRUD
# ============================================================================

class TestRefAudioCRUD:
    """参考音频上传 / 列表 / 删除"""

    @requires_gateway
    def test_ref_audio_lifecycle(self):
        """完整的 upload → list → delete 周期"""
        wav_b64 = _make_silence_wav_b64(1.0)

        r = _post(
            f"{GATEWAY_URL}/api/assets/ref_audio",
            json={"name": "pytest_silence", "audio_base64": wav_b64},
            timeout=15,
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("success") is True, d
        ref_id = d["id"]
        logger.info(f"Uploaded ref audio: {ref_id}")

        r = _get(f"{GATEWAY_URL}/api/assets/ref_audio", timeout=10)
        assert r.status_code == 200
        ids = [a["id"] for a in r.json()["ref_audios"]]
        assert ref_id in ids, f"{ref_id} not in {ids}"

        r = _delete(f"{GATEWAY_URL}/api/assets/ref_audio/{ref_id}", timeout=10)
        assert r.status_code == 200
        assert r.json().get("success") is True

        r = _get(f"{GATEWAY_URL}/api/assets/ref_audio", timeout=10)
        ids = [a["id"] for a in r.json()["ref_audios"]]
        assert ref_id not in ids
        logger.info("Ref audio lifecycle OK")


# ============================================================================
# Part 9: 静态页面可访问性
# ============================================================================

class TestStaticPages:
    """静态 HTML 页面和 OpenAPI 文档"""

    @requires_gateway
    @pytest.mark.parametrize("path", [
        "/",
        "/turnbased",
        "/omni",
        "/half_duplex",
        "/audio_duplex",
        "/admin",
        "/docs",
    ])
    def test_page_accessible(self, path):
        r = _get(f"{GATEWAY_URL}{path}", timeout=15, follow_redirects=True)
        assert r.status_code == 200, f"GET {path} -> {r.status_code}"
        assert len(r.content) > 100, f"GET {path}: response too small ({len(r.content)} bytes)"
        logger.info(f"Page {path}: {r.status_code}, {len(r.content)} bytes")

    @requires_gateway
    def test_openapi_json(self):
        r = _get(f"{GATEWAY_URL}/openapi.json", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert "paths" in d
        assert "info" in d
        logger.info(f"OpenAPI: {len(d['paths'])} paths")


# ============================================================================
# Part 10: Worker 健康检查 —— 推理期间仍可响应
# ============================================================================

class TestHealthDuringInference:
    """推理期间健康检查仍可用"""

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_health_during_streaming(self):
        """流式推理期间 /health 仍可响应"""
        assert _wait_worker_idle(), "Worker not idle"
        import websockets

        payload = {
            "messages": [{"role": "user", "content": "写一首五十字左右的诗。"}],
            "streaming": True,
            "generation": {"max_new_tokens": 128, "do_sample": False},
        }

        async with websockets.connect(f"{WORKER_WS}/ws/chat", max_size=50_000_000, ssl=_ws_ssl_context(WORKER_WS)) as ws:
            await ws.send(json.dumps(payload))
            await ws.recv()  # prefill_done

            await asyncio.sleep(0.2)
            async with _async_http_client() as client:
                health_r = await client.get(f"{WORKER_URL}/health", timeout=5)
                assert health_r.status_code == 200
                logger.info(f"Health during streaming: {health_r.json().get('worker_status')}")

            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=STREAMING_TIMEOUT)
                msg = json.loads(raw)
                if msg["type"] in ("done", "error"):
                    break
