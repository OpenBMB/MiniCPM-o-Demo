"""针对已部署 8020 服务的完整功能测试

运行方式：
  # 快速测试（不含推理）
  PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_full_service.py -v -s -m "not slow"

  # 全部测试（含推理，需 GPU 后端在线）
  PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_full_service.py -v -s

  # 只跑某一类
  PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_full_service.py -v -s -k "TestChat"

环境变量：
  GATEWAY_URL   默认读 config.json 的 gateway_port（https://localhost:8020）
  WORKER_URL    默认读 config.json 的 worker_base_port（http://localhost:22400）
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pytest

logger = logging.getLogger("test_full_service")
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).parent.parent

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

VERIFY_SSL = False

REF_AUDIO_WAV = PROJECT_ROOT / "assets" / "ref_audio" / "ref_minicpm_signature.wav"

CHAT_TIMEOUT = 120
STREAMING_TIMEOUT = 120
DUPLEX_TIMEOUT = 60

# ---------------------------------------------------------------------------
# HTTP helpers (skip SSL verification for self-signed certs)
# ---------------------------------------------------------------------------

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


def _async_http_client(**kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=VERIFY_SSL, **kwargs)


# ---------------------------------------------------------------------------
# Service availability checks
# ---------------------------------------------------------------------------

def _gateway_ok() -> bool:
    try:
        r = _get(f"{GATEWAY_URL}/health", timeout=5)
        return r.status_code == 200
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


# ---------------------------------------------------------------------------
# Audio / image helpers
# ---------------------------------------------------------------------------

def _make_silence_f32_b64(duration_s: float = 1.0) -> str:
    n_samples = int(16000 * duration_s)
    pcm = np.zeros(n_samples, dtype=np.float32)
    return base64.b64encode(pcm.tobytes()).decode()


def _load_user_audio_f32_b64(duration_s: float = 2.0) -> str:
    """Generate a speech-like sine-wave signal as float32 base64.

    A 440 Hz tone at amplitude 0.3 is enough to trigger Silero VAD.
    Falls back to loading REF_AUDIO_WAV if available.
    """
    if REF_AUDIO_WAV.exists():
        try:
            import soundfile as sf
            audio, sr = sf.read(str(REF_AUDIO_WAV), dtype="float32")
            target_len = int(sr * duration_s)
            if len(audio) > target_len:
                audio = audio[:target_len]
            elif len(audio) < target_len:
                audio = np.tile(audio, (target_len // len(audio)) + 1)[:target_len]
            return base64.b64encode(audio.astype(np.float32).tobytes()).decode()
        except Exception:
            pass
    n_samples = int(16000 * duration_s)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return base64.b64encode(tone.tobytes()).decode()


def _load_ref_audio_f32_b64() -> Optional[str]:
    if not REF_AUDIO_WAV.exists():
        return None
    try:
        import soundfile as sf
        audio, sr = sf.read(str(REF_AUDIO_WAV), dtype="float32")
        return base64.b64encode(audio.tobytes()).decode()
    except Exception:
        return None


def _make_test_image_b64() -> str:
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _ws_ssl_context(ws_url: str = ""):
    target = ws_url or GATEWAY_WS
    if target.startswith("wss://"):
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None


# ============================================================================
# Part 1: Health & Status
# ============================================================================

class TestHealthStatus:
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
        assert "total_workers" in d
        assert d["gateway_healthy"] is True
        logger.info(f"Gateway status: {d['total_workers']} workers, {d['idle_workers']} idle")

    @requires_gateway
    def test_workers_list(self):
        r = _get(f"{GATEWAY_URL}/workers")
        assert r.status_code == 200
        d = r.json()
        assert "workers" in d
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
    @requires_gateway
    @requires_worker
    @slow
    def test_simple_text_chat(self):
        assert _wait_worker_idle(), "Worker not idle"
        r = _post(
            f"{GATEWAY_URL}/api/chat",
            json={
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "generation": {"max_new_tokens": 32, "do_sample": False},
            },
            timeout=CHAT_TIMEOUT,
        )
        assert r.status_code == 200, f"Chat failed: {r.text}"
        d = r.json()
        assert "text" in d and len(d["text"]) > 0
        logger.info(f"Chat response: {d['text'][:80]}")

    @requires_gateway
    @requires_worker
    @slow
    def test_multi_turn_chat(self):
        assert _wait_worker_idle(), "Worker not idle"
        r = _post(
            f"{GATEWAY_URL}/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "generation": {"max_new_tokens": 32, "do_sample": False},
            },
            timeout=CHAT_TIMEOUT,
        )
        assert r.status_code == 200
        d = r.json()
        assert "text" in d and len(d["text"]) > 0, f"Empty response: {d}"
        logger.info(f"Multi-turn: {d['text'][:80]}")

    @requires_gateway
    @requires_worker
    @slow
    def test_chat_with_tts(self):
        assert _wait_worker_idle(), "Worker not idle"
        ref_b64 = _load_ref_audio_f32_b64()
        tts_cfg: Dict[str, Any] = {"enabled": True}
        if ref_b64:
            tts_cfg["ref_audio_data"] = ref_b64
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "Count 1 to 3."}],
            "generation": {"max_new_tokens": 32, "do_sample": False},
            "tts": tts_cfg,
        }
        r = _post(f"{GATEWAY_URL}/api/chat", json=payload, timeout=CHAT_TIMEOUT)
        assert r.status_code == 200
        d = r.json()
        assert "text" in d and len(d["text"]) > 0
        has_audio = d.get("audio_base64") or d.get("audio_data") or d.get("audio_duration_s", 0) > 0
        logger.info(f"TTS chat: text={d['text'][:60]}, has_audio={has_audio}")

    @requires_worker
    @slow
    def test_worker_direct_chat(self):
        assert _wait_worker_idle(), "Worker not idle"
        r = _post(
            f"{WORKER_URL}/chat",
            json={
                "messages": [{"role": "user", "content": "2+3=?"}],
                "generation": {"max_new_tokens": 16, "do_sample": False},
            },
            timeout=CHAT_TIMEOUT,
        )
        assert r.status_code == 200
        d = r.json()
        assert "text" in d
        logger.info(f"Direct worker chat: {d['text'][:60]}")


# ============================================================================
# Part 3: Chat WebSocket Streaming
# ============================================================================

class TestChatStreaming:
    @staticmethod
    async def _run_streaming(ws_url: str, payload: dict) -> dict:
        import websockets
        chunks = []
        done_msg = None
        async with websockets.connect(
            f"{ws_url}/ws/chat", max_size=50_000_000, open_timeout=30,
            ssl=_ws_ssl_context(ws_url),
        ) as ws:
            await ws.send(json.dumps(payload))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=STREAMING_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") == "done":
                    done_msg = msg
                    break
                if msg.get("type") == "prefill_done":
                    continue
                if msg.get("type") == "heartbeat":
                    continue
                chunks.append(msg)
        return {"chunks": chunks, "done": done_msg}

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_text_only(self):
        assert _wait_worker_idle(), "Worker not idle"
        result = await self._run_streaming(GATEWAY_WS, {
            "messages": [{"role": "user", "content": "Say hi in one sentence."}],
            "generation": {"max_new_tokens": 32, "do_sample": False},
        })
        assert result["done"] is not None
        full = "".join(c.get("text_delta", "") or c.get("text", "") for c in result["chunks"])
        done_text = result["done"].get("text", "")
        combined = full or done_text
        assert len(combined) > 0, "No text in streaming response"
        logger.info(f"Streaming text: '{combined[:80]}' ({len(result['chunks'])} chunks)")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_with_tts(self):
        assert _wait_worker_idle(), "Worker not idle"
        ref_b64 = _load_ref_audio_f32_b64()
        tts_cfg: Dict[str, Any] = {"enabled": True}
        if ref_b64:
            tts_cfg["ref_audio_data"] = ref_b64
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "Count 1 to 3."}],
            "generation": {"max_new_tokens": 32, "do_sample": False},
            "tts": tts_cfg,
        }
        result = await self._run_streaming(GATEWAY_WS, payload)
        assert result["done"] is not None
        audio_chunks = [c for c in result["chunks"] if c.get("audio_base64") or c.get("audio_data")]
        logger.info(f"Streaming TTS: {len(result['chunks'])} chunks, {len(audio_chunks)} with audio")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_multi_turn(self):
        assert _wait_worker_idle(), "Worker not idle"
        result = await self._run_streaming(GATEWAY_WS, {
            "messages": [
                {"role": "user", "content": "My name is Bob."},
                {"role": "assistant", "content": "Hi Bob!"},
                {"role": "user", "content": "What's my name?"},
            ],
            "generation": {"max_new_tokens": 32, "do_sample": False},
        })
        assert result["done"] is not None
        full = "".join(c.get("text_delta", "") or c.get("text", "") for c in result["chunks"])
        done_text = result["done"].get("text", "")
        combined = full or done_text
        assert len(combined) > 0, "No text in multi-turn streaming"
        logger.info(f"Streaming multi-turn: '{combined[:80]}'")

    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_direct_worker(self):
        assert _wait_worker_idle(), "Worker not idle"
        result = await self._run_streaming(WORKER_WS, {
            "messages": [{"role": "user", "content": "1+1=?"}],
            "generation": {"max_new_tokens": 16, "do_sample": False},
        })
        assert result["done"] is not None
        full = "".join(c.get("text_delta", "") or c.get("text", "") for c in result["chunks"])
        done_text = result["done"].get("text", "")
        combined = full or done_text
        assert len(combined) > 0, "No text from direct worker streaming"
        logger.info(f"Direct worker streaming: '{combined[:60]}'")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_streaming_stream_vs_done_consistency(self):
        assert _wait_worker_idle(), "Worker not idle"
        result = await self._run_streaming(GATEWAY_WS, {
            "messages": [{"role": "user", "content": "Say hello."}],
            "generation": {"max_new_tokens": 32, "do_sample": False},
        })
        streamed = "".join(c.get("text_delta", "") or c.get("text", "") for c in result["chunks"])
        done_text = result["done"].get("text", "")
        if streamed and done_text:
            assert streamed == done_text, f"Mismatch: streamed={streamed!r} vs done={done_text!r}"
        assert len(streamed) > 0 or len(done_text) > 0, "No text from streaming at all"
        logger.info(f"Stream vs done consistency OK (streamed={len(streamed)}, done={len(done_text)})")


# ============================================================================
# Part 4: Half-Duplex WebSocket
# ============================================================================

class TestHalfDuplex:
    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_half_duplex_with_real_audio(self):
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        import websockets

        sid = f"test_hd_{int(time.time())}"
        audio_b64 = _load_user_audio_f32_b64(2.0)

        async with websockets.connect(
            f"{GATEWAY_WS}/ws/half_duplex/{sid}", max_size=50_000_000, open_timeout=60,
            ssl=_ws_ssl_context(GATEWAY_WS),
        ) as ws:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") in ("queue_done", "error"):
                    break
            if msg.get("type") == "error":
                pytest.skip(f"Queue error: {msg}")

            prepare_msg: Dict[str, Any] = {
                "type": "prepare",
                "system_content": [{"type": "text", "text": "You are a helpful assistant."}],
                "config": {
                    "generation": {"max_new_tokens": 64},
                    "tts": {"enabled": True},
                },
            }
            await ws.send(json.dumps(prepare_msg))

            for _ in range(20):
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "audio_base64": audio_b64,
                }))
                await asyncio.sleep(0.1)

            msgs = []
            deadline = time.time() + STREAMING_TIMEOUT
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    m = json.loads(raw)
                    msgs.append(m)
                    if m.get("type") in ("turn_done", "timeout"):
                        break
                except asyncio.TimeoutError:
                    break

            msg_types = [m.get("type") for m in msgs]
            logger.info(f"Half-duplex: message types={msg_types}")
            assert len(msgs) >= 1, f"No messages received from half-duplex"

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_half_duplex_prepare_only(self):
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        import websockets

        sid = f"test_hd_prep_{int(time.time())}"
        async with websockets.connect(
            f"{GATEWAY_WS}/ws/half_duplex/{sid}", max_size=50_000_000, open_timeout=60,
            ssl=_ws_ssl_context(GATEWAY_WS),
        ) as ws:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") in ("queue_done", "error"):
                    break
            assert msg.get("type") == "queue_done"
        logger.info("Half-duplex prepare-only OK")


# ============================================================================
# Part 5: Omni Duplex WebSocket
# ============================================================================

class TestOmniDuplex:
    @staticmethod
    async def _run_duplex(
        ws_url: str,
        session_id: str,
        num_audio_chunks: int = 5,
        *,
        send_frame: bool = False,
        force_listen_first_n: int = 3,
        max_prepare_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        import websockets

        audio_b64 = _make_silence_f32_b64(1.0)
        frame_b64 = _make_test_image_b64() if send_frame else None
        ws_msgs: List[Dict[str, Any]] = []

        for attempt in range(max_prepare_retries):
            ws_msgs = []
            need_retry = False
            try:
                actual_sid = session_id if attempt == 0 else f"{session_id}_r{attempt}"
                async with websockets.connect(
                    f"{ws_url}/ws/duplex/{actual_sid}",
                    max_size=50_000_000,
                    open_timeout=60,
                    ssl=_ws_ssl_context(ws_url),
                ) as ws:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                        msg = json.loads(raw)
                        ws_msgs.append(msg)
                        if msg.get("type") in ("queue_done", "error"):
                            break

                    if ws_msgs[-1].get("type") == "error":
                        if attempt < max_prepare_retries - 1:
                            logger.warning(f"Queue error on attempt {attempt}, retrying: {ws_msgs[-1]}")
                            need_retry = True
                        else:
                            return ws_msgs

                    if need_retry:
                        await asyncio.sleep(3.0)
                        continue

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
                        if attempt < max_prepare_retries - 1:
                            logger.warning(f"Prepare failed on attempt {attempt}, retrying: {msg}")
                            need_retry = True
                        else:
                            return ws_msgs

                    if need_retry:
                        await asyncio.sleep(5.0)
                        continue

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

                break  # success
            except Exception as e:
                ws_msgs.append({"type": "error", "error": f"Exception: {e}"})
                if attempt < max_prepare_retries - 1:
                    await asyncio.sleep(3.0)
                    continue

        return ws_msgs

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_omni_duplex_audio_only(self):
        """Omni Duplex: audio only"""
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
        """Omni Duplex: audio + video frames"""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        sid = f"test_omni_video_{int(time.time())}"
        msgs = await self._run_duplex(GATEWAY_WS, sid, num_audio_chunks=3, send_frame=True)

        types = [m["type"] for m in msgs]
        assert "prepared" in types, f"Missing prepared: {types}"
        error_msgs = [m for m in msgs if m.get("type") == "error"]
        result_msgs = [m for m in msgs if m.get("type") == "result"]
        if error_msgs:
            error_details = [m.get("error", str(m)) for m in error_msgs]
            logger.warning(f"Omni duplex+video errors: {error_details}")
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
        """Omni Duplex: pause / resume lifecycle"""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        import websockets

        sid = f"test_omni_pr_{int(time.time())}"
        audio_b64 = _make_silence_f32_b64(1.0)

        async with websockets.connect(
            f"{GATEWAY_WS}/ws/duplex/{sid}", max_size=50_000_000, open_timeout=60,
            ssl=_ws_ssl_context(GATEWAY_WS),
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
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") == "paused":
                    break
                if msg.get("type") == "error":
                    pytest.fail(f"Error during pause: {msg}")

            await ws.send(json.dumps({"type": "resume"}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") == "resumed":
                    break
                if msg.get("type") == "error":
                    pytest.fail(f"Error during resume: {msg}")

            await ws.send(json.dumps({
                "type": "audio_chunk", "audio_base64": audio_b64, "force_listen": True,
            }))
            raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
            msg = json.loads(raw)
            assert msg.get("type") in ("result", "error", "paused", "resumed"), \
                f"Unexpected after resume: {msg}"

            await ws.send(json.dumps({"type": "stop"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=30)

        logger.info("Omni duplex pause/resume OK")


# ============================================================================
# Part 6: Audio Duplex WebSocket (adx_ prefix)
# ============================================================================

class TestAudioDuplex:
    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_audio_duplex(self):
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        import websockets

        sid = f"adx_test_{int(time.time())}"
        audio_b64 = _make_silence_f32_b64(1.0)

        async with websockets.connect(
            f"{GATEWAY_WS}/ws/duplex/{sid}", max_size=50_000_000, open_timeout=60,
            ssl=_ws_ssl_context(GATEWAY_WS),
        ) as ws:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") in ("queue_done", "error"):
                    break
            if msg.get("type") == "error":
                pytest.skip(f"Queue error: {msg}")

            prepare_msg: Dict[str, Any] = {
                "type": "prepare",
                "system_prompt": "Audio duplex test",
                "deferred_finalize": True,
            }
            ref_b64 = _load_ref_audio_f32_b64()
            if ref_b64:
                prepare_msg["tts_ref_audio_base64"] = ref_b64

            await ws.send(json.dumps(prepare_msg))
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            msg = json.loads(raw)
            assert msg["type"] == "prepared", f"Expected prepared, got: {msg}"

            results = []
            for i in range(3):
                await ws.send(json.dumps({
                    "type": "audio_chunk", "audio_base64": audio_b64, "force_listen": True,
                }))
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                m = json.loads(raw)
                results.append(m)
                await asyncio.sleep(0.05)

            await ws.send(json.dumps({"type": "stop"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=30)

        result_types = [m.get("type") for m in results]
        logger.info(f"Audio duplex: {result_types}")


# ============================================================================
# Part 7: Admin APIs
# ============================================================================

class TestAdminAPIs:
    @requires_gateway
    def test_frontend_defaults(self):
        r = _get(f"{GATEWAY_URL}/api/frontend_defaults")
        assert r.status_code == 200
        d = r.json()
        logger.info(f"Frontend defaults keys: {list(d.keys())}")

    @requires_gateway
    def test_presets_list(self):
        r = _get(f"{GATEWAY_URL}/api/presets")
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d, (list, dict))
        logger.info(f"Presets: {len(d)} entries")

    @requires_gateway
    def test_queue_status(self):
        r = _get(f"{GATEWAY_URL}/api/queue")
        assert r.status_code == 200
        logger.info(f"Queue: {r.json()}")

    @requires_gateway
    def test_eta_config(self):
        r = _get(f"{GATEWAY_URL}/api/config/eta")
        assert r.status_code == 200
        logger.info(f"ETA config: {r.json()}")

    @requires_gateway
    def test_eta_update(self):
        r = _put(f"{GATEWAY_URL}/api/config/eta", json={})
        assert r.status_code in (200, 422)

    @requires_gateway
    def test_sessions_list(self):
        r = _get(f"{GATEWAY_URL}/api/sessions")
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d, (list, dict))
        logger.info(f"Sessions: {d if isinstance(d, dict) else len(d)}")

    @requires_gateway
    def test_apps_list(self):
        r = _get(f"{GATEWAY_URL}/api/apps")
        assert r.status_code == 200

    @requires_gateway
    def test_admin_apps(self):
        r = _get(f"{GATEWAY_URL}/api/admin/apps")
        assert r.status_code == 200

    @requires_gateway
    def test_default_ref_audio(self):
        r = _get(f"{GATEWAY_URL}/api/default_ref_audio")
        assert r.status_code in (200, 404)

    @requires_gateway
    def test_cache_endpoint(self):
        r = _get(f"{GATEWAY_URL}/cache")
        assert r.status_code in (200, 404, 405)


# ============================================================================
# Part 8: Reference Audio CRUD
# ============================================================================

class TestRefAudioCRUD:
    @requires_gateway
    def test_ref_audio_lifecycle(self):
        r = _get(f"{GATEWAY_URL}/api/assets/ref_audio")
        assert r.status_code == 200
        initial = r.json()
        logger.info(f"Initial ref_audios: {len(initial) if isinstance(initial, list) else initial}")

        if REF_AUDIO_WAV.exists():
            with open(REF_AUDIO_WAV, "rb") as f:
                upload_r = _post(
                    f"{GATEWAY_URL}/api/assets/ref_audio",
                    files={"file": ("test_upload.wav", f, "audio/wav")},
                    timeout=30,
                )
            if upload_r.status_code in (200, 201):
                uploaded = upload_r.json()
                ref_id = uploaded.get("id") or uploaded.get("name", "test_upload")
                logger.info(f"Uploaded ref audio: {ref_id}")

                del_r = _delete(f"{GATEWAY_URL}/api/assets/ref_audio/{ref_id}")
                logger.info(f"Delete ref audio: {del_r.status_code}")


# ============================================================================
# Part 9: Static Pages
# ============================================================================

class TestStaticPages:
    @requires_gateway
    @pytest.mark.parametrize("path", [
        "/", "/turnbased", "/omni", "/half_duplex", "/audio_duplex", "/admin", "/docs",
    ])
    def test_page_accessible(self, path):
        r = _get(f"{GATEWAY_URL}{path}", follow_redirects=True)
        assert r.status_code == 200, f"{path} returned {r.status_code}"

    @requires_gateway
    def test_openapi_json(self):
        r = _get(f"{GATEWAY_URL}/openapi.json")
        assert r.status_code == 200
        d = r.json()
        assert "paths" in d


# ============================================================================
# Part 10: Health During Inference
# ============================================================================

class TestHealthDuringInference:
    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_health_during_streaming(self):
        assert _wait_worker_idle(), "Worker not idle"
        import websockets

        async with websockets.connect(
            f"{GATEWAY_WS}/ws/chat", max_size=50_000_000, open_timeout=30,
            ssl=_ws_ssl_context(GATEWAY_WS),
        ) as ws:
            payload = {
                "messages": [{"role": "user", "content": "Count from 1 to 10 slowly."}],
                "generation": {"max_new_tokens": 64, "do_sample": False},
            }
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
                if msg.get("type") == "done":
                    break

        logger.info("Health during inference OK")


# ============================================================================
# Part 11: Half-Duplex Deep Tests (VAD trigger + multi-turn)
# ============================================================================

class TestHalfDuplexDeep:
    """Deeper half-duplex tests that verify VAD triggering and multi-turn stability."""

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_hd_generates_audio(self):
        """Half-duplex should produce text and/or audio when given real speech."""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        import websockets

        sid = f"test_hd_deep_{int(time.time())}"
        audio_b64 = _load_user_audio_f32_b64(2.0)

        async with websockets.connect(
            f"{GATEWAY_WS}/ws/half_duplex/{sid}", max_size=50_000_000, open_timeout=60,
            ssl=_ws_ssl_context(GATEWAY_WS),
        ) as ws:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                msg = json.loads(raw)
                if msg.get("type") in ("queue_done", "error"):
                    break
            if msg.get("type") == "error":
                pytest.skip(f"Queue error: {msg}")

            prepare_msg: Dict[str, Any] = {
                "type": "prepare",
                "system_content": [{"type": "text", "text": "You are a helpful assistant."}],
                "config": {
                    "generation": {"max_new_tokens": 128},
                    "tts": {"enabled": True},
                },
            }
            ref_b64 = _load_ref_audio_f32_b64()
            if ref_b64:
                prepare_msg["config"]["tts"]["ref_audio_data"] = ref_b64
            await ws.send(json.dumps(prepare_msg))

            for _ in range(30):
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "audio_base64": audio_b64,
                }))
                await asyncio.sleep(0.1)

            all_msgs: List[Dict[str, Any]] = []
            text_parts: List[str] = []
            has_audio = False
            deadline = time.time() + STREAMING_TIMEOUT
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    m = json.loads(raw)
                    all_msgs.append(m)
                    if m.get("text_delta"):
                        text_parts.append(m["text_delta"])
                    if m.get("text"):
                        text_parts.append(m["text"])
                    if m.get("audio_base64") or m.get("audio_data"):
                        has_audio = True
                    if m.get("type") in ("turn_done", "timeout"):
                        break
                except asyncio.TimeoutError:
                    break

        text = "".join(text_parts)
        msg_types = [m.get("type") for m in all_msgs]
        logger.info(f"[HD deep] msg_types={msg_types}, text='{text[:80]}', has_audio={has_audio}")
        assert len(text) > 0 or has_audio, (
            f"Half-duplex produced no output. msg_types={msg_types}"
        )

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_hd_multi_turn_not_stuck(self):
        """Two consecutive half-duplex turns should both produce output (not stuck at thinking)."""
        assert _wait_worker_idle(timeout=60), "Worker not idle"
        import websockets

        sid = f"test_hd_mt_{int(time.time())}"
        audio_b64 = _load_user_audio_f32_b64(2.0)

        async with websockets.connect(
            f"{GATEWAY_WS}/ws/half_duplex/{sid}", max_size=50_000_000, open_timeout=60,
            ssl=_ws_ssl_context(GATEWAY_WS),
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
                "system_content": [{"type": "text", "text": "You are a helpful assistant."}],
                "config": {
                    "generation": {"max_new_tokens": 64},
                    "tts": {"enabled": True},
                },
            }))

            for turn in range(2):
                logger.info(f"[HD multi-turn] Starting turn {turn + 1}")
                for _ in range(25):
                    await ws.send(json.dumps({
                        "type": "audio_chunk",
                        "audio_base64": audio_b64,
                    }))
                    await asyncio.sleep(0.1)

                got_output = False
                deadline = time.time() + STREAMING_TIMEOUT
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                        m = json.loads(raw)
                        if m.get("text_delta") or m.get("text") or m.get("audio_base64"):
                            got_output = True
                        if m.get("type") in ("turn_done", "timeout"):
                            break
                    except asyncio.TimeoutError:
                        break

                logger.info(f"[HD multi-turn] Turn {turn + 1}: got_output={got_output}")
                assert got_output, f"Turn {turn + 1} stuck: no output received"
                await asyncio.sleep(1.0)


# ============================================================================
# Part 12: Mixed Mode Tests
# ============================================================================

class TestMixedModes:
    """Test that all four modes can be used interchangeably without state residue."""

    @staticmethod
    async def _do_chat_turn(with_tts: bool = False) -> Dict[str, Any]:
        """Execute one turn-based chat and return result info."""
        _wait_worker_idle(timeout=30)
        tts_cfg: Optional[Dict[str, Any]] = None
        if with_tts:
            tts_cfg = {"enabled": True}
            ref_b64 = _load_ref_audio_f32_b64()
            if ref_b64:
                tts_cfg["ref_audio_data"] = ref_b64
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "generation": {"max_new_tokens": 32, "do_sample": False},
        }
        if tts_cfg:
            payload["tts"] = tts_cfg
        r = _post(f"{GATEWAY_URL}/api/chat", json=payload, timeout=CHAT_TIMEOUT)
        d = r.json() if r.status_code == 200 else {}
        has_audio = bool(d.get("audio_base64") or d.get("audio_data"))
        return {"ok": r.status_code == 200, "text": d.get("text", ""), "has_audio": has_audio}

    @staticmethod
    async def _do_streaming_turn() -> Dict[str, Any]:
        """Execute one streaming chat and return result info."""
        _wait_worker_idle(timeout=30)
        import websockets
        chunks: List[Dict[str, Any]] = []
        done_msg = None
        try:
            async with websockets.connect(
                f"{GATEWAY_WS}/ws/chat", max_size=50_000_000, open_timeout=30,
                ssl=_ws_ssl_context(GATEWAY_WS),
            ) as ws:
                await ws.send(json.dumps({
                    "messages": [{"role": "user", "content": "Count 1 to 3."}],
                    "generation": {"max_new_tokens": 32, "do_sample": False},
                }))
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=STREAMING_TIMEOUT)
                    msg = json.loads(raw)
                    if msg.get("type") == "done":
                        done_msg = msg
                        break
                    if msg.get("type") in ("prefill_done", "heartbeat"):
                        continue
                    chunks.append(msg)
        except Exception as e:
            return {"ok": False, "text": "", "has_audio": False, "error": str(e)}

        text = "".join(c.get("text_delta", "") or c.get("text", "") for c in chunks)
        if not text and done_msg:
            text = done_msg.get("text", "")
        return {"ok": done_msg is not None, "text": text, "has_audio": False}

    @staticmethod
    async def _do_half_duplex_turn() -> Dict[str, Any]:
        """Execute one half-duplex turn with real f32 audio and return result info."""
        _wait_worker_idle(timeout=60)
        import websockets

        sid = f"mix_hd_{int(time.time())}_{id(asyncio.get_event_loop()) % 10000}"
        audio_b64 = _load_user_audio_f32_b64(2.0)
        text_parts: List[str] = []
        has_audio = False

        try:
            async with websockets.connect(
                f"{GATEWAY_WS}/ws/half_duplex/{sid}", max_size=50_000_000, open_timeout=60,
                ssl=_ws_ssl_context(GATEWAY_WS),
            ) as ws:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=DUPLEX_TIMEOUT)
                    msg = json.loads(raw)
                    if msg.get("type") in ("queue_done", "error"):
                        break
                if msg.get("type") == "error":
                    return {"ok": False, "text": "", "has_audio": False, "error": f"queue: {msg}"}

                await ws.send(json.dumps({
                    "type": "prepare",
                    "system_content": [{"type": "text", "text": "You are a helpful assistant."}],
                    "config": {
                        "generation": {"max_new_tokens": 64},
                        "tts": {"enabled": True},
                    },
                }))

                for _ in range(25):
                    await ws.send(json.dumps({
                        "type": "audio_chunk",
                        "audio_base64": audio_b64,
                    }))
                    await asyncio.sleep(0.1)

                deadline = time.time() + STREAMING_TIMEOUT
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                        m = json.loads(raw)
                        if m.get("text_delta"):
                            text_parts.append(m["text_delta"])
                        if m.get("text"):
                            text_parts.append(m["text"])
                        if m.get("audio_base64") or m.get("audio_data"):
                            has_audio = True
                        if m.get("type") in ("turn_done", "timeout"):
                            break
                    except asyncio.TimeoutError:
                        break
        except Exception as e:
            return {"ok": False, "text": "", "has_audio": False, "error": str(e)}

        text = "".join(text_parts)
        return {"ok": len(text) > 0 or has_audio, "text": text, "has_audio": has_audio}

    @staticmethod
    async def _do_omni_duplex_turn() -> Dict[str, Any]:
        """Execute one omni-duplex turn and return result info."""
        _wait_worker_idle(timeout=60)
        sid = f"mix_omni_{int(time.time())}_{id(asyncio.get_event_loop()) % 10000}"
        msgs = await TestOmniDuplex._run_duplex(
            GATEWAY_WS, sid, num_audio_chunks=5, send_frame=False, force_listen_first_n=3,
        )
        types = [m.get("type") for m in msgs]
        result_msgs = [m for m in msgs if m.get("type") == "result"]
        has_error = any(m.get("type") == "error" for m in msgs)
        return {
            "ok": "prepared" in types and len(result_msgs) >= 1,
            "text": "",
            "has_audio": False,
            "n_results": len(result_msgs),
            "has_error": has_error,
        }

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_sequential_mode_switching(self):
        """Cycle through chat → streaming → omni_duplex → half_duplex in sequence."""
        steps = [
            ("chat+tts", self._do_chat_turn, {"with_tts": True}),
            ("streaming", self._do_streaming_turn, {}),
            ("omni_duplex", self._do_omni_duplex_turn, {}),
            ("half_duplex", self._do_half_duplex_turn, {}),
        ]
        results = []
        for name, fn, kwargs in steps:
            logger.info(f"[MixedModes] Step: {name}")
            res = await fn(**kwargs)
            logger.info(f"  [{('OK' if res['ok'] else 'FAIL')}] text='{res.get('text', '')[:40]}' audio={res.get('has_audio')}")
            results.append((name, res))
            await asyncio.sleep(2.0)

        failures = [(name, r) for name, r in results if not r["ok"]]
        if failures:
            fail_detail = "; ".join(f"{n}: {r}" for n, r in failures)
            pytest.fail(f"Mode switching failures: {fail_detail}")

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_parallel_different_modes(self):
        """Run chat and streaming concurrently (different mode, same gateway)."""
        chat_task = asyncio.create_task(self._do_chat_turn(with_tts=False))
        stream_task = asyncio.create_task(self._do_streaming_turn())
        chat_res, stream_res = await asyncio.gather(chat_task, stream_task)
        logger.info(f"[Parallel] chat={chat_res['ok']}, stream={stream_res['ok']}")
        assert chat_res["ok"], f"Parallel chat failed: {chat_res}"
        assert stream_res["ok"], f"Parallel stream failed: {stream_res}"

    @requires_gateway
    @requires_worker
    @slow
    @pytest.mark.asyncio
    async def test_duplex_chat_duplex_sandwich(self):
        """omni_duplex → chat → half_duplex: the 'sandwich' pattern."""
        steps = [
            ("omni_duplex_1", self._do_omni_duplex_turn, {}),
            ("chat", self._do_chat_turn, {"with_tts": True}),
            ("half_duplex", self._do_half_duplex_turn, {}),
        ]
        results = []
        for name, fn, kwargs in steps:
            logger.info(f"[Sandwich] Step: {name}")
            res = await fn(**kwargs)
            logger.info(f"  [{('OK' if res['ok'] else 'FAIL')}] text='{res.get('text', '')[:40]}' audio={res.get('has_audio')}")
            results.append((name, res))
            await asyncio.sleep(2.0)

        failures = [(name, r) for name, r in results if not r["ok"]]
        if failures:
            fail_detail = "; ".join(f"{n}: {r}" for n, r in failures)
            pytest.fail(f"Sandwich test failures: {fail_detail}")
