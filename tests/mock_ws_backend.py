"""Mock WebSocket Backend Server — 模拟 C++ llama-server 的新协议端点

用于测试 CppWsBackendWorker 和协议交互，无需真实 GPU/模型。

启动方式：
    PYTHONPATH=. python tests/mock_ws_backend.py --port 19060

端点：
    WS  /backend              — 新协议 WS endpoint
    HTTP GET  /health         — 健康检查
    HTTP POST /sessions/{id}/close — 会话关闭
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
import wave
import io
import struct
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] mock-ws: %(message)s",
)
logger = logging.getLogger("mock_ws_backend")

# ============ 可配参数 ============

RESPONSE_TEXT = "This is a mock response from the MiniCPM-Omni backend server."
RESPONSE_CHUNKS = 3       # text_delta 分成几个 chunk
AUDIO_DURATION_S = 1.0    # 模拟 TTS 输出时长
AUDIO_SAMPLE_RATE = 24000
AUDIO_SAMPLES_PER_DELTA = 2400  # 0.1s 每块
AUDIO_CHUNKS = 10

# ============ 会话管理 ============

_active_session_id: Optional[str] = None
_active_sessions: Dict[str, dict] = {}


def _gen_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _gen_wav_bytes(duration_s: float = 0.1, sample_rate: int = 24000) -> bytes:
    """生成模拟 float32 PCM WAV 文件（440Hz 正弦波）"""
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.3).astype(np.float32)

    buf = io.BytesIO()
    with wave.open(buf, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(sample_rate)
        pcm = (audio * 32767).astype(np.int16)
        w.writeframes(pcm.tobytes())
    return audio.tobytes()  # return float32 PCM bytes


def create_app(delay_s: float = 0.05) -> FastAPI:
    app = FastAPI(title="Mock WS Backend")

    # ============ Health ============
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ============ HTTP Close ============
    @app.post("/sessions/{session_id}/close")
    async def close_session(session_id: str):
        global _active_session_id
        if session_id in _active_sessions:
            del _active_sessions[session_id]
            if _active_session_id == session_id:
                _active_session_id = None
            logger.info(f"Session closed via HTTP: {session_id}")
            return {"ok": True, "session_id": session_id, "closed": True}
        return {"ok": False, "error": "session not found"}

    # ============ WebSocket /backend ============
    @app.websocket("/backend")
    async def ws_backend(ws: WebSocket):
        global _active_session_id
        await ws.accept()

        # --- 读取 session.init ---
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for session.init")
            await ws.close()
            return

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in init")
            await ws.close()
            return

        if msg.get("type") != "session.init":
            logger.error(f"Expected session.init, got {msg.get('type')}")
            await ws.close()
            return

        # --- 单会话互斥 ---
        if _active_session_id:
            logger.warning(f"Rejecting init: active session {_active_session_id} exists")
            await ws.send_text(json.dumps({
                "type": "session.closed",
                "session_id": "",
                "reason": "active_session_exists",
                "message": "Another session is already active",
            }))
            await ws.close()
            return

        # --- 分配 session ---
        session_id = _gen_session_id()
        _active_session_id = session_id
        _active_sessions[session_id] = {"ws": ws, "mode": msg["payload"].get("mode", "full_duplex")}

        mode = msg["payload"].get("mode", "full_duplex")
        logger.info(f"Session created: {session_id} mode={mode}")

        # 发送 session.created
        await ws.send_text(json.dumps({
            "type": "session.created",
            "session_id": session_id,
            "mode": mode,
            "metrics": {
                "backend": "mock-llama.cpp-omni",
                "kv_cache_length": 0,
                "prefill_ms": 50.0,
                "wall_clock_ms": 100.0,
            },
        }))

        # --- 消息循环 ---
        response_seq = 0

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Session {session_id} timed out")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {session_id}")
                    await ws.send_text(json.dumps({
                        "type": "session.closed",
                        "session_id": session_id,
                        "reason": "invalid_json",
                    }))
                    break

                msg_type = msg.get("type", "")

                if msg_type != "input.append":
                    logger.error(f"Unexpected type {msg_type} from {session_id}")
                    await ws.send_text(json.dumps({
                        "type": "session.closed",
                        "session_id": session_id,
                        "reason": "unexpected_message_type",
                    }))
                    break

                response_seq += 1
                response_id = f"{session_id}_resp_{response_seq}"
                input_data = msg.get("input", {})
                is_streaming = input_data.get("streaming", True)

                # --- 模拟 prefill + decode ---
                await asyncio.sleep(delay_s * 2)  # 模拟 prefill 延迟

                # 发送 listen_delta（full_duplex 模式）
                if mode == "full_duplex":
                    await ws.send_text(json.dumps({
                        "type": "response.output.delta",
                        "kind": "listen",
                        "session_id": session_id,
                        "response_id": response_id,
                        "server_send_ts": int(time.time() * 1000),
                    }))
                    continue

                # turn_based 模式：发送 text_delta + audio_delta
                if is_streaming:
                    # 文本 delta
                    words = RESPONSE_TEXT.split()
                    chunk_size = max(1, len(words) // RESPONSE_CHUNKS)
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        if i > 0:
                            chunk = " " + chunk
                        await asyncio.sleep(delay_s)
                        await ws.send_text(json.dumps({
                            "type": "response.output.delta",
                            "kind": "text",
                            "delta": chunk,
                            "session_id": session_id,
                            "response_id": response_id,
                            "server_send_ts": int(time.time() * 1000),
                        }))

                    # 音频 delta
                    if input_data.get("tts", {}).get("enabled", True):
                        for i in range(AUDIO_CHUNKS):
                            wav_bytes = _gen_wav_bytes(0.1, AUDIO_SAMPLE_RATE)
                            import base64
                            b64 = base64.b64encode(wav_bytes).decode()
                            await asyncio.sleep(delay_s * 0.5)
                            await ws.send_text(json.dumps({
                                "type": "response.output.delta",
                                "kind": "audio",
                                "delta": b64,
                                "session_id": session_id,
                                "response_id": response_id,
                                "server_send_ts": int(time.time() * 1000),
                            }))

                # 发送 response.done
                await asyncio.sleep(delay_s)
                await ws.send_text(json.dumps({
                    "type": "response.done",
                    "session_id": session_id,
                    "response_id": response_id,
                    "full_text": RESPONSE_TEXT,
                    "reason": "turn_end",
                    "server_send_ts": int(time.time() * 1000),
                    "metrics": {
                        "kv_cache_length": 128,
                        "generate_ms": 500.0,
                        "wall_clock_ms": 800.0,
                        "n_tokens": len(RESPONSE_TEXT.split()),
                    },
                }))

        except WebSocketDisconnect:
            logger.info(f"Session {session_id} disconnected")

        # --- 清理（立即清除 active 状态，避免竞态）---
        if _active_session_id == session_id:
            _active_session_id = None
        if session_id in _active_sessions:
            del _active_sessions[session_id]

        # 尝试发送 session.closed（best-effort）
        try:
            await ws.send_text(json.dumps({
                "type": "session.closed",
                "session_id": session_id,
                "reason": "client_disconnected",
            }))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

    return app


def main():
    parser = argparse.ArgumentParser(description="Mock WS Backend Server")
    parser.add_argument("--port", type=int, default=19060, help="Server port (default: 19060)")
    parser.add_argument("--delay", type=float, default=0.05, help="Response delay in seconds (default: 0.05)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    app = create_app(delay_s=args.delay)
    logger.info(f"Starting mock WS backend on {args.host}:{args.port} (delay={args.delay}s)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
