#!/usr/bin/env python3
"""手动端到端测试 — 连接真实 C++ llama-server /backend 端点

用法:
    # 1. 启动 C++ server
    cd /path/to/llama.cpp-omni
    ./build-arm64-apple-clang-release/bin/llama-server -m <model.gguf> --mmproj <mmproj.gguf> --port 19060

    # 2. 运行测试
    PYTHONPATH=. python tests/test_ws_real.py [--port 19060] [--mode full_duplex|turn_based|chat|all]

    如果省略模型路径（无法连接 server），脚本自动跳过所有测试。
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

TESTS_DIR = Path(__file__).parent
USER_AUDIO_PATH = TESTS_DIR / "cases" / "common" / "user_audio" / "000_user_audio0.wav"
REF_AUDIO_PATH = TESTS_DIR / "cases" / "common" / "ref_audio" / "BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav"
IMAGE_PATH = TESTS_DIR / "cases" / "common" / "images" / "image.png"


def _load_audio_b64(path: Path) -> str:
    import soundfile as sf
    data, sr = sf.read(str(path))
    if sr != 16000:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return base64.b64encode(data.astype(np.float32).tobytes()).decode()


def _load_image_b64(path: Path) -> str:
    with open(str(path), "rb") as f:
        return base64.b64encode(f.read()).decode()


def _check_server(url: str) -> bool:
    import httpx
    try:
        resp = httpx.get(f"{url}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _ws_connect(ws_url: str):
    import websocket
    return websocket.create_connection(ws_url, timeout=30)


_ok = 0
_fail = 0


def _assert(cond, msg):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  ✅ {msg}")
    else:
        _fail += 1
        print(f"  ❌ {msg}")


def test_full_duplex(http_url: str, ws_url: str):
    """full_duplex: session.init → input.append(audio+image) → listen"""
    global _ok, _fail
    _ok = _fail = 0
    print("\n=== Full Duplex Test ===")

    ws = _ws_connect(ws_url)

    # --- session.init ---
    ref_b64 = _load_audio_b64(REF_AUDIO_PATH)
    ws.send(json.dumps({
        "type": "session.init",
        "payload": {
            "mode": "full_duplex",
            "voice": {"ref_audio": ref_b64},
            "system_prompt": "Streaming Duplex Conversation! You are a helpful assistant.",
        },
    }))

    resp = json.loads(ws.recv())
    _assert(resp["type"] == "session.created", f"session.created: session_id={resp.get('session_id', 'N/A')}")

    sid = resp.get("session_id", "")

    # --- input.append (audio only) ---
    user_b64 = _load_audio_b64(USER_AUDIO_PATH)
    ws.send(json.dumps({
        "type": "input.append",
        "input": {"audio": user_b64, "max_slice_nums": 3},
    }))

    # --- collect events ---
    events = []
    try:
        ws.settimeout(30.0)
        while True:
            raw = ws.recv()
            ev = json.loads(raw)
            events.append(ev)
            print(f"    event: {ev['type']}")
            if ev.get("kind"):
                print(f"      kind={ev['kind']}, delta={str(ev.get('delta', ''))[:60]}...")
            if ev["type"] in ("response.done", "session.closed"):
                break
    except Exception:
        pass

    _assert(len(events) > 0, f"received {len(events)} events")
    _assert(any(e.get("kind") == "listen" for e in events), "received listen event")

    # --- input.append (audio + image) ---
    ws.send(json.dumps({
        "type": "input.append",
        "input": {
            "audio": user_b64,
            "video_frames": [_load_image_b64(IMAGE_PATH)],
            "max_slice_nums": 3,
        },
    }))

    events2 = []
    try:
        while True:
            raw = ws.recv()
            ev = json.loads(raw)
            events2.append(ev)
            if ev["type"] in ("response.done", "session.closed"):
                break
    except Exception:
        pass

    _assert(len(events2) > 0, f"received {len(events2)} events (with image)")

    # --- HTTP close ---
    import httpx
    client = httpx.Client(timeout=httpx.Timeout(10.0), trust_env=False)
    r = client.post(f"{http_url}/sessions/{sid}/close")
    _assert(r.status_code == 200, f"HTTP close: {r.status_code}")
    client.close()

    ws.close()

    print(f"\n  Results: {_ok} ✅ / {_fail} ❌")


def test_turn_based_streaming(http_url: str, ws_url: str):
    """turn_based: session.init → input.append(messages, streaming=true) → deltas → done"""
    global _ok, _fail
    _ok = _fail = 0
    print("\n=== Turn-based Streaming Test ===")

    ws = _ws_connect(ws_url)

    # --- session.init (turn_based) ---
    ws.send(json.dumps({
        "type": "session.init",
        "payload": {"mode": "turn_based", "system_prompt": "You are a helpful assistant."},
    }))
    resp = json.loads(ws.recv())
    _assert(resp["type"] == "session.created", f"session.created: {resp.get('session_id', 'N/A')}")

    # --- input.append (text only) ---
    ws.send(json.dumps({
        "type": "input.append",
        "input": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
            ],
            "streaming": True,
            "generation": {"max_new_tokens": 200, "length_penalty": 1.1},
            "tts": {"enabled": True},
        },
    }))

    events = []
    try:
        ws.settimeout(60.0)
        while True:
            raw = ws.recv()
            ev = json.loads(raw)
            events.append(ev)
            if ev.get("kind") == "text":
                print(f"    text_delta: {ev.get('delta', '')[:80]}")
            elif ev.get("kind") == "audio":
                print(f"    audio_delta: len={len(ev.get('delta', ''))} chars")
            elif ev.get("kind") == "listen":
                print(f"    listen")
            if ev["type"] in ("response.done", "session.closed"):
                break
    except Exception:
        pass

    _assert(len(events) > 0, f"received {len(events)} events")
    _assert(any(e["type"] == "response.done" for e in events), "received response.done")
    _assert(any(e.get("kind") == "text" for e in events), "received text_delta")

    done = [e for e in events if e["type"] == "response.done"][0]
    print(f"    full_text: {done.get('full_text', '')[:200]}")

    ws.close()

    print(f"\n  Results: {_ok} ✅ / {_fail} ❌")


def test_chat_non_streaming(http_url: str, ws_url: str):
    """chat: turn_based, streaming=false → 聚合 response.done"""
    global _ok, _fail
    _ok = _fail = 0
    print("\n=== Chat Non-streaming Test ===")

    ws = _ws_connect(ws_url)

    ws.send(json.dumps({
        "type": "session.init",
        "payload": {"mode": "turn_based"},
    }))
    resp = json.loads(ws.recv())
    _assert(resp["type"] == "session.created", f"session.created OK")

    ws.send(json.dumps({
        "type": "input.append",
        "input": {
            "messages": [
                {"role": "user", "content": "What is 1+1? Answer briefly."},
            ],
            "streaming": False,
            "generation": {"max_new_tokens": 50, "length_penalty": 1.1},
        },
    }))

    events = []
    try:
        ws.settimeout(60.0)
        while True:
            raw = ws.recv()
            ev = json.loads(raw)
            events.append(ev)
            print(f"    event: {ev['type']}")
            if ev["type"] in ("response.done", "session.closed"):
                break
    except Exception:
        pass

    _assert(len(events) == 1 or events[-1]["type"] == "response.done",
            f"non-streaming: only response.done (got {len(events)} events)")

    done = [e for e in events if e["type"] == "response.done"]
    if done:
        print(f"    full_text: {done[0].get('full_text', '')[:200]}")
        _assert(len(done[0].get("full_text", "")) > 0, "full_text non-empty")

    ws.close()

    print(f"\n  Results: {_ok} ✅ / {_fail} ❌")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Manual E2E test against real C++ server")
    parser.add_argument("--port", type=int, default=19060, help="llama-server port")
    parser.add_argument("--mode", choices=["full_duplex", "turn_based", "chat", "all"],
                        default="all", help="Test mode")
    args = parser.parse_args()

    http_url = f"http://127.0.0.1:{args.port}"
    ws_url = f"ws://127.0.0.1:{args.port}/backend"

    if not _check_server(http_url):
        print(f"❌ Cannot connect to {http_url}/health — make sure llama-server is running")
        print(f"   cd llama.cpp-omni && ./build-arm64-apple-clang-release/bin/llama-server -m <model> --mmproj <mmproj> --port {args.port}")
        return

    print(f"✅ Server is running on port {args.port}")

    if args.mode in ("full_duplex", "all"):
        test_full_duplex(http_url, ws_url)

    if args.mode in ("turn_based", "all"):
        test_turn_based_streaming(http_url, ws_url)

    if args.mode in ("chat", "all"):
        test_chat_non_streaming(http_url, ws_url)


if __name__ == "__main__":
    main()
