"""End-to-end realtime protocol smoke test (chat + video duplex).

This is the canonical e2e check for the backend-server-protocol chain:

    frontend protocol (WS /v1/realtime)
      -> gateway
      -> worker (RemoteBackendWorker, pure forward)
      -> BackendRuntimeSession (init / push / pull / unary)
      -> py_backend/server.py
      -> PyTorchBackend -> UnifiedProcessor -> inference

It verifies the protocol chain, NOT model quality.

Prereqs — start the three services first (worker points at the backend server
so it goes through the new chain; --backend was removed, pytorch is the only
backend, and the module is py_backend.server):

    cd /user/weihongliang/MiniCPM-o-Demo-wt-backend-server-protocol-2026-05-26

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/base/bin/python -m py_backend.server \
        --host 127.0.0.1 --port 22500 --gpu-id 0 \
        --model-path /user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/base/bin/python worker.py \
        --host 127.0.0.1 --port 22400 --gpu-id 0 \
        --backend-server-url http://127.0.0.1:22500

    PYTHONPATH=. .venv/base/bin/python gateway.py \
        --host 0.0.0.0 --port 8006 --http --workers localhost:22400

Health:
    curl http://127.0.0.1:22500/health
    curl http://127.0.0.1:22400/health
    curl http://127.0.0.1:8006/health

Run:
    PYTHONPATH=. .venv/base/bin/python tests/e2e_realtime.py            # chat + video
    PYTHONPATH=. .venv/base/bin/python tests/e2e_realtime.py chat       # chat only
    PYTHONPATH=. .venv/base/bin/python tests/e2e_realtime.py chat-stream # streaming chat
    PYTHONPATH=. .venv/base/bin/python tests/e2e_realtime.py video      # video duplex only

Sample video (downloaded once to tmp/realtime_sample.mp4):
    https://pub-4c730b83fc564d6a85ec9be6da99f10c.r2.dev/minicpmo/realtime-api/examples/VID_20260511_174245.mp4
"""

import asyncio
import base64
import json
import subprocess
import sys

import websockets

GATEWAY = "ws://127.0.0.1:8006/v1/realtime"
MAX_WS = 128 * 1024 * 1024
SAMPLE = "tmp/realtime_sample.mp4"


# ----------------------------------------------------------------------------- helpers
async def _handshake(ws, init_payload):
    """Wait for queue, send session.init, wait for session.created. Returns SID."""
    while True:
        msg = json.loads(await ws.recv())
        t = msg.get("type")
        if t in ("session.queue_done", "queue_done"):
            await ws.send(json.dumps({"type": "session.init", "payload": init_payload}))
        elif t == "session.created":
            return msg.get("session_id")
        elif t == "error":
            raise RuntimeError(f"error during handshake: {msg}")


def _audio_b64(seconds=1.5):
    out = subprocess.run(
        ["ffmpeg", "-i", SAMPLE, "-t", str(seconds), "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
        capture_output=True,
    ).stdout
    return base64.b64encode(out).decode()


def _frames_b64(n=1, fps=2):
    """Extract n jpeg frames as base64, matching the page's `video_frames` field."""
    out = subprocess.run(
        ["ffmpeg", "-i", SAMPLE, "-vf", f"fps={fps}", "-vframes", str(n),
         "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True,
    ).stdout
    # split the mjpeg stream on JPEG SOI/EOI markers
    frames, start = [], 0
    while True:
        soi = out.find(b"\xff\xd8", start)
        if soi < 0:
            break
        eoi = out.find(b"\xff\xd9", soi)
        if eoi < 0:
            break
        frames.append(base64.b64encode(out[soi:eoi + 2]).decode())
        start = eoi + 2
    return frames


# ----------------------------------------------------------------------------- chat
async def run_chat(streaming=False):
    print(f"\n=== CHAT (streaming={streaming}) ===")
    async with websockets.connect(f"{GATEWAY}?mode=chat", max_size=MAX_WS) as ws:
        sid = await _handshake(ws, {})
        print("SID =", sid)
        body = {
            "messages": [{"role": "user", "content": "请只回答：测试"}],
            "streaming": streaming,
            "generation": {"max_new_tokens": 32, "length_penalty": 1.1},
            "image": {"max_slice_nums": 1},
            "omni_mode": False,
            "tts": {"enabled": False},
            "use_tts_template": False,
        }
        await ws.send(json.dumps({"type": "input.append", "input": body}, ensure_ascii=False))
        text = ""
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            t = msg.get("type")
            if t == "response.output.delta":
                if msg.get("kind") == "text":
                    text += msg.get("text", "")
            elif t == "response.done":
                print("response.done text =", repr(msg.get("text") or text))
                await ws.send(json.dumps({"type": "session.close", "reason": "turn_done"}))
            elif t in ("session.closed", "error"):
                print("<-", t, msg.get("reason") or msg.get("error"))
                break
    assert sid, "no session_id"
    print("CHAT OK")


# ----------------------------------------------------------------------------- video duplex
async def run_video(chunks=4):
    print("\n=== VIDEO DUPLEX ===")
    audio = _audio_b64(seconds=1.0)
    frames = _frames_b64(n=2, fps=2)
    print(f"prepared audio + {len(frames)} jpeg frame(s)")
    async with websockets.connect(f"{GATEWAY}?mode=video", max_size=MAX_WS) as ws:
        sid = await _handshake(ws, {"system_prompt": "你是一个有用的助手"})
        print("SID =", sid)
        # send chunks; each carries audio + video frames, matching the page's
        # sendChunk() which puts frames under `video_frames`.
        for i in range(chunks):
            await ws.send(json.dumps({
                "type": "input.append",
                "input": {"audio": audio, "video_frames": frames, "max_slice_nums": 1},
            }))
        kinds = {"listen": 0, "text": 0, "audio": 0}
        vision_seen = False
        deadline = chunks * 2 + 8
        try:
            while sum(kinds.values()) < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                t = msg.get("type")
                if t == "response.output.delta":
                    kind = msg.get("kind", "?")
                    kinds[kind] = kinds.get(kind, 0) + 1
                    met = msg.get("metrics") or {}
                    vs, vt = met.get("vision_slices"), met.get("vision_tokens")
                    if vs is not None:
                        vision_seen = True
                    print(f"  delta kind={kind} vision_slices={vs} vision_tokens={vt} "
                          f"wall={met.get('wall_clock_ms')}")
                elif t in ("session.closed", "error"):
                    print("<-", t, msg.get("reason") or msg.get("error"))
                    break
        except asyncio.TimeoutError:
            print("  (no more events)")
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
    print("kinds:", kinds, "| vision metrics seen:", vision_seen)
    assert sid, "no session_id"
    print("VIDEO OK")


# ----------------------------------------------------------------------------- main
async def main(which):
    if which in ("all", "chat"):
        await run_chat(streaming=False)
    if which in ("all", "chat-stream"):
        await run_chat(streaming=True)
    if which in ("all", "video"):
        await run_video()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    asyncio.run(main(arg))
