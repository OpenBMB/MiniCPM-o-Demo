#!/usr/bin/env python3
"""Minimal MiniCPM-o Realtime video API client."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import shutil
import ssl
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import websockets


INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


def now() -> float:
    return time.perf_counter()


def ms(seconds: float) -> float:
    return seconds * 1000.0


def normalize_realtime_url(url: str, mode: str = "video") -> str:
    parsed = urlparse(url)
    if parsed.scheme in ("ws", "wss") and parsed.path:
        return url
    if parsed.scheme not in ("http", "https", "ws", "wss"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    host = parsed.netloc or parsed.path
    return f"{scheme}://{host.rstrip('/')}/v1/realtime?mode={mode}"


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def b64_float32(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype("<f4", copy=False).tobytes()).decode("ascii")


def decode_audio_duration_ms(audio_b64: str) -> float:
    raw = base64.b64decode(audio_b64)
    return (len(raw) // 4) / OUTPUT_SAMPLE_RATE * 1000.0


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(values: list[float], prefix: str) -> dict[str, float | None]:
    return {
        f"{prefix}_min_ms": min(values) if values else None,
        f"{prefix}_p50_ms": percentile(values, 50),
        f"{prefix}_p90_ms": percentile(values, 90),
        f"{prefix}_p99_ms": percentile(values, 99),
        f"{prefix}_max_ms": max(values) if values else None,
    }


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def prepare_media(video_path: Path, work_dir: Path, frame_fps: float, jpeg_width: int) -> tuple[np.ndarray, list[str]]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but not found in PATH")

    wav_path = work_dir / "audio_16k.wav"
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(INPUT_SAMPLE_RATE), "-sample_fmt", "s16",
        str(wav_path),
    ])

    # Extract approximately one JPEG per second by default. Scale down to keep JSON frames small.
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={frame_fps},scale={jpeg_width}:-2",
        "-q:v", "5",
        str(frames_dir / "frame_%06d.jpg"),
    ])

    samples = load_wav_as_16k_float32(wav_path)
    frames = [b64_file(p) for p in sorted(frames_dir.glob("frame_*.jpg"))]
    if not frames:
        raise RuntimeError("No video frames extracted")
    return samples, frames


def load_wav_as_16k_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if sample_rate != INPUT_SAMPLE_RATE:
        old_x = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        new_len = int(round(len(data) * INPUT_SAMPLE_RATE / sample_rate))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        data = np.interp(new_x, old_x, data).astype(np.float32)

    return np.clip(data, -1.0, 1.0).astype(np.float32)


def iter_audio_chunks(samples: np.ndarray, chunk_ms: int, tail_silence_s: float) -> list[np.ndarray]:
    if tail_silence_s > 0:
        silence = np.zeros(int(round(INPUT_SAMPLE_RATE * tail_silence_s)), dtype=np.float32)
        samples = np.concatenate([samples, silence])
    chunk_len = int(INPUT_SAMPLE_RATE * chunk_ms / 1000)
    chunks = []
    for start in range(0, len(samples), chunk_len):
        chunk = samples[start : start + chunk_len]
        if len(chunk):
            chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


@dataclass
class ProbeState:
    started_s: float
    ws_open_s: float | None = None
    session_id: str = ""
    first_text_s: float | None = None
    first_audio_s: float | None = None
    text_chunks: list[str] = field(default_factory=list)
    output_chunk_times_s: list[float] = field(default_factory=list)
    output_chunk_durations_ms: list[float] = field(default_factory=list)
    output_chunk_eot: list[bool] = field(default_factory=list)
    input_chunks_sent: int = 0
    input_frames_sent: int = 0
    listen_events: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def event(self, typ: str, **kwargs: Any) -> None:
        item = {"t_ms": ms(now() - self.started_s), "type": typ}
        item.update(kwargs)
        self.events.append(item)


async def sender(
    ws: websockets.ClientConnection,
    state: ProbeState,
    audio_chunks: list[np.ndarray],
    frames: list[str],
    chunk_ms: int,
    stop_event: asyncio.Event,
    continue_silence: bool,
    max_slice_nums: int,
) -> None:
    next_send = now()
    idx = 0
    silence = np.zeros(int(INPUT_SAMPLE_RATE * chunk_ms / 1000), dtype=np.float32)

    while not stop_event.is_set():
        if idx < len(audio_chunks):
            chunk = audio_chunks[idx]
        elif continue_silence:
            chunk = silence
        else:
            break

        frame = frames[min(idx, len(frames) - 1)]
        sleep_s = next_send - now()
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

        msg = {
            "type": "input.append",
            "input": {
                "audio": b64_float32(chunk),
                "video_frames": [frame],
                "force_listen": False,
                "max_slice_nums": max_slice_nums,
            },
        }
        await ws.send(json.dumps(msg))
        state.input_chunks_sent += 1
        state.input_frames_sent += 1
        state.event("client.input.append", chunk_idx=idx + 1)

        idx += 1
        next_send += chunk_ms / 1000.0


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args.url = normalize_realtime_url(args.url, "video")

    with tempfile.TemporaryDirectory(prefix="realtime-video-probe-") as tmp:
        work_dir = Path(tmp)
        samples, frames = prepare_media(Path(args.video), work_dir, args.frame_fps, args.jpeg_width)
        audio_chunks = iter_audio_chunks(samples, args.chunk_ms, args.tail_silence_s)

        state = ProbeState(started_s=now())
        ssl_ctx = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
        connect_t0 = now()
        stop_sender = asyncio.Event()
        sender_task: asyncio.Task[None] | None = None

        async with websockets.connect(
            args.url,
            ssl=ssl_ctx,
            open_timeout=args.open_timeout,
            max_size=args.max_message_mb * 1024 * 1024,
        ) as ws:
            state.ws_open_s = now()
            state.event("ws.open", ws_ready_ms=ms(state.ws_open_s - connect_t0))

            close_after_s = now() + args.max_session_s if args.max_session_s is not None else None

            try:
                while True:
                    if close_after_s is not None and now() >= close_after_s:
                        state.event("client.max_session_reached")
                        break
                    timeout = max(0.1, close_after_s - now()) if close_after_s is not None else None
                    raw = await (ws.recv() if timeout is None else asyncio.wait_for(ws.recv(), timeout))
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")
                    state.event(f"server.{msg_type}")

                    if msg_type in ("session.queue_done", "queue_done") and sender_task is None:
                        await ws.send(json.dumps({
                            "type": "session.init",
                            "payload": {
                                "system_prompt": args.instructions,
                            },
                        }))
                        state.event("client.session.init")

                    elif msg_type == "session.created":
                        state.session_id = msg.get("session_id", "")
                        sender_task = asyncio.create_task(
                            sender(
                                ws, state, audio_chunks, frames, args.chunk_ms,
                                stop_sender, args.continue_silence, args.max_slice_nums,
                            )
                        )
                        state.event("client.sender_started", audio_chunks=len(audio_chunks), frames=len(frames))

                    elif msg_type == "response.output.delta" and msg.get("kind") == "listen":
                        state.listen_events += 1
                        if msg.get("metrics"):
                            state.event("server.response.output.delta.listen", metrics=msg.get("metrics"))
                        if args.stop_on_end_of_turn and (state.text_chunks or state.output_chunk_times_s):
                            break

                    elif msg_type == "response.output.delta" and msg.get("kind") == "text":
                        recv_s = now()
                        text = msg.get("text") or ""
                        if text:
                            if state.first_text_s is None:
                                state.first_text_s = recv_s
                            state.text_chunks.append(text)

                    elif msg_type == "response.output.delta" and msg.get("kind") == "audio":
                        recv_s = now()
                        if msg.get("audio"):
                            if state.first_audio_s is None:
                                state.first_audio_s = recv_s
                            state.output_chunk_times_s.append(recv_s)
                            state.output_chunk_durations_ms.append(decode_audio_duration_ms(msg["audio"]))
                            state.output_chunk_eot.append(False)

                    elif msg_type == "response.done":
                        text = msg.get("text") or ""
                        if text and not state.text_chunks:
                            state.text_chunks.append(text)
                        if args.stop_on_end_of_turn:
                            break

                    elif msg_type == "session.closed":
                        break

                    elif msg_type == "error":
                        raise RuntimeError(json.dumps(msg, ensure_ascii=False))
            finally:
                stop_sender.set()
                if sender_task is not None:
                    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                        await asyncio.wait_for(sender_task, timeout=1.0)
                with contextlib.suppress(Exception):
                    await ws.send(json.dumps({"type": "session.close", "reason": "user_stop"}))

        return build_result(args, state, len(audio_chunks), len(frames))


def build_result(args: argparse.Namespace, state: ProbeState, source_audio_chunks: int, source_frames: int) -> dict[str, Any]:
    chunk_intervals_ms = [
        ms(b - a)
        for a, b in zip(state.output_chunk_times_s, state.output_chunk_times_s[1:])
    ]
    return {
        "region": args.region,
        "url": args.url,
        "success": True,
        "session_id": state.session_id,
        "ws_ready_ms": ms(state.ws_open_s - state.started_s) if state.ws_open_s else None,
        "first_text_ms": ms(state.first_text_s - state.started_s) if state.first_text_s else None,
        "first_audio_ms": ms(state.first_audio_s - state.started_s) if state.first_audio_s else None,
        "listen_events": state.listen_events,
        "source_audio_chunks": source_audio_chunks,
        "source_frames": source_frames,
        "input_chunks_sent": state.input_chunks_sent,
        "input_frames_sent": state.input_frames_sent,
        "text_delta_chunks": len(state.text_chunks),
        "text": "".join(state.text_chunks),
        "output_audio_chunks": len(state.output_chunk_times_s),
        **summarize(chunk_intervals_ms, "chunk_interarrival"),
        "events": state.events if args.include_events else None,
    }


def print_human(result: dict[str, Any]) -> None:
    for key in [
        "region", "url", "session_id", "ws_ready_ms", "first_text_ms", "first_audio_ms",
        "listen_events", "source_audio_chunks", "source_frames", "input_chunks_sent",
        "input_frames_sent", "text_delta_chunks", "output_audio_chunks",
        "chunk_interarrival_p50_ms", "chunk_interarrival_p90_ms",
    ]:
        print(f"{key}: {result.get(key)}")
    if result.get("text"):
        print("text:")
        print(result["text"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Host URL or full WSS URL")
    parser.add_argument("--video", required=True, help="Input video file, e.g. mp4")
    parser.add_argument("--region", default="video-probe")
    parser.add_argument("--instructions", default="你是一个视频语音助手，请根据用户语音和画面内容自然、完整地回答。")
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--frame-fps", type=float, default=1.0)
    parser.add_argument("--jpeg-width", type=int, default=640)
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--tail-silence-s", type=float, default=3.0)
    parser.add_argument("--continue-silence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-end-of-turn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-session-s", type=float, default=None)
    parser.add_argument("--open-timeout", type=float, default=15.0)
    parser.add_argument("--max-message-mb", type=int, default=128)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--pretty-json", action="store_true")
    parser.add_argument("--include-events", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = asyncio.run(run_probe(args))
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1

    if args.json:
        result = {k: v for k, v in result.items() if v is not None}
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    elif args.pretty_json:
        result = {k: v for k, v in result.items() if v is not None}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
