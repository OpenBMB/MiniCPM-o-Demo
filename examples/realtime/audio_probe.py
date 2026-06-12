#!/usr/bin/env python3
"""CLI latency and streaming smoothness probe for MiniCPM-o Realtime API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import ssl
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
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


def load_wav_as_16k_float32(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        # Most PCM WAV files are int32. Float32 WAV is uncommon with Python wave.
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


def iter_audio_chunks(samples: np.ndarray, chunk_ms: int) -> list[np.ndarray]:
    chunk_len = int(INPUT_SAMPLE_RATE * chunk_ms / 1000)
    if chunk_len <= 0:
        raise ValueError("chunk_ms must be positive")
    chunks = []
    for start in range(0, len(samples), chunk_len):
        chunk = samples[start : start + chunk_len]
        if len(chunk) == 0:
            continue
        chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


def append_tail_silence(samples: np.ndarray, silence_s: float) -> np.ndarray:
    if silence_s <= 0:
        return samples
    silence = np.zeros(int(round(INPUT_SAMPLE_RATE * silence_s)), dtype=np.float32)
    return np.concatenate([samples, silence])


def b64_float32(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype("<f4", copy=False).tobytes()).decode("ascii")


def decode_audio_duration_ms(audio_b64: str) -> float:
    raw = base64.b64decode(audio_b64)
    sample_count = len(raw) // 4
    return sample_count / OUTPUT_SAMPLE_RATE * 1000.0


def normalize_realtime_url(url: str, mode: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in ("ws", "wss") and parsed.path:
        return url
    if parsed.scheme not in ("http", "https", "ws", "wss"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    host = parsed.netloc or parsed.path
    return f"{scheme}://{host.rstrip('/')}/v1/realtime?mode={mode}"


@dataclass
class PlaybackProbe:
    playback_delay_ms: float
    underrun_threshold_ms: float
    turn_active: bool = False
    playback_end_s: float | None = None
    ahead_samples_ms: list[float] = field(default_factory=list)
    underrun_count: int = 0
    underrun_total_ms: float = 0.0
    underruns: list[dict[str, float]] = field(default_factory=list)

    def on_audio(self, recv_s: float, duration_ms: float, end_of_turn: bool) -> float:
        if not self.turn_active:
            self.turn_active = True
            self.playback_end_s = recv_s + self.playback_delay_ms / 1000.0

        assert self.playback_end_s is not None
        gap_ms = ms(recv_s - self.playback_end_s)
        if gap_ms > self.underrun_threshold_ms:
            self.underrun_count += 1
            self.underrun_total_ms += gap_ms
            self.underruns.append({"at_ms": ms(recv_s), "gap_ms": gap_ms})
            self.playback_end_s = recv_s

        self.playback_end_s += duration_ms / 1000.0
        ahead_ms = max(0.0, ms(self.playback_end_s - recv_s))
        self.ahead_samples_ms.append(ahead_ms)

        if end_of_turn:
            self.turn_active = False
            self.playback_end_s = None

        return ahead_ms


@dataclass
class ProbeState:
    started_s: float
    ws_open_s: float | None = None
    first_text_s: float | None = None
    first_audio_s: float | None = None
    text_chunks: list[str] = field(default_factory=list)
    output_chunk_times_s: list[float] = field(default_factory=list)
    output_chunk_durations_ms: list[float] = field(default_factory=list)
    output_chunk_eot: list[bool] = field(default_factory=list)
    ping_rtts_ms: list[float] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def event(self, typ: str, **kwargs: Any) -> None:
        item = {"t_ms": ms(now() - self.started_s), "type": typ}
        item.update(kwargs)
        self.events.append(item)


async def measure_ping(ws: websockets.ClientConnection, count: int, interval_s: float) -> list[float]:
    rtts: list[float] = []
    for _ in range(count):
        t0 = now()
        pong = await ws.ping()
        await pong
        rtts.append(ms(now() - t0))
        if interval_s > 0:
            await asyncio.sleep(interval_s)
    return rtts


async def sender(
    ws: websockets.ClientConnection,
    chunks: list[np.ndarray],
    chunk_ms: int,
    stop_event: asyncio.Event,
    continue_silence: bool,
) -> None:
    # Send in real-time cadence to match the browser demo. In duplex mode the
    # browser keeps sending mic ticks even during silence, so optionally keep
    # sending zero chunks after the WAV finishes.
    next_send = now()
    silence = np.zeros(int(INPUT_SAMPLE_RATE * chunk_ms / 1000), dtype=np.float32)
    idx = 0
    while not stop_event.is_set():
        if idx < len(chunks):
            chunk = chunks[idx]
            idx += 1
        elif continue_silence:
            chunk = silence
        else:
            break

        sleep_s = next_send - now()
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        await ws.send(json.dumps({
            "type": "input.append",
            "input": {
                "audio": b64_float32(chunk),
                "force_listen": False,
            },
        }))
        next_send += chunk_ms / 1000.0


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args.url = normalize_realtime_url(args.url, args.mode)
    state = ProbeState(started_s=now())
    parsed_url = urlparse(args.url)
    ssl_ctx = None
    if parsed_url.scheme == "wss":
        ssl_ctx = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
    connect_t0 = now()
    async with websockets.connect(
        args.url,
        ssl=ssl_ctx,
        open_timeout=args.open_timeout,
        max_size=args.max_message_mb * 1024 * 1024,
    ) as ws:
        state.ws_open_s = now()
        state.event("ws.open", ws_ready_ms=ms(state.ws_open_s - connect_t0))

        if args.ping_count > 0:
            state.ping_rtts_ms = await measure_ping(ws, args.ping_count, args.ping_interval)
            state.event("ws.ping_done", count=len(state.ping_rtts_ms))

        if args.ping_only:
            return build_result(args, state, None)

        if not args.input_wav:
            raise ValueError("--input-wav is required unless --ping-only is set")

        samples = append_tail_silence(load_wav_as_16k_float32(args.input_wav), args.tail_silence_s)
        chunks = iter_audio_chunks(samples, args.chunk_ms)
        playback = PlaybackProbe(
            playback_delay_ms=args.playback_delay_ms,
            underrun_threshold_ms=args.underrun_threshold_ms,
        )
        stop_sender = asyncio.Event()

        session_created = False
        sender_task: asyncio.Task[None] | None = None
        close_after_s = now() + args.max_session_s if args.max_session_s is not None else None

        while True:
            if close_after_s is not None and now() >= close_after_s:
                state.event("client.max_session_reached")
                break
            timeout = max(0.1, close_after_s - now()) if close_after_s is not None else None
            try:
                if timeout is None:
                    raw = await ws.recv()
                else:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                state.event("client.max_session_reached")
                break
            msg = json.loads(raw)
            msg_type = msg.get("type", "")
            state.event(f"server.{msg_type}")

            if msg_type in ("session.queue_done", "queue_done") and not session_created:
                await ws.send(json.dumps({
                    "type": "session.init",
                    "payload": {
                        "system_prompt": args.instructions,
                    },
                }))
                state.event("client.session.init")

            elif msg_type == "session.created":
                session_created = True
                if sender_task is None:
                    sender_task = asyncio.create_task(
                        sender(ws, chunks, args.chunk_ms, stop_sender, args.continue_silence)
                    )
                    state.event("client.audio_sender_started", chunks=len(chunks))

            elif msg_type == "response.output.delta" and msg.get("kind") == "text":
                recv_s = now()
                text = msg.get("text") or ""
                if text:
                    if state.first_text_s is None:
                        state.first_text_s = recv_s
                    state.text_chunks.append(text)

            elif msg_type == "response.output.delta" and msg.get("kind") == "audio" and msg.get("audio"):
                recv_s = now()
                if state.first_audio_s is None:
                    state.first_audio_s = recv_s
                duration_ms = decode_audio_duration_ms(msg["audio"])
                end_of_turn = False
                ahead_ms = playback.on_audio(recv_s, duration_ms, end_of_turn)
                state.output_chunk_times_s.append(recv_s)
                state.output_chunk_durations_ms.append(duration_ms)
                state.output_chunk_eot.append(end_of_turn)
                state.event(
                    "client.audio_chunk_received",
                    duration_ms=duration_ms,
                    ahead_ms=ahead_ms,
                    text_chars=0,
                    end_of_turn=end_of_turn,
                )

            elif msg_type == "response.output.delta" and msg.get("kind") == "listen":
                if args.stop_on_end_of_turn and state.output_chunk_times_s:
                    break

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

        if sender_task is not None:
            stop_sender.set()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(sender_task, timeout=1.0)

        try:
            await ws.send(json.dumps({"type": "session.close", "reason": "user_stop"}))
        except Exception:
            pass

        return build_result(args, state, playback)


def build_result(args: argparse.Namespace, state: ProbeState, playback: PlaybackProbe | None) -> dict[str, Any]:
    chunk_intervals_ms = [
        ms(b - a)
        for a, b in zip(state.output_chunk_times_s, state.output_chunk_times_s[1:])
    ]

    jitter_samples: list[float] = []
    late_jitter_samples: list[float] = []
    early_jitter_samples: list[float] = []
    for i, interval in enumerate(chunk_intervals_ms, start=1):
        prev_duration = state.output_chunk_durations_ms[i - 1]
        cur_duration = state.output_chunk_durations_ms[i]
        prev_eot = state.output_chunk_eot[i - 1]
        cur_eot = state.output_chunk_eot[i]
        if (
            prev_duration >= args.min_jitter_audio_ms
            and cur_duration >= args.min_jitter_audio_ms
            and not prev_eot
            and not cur_eot
        ):
            delta = interval - args.expected_chunk_ms
            jitter_samples.append(abs(delta))
            late_jitter_samples.append(max(0.0, delta))
            early_jitter_samples.append(max(0.0, -delta))

    result: dict[str, Any] = {
        "region": args.region,
        "url": args.url,
        "success": True,
        "ws_ready_ms": ms(state.ws_open_s - state.started_s) if state.ws_open_s else None,
        "first_text_ms": ms(state.first_text_s - state.started_s) if state.first_text_s else None,
        "first_audio_ms": ms(state.first_audio_s - state.started_s) if state.first_audio_s else None,
        "text_delta_chunks": len(state.text_chunks),
        "text": "".join(state.text_chunks),
        "output_audio_chunks": len(state.output_chunk_times_s),
        "expected_chunk_ms": args.expected_chunk_ms,
        **summarize(state.ping_rtts_ms, "ping_rtt"),
        **summarize(chunk_intervals_ms, "chunk_interarrival"),
        **summarize(jitter_samples, "chunk_jitter"),
        **summarize(late_jitter_samples, "chunk_late_jitter"),
        **summarize(early_jitter_samples, "chunk_early_jitter"),
    }

    if state.ping_rtts_ms:
        result["ping_jitter_ms"] = statistics.pstdev(state.ping_rtts_ms) if len(state.ping_rtts_ms) > 1 else 0.0

    if playback is not None:
        result.update({
            **summarize(playback.ahead_samples_ms, "audio_ahead"),
            "underrun_count": playback.underrun_count,
            "underrun_total_ms": playback.underrun_total_ms,
            "underruns": playback.underruns,
        })

    if args.include_events:
        result["events"] = state.events

    return result


def print_human(result: dict[str, Any]) -> None:
    print(f"region: {result.get('region')}")
    print(f"url: {result.get('url')}")
    print(f"ws_ready_ms: {fmt(result.get('ws_ready_ms'))}")
    print(f"first_text_ms: {fmt(result.get('first_text_ms'))}")
    print(f"first_audio_ms: {fmt(result.get('first_audio_ms'))}")
    print(f"text_delta_chunks: {result.get('text_delta_chunks')}")
    if result.get("text"):
        print(f"text: {result.get('text')}")
    print(f"output_audio_chunks: {result.get('output_audio_chunks')}")
    print(f"ping_rtt_p50/p90/p99_ms: {fmt(result.get('ping_rtt_p50_ms'))} / {fmt(result.get('ping_rtt_p90_ms'))} / {fmt(result.get('ping_rtt_p99_ms'))}")
    print(f"chunk_interarrival_p50/p90/p99_ms: {fmt(result.get('chunk_interarrival_p50_ms'))} / {fmt(result.get('chunk_interarrival_p90_ms'))} / {fmt(result.get('chunk_interarrival_p99_ms'))}")
    print(f"chunk_jitter_p50/p90/p99_ms: {fmt(result.get('chunk_jitter_p50_ms'))} / {fmt(result.get('chunk_jitter_p90_ms'))} / {fmt(result.get('chunk_jitter_p99_ms'))}")
    print(f"chunk_late_jitter_p50/p90/p99_ms: {fmt(result.get('chunk_late_jitter_p50_ms'))} / {fmt(result.get('chunk_late_jitter_p90_ms'))} / {fmt(result.get('chunk_late_jitter_p99_ms'))}")
    print(f"chunk_early_jitter_p50/p90/p99_ms: {fmt(result.get('chunk_early_jitter_p50_ms'))} / {fmt(result.get('chunk_early_jitter_p90_ms'))} / {fmt(result.get('chunk_early_jitter_p99_ms'))}")
    print(f"audio_ahead_min/p50/p90_ms: {fmt(result.get('audio_ahead_min_ms'))} / {fmt(result.get('audio_ahead_p50_ms'))} / {fmt(result.get('audio_ahead_p90_ms'))}")
    print(f"underrun_count: {fmt(result.get('underrun_count'))}")
    print(f"underrun_total_ms: {fmt(result.get('underrun_total_ms'))}")


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True,
                        help="Host or WSS URL, e.g. https://host or wss://host/v1/realtime?mode=audio")
    parser.add_argument("--mode", default="audio", choices=["audio", "video"],
                        help="Used when --url is a host URL instead of a full WebSocket URL")
    parser.add_argument("--region", default="unknown", help="Label written to output")
    parser.add_argument("--input-wav", help="PCM WAV to send as user audio")
    parser.add_argument("--instructions", default="你是一个语音助手，请自然、完整地回答用户的问题。")
    parser.add_argument("--chunk-ms", type=int, default=1000, help="Uplink audio chunk duration")
    parser.add_argument("--tail-silence-s", type=float, default=3.0, help="Append trailing silence so VAD can finalize")
    parser.add_argument("--continue-silence", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep sending silent chunks after the WAV, matching browser mic behavior")
    parser.add_argument("--expected-chunk-ms", type=float, default=1000.0, help="Expected downlink chunk cadence")
    parser.add_argument("--min-jitter-audio-ms", type=float, default=900.0, help="Only chunks at least this long are used for jitter")
    parser.add_argument("--playback-delay-ms", type=float, default=200.0, help="Simulated client playback delay")
    parser.add_argument("--underrun-threshold-ms", type=float, default=10.0)
    parser.add_argument("--max-session-s", type=float, default=None,
                        help="Optional safety timeout. By default, wait for end_of_turn/session close.")
    parser.add_argument("--stop-on-end-of-turn", action="store_true", default=True)
    parser.add_argument("--ping-only", action="store_true", help="Only measure WebSocket connect and ping RTT")
    parser.add_argument("--ping-count", type=int, default=5)
    parser.add_argument("--ping-interval", type=float, default=0.2)
    parser.add_argument("--open-timeout", type=float, default=15.0)
    parser.add_argument("--max-message-mb", type=int, default=128)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    parser.add_argument("--json", action="store_true", help="Print compact JSON")
    parser.add_argument("--pretty-json", action="store_true", help="Print formatted JSON")
    parser.add_argument("--include-events", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = asyncio.run(run_probe(args))
    except Exception as exc:
        error = {"region": args.region, "url": args.url, "success": False, "error": str(exc)}
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    elif args.pretty_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
