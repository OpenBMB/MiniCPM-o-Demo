"""Replay a recorded FC realtime API session from gateway session logs.

The gateway recorder stores faithful WebSocket frames under
``data/sessions/<session_id>/stream.jsonl``. Large audio payloads are replaced
with ``@blob/NNNN.wav`` pointers. This script reconstructs the upstream frames
and sends them to ``/v1/realtime?mode=audio`` so the API can be regression tested
from CLI with a real previous session.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import ssl
import time
import wave
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import httpx
import websockets


DEFAULT_SESSION_ID = "sess_6fef8139e867"


def ws_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/") + "/v1/realtime?mode=audio")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def read_wav_as_float32_base64(path: Path) -> str:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise RuntimeError(f"unsupported wav sample width {width}: {path}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return base64.b64encode(np.asarray(audio, dtype=np.float32).tobytes()).decode("ascii")


def restore_blobs(frame: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    if frame.get("type") != "input.append" or not isinstance(frame.get("input"), dict):
        return frame
    payload = dict(frame)
    inp = dict(payload["input"])

    # Older frontend frames used audio_base64; gateway recording externalizes the
    # canonical input.audio field. Replay both shapes if present.
    for key in ("audio", "audio_base64"):
        value = inp.get(key)
        if isinstance(value, str) and value.startswith("@blob/"):
            inp[key] = read_wav_as_float32_base64(session_dir / value[1:])
    if "audio" in inp and "audio_base64" not in inp:
        inp["audio_base64"] = inp["audio"]
    payload["input"] = inp
    return payload


def load_upstream_frames(session_dir: Path, *, skip_recorded_tool_results: bool) -> list[tuple[float, dict[str, Any]]]:
    frames: list[tuple[float, dict[str, Any]]] = []
    with (session_dir / "stream.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("dir") != "up":
                continue
            frame = record.get("frame") or {}
            if skip_recorded_tool_results and str(frame.get("type") or "").startswith("input.tool_result"):
                continue
            if frame.get("type") not in {
                "session.init",
                "input.append",
                "input.tool_result",
                "input.tool_result.delta",
                "input.tool_result.done",
                "session.close",
            }:
                continue
            frames.append((float(record.get("ts") or 0.0), restore_blobs(frame, session_dir)))
    return frames


def summarize_event(event: dict[str, Any]) -> str:
    typ = event.get("type") or ""
    if typ == "response.output.delta":
        kind = event.get("kind")
        if kind == "audio":
            return f"{typ} audio={len(str(event.get('audio') or ''))}"
        return f"{typ} kind={kind} text={event.get('text') or ''}"
    if typ.startswith("response.tool_call"):
        raw = event.get("raw")
        return f"{typ} id={event.get('tool_call_id')} delta={event.get('delta') or ''} raw={raw or ''}"
    if typ.startswith("response.think"):
        return f"{typ} {event.get('delta') or ''}"
    if typ == "response.output.sp_tokens":
        return f"{typ} token={event.get('token')}"
    if typ == "response.tool_result":
        return f"{typ} id={event.get('tool_call_id')}"
    return typ


async def _execute_board_tool(
    event: dict[str, Any],
    *,
    base_url: str,
    insecure: bool,
) -> str:
    """Execute the same board tool endpoint used by the browser Demo."""

    raw = dict(event.get("raw") or {})
    arguments = json.loads(str(raw.get("arguments") or "{}"))
    name = str(arguments.get("name") or "").strip()
    if raw.get("name") != "display_object_on_board" or not name:
        raise RuntimeError(f"unsupported auto tool call: {raw}")
    async with httpx.AsyncClient(verify=not insecure, timeout=30.0) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/fc_board/tools/display_object_on_board",
            json={
                "name": name,
                "tool_call_id": event.get("tool_call_id"),
            },
        )
        response.raise_for_status()
        result = response.json()
    return str(
        result.get("tool_response_content")
        or json.dumps(result, ensure_ascii=False)
    )


async def drain(
    ws: websockets.WebSocketClientProtocol,
    out,
    *,
    idle_timeout: float,
    max_idle: int,
    auto_execute_board_tool: bool,
    auto_tool_result_ids: set[str],
    base_url: str,
    insecure: bool,
) -> bool:
    idle = 0
    closed = False
    while idle < max_idle:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=idle_timeout)
        except asyncio.TimeoutError:
            idle += 1
            continue
        except websockets.ConnectionClosedOK:
            break
        event = json.loads(raw)
        out.write(json.dumps({"ts": time.time(), "frame": event}, ensure_ascii=False) + "\n")
        out.flush()
        print("RX", summarize_event(event)[:500], flush=True)
        if (
            auto_execute_board_tool
            and event.get("type") == "response.tool_call.args.raw"
            and not dict(event.get("raw") or {}).get("error")
        ):
            tool_call_id = str(event.get("tool_call_id") or "")
            if tool_call_id and tool_call_id not in auto_tool_result_ids:
                content = await _execute_board_tool(
                    event,
                    base_url=base_url,
                    insecure=insecure,
                )
                tool_result = {
                    "type": "input.tool_result",
                    "tool_call_id": tool_call_id,
                    "contents": [{"kind": "text", "text": content}],
                }
                await ws.send(json.dumps(tool_result, ensure_ascii=False))
                out.write(json.dumps({
                    "ts": time.time(),
                    "dir": "up",
                    "frame": tool_result,
                }, ensure_ascii=False) + "\n")
                out.flush()
                auto_tool_result_ids.add(tool_call_id)
                print("TX auto input.tool_result", tool_call_id, flush=True)
        if event.get("type") == "session.closed":
            closed = True
            break
    return closed


async def replay(args: argparse.Namespace) -> None:
    session_dir = Path(args.session_dir or Path(args.data_dir) / "sessions" / args.session_id)
    frames = load_upstream_frames(session_dir, skip_recorded_tool_results=args.skip_recorded_tool_results)
    if not frames:
        raise RuntimeError(f"no replayable upstream frames found: {session_dir}")
    print(f"session_dir={session_dir}")
    print(f"upstream_frames={len(frames)}")
    if args.dry_run:
        for _, frame in frames[: args.print_limit]:
            print(json.dumps({"type": frame.get("type"), "keys": list(frame.keys())}, ensure_ascii=False))
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl._create_unverified_context() if args.insecure else None
    auto_tool_result_ids: set[str] = set()
    async with websockets.connect(ws_url(args.base_url), ssl=ssl_context, max_size=128 * 1024 * 1024) as ws:
        with output.open("w") as out:
            previous_ts: float | None = None
            for index, (record_ts, frame) in enumerate(frames):
                if previous_ts is not None and args.timing != "fast":
                    delay = max(0.0, record_ts - previous_ts)
                    if args.timing == "scaled":
                        delay /= max(args.speed, 0.001)
                    await asyncio.sleep(min(delay, args.max_sleep))
                previous_ts = record_ts
                await ws.send(json.dumps(frame, ensure_ascii=False))
                print("TX", index, frame.get("type"), flush=True)
                if frame.get("type") == "session.init":
                    await drain(
                        ws,
                        out,
                        idle_timeout=args.idle_timeout,
                        max_idle=1,
                        auto_execute_board_tool=args.auto_execute_board_tool,
                        auto_tool_result_ids=auto_tool_result_ids,
                        base_url=args.base_url,
                        insecure=args.insecure,
                    )
                elif frame.get("type") == "session.close":
                    await drain(
                        ws,
                        out,
                        idle_timeout=args.idle_timeout,
                        max_idle=args.close_idle_rounds,
                        auto_execute_board_tool=args.auto_execute_board_tool,
                        auto_tool_result_ids=auto_tool_result_ids,
                        base_url=args.base_url,
                        insecure=args.insecure,
                    )
                else:
                    await drain(
                        ws,
                        out,
                        idle_timeout=args.idle_timeout,
                        max_idle=args.between_idle_rounds,
                        auto_execute_board_tool=args.auto_execute_board_tool,
                        auto_tool_result_ids=auto_tool_result_ids,
                        base_url=args.base_url,
                        insecure=args.insecure,
                    )
            await drain(
                ws,
                out,
                idle_timeout=args.idle_timeout,
                max_idle=args.close_idle_rounds,
                auto_execute_board_tool=args.auto_execute_board_tool,
                auto_tool_result_ids=auto_tool_result_ids,
                base_url=args.base_url,
                insecure=args.insecure,
            )
    print(f"wrote={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded FC realtime session")
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-url", default="https://82.157.64.212:8444")
    parser.add_argument("--output", default="/tmp/fc_duplex_replay_events.jsonl")
    parser.add_argument("--timing", choices=["fast", "scaled", "realtime"], default="scaled")
    parser.add_argument("--speed", type=float, default=4.0, help="Replay speedup for --timing scaled")
    parser.add_argument("--max-sleep", type=float, default=1.0)
    parser.add_argument("--idle-timeout", type=float, default=0.5)
    parser.add_argument("--between-idle-rounds", type=int, default=1)
    parser.add_argument("--close-idle-rounds", type=int, default=6)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    parser.add_argument(
        "--auto-execute-board-tool",
        action="store_true",
        help="Execute valid display_object_on_board calls and send input.tool_result.",
    )
    parser.add_argument(
        "--skip-recorded-tool-results",
        action="store_true",
        help="Replay only recorded init/audio/close frames; do not send recorded tool results with old tool_call_id values.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-limit", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
