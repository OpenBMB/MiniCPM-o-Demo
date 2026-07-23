"""FC Duplex stateless Unit-boundary resume end-to-end smoke.

The script opens a live `/v1/realtime?mode=audio` session, sends silent one-second
Units until the API exposes an available checkpoint, disconnects, then opens a
new connection and resumes exclusively from the saved public bidirectional
history.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import websockets


def _ws_url(base_url: str) -> str:
    """Convert an HTTP(S) service URL into the FC realtime WebSocket URL."""

    parsed = urlsplit(base_url.rstrip("/") + "/v1/realtime?mode=audio")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _silent_audio_base64(sample_rate: int = 16000) -> str:
    """Return one second of float32 PCM silence."""

    audio = np.zeros(sample_rate, dtype=np.float32)
    return base64.b64encode(audio.tobytes()).decode("ascii")


async def _wait_queue_done(ws: Any) -> None:
    """Drain queue events until the Gateway assigned a Worker."""

    while True:
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
        if event.get("type") == "session.queue_done":
            return
        if event.get("type") in {"error", "session.closed"}:
            raise RuntimeError(f"queue failed: {event}")


async def _collect_until_checkpoint(
    ws: Any,
    *,
    history: list[dict[str, Any]],
    target_unit_index: int,
) -> dict[str, Any]:
    """Collect downstream events until one Unit checkpoint arrives."""

    while True:
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
        history.append(event)
        if (
            event.get("type") == "response.unit.committed"
            and event.get("unit_index") == target_unit_index
        ):
            return event
        if event.get("type") in {"session.closed", "session.resume.failed"}:
            raise RuntimeError(f"session ended before checkpoint: {event}")


async def run_smoke(args: argparse.Namespace) -> None:
    """Run live generation followed by stateless resume."""

    ssl_context = ssl._create_unverified_context() if args.insecure else None
    ws_url = _ws_url(args.base_url)
    history: list[dict[str, Any]] = []
    resume_identity: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    live_session_id: str | None = None
    resumed_session_id: str | None = None

    async with websockets.connect(
        ws_url,
        ssl=ssl_context,
        max_size=128 * 1024 * 1024,
    ) as ws:
        await _wait_queue_done(ws)
        init_frame = {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "checkpoint_profile_id": args.checkpoint_profile_id,
                "system_prompt": "你是一个简洁的语音助手。",
                "tools": [],
                "generate_audio": False,
                "config": {
                    "runtime": "fc_duplex",
                    "non_spoken_scheduling": args.non_spoken_scheduling,
                    "non_spoken_budget_while_listening": (
                        args.non_spoken_budget_while_listening
                    ),
                    "non_spoken_budget_while_speaking": (
                        args.non_spoken_budget_while_speaking
                    ),
                    "max_spoken_tokens": 24,
                    "decode_mode": "greedy",
                    "sample_rate": 16000,
                },
            },
        }
        history.append(init_frame)
        await ws.send(json.dumps(init_frame, ensure_ascii=False))

        while resume_identity is None:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            history.append(event)
            if event.get("type") == "session.created":
                live_session_id = str(event.get("session_id") or "")
                resume_identity = dict(event.get("resume") or {})
            elif event.get("type") == "session.closed":
                raise RuntimeError(f"session closed during init: {event}")
        if not resume_identity:
            raise RuntimeError("session.created did not include resume identity")

        for unit_index in range(args.max_units):
            input_frame = {
                "type": "input.append",
                "input": {
                    "input_id": f"resume_smoke_{unit_index:04d}",
                    "audio_base64": _silent_audio_base64(),
                    "sample_rate": 16000,
                },
            }
            history.append(input_frame)
            await ws.send(json.dumps(input_frame))
            checkpoint = await _collect_until_checkpoint(
                ws,
                history=history,
                target_unit_index=unit_index,
            )
            print(
                f"unit={unit_index} resume={checkpoint.get('resume')}",
                flush=True,
            )
            if checkpoint.get("resume", {}).get("status") == "available":
                break

    if checkpoint is None or checkpoint.get("resume", {}).get("status") != "available":
        raise RuntimeError(
            f"no resumable checkpoint within {args.max_units} Units"
        )

    await asyncio.sleep(1.0)
    through_unit_index = int(checkpoint["unit_index"])
    resume_frame = {
        "type": "session.resume",
        "payload": {
            **resume_identity,
            "through_unit_index": through_unit_index,
            "history": history,
        },
    }
    async with websockets.connect(
        ws_url,
        ssl=ssl_context,
        max_size=128 * 1024 * 1024,
    ) as ws:
        await _wait_queue_done(ws)
        await ws.send(json.dumps(resume_frame, ensure_ascii=False))
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
        if event.get("type") != "session.resumed":
            raise RuntimeError(f"resume failed: {event}")
        if event.get("through_unit_index") != through_unit_index:
            raise RuntimeError(f"resume returned wrong checkpoint: {event}")
        resumed_session_id = str(event.get("session_id") or "")
        print(json.dumps(event, ensure_ascii=False), flush=True)

    if args.trace_dir:
        await asyncio.sleep(1.0)
        if not live_session_id or not resumed_session_id:
            raise RuntimeError("missing live/resumed session ids for trace comparison")
        trace_dir = Path(args.trace_dir)

        def load_trace(session_id: str) -> dict[str, Any]:
            matches = sorted(
                trace_dir.glob(f"fc_trace_{session_id}_*.json"),
                key=lambda path: path.stat().st_mtime,
            )
            if not matches:
                raise RuntimeError(
                    f"missing model trace for {session_id} under {trace_dir}"
                )
            return json.loads(matches[-1].read_text(encoding="utf-8"))

        live_trace = load_trace(live_session_id)
        resumed_trace = load_trace(resumed_session_id)
        for field in ("output_ids", "kv_cache_length", "current_unit_idx"):
            if live_trace.get(field) != resumed_trace.get(field):
                raise RuntimeError(
                    f"trace mismatch for {field}: "
                    f"live={live_trace.get(field)!r}, "
                    f"resumed={resumed_trace.get(field)!r}"
                )
        print(
            json.dumps(
                {
                    "trace_equivalent": True,
                    "output_id_count": len(live_trace.get("output_ids") or []),
                    "kv_cache_length": live_trace.get("kv_cache_length"),
                    "current_unit_idx": live_trace.get("current_unit_idx"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://127.0.0.1:8009")
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument(
        "--checkpoint-profile-id",
        default=os.environ.get("CHECKPOINT_PROFILE_ID"),
    )
    parser.add_argument(
        "--non-spoken-scheduling",
        choices=("quality", "latency"),
        default=os.environ.get("FC_DUPLEX_NON_SPOKEN_SCHEDULING"),
    )
    parser.add_argument(
        "--non-spoken-budget-while-listening",
        type=int,
        default=os.environ.get("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING"),
    )
    parser.add_argument(
        "--non-spoken-budget-while-speaking",
        type=int,
        default=os.environ.get("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING"),
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Optional shared FC trace directory for live/replay equivalence checks.",
    )
    args = parser.parse_args()
    if not args.checkpoint_profile_id:
        parser.error("--checkpoint-profile-id or CHECKPOINT_PROFILE_ID is required")
    if not args.non_spoken_scheduling:
        parser.error("non-spoken scheduling must be provided by Checkpoint Profile")
    if not args.non_spoken_budget_while_listening:
        parser.error("listening budget must be provided by Checkpoint Profile")
    if not args.non_spoken_budget_while_speaking:
        parser.error("speaking budget must be provided by Checkpoint Profile")
    asyncio.run(run_smoke(args))


if __name__ == "__main__":
    main()
