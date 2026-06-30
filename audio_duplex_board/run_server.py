"""Standalone HTTP / WebSocket server for the Audio Duplex Board prototype.

Three entry points:

1. `GET  /`              → standalone web page (web/index.html)
2. `GET  /api/defaults`  → runtime config defaults (used by the page to suggest case path etc.)
3. `WS   /ws/session`    → live board session with async tool dispatch and
                           streaming two-layer non-spoken events. Driven by
                           `AudioDuplexBoardSession` (see session.py).
4. `POST /api/replay-case` → blocking debug endpoint that runs one TrainingData
                             JSON case end-to-end via the legacy
                             `events_from_train_data_result` path. Note this
                             path is synchronous and does NOT exercise the
                             async streaming-two-layer protocol — it is for
                             checkpoint smoke / event log inspection only.

Concurrency model:

- The ws handler is fully async. Each ws frame is processed sequentially by
  one coroutine; tool image searches are spawned as background `asyncio.Task`s
  inside the session. A single asyncio.Lock inside the handler serializes
  `await ws.send_json` calls between the unit-loop coroutine and the
  background tool-task coroutines.
- The model itself is shared across ws connections (single FcDuplexView state),
  so we accept only one ws session at a time. A second connecting client is
  rejected with code 1008.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_duplex_board.config import AudioDuplexBoardConfig, make_default_config
from audio_duplex_board.events import events_from_train_data_result
from audio_duplex_board.mock_view import MockUnifiedProcessor
from audio_duplex_board.schemas import (
    BoardEvent,
    ReplayCaseRequest,
    ReplayCaseResponse,
    StreamAudioChunkRequest,
    StreamFinishRequest,
    StreamPrepareRequest,
)
from audio_duplex_board.session import AudioDuplexBoardSession
from audio_duplex_board.tools.display_object_on_board.service import (
    DisplayObjectOnBoardService,
)
from core.processors import UnifiedProcessor
from core.schemas.fc_duplex import (
    FcDuplexConfig,
    FcDuplexTrainDataRequest,
)


def create_app(config: AudioDuplexBoardConfig) -> FastAPI:
    """Create the standalone FastAPI app."""

    _prepend_sdk_src(config.sdk_src)
    if config.use_mock_view:
        processor = MockUnifiedProcessor(energy_threshold=config.mock_energy_threshold)
    else:
        processor = UnifiedProcessor(
            model_path=config.model_path,
            pt_path=config.pt_path,
            device="cuda",
            compile=False,
            attn_implementation="sdpa",
        )
    tool_service = DisplayObjectOnBoardService()
    web_dir = Path(__file__).resolve().parent / "web"
    live_image_dir = (
        Path(__file__).resolve().parent
        / "tools"
        / "display_object_on_board"
        / "live_image_downloads"
    )
    live_image_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Audio Duplex Board Prototype")
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="audio_duplex_board_static")
    app.mount(
        "/live-image-downloads",
        StaticFiles(directory=str(live_image_dir)),
        name="live_image_downloads",
    )

    # Single-session gate: FcDuplexView is stateful; we do not multiplex.
    ws_session_active = asyncio.Lock()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/api/defaults")
    def defaults() -> dict[str, object]:
        # Extract the default training-aligned system_prompt + ref_audio_path
        # from the first case in the default case folder, so the frontend
        # mic-live flow can send the same prepare request the model was
        # trained on. Without this, prepare() uses a custom prompt + no
        # reference audio, the model is heavily OOD, and the AI never speaks.
        default_system_prompt: str | None = None
        default_ref_audio_path: str | None = None
        default_tools: list[dict[str, object]] | None = None
        default_case_path: str | None = None
        case_folder_path = (
            Path(config.case_folder) if config.case_folder else None
        )
        if case_folder_path and case_folder_path.exists():
            cases = sorted(case_folder_path.glob("*.json"))
            if cases:
                default_case_path = str(cases[0])
                try:
                    extracted = _extract_prepare_defaults_from_case(cases[0])
                    default_system_prompt = extracted.get("system_prompt")
                    default_ref_audio_path = extracted.get("ref_audio_path")
                    default_tools = extracted.get("tools")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[api/defaults] failed to extract prepare defaults "
                        f"from {cases[0]}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
        return {
            "model_path": config.model_path,
            "pt_path": config.pt_path,
            "sdk_src": config.sdk_src,
            "case_folder": config.case_folder,
            "max_board_cards": config.max_board_cards,
            "use_mock_view": config.use_mock_view,
            "mock_energy_threshold": config.mock_energy_threshold,
            "default_case_path": default_case_path,
            "default_system_prompt": default_system_prompt,
            "default_ref_audio_path": default_ref_audio_path,
            "default_tools": default_tools,
        }

    @app.post("/api/replay-case", response_model=ReplayCaseResponse)
    def replay_case(request: ReplayCaseRequest) -> ReplayCaseResponse:
        case_path = Path(request.case_path)
        if not case_path.exists():
            raise HTTPException(status_code=404, detail=f"case_path not found: {case_path}")
        data_root = Path(request.data_root) if request.data_root else case_path.parent
        result = processor.fc_duplex.offline_inference_from_train_data(
            FcDuplexTrainDataRequest(
                train_data_path=str(case_path),
                data_root=str(data_root),
                config=FcDuplexConfig(decode_mode="greedy"),
                generate_audio=request.generate_audio,
                use_train_tool_call_ids=True,
                inject_train_tool_responses=True,
            )
        )
        session_id = request.session_id or f"board-replay-{Path(case_path).stem}"
        events = events_from_train_data_result(
            result=result,
            session_id=session_id,
            tool_service=tool_service,
        )
        summary = {
            "pred_spoken_text": result.pred_spoken_text,
            "pred_think_text": result.pred_think_text,
            "pred_tool_call_count": len(result.pred_tool_calls),
            "token_ids_exact": (
                result.comparison.token_ids_exact if result.comparison else None
            ),
            "tool_calls_semantic_exact": (
                result.comparison.tool_calls_semantic_exact if result.comparison else None
            ),
        }
        return ReplayCaseResponse(
            session_id=session_id,
            sample_id=result.sample_id or case_path.stem,
            success=result.success,
            error=result.error,
            events=events,
            summary=summary,
        )

    @app.websocket("/ws/session")
    async def ws_session(ws: WebSocket) -> None:
        await ws.accept()
        if ws_session_active.locked():
            await ws.close(code=1008, reason="another board session is active")
            return
        async with ws_session_active:
            await _run_ws_session(
                ws=ws,
                processor=processor,
                config=config,
                tool_service=tool_service,
            )

    return app


async def _run_ws_session(
    *,
    ws: WebSocket,
    processor,
    config: AudioDuplexBoardConfig,
    tool_service: DisplayObjectOnBoardService,
) -> None:
    """Run one full ws session lifecycle.

    Send-side synchronization: ws.send_json is not safe to call from multiple
    coroutines concurrently (background tool tasks + main unit loop both try).
    We wrap it with an asyncio.Lock and pass the wrapped send to the session.
    """

    send_lock = asyncio.Lock()

    async def send_event(event: BoardEvent) -> None:
        async with send_lock:
            try:
                await ws.send_json(event.model_dump(mode="json"))
            except RuntimeError:
                # ws may be closed when a tool task finishes after disconnect;
                # swallow so the background task does not raise.
                return

    session = AudioDuplexBoardSession(
        config=config,
        processor=processor,
        send_event=send_event,
        tool_service=tool_service,
    )

    try:
        while True:
            message = await ws.receive_json()
            message_type = str(message.get("type") or "")
            payload = message.get("payload") or {}
            if message_type == "prepare":
                await session.prepare_stream(
                    StreamPrepareRequest.model_validate(payload)
                )
            elif message_type == "audio_chunk":
                await session.process_audio_chunk(
                    StreamAudioChunkRequest.model_validate(payload)
                )
            elif message_type == "finish":
                await session.finish_stream(
                    StreamFinishRequest.model_validate(payload).reason
                )
                break
            else:
                await send_event(
                    BoardEvent(
                        type="session_error",
                        session_id=session.session_id,
                        text=f"unsupported message type: {message_type}",
                    )
                )
    except WebSocketDisconnect:
        # Best effort: still wait for in-flight tool tasks so they emit on a
        # dead socket (silently swallowed by send_event) and clean up state.
        await session.finish_stream(reason="websocket_disconnected")
    except Exception as exc:  # noqa: BLE001
        # Print the full traceback to server stdout (= cctl job log) so we can
        # diagnose what blew up inside session.prepare / process_audio_chunk
        # without relying on browser console.
        import traceback as _tb
        print(
            f"[ws_session_error] session_id={session.session_id} "
            f"exc={type(exc).__name__}: {exc}\n{_tb.format_exc()}",
            flush=True,
        )
        await send_event(
            BoardEvent(
                type="session_error",
                session_id=session.session_id,
                text=f"{type(exc).__name__}: {exc}",
            )
        )
        await session.finish_stream(reason="session_error")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    defaults = make_default_config()
    parser = argparse.ArgumentParser(description="Run standalone Audio Duplex Board prototype")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--pt-path", default=defaults.pt_path)
    parser.add_argument("--sdk-src", default=defaults.sdk_src)
    parser.add_argument("--case-folder", default=defaults.case_folder)
    parser.add_argument("--max-board-cards", type=int, default=defaults.max_board_cards)
    parser.add_argument("--mock", action="store_true", help="Use GPU-free mock FcDuplexView")
    parser.add_argument(
        "--mock-energy-threshold", type=float, default=defaults.mock_energy_threshold
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to TLS private key (PEM). HTTPS is required for getUserMedia mic access on non-localhost URLs.",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to TLS certificate chain (PEM). Pair with --ssl-keyfile to enable HTTPS.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = AudioDuplexBoardConfig(
        model_path=args.model_path,
        pt_path=args.pt_path,
        sdk_src=args.sdk_src,
        case_folder=args.case_folder,
        host=args.host,
        port=args.port,
        max_board_cards=args.max_board_cards,
        use_mock_view=args.mock,
        mock_energy_threshold=args.mock_energy_threshold,
    )
    uvicorn_kwargs: dict[str, object] = {"host": config.host, "port": config.port}
    if args.ssl_keyfile and args.ssl_certfile:
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile
    elif args.ssl_keyfile or args.ssl_certfile:
        raise SystemExit("--ssl-keyfile and --ssl-certfile must be supplied together")
    uvicorn.run(create_app(config), **uvicorn_kwargs)


def _prepend_sdk_src(sdk_src: str | None) -> None:
    if not sdk_src:
        return
    path = str(Path(sdk_src))
    if path not in sys.path:
        sys.path.insert(0, path)


def _extract_prepare_defaults_from_case(case_path: Path) -> dict[str, object]:
    """Read a TrainingData JSON case and pull out the training-aligned
    system_prompt, AI reference audio path, and tool definitions.

    Mirrors the logic FcDuplexView's offline_inference_from_train_data uses
    internally, but exposed as a tiny helper so the live ws path can show
    the same prepare to the model without re-implementing the SDK contract.
    """

    import json as _json

    with case_path.open("r", encoding="utf-8") as fp:
        structure = _json.load(fp)
    data_root = case_path.parent

    system_prompt_parts: list[str] = []
    ref_audio_path: str | None = None
    for segment in (structure.get("system", {}) or {}).get("segments", []) or []:
        kind = segment.get("kind")
        if kind == "text":
            text = segment.get("text") or ""
            if text:
                system_prompt_parts.append(text)
        elif kind == "audio":
            audio = segment.get("audio") or {}
            file_path = audio.get("file_path")
            if file_path:
                candidate = (data_root / file_path).resolve()
                if candidate.exists():
                    ref_audio_path = str(candidate)

    # tools: training case stores them under top-level "tools" if present;
    # display_object_on_board cases tend to encode the tool implicitly via
    # ai_non_spoken tool_call segments, so we fall back to a hardcoded
    # display_object_on_board tool definition matching the training schema.
    tools = structure.get("tools")
    if not tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "display_object_on_board",
                    "description": (
                        "Display a named concrete object on the visual board "
                        "so the user can see it. Use only for concrete, "
                        "visualizable objects mentioned in user speech."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            }
        ]
    return {
        "system_prompt": "\n".join(system_prompt_parts) or None,
        "ref_audio_path": ref_audio_path,
        "tools": tools,
    }


if __name__ == "__main__":
    main()
