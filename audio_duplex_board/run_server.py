"""Standalone HTTP server for the Audio Duplex Board prototype."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_duplex_board.config import AudioDuplexBoardConfig, make_default_config
from audio_duplex_board.schemas import (
    ReplayCaseRequest,
    ReplayCaseResponse,
    StreamAudioChunkRequest,
    StreamFinishRequest,
    StreamPrepareRequest,
)
from audio_duplex_board.session import AudioDuplexBoardSession
from core.processors import UnifiedProcessor


def create_app(config: AudioDuplexBoardConfig) -> FastAPI:
    """Create the standalone FastAPI app.

    Args:
        config: Runtime config.

    Returns:
        FastAPI app serving the static page and replay endpoint.
    """

    _prepend_sdk_src(config.sdk_src)
    processor = UnifiedProcessor(
        model_path=config.model_path,
        pt_path=config.pt_path,
        device="cuda",
        compile=False,
        attn_implementation="sdpa",
    )
    session = AudioDuplexBoardSession(config=config, processor=processor)
    web_dir = Path(__file__).resolve().parent / "web"
    live_image_dir = Path(__file__).resolve().parent / "tools" / "display_object_on_board" / "live_image_downloads"
    live_image_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Audio Duplex Board Prototype")
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="audio_duplex_board_static")
    app.mount("/live-image-downloads", StaticFiles(directory=str(live_image_dir)), name="live_image_downloads")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.post("/api/replay-case", response_model=ReplayCaseResponse)
    def replay_case(request: ReplayCaseRequest) -> ReplayCaseResponse:
        case_path = Path(request.case_path)
        if not case_path.exists():
            raise HTTPException(status_code=404, detail=f"case_path not found: {case_path}")
        return session.replay_case(request)

    @app.get("/api/defaults")
    def defaults() -> dict[str, object]:
        return {
            "model_path": config.model_path,
            "pt_path": config.pt_path,
            "sdk_src": config.sdk_src,
            "case_folder": config.case_folder,
            "max_board_cards": config.max_board_cards,
        }

    @app.websocket("/ws/session")
    async def ws_session(ws: WebSocket) -> None:
        await ws.accept()
        ws_session_obj = AudioDuplexBoardSession(config=config, processor=processor)
        try:
            while True:
                message = await ws.receive_json()
                message_type = str(message.get("type") or "")
                payload = message.get("payload") or {}
                if message_type == "prepare":
                    events = ws_session_obj.prepare_stream(StreamPrepareRequest.model_validate(payload))
                elif message_type == "audio_chunk":
                    events = ws_session_obj.process_audio_chunk(StreamAudioChunkRequest.model_validate(payload))
                elif message_type == "finish":
                    events = ws_session_obj.finish_stream(
                        StreamFinishRequest.model_validate(payload).reason
                    )
                    for event in events:
                        await ws.send_json(event.model_dump(mode="json"))
                    break
                else:
                    await ws.send_json(
                        {
                            "type": "session_error",
                            "session_id": ws_session_obj.session_id,
                            "text": f"unsupported message type: {message_type}",
                        }
                    )
                    continue
                for event in events:
                    await ws.send_json(event.model_dump(mode="json"))
        except WebSocketDisconnect:
            return

    return app


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
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port)


def _prepend_sdk_src(sdk_src: str | None) -> None:
    if not sdk_src:
        return
    path = str(Path(sdk_src))
    if path not in sys.path:
        sys.path.insert(0, path)


if __name__ == "__main__":
    main()
