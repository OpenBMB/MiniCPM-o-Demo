"""Typed event schema for the standalone Audio Duplex Board prototype.

The schema in this file describes the page-facing board event stream. It does
not redefine FC duplex model protocol; model requests and responses continue to
use `core.schemas.fc_duplex`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


BoardEventType = Literal[
    "session_started",
    "unit_started",
    "spoken_final",
    "think_final",
    "tool_call_final",
    "board_card_created",
    "board_card_updated",
    "unit_finished",
    "session_finished",
    "session_error",
]


class BoardImageResult(BaseModel):
    """Image result shown on the board.

    Args:
        query: Search query from `display_object_on_board(name=...)`.
        asset_id: Stable local asset id.
        image_url: Frontend-loadable image URL or data URL.
        source_url: Original image URL when using a real search backend.
        title: Display title.
        elapsed_ms: Search or placeholder generation time.
        error: Optional error message.
    """

    query: str
    asset_id: str
    image_url: str | None = None
    source_url: str | None = None
    title: str = ""
    elapsed_ms: float = 0.0
    error: str | None = None


class BoardCard(BaseModel):
    """A single visual card on the board."""

    card_id: str
    tool_call_id: str | None = None
    query: str
    status: Literal["searching", "ready", "error"] = "searching"
    image: BoardImageResult | None = None
    error: str | None = None


class ToolCallView(BaseModel):
    """Frontend-friendly tool call view."""

    tool_call_id: str | None = None
    name: str | None = None
    arguments: Any = None
    error: str | None = None
    wire: str | None = None


class BoardEvent(BaseModel):
    """One page-facing event emitted by the board prototype."""

    type: BoardEventType
    session_id: str
    unit_index: int | None = None
    text: str | None = None
    think_text: str | None = None
    tool_call: ToolCallView | None = None
    card: BoardCard | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ReplayCaseRequest(BaseModel):
    """Request to replay one TrainingData JSON case."""

    case_path: str = Field(..., description="Path to a TrainingData JSON case")
    data_root: str | None = Field(None, description="Media root; defaults to case parent")
    session_id: str | None = None
    generate_audio: bool = False


class ReplayCaseResponse(BaseModel):
    """Response returned by the blocking replay endpoint."""

    session_id: str
    sample_id: str
    success: bool
    error: str | None = None
    events: list[BoardEvent] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class StreamPrepareRequest(BaseModel):
    """Prepare a streaming board session."""

    session_id: str | None = None
    system_prompt: str = ""
    tools: list[dict[str, Any]] | None = None
    generate_audio: bool = False
    max_spoken_tokens: int = 24
    decode_mode: str = "greedy"


class StreamAudioChunkRequest(BaseModel):
    """One user audio unit sent by the browser."""

    audio_base64: str
    sample_rate: int = 16000


class StreamFinishRequest(BaseModel):
    """Finish the streaming board session."""

    reason: str = "client_finished"
