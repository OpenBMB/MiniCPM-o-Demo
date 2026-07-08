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
    # 非 spoken slot 的两层 streaming：
    #   non_spoken_block_started  → 前端建一个 think 或 tool_call 的 block 容器（含两层 streaming/full）
    #   non_spoken_delta          → 后端 step 出 token 时，追加到 streaming 层（携带 token_id + token_str）
    #   non_spoken_block_closed   → block 闭合，前端把 full 层填上（携带后端 BPE 整体 decode 的 full_text；
    #                                若 tool_call wire 非法导致 parser 拿不到 full，full_text 为空）
    # 旧的粗粒度 think_final / tool_call_final 保留作为派生事件，便于事件 log 阅读，
    # 但前端两层 UI 的真相源是 *_block_started/_delta/_closed。
    "non_spoken_block_started",
    "non_spoken_delta",
    "non_spoken_block_closed",
    "think_final",
    "tool_call_final",
    # board card 两步异步：先 created（searching 状态），search 完成回填 updated
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
    """One page-facing event emitted by the board prototype.

    新增字段（非 spoken streaming 两层 UI 协议）：

    - `block_id`：non_spoken block 在一个 session 内的稳定 id（格式 `nsb-<unit>-<seq>`），
      `*_block_started` / `*_delta` / `*_block_closed` 必须带同一个 id，前端据此找到容器。
    - `block_kind`：`think` | `tool_call` | `unknown`（首个内容 token 未到时还无法判断；
      允许后端先发 started + delta，等命中已知的 think/tool_call 起始 token id 后判定，
      前端先不渲染 kind 标签）。
    - `token_ids`：本次 step 新增的原始 token id（业务侧用来跟 SDK 协议里已知的
      think/tool_call 起始 id 做精确匹配，见 `session.py::_guess_block_kind_from_token_ids`）。
    - `step_text`：本次 step 后端正常 BPE decode 出的增量文本，是 streaming 层展示的
      唯一数据源（`tokenizer.convert_ids_to_tokens` 反查得到的 id-to-token vocab piece
      对 byte-level BPE 不是合法可读文本，已废弃，不再作为协议字段）。
    - `full_text`：闭合时整体 BPE decode 的文本，前端填到 full 层；如果 tool_call 非法
      或 think 提前截断，可能为空字符串。
    """

    type: BoardEventType
    session_id: str
    unit_index: int | None = None
    text: str | None = None
    think_text: str | None = None
    tool_call: ToolCallView | None = None
    card: BoardCard | None = None
    block_id: str | None = None
    block_kind: Literal["think", "tool_call", "unknown"] | None = None
    token_ids: list[int] = Field(default_factory=list)
    step_text: str | None = None
    full_text: str | None = None
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
    """Prepare a streaming board session.

    Args:
        ref_audio_path: AI voice reference audio path (the same wav that the
            training data's `system.segments[*].kind == "audio"` entry points
            at — e.g. `media/system_reference/HTRef06.wav`). MUST be passed for
            real-model sessions, otherwise the model is OOD relative to its
            training distribution and will likely stay in listen mode forever.
        prompt_wav_path: Optional Token2Wav prompt wav for TTS streaming cache.
            Only meaningful when `generate_audio=True`.
    """

    session_id: str | None = None
    system_prompt: str = ""
    tools: list[dict[str, Any]] | None = None
    ref_audio_path: str | None = None
    prompt_wav_path: str | None = None
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
