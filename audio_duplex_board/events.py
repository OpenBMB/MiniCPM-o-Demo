"""Convert FC duplex offline replay results into board prototype events.

Scope: this path is used by `POST /api/replay-case` for one-shot debug replay
of TrainingData JSON cases. It builds a flat event list (no streaming-two-layer
`non_spoken_block_*` events) because offline_inference_from_train_data is a
synchronous batch call — there is no token-by-token streaming to surface.

The live ws path (`AudioDuplexBoardSession`) is the source of truth for the
async streaming-two-layer protocol; see session.py.
"""

from __future__ import annotations

from typing import Any

from core.schemas.fc_duplex import FcDecodedToolCall, FcDuplexTrainDataResult

from audio_duplex_board.schemas import BoardCard, BoardEvent, ToolCallView
from audio_duplex_board.tools.display_object_on_board.service import (
    DisplayObjectOnBoardService,
    board_image_result_from_tool_result,
)


def events_from_train_data_result(
    *,
    result: FcDuplexTrainDataResult,
    session_id: str,
    tool_service: DisplayObjectOnBoardService,
) -> list[BoardEvent]:
    """Build page-facing board events from an FC train-data replay result.

    Args:
        result: Output returned by `FcDuplexView.offline_inference_from_train_data`.
        session_id: Board session id.
        tool_service: Tool service used to enrich decoded tool calls with board
            image cards.

    Returns:
        Ordered event list for the prototype page.
    """

    events: list[BoardEvent] = [
        BoardEvent(
            type="session_started",
            session_id=session_id,
            payload={
                "sample_id": result.sample_id,
                "source_path": result.source_path,
            },
        )
    ]
    if not result.success:
        events.append(
            BoardEvent(
                type="session_error",
                session_id=session_id,
                text=result.error,
            )
        )
        return events

    for unit in result.units_info:
        events.append(
            BoardEvent(
                type="unit_started",
                session_id=session_id,
                unit_index=unit.unit,
                payload={
                    "n_audio": unit.n_audio,
                    "is_listen": unit.is_listen,
                    "is_speaking": unit.is_speaking,
                },
            )
        )
        if unit.is_speaking and unit.spoken_ids:
            events.append(
                BoardEvent(
                    type="spoken_final",
                    session_id=session_id,
                    unit_index=unit.unit,
                    text="",
                    payload={"token_ids": list(unit.spoken_ids)},
                )
            )
        for span in unit.closed_spans:
            if span.type == "think":
                events.append(
                    BoardEvent(
                        type="think_final",
                        session_id=session_id,
                        unit_index=unit.unit,
                        think_text=span.text or "",
                    )
                )
            elif span.type == "tool_call":
                tool_call = _tool_call_view(span)
                events.append(
                    BoardEvent(
                        type="tool_call_final",
                        session_id=session_id,
                        unit_index=unit.unit,
                        tool_call=tool_call,
                    )
                )
                card = _card_from_tool_call(
                    tool_call=tool_call,
                    tool_service=tool_service,
                )
                events.append(
                    BoardEvent(
                        type="board_card_created",
                        session_id=session_id,
                        unit_index=unit.unit,
                        card=card.model_copy(update={"status": "searching", "image": None}),
                    )
                )
                events.append(
                    BoardEvent(
                        type="board_card_updated",
                        session_id=session_id,
                        unit_index=unit.unit,
                        card=card,
                    )
                )
        events.append(
            BoardEvent(
                type="unit_finished",
                session_id=session_id,
                unit_index=unit.unit,
                payload={
                    "non_spoken_terminator": unit.non_spoken_terminator,
                    "closed_span_count": len(unit.closed_spans),
                },
            )
        )

    events.append(
        BoardEvent(
            type="session_finished",
            session_id=session_id,
            payload={
                "success": result.success,
                "spoken_text": result.pred_spoken_text,
                "think_text": result.pred_think_text,
                "tool_call_count": len(result.pred_tool_calls),
                "total_units": len(result.units_info),
            },
        )
    )
    return events


def _tool_call_view(span: Any) -> ToolCallView:
    raw_tool_call = span.tool_call or {}
    return ToolCallView(
        tool_call_id=span.tool_call_id,
        name=raw_tool_call.get("name"),
        arguments=raw_tool_call.get("arguments"),
        error=span.error or raw_tool_call.get("error"),
        wire=span.wire,
    )


def _card_from_tool_call(
    *,
    tool_call: ToolCallView,
    tool_service: DisplayObjectOnBoardService,
) -> BoardCard:
    query = _query_from_tool_call(tool_call)
    result = tool_service.search(query)
    image = board_image_result_from_tool_result(result, tool_call_id=tool_call.tool_call_id)
    return BoardCard(
        card_id=f"card:{tool_call.tool_call_id or query}",
        tool_call_id=tool_call.tool_call_id,
        query=query,
        status="error" if result.error else "ready",
        image=image,
        error=result.error,
    )


def _query_from_tool_call(tool_call: ToolCallView) -> str:
    args = tool_call.arguments
    if isinstance(args, dict):
        value = args.get("name") or args.get("query")
        if value is not None:
            return str(value)
    return tool_call.name or "unknown object"
