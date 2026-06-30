"""Board prototype session wrapper around `FcDuplexView`."""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from core.processors import UnifiedProcessor
from core.schemas.fc_duplex import (
    FcDuplexConfig,
    FcDuplexPrepareRequest,
    FcDuplexPrefillRequest,
    FcDuplexTrainDataRequest,
    FcNonSpokenGenerateRequest,
    FcSpokenGenerateRequest,
    FcToolResponse,
)

from audio_duplex_board.config import AudioDuplexBoardConfig
from audio_duplex_board.events import events_from_train_data_result
from audio_duplex_board.schemas import (
    BoardCard,
    BoardEvent,
    BoardImageResult,
    ReplayCaseRequest,
    ReplayCaseResponse,
    StreamAudioChunkRequest,
    StreamPrepareRequest,
    ToolCallView,
)
from audio_duplex_board.tools.display_object_on_board.service import (
    DisplayObjectOnBoardService,
)


class AudioDuplexBoardSession:
    """One independent board prototype session.

    Args:
        config: Board prototype runtime config.
        processor: Shared `UnifiedProcessor` instance.
        tool_service: Board tool service.
    """

    def __init__(
        self,
        *,
        config: AudioDuplexBoardConfig,
        processor: UnifiedProcessor,
        tool_service: DisplayObjectOnBoardService | None = None,
    ) -> None:
        self.config = config
        self.processor = processor
        self.fc = processor.fc_duplex
        self.tool_service = tool_service or DisplayObjectOnBoardService()
        self.session_id = f"board-{uuid.uuid4().hex[:10]}"
        self._prepared = False
        self._unit_index = 0
        self._pending_tool_responses: list[FcToolResponse] = []
        self._max_spoken_tokens = 24
        self._decode_mode = "greedy"

    def replay_case(self, request: ReplayCaseRequest) -> ReplayCaseResponse:
        """Replay one TrainingData JSON case and return board events.

        Args:
            request: Replay request with case path and optional data root.

        Returns:
            Board replay response containing event list and summary.
        """

        session_id = request.session_id or f"board-{uuid.uuid4().hex[:10]}"
        case_path = Path(request.case_path)
        data_root = Path(request.data_root) if request.data_root else case_path.parent
        result = self.fc.offline_inference_from_train_data(
            FcDuplexTrainDataRequest(
                train_data_path=str(case_path),
                data_root=str(data_root),
                config=FcDuplexConfig(decode_mode="greedy"),
                generate_audio=request.generate_audio,
                use_train_tool_call_ids=True,
                inject_train_tool_responses=True,
            )
        )
        events = events_from_train_data_result(
            result=result,
            session_id=session_id,
            tool_service=self.tool_service,
        )
        summary = {
            "pred_spoken_text": result.pred_spoken_text,
            "pred_think_text": result.pred_think_text,
            "pred_tool_call_count": len(result.pred_tool_calls),
            "token_ids_exact": result.comparison.token_ids_exact if result.comparison else None,
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

    def prepare_stream(self, request: StreamPrepareRequest) -> list[BoardEvent]:
        """Prepare a streaming board session.

        Args:
            request: Streaming prepare request from the browser.

        Returns:
            Initial board events.
        """

        if request.session_id:
            self.session_id = request.session_id
        self._max_spoken_tokens = request.max_spoken_tokens
        self._decode_mode = request.decode_mode
        self._pending_tool_responses = []
        self._unit_index = 0
        self.fc.prepare(
            FcDuplexPrepareRequest(
                system_prompt=request.system_prompt,
                tools=request.tools,
                generate_audio=request.generate_audio,
            )
        )
        self._prepared = True
        return [
            BoardEvent(
                type="session_started",
                session_id=self.session_id,
                payload={
                    "mode": "stream",
                    "generate_audio": request.generate_audio,
                },
            )
        ]

    def process_audio_chunk(self, request: StreamAudioChunkRequest) -> list[BoardEvent]:
        """Process one browser-sent audio unit.

        Args:
            request: Audio chunk request containing one base64 float32 PCM unit.

        Returns:
            Events produced by this unit.
        """

        if not self._prepared:
            raise RuntimeError("stream session is not prepared")

        responses = list(self._pending_tool_responses)
        self._pending_tool_responses = []
        prefill = self.fc.streaming_prefill(
            FcDuplexPrefillRequest(
                audio_data=request.audio_base64,
                sample_rate=request.sample_rate,
                tool_responses=responses or None,
            )
        )
        unit_index = prefill.unit_index
        events: list[BoardEvent] = [
            BoardEvent(
                type="unit_started",
                session_id=self.session_id,
                unit_index=unit_index,
                payload={
                    "n_audio": prefill.n_audio_placeholders,
                    "tool_response_count": len(responses),
                },
            )
        ]

        spoken = self.fc.streaming_spoken_generate(
            FcSpokenGenerateRequest(
                max_tokens=self._max_spoken_tokens,
                decode_mode=self._decode_mode,
            )
        )
        if spoken.is_speaking or spoken.spoken_token_ids:
            audio_wav_base64 = _audio_waveform_to_wav_base64(
                spoken.audio_waveform,
                spoken.audio_sample_rate or 24000,
            )
            events.append(
                BoardEvent(
                    type="spoken_final",
                    session_id=self.session_id,
                    unit_index=unit_index,
                    text=spoken.spoken_text,
                    payload={
                        "token_ids": list(spoken.spoken_token_ids),
                        "spoken_turn_eos": spoken.spoken_turn_eos,
                        "audio_wav_base64": audio_wav_base64,
                        "audio_sample_rate": spoken.audio_sample_rate,
                    },
                )
            )

        unit_budget = 15 if spoken.is_speaking else 30
        terminated = False
        for _ in range(unit_budget):
            step = self.fc.streaming_non_spoken_generate(
                FcNonSpokenGenerateRequest(max_tokens=1, decode_mode=self._decode_mode)
            )
            events.extend(self._events_from_closed_spans(step.closed_spans, unit_index))
            if step.terminated:
                terminated = True
                break
        if not terminated:
            close_step = self.fc.streaming_non_spoken_generate(
                FcNonSpokenGenerateRequest(
                    max_tokens=0,
                    decode_mode=self._decode_mode,
                    close_reason="budget_reached",
                )
            )
            events.extend(self._events_from_closed_spans(close_step.closed_spans, unit_index))

        unit = self.fc.finalize_unit()
        events.append(
            BoardEvent(
                type="unit_finished",
                session_id=self.session_id,
                unit_index=unit.unit,
                payload={
                    "is_listen": unit.is_listen,
                    "is_speaking": unit.is_speaking,
                    "non_spoken_terminator": unit.non_spoken_terminator,
                    "closed_span_count": len(unit.closed_spans),
                },
            )
        )
        self._unit_index = unit_index + 1
        return events

    def finish_stream(self, reason: str = "client_finished") -> list[BoardEvent]:
        """Finish the streaming session and emit a final event."""

        self._prepared = False
        return [
            BoardEvent(
                type="session_finished",
                session_id=self.session_id,
                payload={"reason": reason, "unit_count": self._unit_index},
            )
        ]

    def _events_from_closed_spans(self, spans: list[object], unit_index: int) -> list[BoardEvent]:
        events: list[BoardEvent] = []
        for span in spans:
            span_type = getattr(span, "type", None)
            if span_type == "think":
                events.append(
                    BoardEvent(
                        type="think_final",
                        session_id=self.session_id,
                        unit_index=unit_index,
                        think_text=getattr(span, "text", "") or "",
                    )
                )
            elif span_type == "tool_call":
                tool_call = self._tool_call_view(span)
                events.append(
                    BoardEvent(
                        type="tool_call_final",
                        session_id=self.session_id,
                        unit_index=unit_index,
                        tool_call=tool_call,
                    )
                )
                events.extend(self._board_events_from_tool_call(tool_call, unit_index))
        return events

    def _tool_call_view(self, span: object) -> ToolCallView:
        raw_tool_call = getattr(span, "tool_call", None) or {}
        return ToolCallView(
            tool_call_id=getattr(span, "tool_call_id", None),
            name=raw_tool_call.get("name") if isinstance(raw_tool_call, dict) else None,
            arguments=raw_tool_call.get("arguments") if isinstance(raw_tool_call, dict) else None,
            error=(getattr(span, "error", None) or raw_tool_call.get("error"))
            if isinstance(raw_tool_call, dict)
            else getattr(span, "error", None),
            wire=getattr(span, "wire", None),
        )

    def _board_events_from_tool_call(
        self,
        tool_call: ToolCallView,
        unit_index: int,
    ) -> list[BoardEvent]:
        query = self._query_from_tool_call(tool_call)
        card_id = f"card:{tool_call.tool_call_id or query}"
        created = BoardCard(
            card_id=card_id,
            tool_call_id=tool_call.tool_call_id,
            query=query,
            status="searching",
        )
        result = self.tool_service.search(query)
        if tool_call.tool_call_id:
            self._pending_tool_responses.append(
                FcToolResponse(
                    call_id=tool_call.tool_call_id,
                    content=result.tool_response_content,
                )
            )
        image = BoardImageResult(
            query=query,
            asset_id=f"tool:{tool_call.tool_call_id or query}",
            image_url=result.image_url,
            source_url=result.source_url,
            title=result.title or query,
            elapsed_ms=result.elapsed_ms,
            error=result.error,
        )
        updated = created.model_copy(
            update={
                "status": "error" if result.error else "ready",
                "image": image,
                "error": result.error,
            }
        )
        return [
            BoardEvent(
                type="board_card_created",
                session_id=self.session_id,
                unit_index=unit_index,
                card=created,
            ),
            BoardEvent(
                type="board_card_updated",
                session_id=self.session_id,
                unit_index=unit_index,
                card=updated,
            ),
        ]

    @staticmethod
    def _query_from_tool_call(tool_call: ToolCallView) -> str:
        args = tool_call.arguments
        if isinstance(args, dict):
            value = args.get("name") or args.get("query")
            if value is not None:
                return str(value)
        return tool_call.name or "unknown object"


def _audio_waveform_to_wav_base64(audio_waveform: object, sample_rate: int) -> str | None:
    """Encode a generated waveform as base64 wav for browser playback."""

    if audio_waveform is None:
        return None
    array = np.asarray(audio_waveform, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return None
    buffer = io.BytesIO()
    sf.write(buffer, array, sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
