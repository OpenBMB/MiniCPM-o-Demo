"""Board prototype session wrapper around `FcDuplexView`."""

from __future__ import annotations

import uuid
from pathlib import Path

from core.processors import UnifiedProcessor
from core.schemas.fc_duplex import FcDuplexConfig, FcDuplexTrainDataRequest

from audio_duplex_board.config import AudioDuplexBoardConfig
from audio_duplex_board.events import events_from_train_data_result
from audio_duplex_board.schemas import ReplayCaseRequest, ReplayCaseResponse
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
