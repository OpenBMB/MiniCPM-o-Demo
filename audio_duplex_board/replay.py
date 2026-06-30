"""Replay helpers for the independent Audio Duplex Board prototype."""

from __future__ import annotations

from pathlib import Path

from audio_duplex_board.schemas import ReplayCaseRequest, ReplayCaseResponse
from audio_duplex_board.session import AudioDuplexBoardSession


def replay_case_path(
    *,
    session: AudioDuplexBoardSession,
    case_path: Path,
    data_root: Path | None = None,
    generate_audio: bool = False,
) -> ReplayCaseResponse:
    """Replay a TrainingData case path through a board session.

    Args:
        session: Initialized board session.
        case_path: TrainingData JSON path.
        data_root: Optional media root. Defaults to `case_path.parent`.
        generate_audio: Whether to generate TTS waveform.

    Returns:
        Board replay response.
    """

    return session.replay_case(
        ReplayCaseRequest(
            case_path=str(case_path),
            data_root=str(data_root) if data_root else None,
            generate_audio=generate_audio,
        )
    )
