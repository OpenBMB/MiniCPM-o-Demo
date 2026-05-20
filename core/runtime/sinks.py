"""Runtime output sinks."""

from __future__ import annotations

import base64
import logging
from typing import Iterable

import numpy as np

from core.runtime.events import OutputSink, RuntimeEvent
from core.runtime.legacy_duplex import LegacyDuplexInput
from core.runtime.legacy_duplex import legacy_result_payload_from_event

logger = logging.getLogger(__name__)


class CompositeSink:
    """Fan out runtime events to multiple sinks in order."""

    def __init__(self, sinks: Iterable[OutputSink]):
        self._sinks = list(sinks)

    async def emit(self, event: RuntimeEvent) -> None:
        for sink in self._sinks:
            await sink.emit(event)


class LegacyDuplexWebSocketSink:
    """Emit runtime duplex events using the legacy /ws/duplex payload shape."""

    def __init__(self, ws):
        self.ws = ws

    async def emit(self, event: RuntimeEvent) -> None:
        await self.ws.send_json(legacy_result_payload_from_event(event))


class DuplexRecordingSink:
    """Persist duplex runtime events with the existing DuplexSessionRecorder."""

    def __init__(self, recorder):
        self.recorder = recorder
        self._chunk_index = 0
        self._pending_input = None

    @property
    def chunk_index(self) -> int:
        return self._chunk_index

    def capture_input(self, legacy_input: LegacyDuplexInput, *, receive_ts_ms: float) -> None:
        """Persist input artifacts and remember them for the next output event."""

        user_audio_rel = self.recorder.save_user_audio(
            self._chunk_index,
            legacy_input.frame.audio_waveform,
        )
        user_frame_rel = None
        if legacy_input.first_frame_bytes is not None:
            user_frame_rel = self.recorder.save_user_frame(
                self._chunk_index,
                legacy_input.first_frame_bytes,
            )

        self._pending_input = {
            "index": self._chunk_index,
            "receive_ts_ms": receive_ts_ms,
            "user_audio_rel": user_audio_rel,
            "user_frame_rel": user_frame_rel,
        }

    async def emit(self, event: RuntimeEvent) -> None:
        if event.channel != "output.duplex_result":
            return
        if self._pending_input is None:
            logger.warning("DuplexRecordingSink received output without pending input")
            return

        frame = event.payload["frame"]
        result = frame.result
        result_dict = frame.result_dict

        ai_audio_rel = None
        ai_audio_n_samples = 0
        if not result.is_listen and result.audio_data:
            try:
                ai_bytes = base64.b64decode(result.audio_data)
                ai_ndarray = np.frombuffer(ai_bytes, dtype=np.float32)
                ai_audio_rel = self.recorder.save_ai_audio(
                    self.recorder.turn_index,
                    self._pending_input["index"],
                    ai_ndarray,
                )
                ai_audio_n_samples = len(ai_ndarray)
            except Exception as exc:
                logger.warning("[SessionRecorder] failed to save AI audio: %s", exc)

        self.recorder.record_chunk(
            index=self._pending_input["index"],
            receive_ts_ms=self._pending_input["receive_ts_ms"],
            result_dict=result_dict,
            prefill_ms=frame.prefill_ms,
            user_audio_rel=self._pending_input["user_audio_rel"],
            user_frame_rel=self._pending_input["user_frame_rel"],
            ai_audio_rel=ai_audio_rel,
            ai_audio_samples=ai_audio_n_samples,
        )
        self._pending_input = None
        self._chunk_index += 1

