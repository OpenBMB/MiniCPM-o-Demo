"""FcDuplexCapability deterministic Unit replay 的无 GPU 编排测试。"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

from MiniCPMO45.fc_duplex_capability import FcDuplexCapability


def test_replay_completed_unit_feeds_expected_slot_skeleton_without_sampling() -> None:
    """Safe Unit replay 应按 live 模板 feed 历史输出并闭合 Unit。"""

    capability = object.__new__(FcDuplexCapability)
    capability.K = SimpleNamespace(
        LISTEN="listen",
        SPEAK="speak",
        AI_SPOKEN_SLOT_END="ai_spoken_slot_end",
        AI_NON_SPOKEN_SLOT_END="ai_non_spoken_slot_end",
        UNIT_END="unit_end",
        NO_ACTION="no_action",
        NON_SPOKEN_EOS="non_spoken_eos",
        NON_SPOKEN_HOLD="non_spoken_hold",
        NON_SPOKEN_ABORT="non_spoken_abort",
    )
    token_ids = {
        "listen": 1,
        "speak": 2,
        "ai_spoken_slot_end": 3,
        "ai_non_spoken_slot_end": 4,
        "unit_end": 5,
        "no_action": 6,
        "non_spoken_eos": 7,
        "non_spoken_hold": 8,
        "non_spoken_abort": 9,
        "ai_non_spoken_slot_start": 10,
    }
    capability._current_unit_idx = 0
    capability._current_unit_info = None
    capability._spoken_slot_open = False
    capability._non_spoken_slot_open = False
    capability._spoken_logits = None
    capability._non_spoken_logits = None
    capability._non_spoken_mode = None
    capability.fed_ids = []

    def sid(self: Any, key: str) -> int:
        return token_ids[key]

    def streaming_prefill(self: Any, **_: Any) -> None:
        self._current_unit_info = {
            "unit": self._current_unit_idx,
            "spoken_ids": [],
            "non_spoken_ids": [],
            "closed_spans": [],
        }
        self._spoken_slot_open = True

    def feed_ids(self: Any, ids: list[int]) -> None:
        self.fed_ids.extend(ids)

    def open_non_spoken(self: Any) -> None:
        self.fed_ids.append(token_ids["ai_non_spoken_slot_start"])
        self._non_spoken_slot_open = True

    def track_non_spoken(self: Any, _: int) -> list[Any]:
        return []

    def mark_finalized(self: Any, _: dict[str, Any]) -> None:
        self._current_unit_idx += 1
        self._current_unit_info = None

    capability.sid = MethodType(sid, capability)
    capability.streaming_prefill = MethodType(streaming_prefill, capability)
    capability._feed_ids = MethodType(feed_ids, capability)
    capability._open_non_spoken_slot = MethodType(open_non_spoken, capability)
    capability._track_non_spoken_token = MethodType(track_non_spoken, capability)
    capability._mark_unit_finalized = MethodType(mark_finalized, capability)
    capability._record_trace = MethodType(lambda self, *args, **kwargs: None, capability)

    info = capability.replay_completed_unit(
        spoken_token_ids=[token_ids["listen"]],
        non_spoken_token_ids=[token_ids["no_action"]],
    )

    assert capability.fed_ids == [
        token_ids["listen"],
        token_ids["ai_spoken_slot_end"],
        token_ids["ai_non_spoken_slot_start"],
        token_ids["no_action"],
        token_ids["ai_non_spoken_slot_end"],
        token_ids["unit_end"],
    ]
    assert info["is_listen"] is True
    assert info["non_spoken_terminator"] == "no_action"
    assert capability._current_unit_idx == 1
