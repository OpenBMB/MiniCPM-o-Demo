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
        NON_SPOKEN_BUDGET_REACHED="non_spoken_budget_reached",
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
        "non_spoken_budget_reached": 11,
    }
    capability._current_unit_idx = 0
    capability._current_unit_info = None
    capability._spoken_slot_open = False
    capability._non_spoken_slot_open = False
    capability._spoken_logits = None
    capability._non_spoken_logits = None
    capability._non_spoken_mode = None
    capability._pending_prefill_close_ids = []
    capability._pending_prefill_unit_info = None
    capability._think_buf = []
    capability._tool_call_buf = []
    capability.output_ids = []
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

    capability.replay_completed_unit(
        spoken_token_ids=[token_ids["listen"]],
        non_spoken_token_ids=[token_ids["non_spoken_budget_reached"]],
        deferred_non_spoken_close=True,
    )

    assert capability._pending_prefill_close_ids == [
        token_ids["non_spoken_budget_reached"],
        token_ids["ai_non_spoken_slot_end"],
        token_ids["unit_end"],
    ]
    assert capability.output_ids == capability._pending_prefill_close_ids


def test_tool_call_semantic_buffer_survives_budget_boundary() -> None:
    """Budget 只终止 View decoder，不能丢失 capability 的完整 tool XML。"""

    capability = object.__new__(FcDuplexCapability)
    capability.tool_format = "minicpm4_xml"
    capability._registry = None
    capability._serializer = None
    capability._sdk_tokenizer = None
    capability._normalize_tool_response_content = None
    capability._tools = [
        {
            "type": "function",
            "function": {
                "name": "display_object_on_board",
                "description": "Display an object.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
    ]
    capability._non_spoken_mode = None
    capability._think_buf = []
    capability._tool_call_buf = []
    capability._non_spoken_slot_open = True
    capability._pending_prefill_close_ids = []
    capability._current_unit_info = {
        "non_spoken_ids": [],
        "closed_spans": [],
    }
    capability.output_ids = []
    capability._record_trace = MethodType(
        lambda self, *args, **kwargs: None,
        capability,
    )
    capability._ensure_protocol()

    wire = (
        '<function name="display_object_on_board">'
        '<param name="name">小白兔</param>'
        "</function>"
    )
    wire_ids = capability.encode_text(wire)
    split_at = len(wire_ids) // 2
    capability._track_non_spoken_token(
        capability.sid(capability.K.TOOL_CALL_START)
    )
    for token_id in wire_ids[:split_at]:
        capability._track_non_spoken_token(token_id)

    capability._close_non_spoken_slot(
        "budget_reached",
        defer_feed=True,
    )
    capability._non_spoken_slot_open = True
    for token_id in wire_ids[split_at:]:
        capability._track_non_spoken_token(token_id)
    closed = capability._track_non_spoken_token(
        capability.sid(capability.K.TOOL_CALL_END)
    )

    assert len(closed) == 1
    assert closed[0]["wire"] == wire
    assert closed[0]["tool_call"]["error"] is None
    assert closed[0]["tool_call"]["name"] == "display_object_on_board"
