"""FC Duplex 公共事件历史的 stateless resume canonicalizer。

本模块只负责把公共双向历史校验并转换成 Unit 级 replay plan：

- safe text delta 使用同 target SDK tokenizer 逐项重新 encode；
- protocol semantic key 映射回固定 token ID；
- generation step 与 Unit 归属保持不变；
- pending、缺失历史、target 错配和 round-trip 错误全部 fail-fast。

模型/KV 的 deterministic feed 由 FC Duplex capability 执行，不在本模块中实现。
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from minicpm_o5_sdk import O5SpecialTokenKey, O5TokenizerID, load_builtin_tokenizer
from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenRegistry


FcResumeFailureCode = Literal[
    "non_resumable_text_boundary",
    "incomplete_event_history",
    "model_or_tokenizer_mismatch",
    "text_delta_roundtrip_mismatch",
    "unsupported_open_span",
    "unsupported_spoken_turn_state",
    "unsupported_deferred_close",
    "unsupported_tool_state",
]


class FcDuplexResumeError(RuntimeError):
    """FC Duplex stateless resume 历史无法安全恢复。"""

    def __init__(
        self,
        code: FcResumeFailureCode,
        message: str,
        *,
        unit_index: int | None = None,
        stream_id: str | None = None,
        pending_from_step: int | None = None,
    ) -> None:
        """保存机器可读失败码与边界上下文。"""

        super().__init__(message)
        self.code = code
        self.unit_index = unit_index
        self.stream_id = stream_id
        self.pending_from_step = pending_from_step


class FcDuplexResumeUnitPlan(BaseModel):
    """一个已完成 Unit 的 deterministic replay 输入。"""

    unit_index: int = Field(..., ge=0)
    input_payload: dict[str, Any]
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
    spoken_token_ids: list[int] = Field(default_factory=list)
    non_spoken_token_ids: list[int] = Field(default_factory=list)
    deferred_non_spoken_close: bool = False


class FcDuplexResumePlan(BaseModel):
    """从公共事件历史恢复到目标 Unit 所需的完整 replay plan。"""

    protocol_version: str
    model: str
    tokenizer_target: Literal["o45_fc", "o5"]
    through_unit_index: int
    next_event_index: int
    next_step_index: int
    next_batch_index: int
    next_stream_sequence: int
    next_delta_index_by_stream: dict[str, int]
    seen_input_ids: list[str]
    tool_call_count: int = 0
    api_tool_call_sequence: int = 0
    session_init_payload: dict[str, Any]
    units: list[FcDuplexResumeUnitPlan]


@dataclass
class _StepSlot:
    """一个 public generation step 对应的待恢复 token 槽位。"""

    step_index: int
    unit_index: int
    stream_id: str
    track: Literal["spoken", "non_spoken"]
    token_id: int | None = None
    deferred_model_feed: bool = False


def _resume_error(
    code: FcResumeFailureCode,
    message: str,
    *,
    unit_index: int | None = None,
    stream_id: str | None = None,
    pending_from_step: int | None = None,
) -> FcDuplexResumeError:
    """构造带结构上下文的 resume 错误。"""

    return FcDuplexResumeError(
        code,
        message,
        unit_index=unit_index,
        stream_id=stream_id,
        pending_from_step=pending_from_step,
    )


def _extract_replay_audio_base64(input_payload: dict[str, Any]) -> str | None:
    """Extract a supported audio payload shape for deterministic replay validation."""

    for key in ("audio_base64", "audio_data"):
        value = input_payload.get(key)
        if isinstance(value, str) and value:
            return value
    audio = input_payload.get("audio")
    if isinstance(audio, str) and audio:
        return audio
    if isinstance(audio, dict):
        value = (
            audio.get("data")
            or audio.get("base64")
            or audio.get("audio_base64")
        )
        if isinstance(value, str) and value:
            return value
    return None


def _validate_replay_audio(input_payload: dict[str, Any], *, input_id: str) -> None:
    """Fail fast unless input contains non-empty, valid float32 PCM bytes."""

    audio_base64 = _extract_replay_audio_base64(input_payload)
    if audio_base64 is None:
        raise _resume_error(
            "incomplete_event_history",
            f"input.append 缺少可重放音频: input_id={input_id}",
        )
    try:
        raw_audio = base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise _resume_error(
            "incomplete_event_history",
            f"input.append 音频不是合法 base64: input_id={input_id}",
        ) from exc
    if not raw_audio or len(raw_audio) % 4 != 0:
        raise _resume_error(
            "incomplete_event_history",
            f"input.append 音频不是非空 float32 PCM: input_id={input_id}, "
            f"n_bytes={len(raw_audio)}",
        )


def _tool_result_content(contents: Any) -> str:
    """Normalize public tool-result contents exactly like the live runtime."""

    if not isinstance(contents, list):
        return json.dumps(contents, ensure_ascii=False)
    parts: list[str] = []
    for item in contents:
        if isinstance(item, dict):
            if item.get("kind") == "text" or "text" in item:
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))
    return "".join(parts)


def build_fc_duplex_resume_plan(
    *,
    protocol_version: str,
    model: str,
    tokenizer_target: Literal["o45_fc", "o5"],
    tokenizer_fingerprint: dict[str, str],
    through_unit_index: int,
    history: list[dict[str, Any]],
) -> FcDuplexResumePlan:
    """把公共双向事件历史转换成可 deterministic feed 的 Unit replay plan。

    参数:
        protocol_version: Resume 协议版本；当前只支持
            ``fc-duplex-resume-v1``。
        model: 调用方声明的模型标识。
        tokenizer_target: 当前模型对应的 SDK tokenizer target。
        tokenizer_fingerprint: `session.created.resume` 返回的 tokenizer 指纹。
        through_unit_index: 目标可恢复 Unit。
        history: 从 Session 开始到目标 checkpoint 的完整双向事件列表。

    返回:
        经过连续性、checkpoint 与 text round-trip 校验的 replay plan。

    异常:
        FcDuplexResumeError: 历史不完整、目标边界不可恢复、target 错配或
            text delta 无法精确映射回 generation steps。
    """

    if protocol_version == "fc-duplex-semantic-v2":
        return _build_fc_duplex_semantic_v2_resume_plan(
            model=model,
            tokenizer_target=tokenizer_target,
            tokenizer_fingerprint=tokenizer_fingerprint,
            through_unit_index=through_unit_index,
            history=history,
        )
    if protocol_version != "fc-duplex-resume-v1":
        raise _resume_error(
            "model_or_tokenizer_mismatch",
            f"unsupported protocol_version: {protocol_version}",
        )
    if through_unit_index < 0:
        raise _resume_error(
            "incomplete_event_history",
            f"through_unit_index 必须 >= 0: {through_unit_index}",
        )
    if (
        not history
        or not isinstance(history[0], dict)
        or history[0].get("type") != "session.init"
    ):
        raise _resume_error(
            "incomplete_event_history",
            "resume history 必须从 session.init 开始",
        )

    session_init_payload = dict(history[0].get("payload") or {})
    history_target = session_init_payload.get("tokenizer_target")
    if history_target is not None and history_target != tokenizer_target:
        raise _resume_error(
            "model_or_tokenizer_mismatch",
            f"tokenizer target mismatch: history={history_target}, request={tokenizer_target}",
        )

    tokenizer = load_builtin_tokenizer(O5TokenizerID(tokenizer_target))
    expected_fingerprint = {
        "vocab_hash": tokenizer.fingerprint.vocab_hash,
        "merges_hash": tokenizer.fingerprint.merges_hash,
    }
    if tokenizer_fingerprint != expected_fingerprint:
        raise _resume_error(
            "model_or_tokenizer_mismatch",
            "tokenizer fingerprint mismatch: "
            f"history={tokenizer_fingerprint}, current={expected_fingerprint}",
        )
    registry = O5SpecialTokenRegistry.from_tokenizer(tokenizer)
    step_slots: dict[int, _StepSlot] = {}
    delta_index_by_stream: dict[str, int] = {}
    input_payload_by_id: dict[str, dict[str, Any]] = {}
    checkpoint_by_unit: dict[int, dict[str, Any]] = {}
    track_phase_by_unit: dict[int, Literal["spoken", "non_spoken"]] = {}
    tool_events_by_unit: dict[int, list[dict[str, Any]]] = {}
    pending_tool_started_events: list[dict[str, Any]] = []
    pending_tool_error_events: list[dict[str, Any]] = []
    pending_tool_response_events: list[dict[str, Any]] = []
    api_to_internal_tool_call_id: dict[str, str] = {}
    pending_valid_tool_call_ids: set[str] = set()
    completed_tool_result_ids: set[str] = set()
    tool_call_count = 0
    maximum_api_tool_call_sequence = 0
    expected_event_index = 0
    expected_batch_index = 0
    expected_step_index = 0
    last_step_unit_index = -1
    maximum_batch_index = -1
    maximum_stream_sequence = 0
    active_spoken_stream_id: str | None = None
    pending_spoken_slot_eos_stream_id: str | None = None
    active_non_spoken_stream: tuple[str, str] | None = None
    pending_non_spoken_continuation_kind: str | None = None
    target_checkpoint: dict[str, Any] | None = None

    for event in history[1:]:
        if not isinstance(event, dict):
            raise _resume_error(
                "incomplete_event_history",
                f"resume history event 必须是 object: {event!r}",
            )
        event_type = str(event.get("type") or "")
        if event_type.startswith("response.tool_call"):
            api_sequence_match = re.fullmatch(
                r"tc_(\d+)",
                str(event.get("tool_call_id") or ""),
            )
            if api_sequence_match is not None:
                maximum_api_tool_call_sequence = max(
                    maximum_api_tool_call_sequence,
                    int(api_sequence_match.group(1)),
                )
        if event_type == "response.tool_call.args.raw":
            api_call_id = str(event.get("tool_call_id") or "")
            if (
                not api_call_id
                or api_call_id in api_to_internal_tool_call_id
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    f"tool_call raw 缺少唯一 id: {api_call_id!r}",
                )
            tool_call_count += 1
            internal_call_id = (
                f"fc_call_{tool_call_count:06d}"
            )
            api_to_internal_tool_call_id[api_call_id] = internal_call_id
            pending_tool_started_events.append(
                {
                    "type": "tool_started",
                    "call_id": internal_call_id,
                }
            )
            raw = dict(event.get("raw") or {})
            if raw.get("error"):
                pending_tool_error_events.append(
                    {
                        "type": "tool_response",
                        "call_id": internal_call_id,
                        "content": (
                            "工具调用解析失败，无法执行该工具调用。错误信息："
                            + str(raw["error"])
                        ),
                    }
                )
            else:
                pending_valid_tool_call_ids.add(api_call_id)
            continue
        if event_type in {
            "input.tool_result",
            "input.tool_result.delta",
            "input.tool_result.done",
        }:
            if event_type != "input.tool_result":
                raise _resume_error(
                    "unsupported_tool_state",
                    "当前 resume 仅支持完整 input.tool_result，"
                    "不支持 streaming tool result",
                )
            api_call_id = str(event.get("tool_call_id") or "")
            response_internal_call_id = api_to_internal_tool_call_id.get(
                api_call_id
            )
            if (
                response_internal_call_id is None
                or api_call_id not in pending_valid_tool_call_ids
                or api_call_id in completed_tool_result_ids
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    f"tool result 无匹配 pending call: {api_call_id!r}",
                )
            pending_tool_response_events.append(
                {
                    "type": "tool_response",
                    "call_id": response_internal_call_id,
                    "content": _tool_result_content(event.get("contents")),
                }
            )
            pending_valid_tool_call_ids.remove(api_call_id)
            completed_tool_result_ids.add(api_call_id)
            continue
        if event_type == "input.append":
            input_payload = dict(event.get("input") or {})
            input_id = str(input_payload.get("input_id") or "")
            if not input_id or input_id in input_payload_by_id:
                raise _resume_error(
                    "incomplete_event_history",
                    f"input.append 缺少唯一 input_id: {input_id!r}",
                )
            _validate_replay_audio(input_payload, input_id=input_id)
            input_payload_by_id[input_id] = input_payload
            continue
        if event_type not in {
            "response.unit.input_events",
            "response.generation.step_batch",
            "response.unit.committed",
        }:
            continue

        event_index = event.get("event_index")
        if event_index != expected_event_index:
            raise _resume_error(
                "incomplete_event_history",
                f"event_index 不连续: expected={expected_event_index}, actual={event_index}",
            )
        expected_event_index += 1

        if event_type == "response.unit.input_events":
            unit_index = int(event.get("unit_index", -1))
            if unit_index in tool_events_by_unit:
                raise _resume_error(
                    "incomplete_event_history",
                    f"重复 response.unit.input_events: unit={unit_index}",
                )
            canonical_events: list[dict[str, Any]] = []
            for public_event in list(event.get("events") or []):
                if not isinstance(public_event, dict):
                    raise _resume_error(
                        "incomplete_event_history",
                        f"unit input event 必须是 object: {public_event!r}",
                    )
                api_call_id = str(public_event.get("tool_call_id") or "")
                attributed_internal_call_id = api_to_internal_tool_call_id.get(
                    api_call_id
                )
                if attributed_internal_call_id is None:
                    raise _resume_error(
                        "incomplete_event_history",
                        f"unit input event 引用未知 tool_call_id: {api_call_id}",
                    )
                canonical_event = {
                    "type": str(
                        public_event.get("type") or "tool_response"
                    ),
                    "call_id": attributed_internal_call_id,
                }
                if "content" in public_event:
                    canonical_event["content"] = public_event["content"]
                canonical_events.append(canonical_event)
            expected_tool_events = [
                *pending_tool_started_events,
                *pending_tool_error_events,
                *pending_tool_response_events,
            ]
            if canonical_events != expected_tool_events:
                raise _resume_error(
                    "incomplete_event_history",
                    "response.unit.input_events 与 public tool history 不一致",
                )
            tool_events_by_unit[unit_index] = canonical_events
            pending_tool_started_events = []
            pending_tool_error_events = []
            pending_tool_response_events = []
            continue

        if event_type == "response.unit.committed":
            unit_index = int(event.get("unit_index", -1))
            if unit_index != len(checkpoint_by_unit):
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit checkpoint 不连续: expected={len(checkpoint_by_unit)}, "
                    f"actual={unit_index}",
                )
            if int(event.get("last_step_index", -1)) != expected_step_index - 1:
                raise _resume_error(
                    "incomplete_event_history",
                    f"checkpoint last_step_index 不匹配: "
                    f"expected={expected_step_index - 1}, "
                    f"actual={event.get('last_step_index')}",
                )
            checkpoint_by_unit[unit_index] = event
            if unit_index == through_unit_index:
                target_checkpoint = event
                break
            continue

        stream_id = str(event.get("stream_id") or "")
        track = event.get("track")
        batch_index = int(event.get("batch_index", -1))
        if batch_index != expected_batch_index:
            raise _resume_error(
                "incomplete_event_history",
                f"batch_index 不连续: expected={expected_batch_index}, "
                f"actual={batch_index}",
            )
        expected_batch_index += 1
        stream_match = re.search(r"_(\d+)$", stream_id)
        if stream_match is not None:
            maximum_stream_sequence = max(
                maximum_stream_sequence,
                int(stream_match.group(1)),
            )
        maximum_batch_index = max(
            maximum_batch_index,
            int(event.get("batch_index", -1)),
        )
        if not stream_id or track not in {"spoken", "non_spoken"}:
            raise _resume_error(
                "incomplete_event_history",
                f"generation batch 缺少合法 stream_id/track: {event}",
            )

        for raw_step in list(event.get("steps") or []):
            if not isinstance(raw_step, dict):
                raise _resume_error(
                    "incomplete_event_history",
                    f"generation step 必须是 object: {raw_step!r}",
                )
            step_index = int(raw_step.get("step_index", -1))
            unit_index = int(raw_step.get("unit_index", -1))
            if (
                unit_index not in tool_events_by_unit
                and (
                    pending_tool_started_events
                    or pending_tool_error_events
                    or pending_tool_response_events
                )
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit {unit_index} 缺少 response.unit.input_events",
                )
            if (
                step_index != expected_step_index
                or unit_index != len(checkpoint_by_unit)
                or unit_index < last_step_unit_index
                or unit_index > through_unit_index
                or step_index in step_slots
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    "非法、乱序或重复 generation step: "
                    f"expected_step={expected_step_index}, "
                    f"step={step_index}, unit={unit_index}, "
                    f"last_unit={last_step_unit_index}",
                )
            current_phase = track_phase_by_unit.get(unit_index, "spoken")
            if track == "spoken" and current_phase == "non_spoken":
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit {unit_index} 在 non_spoken 后又出现 spoken step",
                )
            if track == "non_spoken":
                track_phase_by_unit[unit_index] = "non_spoken"
            else:
                track_phase_by_unit.setdefault(unit_index, "spoken")
            expected_step_index += 1
            last_step_unit_index = unit_index
            output = dict(raw_step.get("output") or {})
            kind = output.get("kind")
            slot = _StepSlot(
                step_index=step_index,
                unit_index=unit_index,
                stream_id=stream_id,
                track=track,
            )
            step_slots[step_index] = slot

            if kind == "text_pending":
                if (
                    track == "non_spoken"
                    and active_non_spoken_stream is None
                    and pending_non_spoken_continuation_kind is not None
                ):
                    active_non_spoken_stream = (
                        stream_id,
                        pending_non_spoken_continuation_kind,
                    )
                    pending_non_spoken_continuation_kind = None
                if (
                    track == "spoken"
                    and active_spoken_stream_id != stream_id
                ) or (
                    track == "non_spoken"
                    and (
                        active_non_spoken_stream is None
                        or active_non_spoken_stream[0] != stream_id
                    )
                ):
                    raise _resume_error(
                        "incomplete_event_history",
                        f"ordinary step 没有匹配的 active stream: "
                        f"track={track}, stream_id={stream_id}",
                    )
                continue
            if kind == "protocol":
                semantic_key = str(output.get("semantic_key") or "")
                deferred_model_feed = bool(
                    output.get("deferred_model_feed", False)
                )
                if deferred_model_feed and semantic_key != "non_spoken_budget_reached":
                    raise _resume_error(
                        "incomplete_event_history",
                        "deferred_model_feed 只允许用于 "
                        "non_spoken_budget_reached",
                    )
                try:
                    slot.token_id = registry.get(O5SpecialTokenKey(semantic_key)).token_id
                except (KeyError, ValueError) as exc:
                    raise _resume_error(
                        "incomplete_event_history",
                        f"未知 protocol semantic key: {semantic_key}",
                    ) from exc
                slot.deferred_model_feed = deferred_model_feed
                if track == "spoken":
                    if semantic_key == "speak":
                        if pending_spoken_slot_eos_stream_id is not None:
                            raise _resume_error(
                                "incomplete_event_history",
                                "spoken_turn_eos 后缺少 synthetic spoken_slot_eos",
                            )
                        if active_spoken_stream_id is None:
                            active_spoken_stream_id = stream_id
                        elif active_spoken_stream_id != stream_id:
                            raise _resume_error(
                                "incomplete_event_history",
                                "spoken continuation 改变了 stream_id",
                            )
                    elif semantic_key == "listen":
                        if (
                            active_spoken_stream_id is not None
                            or pending_spoken_slot_eos_stream_id is not None
                        ):
                            raise _resume_error(
                                "incomplete_event_history",
                                "active spoken turn 在 turn_eos 前出现 listen",
                            )
                    elif semantic_key in {"spoken_slot_eos", "tts_pad"}:
                        if semantic_key == "spoken_slot_eos" and (
                            pending_spoken_slot_eos_stream_id == stream_id
                        ):
                            pending_spoken_slot_eos_stream_id = None
                        elif active_spoken_stream_id != stream_id:
                            raise _resume_error(
                                "incomplete_event_history",
                                f"{semantic_key} 没有匹配的 active spoken stream",
                            )
                    elif semantic_key == "spoken_turn_eos":
                        if active_spoken_stream_id != stream_id:
                            raise _resume_error(
                                "incomplete_event_history",
                                "spoken_turn_eos 没有匹配的 active stream",
                            )
                        active_spoken_stream_id = None
                        pending_spoken_slot_eos_stream_id = stream_id
                else:
                    opener_kind = {
                        "think_start": "think",
                        "tool_call_start": "tool_call",
                    }.get(semantic_key)
                    closer_kind = {
                        "think_end": "think",
                        "tool_call_end": "tool_call",
                    }.get(semantic_key)
                    if opener_kind is not None:
                        if active_non_spoken_stream is not None:
                            raise _resume_error(
                                "incomplete_event_history",
                                "nested non-spoken stream opener",
                            )
                        active_non_spoken_stream = (
                            stream_id,
                            opener_kind,
                        )
                        pending_non_spoken_continuation_kind = None
                    elif closer_kind is not None:
                        if active_non_spoken_stream != (
                            stream_id,
                            closer_kind,
                        ):
                            raise _resume_error(
                                "incomplete_event_history",
                                f"{semantic_key} 没有匹配的 active stream",
                            )
                        active_non_spoken_stream = None
                    elif semantic_key == "non_spoken_budget_reached":
                        if (
                            active_non_spoken_stream is not None
                            and active_non_spoken_stream[0] != stream_id
                        ):
                            raise _resume_error(
                                "incomplete_event_history",
                                "budget_reached 改变了 active stream_id",
                            )
                        if active_non_spoken_stream is not None:
                            pending_non_spoken_continuation_kind = (
                                active_non_spoken_stream[1]
                            )
                        active_non_spoken_stream = None
                    elif semantic_key in {
                        "no_action",
                        "non_spoken_eos",
                        "non_spoken_hold",
                        "non_spoken_abort",
                    }:
                        pending_non_spoken_continuation_kind = None
                continue
            if kind != "text_delta":
                raise _resume_error(
                    "incomplete_event_history",
                    f"未知 generation step output kind: {kind}",
                )

            expected_delta_index = delta_index_by_stream.get(stream_id, 0)
            delta_index = int(output.get("delta_index", -1))
            if delta_index != expected_delta_index:
                raise _resume_error(
                    "incomplete_event_history",
                    f"delta_index 不连续: stream={stream_id}, "
                    f"expected={expected_delta_index}, actual={delta_index}",
                )
            delta_index_by_stream[stream_id] = expected_delta_index + 1

            if (
                track == "non_spoken"
                and active_non_spoken_stream is None
                and pending_non_spoken_continuation_kind is not None
            ):
                active_non_spoken_stream = (
                    stream_id,
                    pending_non_spoken_continuation_kind,
                )
                pending_non_spoken_continuation_kind = None

            if (
                track == "spoken"
                and active_spoken_stream_id != stream_id
            ) or (
                track == "non_spoken"
                and (
                    active_non_spoken_stream is None
                    or active_non_spoken_stream[0] != stream_id
                )
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    f"text delta 没有匹配的 active stream: "
                    f"track={track}, stream_id={stream_id}",
                )

            source_step_indices = output.get("source_step_indices")
            if (
                not isinstance(source_step_indices, list)
                or not source_step_indices
                or any(
                    not isinstance(index, int) or isinstance(index, bool) or index < 0
                    for index in source_step_indices
                )
                or source_step_indices != sorted(set(source_step_indices))
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    f"非法 text delta source_step_indices: {source_step_indices}",
                )
            source_slots: list[_StepSlot] = []
            for source_step_index in source_step_indices:
                source_slot = step_slots.get(source_step_index)
                if (
                    source_slot is None
                    or source_slot.stream_id != stream_id
                    or source_slot.track != track
                    or source_slot.token_id is not None
                ):
                    raise _resume_error(
                        "incomplete_event_history",
                        f"text delta 引用缺失或不匹配 step: {source_step_index}",
                    )
                source_slots.append(source_slot)

            text = output.get("text")
            if not isinstance(text, str) or not text:
                raise _resume_error(
                    "text_delta_roundtrip_mismatch",
                    "text_delta.text 必须是非空字符串",
                )
            recovered_ids = tokenizer.encode_ordinary(text)
            if len(recovered_ids) != len(source_slots):
                raise _resume_error(
                    "text_delta_roundtrip_mismatch",
                    f"text delta re-encode 数量不匹配: ids={len(recovered_ids)}, "
                    f"steps={len(source_slots)}",
                )
            for source_slot, token_id in zip(source_slots, recovered_ids):
                source_slot.token_id = token_id

    if target_checkpoint is None:
        raise _resume_error(
            "incomplete_event_history",
            f"缺少目标 Unit checkpoint: {through_unit_index}",
            unit_index=through_unit_index,
        )
    if pending_valid_tool_call_ids:
        raise _resume_error(
            "unsupported_tool_state",
            "目标 checkpoint 仍有 pending tool result: "
            + ", ".join(sorted(pending_valid_tool_call_ids)),
        )
    if (
        pending_tool_started_events
        or pending_tool_error_events
        or pending_tool_response_events
    ):
        raise _resume_error(
            "unsupported_tool_state",
            "目标 checkpoint 仍有尚未注入下一 Unit 的 tool events",
        )
    resume_info = dict(target_checkpoint.get("resume") or {})
    if resume_info.get("status") != "available":
        raise _resume_error(
            "non_resumable_text_boundary",
            f"Unit {through_unit_index} checkpoint 不可恢复",
            unit_index=through_unit_index,
            stream_id=resume_info.get("stream_id"),
            pending_from_step=resume_info.get("pending_from_step"),
        )
    if (
        active_spoken_stream_id is not None
        or pending_spoken_slot_eos_stream_id is not None
        or active_non_spoken_stream is not None
        or pending_non_spoken_continuation_kind is not None
    ):
        raise _resume_error(
            "incomplete_event_history",
            "available checkpoint 仍有 active semantic stream",
        )

    last_step_index = int(target_checkpoint.get("last_step_index", -1))
    for step_index in range(last_step_index + 1):
        checkpoint_slot = step_slots.get(step_index)
        if checkpoint_slot is None:
            raise _resume_error(
                "incomplete_event_history",
                f"缺少 generation step: {step_index}",
            )
        if checkpoint_slot.token_id is None:
            raise _resume_error(
                "non_resumable_text_boundary",
                f"generation step {step_index} 仍为 pending",
                unit_index=checkpoint_slot.unit_index,
                stream_id=checkpoint_slot.stream_id,
                pending_from_step=step_index,
            )

    expected_unit_count = through_unit_index + 1
    ordered_input_payloads: list[dict[str, Any]] = []
    ordered_tool_events: list[list[dict[str, Any]]] = []
    used_checkpoint_input_ids: set[str] = set()
    for unit_index in range(expected_unit_count):
        checkpoint = checkpoint_by_unit.get(unit_index)
        if checkpoint is None:
            raise _resume_error(
                "incomplete_event_history",
                f"缺少 Unit checkpoint: {unit_index}",
            )
        checkpoint_input_id = str(checkpoint.get("input_id") or "")
        if not checkpoint_input_id or checkpoint_input_id in used_checkpoint_input_ids:
            raise _resume_error(
                "incomplete_event_history",
                f"checkpoint input_id 缺失或重复: "
                f"unit={unit_index}, input_id={checkpoint_input_id!r}",
            )
        used_checkpoint_input_ids.add(checkpoint_input_id)
        checkpoint_input_payload = input_payload_by_id.get(checkpoint_input_id)
        if checkpoint_input_payload is None:
            raise _resume_error(
                "incomplete_event_history",
                f"checkpoint input_id 无对应 input.append: "
                f"unit={unit_index}, input_id={checkpoint_input_id!r}",
            )
        ordered_input_payloads.append(checkpoint_input_payload)
        ordered_tool_events.append(tool_events_by_unit.get(unit_index, []))
    units = [
        FcDuplexResumeUnitPlan(
            unit_index=unit_index,
            input_payload=ordered_input_payloads[unit_index],
            tool_events=ordered_tool_events[unit_index],
            spoken_token_ids=[
                int(slot.token_id)
                for slot in sorted(step_slots.values(), key=lambda item: item.step_index)
                if slot.unit_index == unit_index
                and slot.track == "spoken"
                and slot.token_id is not None
            ],
            non_spoken_token_ids=[
                int(slot.token_id)
                for slot in sorted(step_slots.values(), key=lambda item: item.step_index)
                if slot.unit_index == unit_index
                and slot.track == "non_spoken"
                and slot.token_id is not None
            ],
            deferred_non_spoken_close=any(
                slot.unit_index == unit_index
                and slot.track == "non_spoken"
                and slot.deferred_model_feed
                for slot in step_slots.values()
            ),
        )
        for unit_index in range(expected_unit_count)
    ]
    return FcDuplexResumePlan(
        protocol_version=protocol_version,
        model=model,
        tokenizer_target=tokenizer_target,
        through_unit_index=through_unit_index,
        next_event_index=expected_event_index,
        next_step_index=last_step_index + 1,
        next_batch_index=maximum_batch_index + 1,
        next_stream_sequence=maximum_stream_sequence + 1,
        next_delta_index_by_stream=dict(delta_index_by_stream),
        seen_input_ids=list(input_payload_by_id),
        tool_call_count=tool_call_count,
        api_tool_call_sequence=maximum_api_tool_call_sequence,
        session_init_payload=session_init_payload,
        units=units,
    )


def _build_fc_duplex_semantic_v2_resume_plan(
    *,
    model: str,
    tokenizer_target: Literal["o45_fc", "o5"],
    tokenizer_fingerprint: dict[str, str],
    through_unit_index: int,
    history: list[dict[str, Any]],
) -> FcDuplexResumePlan:
    """Canonicalize the minimal ordered semantic-event v2 history."""

    if through_unit_index < 0:
        raise _resume_error(
            "incomplete_event_history",
            f"through_unit_index 必须 >= 0: {through_unit_index}",
        )
    if (
        not history
        or not isinstance(history[0], dict)
        or history[0].get("type") != "session.init"
    ):
        raise _resume_error(
            "incomplete_event_history",
            "semantic v2 history 必须从 session.init 开始",
        )
    session_init_payload = dict(history[0].get("payload") or {})
    history_target = session_init_payload.get("tokenizer_target")
    if history_target is not None and history_target != tokenizer_target:
        raise _resume_error(
            "model_or_tokenizer_mismatch",
            f"tokenizer target mismatch: {history_target} != {tokenizer_target}",
        )
    tokenizer = load_builtin_tokenizer(O5TokenizerID(tokenizer_target))
    expected_fingerprint = {
        "vocab_hash": tokenizer.fingerprint.vocab_hash,
        "merges_hash": tokenizer.fingerprint.merges_hash,
    }
    if tokenizer_fingerprint != expected_fingerprint:
        raise _resume_error(
            "model_or_tokenizer_mismatch",
            "tokenizer fingerprint mismatch",
        )
    registry = O5SpecialTokenRegistry.from_tokenizer(tokenizer)

    input_payload_by_id: dict[str, dict[str, Any]] = {}
    unit_input_id: dict[int, str] = {}
    unit_tool_events: dict[int, list[dict[str, Any]]] = {}
    unit_non_spoken_ends: dict[int, str] = {}
    unit_commits: dict[int, dict[str, Any]] = {}
    slots: list[_StepSlot] = []
    pending_slots: dict[str, list[_StepSlot]] = {}
    tool_api_to_internal: dict[str, str] = {}
    tool_result_content_by_api: dict[str, str] = {}
    pending_tool_started: list[dict[str, Any]] = []
    pending_tool_error: list[dict[str, Any]] = []
    pending_tool_response: list[dict[str, Any]] = []
    pending_valid_tool_calls: set[str] = set()
    tool_call_count = 0
    maximum_api_tool_sequence = 0
    expected_started_unit = 0
    expected_committed_unit = 0
    active_think = False
    think_parts: list[str] = []
    active_tool_call_id: str | None = None
    tool_parts: list[str] = []
    spoken_turn_active = False
    spoken_parts: list[str] = []
    spoken_started_units: set[int] = set()
    history_unsafe = False

    def append_token(
        *,
        unit_index: int,
        track: Literal["spoken", "non_spoken"],
        token_id: int,
    ) -> None:
        slots.append(
            _StepSlot(
                step_index=len(slots),
                unit_index=unit_index,
                stream_id=track,
                track=track,
                token_id=token_id,
            )
        )

    def consume_text_steps(
        *,
        unit_index: int,
        track: Literal["spoken", "non_spoken"],
        decoder_key: str,
        raw_steps: Any,
        aggregate_parts: list[str],
    ) -> None:
        nonlocal history_unsafe
        if not isinstance(raw_steps, list):
            raise _resume_error(
                "incomplete_event_history",
                "semantic delta.steps 必须是 list",
            )
        pending = pending_slots.setdefault(decoder_key, [])
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                raise _resume_error(
                    "incomplete_event_history",
                    f"semantic text step 必须是 object: {raw_step!r}",
                )
            slot = _StepSlot(
                step_index=len(slots),
                unit_index=unit_index,
                stream_id=decoder_key,
                track=track,
            )
            slots.append(slot)
            pending.append(slot)
            kind = raw_step.get("kind")
            if kind == "pending":
                continue
            if kind != "text":
                raise _resume_error(
                    "incomplete_event_history",
                    f"未知 semantic text step: {kind}",
                )
            text = raw_step.get("text")
            if not isinstance(text, str) or not text:
                raise _resume_error(
                    "text_delta_roundtrip_mismatch",
                    "semantic text step 缺少非空文本",
                )
            recovered_ids = tokenizer.encode_ordinary(text)
            if len(recovered_ids) != len(pending):
                raise _resume_error(
                    "text_delta_roundtrip_mismatch",
                    "semantic text re-encode 数量与有序 pending steps 不匹配",
                )
            for pending_slot, token_id in zip(pending, recovered_ids):
                pending_slot.token_id = token_id
            pending.clear()
            aggregate_parts.append(text)

    def require_active_unit(event: dict[str, Any]) -> int:
        """Require semantic output to belong to the started, uncommitted Unit."""

        unit_index = int(event.get("unit_index", -1))
        if (
            unit_index != expected_committed_unit
            or unit_index not in unit_input_id
            or unit_index in unit_commits
            or unit_index in unit_non_spoken_ends
        ):
            raise _resume_error(
                "incomplete_event_history",
                f"semantic event 不属于当前 active Unit: {unit_index}",
            )
        return unit_index

    for event in history[1:]:
        if not isinstance(event, dict):
            raise _resume_error(
                "incomplete_event_history",
                f"history event 必须是 object: {event!r}",
            )
        event_type = str(event.get("type") or "")
        if event_type == "input.append":
            payload = dict(event.get("input") or {})
            input_id = str(payload.get("input_id") or "")
            if not input_id or input_id in input_payload_by_id:
                raise _resume_error(
                    "incomplete_event_history",
                    f"input.append 缺少唯一 input_id: {input_id!r}",
                )
            _validate_replay_audio(payload, input_id=input_id)
            input_payload_by_id[input_id] = payload
            continue
        if event_type == "input.tool_result":
            api_id = str(event.get("tool_call_id") or "")
            if api_id not in pending_valid_tool_calls:
                raise _resume_error(
                    "incomplete_event_history",
                    f"tool result 无 pending call: {api_id}",
                )
            tool_result_content_by_api[api_id] = _tool_result_content(
                event.get("content", event.get("contents"))
            )
            pending_valid_tool_calls.remove(api_id)
            pending_tool_response.append(
                {
                    "type": "tool_response",
                    "call_id": tool_api_to_internal[api_id],
                    "content": tool_result_content_by_api[api_id],
                }
            )
            continue
        if event_type == "response.unit.started":
            unit_index = int(event.get("unit_index", -1))
            if (
                unit_index != expected_started_unit
                or unit_index != expected_committed_unit
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit started 不连续: {unit_index}",
                )
            expected_started_unit += 1
            input_id = str(event.get("input_id") or "")
            if input_id not in input_payload_by_id:
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit started 引用未知 input_id: {input_id}",
                )
            expected_refs: list[dict[str, Any]] = []
            for internal_event in [
                *pending_tool_started,
                *pending_tool_error,
                *pending_tool_response,
            ]:
                attributed_api_id = next(
                    (
                        api
                        for api, internal in tool_api_to_internal.items()
                        if internal == internal_event["call_id"]
                    ),
                    None,
                )
                if attributed_api_id is None:
                    raise _resume_error(
                        "incomplete_event_history",
                        "internal tool event 无 API id",
                    )
                expected_refs.append(
                    {
                        "type": (
                            "tool_result"
                            if internal_event["type"] == "tool_response"
                            else internal_event["type"]
                        ),
                        "tool_call_id": attributed_api_id,
                    }
                )
            if list(event.get("tool_events") or []) != expected_refs:
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit {unit_index} tool_events 与实际 history 不一致",
                )
            unit_input_id[unit_index] = input_id
            unit_tool_events[unit_index] = [
                *pending_tool_started,
                *pending_tool_error,
                *pending_tool_response,
            ]
            pending_tool_started = []
            pending_tool_error = []
            pending_tool_response = []
            continue
        if event_type == "response.think.begin":
            if active_think:
                raise _resume_error(
                    "incomplete_event_history",
                    "nested think.begin",
                )
            unit_index = require_active_unit(event)
            active_think = True
            think_parts = []
            append_token(
                unit_index=unit_index,
                track="non_spoken",
                token_id=registry.get(O5SpecialTokenKey.THINK_START).token_id,
            )
            continue
        if event_type == "response.think.delta":
            if not active_think:
                raise _resume_error(
                    "incomplete_event_history",
                    "think.delta without begin",
                )
            consume_text_steps(
                unit_index=require_active_unit(event),
                track="non_spoken",
                decoder_key="think",
                raw_steps=event.get("steps"),
                aggregate_parts=think_parts,
            )
            continue
        if event_type == "response.think.end":
            if not active_think:
                raise _resume_error(
                    "incomplete_event_history",
                    "think.end without begin",
                )
            if "".join(think_parts) != str(event.get("full_text") or ""):
                raise _resume_error(
                    "text_delta_roundtrip_mismatch",
                    "think streaming/full mismatch",
                )
            append_token(
                unit_index=require_active_unit(event),
                track="non_spoken",
                token_id=registry.get(O5SpecialTokenKey.THINK_END).token_id,
            )
            active_think = False
            think_parts = []
            pending_slots.pop("think", None)
            continue
        if event_type == "response.tool_call.begin":
            if active_tool_call_id is not None:
                raise _resume_error(
                    "incomplete_event_history",
                    "nested tool_call.begin",
                )
            api_id = str(event.get("tool_call_id") or "")
            match = re.fullmatch(r"tc_(\d+)", api_id)
            if not match or api_id in tool_api_to_internal:
                raise _resume_error(
                    "incomplete_event_history",
                    f"非法 tool_call_id: {api_id}",
                )
            maximum_api_tool_sequence = max(
                maximum_api_tool_sequence,
                int(match.group(1)),
            )
            tool_call_count += 1
            tool_api_to_internal[api_id] = f"fc_call_{tool_call_count:06d}"
            active_tool_call_id = api_id
            tool_parts = []
            append_token(
                unit_index=require_active_unit(event),
                track="non_spoken",
                token_id=registry.get(O5SpecialTokenKey.TOOL_CALL_START).token_id,
            )
            continue
        if event_type == "response.tool_call.delta":
            api_id = str(event.get("tool_call_id") or "")
            if api_id != active_tool_call_id:
                raise _resume_error(
                    "incomplete_event_history",
                    "tool_call.delta 无 active call",
                )
            consume_text_steps(
                unit_index=require_active_unit(event),
                track="non_spoken",
                decoder_key=f"tool:{api_id}",
                raw_steps=event.get("steps"),
                aggregate_parts=tool_parts,
            )
            continue
        if event_type == "response.tool_call.done":
            api_id = str(event.get("tool_call_id") or "")
            if api_id != active_tool_call_id:
                raise _resume_error(
                    "incomplete_event_history",
                    "tool_call.done 无 active call",
                )
            full_text = str(event.get("full_text") or "")
            if full_text and "".join(tool_parts) != full_text:
                raise _resume_error(
                    "text_delta_roundtrip_mismatch",
                    "tool-call streaming/full mismatch",
                )
            append_token(
                unit_index=require_active_unit(event),
                track="non_spoken",
                token_id=registry.get(O5SpecialTokenKey.TOOL_CALL_END).token_id,
            )
            internal_id = tool_api_to_internal[api_id]
            pending_tool_started.append(
                {"type": "tool_started", "call_id": internal_id}
            )
            if event.get("error"):
                pending_tool_error.append(
                    {
                        "type": "tool_response",
                        "call_id": internal_id,
                        "content": (
                            "工具调用解析失败，无法执行该工具调用。错误信息："
                            + str(event["error"])
                        ),
                    }
                )
            else:
                pending_valid_tool_calls.add(api_id)
            active_tool_call_id = None
            tool_parts = []
            pending_slots.pop(f"tool:{api_id}", None)
            continue
        if event_type == "response.spoken.delta":
            unit_index = require_active_unit(event)
            if unit_index not in spoken_started_units:
                append_token(
                    unit_index=unit_index,
                    track="spoken",
                    token_id=registry.get(O5SpecialTokenKey.SPEAK).token_id,
                )
                spoken_started_units.add(unit_index)
                spoken_turn_active = True
            if event.get("steps") is not None:
                consume_text_steps(
                    unit_index=unit_index,
                    track="spoken",
                    decoder_key="spoken",
                    raw_steps=event.get("steps"),
                    aggregate_parts=spoken_parts,
                )
            continue
        if event_type == "response.spoken.end":
            unit_index = require_active_unit(event)
            reason = str(event.get("reason") or "")
            if reason == "listen":
                if spoken_turn_active:
                    raise _resume_error(
                        "incomplete_event_history",
                        "listen before spoken turn eos",
                    )
                append_token(
                    unit_index=unit_index,
                    track="spoken",
                    token_id=registry.get(O5SpecialTokenKey.LISTEN).token_id,
                )
            elif reason == "tts_pad":
                append_token(
                    unit_index=unit_index,
                    track="spoken",
                    token_id=registry.get(O5SpecialTokenKey.TTS_PAD).token_id,
                )
            elif reason == "slot_end":
                pass
            elif reason == "slot_eos":
                if not spoken_turn_active:
                    raise _resume_error(
                        "incomplete_event_history",
                        "slot_eos without spoken turn",
                    )
                append_token(
                    unit_index=unit_index,
                    track="spoken",
                    token_id=registry.get(O5SpecialTokenKey.SPOKEN_SLOT_EOS).token_id,
                )
            elif reason == "turn_eos":
                if not spoken_turn_active:
                    raise _resume_error(
                        "incomplete_event_history",
                        "turn_eos without spoken turn",
                    )
                if "".join(spoken_parts) != str(event.get("full_text") or ""):
                    raise _resume_error(
                        "text_delta_roundtrip_mismatch",
                        "spoken streaming/full mismatch",
                    )
                append_token(
                    unit_index=unit_index,
                    track="spoken",
                    token_id=registry.get(O5SpecialTokenKey.SPOKEN_TURN_EOS).token_id,
                )
                append_token(
                    unit_index=unit_index,
                    track="spoken",
                    token_id=registry.get(O5SpecialTokenKey.SPOKEN_SLOT_EOS).token_id,
                )
                spoken_turn_active = False
                spoken_parts = []
                pending_slots.pop("spoken", None)
            else:
                raise _resume_error(
                    "incomplete_event_history",
                    f"未知 spoken end reason: {reason}",
                )
            continue
        if event_type == "response.warning":
            if event.get("code") == "incomplete_bpe_at_stream_end":
                history_unsafe = True
            continue
        if event_type == "response.non_spoken.end":
            unit_index = int(event.get("unit_index", -1))
            if unit_index in unit_non_spoken_ends:
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit {unit_index} 重复 non_spoken.end",
                )
            unit_index = require_active_unit(event)
            reason = str(event.get("reason") or "")
            key_by_reason = {
                "no_action": O5SpecialTokenKey.NO_ACTION,
                "eos": O5SpecialTokenKey.NON_SPOKEN_EOS,
                "budget_reached": O5SpecialTokenKey.NON_SPOKEN_BUDGET_REACHED,
            }
            if reason not in key_by_reason:
                raise _resume_error(
                    "incomplete_event_history",
                    f"未知 non_spoken.end reason: {reason}",
                )
            if reason in {"no_action", "eos"} and (
                active_think or active_tool_call_id is not None
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    f"{reason} 不能结束仍 active 的 semantic message",
                )
            append_token(
                unit_index=unit_index,
                track="non_spoken",
                token_id=registry.get(key_by_reason[reason]).token_id,
            )
            if reason == "budget_reached":
                if active_think:
                    pending_slots["think"] = []
                if active_tool_call_id is not None:
                    pending_slots[f"tool:{active_tool_call_id}"] = []
            unit_non_spoken_ends[unit_index] = reason
            continue
        if event_type == "response.unit.committed":
            unit_index = int(event.get("unit_index", -1))
            if (
                unit_index != expected_committed_unit
                or unit_index not in unit_input_id
            ):
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit committed 不连续: {unit_index}",
                )
            if unit_index not in unit_non_spoken_ends:
                raise _resume_error(
                    "incomplete_event_history",
                    f"Unit {unit_index} committed 前缺少 response.non_spoken.end",
                )
            expected_committed_unit += 1
            unit_commits[unit_index] = event
            if unit_index == through_unit_index:
                break

    target_commit = unit_commits.get(through_unit_index)
    if target_commit is None:
        raise _resume_error(
            "incomplete_event_history",
            f"缺少目标 Unit committed: {through_unit_index}",
        )
    if dict(target_commit.get("resume") or {}).get("status") != "available":
        raise _resume_error(
            "non_resumable_text_boundary",
            f"目标 Unit {through_unit_index} 不可恢复",
            unit_index=through_unit_index,
        )
    if (
        active_think
        or active_tool_call_id is not None
        or spoken_turn_active
        or pending_valid_tool_calls
        or pending_tool_started
        or pending_tool_error
        or pending_tool_response
        or history_unsafe
        or any(slot.token_id is None for slot in slots)
    ):
        raise _resume_error(
            "non_resumable_text_boundary",
            "available checkpoint 的 semantic history 不闭合",
            unit_index=through_unit_index,
        )
    expected_units = through_unit_index + 1
    if set(unit_input_id) != set(range(expected_units)):
        raise _resume_error(
            "incomplete_event_history",
            "Unit started 不完整",
        )
    if set(unit_commits) != set(range(expected_units)):
        raise _resume_error(
            "incomplete_event_history",
            "Unit committed 不完整",
        )
    if set(unit_non_spoken_ends) != set(range(expected_units)):
        raise _resume_error(
            "incomplete_event_history",
            "Unit non_spoken.end 不完整",
        )
    units = [
        FcDuplexResumeUnitPlan(
            unit_index=unit_index,
            input_payload=input_payload_by_id[unit_input_id[unit_index]],
            tool_events=unit_tool_events.get(unit_index, []),
            spoken_token_ids=[
                int(slot.token_id)
                for slot in slots
                if slot.unit_index == unit_index
                and slot.track == "spoken"
                and slot.token_id is not None
            ],
            non_spoken_token_ids=[
                int(slot.token_id)
                for slot in slots
                if slot.unit_index == unit_index
                and slot.track == "non_spoken"
                and slot.token_id is not None
            ],
            deferred_non_spoken_close=(
                unit_non_spoken_ends[unit_index] == "budget_reached"
            ),
        )
        for unit_index in range(expected_units)
    ]
    return FcDuplexResumePlan(
        protocol_version="fc-duplex-semantic-v2",
        model=model,
        tokenizer_target=tokenizer_target,
        through_unit_index=through_unit_index,
        next_event_index=0,
        next_step_index=len(slots),
        next_batch_index=0,
        next_stream_sequence=1,
        next_delta_index_by_stream={},
        seen_input_ids=list(input_payload_by_id),
        tool_call_count=tool_call_count,
        api_tool_call_sequence=maximum_api_tool_sequence,
        session_init_payload=session_init_payload,
        units=units,
    )
