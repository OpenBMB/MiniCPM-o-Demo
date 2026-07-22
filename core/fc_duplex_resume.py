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
    tool_result_payloads: list[dict[str, Any]] = Field(default_factory=list)
    spoken_token_ids: list[int] = Field(default_factory=list)
    non_spoken_token_ids: list[int] = Field(default_factory=list)


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
    tool_results_by_input_id: dict[str, list[dict[str, Any]]] = {}
    pending_tool_results: list[dict[str, Any]] = []
    expected_event_index = 0
    expected_batch_index = 0
    expected_step_index = 0
    last_step_unit_index = -1
    maximum_batch_index = -1
    maximum_stream_sequence = 0
    target_checkpoint: dict[str, Any] | None = None

    for event in history[1:]:
        if not isinstance(event, dict):
            raise _resume_error(
                "incomplete_event_history",
                f"resume history event 必须是 object: {event!r}",
            )
        event_type = str(event.get("type") or "")
        if event_type in {
            "input.tool_result",
            "input.tool_result.delta",
            "input.tool_result.done",
        }:
            pending_tool_results.append(dict(event))
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
            tool_results_by_input_id[input_id] = pending_tool_results
            pending_tool_results = []
            continue
        if event_type not in {
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
                continue
            if kind == "protocol":
                semantic_key = str(output.get("semantic_key") or "")
                try:
                    slot.token_id = registry.get(O5SpecialTokenKey(semantic_key)).token_id
                except (KeyError, ValueError) as exc:
                    raise _resume_error(
                        "incomplete_event_history",
                        f"未知 protocol semantic key: {semantic_key}",
                    ) from exc
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
    resume_info = dict(target_checkpoint.get("resume") or {})
    if resume_info.get("status") != "available":
        raise _resume_error(
            "non_resumable_text_boundary",
            f"Unit {through_unit_index} checkpoint 不可恢复",
            unit_index=through_unit_index,
            stream_id=resume_info.get("stream_id"),
            pending_from_step=resume_info.get("pending_from_step"),
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
    ordered_tool_results: list[list[dict[str, Any]]] = []
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
        ordered_tool_results.append(
            tool_results_by_input_id.get(checkpoint_input_id, [])
        )
    if any(ordered_tool_results):
        raise _resume_error(
            "unsupported_tool_state",
            "当前 resume MVP 不支持包含 tool result 的历史",
        )

    units = [
        FcDuplexResumeUnitPlan(
            unit_index=unit_index,
            input_payload=ordered_input_payloads[unit_index],
            tool_result_payloads=ordered_tool_results[unit_index],
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
        session_init_payload=session_init_payload,
        units=units,
    )
