#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FC duplex capability for the o45 demo unified wrapper.

This module intentionally keeps FC slot protocol logic outside
``modeling_minicpmo_unified.py`` so the unified model can remain a thin
wrapper around the vendored/weight modeling implementation.
"""

import json
import logging
import os
import re
import time
from typing import List
from typing import Optional

import numpy as np
import torch

from .modeling_minicpmo import gen_logits
from .processing_minicpmo import MiniCPMOProcessor
from .utils import StreamDecoder
from .utils import torch_clone_recursive

logger = logging.getLogger(__name__)

class FcDuplexCapability:
    """FC slot duplex capability.

    This is a parallel implementation to ``DuplexCapability`` for the new
    FC slot protocol. It keeps similar lifecycle method names while splitting
    generation into spoken and non-spoken phases.
    """

    def __init__(
        self,
        model: "MiniCPMO",
        device: str = "cuda",
        temperature: float = 0.7,
        tool_format: str = "minicpm4_xml",
        forbidden_token_ids=None,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.tool_format = tool_format
        self.extra_config = kwargs
        self.generate_audio = bool(kwargs.get("generate_audio", False))
        self.tts_temperature = torch.tensor(
            [kwargs.get("tts_temperature", 0.8)],
            dtype=torch.float,
            device=self.device,
        )
        self.tts_repetition_penalty = kwargs.get("tts_repetition_penalty", 1.05)
        self.prompt_wav_path = None

        if not hasattr(self.model, "processor") or self.model.processor is None:
            self.model.processor = MiniCPMOProcessor.from_pretrained(
                self.model.config._name_or_path, trust_remote_code=True
            )
        self.processor = self.model.processor
        self.tokenizer = self.processor.tokenizer

        if forbidden_token_ids is None:
            tts_pad_id = self.tokenizer.convert_tokens_to_ids("<|tts_pad|>")
            bad_token_ids = getattr(self.tokenizer, "bad_token_ids", [])
            forbidden_token_ids = [tts_pad_id] + list(bad_token_ids)
        self.forbidden_token_ids = forbidden_token_ids
        self.decoder = StreamDecoder(
            llm=self.model.llm,
            tokenizer=self.tokenizer,
            forbidden_token_ids=self.forbidden_token_ids,
        )

        self._sdk_tokenizer = None
        self._registry = None
        self._serializer = None
        self._normalize_tool_response_content = None
        self.K = None
        self.ids = {}
        self.id2name = {}
        self.max_special_id = 0
        self._resized = False
        self.tts_logits_processors = None
        self.tts_eos_token = None
        if getattr(self.model, "tts", None) is not None:
            self.tts_logits_processors = gen_logits(
                num_code=self.model.tts.config.num_audio_tokens,
                repetition_penalty=self.tts_repetition_penalty,
            )
            self.tts_eos_token = torch.tensor(
                [self.model.tts.config.num_audio_tokens - 1],
                dtype=torch.long,
                device=self.device,
            )

        self._reset_streaming_state()
        logger.info("[FcDuplexCapability] initialized")

    @property
    def protocol(self):
        """Compatibility accessor: this object owns the protocol helpers."""
        self._ensure_protocol()
        return self

    @property
    def protocol_tokenizer(self):
        """Return the SDK tokenizer bundle used by this FC protocol instance."""
        self._ensure_protocol()
        return self._sdk_tokenizer

    def _ensure_protocol(self):
        if self._registry is not None:
            return
        from minicpm_o5_sdk import (
            get_o5_tool_serializer,
            load_builtin_o45_fc_tokenizer,
            normalize_tool_response_content,
        )
        from minicpm_o5_sdk.protocols.duplex.special_tokens import (
            O5SpecialTokenKey,
            O5SpecialTokenRegistry,
        )

        self.K = O5SpecialTokenKey
        self._normalize_tool_response_content = normalize_tool_response_content
        self._sdk_tokenizer = load_builtin_o45_fc_tokenizer()
        self._registry = O5SpecialTokenRegistry.from_tokenizer(self._sdk_tokenizer)
        self._serializer = get_o5_tool_serializer(self.tool_format)

        self.ids = {}
        self.id2name = {}
        for key in self.K:
            try:
                resolved = self._registry.get(key)
            except Exception:
                continue
            self.ids[key.value] = resolved.token_id
            self.id2name[resolved.token_id] = resolved.display_name
        self.max_special_id = max(self.id2name) if self.id2name else 0
        logger.info(
            "[FcDuplexCapability] protocol ready: %d special tokens, max_id=%d",
            len(self.id2name),
            self.max_special_id,
        )

    def sid(self, key) -> int:
        self._ensure_protocol()
        return self._registry.get(key).token_id

    def is_special(self, tid: int) -> bool:
        self._ensure_protocol()
        return tid in self.id2name

    def encode_text(self, text: str) -> list:
        self._ensure_protocol()
        if not text:
            return []
        return [t.token_id for t in self._sdk_tokenizer.encode_ordinary_with_offsets(text)]

    def decode_text(self, ids: list) -> str:
        self._ensure_protocol()
        if not ids:
            return ""
        return self._sdk_tokenizer.decode_ordinary(ids)

    def _flush(self, buf: list) -> str:
        try:
            return self.decode_text(buf)
        except Exception:
            parts = []
            for tid in buf:
                try:
                    parts.append(self.decode_text([tid]))
                except Exception:
                    parts.append(f"<id:{tid}>")
            return "".join(parts)

    def safe_decode_text(self, ids: list) -> str:
        self._ensure_protocol()
        out = []
        buf = []
        for tid in ids:
            if self.is_special(tid):
                if buf:
                    out.append(self._flush(buf))
                    buf = []
                out.append(self.id2name[tid])
            else:
                buf.append(tid)
        if buf:
            out.append(self._flush(buf))
        return "".join(out)

    def render_token_stream(self, ids: list) -> str:
        return self.safe_decode_text(ids)

    def _normalize_tools(self, tools):
        if not tools:
            return None
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        if all(not isinstance(t, dict) for t in tools):
            return list(tools)
        try:
            from minicpm_o5_sdk import OpenAIToolDefinition

            return [OpenAIToolDefinition.model_validate(t) if isinstance(t, dict) else t for t in tools]
        except Exception:
            return list(tools)

    def _system_prefill_parts(self, system_prompt: str, tools=None, has_ref_audio: bool = False) -> tuple[list, list]:
        self._ensure_protocol()
        tools = self._normalize_tools(tools)
        prefix_ids = [self.sid(self.K.IM_START)]
        prefix_ids += self.encode_text(system_prompt or "")
        suffix_ids = []
        if has_ref_audio:
            prefix_ids += [self.sid(self.K.AUDIO_START)]
            suffix_ids += [self.sid(self.K.AUDIO_END)]
        if tools:
            block = self._serializer.render_tool_system_block(list(tools))
            suffix_ids += self.encode_text(block.preamble)
            suffix_ids += self.encode_text(block.definitions)
            suffix_ids += self.encode_text(block.guidelines)
        suffix_ids += [self.sid(self.K.IM_END)]
        return prefix_ids, suffix_ids

    def _system_prefill_ids(self, system_prompt: str, tools=None, has_ref_audio: bool = False) -> list:
        prefix_ids, suffix_ids = self._system_prefill_parts(system_prompt, tools, has_ref_audio=has_ref_audio)
        return prefix_ids + suffix_ids

    def _user_video_slot_ids(self, n_image: int = 0, n_slice: int = 0) -> list:
        self._ensure_protocol()
        if n_image <= 0 and n_slice <= 0:
            return []
        ids = [self.sid(self.K.USER_VIDEO_SLOT_START)]
        if n_image > 0:
            ids += [self.sid(self.K.IMAGE_START)]
            ids += [self.sid(self.K.IMAGE_PLACEHOLDER)] * n_image
            ids += [self.sid(self.K.IMAGE_END)]
        if n_slice > 0:
            ids += [self.sid(self.K.SLICE_START)]
            ids += [self.sid(self.K.IMAGE_PLACEHOLDER)] * n_slice
            ids += [self.sid(self.K.SLICE_END)]
        ids += [self.sid(self.K.USER_VIDEO_SLOT_END)]
        return ids

    def _user_audio_slot_ids(self, n_audio: int = 0) -> list:
        self._ensure_protocol()
        if n_audio <= 0:
            return []
        return (
            [self.sid(self.K.USER_AUDIO_SLOT_START)]
            + [self.sid(self.K.AUDIO_PLACEHOLDER)] * n_audio
            + [self.sid(self.K.USER_AUDIO_SLOT_END)]
        )

    def _input_event_slot_ids(self, tool_responses=None) -> list:
        self._ensure_protocol()
        if not tool_responses:
            return []
        ids = [self.sid(self.K.INPUT_EVENT_SLOT_START)]
        for item in tool_responses:
            if isinstance(item, dict):
                call_id = item.get("call_id") or item.get("id") or item.get("tool_call_id")
                event_type = item.get("type") or item.get("event") or "tool_response"
                raw = item.get("content") if "content" in item else item.get("response")
            else:
                call_id, raw = item
                event_type = "tool_response"
            if not call_id:
                raise ValueError("tool event missing call_id/tool_call_id")
            ids += [self.sid(self.K.TOOL_RESPONSE_EVENT_START), self.sid(self.K.TOOL_CALL_ID_START)]
            ids += self.encode_text(str(call_id))
            ids += [self.sid(self.K.TOOL_CALL_ID_END)]
            if event_type in ("tool_started", "started"):
                ids += [self.sid(self.K.TOOL_STARTED), self.sid(self.K.TOOL_RESPONSE_EVENT_END)]
                continue
            content = self._normalize_tool_response_content(raw)
            ids += [self.sid(self.K.TOOL_RESPONSE_START)]
            ids += self.encode_text(content)
            ids += [self.sid(self.K.TOOL_RESPONSE_END), self.sid(self.K.TOOL_RESPONSE_EVENT_END)]
        ids += [self.sid(self.K.INPUT_EVENT_SLOT_END)]
        return ids

    def _unit_input_ids(self, n_audio=0, n_image=0, n_slice=0, tool_responses=None) -> list:
        return (
            self._user_video_slot_ids(n_image, n_slice)
            + self._user_audio_slot_ids(n_audio)
            + self._input_event_slot_ids(tool_responses or [])
        )

    def _resize_embeddings(self) -> dict:
        self._ensure_protocol()
        need = self.max_special_id + 1
        emb = self.model.llm.get_input_embeddings()
        cur = emb.weight.shape[0]
        info = {"old_vocab": int(cur), "new_vocab": int(cur), "resized": False, "need": int(need)}
        if cur < need:
            self.model.llm.resize_token_embeddings(need)
            new_cur = self.model.llm.get_input_embeddings().weight.shape[0]
            info.update({"new_vocab": int(new_cur), "resized": True})
            logger.warning(
                "[FcDuplexCapability] resized token embeddings %d -> %d; new rows must be trained",
                cur,
                new_cur,
            )
        self._resized = True
        return info

    def _feed_ids(self, ids: list, want_logits: bool = False):
        if not ids:
            return None
        self.output_ids.extend(ids)
        out = self.decoder.feed(self.decoder.embed_tokens(ids), return_logits=want_logits)
        return out[0] if want_logits else None

    def _feed_mixed_embeddings(self, chunks: list, want_logits: bool = False):
        embeds = []
        for kind, value in chunks:
            if kind == "ids":
                ids = list(value or [])
                if not ids:
                    continue
                self.output_ids.extend(ids)
                embeds.append(self.decoder.embed_tokens(ids))
            elif kind == "ids_no_record":
                ids = list(value or [])
                if ids:
                    embeds.append(self.decoder.embed_tokens(ids))
            elif kind == "embeds":
                if value is not None and int(value.shape[0]) > 0:
                    embeds.append(value)
            else:
                raise ValueError(f"unsupported feed chunk kind: {kind}")
        if not embeds:
            return None
        out = self.decoder.feed(torch.cat(embeds, dim=0), return_logits=want_logits)
        return out[0] if want_logits else None

    def _audio_embeds(self, audio_waveform, sample_rate: int = 16000):
        del sample_rate
        if audio_waveform is None or len(audio_waveform) == 0:
            return None, 0
        audio = np.asarray(audio_waveform, dtype=np.float32)
        data = self.processor.process_audio([audio])
        embeds_nested = self.model.get_audio_embedding(
            data,
            chunk_length=self.model.config.audio_chunk_length,
        )
        if not embeds_nested:
            return None, 0
        audio_embeds = torch.cat([t for group in embeds_nested for t in group], dim=0)
        return audio_embeds, int(audio_embeds.shape[0])

    def _feed_audio(self, audio_waveform, sample_rate: int = 16000) -> int:
        audio_embeds, n_audio = self._audio_embeds(audio_waveform, sample_rate=sample_rate)
        if n_audio <= 0:
            return 0
        self.output_ids.extend([self.sid(self.K.AUDIO_PLACEHOLDER)] * n_audio)
        self.decoder.feed(audio_embeds)
        return n_audio

    def _sample(self, logits, decode_mode: str) -> int:
        if decode_mode in ("greedy", "argmax"):
            return int(torch.argmax(logits[0]).item())
        probs = torch.softmax(logits[0] / max(float(self.temperature), 1e-5), dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _lookup_tool_definition(self, wire_text: str, tool_definitions):
        tool_definitions = self._normalize_tools(tool_definitions)
        if not tool_definitions:
            return None
        match = re.search(r'name="([^"]+)"', wire_text) or re.search(r'"name"\s*:\s*"([^"]+)"', wire_text)
        if match:
            target = match.group(1)
            for definition in tool_definitions:
                if getattr(getattr(definition, "function", None), "name", None) == target:
                    return definition
        return tool_definitions[0]

    def _safe_deserialize_tool_call(self, wire: str, tool_definitions=None) -> dict:
        self._ensure_protocol()
        definition = self._lookup_tool_definition(wire, tool_definitions or self._tools)
        result = {"wire": wire, "name": None, "arguments": None, "error": None}
        if definition is None:
            result["error"] = "no tool definition available to deserialize"
            return result
        try:
            call = self._serializer.deserialize_tool_call(wire, definition=definition)
            result["name"] = call.function.name
            result["arguments"] = call.function.arguments
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _reset_streaming_state(self) -> None:
        self.decoder.reset()
        self.output_ids = []
        self.units_info = []
        self.trace_events = []
        self._trace_step = 0
        self._trace_started_at = time.time()
        self._tools = None
        self._current_unit_idx = 0
        self._current_unit_open = False
        self._current_unit_info = None
        self._spoken_slot_open = False
        self._non_spoken_slot_open = False
        self._pending_prefill_close_ids = []
        self._pending_prefill_unit_info = None
        self._spoken_logits = None
        self._non_spoken_logits = None
        self._non_spoken_mode = None
        self._think_buf = []
        self._tool_call_buf = []
        self.tts_text_start_pos = 0
        self.tts_past_key_values = None
        self.tts_current_turn_start_time = None
        self.token2wav_initialized = False
        self.token2wav_buffer = []
        self.flow_cache_base = None
        self.hift_cache_base = None
        self.pre_lookahead = 0

    def _token_pieces(self, ids: list) -> list:
        if not ids:
            return []
        pieces = []
        ordinary = []
        ordinary_positions = []
        for tid in ids:
            tid = int(tid)
            if tid in self.id2name:
                pieces.append(str(self.id2name[tid]))
            else:
                pieces.append(None)
                ordinary.append(tid)
                ordinary_positions.append(len(pieces) - 1)
        try:
            converted = list(self.tokenizer.convert_ids_to_tokens(ordinary)) if ordinary else []
        except Exception:
            converted = []
        for index, pos in enumerate(ordinary_positions):
            piece = converted[index] if index < len(converted) else None
            pieces[pos] = str(piece) if piece is not None else f"<id:{ordinary[index]}>"
        return [str(piece) for piece in pieces]

    def _trace_span(self, span: dict) -> dict:
        if not isinstance(span, dict):
            return {"repr": repr(span)}
        item = dict(span)
        tool_call = item.get("tool_call")
        if isinstance(tool_call, dict):
            item["tool_call"] = dict(tool_call)
        return item

    def _record_trace(self, event: str, **fields) -> None:
        self._trace_step = getattr(self, "_trace_step", 0) + 1
        item = {
            "step": self._trace_step,
            "event": event,
            "ts": time.time(),
            "unit": self._current_unit_idx,
            "output_len": len(self.output_ids),
        }
        item.update(fields)
        self.trace_events.append(item)

    def trace_snapshot(self, *, session_id: Optional[str] = None, reason: Optional[str] = None) -> dict:
        ids = list(getattr(self, "output_ids", []) or [])
        snapshot = {
            "schema": "fc_duplex_model_trace.v1",
            "session_id": session_id,
            "reason": reason,
            "created_at": time.time(),
            "started_at": getattr(self, "_trace_started_at", None),
            "tool_format": self.tool_format,
            "generate_audio": self.generate_audio,
            "current_unit_idx": self._current_unit_idx,
            "kv_cache_length": self.decoder.get_cache_length(),
            "current_non_spoken_mode": self._non_spoken_mode,
            "open_think_token_ids": list(self._think_buf),
            "open_tool_call_token_ids": list(self._tool_call_buf),
            "output_ids": ids,
            "output_token_strs": self._token_pieces(ids),
            "output_render": self.render_token_stream(ids) if ids else "",
            "units_info": list(getattr(self, "units_info", []) or []),
            "current_unit_info": dict(self._current_unit_info) if self._current_unit_info else None,
            "events": list(getattr(self, "trace_events", []) or []),
        }
        if snapshot["open_think_token_ids"]:
            snapshot["open_think_text"] = self._flush(snapshot["open_think_token_ids"])
        if snapshot["open_tool_call_token_ids"]:
            snapshot["open_tool_call_wire"] = self._flush(snapshot["open_tool_call_token_ids"])
        return snapshot

    def dump_trace(self, path: str, *, session_id: Optional[str] = None, reason: Optional[str] = None) -> dict:
        snapshot = self.trace_snapshot(session_id=session_id, reason=reason)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        logger.info("[FcDuplexCapability] trace dumped: %s", path)
        return {"path": path, "n_output_ids": len(snapshot["output_ids"]), "n_events": len(snapshot["events"])}

    def _init_token2wav_cache(self, prompt_wav_path: str) -> None:
        if getattr(self.model, "tts", None) is None:
            raise RuntimeError("TTS model is not initialized")
        self.model.tts.audio_tokenizer.cache = None
        flow_cache, hift_cache = self.model.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)
        self.flow_cache_base = torch_clone_recursive(flow_cache)
        self.hift_cache_base = torch_clone_recursive(hift_cache)
        self.pre_lookahead = int(self.model.tts.audio_tokenizer.flow.pre_lookahead_len)
        self.token2wav_initialized = True

    def _reset_token2wav_for_new_turn(self) -> None:
        if not self.token2wav_initialized:
            return
        self.model.tts.audio_tokenizer.stream_cache = torch_clone_recursive(self.flow_cache_base)
        self.model.tts.audio_tokenizer.hift_cache_dict = torch_clone_recursive(self.hift_cache_base)
        self.token2wav_buffer = [4218] * 3

    def _convert_results_to_tts_input(self, results):
        if len(results) == 0:
            audio_bos = self.model.tts.emb_text(
                torch.tensor(
                    [self.model.tts.audio_bos_token_id],
                    device=self.model.tts.emb_text.weight.device,
                    dtype=torch.long,
                )
            )
            return audio_bos.unsqueeze(0)

        llm_tokens = []
        llm_hidden = []
        for token_id, hidden, _end_of_turn in results:
            llm_tokens.append(token_id)
            llm_hidden.append(hidden.squeeze(0))

        llm_tokens_tensor = torch.tensor(llm_tokens, device=self.device, dtype=torch.long)
        llm_embeds = self.model.tts.emb_text(llm_tokens_tensor)

        llm_hidden_tensor = torch.cat(llm_hidden, dim=0)
        llm_hidden_tensor = self.model.tts.projector_semantic(llm_hidden_tensor)
        llm_hidden_tensor = torch.nn.functional.normalize(llm_hidden_tensor, p=2, dim=-1)
        tts_embeds = llm_embeds + llm_hidden_tensor

        audio_bos = self.model.tts.emb_text(
            torch.tensor(
                [self.model.tts.audio_bos_token_id],
                device=self.model.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        )
        return torch.cat([tts_embeds, audio_bos], dim=0).unsqueeze(0)

    def _generate_waveform_from_tokens(
        self,
        new_tokens: torch.Tensor,
        prompt_wav_path: Optional[str],
        is_last_chunk: bool = False,
        force_flush: bool = False,
    ) -> Optional[np.ndarray]:
        if not self.token2wav_initialized:
            logger.warning("[FcDuplexCapability] Token2Wav is not initialized")
            return None

        chunk_size = 25
        token_ids = torch.reshape(new_tokens, (-1,)).tolist()
        self.token2wav_buffer += token_ids
        eos_id = int(self.tts_eos_token.item()) if self.tts_eos_token is not None else None
        has_chunk_eos = eos_id is not None and eos_id in token_ids

        pcm_bytes_list = []
        if has_chunk_eos or force_flush:
            while len(self.token2wav_buffer) >= self.pre_lookahead + 5:
                chunk_to_process = min(chunk_size + self.pre_lookahead, len(self.token2wav_buffer))
                pcm_bytes_list.append(
                    self.model.tts.audio_tokenizer.stream(
                        self.token2wav_buffer[:chunk_to_process],
                        prompt_wav=prompt_wav_path,
                    )
                )
                self.token2wav_buffer = self.token2wav_buffer[
                    min(chunk_size, chunk_to_process - self.pre_lookahead) :
                ]
        else:
            while len(self.token2wav_buffer) >= chunk_size + self.pre_lookahead:
                pcm_bytes_list.append(
                    self.model.tts.audio_tokenizer.stream(
                        self.token2wav_buffer[: chunk_size + self.pre_lookahead],
                        prompt_wav=prompt_wav_path,
                    )
                )
                self.token2wav_buffer = self.token2wav_buffer[chunk_size:]

        if is_last_chunk and len(self.token2wav_buffer) > 0:
            pcm_bytes_list.append(
                self.model.tts.audio_tokenizer.stream(
                    self.token2wav_buffer,
                    prompt_wav=prompt_wav_path,
                    last_chunk=True,
                )
            )
            self.token2wav_buffer = []

        if not pcm_bytes_list:
            return None

        all_pcm = b"".join(pcm_bytes_list)
        if len(all_pcm) == 0:
            return None
        audio_waveform = np.frombuffer(all_pcm, dtype="<i2").astype(np.float32) / 32768.0
        if not is_last_chunk and len(audio_waveform) < 24000:
            audio_waveform = np.pad(audio_waveform, (24000 - len(audio_waveform), 0), mode="constant")
        return audio_waveform

    def _generate_spoken_audio(self, tts_hidden_in_unit: list, end_of_turn: bool) -> dict:
        if not self.generate_audio or (not tts_hidden_in_unit and not end_of_turn):
            return {
                "audio_waveform": None,
                "audio_sample_rate": None,
                "n_tts_tokens": 0,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
            }
        if not self.prompt_wav_path:
            raise ValueError("prompt_wav_path is required when generate_audio=True")
        if not self.token2wav_initialized:
            self._init_token2wav_cache(self.prompt_wav_path)
            self._reset_token2wav_for_new_turn()

        tts_prep_start = time.time()
        tts_condition = self._convert_results_to_tts_input(tts_hidden_in_unit)
        tts_prep_end = time.time()

        min_token_per_chunk = 0 if end_of_turn or self.tts_text_start_pos == 0 else 26
        force_flush = self.tts_text_start_pos == 0

        tts_start = time.time()
        new_tokens, old_kv = self.model.tts.generate_chunk(
            inputs_embeds=tts_condition,
            temperature=self.tts_temperature,
            repetition_penalty=self.tts_repetition_penalty,
            eos_token=self.tts_eos_token,
            force_no_stop=False,
            max_new_token=26,
            min_new_tokens=min_token_per_chunk,
            past_key_values=self.tts_past_key_values,
            logits_processors=self.tts_logits_processors,
            text_start_pos=self.tts_text_start_pos,
        )
        tts_end = time.time()

        if end_of_turn:
            self.tts_text_start_pos = 0
            self.tts_past_key_values = None
            self.tts_current_turn_start_time = None
        else:
            self.tts_past_key_values = old_kv
            self.tts_text_start_pos += tts_condition.shape[1] + new_tokens.shape[1]

        token2wav_start = time.time()
        audio_waveform = self._generate_waveform_from_tokens(
            new_tokens,
            self.prompt_wav_path,
            is_last_chunk=end_of_turn,
            force_flush=force_flush,
        )
        token2wav_end = time.time()

        if end_of_turn:
            self._reset_token2wav_for_new_turn()

        return {
            "audio_waveform": audio_waveform,
            "audio_sample_rate": 24000 if audio_waveform is not None else None,
            "n_tts_tokens": int(new_tokens.numel()),
            "cost_tts_prep": tts_prep_end - tts_prep_start,
            "cost_tts": tts_end - tts_start,
            "cost_token2wav": token2wav_end - token2wav_start,
        }

    def prepare(
        self,
        system_prompt: str,
        tools=None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        generate_audio: Optional[bool] = None,
    ) -> dict:
        self._ensure_protocol()
        resize_info = self._resize_embeddings()
        self._reset_streaming_state()
        self._tools = self._normalize_tools(tools)
        if generate_audio is not None:
            self.generate_audio = bool(generate_audio)
        self.prompt_wav_path = prompt_wav_path
        self.model.init_streaming_processor()
        if self.generate_audio:
            if not self.prompt_wav_path:
                raise ValueError("prompt_wav_path is required when generate_audio=True")
            self._init_token2wav_cache(self.prompt_wav_path)
            self._reset_token2wav_for_new_turn()
        has_ref_audio = ref_audio is not None
        prefix_ids, suffix_ids = self._system_prefill_parts(
            system_prompt,
            self._tools,
            has_ref_audio=has_ref_audio,
        )
        self._feed_ids(prefix_ids)
        if ref_audio is not None:
            data = self.processor.process_audio([np.asarray(ref_audio, dtype=np.float32)])
            embeds_nested = self.model.get_audio_embedding(
                data,
                chunk_length=self.model.config.audio_chunk_length,
            )
            if embeds_nested:
                self.decoder.feed(torch.cat([t for group in embeds_nested for t in group], dim=0))
        self._feed_ids(suffix_ids)
        prefill_ids = prefix_ids + suffix_ids
        self._record_trace(
            "prepare",
            token_ids=prefill_ids,
            token_strs=self._token_pieces(prefill_ids),
            output_render=self.render_token_stream(prefill_ids),
            has_ref_audio=ref_audio is not None,
            generate_audio=self.generate_audio,
            resize_info=resize_info,
        )
        return {
            "prefill_ids": prefill_ids,
            "resize_info": resize_info,
            "output_render": self.render_token_stream(prefill_ids),
            "generate_audio": self.generate_audio,
            "prompt_wav_path": self.prompt_wav_path,
            "has_ref_audio": ref_audio is not None,
        }

    def _ensure_previous_unit_closed(self) -> None:
        if self._non_spoken_slot_open:
            self.streaming_non_spoken_generate(close_reason="budget_reached", defer_feed=True)
        if self._spoken_slot_open:
            self._close_spoken_slot()
        if self._current_unit_open:
            self.finalize_unit()

    def _mark_unit_finalized(self, info: dict) -> None:
        self.units_info.append(info)
        self._current_unit_idx += 1
        self._current_unit_info = None
        self._current_unit_open = False
        self._record_trace("finalize_unit", unit_info=info)

    def _flush_pending_prefill_close(self) -> None:
        pending_close_ids = list(self._pending_prefill_close_ids or [])
        if not pending_close_ids:
            return
        self._pending_prefill_close_ids = []
        self._pending_prefill_unit_info = None
        # output_ids were already updated when the close was deferred; only the
        # decoder/KV state still needs to catch up.
        self.decoder.feed(self.decoder.embed_tokens(pending_close_ids))
        self._record_trace(
            "flush_pending_prefill_close",
            token_ids=pending_close_ids,
            token_strs=self._token_pieces(pending_close_ids),
        )

    def _close_spoken_slot(self, *, append_spoken_slot_eos: bool = False) -> None:
        close_ids = []
        if append_spoken_slot_eos:
            close_ids.append(self.sid(self.K.SPOKEN_SLOT_EOS))
        close_ids.append(self.sid(self.K.AI_SPOKEN_SLOT_END))
        self._feed_ids(close_ids)
        self._spoken_slot_open = False

    def streaming_prefill(
        self,
        audio_waveform=None,
        frame_list=None,
        tool_responses=None,
        sample_rate: int = 16000,
        max_slice_nums: int = 1,
    ) -> dict:
        self._ensure_protocol()
        self._ensure_previous_unit_closed()
        unit = self._current_unit_idx
        self._current_unit_info = {
            "unit": unit,
            "n_audio": 0,
            "has_event": bool(tool_responses),
            "is_listen": None,
            "is_speaking": False,
            "spoken_ids": [],
            "non_spoken_ids": [],
            "non_spoken_terminator": None,
            "closed_spans": [],
        }
        chunks = []
        pending_close_ids = list(self._pending_prefill_close_ids or [])
        if pending_close_ids:
            chunks.append(("ids_no_record", pending_close_ids))
            self._pending_prefill_close_ids = []
        chunks.append(("ids", [self.sid(self.K.UNIT_START)]))
        self._current_unit_open = True

        # TODO: convert frame_list to image embeddings if FC video support is needed.
        del frame_list, max_slice_nums

        if audio_waveform is not None and len(audio_waveform) > 0:
            chunks.append(("ids", [self.sid(self.K.USER_AUDIO_SLOT_START)]))
            audio_embeds, n_audio = self._audio_embeds(audio_waveform, sample_rate=sample_rate)
            self._current_unit_info["n_audio"] = n_audio
            if n_audio > 0:
                self.output_ids.extend([self.sid(self.K.AUDIO_PLACEHOLDER)] * n_audio)
                chunks.append(("embeds", audio_embeds))
            chunks.append(("ids", [self.sid(self.K.USER_AUDIO_SLOT_END)]))

        if tool_responses:
            chunks.append(("ids", self._input_event_slot_ids(tool_responses)))

        chunks.append(("ids", [self.sid(self.K.AI_SPOKEN_SLOT_START)]))
        self._spoken_logits = self._feed_mixed_embeddings(chunks, want_logits=True)
        self._spoken_slot_open = True
        self._record_trace(
            "streaming_prefill",
            n_audio=self._current_unit_info["n_audio"],
            has_event=self._current_unit_info["has_event"],
            tool_responses=[getattr(item, "model_dump", lambda: item)() for item in (tool_responses or [])],
            unit_info=dict(self._current_unit_info),
        )
        return dict(self._current_unit_info)

    def streaming_spoken_generate(self, max_tokens: int = 24, decode_mode: str = "greedy") -> dict:
        self._ensure_protocol()
        if not self._spoken_slot_open:
            raise RuntimeError("spoken slot is not open; call streaming_prefill() first")

        start_time = time.time()
        K = self.K
        spoken_terms = {
            self.sid(K.SPOKEN_SLOT_EOS),
            self.sid(K.SPOKEN_TURN_EOS),
            self.sid(K.LISTEN),
            self.sid(K.TTS_PAD),
        }
        spoken_ids = []
        text_ids = []
        tts_hidden_in_unit = []
        is_listen = False
        is_speaking = False
        turn_eos = False
        logits = self._spoken_logits

        for _ in range(max_tokens):
            nid = self._sample(logits, decode_mode)
            if nid == self.sid(K.AI_SPOKEN_SLOT_END):
                break
            spoken_ids.append(nid)
            self.output_ids.append(nid)
            if nid == self.sid(K.LISTEN):
                is_listen = True
            elif nid == self.sid(K.SPEAK):
                is_speaking = True
            elif nid == self.sid(K.SPOKEN_TURN_EOS):
                turn_eos = True
            elif not self.is_special(nid):
                text_ids.append(nid)
            fed = self.decoder.feed(self.decoder.embed_token(nid), return_logits=True)
            logits = fed[0]
            hidden = fed[1] if len(fed) > 1 else None
            if (
                hidden is not None
                and is_speaking
                and nid not in {self.sid(K.SPEAK), self.sid(K.LISTEN), self.sid(K.TTS_PAD)}
            ):
                tts_hidden_in_unit.append([nid, hidden, bool(turn_eos)])
            if nid in spoken_terms:
                break

        self._close_spoken_slot(append_spoken_slot_eos=turn_eos)
        text = self._flush(text_ids) if text_ids else ""
        llm_end_time = time.time()
        audio_info = self._generate_spoken_audio(
            tts_hidden_in_unit,
            end_of_turn=bool(turn_eos),
        )
        if self._current_unit_info is not None:
            self._current_unit_info["is_listen"] = bool(is_listen)
            self._current_unit_info["is_speaking"] = bool(is_speaking)
            self._current_unit_info["spoken_ids"] = spoken_ids
            if audio_info.get("audio_waveform") is not None:
                self._current_unit_info["audio_sample_rate"] = audio_info.get("audio_sample_rate")
                self._current_unit_info["n_audio_samples"] = int(len(audio_info["audio_waveform"]))
        self._record_trace(
            "streaming_spoken_generate",
            token_ids=list(spoken_ids),
            token_strs=self._token_pieces(spoken_ids),
            text=text,
            is_listen=bool(is_listen),
            is_speaking=bool(is_speaking),
            spoken_turn_eos=bool(turn_eos),
            n_audio_samples=int(len(audio_info["audio_waveform"])) if audio_info.get("audio_waveform") is not None else 0,
        )
        return {
            "is_listen": bool(is_listen),
            "is_speaking": bool(is_speaking),
            "spoken_ids": spoken_ids,
            "spoken_text": text,
            "text": text,
            "spoken_turn_eos": bool(turn_eos),
            "end_of_turn": bool(turn_eos),
            "cost_llm": llm_end_time - start_time,
            **audio_info,
        }

    def _open_non_spoken_slot(self):
        if self._non_spoken_slot_open:
            return
        self._non_spoken_logits = self._feed_ids([self.sid(self.K.AI_NON_SPOKEN_SLOT_START)], want_logits=True)
        self._non_spoken_slot_open = True

    def _close_non_spoken_slot(self, reason: str, *, defer_feed: bool = False) -> dict:
        self._ensure_protocol()
        self._open_non_spoken_slot()
        term_key = {
            "eos": self.K.NON_SPOKEN_EOS,
            "no_action": self.K.NO_ACTION,
            "budget_reached": self.K.NON_SPOKEN_BUDGET_REACHED,
            "hold": self.K.NON_SPOKEN_HOLD,
            "abort": self.K.NON_SPOKEN_ABORT,
        }.get(reason)
        if term_key is None:
            raise ValueError(f"unsupported non-spoken close reason: {reason}")
        tid = self.sid(term_key)
        close_ids = [tid, self.sid(self.K.AI_NON_SPOKEN_SLOT_END)]
        if defer_feed:
            self.output_ids.extend(close_ids)
            self._pending_prefill_close_ids.extend(close_ids)
        else:
            self._feed_ids(close_ids)
        self._non_spoken_slot_open = False
        if self._current_unit_info is not None:
            self._current_unit_info["non_spoken_ids"].append(tid)
            self._current_unit_info["non_spoken_terminator"] = reason
        self._record_trace(
            "streaming_non_spoken_close",
            token_ids=[tid],
            token_strs=self._token_pieces([tid]),
            terminated=True,
            close_reason=reason,
            closed_spans=[],
            text="",
            deferred_feed=defer_feed,
        )
        return {"token_ids": [tid], "terminated": True, "close_reason": reason, "closed_spans": [], "text": ""}

    def _track_non_spoken_token(self, nid: int) -> list:
        closed_spans = []
        K = self.K
        if nid in {
            self.sid(K.NO_ACTION),
            self.sid(K.NON_SPOKEN_EOS),
            self.sid(K.NON_SPOKEN_HOLD),
            self.sid(K.NON_SPOKEN_ABORT),
        }:
            self._non_spoken_mode = None
            self._think_buf = []
            self._tool_call_buf = []
            return closed_spans
        if self._non_spoken_mode is None:
            if nid == self.sid(K.THINK_START):
                self._non_spoken_mode = "think"
                self._think_buf = []
            elif nid == self.sid(K.TOOL_CALL_START):
                self._non_spoken_mode = "tool_call"
                self._tool_call_buf = []
        elif self._non_spoken_mode == "think":
            if nid == self.sid(K.THINK_END):
                text = self._flush(self._think_buf) if self._think_buf else ""
                closed_spans.append({"type": "think", "text": text})
                self._non_spoken_mode = None
                self._think_buf = []
            elif not self.is_special(nid):
                self._think_buf.append(nid)
        elif self._non_spoken_mode == "tool_call":
            if nid == self.sid(K.TOOL_CALL_END):
                wire = self._flush(self._tool_call_buf) if self._tool_call_buf else ""
                closed_spans.append({
                    "type": "tool_call",
                    "wire": wire,
                    "tool_call": self._safe_deserialize_tool_call(wire),
                })
                self._non_spoken_mode = None
                self._tool_call_buf = []
            elif not self.is_special(nid):
                self._tool_call_buf.append(nid)
        return closed_spans

    def streaming_non_spoken_generate(
        self,
        decode_mode: str = "greedy",
        max_tokens: int = 1,
        close_reason: Optional[str] = None,
        defer_feed: bool = False,
    ) -> dict:
        self._ensure_protocol()
        if close_reason is not None:
            return self._close_non_spoken_slot(close_reason, defer_feed=defer_feed)

        self._open_non_spoken_slot()
        K = self.K
        natural_terms = {
            self.sid(K.NON_SPOKEN_EOS): "eos",
            self.sid(K.NO_ACTION): "no_action",
            self.sid(K.NON_SPOKEN_ABORT): "abort",
        }
        token_ids = []
        closed_spans = []
        text_ids = []
        terminated = False
        close = None
        logits = self._non_spoken_logits

        for _ in range(max_tokens):
            nid = self._sample(logits, decode_mode)
            if nid == self.sid(K.AI_NON_SPOKEN_SLOT_END):
                close = "eos"
                terminated = True
                break
            token_ids.append(nid)
            self.output_ids.append(nid)
            if self._current_unit_info is not None:
                self._current_unit_info["non_spoken_ids"].append(nid)
            closed_spans.extend(self._track_non_spoken_token(nid))
            if not self.is_special(nid):
                text_ids.append(nid)
            self._non_spoken_logits = self.decoder.feed(self.decoder.embed_token(nid), return_logits=True)[0]
            logits = self._non_spoken_logits
            if nid in natural_terms:
                close = natural_terms[nid]
                terminated = True
                break

        if terminated:
            self._feed_ids([self.sid(K.AI_NON_SPOKEN_SLOT_END)])
            self._non_spoken_slot_open = False
            if self._current_unit_info is not None:
                self._current_unit_info["non_spoken_terminator"] = close
                self._current_unit_info["closed_spans"].extend(closed_spans)
        elif self._current_unit_info is not None:
            self._current_unit_info["closed_spans"].extend(closed_spans)

        text = self._flush(text_ids) if text_ids else ""
        self._record_trace(
            "streaming_non_spoken_generate",
            token_ids=list(token_ids),
            token_strs=self._token_pieces(token_ids),
            terminated=terminated,
            close_reason=close,
            closed_spans=[self._trace_span(span) for span in closed_spans],
            text=text,
            non_spoken_mode=self._non_spoken_mode,
        )
        return {
            "token_ids": token_ids,
            "terminated": terminated,
            "close_reason": close,
            "closed_spans": closed_spans,
            "text": text,
        }

    def finalize_unit(self) -> dict:
        if not self._current_unit_open:
            return {}
        if self._non_spoken_slot_open:
            self.streaming_non_spoken_generate(close_reason="budget_reached", defer_feed=True)
        if self._spoken_slot_open:
            self._close_spoken_slot()
        info = dict(self._current_unit_info or {"unit": self._current_unit_idx})
        if self._pending_prefill_close_ids:
            self.output_ids.append(self.sid(self.K.UNIT_END))
            self._pending_prefill_close_ids.append(self.sid(self.K.UNIT_END))
            self._pending_prefill_unit_info = info
            self._mark_unit_finalized(info)
            return info
        self._feed_ids([self.sid(self.K.UNIT_END)])
        self._mark_unit_finalized(info)
        return info

    def replay_completed_unit(
        self,
        *,
        audio_waveform=None,
        frame_list=None,
        tool_responses=None,
        sample_rate: int = 16000,
        spoken_token_ids: list[int],
        non_spoken_token_ids: list[int],
        deferred_non_spoken_close: bool = False,
    ) -> dict:
        """Deterministically feed one completed Unit without re-sampling outputs.

        The caller must only use histories whose checkpoint declared no open
        spoken turn, open non-spoken span, pending text delta, or deferred close.
        TTS/audio is intentionally not regenerated during replay.
        """

        self.streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            tool_responses=tool_responses,
            sample_rate=sample_rate,
        )
        K = self.K
        spoken_ids = [int(token_id) for token_id in spoken_token_ids]
        non_spoken_ids = [int(token_id) for token_id in non_spoken_token_ids]

        if spoken_ids:
            self._feed_ids(spoken_ids)
        is_listen = self.sid(K.LISTEN) in spoken_ids
        is_speaking = self.sid(K.SPEAK) in spoken_ids
        if self._current_unit_info is not None:
            self._current_unit_info["is_listen"] = is_listen
            self._current_unit_info["is_speaking"] = is_speaking
            self._current_unit_info["spoken_ids"] = list(spoken_ids)
        self._feed_ids([self.sid(K.AI_SPOKEN_SLOT_END)])
        self._spoken_slot_open = False
        self._spoken_logits = None

        self._open_non_spoken_slot()
        closed_spans = []
        budget_reached_id = self.sid(K.NON_SPOKEN_BUDGET_REACHED)
        if deferred_non_spoken_close:
            if not non_spoken_ids or non_spoken_ids[-1] != budget_reached_id:
                raise RuntimeError(
                    "deferred non-spoken replay requires trailing "
                    "non_spoken_budget_reached"
                )
            immediate_non_spoken_ids = non_spoken_ids[:-1]
        else:
            immediate_non_spoken_ids = non_spoken_ids
        for token_id in immediate_non_spoken_ids:
            closed_spans.extend(self._track_non_spoken_token(token_id))
        if immediate_non_spoken_ids:
            self._feed_ids(immediate_non_spoken_ids)
        if self._current_unit_info is not None:
            self._current_unit_info["non_spoken_ids"] = list(non_spoken_ids)
            self._current_unit_info["closed_spans"].extend(closed_spans)
            close_reason_by_id = {
                self.sid(K.NO_ACTION): "no_action",
                self.sid(K.NON_SPOKEN_EOS): "eos",
                self.sid(K.NON_SPOKEN_HOLD): "hold",
                self.sid(K.NON_SPOKEN_ABORT): "abort",
                budget_reached_id: "budget_reached",
            }
            for token_id in reversed(non_spoken_ids):
                if token_id in close_reason_by_id:
                    self._current_unit_info["non_spoken_terminator"] = (
                        close_reason_by_id[token_id]
                    )
                    break
        if deferred_non_spoken_close:
            deferred_ids = [
                budget_reached_id,
                self.sid(K.AI_NON_SPOKEN_SLOT_END),
                self.sid(K.UNIT_END),
            ]
            self.output_ids.extend(deferred_ids)
            self._pending_prefill_close_ids.extend(deferred_ids)
            self._pending_prefill_unit_info = dict(
                self._current_unit_info or {"unit": self._current_unit_idx}
            )
        else:
            self._feed_ids(
                [
                    self.sid(K.AI_NON_SPOKEN_SLOT_END),
                    self.sid(K.UNIT_END),
                ]
            )
        self._non_spoken_slot_open = False
        self._non_spoken_logits = None

        info = dict(self._current_unit_info or {"unit": self._current_unit_idx})
        self._mark_unit_finalized(info)
        self._record_trace(
            "replay_completed_unit",
            spoken_token_ids=spoken_ids,
            non_spoken_token_ids=non_spoken_ids,
            deferred_non_spoken_close=deferred_non_spoken_close,
            unit_info=info,
        )
        return info

    def resume_boundary_status(self) -> dict:
        """Return model-state constraints for stateless Unit-boundary resume."""

        if self._pending_prefill_close_ids:
            return {
                "status": "unavailable",
                "reason": "unsupported_deferred_close",
            }
        if self._non_spoken_mode is not None:
            return {
                "status": "unavailable",
                "reason": "unsupported_open_span",
                "stream_kind": self._non_spoken_mode,
            }
        if (
            self.tts_past_key_values is not None
            or self.tts_text_start_pos != 0
            or self.tts_current_turn_start_time is not None
        ):
            return {
                "status": "unavailable",
                "reason": "unsupported_spoken_turn_state",
            }
        return {"status": "available"}

    def decode_output_ids(self, output_ids=None, tools=None) -> dict:
        self._ensure_protocol()
        ids = list(self.output_ids if output_ids is None else output_ids)
        K = self.K
        unit_start = self.sid(K.UNIT_START)
        unit_end = self.sid(K.UNIT_END)
        spk_start = self.sid(K.AI_SPOKEN_SLOT_START)
        spk_end = self.sid(K.AI_SPOKEN_SLOT_END)
        nsp_start = self.sid(K.AI_NON_SPOKEN_SLOT_START)
        nsp_end = self.sid(K.AI_NON_SPOKEN_SLOT_END)
        listen = self.sid(K.LISTEN)
        speak = self.sid(K.SPEAK)
        spoken_eos = {self.sid(K.SPOKEN_SLOT_EOS), self.sid(K.SPOKEN_TURN_EOS)}
        think_start = self.sid(K.THINK_START)
        think_end = self.sid(K.THINK_END)
        tc_start = self.sid(K.TOOL_CALL_START)
        tc_end = self.sid(K.TOOL_CALL_END)
        no_action = self.sid(K.NO_ACTION)
        nsp_terms = {
            self.sid(K.NON_SPOKEN_EOS): "eos",
            self.sid(K.NON_SPOKEN_BUDGET_REACHED): "budget_reached",
            self.sid(K.NON_SPOKEN_HOLD): "hold",
            self.sid(K.NON_SPOKEN_ABORT): "abort",
        }

        units = []
        cur = None
        slot = None
        spoken_buf = []
        mode = None
        think_buf = []
        tc_buf = []
        think_completed = []
        tool_calls = []

        def new_unit():
            return {"is_listen": None, "spoken_text": "", "non_spoken_terminator": None, "raw_non_spoken": ""}

        for tid in ids:
            if tid == unit_start:
                cur = new_unit()
                slot = None
                continue
            if tid == unit_end:
                if cur is not None:
                    units.append(cur)
                cur = None
                slot = None
                continue
            if cur is None:
                continue
            if tid == spk_start:
                slot = "spoken"
                spoken_buf = []
                if cur["is_listen"] is None:
                    cur["is_listen"] = False
                continue
            if tid == spk_end:
                cur["spoken_text"] += self._flush(spoken_buf) if spoken_buf else ""
                spoken_buf = []
                slot = None
                continue
            if tid == nsp_start:
                slot = "non_spoken"
                continue
            if tid == nsp_end:
                slot = None
                continue
            if slot == "spoken":
                if tid == listen:
                    cur["is_listen"] = True
                elif tid == speak:
                    cur["is_listen"] = False
                elif tid in spoken_eos:
                    pass
                elif not self.is_special(tid):
                    spoken_buf.append(tid)
                continue
            if slot == "non_spoken":
                cur["raw_non_spoken"] += self.id2name[tid] if self.is_special(tid) else self._flush([tid])
                if mode is None:
                    if tid == think_start:
                        mode = "think"
                        think_buf = []
                    elif tid == tc_start:
                        mode = "tool_call"
                        tc_buf = []
                    elif tid == no_action:
                        cur["non_spoken_terminator"] = "no_action"
                    elif tid in nsp_terms:
                        cur["non_spoken_terminator"] = nsp_terms[tid]
                elif mode == "think":
                    if tid == think_end:
                        think_completed.append(self._flush(think_buf) if think_buf else "")
                        mode = None
                    elif tid in nsp_terms:
                        cur["non_spoken_terminator"] = nsp_terms[tid]
                    elif not self.is_special(tid):
                        think_buf.append(tid)
                elif mode == "tool_call":
                    if tid == tc_end:
                        tool_calls.append(self._safe_deserialize_tool_call(self._flush(tc_buf), tools or self._tools))
                        mode = None
                    elif tid in nsp_terms:
                        cur["non_spoken_terminator"] = nsp_terms[tid]
                    elif not self.is_special(tid):
                        tc_buf.append(tid)

        if mode == "think" and think_buf:
            think_completed.append(self._flush(think_buf))
        elif mode == "tool_call" and tc_buf:
            tool_calls.append(self._safe_deserialize_tool_call(self._flush(tc_buf), tools or self._tools))

        return {
            "units": units,
            "spoken_text": "".join(u["spoken_text"] for u in units),
            "think_text": "".join(think_completed),
            "tool_calls": tool_calls,
            "output_ids": ids,
            "output_render": self.render_token_stream(ids),
        }

    def cleanup(self) -> None:
        self._flush_pending_prefill_close()
        self._reset_streaming_state()
