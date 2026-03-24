"""C++ llama.cpp-omni 推理后端适配层

通过 HTTP 调用 C++ llama-server 的 omni 接口，实现与 MiniCPMOWorker 相同的方法签名，
作为 PyTorch 后端的 drop-in 替换。

生命周期映射：
    服务启动   → omni_init（加载 APM/VPM/TTS/Token2Wav，复用 LLM）
    新会话     → update_session_config（清空 KV cache，重新 prefill system prompt）
    每个 chunk → /v1/stream/prefill + /v1/stream/decode
    打断       → /v1/stream/break
    会话结束   → 清理输出目录
"""

import os
import re
import sys
import io
import gc
import json
import time
import base64
import shutil
import signal
import logging
import tempfile
import platform
import threading
import subprocess
from typing import Optional, List, Dict, Any, Iterator
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger("cpp_backend")

_AUDIO_INPUT_SR = 16000
_AUDIO_OUTPUT_SR = 24000

# System prompt 模板 — 来自 modeling_minicpmo.py audio_assistant 模式
# key: (duplex, lang) → (voice_clone_prompt, assistant_prompt)
_SYSTEM_PROMPTS: Dict[tuple, Dict[str, str]] = {
    # 双工模式 — 语言无关，固定英文 prompt
    (True, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|><|im_end|>\n",
    },
    (True, "en"): {
        "voice_clone_prompt": "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|><|im_end|>\n",
    },
    # 非双工 — 中文
    (False, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。"
                              "请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
                              "<|im_end|>\n<|im_start|>user\n",
    },
    # 非双工 — 英文
    (False, "en"): {
        "voice_clone_prompt": "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|>Please assist users while maintaining this voice style. "
                              "Please answer the user's questions seriously and in a high quality. "
                              "Please chat with the user in a highly human-like and oral style. "
                              "You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
                              "<|im_end|>\n<|im_start|>user\n",
    },
}


def _get_system_prompts(duplex: bool, lang: str = "zh") -> Dict[str, str]:
    """根据模式和语言返回 voice_clone_prompt / assistant_prompt"""
    return _SYSTEM_PROMPTS.get((duplex, lang), _SYSTEM_PROMPTS[(duplex, "zh")])


class CppBackendWorker:
    """C++ llama-server 推理后端

    实现与 MiniCPMOWorker 相同的方法签名，内部通过 HTTP 调用 C++ 服务。
    """

    def __init__(
        self,
        llamacpp_root: str,
        model_dir: str,
        gpu_id: int = 0,
        ref_audio_path: Optional[str] = None,
        duplex_pause_timeout: float = 60.0,
        llm_model: str = "",
        cpp_server_port: Optional[int] = None,
        ctx_size: int = 8192,
        n_gpu_layers: int = 99,
        **kwargs,
    ):
        self.llamacpp_root = llamacpp_root
        self.model_dir = model_dir
        self.gpu_id = gpu_id
        self.ref_audio_path = ref_audio_path
        self.duplex_pause_timeout = duplex_pause_timeout
        self.llm_model = llm_model or self._auto_detect_llm_model(model_dir)
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers

        from worker import WorkerState, WorkerStatus
        self.state = WorkerState()
        self.processor = None  # compatibility — used for kv_cache_length etc.

        self._cpp_server_port = cpp_server_port or (19060 + gpu_id)
        self._cpp_server_url = f"http://127.0.0.1:{self._cpp_server_port}"
        self._cpp_process: Optional[subprocess.Popen] = None
        self._http_client = None  # httpx.Client (sync)
        self._temp_dir = tempfile.mkdtemp(prefix="cpp_backend_")
        self._output_dir = os.path.join(llamacpp_root, f"tools/omni/output_{self._cpp_server_port}")
        self._last_duplex_mode: Optional[bool] = None
        self._last_media_type: int = 2
        self._last_lang: str = "zh"

        self._duplex_chunk_counter: int = 0
        self._current_session_id: Optional[str] = None
        self._round_number: int = 0
        self._sent_wav_files: set = set()

    # ================================================================
    # Model loading (maps to omni_init)
    # ================================================================

    def load_model(self) -> None:
        """启动 C++ llama-server 并调用 omni_init 加载所有子模型"""
        from worker import WorkerStatus
        self.state.status = WorkerStatus.LOADING
        logger.info(f"[GPU {self.gpu_id}] Starting C++ llama-server...")

        import httpx
        self._http_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))

        self._start_cpp_server()
        self._call_omni_init(media_type=2, duplex_mode=True)
        self._last_duplex_mode = True

        self.state.status = WorkerStatus.IDLE
        logger.info(f"[GPU {self.gpu_id}] C++ backend ready")

    @property
    def kv_cache_length(self) -> int:
        return 0

    # ================================================================
    # Duplex
    # ================================================================

    def duplex_prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        media_type: int = 2,
        lang: Optional[str] = None,
    ) -> str:
        """Duplex 准备 → update_session_config"""
        self._reset_output_dir()
        self._duplex_chunk_counter = 0
        self._round_number = 0
        self._sent_wav_files = set()
        voice_audio = ref_audio_path or self.ref_audio_path or ""
        self._call_update_session_config(
            media_type=media_type,
            duplex_mode=True,
            voice_audio=voice_audio,
            lang=lang,
        )
        os.makedirs(os.path.join(self._output_dir, "tts_wav"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "tts_txt"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "llm_debug"), exist_ok=True)
        return system_prompt_text or "Streaming Duplex Conversation."

    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[list] = None,
        max_slice_nums: int = 1,
    ) -> Dict[str, Any]:
        """Duplex 预填充 → /v1/stream/prefill"""
        cnt = self._duplex_chunk_counter
        self._duplex_chunk_counter += 1

        temp_audio = ""
        if audio_waveform is not None and len(audio_waveform) > 0:
            temp_audio = self._save_audio_to_temp(audio_waveform, f"duplex_{cnt}")

        temp_image = ""
        n_vision_images = 0
        if frame_list:
            temp_image = self._save_pil_image_to_temp(frame_list[0], f"duplex_{cnt}")
            n_vision_images = 1

        self._call_prefill(temp_audio, temp_image, cnt, max_slice_nums)

        if frame_list and len(frame_list) > 1:
            for i, frame in enumerate(frame_list[1:], 1):
                extra_img = self._save_pil_image_to_temp(frame, f"duplex_{cnt}_f{i}")
                self._call_prefill("", extra_img, cnt + i, max_slice_nums)
                n_vision_images += 1

        self._cleanup_temp_files(temp_audio, temp_image)
        return {"n_vision_images": n_vision_images}

    def duplex_generate(self, force_listen: bool = False) -> "DuplexGenerateResult":
        """Duplex 生成 → /v1/stream/decode + scan WAV files"""
        from core.schemas.duplex import DuplexGenerateResult

        t0 = time.perf_counter()

        resp = self._http_client.post(
            f"{self._cpp_server_url}/v1/stream/decode",
            json={"stream": True},
            timeout=600.0,
        )

        is_listen = True
        end_of_turn = False
        texts = []

        if resp.status_code == 200:
            for line in resp.text.splitlines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if "is_listen" in event:
                    is_listen = event["is_listen"]
                if "end_of_turn" in event:
                    end_of_turn = event["end_of_turn"]
                if event.get("text"):
                    texts.append(event["text"])
                if event.get("content"):
                    texts.append(event["content"])
                # if event.get("stop"):
                #     end_of_turn = True

        text = "".join(texts)
        cost_all_ms = (time.perf_counter() - t0) * 1000

        return DuplexGenerateResult(
            is_listen=is_listen,
            text=text,
            audio_data=None,
            end_of_turn=end_of_turn,
            current_time=self._duplex_chunk_counter,
            cost_all_ms=round(cost_all_ms, 1),
        )

    def duplex_finalize(self) -> None:
        """C++ 内部自动管理 KV cache，此处为空操作"""
        pass

    def duplex_stop(self) -> None:
        """Duplex 停止 → /v1/stream/break"""
        self._last_break_time = time.time()
        try:
            self._http_client.post(
                f"{self._cpp_server_url}/v1/stream/break",
                json={"reason": "duplex_stop"},
                timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"duplex_stop break call failed: {e}")

    def duplex_cleanup(self) -> None:
        """清理输出目录 + 清空 KV cache，确保下次会话从干净状态开始"""
        self._reset_output_dir()
        try:
            self._call_update_session_config(
                media_type=self._last_media_type,
                duplex_mode=self._last_duplex_mode if self._last_duplex_mode is not None else True,
                voice_audio=self.ref_audio_path or "",
            )
        except Exception as e:
            logger.warning(f"duplex_cleanup session reset failed: {e}")
        gc.collect()

    def is_cpp_healthy(self) -> bool:
        """检查底层 C++ llama-server 是否活着（进程 + HTTP /health）"""
        proc = self._cpp_process
        if proc is None or proc.poll() is not None:
            return False
        try:
            import requests as _req
            r = _req.get(f"{self._cpp_server_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _stop_cpp_server(self) -> None:
        if self._cpp_process is not None:
            proc = self._cpp_process
            try:
                if proc.poll() is None:
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        proc.terminate()

                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            pgid = os.getpgid(proc.pid)
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except Exception:
                            proc.kill()
                        proc.wait(timeout=5)
            except Exception as e:
                logger.warning(f"_stop_cpp_server: {e}")
            finally:
                self._cpp_process = None
                logger.info("llama-server stopped")

    def full_reinit(self) -> None:
        """每次会话结束后完全重启 llama-server，保证下次会话状态绝对干净。"""
        self._round_number = 0
        self._last_duplex_mode = None
        self._last_media_type = 2
        t_break = getattr(self, '_last_break_time', 0.0)
        if t_break > 0.0:
            flag_paths = [
                os.path.join(self._output_dir, "generation_done.flag"),
                os.path.join(self._output_dir, "tts_wav", "generation_done.flag"),
            ]

            def _flag_exists() -> bool:
                latest_round = self._find_latest_round_dir()
                if latest_round:
                    latest_flag = os.path.join(latest_round, "tts_wav", "generation_done.flag")
                    if latest_flag not in flag_paths:
                        flag_paths.append(latest_flag)
                return any(os.path.exists(p) for p in flag_paths)

            if not _flag_exists():
                for _ in range(20):
                    time.sleep(0.5)
                    if _flag_exists():
                        break
                else:
                    logger.warning(
                        "full_reinit: T2W completion flag not seen within 10s, proceeding; "
                        f"checked_paths={flag_paths}"
                    )
        try:
            logger.info("full_reinit: stopping llama-server...")
            self._stop_cpp_server()
            logger.info("full_reinit: restarting llama-server...")
            self._start_cpp_server()
            self._call_omni_init(media_type=2, duplex_mode=True)
            logger.info("full_reinit: omni context re-initialized successfully")
        except Exception as e:
            logger.error(f"full_reinit failed: {e}", exc_info=True)
            # 关键约束：重初始化失败时必须向上抛出，禁止调用方误判为可分配。
            raise

    # ================================================================
    # Half-Duplex
    # ================================================================

    def half_duplex_prefill(self, request) -> str:
        """Half-Duplex 预填充"""
        from core.processors.base import MiniCPMOProcessorMixin
        mixin = MiniCPMOProcessorMixin()

        for msg in request.messages:
            content = mixin._convert_content_to_model_format(msg.content)
            for item in content:
                if isinstance(item, np.ndarray):
                    temp_audio = self._save_audio_to_temp(item, f"hdx_{self._duplex_chunk_counter}")
                    self._call_prefill(temp_audio, "", self._duplex_chunk_counter)
                    self._duplex_chunk_counter += 1
                    self._cleanup_temp_files(temp_audio)

        return "prefilled"

    def half_duplex_init_tts(self, ref_audio_data: Optional[np.ndarray] = None) -> None:
        """TTS 在 omni_init 时已初始化，此处为空操作"""
        pass

    def _parse_sse_text(self, resp_text: str) -> str:
        """从 C++ decode SSE 响应中提取所有文本内容"""
        pieces = []
        for line in resp_text.splitlines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            content = event.get("content", "")
            if content:
                pieces.append(content)
            if event.get("stop"):
                break
        return "".join(pieces)

    def half_duplex_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        length_penalty: float = 1.1,
    ) -> "Iterator[StreamingChunk]":
        """Half-Duplex 生成：流式读取 SSE 文本，同时交错 yield 已生成的 WAV 音频"""
        from core.schemas.streaming import StreamingChunk
        import soundfile as sf

        t0 = time.perf_counter()
        cur_round = self._round_number
        chunk_idx = 0

        logger.info(f"[HalfDuplex] decode start, round_idx={cur_round}")

        sent_wav: set = set()
        tts_dir_cache: Optional[str] = None

        def _find_tts_dir():
            nonlocal tts_dir_cache
            if tts_dir_cache and os.path.isdir(tts_dir_cache):
                return tts_dir_cache
            rd = self._find_latest_round_dir()
            if rd:
                d = os.path.join(rd, "tts_wav")
                if os.path.isdir(d):
                    tts_dir_cache = d
                    return d
            return None

        def _yield_new_wavs():
            d = _find_tts_dir()
            if not d:
                return
            try:
                files = sorted(
                    [f for f in os.listdir(d) if f.startswith("wav_") and f.endswith(".wav")],
                    key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
                )
            except OSError:
                return
            for wf in files:
                if wf in sent_wav:
                    continue
                wp = os.path.join(d, wf)
                try:
                    data, _sr = sf.read(wp)
                    if len(data) > 0:
                        if data.dtype != np.float32:
                            data = data.astype(np.float32)
                        yield base64.b64encode(data.tobytes()).decode("utf-8")
                        sent_wav.add(wf)
                except Exception:
                    pass

        sse_text_pieces = []
        try:
            with self._http_client.stream(
                "POST",
                f"{self._cpp_server_url}/v1/stream/decode",
                json={"stream": True, "round_idx": cur_round},
                timeout=600.0,
            ) as resp:
                if resp.status_code != 200:
                    logger.error(f"C++ decode failed: {resp.status_code}")
                    self._round_number += 1
                    yield StreamingChunk(chunk_index=0, is_final=True)
                    return

                buf = ""
                for raw_chunk in resp.iter_text():
                    buf += raw_chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        content = event.get("content", "")
                        if content:
                            sse_text_pieces.append(content)
                            yield StreamingChunk(
                                chunk_index=chunk_idx,
                                text_delta=content,
                                is_final=False,
                                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                            )
                            chunk_idx += 1
                        if event.get("stop"):
                            break

                    if generate_audio:
                        for audio_b64 in _yield_new_wavs():
                            yield StreamingChunk(
                                chunk_index=chunk_idx,
                                audio_data=audio_b64,
                                is_final=False,
                                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                            )
                            chunk_idx += 1
        except Exception as e:
            logger.error(f"[HalfDuplex] SSE stream error: {e}")

        decode_elapsed = (time.perf_counter() - t0) * 1000
        sse_text = "".join(sse_text_pieces)
        wav_during_sse = len(sent_wav)
        logger.info(f"[HalfDuplex] decode done in {decode_elapsed:.0f}ms, text={len(sse_text)} chars, "
                     f"wav_sent_during_sse={wav_during_sse}")

        self._round_number += 1

        if not generate_audio:
            if chunk_idx == 0 and sse_text:
                yield StreamingChunk(chunk_index=0, text_delta=sse_text, is_final=True)
            elif chunk_idx > 0:
                yield StreamingChunk(chunk_index=chunk_idx, is_final=True)
            else:
                yield StreamingChunk(chunk_index=0, is_final=True)
            return

        audio_chunk_count = wav_during_sse
        t_post = time.time()
        while time.time() - t_post < 120.0:
            for audio_b64 in _yield_new_wavs():
                yield StreamingChunk(
                    chunk_index=chunk_idx,
                    audio_data=audio_b64,
                    is_final=False,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                chunk_idx += 1
                audio_chunk_count += 1

            d = _find_tts_dir()
            if d and os.path.exists(os.path.join(d, "generation_done.flag")):
                for audio_b64 in _yield_new_wavs():
                    yield StreamingChunk(
                        chunk_index=chunk_idx,
                        audio_data=audio_b64,
                        is_final=False,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                    )
                    chunk_idx += 1
                    audio_chunk_count += 1
                break

            time.sleep(0.15)

        logger.info(f"[HalfDuplex] streamed {audio_chunk_count} wav chunks "
                     f"({wav_during_sse} during SSE, {audio_chunk_count - wav_during_sse} after)")

        if chunk_idx == 0:
            yield StreamingChunk(chunk_index=0, is_final=True, text_delta=sse_text or None)
        else:
            yield StreamingChunk(chunk_index=chunk_idx, is_final=True)

    def reset_half_duplex_session(self, lang: Optional[str] = None, ref_audio_path: Optional[str] = None) -> None:
        """重置 Half-Duplex 会话"""
        voice_audio = ref_audio_path or self.ref_audio_path or ""
        self._call_update_session_config(
            media_type=1,
            duplex_mode=False,
            voice_audio=voice_audio,
            lang=lang,
        )
        self._duplex_chunk_counter = 0
        self._round_number = 0
        self._reset_output_dir()

    # ================================================================
    # Chat
    # ================================================================

    def chat(self, request) -> "ChatResponse":
        """Chat 推理"""
        from core.schemas.chat import ChatResponse
        from core.processors.base import MiniCPMOProcessorMixin

        self._call_update_session_config(
            media_type=2,
            duplex_mode=False,
            voice_audio=self.ref_audio_path or "",
        )
        self._reset_output_dir()
        self._round_number = 0

        mixin = MiniCPMOProcessorMixin()
        cnt = 0
        for msg in request.messages:
            content = mixin._convert_content_to_model_format(msg.content)
            for item in content:
                if isinstance(item, np.ndarray):
                    temp_audio = self._save_audio_to_temp(item, f"chat_{cnt}")
                    self._call_prefill(temp_audio, "", cnt)
                    self._cleanup_temp_files(temp_audio)
                    cnt += 1
                elif isinstance(item, str):
                    pass

        resp = self._http_client.post(
            f"{self._cpp_server_url}/v1/stream/decode",
            json={"stream": True, "round_idx": self._round_number},
            timeout=600.0,
        )
        self._round_number += 1

        sse_text = self._parse_sse_text(resp.text) if resp.status_code == 200 else ""
        wav_b64, _ = self._collect_wav_output(sse_text=sse_text)

        return ChatResponse(
            text=sse_text,
            audio_data=wav_b64,
            audio_sample_rate=_AUDIO_OUTPUT_SR if wav_b64 else None,
            success=True,
        )

    def chat_prefill(self, session_id, msgs, omni_mode=False, max_slice_nums=None,
                     use_tts_template=False, enable_thinking=False, lang: Optional[str] = None,
                     ref_audio_path: Optional[str] = None,
                     reset_context: bool = True) -> str:
        """Chat prefill — reset_context=True 时重置会话上下文"""
        media_type = 2 if omni_mode else 1
        if reset_context:
            self._call_update_session_config(
                media_type=media_type,
                duplex_mode=False,
                voice_audio=ref_audio_path or self.ref_audio_path or "",
                lang=lang,
            )
        logger.info(
            f"[ChatPrefill] session={session_id} omni_mode={omni_mode} media_type={media_type} "
            f"lang={lang or self._last_lang} reset_context={reset_context} "
            f"ref_audio_path={ref_audio_path or self.ref_audio_path or ''}"
        )
        if reset_context:
            self._reset_output_dir()
            self._round_number = 0

        cnt = self._round_number
        prefill_msgs: List[Dict[str, Any]] = []
        # if msgs and reset_context and len(msgs) > 1:
        #     prefill_msgs.append(msgs[0])
        if msgs:
            prefill_msgs.append(msgs[-1])

        for msg in prefill_msgs:
            content_list = msg.get("content", [])
            if isinstance(content_list, str):
                continue
            if not isinstance(content_list, list):
                content_list = [content_list]
            for item in content_list:
                if isinstance(item, np.ndarray):
                    temp_audio = self._save_audio_to_temp(item, f"chat_pf_{cnt}")
                    self._call_prefill(temp_audio, "", cnt)
                    self._cleanup_temp_files(temp_audio)
                    cnt += 1
                elif hasattr(item, 'size'):
                    temp_img = self._save_pil_image_to_temp(item, f"chat_pf_{cnt}")
                    self._call_prefill("", temp_img, cnt, max_slice_nums or -1)
                    self._cleanup_temp_files("", temp_img)
                    cnt += 1
        return "prefilled"

    def chat_non_streaming_generate(self, session_id, **kwargs):
        """Chat 非流式生成"""
        cur_round = self._round_number
        resp = self._http_client.post(
            f"{self._cpp_server_url}/v1/stream/decode",
            json={"stream": True, "round_idx": cur_round},
            timeout=600.0,
        )
        self._round_number += 1

        sse_text = self._parse_sse_text(resp.text) if resp.status_code == 200 else ""
        wav_b64, _ = self._collect_wav_output(sse_text=sse_text)

        if wav_b64:
            audio_bytes = base64.b64decode(wav_b64)
            waveform = np.frombuffer(audio_bytes, dtype=np.float32)
            return sse_text, waveform
        return sse_text

    def chat_streaming_generate(self, session_id, generate_audio=True,
                                max_new_tokens=256, length_penalty=1.1):
        """Chat 流式生成"""
        yield from self.half_duplex_generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    # ================================================================
    # Internal: C++ server management
    # ================================================================

    def _start_cpp_server(self) -> None:
        server_bin = self._find_server_binary()
        model_path = os.path.join(self.model_dir, self.llm_model)

        if not os.path.exists(server_bin):
            raise RuntimeError(f"llama-server not found: {server_bin}")
        if not os.path.exists(model_path):
            raise RuntimeError(f"LLM model not found: {model_path}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        cmd = [
            server_bin,
            "--host", "0.0.0.0",
            "--port", str(self._cpp_server_port),
            "--model", model_path,
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--repeat-penalty", "1.05",
            "--temp", "0.7",
        ]

        logger.info(f"Starting C++ server: {' '.join(cmd)}")

        self._cpp_process = subprocess.Popen(
            cmd, env=env, cwd=self.llamacpp_root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, encoding="utf-8", errors="replace",
            start_new_session=True,
        )

        def _log_reader():
            try:
                for line in self._cpp_process.stdout:
                    stripped = line.rstrip()
                    if any(kw in stripped for kw in ("TTS", "T2W", "LLM->TTS", "wav_", "tts_thread", "generate_audio", "speek_done", "break_event", "lang", "language", "omni_set_language", "prefill", "change")):
                        logger.info(f"[CPP] {stripped}")
                    else:
                        logger.debug(f"[CPP] {stripped}")
            except Exception:
                pass

        threading.Thread(target=_log_reader, daemon=True).start()

        import requests
        for i in range(300):
            try:
                r = requests.get(f"{self._cpp_server_url}/health", timeout=2)
                if r.status_code == 200:
                    logger.info(f"C++ server ready after {i+1}s")
                    return
            except Exception:
                pass
            time.sleep(1)

        raise RuntimeError("C++ server startup timeout (300s)")

    def _find_server_binary(self) -> str:
        candidates = [
            os.path.join(self.llamacpp_root, "build/bin/llama-server"),
            os.path.join(self.llamacpp_root, "build/bin/Release/llama-server"),
        ]
        if platform.system() != "Windows":
            candidates.append(os.path.join(self.llamacpp_root, "build-x64-linux-cuda-release/bin/llama-server"))
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def _call_omni_init(
        self,
        media_type: int = 2,
        duplex_mode: bool = True,
        lang: Optional[str] = None,
    ) -> None:
        tts_bin_dir = os.path.join(self.model_dir, "tts")
        os.makedirs(self._output_dir, exist_ok=True)

        req_body = {
            "media_type": media_type,
            "use_tts": True,
            "duplex_mode": duplex_mode,
            "model_dir": self.model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": "gpu:0",
            "output_dir": self._output_dir,
        }

        if self.ref_audio_path and os.path.exists(self.ref_audio_path):
            req_body["voice_audio"] = self.ref_audio_path

        effective_lang = lang or self._last_lang
        prompts = _get_system_prompts(duplex_mode, effective_lang)
        req_body["voice_clone_prompt"] = prompts["voice_clone_prompt"]
        req_body["assistant_prompt"] = prompts["assistant_prompt"]
        if lang:
            self._last_lang = lang

        logger.info(
            f"Calling omni_init: media_type={media_type}, duplex={duplex_mode}, "
            f"lang={effective_lang}"
        )
        resp = self._http_client.post(
            f"{self._cpp_server_url}/v1/stream/omni_init",
            json=req_body,
            timeout=120.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"omni_init failed: {resp.text}")
        logger.info(f"omni_init success: {resp.json()}")

    def _call_update_session_config(
        self,
        media_type: int = 2,
        duplex_mode: bool = True,
        voice_audio: str = "",
        lang: Optional[str] = None,
    ) -> None:
        mode_changed = (
            self._last_duplex_mode is not None and
            self._last_duplex_mode != duplex_mode
        )
        media_type_changed = (self._last_media_type != media_type)

        if mode_changed or media_type_changed:
            # Full re-init when switching duplex_mode or media_type.
            # This restarts all TTS/T2W threads with the correct function,
            # avoiding subtle state corruption from mode switches.
            logger.info(
                "session mode changed "
                f"(duplex: {self._last_duplex_mode} -> {duplex_mode}, "
                f"media_type: {self._last_media_type} -> {media_type}), "
                "calling omni_init for clean restart"
            )
            self._call_omni_init(media_type=media_type, duplex_mode=duplex_mode, lang=lang)
            self._last_duplex_mode = duplex_mode
            self._last_media_type = media_type
            # omni_init already prefills voice_audio if set in ref_audio_path
            # return

        self._last_duplex_mode = duplex_mode
        self._last_media_type = media_type

        # Same mode — lightweight reset via break + update_session_config
        try:
            self._http_client.post(
                f"{self._cpp_server_url}/v1/stream/break",
                json={"reason": "session_config_change"},
                timeout=10.0,
            )
            time.sleep(0.1)
        except Exception:
            pass

        effective_lang = lang or self._last_lang
        prompts = _get_system_prompts(duplex_mode, effective_lang)

        req_body: Dict[str, Any] = {
            "media_type": media_type,
            "duplex_mode": duplex_mode,
            "voice_clone_prompt": prompts["voice_clone_prompt"],
            "assistant_prompt": prompts["assistant_prompt"],
        }
        if voice_audio and os.path.exists(voice_audio):
            req_body["voice_audio"] = voice_audio
        if lang:
            self._last_lang = lang

        resp = self._http_client.post(
            f"{self._cpp_server_url}/v1/stream/update_session_config",
            json=req_body,
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"update_session_config failed: {resp.text}")

    def _call_prefill(self, audio_path: str, img_path: str, cnt: int,
                      max_slice_nums: int = -1) -> None:
        req_body: Dict[str, Any] = {
            "audio_path_prefix": audio_path,
            "img_path_prefix": img_path,
            "cnt": cnt,
        }
        if max_slice_nums > 0:
            req_body["max_slice_nums"] = max_slice_nums

        resp = self._http_client.post(
            f"{self._cpp_server_url}/v1/stream/prefill",
            json=req_body,
            timeout=30.0,
        )
        if resp.status_code != 200:
            logger.error(f"prefill failed (cnt={cnt}): {resp.text}")

    # ================================================================
    # Internal: data conversion helpers
    # ================================================================

    def _save_audio_to_temp(self, audio_np: np.ndarray, prefix: str) -> str:
        import soundfile as sf

        MIN_SAMPLES = 1600
        if len(audio_np) < MIN_SAMPLES:
            audio_np = np.pad(audio_np, (0, MIN_SAMPLES - len(audio_np)), mode="constant")

        path = os.path.join(self._temp_dir, f"{prefix}.wav")
        audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
        sf.write(path, audio_np, _AUDIO_INPUT_SR, format="WAV", subtype="PCM_16")
        return path

    def _save_pil_image_to_temp(self, pil_image, prefix: str) -> str:
        path = os.path.join(self._temp_dir, f"{prefix}.png")
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(path, format="PNG")
        return path

    def _cleanup_temp_files(self, *paths: str) -> None:
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ================================================================
    # Internal: collect WAV output from C++ tts_wav directory
    # ================================================================

    def _iter_wav_chunks_incremental(self, timeout: float = 120.0) -> "Iterator[str]":
        """增量式收集 WAV：每出现一个新 WAV 文件就立即 yield base64 音频，不等全部完成"""
        import soundfile as sf

        # Wait up to 15s for the round directory to appear (C++ TTS creates it async)
        round_dir = None
        t_wait = time.time()
        while time.time() - t_wait < 15.0:
            round_dir = self._find_latest_round_dir()
            if round_dir:
                break
            # Also check base output dir for duplex-mode WAV files
            direct_tts = os.path.join(self._output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                round_dir = self._output_dir
                break
            time.sleep(0.2)
        if not round_dir:
            logger.warning("_iter_wav_chunks_incremental: no round/tts_wav dir found after 15s")
            return

        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        flag_path = os.path.join(tts_wav_dir, "generation_done.flag")
        sent_files: set = set()
        t0 = time.time()

        while time.time() - t0 < timeout:
            if not os.path.exists(tts_wav_dir):
                time.sleep(0.1)
                continue

            current_files = sorted(
                [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
                key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
            )

            new_files = [f for f in current_files if f not in sent_files]
            for wf in new_files:
                wp = os.path.join(tts_wav_dir, wf)
                try:
                    data, _sr = sf.read(wp)
                    if len(data) == 0:
                        continue
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    yield base64.b64encode(data.tobytes()).decode("utf-8")
                    sent_files.add(wf)
                except Exception as e:
                    logger.warning(f"Failed to read {wf}: {e}")

            if os.path.exists(flag_path):
                final_files = sorted(
                    [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
                    key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
                )
                for wf in final_files:
                    if wf in sent_files:
                        continue
                    wp = os.path.join(tts_wav_dir, wf)
                    try:
                        data, _sr = sf.read(wp)
                        if len(data) > 0:
                            if data.dtype != np.float32:
                                data = data.astype(np.float32)
                            yield base64.b64encode(data.tobytes()).decode("utf-8")
                            sent_files.add(wf)
                    except Exception:
                        pass
                return

            time.sleep(0.15)

        logger.warning(f"_iter_wav_chunks_incremental timed out after {timeout}s")

    def _find_latest_round_dir(self) -> Optional[str]:
        """找到最新的 round_NNN 目录"""
        if not os.path.exists(self._output_dir):
            return None
        rounds = sorted(
            [d for d in os.listdir(self._output_dir)
             if d.startswith("round_") and os.path.isdir(os.path.join(self._output_dir, d))],
            reverse=True,
        )
        if rounds:
            return os.path.join(self._output_dir, rounds[0])
        return None

    def _wait_for_generation_done(self, round_dir: str, timeout: float = 120.0) -> bool:
        """等待 C++ TTS 异步生成完成（generation_done.flag 出现）"""
        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        flag_path = os.path.join(tts_wav_dir, "generation_done.flag")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(flag_path):
                return True
            time.sleep(0.1)
        logger.warning(f"Timed out waiting for generation_done.flag ({timeout}s)")
        return False

    def _collect_wav_output_nowait(self, sse_text: str = "") -> tuple:
        """非阻塞版 WAV 收集：只拿新增的 WAV 文件，跳过已发送的，不做任何等待。

        用于 duplex 场景——TTS 异步生成 WAV，每个 chunk 只取增量部分。
        """
        import soundfile as sf

        round_dir = self._find_latest_round_dir()
        if not round_dir:
            direct_tts = os.path.join(self._output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                round_dir = self._output_dir
        if not round_dir:
            return None, sse_text

        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(tts_wav_dir):
            return None, sse_text

        all_files = os.listdir(tts_wav_dir)
        if all_files:
            logger.info(f"[WAV nowait] dir={tts_wav_dir}, all_files={sorted(all_files)[:10]}, sent={len(self._sent_wav_files)}")

        wav_files = sorted(
            [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
            key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
        )
        new_files = [f for f in wav_files if f not in self._sent_wav_files]
        if not new_files:
            return None, sse_text

        all_audio = []
        for wf in new_files:
            wp = os.path.join(tts_wav_dir, wf)
            try:
                data, sr = sf.read(wp)
                if len(data) > 0:
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    all_audio.append(data)
                    self._sent_wav_files.add(wf)
            except Exception:
                pass

        if not all_audio:
            return None, sse_text

        combined = np.concatenate(all_audio)
        audio_b64 = base64.b64encode(combined.astype(np.float32).tobytes()).decode("utf-8")
        return audio_b64, sse_text

    def _collect_wav_output(self, sse_text: str = "") -> tuple:
        """收集所有 WAV 文件，合并为一个 base64 float32 PCM 字符串 + 文本

        Returns:
            (audio_base64_float32, combined_text)
        """
        import soundfile as sf

        round_dir = self._find_latest_round_dir()
        # Also check base output dir for duplex-mode WAV files
        if not round_dir:
            direct_tts = os.path.join(self._output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                round_dir = self._output_dir
        if not round_dir:
            # Wait briefly — TTS may still be creating the directory
            for _ in range(30):
                time.sleep(0.2)
                round_dir = self._find_latest_round_dir()
                if round_dir:
                    break
                direct_tts = os.path.join(self._output_dir, "tts_wav")
                if os.path.isdir(direct_tts):
                    round_dir = self._output_dir
                    break
        if not round_dir:
            return None, sse_text

        self._wait_for_generation_done(round_dir)

        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(tts_wav_dir):
            return None, sse_text

        wav_files = sorted(
            [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
            key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
        )

        if not wav_files:
            return None, sse_text

        all_audio = []
        for wf in wav_files:
            wp = os.path.join(tts_wav_dir, wf)
            try:
                data, sr = sf.read(wp)
                if len(data) > 0:
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    all_audio.append(data)
            except Exception as e:
                logger.warning(f"Failed to read {wf}: {e}")

        if not all_audio:
            return None, sse_text

        combined = np.concatenate(all_audio)
        audio_b64 = base64.b64encode(combined.astype(np.float32).tobytes()).decode("utf-8")
        return audio_b64, sse_text

    def _collect_all_wav_chunks(self, sse_text: str = "") -> List[tuple]:
        """收集所有 WAV 文件，每个文件作为独立 chunk

        Returns:
            [(audio_base64_float32, text), ...]
        """
        import soundfile as sf

        round_dir = self._find_latest_round_dir()
        if not round_dir:
            direct_tts = os.path.join(self._output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                round_dir = self._output_dir
        if not round_dir:
            if sse_text:
                return [(None, sse_text)]
            return []

        self._wait_for_generation_done(round_dir)

        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(tts_wav_dir):
            if sse_text:
                return [(None, sse_text)]
            return []

        wav_files = sorted(
            [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
            key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
        )

        results = []
        for i, wf in enumerate(wav_files):
            wp = os.path.join(tts_wav_dir, wf)
            try:
                data, sr = sf.read(wp)
                if len(data) == 0:
                    continue
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                audio_b64 = base64.b64encode(data.tobytes()).decode("utf-8")
                text = sse_text if i == 0 else None
                results.append((audio_b64, text))
            except Exception as e:
                logger.warning(f"Failed to read {wf}: {e}")

        if not results and sse_text:
            results.append((None, sse_text))
        return results

    def _read_llm_text(self, llm_debug_dir: str) -> str:
        """从 llm_debug 目录读取所有文本并拼接"""
        text_file = os.path.join(llm_debug_dir, "llm_text.txt")
        if os.path.exists(text_file):
            try:
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                texts = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r"\[chunk_\d+\]\s*(.*)", line)
                    texts.append(m.group(1).strip() if m else line)
                return "".join(texts)
            except Exception:
                pass

        # fallback: read per-chunk text files
        texts = []
        for i in range(100):
            chunk_dir = os.path.join(llm_debug_dir, f"chunk_{i}")
            txt_path = os.path.join(chunk_dir, "llm_text.txt")
            if not os.path.exists(txt_path):
                break
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read().strip())
            except Exception:
                break
        return "".join(texts)

    def _read_llm_text_lines(self, llm_debug_dir: str) -> List[str]:
        """从 llm_debug 目录按 chunk 读取文本列表"""
        text_file = os.path.join(llm_debug_dir, "llm_text.txt")
        if os.path.exists(text_file):
            try:
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                results = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r"\[chunk_\d+\]\s*(.*)", line)
                    results.append(m.group(1).strip() if m else line)
                return results
            except Exception:
                pass

        results = []
        for i in range(100):
            chunk_dir = os.path.join(llm_debug_dir, f"chunk_{i}")
            txt_path = os.path.join(chunk_dir, "llm_text.txt")
            if not os.path.exists(txt_path):
                break
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    results.append(f.read().strip())
            except Exception:
                break
        return results

    # ================================================================
    # Internal: output directory management
    # ================================================================

    def _reset_output_dir(self) -> None:
        if os.path.exists(self._output_dir):
            for item in os.listdir(self._output_dir):
                item_path = os.path.join(self._output_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass
        os.makedirs(self._output_dir, exist_ok=True)

    # ================================================================
    # Internal: auto detect LLM model
    # ================================================================

    @staticmethod
    def _auto_detect_llm_model(model_dir: str) -> str:
        import glob

        # 优先 Q8，再回退到 Q4 / F16（与显式配置 llm_model 时的推荐一致）
        patterns = ["*Q8_0*.gguf", "*Q4_K_M*.gguf", "*Q4_K_S*.gguf", "*F16*.gguf"]
        for pat in patterns:
            matches = glob.glob(os.path.join(model_dir, pat))
            root = [m for m in matches if os.path.dirname(m) == model_dir]
            if root:
                return os.path.basename(sorted(root)[0])

        all_gguf = glob.glob(os.path.join(model_dir, "*.gguf"))
        candidates = [f for f in all_gguf
                      if not any(x in os.path.basename(f).lower()
                                 for x in ("audio", "vision", "tts", "projector"))]
        if candidates:
            return os.path.basename(sorted(candidates)[0])

        raise RuntimeError(f"No LLM GGUF found in {model_dir}")

    # ================================================================
    # Cleanup
    # ================================================================

    def shutdown(self) -> None:
        if self._cpp_process:
            logger.info("Stopping C++ server...")
            self._cpp_process.terminate()
            try:
                self._cpp_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._cpp_process.kill()
            self._cpp_process = None

        if self._http_client:
            self._http_client.close()
            self._http_client = None

        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
