"""C++ llama.cpp-omni WebSocket 协议后端适配层

通过 WebSocket 协议调用 C++ llama-server 的 /backend 端点，
实现与 CppBackendWorker 相同的方法签名，作为 PyTorch 后端的 drop-in 替换。

协议文档：docs/backend-protocol/README.md

生命周期映射：
    服务启动   → 启动 llama-server 进程（模型在 server.cpp 初始化时加载）
    新会话     → WS connect → session.init → session.created
    Duplex     → input.append (audio + video_frames) → text_delta / audio_delta / listen / response.done
    Half-duplex→ input.append (messages) → text_delta / audio_delta / response.done
    Chat       → input.append (messages, streaming=false) → response.done
    打断/关闭  → HTTP POST /sessions/{id}/close
    会话结束   → WS close
"""

import os
import re
import json
import time
import base64
import shutil
import signal
import logging
import tempfile
import threading
import subprocess
from typing import Optional, List, Dict, Any, Iterator, Tuple
from enum import Enum

import numpy as np
import websocket  # websocket-client (sync)

logger = logging.getLogger("cpp_ws_backend")

_AUDIO_INPUT_SR = 16000
_AUDIO_OUTPUT_SR = 24000

# ============================================================================
# System prompts (same as cpp_backend.py)
# ============================================================================

_SYSTEM_PROMPTS: Dict[tuple, Dict[str, str]] = {
    (True, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|><|im_end|>\n",
    },
    (True, "en"): {
        "voice_clone_prompt": "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|><|im_end|>\n",
    },
    (False, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。"
                              "请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
                              "<|im_end|>\n<|im_start|>user\n",
    },
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
    return _SYSTEM_PROMPTS.get((duplex, lang), _SYSTEM_PROMPTS[(duplex, "zh")])


def _sampling_from_generation(generation) -> Dict[str, Any]:
    sampling = {}
    if generation is None:
        return sampling
    for key in ("temperature", "top_p", "top_k", "repetition_penalty",
                "presence_penalty", "frequency_penalty", "min_p",
                "length_penalty", "max_new_tokens"):
        val = getattr(generation, key, None)
        if val is not None:
            sampling[key] = float(val) if isinstance(val, (int, float)) else val
    return sampling


# ============================================================================
# WS protocol helpers
# ============================================================================

def _audio_ndarray_to_b64(audio: np.ndarray) -> str:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return base64.b64encode(audio.tobytes()).decode("utf-8")


def _read_ref_audio_b64(ref_audio_path: Optional[str]) -> str:
    if not ref_audio_path or not os.path.exists(ref_audio_path):
        return ""
    try:
        import soundfile as sf
        data, sr = sf.read(ref_audio_path)
        if sr != _AUDIO_INPUT_SR:
            import librosa
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=_AUDIO_INPUT_SR)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return _audio_ndarray_to_b64(data.astype(np.float32))
    except Exception:
        return ""


def _image_to_b64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


# ============================================================================
# CppWsBackendWorker
# ============================================================================

class CppWsBackendWorker:
    """Backend that communicates with llama-server via the new WebSocket protocol."""

    def __init__(
        self,
        gpu_id: int = 0,
        llamacpp_root: str = "",
        model_dir: str = "",
        ref_audio_path: Optional[str] = None,
        duplex_pause_timeout: float = 60.0,
        **kwargs,
    ):
        self.llamacpp_root = llamacpp_root
        self.model_dir = model_dir
        self.gpu_id = gpu_id
        self.ref_audio_path = ref_audio_path
        self.duplex_pause_timeout = duplex_pause_timeout

        from worker import WorkerState, WorkerStatus
        self.state = WorkerState()
        self.processor = None

        self._cpp_server_port = 19060 + gpu_id
        self._cpp_server_url = f"http://127.0.0.1:{self._cpp_server_port}"
        self._cpp_ws_url = f"ws://127.0.0.1:{self._cpp_server_port}/backend"
        self._cpp_process: Optional[subprocess.Popen] = None
        self._http_client: Any = None
        self._temp_dir = tempfile.mkdtemp(prefix="cpp_ws_backend_")
        self._output_dir = os.path.join(llamacpp_root, f"tools/omni/output_{self._cpp_server_port}")

        self._ws: Optional[websocket.WebSocket] = None
        self._session_id: Optional[str] = None
        self._session_mode: str = ""
        self._response_seq: int = 0
        self._kv_cache_len: int = 0
        self._stop_event = threading.Event()

    # ================================================================
    # Server lifecycle
    # ================================================================

    def load_model(self) -> None:
        from worker import WorkerStatus
        self.state.status = WorkerStatus.LOADING
        logger.info(f"[GPU {self.gpu_id}] Starting C++ llama-server (WS protocol)...")
        self._start_cpp_server()
        self.state.status = WorkerStatus.IDLE
        logger.info(f"[GPU {self.gpu_id}] C++ WS backend ready")

    def _start_cpp_server(self) -> None:
        import httpx
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(600.0, connect=30.0),
            trust_env=False,
        )

        model_path = os.path.join(self.model_dir, "MiniCPM-Omni-gguf")
        mmproj_path = os.path.join(self.model_dir, "mmproj-minicpm-o-gguf", "mmproj-minicpm-o-F16.gguf")
        llm_gguf = self._auto_detect_llm_model(model_path)

        server_bin = os.path.join(self.llamacpp_root, "build-arm64-apple-clang-release/bin/llama-server")
        if not os.path.exists(server_bin):
            server_bin = os.path.join(self.llamacpp_root, "build/bin/llama-server")

        cmd = [
            server_bin,
            "-m", llm_gguf,
            "--mmproj", mmproj_path,
            "--port", str(self._cpp_server_port),
            "--host", "127.0.0.1",
            "--n-gpu-layers", "99",
            "--ctx-size", "32768",
            "-ngl", "99",
        ]

        env = os.environ.copy()
        logger.info(f"[GPU {self.gpu_id}] Launching: {' '.join(cmd)}")
        self._cpp_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        for i in range(120):
            try:
                resp = self._http_client.get(f"{self._cpp_server_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    logger.info(f"[GPU {self.gpu_id}] llama-server ready after {i}s")
                    return
            except Exception:
                pass
            time.sleep(1.0)

        raise RuntimeError(f"llama-server failed to start on port {self._cpp_server_port}")

    def _auto_detect_llm_model(self, model_dir: str) -> str:
        gguf_dir = os.path.join(model_dir, "MiniCPM-Omni-gguf")
        for name in ["Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "MiniCPM-Omni-Q4_K_M.gguf"]:
            path = os.path.join(gguf_dir, name)
            if os.path.exists(path):
                return path
        for f in os.listdir(gguf_dir):
            if f.endswith(".gguf"):
                return os.path.join(gguf_dir, f)
        raise FileNotFoundError(f"No GGUF model found in {gguf_dir}")

    def _stop_cpp_server(self) -> None:
        if self._cpp_process and self._cpp_process.poll() is None:
            logger.info(f"[GPU {self.gpu_id}] Stopping llama-server...")
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(self._cpp_process.pid), signal.SIGTERM)
            else:
                self._cpp_process.terminate()
            try:
                self._cpp_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._cpp_process.kill()
                self._cpp_process.wait()
        self._cpp_process = None

    def full_reinit(self) -> None:
        self._close_ws_safe()
        self._stop_cpp_server()
        self._start_cpp_server()

    @property
    def kv_cache_length(self) -> int:
        return self._kv_cache_len

    # ================================================================
    # WebSocket helpers
    # ================================================================

    def _ws_connect(self, mode: str) -> None:
        self._close_ws_safe()
        self._stop_event.clear()

        ws = websocket.create_connection(self._cpp_ws_url, timeout=30)
        self._ws = ws

        prompts = _get_system_prompts(mode == "full_duplex", "zh")
        ref_audio_b64 = _read_ref_audio_b64(self.ref_audio_path)

        init_msg: Dict[str, Any] = {
            "type": "session.init",
            "payload": {
                "mode": mode,
            }
        }
        if ref_audio_b64:
            init_msg["payload"]["voice"] = {"ref_audio": ref_audio_b64}

        ws.send(json.dumps(init_msg))

        raw = ws.recv()
        resp = json.loads(raw)
        if resp.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {resp.get('type')}: {resp}")

        self._session_id = resp.get("session_id", "")
        self._session_mode = mode
        self._response_seq = 0
        logger.info(f"WS session created: {self._session_id} mode={mode}")

        metrics = resp.get("metrics", {})
        if "kv_cache_length" in metrics:
            self._kv_cache_len = int(metrics["kv_cache_length"])

    def _ws_send_input_append(self, msg: dict) -> None:
        self._response_seq += 1
        if self._ws:
            self._ws.send(json.dumps(msg))

    def _ws_recv_events(self, timeout: Optional[float] = None) -> Iterator[dict]:
        if hasattr(self._ws, 'gettimeout'):
            old = self._ws.gettimeout()
            self._ws.settimeout(timeout or 0.2)
        try:
            while not self._stop_event.is_set():
                try:
                    raw = self._ws.recv()
                except websocket.WebSocketTimeoutException:
                    if timeout is not None:
                        break
                    continue
                except Exception:
                    break
                event = json.loads(raw)
                yield event
                etype = event.get("type", "")
                if etype in ("response.done", "session.closed"):
                    break
        finally:
            if hasattr(self._ws, 'settimeout'):
                self._ws.settimeout(old)

    def _close_ws_safe(self) -> None:
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._session_id = None

    def _http_close_session(self) -> None:
        if not self._session_id or not self._http_client:
            return
        try:
            self._http_client.post(
                f"{self._cpp_server_url}/sessions/{self._session_id}/close",
                timeout=10.0,
            )
        except Exception:
            pass
        self._close_ws_safe()

    # ================================================================
    # Temp file helpers
    # ================================================================

    def _save_image_to_temp(self, image: Any, prefix: str) -> str:
        path = os.path.join(self._temp_dir, f"{prefix}_image.jpg")
        if hasattr(image, "save"):
            image.save(path, "JPEG", quality=95)
        return path

    def _cleanup_temp_files(self, *paths: str) -> None:
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ================================================================
    # Duplex (full_duplex)
    # ================================================================

    def duplex_prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        media_type: int = 2,
        lang: Optional[str] = None,
        system_content: Any = None,
        length_penalty: float = 1.1,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._ws_connect(mode="full_duplex")
        return "ok"

    def duplex_prefill(
        self,
        audio_chunk: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        max_slice_nums: int = -1,
        text: str = "",
    ) -> Dict[str, Any]:
        cnt = getattr(self, '_duplex_chunk_counter', 0) + 1
        setattr(self, '_duplex_chunk_counter', cnt)

        audio_b64 = ""
        temp_files = []
        video_frames_b64 = []

        if audio_chunk is not None:
            audio_b64 = _audio_ndarray_to_b64(audio_chunk)
        if frame_list:
            for frame in frame_list:
                img_path = self._save_image_to_temp(frame, f"duplex_{cnt}")
                temp_files.append(img_path)
                video_frames_b64.append(_image_to_b64(img_path))

        msg: Dict[str, Any] = {"type": "input.append", "input": {}}
        if audio_b64:
            msg["input"]["audio"] = audio_b64
        if video_frames_b64:
            msg["input"]["video_frames"] = video_frames_b64
        if max_slice_nums >= 1:
            msg["input"]["max_slice_nums"] = max_slice_nums

        self._ws_send_input_append(msg)
        self._cleanup_temp_files(*temp_files)
        return {"n_vision_images": len(video_frames_b64)}

    def duplex_generate(self, force_listen: bool = False) -> Any:
        from core.schemas.duplex import DuplexGenerateResult

        is_listen = True
        end_of_turn = False
        text_parts: List[str] = []
        audio_bytes_list: List[bytes] = []

        for event in self._ws_recv_events(timeout=None):
            etype = event.get("type", "")
            if etype == "response.output.delta":
                kind = event.get("kind", "")
                if kind == "text":
                    delta = event.get("delta", "")
                    if delta:
                        text_parts.append(delta)
                elif kind == "audio":
                    delta_b64 = event.get("delta", "")
                    if delta_b64:
                        audio_bytes_list.append(base64.b64decode(delta_b64))
                elif kind == "listen":
                    is_listen = True
                    break
            elif etype == "response.done":
                full_text = event.get("full_text") or ""
                audio_b64 = event.get("audio") or ""
                if full_text:
                    text_parts.append(full_text)
                if audio_b64:
                    audio_bytes_list.append(base64.b64decode(audio_b64))
                reason = event.get("reason", "")
                is_listen = (reason == "listen")
                end_of_turn = (reason == "turn_end")
                break
            elif etype == "session.closed":
                end_of_turn = True
                break

        full_text = "".join(text_parts)
        full_audio = b"".join(audio_bytes_list) if audio_bytes_list else None

        return DuplexGenerateResult(
            is_listen=is_listen,
            end_of_turn=end_of_turn,
            text=full_text,
            audio_data=full_audio,
        )

    def duplex_stop(self) -> None:
        self._http_close_session()

    def duplex_cleanup(self) -> None:
        self._close_ws_safe()

    # ================================================================
    # Half-duplex (turn_based streaming)
    # ================================================================

    def reset_half_duplex_session(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        lang: str = "zh",
        streaming: bool = True,
        length_penalty: float = 1.1,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self._ws_connect(mode="turn_based")
        return None

    def half_duplex_prefill(
        self,
        audio_chunk: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        text: str = "",
        max_slice_nums: int = -1,
    ) -> Dict[str, Any]:
        return {"n_vision_images": 0}

    def half_duplex_generate(
        self,
        messages: List[Dict],
        streaming: bool = True,
        length_penalty: float = 1.1,
        max_new_tokens: int = 512,
        tts_enabled: bool = True,
        cur_round: int = 0,
    ) -> Iterator:
        from core.schemas.streaming import StreamingChunk

        msg = {
            "type": "input.append",
            "input": {
                "messages": messages,
                "streaming": streaming,
                "generation": {
                    "max_new_tokens": max_new_tokens,
                    "length_penalty": float(length_penalty),
                },
                "tts": {"enabled": tts_enabled},
            },
        }
        self._ws_send_input_append(msg)

        texts: List[str] = []
        chunk_idx = 0

        for event in self._ws_recv_events(timeout=None):
            etype = event.get("type", "")
            if etype == "response.output.delta":
                kind = event.get("kind", "")
                if kind == "text":
                    delta = event.get("delta", "")
                    if delta:
                        texts.append(delta)
                        yield StreamingChunk(chunk_index=chunk_idx, text_delta=delta, is_final=False)
                        chunk_idx += 1
                elif kind == "audio":
                    delta_b64 = event.get("delta", "")
                    if delta_b64:
                        audio_data = base64.b64decode(delta_b64)
                        yield StreamingChunk(chunk_index=chunk_idx, audio_data=audio_data, is_final=False)
                        chunk_idx += 1
                elif kind == "listen":
                    yield StreamingChunk(chunk_index=chunk_idx, text_delta="", is_final=True)
                    return
            elif etype == "response.done":
                full_text = event.get("full_text") or "".join(texts)
                audio_b64 = event.get("audio") or ""
                audio_data = base64.b64decode(audio_b64) if audio_b64 else None
                yield StreamingChunk(chunk_index=chunk_idx, text_delta=full_text, audio_data=audio_data, is_final=True)
                return
            elif etype == "session.closed":
                yield StreamingChunk(chunk_index=chunk_idx, text_delta="", is_final=True)
                return

        yield StreamingChunk(chunk_index=0, is_final=True)

    # ================================================================
    # Chat (turn_based non-streaming)
    # ================================================================

    def chat(self, request) -> Any:
        from core.schemas.chat import ChatResponse
        from core.processors.base import MiniCPMOProcessorMixin

        generation = getattr(request, "generation", None)
        length_penalty = float(getattr(generation, "length_penalty", 1.1) or 1.1)
        max_new_tokens = int(getattr(generation, "max_new_tokens", 512) or 512)

        self._ws_connect(mode="turn_based")

        mixin = MiniCPMOProcessorMixin()
        messages = []
        for msg in request.messages:
            content = mixin._convert_content_to_model_format(msg.content)
            message_dict = {"role": msg.role, "content": []}
            for item in content:
                if isinstance(item, np.ndarray):
                    message_dict["content"].append({
                        "type": "audio",
                        "audio": _audio_ndarray_to_b64(item),
                    })
                elif isinstance(item, str):
                    message_dict["content"].append({"type": "text", "text": item})
                elif hasattr(item, "save"):
                    img_path = self._save_image_to_temp(item, f"chat_{len(messages)}")
                    img_b64 = _image_to_b64(img_path)
                    message_dict["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                    self._cleanup_temp_files(img_path)
            messages.append(message_dict)

        ws_msg = {
            "type": "input.append",
            "input": {
                "messages": messages,
                "streaming": False,
                "generation": {
                    "max_new_tokens": max_new_tokens,
                    "length_penalty": float(length_penalty),
                },
            },
        }
        self._ws_send_input_append(ws_msg)

        full_text = ""
        full_audio = ""
        for event in self._ws_recv_events(timeout=None):
            etype = event.get("type", "")
            if etype == "response.done":
                full_text = event.get("full_text") or ""
                full_audio = event.get("audio") or ""
                break
            elif etype == "session.closed":
                break

        self._close_ws_safe()
        return ChatResponse(text=full_text, audio_data=base64.b64decode(full_audio) if full_audio else None)
