"""MiniCPMO45 推理 Worker

每个 Worker 占用一张 GPU，持有一个 UnifiedProcessor 实例，
提供 Chat (HTTP) / Streaming (WebSocket) / Duplex (WebSocket) 三种推理 API。

启动方式：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python worker.py \\
        --port 10031 \\
        --model-path /path/to/base_model \\
        --pt-path /path/to/custom.pt \\
        --ref-audio-path /path/to/ref.wav
"""

import gc
import re
import json
import time
import uuid
import asyncio
import argparse
import logging
import base64
import threading
from enum import Enum
from typing import Optional, List, Dict, Any, Iterator
from datetime import datetime

import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from pydantic import BaseModel, Field

from core.schemas.common import Message, Role, TextContent, AudioContent, ImageContent, VideoContent, ContentItem
from core.schemas.chat import ChatRequest, ChatResponse
from core.schemas.streaming import (
    StreamingRequest, StreamingChunk, StreamingResponse, StreamingConfig,
)
from core.schemas.duplex import (
    DuplexConfig, DuplexGenerateResult, DuplexPrefillRequest,
)
from session_recorder import DuplexSessionRecorder, TurnBasedSessionRecorder, generate_session_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")


# ============ Worker 状态 ============

class WorkerStatus(str, Enum):
    """Worker 状态"""
    LOADING = "loading"        # 正在加载模型
    IDLE = "idle"              # 空闲（可接受新请求）
    BUSY_CHAT = "busy_chat"    # 正在处理 Chat 请求
    BUSY_HALF_DUPLEX = "busy_half_duplex"  # 正在处理 Half-Duplex 请求
    DUPLEX_ACTIVE = "duplex_active"    # Duplex 活跃中
    DUPLEX_PAUSED = "duplex_paused"    # Duplex 暂停中
    ERROR = "error"            # 异常状态


class WorkerState(BaseModel):
    """Worker 运行时状态"""
    status: WorkerStatus = WorkerStatus.LOADING
    current_session_id: Optional[str] = None
    duplex_pause_time: Optional[float] = None  # Duplex 暂停的时间戳
    total_requests: int = 0
    total_inference_time_ms: float = 0.0
    last_activity: Optional[str] = None

    @property
    def is_idle(self) -> bool:
        return self.status == WorkerStatus.IDLE

    @property
    def is_busy(self) -> bool:
        return self.status in (
            WorkerStatus.BUSY_CHAT,
            WorkerStatus.BUSY_HALF_DUPLEX,
            WorkerStatus.DUPLEX_ACTIVE,
            WorkerStatus.DUPLEX_PAUSED,
        )


# ============ 请求/响应模型 ============

class WorkerHealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    worker_status: WorkerStatus
    gpu_id: int
    model_loaded: bool
    current_session_id: Optional[str] = None
    total_requests: int = 0
    avg_inference_time_ms: float = 0.0
    kv_cache_length: int = 0  # 当前 LLM KV cache token 总数


class StreamingWsMessage(BaseModel):
    """Streaming WebSocket 消息（Client → Server）"""
    type: str  # "prefill" | "generate" | "complete_turn" | "close"
    # prefill 参数
    messages: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    is_last_chunk: bool = True
    # generate 参数
    generate_audio: bool = True
    max_new_tokens: int = 256
    # complete_turn 参数
    output_audio_path: Optional[str] = None


class DuplexWsMessage(BaseModel):
    """Duplex WebSocket 消息（Client → Server）"""
    type: str  # "prepare" | "audio_chunk" | "pause" | "resume" | "stop"
    # prepare 参数
    system_prompt: Optional[str] = None
    ref_audio_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    # audio_chunk 参数
    audio_base64: Optional[str] = None
    frame_base64_list: Optional[List[str]] = None
    force_listen: Optional[bool] = None  # Force Listen 开关（per-chunk）


# ============ Worker 主类 ============

class MiniCPMOWorker:
    """MiniCPMO45 推理 Worker

    持有一个 UnifiedProcessor 实例，提供三种推理模式。
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        pt_path: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        duplex_pause_timeout: float = 60.0,
        compile: bool = False,
        chat_vocoder: str = "token2wav",
        attn_implementation: str = "auto",
    ):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.pt_path = pt_path
        self.ref_audio_path = ref_audio_path
        self.duplex_pause_timeout = duplex_pause_timeout
        self.compile = compile
        self.chat_vocoder = chat_vocoder
        self.attn_implementation = attn_implementation

        self.state = WorkerState()
        self.processor = None

        # Duplex 暂停超时监控 task
        self._duplex_timeout_task: Optional[asyncio.Task] = None

    def load_model(self) -> None:
        """加载模型（同步，在启动时调用）"""
        self.state.status = WorkerStatus.LOADING
        logger.info(f"[GPU {self.gpu_id}] Loading model from {self.model_path}...")

        from core.processors.unified import UnifiedProcessor

        self.processor = UnifiedProcessor(
            model_path=self.model_path,
            pt_path=self.pt_path,
            ref_audio_path=self.ref_audio_path,
            compile=self.compile,
            chat_vocoder=self.chat_vocoder,
            attn_implementation=self.attn_implementation,
        )

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

        self.state.status = WorkerStatus.IDLE
        logger.info(f"[GPU {self.gpu_id}] Model loaded successfully")

        # 检查模型各组件的 device 分布
        self._log_device_map()

    def _log_device_map(self) -> None:
        """打印模型各关键组件的 device，用于确认是否全部在 GPU 上"""
        if self.processor is None:
            return
        model = self.processor.model
        checks: list[tuple[str, str]] = []

        # LLM
        try:
            p = next(model.llm.parameters())
            checks.append(("LLM", str(p.device)))
        except Exception:
            checks.append(("LLM", "N/A"))

        # Vision encoder
        try:
            p = next(model.vpm.parameters())
            checks.append(("Vision (vpm)", str(p.device)))
        except Exception:
            checks.append(("Vision (vpm)", "N/A"))

        # Whisper / audio encoder
        for name in ("apm", "audio_encoder", "whisper"):
            if hasattr(model, name):
                try:
                    p = next(getattr(model, name).parameters())
                    checks.append((f"Audio ({name})", str(p.device)))
                except Exception:
                    checks.append((f"Audio ({name})", "no params"))
                break

        # TTS 模块
        if hasattr(model, "tts"):
            tts = model.tts
            # TTS 主体
            try:
                p = next(tts.parameters())
                checks.append(("TTS (main)", str(p.device)))
            except Exception:
                checks.append(("TTS (main)", "N/A"))

            # audio_tokenizer (Token2Wav 关键组件)
            if hasattr(tts, "audio_tokenizer"):
                tok = tts.audio_tokenizer
                try:
                    p = next(tok.parameters())
                    checks.append(("TTS audio_tokenizer", str(p.device)))
                except Exception:
                    checks.append(("TTS audio_tokenizer", "no params"))

                # hift (vocoder in Token2Wav)
                if hasattr(tok, "hift"):
                    try:
                        p = next(tok.hift.parameters())
                        checks.append(("TTS hift (vocoder)", str(p.device)))
                    except Exception:
                        checks.append(("TTS hift (vocoder)", "no params"))

            # CosyVoice2 / flow model
            for attr_name in ("cosyvoice", "cosyvoice2", "flow"):
                if hasattr(tts, attr_name):
                    try:
                        p = next(getattr(tts, attr_name).parameters())
                        checks.append((f"TTS {attr_name}", str(p.device)))
                    except Exception:
                        checks.append((f"TTS {attr_name}", "no params"))

        # Duplex decoder
        if hasattr(model, "duplex") and model.duplex is not None:
            try:
                p = next(model.duplex.decoder.parameters())
                checks.append(("Duplex decoder", str(p.device)))
            except Exception:
                checks.append(("Duplex decoder", "N/A"))

        logger.info(f"[GPU {self.gpu_id}] === Device Map ===")
        for name, device in checks:
            on_gpu = "cuda" in device
            marker = "✓" if on_gpu else "⚠ CPU!"
            logger.info(f"[GPU {self.gpu_id}]   {marker} {name}: {device}")

    # ========== Chat ==========

    def chat(self, request: ChatRequest) -> ChatResponse:
        """执行 Chat 推理（无状态）

        Chat 模式下 cached_tokens 始终为 0（每次从头 prefill）。
        token_stats 中的 input_tokens/generated_tokens 从模型输出精确获取：
        - input_tokens: tokenizer 级别（含 audio/image 占位符，不含 embedding 展开）
        - generated_tokens: LLM 实际生成的 token 数
        """
        if not self.state.is_idle:
            raise RuntimeError(f"Worker not idle, status: {self.state.status}")

        self.state.status = WorkerStatus.BUSY_CHAT
        self.state.last_activity = datetime.now().isoformat()
        start_time = time.perf_counter()

        try:
            chat_view = self.processor.set_chat_mode()
            response = chat_view.chat(
                request,
                max_new_tokens=request.generation.max_new_tokens,
                do_sample=request.generation.do_sample,
                generate_audio=request.tts.enabled if request.tts else False,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.state.total_requests += 1
            self.state.total_inference_time_ms += elapsed_ms

            # Chat token 统计已在 ChatView._chat_impl() 中从模型输出精确获取
            # input_tokens: tokenizer 级别（含 audio/image 占位符）
            # generated_tokens: LLM 实际生成的 token 数

            ts = response.token_stats or {}
            logger.info(
                f"[GPU {self.gpu_id}] Chat completed: "
                f"{len(response.text)} chars, {elapsed_ms:.0f}ms, "
                f"tokens: in={ts.get('input_tokens', '?')} "
                f"gen={ts.get('generated_tokens', '?')} "
                f"total={ts.get('total_tokens', '?')}"
            )
            return response
        finally:
            # Chat 是无状态的，完成后清除 KV Cache 映射
            self.state.status = WorkerStatus.IDLE
            self.state.current_session_id = None

    # ========== Chat prefill + generate（KV cache 模式） ==========

    def chat_prefill(
        self,
        session_id: str,
        msgs,
        omni_mode: bool = False,
        max_slice_nums=None,
        use_tts_template: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        """Chat prefill：一次性 prefill 所有消息到 KV cache"""
        chat_view = self.processor.set_chat_mode()
        prompt = chat_view.prefill(
            session_id=session_id,
            msgs=msgs,
            omni_mode=omni_mode,
            max_slice_nums=max_slice_nums,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )
        return prompt

    def chat_non_streaming_generate(
        self,
        session_id: str,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        generate_audio: bool = False,
        use_tts_template: bool = True,
        enable_thinking: bool = False,
        tts_ref_audio=None,
        tts_sampling_params=None,
        length_penalty: float = 1.1,
    ):
        """Chat 非流式 generate：基于 KV cache 做 HF generate + 可选 TTS"""
        chat_view = self.processor.set_chat_mode()
        result = chat_view.generate(
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            generate_audio=generate_audio,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
            tts_ref_audio=tts_ref_audio,
            tts_sampling_params=tts_sampling_params,
            length_penalty=length_penalty,
        )
        return result

    def chat_streaming_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        length_penalty: float = 1.1,
    ) -> "Iterator[StreamingChunk]":
        """Chat 流式 generate：基于 KV cache 做 streaming_generate"""
        chat_view = self.processor.set_chat_mode()
        yield from chat_view.streaming_generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    # ========== Half-Duplex ==========

    def half_duplex_prefill(self, request: StreamingRequest) -> str:
        """Half-Duplex 预填充"""
        half_duplex_view = self.processor.set_half_duplex_mode()
        prompt = half_duplex_view.prefill(request)
        return prompt

    def half_duplex_init_tts(self, ref_audio_data: Optional[np.ndarray] = None) -> None:
        """初始化 Half-Duplex TTS（在 generate 前调用，如需生成音频）
        
        Args:
            ref_audio_data: 前端上传的 ref audio ndarray (16kHz mono float32)。
                若提供则使用此数据，否则使用 worker 默认的 ref_audio_path。
        """
        half_duplex_view = self.processor.set_half_duplex_mode()
        if ref_audio_data is not None:
            half_duplex_view.init_ref_audio_from_data(ref_audio_data)
        else:
            half_duplex_view.init_ref_audio(self.ref_audio_path)

    def half_duplex_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        length_penalty: float = 1.1,
    ) -> Iterator[StreamingChunk]:
        """Half-Duplex 生成（yield StreamingChunk）"""
        half_duplex_view = self.processor.set_half_duplex_mode()
        yield from half_duplex_view.generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    def half_duplex_complete_turn(
        self,
        session_id: str,
        messages: List[Message],
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        output_audio_path: Optional[str] = None,
        length_penalty: float = 1.1,
    ) -> StreamingResponse:
        """Half-Duplex 完成一轮（便捷方法）"""
        half_duplex_view = self.processor.set_half_duplex_mode()
        return half_duplex_view.complete_turn(
            session_id=session_id,
            messages=messages,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            output_audio_path=output_audio_path,
            length_penalty=length_penalty,
        )

    def reset_half_duplex_session(self) -> None:
        """重置 Half-Duplex 模型 session（清除 KV cache）"""
        half_duplex_view = self.processor.set_half_duplex_mode()
        half_duplex_view._model.reset_session(reset_token2wav_cache=False)
        logger.info(f"[GPU {self.gpu_id}] Half-Duplex model session reset (KV cache cleared)")

    # ========== Duplex ==========

    def duplex_prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
    ) -> str:
        """Duplex 准备

        Args:
            system_prompt_text: 系统提示文本
            ref_audio_path: LLM 参考音频路径（嵌入 system prompt）
            prompt_wav_path: TTS 参考音频路径（初始化 vocoder）。
                若不提供则 fallback 到 ref_audio_path。
        """
        duplex_view = self.processor.set_duplex_mode()
        return duplex_view.prepare(
            system_prompt_text=system_prompt_text,
            ref_audio_path=ref_audio_path or self.ref_audio_path,
            prompt_wav_path=prompt_wav_path,
        )

    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[list] = None,
        max_slice_nums: int = 1,
    ) -> Dict[str, Any]:
        """Duplex 预填充"""
        duplex_view = self.processor.set_duplex_mode()
        return duplex_view.prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )

    def duplex_generate(self, force_listen: bool = False) -> DuplexGenerateResult:
        """Duplex 生成
        
        Args:
            force_listen: 前端 Force Listen 开关，强制本次生成为 listen
        """
        duplex_view = self.processor.set_duplex_mode()
        return duplex_view.generate(force_listen=force_listen)

    def duplex_finalize(self) -> None:
        """Duplex 延迟 finalize（feed 终止符 + 滑窗维护）
        
        必须在 duplex_generate 之后、下一次 duplex_prefill 之前调用。
        """
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.finalize()

    def duplex_stop(self) -> None:
        """Duplex 停止"""
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.stop()

    def duplex_cleanup(self) -> None:
        """Duplex 会话结束后释放 GPU 资源，恢复到初始状态

        调用 DuplexView.cleanup() 释放 KV cache、TTS caches 等，
        然后触发 gc + empty_cache 确保显存真正归还。

        诊断数据（40B 参数模型）：
        - stop 后泄漏: ~1,591 MB
        - cleanup 后残留: ~48 MB（忽略不计）
        - 释放量: ~1,543 MB
        """
        if self.processor is None:
            return

        duplex_view = self.processor.set_duplex_mode()
        duplex_view.cleanup()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info(f"[GPU {self.gpu_id}] Duplex cleanup done, GPU memory released")


# ============ FastAPI 应用 ============

worker: Optional[MiniCPMOWorker] = None

# 启动参数（通过 main() 传入）
WORKER_CONFIG: Dict[str, Any] = {}


def _worker_ready() -> bool:
    """检查 worker 是否已完成初始化并可以接受请求（兼容 PyTorch 和 C++ 后端）"""
    if worker is None:
        return False
    if hasattr(worker, 'processor') and worker.processor is not None:
        return True
    if hasattr(worker, '_http_client') and worker._http_client is not None:
        return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型"""
    global worker
    config = WORKER_CONFIG

    backend = config.get("backend", "pytorch")

    if backend == "cpp":
        from core.processors.cpp_backend import CppBackendWorker
        worker = CppBackendWorker(
            llamacpp_root=config["llamacpp_root"],
            model_dir=config["model_dir"],
            gpu_id=config["gpu_id"],
            ref_audio_path=config.get("ref_audio_path"),
            duplex_pause_timeout=config.get("duplex_pause_timeout", 60.0),
            llm_model=config.get("llm_model", ""),
            cpp_server_port=config.get("cpp_server_port"),
            ctx_size=config.get("ctx_size", 8192),
            n_gpu_layers=config.get("n_gpu_layers", 99),
        )
    else:
        worker = MiniCPMOWorker(
            model_path=config["model_path"],
            gpu_id=config["gpu_id"],
            pt_path=config.get("pt_path"),
            ref_audio_path=config.get("ref_audio_path"),
            duplex_pause_timeout=config.get("duplex_pause_timeout", 60.0),
            compile=config.get("compile", False),
            chat_vocoder=config.get("chat_vocoder", "token2wav"),
            attn_implementation=config.get("attn_implementation", "auto"),
        )

    # 模型加载是同步操作（~15s for pytorch, ~2-3min for cpp），在线程中执行避免阻塞
    await asyncio.to_thread(worker.load_model)

    yield

    if backend == "cpp" and hasattr(worker, "shutdown"):
        worker.shutdown()
    logger.info("Worker shutting down")


app = FastAPI(title="MiniCPMO45 Worker", lifespan=lifespan)


# ========== 健康检查 ==========

@app.get("/health", response_model=WorkerHealthResponse)
async def health():
    """健康检查"""
    if worker is None:
        return WorkerHealthResponse(
            status="initializing",
            worker_status=WorkerStatus.LOADING,
            gpu_id=0,
            model_loaded=False,
        )

    avg_time = 0.0
    if worker.state.total_requests > 0:
        avg_time = worker.state.total_inference_time_ms / worker.state.total_requests

    kv_len = 0
    model_loaded = False
    if hasattr(worker, 'processor') and worker.processor is not None:
        kv_len = worker.processor.kv_cache_length
        model_loaded = True
    elif hasattr(worker, '_http_client') and worker._http_client is not None:
        model_loaded = True
        kv_len = getattr(worker, 'kv_cache_length', 0)

    return WorkerHealthResponse(
        status="healthy" if model_loaded else "error",
        worker_status=worker.state.status,
        gpu_id=worker.gpu_id,
        model_loaded=model_loaded,
        current_session_id=worker.state.current_session_id,
        total_requests=worker.state.total_requests,
        avg_inference_time_ms=avg_time,
        kv_cache_length=kv_len,
    )


# ========== Chat API ==========

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat 推理（无状态）"""
    if not _worker_ready():
        raise HTTPException(status_code=503, detail="Worker not ready")

    if not worker.state.is_idle:
        # Gateway 排队机制已保证并发安全，但 Worker 可能还在 cleanup 上一个任务
        # （如 Duplex WS close 后 GPU 资源释放），短暂等待而非立即拒绝
        for _ in range(10):
            await asyncio.sleep(0.5)
            if worker.state.is_idle:
                break
        else:
            raise HTTPException(
                status_code=429,
                detail=f"Worker busy after waiting 5s, status: {worker.state.status.value}",
            )

    # 录制：创建 TurnBasedSessionRecorder
    chat_recorder: Optional[TurnBasedSessionRecorder] = None
    chat_session_id: Optional[str] = None
    from config import get_config
    chat_cfg = get_config()
    if chat_cfg.recording.enabled:
        chat_session_id = generate_session_id("chat")
        sys_prompt = ""
        for m in request.messages:
            if m.role == "system":
                c = m.content
                sys_prompt = c if isinstance(c, str) else str(c)
                break
        chat_recorder = TurnBasedSessionRecorder(
            session_id=chat_session_id,
            app_type="chat",
            worker_id=worker.gpu_id,
            config_snapshot={
                "system_prompt": sys_prompt,
                "ref_audio": chat_cfg.ref_audio_path,
            },
            data_dir=chat_cfg.data_dir,
        )

    try:
        response = await asyncio.to_thread(worker.chat, request)

        # 录制：记录 chat turn
        if chat_recorder and response.success:
            input_summary: Dict[str, Any] = {}
            for m in request.messages:
                if m.role == "user":
                    c = m.content
                    if isinstance(c, str):
                        input_summary["text"] = c
                    elif isinstance(c, list):
                        texts = [it.text for it in c if hasattr(it, "text") and it.text]
                        if texts:
                            input_summary["text"] = " ".join(texts)
            output_audio: Optional[np.ndarray] = None
            if response.audio_data:
                try:
                    audio_bytes = base64.b64decode(response.audio_data)
                    output_audio = np.frombuffer(audio_bytes, dtype=np.float32)
                except Exception:
                    pass
            chat_recorder.record_chat_turn(
                turn_index=0,
                request_ts_ms=0.0,
                input_summary=input_summary,
                output_text=response.text,
                output_audio=output_audio,
                timing={
                    "elapsed_ms": round(response.duration_ms, 1) if response.duration_ms else 0,
                    "tokens": response.tokens_generated or 0,
                },
            )

        if chat_recorder:
            chat_recorder.finalize()

        if chat_session_id and response.success:
            response.recording_session_id = chat_session_id

        return response
    except Exception as e:
        if chat_recorder:
            try:
                chat_recorder.finalize()
            except Exception:
                pass
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Chat WebSocket（统一流式/非流式） ==========

def _parse_raw_messages(raw_messages: List[dict]) -> List[Message]:
    """将前端原始消息列表解析为 Schema Message 列表"""
    messages: List[Message] = []
    for m in raw_messages:
        role = Role(m["role"])
        content = m["content"]
        if isinstance(content, list):
            content_items: List[ContentItem] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        content_items.append(TextContent(text=item["text"]))
                    elif item.get("type") == "audio" and item.get("data"):
                        content_items.append(AudioContent(data=item["data"]))
                    elif item.get("type") == "image" and item.get("data"):
                        content_items.append(ImageContent(data=item["data"]))
                    elif item.get("type") == "video" and item.get("data"):
                        content_items.append(VideoContent(
                            data=item["data"],
                            stack_frames=item.get("stack_frames", 1),
                        ))
            if content_items:
                messages.append(Message(role=role, content=content_items))
        else:
            messages.append(Message(role=role, content=content))
    return messages


def _convert_to_model_msgs(schema_messages: List[Message]) -> list:
    """将 Schema Message 列表转为模型格式 msgs"""
    from core.processors.base import MiniCPMOProcessorMixin
    _mixin = MiniCPMOProcessorMixin()
    model_msgs = []
    for m in schema_messages:
        content = _mixin._convert_content_to_model_format(m.content)
        if len(content) == 1 and isinstance(content[0], str):
            content = content[0]
        model_msgs.append({"role": m.role.value, "content": content})
    return model_msgs


@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    """Chat WebSocket — 统一流式/非流式

    协议：
    1. Client → JSON:
       {
         "messages": [...],
         "streaming": true/false,
         "generation": {"max_new_tokens": 256, "length_penalty": 1.1},
         "image": {"max_slice_nums": null},
         "tts": {"enabled": false, "ref_audio_data": "..."},
         "use_tts_template": false,
         "omni_mode": false,
       }
    2. Server → {"type": "prefill_done", "input_tokens": N}
    3a. streaming=true:
        Server → {"type": "chunk", "text_delta": "...", "audio_data": "..."} × N
        Server → {"type": "done", "text": "...", "generated_tokens": N, "input_tokens": N}
    3b. streaming=false:
        Server → {"type": "done", "text": "...", "audio_data": "...", "generated_tokens": N, "input_tokens": N}
    4. Error:
        Server → {"type": "error", "error": "..."}
    """
    if not _worker_ready():
        await ws.close(code=1013, reason="Worker not ready")
        return

    await ws.accept()
    logger.info("Chat WebSocket connected")

    try:
        raw = await ws.receive_text()
        msg = json.loads(raw)

        # 等待 Worker 空闲
        if not worker.state.is_idle:
            for _ in range(10):
                await asyncio.sleep(0.5)
                if worker.state.is_idle:
                    break
            else:
                await ws.send_json({"type": "error", "error": "Worker busy"})
                await ws.close()
                return

        session_id = "chat_ws_" + uuid.uuid4().hex[:8]
        worker.state.status = WorkerStatus.BUSY_HALF_DUPLEX
        worker.state.current_session_id = session_id

        chat_ws_recorder: Optional[TurnBasedSessionRecorder] = None
        chat_ws_session_id: Optional[str] = None

        try:
            # 1. 解析消息和参数
            messages = _parse_raw_messages(msg.get("messages", []))
            model_msgs = _convert_to_model_msgs(messages)
            streaming = msg.get("streaming", True)

            gen_cfg = msg.get("generation", {})
            max_new_tokens = gen_cfg.get("max_new_tokens", 256)
            length_penalty = gen_cfg.get("length_penalty", 1.1)

            max_slice_nums = None
            img_cfg = msg.get("image", {})
            if img_cfg and img_cfg.get("max_slice_nums") is not None:
                max_slice_nums = int(img_cfg["max_slice_nums"])

            generate_audio = False
            tts_ref_audio_ndarray = None
            use_tts_template = msg.get("use_tts_template", False)
            tts_cfg = msg.get("tts", {})
            if tts_cfg and tts_cfg.get("enabled"):
                generate_audio = True
                use_tts_template = True
                ref_b64 = tts_cfg.get("ref_audio_data")
                if ref_b64:
                    tts_ref_bytes = base64.b64decode(ref_b64)
                    tts_ref_audio_ndarray = np.frombuffer(tts_ref_bytes, dtype=np.float32)

            omni_mode = msg.get("omni_mode", False)
            enable_thinking = msg.get("enable_thinking", False)

            from config import get_config
            _chat_ws_cfg = get_config()
            if _chat_ws_cfg.recording.enabled:
                chat_ws_session_id = generate_session_id("chat")
                raw_messages = msg.get("messages", [])
                sys_prompt = ""
                for m in raw_messages:
                    if m.get("role") == "system":
                        c = m.get("content", "")
                        sys_prompt = c if isinstance(c, str) else str(c)
                        break
                chat_ws_recorder = TurnBasedSessionRecorder(
                    session_id=chat_ws_session_id,
                    app_type="chat",
                    worker_id=worker.gpu_id,
                    config_snapshot={
                        "system_prompt": sys_prompt,
                        "streaming": streaming,
                        "ref_audio": _chat_ws_cfg.ref_audio_path,
                    },
                    data_dir=_chat_ws_cfg.data_dir,
                )

            # 2. Prefill
            def _do_prefill():
                return worker.chat_prefill(
                    session_id=session_id,
                    msgs=model_msgs,
                    omni_mode=omni_mode,
                    max_slice_nums=max_slice_nums,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )

            await asyncio.to_thread(_do_prefill)
            pre_kv = worker.processor.kv_cache_length if worker.processor else 0
            await ws.send_json({"type": "prefill_done", "input_tokens": pre_kv})

            # 3. TTS init (PyTorch only; C++ handles TTS internally via omni_init)
            if generate_audio and worker.processor is not None:
                def _init_tts():
                    if tts_ref_audio_ndarray is not None:
                        worker.processor.model.init_token2wav_cache(prompt_speech_16k=tts_ref_audio_ndarray)
                    elif worker.ref_audio_path:
                        import librosa
                        ref_audio, _ = librosa.load(worker.ref_audio_path, sr=16000, mono=True)
                        worker.processor.model.init_token2wav_cache(prompt_speech_16k=ref_audio)
                await asyncio.to_thread(_init_tts)

            # Build input summary for recording — save all content types
            _chat_input_summary: Dict[str, Any] = {}
            _chat_audio_idx = 0
            _chat_video_idx = 0
            for _rm in msg.get("messages", []):
                if _rm.get("role") == "user":
                    c = _rm.get("content", "")
                    if isinstance(c, str):
                        _chat_input_summary["text"] = c
                    elif isinstance(c, list):
                        texts = [it["text"] for it in c if isinstance(it, dict) and it.get("type") == "text" and it.get("text")]
                        if texts:
                            _chat_input_summary["text"] = " ".join(texts)
                        if chat_ws_recorder:
                            saved_imgs = []
                            saved_videos = []
                            for it in c:
                                if not isinstance(it, dict):
                                    continue
                                if it.get("type") == "image" and it.get("data"):
                                    try:
                                        img_data = base64.b64decode(it["data"])
                                        idx = chat_ws_recorder.next_image_index()
                                        rel = chat_ws_recorder.save_user_image(idx, img_data)
                                        saved_imgs.append(rel)
                                    except Exception:
                                        pass
                                elif it.get("type") == "audio" and it.get("data"):
                                    try:
                                        audio_bytes = base64.b64decode(it["data"])
                                        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                                        rel = chat_ws_recorder.save_user_audio(_chat_audio_idx, audio_np)
                                        _chat_input_summary["audio"] = rel
                                        _chat_audio_idx += 1
                                    except Exception:
                                        pass
                                elif it.get("type") == "video" and it.get("data"):
                                    try:
                                        video_bytes = base64.b64decode(it["data"])
                                        rel = chat_ws_recorder.save_user_video(_chat_video_idx, video_bytes)
                                        saved_videos.append(rel)
                                        _chat_video_idx += 1
                                    except Exception:
                                        pass
                            if saved_imgs:
                                _chat_input_summary["images"] = saved_imgs
                            if saved_videos:
                                _chat_input_summary["videos"] = saved_videos

            _gen_start = time.perf_counter()

            # 4. Generate
            if streaming:
                if chat_ws_recorder:
                    chat_ws_recorder.start_turn(turn_index=0, request_ts_ms=0.0, input_summary=_chat_input_summary)
                chunk_queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def _run_generate():
                    try:
                        for chunk in worker.chat_streaming_generate(
                            session_id=session_id,
                            generate_audio=generate_audio,
                            max_new_tokens=max_new_tokens,
                            length_penalty=length_penalty,
                        ):
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, ("chunk", chunk))
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, ("done", None))
                    except Exception as e:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, ("error", e))

                gen_task = loop.run_in_executor(None, _run_generate)

                full_text = ""
                chunk_count = 0
                while True:
                    try:
                        tag, payload = await asyncio.wait_for(
                            chunk_queue.get(), timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        try:
                            await ws.send_json({"type": "heartbeat"})
                        except Exception:
                            break
                        continue
                    if tag == "chunk":
                        chunk_data = {"type": "chunk"}
                        if payload.text_delta:
                            chunk_data["text_delta"] = payload.text_delta
                            full_text += payload.text_delta
                        if payload.audio_data:
                            chunk_data["audio_data"] = payload.audio_data
                        if len(chunk_data) > 1:
                            await ws.send_json(chunk_data)
                        if chat_ws_recorder:
                            chat_ws_recorder.add_streaming_chunk(
                                text_delta=payload.text_delta,
                                audio_base64=payload.audio_data,
                            )
                        chunk_count += 1
                    elif tag == "done":
                        _gen_ids = getattr(worker.processor.model, '_streaming_generated_token_ids', None) if worker.processor else None
                        generated_tokens = len(_gen_ids) if _gen_ids else chunk_count
                        _elapsed = round((time.perf_counter() - _gen_start) * 1000, 1)
                        if chat_ws_recorder:
                            chat_ws_recorder.end_turn(timing={
                                "elapsed_ms": _elapsed,
                                "tokens": generated_tokens,
                                "input_tokens": pre_kv,
                            })
                        await ws.send_json({
                            "type": "done",
                            "text": full_text,
                            "generated_tokens": generated_tokens,
                            "input_tokens": pre_kv,
                            **({"recording_session_id": chat_ws_session_id} if chat_ws_session_id else {}),
                        })
                        break
                    elif tag == "error":
                        await ws.send_json({"type": "error", "error": str(payload)})
                        break

                try:
                    await asyncio.wait_for(gen_task, timeout=5.0)
                except asyncio.TimeoutError:
                    pass

            else:
                def _run_non_streaming():
                    return worker.chat_non_streaming_generate(
                        session_id=session_id,
                        max_new_tokens=max_new_tokens,
                        generate_audio=generate_audio,
                        use_tts_template=use_tts_template,
                        enable_thinking=enable_thinking,
                        tts_ref_audio=tts_ref_audio_ndarray,
                        length_penalty=length_penalty,
                    )

                result = await asyncio.to_thread(_run_non_streaming)

                text = result
                audio_data = None
                output_audio_np = None
                if isinstance(result, tuple):
                    text, waveform = result
                    if waveform is not None:
                        output_audio_np = waveform.astype(np.float32)
                        audio_bytes = output_audio_np.tobytes()
                        audio_data = base64.b64encode(audio_bytes).decode('utf-8')

                stats = getattr(worker.processor.model, '_last_chat_token_stats', {}) if worker.processor else {}
                _elapsed = round((time.perf_counter() - _gen_start) * 1000, 1)

                if chat_ws_recorder:
                    chat_ws_recorder.record_chat_turn(
                        turn_index=0,
                        request_ts_ms=0.0,
                        input_summary=_chat_input_summary,
                        output_text=text or "",
                        output_audio=output_audio_np,
                        timing={
                            "elapsed_ms": _elapsed,
                            "tokens": stats.get("generated_tokens", 0),
                            "input_tokens": pre_kv,
                        },
                    )

                await ws.send_json({
                    "type": "done",
                    "text": text or "",
                    "audio_data": audio_data,
                    "generated_tokens": stats.get("generated_tokens", 0),
                    "input_tokens": pre_kv,
                    **({"recording_session_id": chat_ws_session_id} if chat_ws_session_id else {}),
                })

        except Exception as e:
            logger.error(f"Chat WS error: {e}", exc_info=True)
            try:
                await ws.send_json({"type": "error", "error": str(e)})
            except Exception:
                pass
        finally:
            if chat_ws_recorder:
                try:
                    chat_ws_recorder.finalize()
                except Exception as _e:
                    logger.error(f"[ChatWS] recorder finalize failed: {_e}", exc_info=True)
            worker.state.status = WorkerStatus.IDLE
            worker.state.current_session_id = None

    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ========== Half-Duplex Stop 信号（每连接独立） ==========
# 每个 WS 连接创建独立的 threading.Event()，按 session_id 索引。
# HTTP POST /half_duplex/stop 广播到所有活跃 session。
# 安全性：
#   - dict 操作在 asyncio 单线程事件循环中，无并发写入
#   - threading.Event 本身线程安全（asyncio 线程 ↔ generate 工作线程）
_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _sanitize_session_id(session_id: str) -> str:
    """校验 session_id 只含安全字符，防止 path traversal"""
    if not _SESSION_ID_RE.match(session_id):
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', session_id)
        return safe
    return session_id


_half_duplex_stop_events: Dict[str, threading.Event] = {}

_half_duplex_ref_audio_cache: Dict[str, np.ndarray] = {}


@app.post("/half_duplex/stop")
async def half_duplex_stop():
    """停止所有正在进行的 Half-Duplex 生成"""
    if not _half_duplex_stop_events:
        return {"success": False, "message": "No active half-duplex session"}
    for sid, evt in _half_duplex_stop_events.items():
        evt.set()
        logger.info(f"Half-Duplex stop signal sent to session {sid}")
    return {"success": True, "message": f"Stop signal sent to {len(_half_duplex_stop_events)} session(s)"}


# ========== Half-Duplex WebSocket ==========

@app.websocket("/ws/half_duplex")
async def half_duplex_ws(ws: WebSocket):
    """Half-Duplex Audio WebSocket

    协议：
    1. Client → {"type": "prepare", "system_content": [...], "config": {...}}
       system_content 格式与 turn-based 相同: [{type:"text",text:...}, {type:"audio",data:...}, ...]
    2. Client → {"type": "audio_chunk", "audio_base64": "..."} (连续)
    3. Server → {"type": "vad_state", "speaking": true/false}
    4. Server → {"type": "generating"}
    5. Server → {"type": "chunk", ...} (流式)
    6. Server → {"type": "turn_done", ...}
    7. Client → {"type": "stop"} / Server → {"type": "timeout"}
    """
    if not _worker_ready():
        await ws.close(code=1013, reason="Worker not ready")
        return

    await ws.accept()
    conn_id = uuid.uuid4().hex[:8]
    session_id = ws.query_params.get("session_id", f"hdx_{conn_id}")
    logger.info(f"Half-Duplex WS connected (conn={conn_id}, session={session_id})")

    worker.state.status = WorkerStatus.BUSY_HALF_DUPLEX
    worker.state.current_session_id = session_id

    from vad import StreamingVAD, VadOptions

    vad: Optional[StreamingVAD] = None
    turn_index = 0
    session_start = time.perf_counter()
    timeout_s = 300
    generate_audio = True
    max_new_tokens = 256
    length_penalty = 1.1
    temperature = 0.7
    is_generating = False
    stop_event = threading.Event()

    vad_armed_at: float = 0.0
    INITIAL_GUARD_S = 0.5
    _half_duplex_stop_events[conn_id] = stop_event

    hdx_recorder: Optional[TurnBasedSessionRecorder] = None
    hdx_session_start_perf: float = 0.0

    try:
        while True:
            elapsed_s = time.perf_counter() - session_start
            if elapsed_s > timeout_s:
                await ws.send_json({"type": "timeout", "elapsed_s": round(elapsed_s, 1)})
                logger.info(f"Half-Duplex session timeout after {elapsed_s:.0f}s")
                break

            remaining = timeout_s - elapsed_s
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "prepare":
                config = msg.get("config", {})

                vad_cfg = config.get("vad", {})
                vad_options = VadOptions(
                    threshold=vad_cfg.get("threshold", 0.7),
                    min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 128),
                    min_silence_duration_ms=vad_cfg.get("min_silence_duration_ms", 500),
                    speech_pad_ms=vad_cfg.get("speech_pad_ms", 30),
                )
                vad = StreamingVAD(options=vad_options)

                gen_cfg = config.get("generation", {})
                max_new_tokens = gen_cfg.get("max_new_tokens", 256)
                length_penalty = gen_cfg.get("length_penalty", 1.1)
                temperature = gen_cfg.get("temperature", 0.7)

                tts_cfg = config.get("tts", {})
                generate_audio = tts_cfg.get("enabled", True)

                session_cfg = config.get("session", {})
                timeout_s = session_cfg.get("timeout_s", 300)
                session_start = time.perf_counter()

                worker.reset_half_duplex_session()

                # 解析 system_content 列表（与 turn-based 相同的 schema）
                ref_audio_ndarray: Optional[np.ndarray] = None
                system_content_items = msg.get("system_content", [])
                content_items: List[ContentItem] = []
                for item in system_content_items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and item.get("text"):
                        content_items.append(TextContent(text=item["text"]))
                    elif item.get("type") == "audio" and item.get("data"):
                        content_items.append(AudioContent(data=item["data"]))
                        if ref_audio_ndarray is None:
                            try:
                                audio_bytes = base64.b64decode(item["data"])
                                ref_audio_ndarray = np.frombuffer(audio_bytes, dtype=np.float32)
                            except Exception:
                                pass
                if content_items:
                    sys_msg = Message(role=Role.SYSTEM, content=content_items)
                else:
                    sys_msg = Message(role=Role.SYSTEM, content="You are a helpful assistant.")
                logger.info(f"[HalfDuplex] system_content: {len(content_items)} items")

                request = StreamingRequest(
                    session_id=session_id,
                    messages=[sys_msg],
                    is_last_chunk=True,
                    use_tts_template=generate_audio,
                )
                await asyncio.to_thread(worker.half_duplex_prefill, request)

                if generate_audio:
                    if ref_audio_ndarray is not None:
                        await asyncio.to_thread(worker.half_duplex_init_tts, ref_audio_ndarray)
                    elif worker.ref_audio_path:
                        await asyncio.to_thread(worker.half_duplex_init_tts)

                vad_armed_at = time.perf_counter()
                hdx_session_start_perf = time.perf_counter()

                from config import get_config
                hdx_cfg = get_config()
                if hdx_cfg.recording.enabled:
                    hdx_recorder = TurnBasedSessionRecorder(
                        session_id=session_id,
                        app_type="half_duplex_audio",
                        worker_id=worker.gpu_id,
                        config_snapshot={
                            "system_content_count": len(content_items),
                            "vad": vad_cfg,
                            "generation": gen_cfg,
                            "tts_enabled": generate_audio,
                            "timeout_s": timeout_s,
                        },
                        data_dir=hdx_cfg.data_dir,
                    )

                rec_sid = hdx_recorder.session_id if hdx_recorder else None
                await ws.send_json({
                    "type": "prepared",
                    "session_id": session_id,
                    "timeout_s": timeout_s,
                    **({"recording_session_id": rec_sid} if rec_sid else {}),
                })
                logger.info(f"[HalfDuplex] prepared: timeout={timeout_s}s, vad_threshold={vad_options.threshold}")

            elif msg_type == "audio_chunk":
                if vad is None:
                    await ws.send_json({"type": "error", "error": "Not prepared yet"})
                    continue
                if is_generating:
                    continue

                if time.perf_counter() - vad_armed_at < INITIAL_GUARD_S:
                    continue

                audio_b64 = msg.get("audio_base64", "")
                if not audio_b64:
                    continue

                audio_bytes = base64.b64decode(audio_b64)
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

                was_speaking = vad.is_speaking
                speech_segment = vad.feed(audio_chunk)

                if vad.is_speaking and not was_speaking:
                    await ws.send_json({"type": "vad_state", "speaking": True})
                elif not vad.is_speaking and was_speaking and speech_segment is None:
                    await ws.send_json({"type": "vad_state", "speaking": False})

                if speech_segment is not None:
                    await ws.send_json({"type": "vad_state", "speaking": False})
                    await ws.send_json({
                        "type": "generating",
                        "speech_duration_ms": round(len(speech_segment) / 16000 * 1000),
                    })
                    is_generating = True

                    try:
                        speech_duration_ms = round(len(speech_segment) / 16000 * 1000)
                        request_ts_ms = (time.perf_counter() - hdx_session_start_perf) * 1000

                        if hdx_recorder:
                            user_audio_rel = hdx_recorder.save_user_audio(turn_index, speech_segment)
                            hdx_recorder.start_turn(
                                turn_index=turn_index,
                                request_ts_ms=request_ts_ms,
                                input_summary={
                                    "type": "voice",
                                    "duration_ms": speech_duration_ms,
                                    "audio": user_audio_rel,
                                },
                            )

                        audio_b64_data = base64.b64encode(speech_segment.tobytes()).decode('utf-8')
                        user_msg = Message(
                            role=Role.USER,
                            content=[AudioContent(data=audio_b64_data)],
                        )
                        prefill_request = StreamingRequest(
                            session_id=session_id,
                            messages=[user_msg],
                            is_last_chunk=True,
                            use_tts_template=generate_audio,
                        )
                        await asyncio.to_thread(worker.half_duplex_prefill, prefill_request)

                        chunk_queue: asyncio.Queue = asyncio.Queue()
                        loop = asyncio.get_event_loop()
                        stop_event.clear()
                        gen_start = time.perf_counter()

                        def _run_generate():
                            try:
                                for chunk in worker.half_duplex_generate(
                                    session_id=session_id,
                                    generate_audio=generate_audio,
                                    max_new_tokens=max_new_tokens,
                                    length_penalty=length_penalty,
                                ):
                                    loop.call_soon_threadsafe(chunk_queue.put_nowait, ("chunk", chunk))
                                    if stop_event.is_set():
                                        break
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("done", None))
                            except Exception as e:
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("error", e))

                        gen_task = loop.run_in_executor(None, _run_generate)

                        full_text = ""
                        while True:
                            try:
                                item_type, payload = await asyncio.wait_for(
                                    chunk_queue.get(), timeout=2.0
                                )
                            except asyncio.TimeoutError:
                                try:
                                    await ws.send_json({"type": "heartbeat"})
                                except Exception:
                                    break
                                continue
                            if item_type == "chunk":
                                await ws.send_json({"type": "chunk", **payload.model_dump()})
                                if payload.text_delta:
                                    full_text += payload.text_delta
                                if hdx_recorder:
                                    hdx_recorder.add_streaming_chunk(
                                        text_delta=payload.text_delta,
                                        audio_base64=payload.audio_data,
                                    )
                            elif item_type == "done":
                                break
                            elif item_type == "error":
                                raise payload

                        await gen_task

                        gen_elapsed_ms = (time.perf_counter() - gen_start) * 1000
                        if hdx_recorder:
                            hdx_recorder.end_turn(timing={
                                "elapsed_ms": round(gen_elapsed_ms, 1),
                                "speech_input_ms": speech_duration_ms,
                            })

                        turn_index += 1
                        await ws.send_json({
                            "type": "turn_done",
                            "turn_index": turn_index,
                            "text": full_text,
                        })

                        vad.reset()
                        logger.info(f"[HalfDuplex] turn {turn_index} done, VAD reset")

                    except Exception as e:
                        logger.error(f"[HalfDuplex] generate failed: {e}", exc_info=True)
                        await ws.send_json({"type": "error", "error": str(e)})
                    finally:
                        is_generating = False

            elif msg_type == "stop":
                stop_event.set()
                logger.info(f"Half-Duplex stop requested (conn={conn_id})")
                break

            else:
                await ws.send_json({"type": "error", "error": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"Half-Duplex WS disconnected (conn={conn_id})")
    except Exception as e:
        logger.error(f"Half-Duplex WS error (conn={conn_id}): {e}", exc_info=True)
    finally:
        if hdx_recorder:
            try:
                hdx_recorder.finalize()
            except Exception as e:
                logger.error(f"[HalfDuplex] recorder finalize failed: {e}", exc_info=True)

        stop_event.set()
        _half_duplex_stop_events.pop(conn_id, None)
        if worker.state.status == WorkerStatus.BUSY_HALF_DUPLEX:
            worker.state.status = WorkerStatus.IDLE
            worker.state.current_session_id = None
            logger.info(f"Half-Duplex session ended (conn={conn_id}, turns={turn_index})")


# ========== Duplex WebSocket ==========

@app.websocket("/ws/duplex")
async def duplex_ws(ws: WebSocket):
    """Duplex WebSocket

    协议流程：
    1. Client 发送 {"type": "prepare", "system_prompt": "...", "ref_audio_path": "..."}
    2. 循环：
       a. Client 发送 {"type": "audio_chunk", "audio_base64": "..."}
       b. Server 执行 prefill + generate
       c. Server 返回 DuplexGenerateResult JSON
    3. Client 发送 {"type": "pause"} → 暂停循环
    4. Client 发送 {"type": "resume"} → 恢复循环
    5. Client 在 audio_chunk 中携带 force_listen: true → 强制模型持续监听（替代旧 interrupt）
    6. Client 发送 {"type": "stop"} → 结束会话
    """
    if not _worker_ready():
        await ws.close(code=1013, reason="Worker not ready")
        return

    if not worker.state.is_idle:
        # Gateway 排队保证并发安全，Worker 可能还在 cleanup，等待
        for _ in range(10):
            await asyncio.sleep(0.5)
            if worker.state.is_idle:
                break
        else:
            await ws.close(code=1013, reason=f"Worker busy after waiting 5s: {worker.state.status.value}")
            return

    await ws.accept()
    client_session_id = _sanitize_session_id(ws.query_params.get("session_id", ""))
    logger.info(f"Duplex WebSocket connected (client_session_id={client_session_id})")

    # Duplex 独占 Worker
    worker.state.status = WorkerStatus.DUPLEX_ACTIVE
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    worker.state.current_session_id = f"{ts}_{client_session_id}" if client_session_id else f"{ts}_duplex"
    # Duplex 会重置模型状态（prepare 会调用），Gateway 侧已清除 cached_hash

    pause_timeout_task: Optional[asyncio.Task] = None

    # finalize 异步栅栏：保证 finalize 完成后才能进入下一轮 prefill
    finalize_done = asyncio.Event()
    finalize_done.set()  # 初始状态：无需等待

    session_max_slice_nums: int = 1

    # Session 录制器（prepare 时初始化）
    session_recorder: Optional[DuplexSessionRecorder] = None
    chunk_idx: int = 0
    session_start_perf: float = time.perf_counter()

    async def pause_timeout_watchdog(timeout: float):
        """暂停超时看门狗"""
        await asyncio.sleep(timeout)
        logger.warning(f"Duplex pause timeout ({timeout}s), releasing Worker")
        worker.state.status = WorkerStatus.IDLE
        worker.state.current_session_id = None
        try:
            await ws.send_json({"type": "timeout", "reason": f"Pause timeout ({timeout}s)"})
            await ws.close(code=1000, reason="Pause timeout")
        except Exception:
            pass

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "prepare":
                system_prompt = msg.get("system_prompt", "You are a helpful assistant.")
                ref_audio_path = msg.get("ref_audio_path")
                ref_audio_b64 = msg.get("ref_audio_base64")
                # TTS ref audio: 独立字段，fallback 到 ref_audio_base64（向后兼容）
                tts_ref_audio_b64 = msg.get("tts_ref_audio_base64") or ref_audio_b64
                config_dict = msg.get("config")
                # Deferred Finalize 策略（默认开启，性能更优）：
                #   True  (模式 A): generate → 返回结果 → 异步 finalize
                #     finalize（feed 终止符 + 滑窗维护，~37ms）与网络传输重叠，
                #     LISTEN 省 ~37ms wall_clock，SPEAK 省 ~37ms wall_clock。
                #     通过 asyncio.Event 栅栏保证 finalize 在下一轮 prefill 前完成。
                #   False (模式 B): generate → 同步 finalize → 返回结果
                #     仍享受合并 feed 优化（终止符 + </unit> 一次 forward），
                #     但 finalize 耗时计入 wall_clock。
                # 实测：模式 A 比 B LISTEN wall_clock 低 ~30ms，SPEAK 低 ~50ms。
                use_deferred_finalize = msg.get("deferred_finalize", True)
                session_max_slice_nums = msg.get("max_slice_nums") or (config_dict.get("max_slice_nums") if config_dict else None) or 1

                if config_dict and worker.processor is not None:
                    duplex_view = worker.processor.set_duplex_mode()
                    duplex_view.config = DuplexConfig(**config_dict)

                # LLM ref audio → ref_audio_path（嵌入 system prompt）
                # TTS ref audio → prompt_wav_path（初始化 vocoder）
                # 两者可以不同，也可以相同（向后兼容：不提供 tts_ref_audio_base64 时复用 ref_audio_base64）
                import tempfile
                import soundfile as sf
                actual_ref_audio_path = ref_audio_path
                actual_tts_audio_path = None
                temp_files: list = []

                # LLM ref audio: base64 → 临时文件
                if ref_audio_b64 and not ref_audio_path:
                    ref_bytes = base64.b64decode(ref_audio_b64)
                    ref_ndarray = np.frombuffer(ref_bytes, dtype=np.float32)
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="duplex_llm_ref_")
                    sf.write(tmp.name, ref_ndarray, 16000)
                    actual_ref_audio_path = tmp.name
                    temp_files.append(tmp.name)
                    logger.info(f"Duplex LLM ref_audio: {len(ref_ndarray)} samples → {tmp.name}")

                # TTS ref audio: 与 LLM 相同则复用，不同则另存
                if tts_ref_audio_b64 and tts_ref_audio_b64 != ref_audio_b64:
                    tts_bytes = base64.b64decode(tts_ref_audio_b64)
                    tts_ndarray = np.frombuffer(tts_bytes, dtype=np.float32)
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="duplex_tts_ref_")
                    sf.write(tmp.name, tts_ndarray, 16000)
                    actual_tts_audio_path = tmp.name
                    temp_files.append(tmp.name)
                    logger.info(f"Duplex TTS ref_audio (independent): {len(tts_ndarray)} samples → {tmp.name}")
                elif actual_ref_audio_path:
                    # TTS 和 LLM 使用同一个文件
                    actual_tts_audio_path = actual_ref_audio_path

                _has_tts_field_d = bool(msg.get("tts_ref_audio_base64"))
                _tts_same_as_llm = (tts_ref_audio_b64 == ref_audio_b64) if tts_ref_audio_b64 else True
                logger.info(
                    f"[Duplex RefAudio] LLM={actual_ref_audio_path}, TTS={actual_tts_audio_path}, "
                    f"tts_field_present={_has_tts_field_d}, tts_same_as_llm={_tts_same_as_llm}"
                )

                try:
                    prompt = await asyncio.to_thread(
                        worker.duplex_prepare,
                        system_prompt_text=system_prompt,
                        ref_audio_path=actual_ref_audio_path,
                        prompt_wav_path=actual_tts_audio_path,
                    )
                    logger.info(f"Duplex prepared (deferred_finalize={use_deferred_finalize})")

                    duplex_type = "omni_duplex" if client_session_id.startswith("omni") else "audio_duplex"
                    from config import get_config
                    cfg = get_config()
                    if cfg.recording.enabled:
                        config_snap = {
                            "system_prompt": system_prompt,
                            "ref_audio": actual_ref_audio_path or cfg.ref_audio_path,
                            "deferred_finalize": use_deferred_finalize,
                            "max_slice_nums": session_max_slice_nums,
                        }
                        if config_dict:
                            config_snap.update({k: v for k, v in config_dict.items() if k != "max_slice_nums"})
                        session_recorder = DuplexSessionRecorder(
                            session_id=worker.state.current_session_id or client_session_id,
                            app_type=duplex_type,
                            worker_id=worker.gpu_id,
                            config_snapshot=config_snap,
                            data_dir=cfg.data_dir,
                        )
                    session_start_perf = time.perf_counter()
                    chunk_idx = 0

                    rec_sid = session_recorder.session_id if session_recorder else None
                    await ws.send_json({
                        "type": "prepared",
                        "prompt_length": len(prompt),
                        **({"recording_session_id": rec_sid} if rec_sid else {}),
                    })
                except Exception as e:
                    logger.error(f"Duplex prepare failed: {e}", exc_info=True)
                    await ws.send_json({"type": "error", "error": str(e)})
                finally:
                    # 清理临时文件
                    import os
                    for tmp_path in temp_files:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

            elif msg_type == "audio_chunk":
                if worker.state.status == WorkerStatus.DUPLEX_PAUSED:
                    await ws.send_json({"type": "error", "error": "Worker is paused"})
                    continue

                audio_b64 = msg.get("audio_base64")
                if not audio_b64:
                    await ws.send_json({"type": "error", "error": "Missing audio_base64"})
                    continue

                t_chunk_start = time.perf_counter()

                # 解码音频
                audio_bytes = base64.b64decode(audio_b64)
                audio_waveform = np.frombuffer(audio_bytes, dtype=np.float32)

                # 录制：保存用户音频 chunk
                user_audio_rel: Optional[str] = None
                if session_recorder:
                    user_audio_rel = session_recorder.save_user_audio(chunk_idx, audio_waveform)

                # 解码图像帧（可选）
                frame_list = None
                user_frame_rel: Optional[str] = None
                frame_b64_list = msg.get("frame_base64_list")
                if frame_b64_list:
                    from PIL import Image
                    import io
                    frame_list = []
                    for fb64 in frame_b64_list:
                        frame_bytes = base64.b64decode(fb64)
                        # 录制：保存原始 JPEG 帧（在 PIL 解码之前）
                        if session_recorder and user_frame_rel is None:
                            user_frame_rel = session_recorder.save_user_frame(chunk_idx, frame_bytes)
                        frame_list.append(Image.open(io.BytesIO(frame_bytes)))

                # per-chunk Force Listen 标记
                chunk_force_listen = bool(msg.get("force_listen", False))
                # per-chunk HD vision override（fallback 到 session 默认值）
                chunk_max_slice_nums: int = msg.get("max_slice_nums", session_max_slice_nums)

                try:
                    # 等待上一轮 finalize 完成（保证 KV cache 状态一致）
                    await finalize_done.wait()

                    # prefill + generate（不含 finalize，延迟执行）
                    def _duplex_step():
                        t0 = time.perf_counter()
                        prefill_result = worker.duplex_prefill(
                            audio_waveform=audio_waveform,
                            frame_list=frame_list,
                            max_slice_nums=chunk_max_slice_nums,
                        )
                        t_prefill = time.perf_counter()
                        gen_result = worker.duplex_generate(force_listen=chunk_force_listen)
                        t_gen = time.perf_counter()

                        prefill_ms = (t_prefill - t0) * 1000
                        kv_len = worker.processor.kv_cache_length if worker.processor else 0
                        return gen_result, prefill_ms, prefill_result, kv_len

                    result, prefill_ms, prefill_cost, kv_cache_len = await asyncio.to_thread(_duplex_step)
                    result.server_send_ts = time.time()

                    wall_clock_ms = (time.perf_counter() - t_chunk_start) * 1000
                    status = "LISTEN" if result.is_listen else "SPEAK"

                    # vision token 从 prefill 返回的实际 slice 数量计算（不再硬编码分辨率假设）
                    n_vision_images = prefill_cost.get("n_vision_images", 0) if isinstance(prefill_cost, dict) else 0
                    vision_tok = n_vision_images * 64
                    if not result.is_listen:
                        llm = result.cost_llm_ms or 0
                        tts_prep = result.cost_tts_prep_ms or 0
                        tts = result.cost_tts_ms or 0
                        t2w = result.cost_token2wav_ms or 0
                        total = result.cost_all_ms or 0
                        n_tok = result.n_tokens or 0
                        n_tts_tok = result.n_tts_tokens or 0
                        logger.info(
                            f"[GPU {worker.gpu_id}] SPEAK t={result.current_time} wall={wall_clock_ms:.0f}ms | "
                            f"prefill={prefill_ms:.0f} llm={llm:.0f} tts_prep={tts_prep:.0f} "
                            f"tts={tts:.0f} t2w={t2w:.0f} total={total:.0f}ms | "
                            f"tokens={n_tok} tts_tokens={n_tts_tok} kv={kv_cache_len} | "
                            f"vimg={n_vision_images} vtok={vision_tok} | "
                            f"text='{(result.text or '')[:20]}'"
                        )
                    else:
                        total = result.cost_all_ms or 0
                        logger.info(
                            f"[GPU {worker.gpu_id}] LISTEN t={result.current_time} wall={wall_clock_ms:.0f}ms | "
                            f"prefill={prefill_ms:.0f} generate={total:.0f}ms kv={kv_cache_len} | "
                            f"vimg={n_vision_images} vtok={vision_tok}"
                        )

                    result_dict = result.model_dump()
                    result_dict["wall_clock_ms"] = round(wall_clock_ms, 1)
                    result_dict["kv_cache_length"] = kv_cache_len
                    result_dict["vision_slices"] = n_vision_images
                    result_dict["vision_tokens"] = vision_tok

                    # 录制：保存 AI 音频并记录 chunk timeline
                    ai_audio_rel: Optional[str] = None
                    ai_audio_n_samples: int = 0
                    if session_recorder:
                        if not result.is_listen and result.audio_data:
                            try:
                                ai_bytes = base64.b64decode(result.audio_data)
                                ai_ndarray = np.frombuffer(ai_bytes, dtype=np.float32)
                                ai_audio_rel = session_recorder.save_ai_audio(
                                    session_recorder.turn_index,
                                    chunk_idx,
                                    ai_ndarray,
                                )
                                ai_audio_n_samples = len(ai_ndarray)
                            except Exception as e:
                                logger.warning(f"[SessionRecorder] failed to save AI audio: {e}")

                        receive_ts_ms = (t_chunk_start - session_start_perf) * 1000
                        session_recorder.record_chunk(
                            index=chunk_idx,
                            receive_ts_ms=receive_ts_ms,
                            result_dict=result_dict,
                            prefill_ms=prefill_ms,
                            user_audio_rel=user_audio_rel,
                            user_frame_rel=user_frame_rel,
                            ai_audio_rel=ai_audio_rel,
                            ai_audio_samples=ai_audio_n_samples,
                        )
                        chunk_idx += 1

                    if use_deferred_finalize:
                        # 模式 A：先发送结果，再异步 finalize（省 ~37ms wall_clock）
                        await ws.send_json({"type": "result", **result_dict})

                        finalize_done.clear()

                        async def _do_finalize():
                            try:
                                await asyncio.to_thread(worker.duplex_finalize)
                            except Exception as e:
                                logger.error(f"Duplex finalize failed: {e}", exc_info=True)
                            finally:
                                finalize_done.set()

                        asyncio.create_task(_do_finalize())
                    else:
                        # 模式 B：同步 finalize 后再发送（仍享受合并 feed 优化）
                        await asyncio.to_thread(worker.duplex_finalize)
                        await ws.send_json({"type": "result", **result_dict})
                except Exception as e:
                    logger.error(f"Duplex prefill/generate failed: {e}", exc_info=True)
                    await ws.send_json({"type": "error", "error": str(e)})

            elif msg_type == "pause":
                logger.info("Duplex paused")
                worker.state.status = WorkerStatus.DUPLEX_PAUSED
                worker.state.duplex_pause_time = time.time()

                # 启动暂停超时看门狗
                timeout = msg.get("timeout", worker.duplex_pause_timeout)
                if pause_timeout_task and not pause_timeout_task.done():
                    pause_timeout_task.cancel()
                pause_timeout_task = asyncio.create_task(pause_timeout_watchdog(timeout))

                await ws.send_json({"type": "paused", "timeout": timeout})

            elif msg_type == "resume":
                logger.info("Duplex resumed")
                worker.state.status = WorkerStatus.DUPLEX_ACTIVE
                worker.state.duplex_pause_time = None

                # 取消暂停超时看门狗
                if pause_timeout_task and not pause_timeout_task.done():
                    pause_timeout_task.cancel()
                    pause_timeout_task = None

                await ws.send_json({"type": "resumed"})

            elif msg_type == "interrupt":
                # 已弃用：interrupt 被 per-chunk force_listen 替代
                logger.warning("Duplex interrupt message received (deprecated, use force_listen in audio_chunk)")
                await ws.send_json({"type": "interrupted"})

            elif msg_type == "stop":
                logger.info("Duplex stopped by client")
                worker.duplex_stop()
                await ws.send_json({"type": "stopped"})
                break

            else:
                await ws.send_json({"type": "error", "error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Duplex WebSocket disconnected")
    except Exception as e:
        logger.error(f"Duplex WebSocket error: {e}", exc_info=True)
    finally:
        # 录制：finalize（flush recording.json + 更新 meta.json）
        if session_recorder:
            try:
                session_recorder.finalize()
            except Exception as e:
                logger.error(f"[SessionRecorder] finalize failed: {e}", exc_info=True)

        # 清理
        if pause_timeout_task and not pause_timeout_task.done():
            pause_timeout_task.cancel()

        try:
            worker.duplex_stop()
        except Exception:
            pass

        # 释放 GPU 显存（KV cache、TTS caches 等，约 1.5 GB）
        try:
            await asyncio.to_thread(worker.duplex_cleanup)
        except Exception as e:
            logger.error(f"Duplex cleanup failed: {e}", exc_info=True)

        worker.state.status = WorkerStatus.IDLE
        worker.state.current_session_id = None
        worker.state.duplex_pause_time = None
        logger.info("Duplex session ended, Worker back to IDLE")


# ============ 缓存状态查询 ==========

@app.get("/cache_info")
async def cache_info():
    """查询当前 Worker 的 KV Cache 状态"""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not ready")

    return {
        "status": worker.state.status.value,
        "note": "KV cache state is now tracked by Gateway (cached_hash on WorkerConnection)",
    }


@app.post("/clear_cache")
async def clear_cache():
    """手动清除 KV Cache（重置 Streaming 模型 session）"""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not ready")

    worker.reset_half_duplex_session()
    return {"success": True, "message": "Cache cleared"}


# ============ 入口 ============

def main():
    from config import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(description="MiniCPMO45 Worker")
    parser.add_argument("--port", type=int, default=None, help=f"Worker port (default: from config, base={cfg.worker_base_port})")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--model-path", type=str, default=None, help="Base model path")
    parser.add_argument("--pt-path", type=str, default=None, help="Custom weights path (.pt)")
    parser.add_argument("--ref-audio-path", type=str, default=None, help="Default ref audio path")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID (inferred from port if not set)")
    parser.add_argument("--worker-index", type=int, default=0, help="Worker index (0, 1, 2, ...)")
    parser.add_argument("--duplex-pause-timeout", type=float, default=None, help="Duplex pause timeout (s)")
    args = parser.parse_args()

    port = args.port or cfg.worker_port(args.worker_index)
    gpu_id = args.gpu_id if args.gpu_id is not None else args.worker_index

    backend = cfg.backend

    if backend == "cpp":
        cpp_cfg = cfg.cpp_backend
        WORKER_CONFIG.update({
            "backend": "cpp",
            "llamacpp_root": cpp_cfg.llamacpp_root,
            "model_dir": cpp_cfg.model_dir,
            "gpu_id": gpu_id,
            "ref_audio_path": args.ref_audio_path or cfg.ref_audio_path,
            "duplex_pause_timeout": args.duplex_pause_timeout or cfg.duplex_pause_timeout,
            "llm_model": cpp_cfg.llm_model,
            "cpp_server_port": cpp_cfg.cpp_server_port,
            "ctx_size": cpp_cfg.ctx_size,
            "n_gpu_layers": cpp_cfg.n_gpu_layers,
        })
        logger.info(f"Starting Worker (C++ backend) on port {port}, GPU {gpu_id}")
    else:
        WORKER_CONFIG.update({
            "backend": "pytorch",
            "model_path": args.model_path or cfg.model.model_path,
            "gpu_id": gpu_id,
            "pt_path": args.pt_path or cfg.model.pt_path,
            "ref_audio_path": args.ref_audio_path or cfg.ref_audio_path,
            "duplex_pause_timeout": args.duplex_pause_timeout or cfg.duplex_pause_timeout,
            "compile": cfg.compile,
            "chat_vocoder": cfg.chat_vocoder,
            "attn_implementation": cfg.attn_implementation,
        })
        logger.info(f"Starting Worker (PyTorch backend) on port {port}, GPU {gpu_id}")

    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
