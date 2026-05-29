"""PyTorch MiniCPM-o backend implementation for session runtimes."""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch

from core.schemas.metrics import BackendMetrics
from core.schemas.common import Message
from core.schemas.duplex import DuplexConfig, DuplexGenerateResult
from core.schemas.streaming import StreamingChunk, StreamingRequest, StreamingResponse

logger = logging.getLogger("pytorch_backend")


class PyTorchBackend:
    """MiniCPMO45 PyTorch inference backend.

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

        self.status = "loading"
        self.processor = None

        # Duplex 暂停超时监控 task
        self._duplex_timeout_task: Optional[asyncio.Task] = None

    def load_model(self) -> None:
        """加载模型（同步，在启动时调用）"""
        self.status = "loading"
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
        torch.cuda.empty_cache()

        self.status = "ready"
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

    # ========== Runtime backend surface ==========

    def metrics(self) -> Dict[str, Any]:
        """Return a sampled PyTorch backend metric snapshot."""
        if self.processor is None:
            return BackendMetrics(backend="pytorch").to_dict()
        return BackendMetrics(
            backend="pytorch",
            kv_cache_length=int(getattr(self.processor, "kv_cache_length", 0) or 0),
        ).to_dict()

    def chat_prefill(
        self,
        session_id: str,
        msgs: list,
        omni_mode: bool = False,
        max_slice_nums: Optional[int] = None,
        use_tts_template: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        chat_view = self.processor.set_chat_mode()
        return chat_view.prefill(
            session_id=session_id,
            msgs=msgs,
            omni_mode=omni_mode,
            max_slice_nums=max_slice_nums,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )

    def chat_init_tts(self, ref_audio: Optional[np.ndarray]) -> None:
        if ref_audio is not None:
            self.processor.model.init_token2wav_cache(prompt_speech_16k=ref_audio)
            return

        if self.ref_audio_path:
            import librosa

            loaded_ref, _ = librosa.load(self.ref_audio_path, sr=16000, mono=True)
            self.processor.model.init_token2wav_cache(prompt_speech_16k=loaded_ref)

    def chat_streaming_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        length_penalty: float = 1.1,
    ) -> Iterator[StreamingChunk]:
        chat_view = self.processor.set_chat_mode()
        yield from chat_view.streaming_generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    def chat_non_streaming_generate(
        self,
        session_id: str,
        max_new_tokens: int = 256,
        generate_audio: bool = False,
        use_tts_template: bool = True,
        enable_thinking: bool = False,
        tts_ref_audio: Optional[np.ndarray] = None,
        length_penalty: float = 1.1,
    ) -> Any:
        chat_view = self.processor.set_chat_mode()
        return chat_view.generate(
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            generate_audio=generate_audio,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
            tts_ref_audio=tts_ref_audio,
            tts_sampling_params=None,
            length_penalty=length_penalty,
        )

    def set_duplex_config(self, config: Optional[Dict[str, Any]]) -> None:
        if self.processor is None or not config:
            return
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.config = DuplexConfig(**config)

    def duplex_prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        length_penalty: float = 1.1,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> str:
        if sampling:
            self.set_duplex_config(sampling)
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
        duplex_view = self.processor.set_duplex_mode()
        return duplex_view.prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )

    def duplex_generate(self, force_listen: bool = False) -> DuplexGenerateResult:
        duplex_view = self.processor.set_duplex_mode()
        return duplex_view.generate(force_listen=force_listen)

    def duplex_finalize(self) -> None:
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.finalize()

    def duplex_stop(self) -> None:
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.stop()

    def duplex_cleanup(self) -> None:
        if self.processor is None:
            return
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.cleanup()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[GPU {self.gpu_id}] Duplex cleanup done, GPU memory released")

    def shutdown(self) -> None:
        """PyTorch backend currently has no external process to shut down."""
        return

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

