#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import Counter
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from enum import Enum

# 相对导入（同目录）
from .fc_duplex_capability import FcDuplexCapability
from .modeling_minicpmo import gen_logits
from .modeling_minicpmo import MiniCPMO as BaseMiniCPMO
from .modeling_minicpmo import MiniCPMODuplex as BaseMiniCPMODuplex
from .modeling_minicpmo import MiniCPMOPreTrainedModel
from .modeling_minicpmo import MiniCPMTTS
from .modeling_minicpmo import Resampler
from .processing_minicpmo import MiniCPMOProcessor
from .utils import as_dynamic_cache
from .utils import ChunkPrefillChunkGenerate
from .utils import drop_tokens_from_cache
from .utils import get_kv_cache_length
from .utils import realign_rotary_suffix
from .utils import streaming_token_decoder
from .utils import torch_clone_recursive
from .utils import TTSSamplingParams
from .utils import TTSStreamingGenerator

logger = logging.getLogger(__name__)


class ProcessorMode(Enum):
    """处理器模式枚举"""
    CHAT = "chat"           # 单工对话（非流式 TTS）
    STREAMING = "streaming" # 流式对话（流式 TTS）
    DUPLEX = "duplex"       # 双工对话（流式 TTS + 双工组件）


class MiniCPMO(BaseMiniCPMO):
    def __init__(self, config):
        super().__init__(config)
        self._init_unified_runtime_state()

    def _init_unified_runtime_state(self):
        self.default_tts_chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n"
            + self.think_str
            + "<|tts_bos|>' }}{% endif %}"
        )

        # ========== 统一模式支持（新增）==========
        self._current_mode: Optional[ProcessorMode] = None
        self._unified_initialized = False
        
        # 双 TTS 缓存（用于毫秒级切换）
        self._tts_streaming = None      # Token2wav (streaming TTS)
        self._tts_non_streaming = None  # CosyVoice2 (non-streaming TTS)
        
        # 双工能力组件（组合模式，通过 init_unified 初始化）
        self.duplex: Optional["DuplexCapability"] = None
        self.fc_duplex: Optional["FcDuplexCapability"] = None
        
        # 双工生成配置（传递给 DuplexCapability）
        self._duplex_config = {
            "generate_audio": True,
            "ls_mode": "explicit",
            "max_new_speak_tokens_per_chunk": 20,
            "text_repetition_penalty": 1.05,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "text_repetition_window_size": 512,
            "listen_prob_scale": 1.0,
            "force_listen_count": 0,
            "tts_temperature": 0.8,
            "tts_repetition_penalty": 1.05,
        }
        self._fc_duplex_config = {
            "generate_audio": True,
            "tool_format": "openai",
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "tts_temperature": 0.8,
            "tts_repetition_penalty": 1.05,
        }

    def init_token2wav(self, streaming=False, model_dir=None, enable_float16=False, n_timesteps=5):
        if streaming:
            if self.config.tts_config.audio_tokenizer_type != "s3tokenizer_step_audio":
                logger.warning("audio tokenizer type is set to s3tokenizer_step_audio")
                self.tts.config.audio_tokenizer_type = "s3tokenizer_step_audio"

            try:
                from stepaudio2 import Token2wav
            except ImportError:
                raise ImportError(f"please install Token2wav via: pip install stepaudio2-minicpmo")

            model_dir = self._ensure_asset_dir("assets/token2wav", model_dir)
            logger.info(f"Token2wav model_dir: {model_dir}, enable_float16: {enable_float16}, n_timesteps: {n_timesteps}")
            self.tts.audio_tokenizer = Token2wav(model_dir, float16=enable_float16, n_timesteps=n_timesteps)
            return self.tts.audio_tokenizer
        else:
            if self.config.tts_config.audio_tokenizer_type != "s3tokenizer":
                logger.warning("audio tokenizer type is set to s3tokenizer")
                self.tts.config.audio_tokenizer_type = "s3tokenizer"

            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2
            except ImportError:
                raise ImportError(f"please install cosyvoice via: pip install cosyvoice-minicpmo")

            model_dir = self._ensure_asset_dir("assets/CosyVoice2-0.5B", model_dir)
            self.tts.audio_tokenizer = CosyVoice2(model_dir=model_dir, load_jit=False, load_trt=False, fp16=False)
            return self.tts.audio_tokenizer

    # ==================== 统一模式方法（新增）====================
    
    def init_unified(
        self,
        pt_path: Optional[str] = None,
        preload_both_tts: bool = True,
        duplex_config: Optional[dict] = None,
        device: str = "cuda",
        chat_vocoder: str = "token2wav",
    ):
        """Unified initialization — load once, hot-switch between three modes.

        Args:
            pt_path: Optional extra .pt weights to override base model weights.
                Typical usage: load base model via from_pretrained, then overlay
                fine-tuned weights via pt_path.
            preload_both_tts: Whether to preload both TTS vocoders (recommended
                True — trades ~0.5 GB VRAM for millisecond-level switching).
            duplex_config: Duplex configuration dict.
            device: Target device.
            chat_vocoder: Vocoder for Chat (non-streaming) mode.
                "token2wav" = Step Audio Token2Wav (default, lightweight);
                "cosyvoice2" = CosyVoice2-0.5B (requires extra dependencies).
                When set to "token2wav", CosyVoice2 is not loaded, saving
                ~0.5 GB VRAM and its dependencies.

        After this call the following mode switches are available:
        - set_mode(ProcessorMode.CHAT)
        - set_mode(ProcessorMode.STREAMING)
        - set_mode(ProcessorMode.DUPLEX)
        """
        logger.info("Initializing unified mode...")
        self._chat_vocoder = chat_vocoder
        logger.info(f"Chat vocoder config: {chat_vocoder}")

        # Load extra .pt weights to override base model (if provided)
        if pt_path is not None:
            logger.info(f"Loading extra weights: {pt_path}")
            state_dict = torch.load(pt_path, map_location="cpu")
            prefix_counts = Counter(key.split(".", 1)[0] for key in state_dict)
            embed_key = "llm.model.embed_tokens.weight"
            resize_info = {"resized": False, "current_vocab": None, "checkpoint_vocab": None}
            if embed_key in state_dict:
                checkpoint_vocab = int(state_dict[embed_key].shape[0])
                current_vocab = int(self.llm.get_input_embeddings().weight.shape[0])
                resize_info = {
                    "resized": checkpoint_vocab > current_vocab,
                    "current_vocab": current_vocab,
                    "checkpoint_vocab": checkpoint_vocab,
                }
                if checkpoint_vocab > current_vocab:
                    logger.info(
                        "Resizing LLM token embeddings before loading extra weights: %d -> %d",
                        current_vocab,
                        checkpoint_vocab,
                    )
                    self.llm.resize_token_embeddings(checkpoint_vocab)
            info = self.load_state_dict(state_dict, strict=False)
            logger.info(f"Weights loaded — missing: {len(info.missing_keys)}, unexpected: {len(info.unexpected_keys)}")
            if info.unexpected_keys:
                logger.warning(f"Unexpected keys: {info.unexpected_keys[:5]}...")
            print(
                "[MiniCPMO45 init_unified] extra_weights_loaded "
                f"pt_path={pt_path} "
                f"num_keys={len(state_dict)} "
                f"prefix_counts={dict(sorted(prefix_counts.items()))} "
                f"resize_info={resize_info} "
                f"missing_count={len(info.missing_keys)} "
                f"unexpected_count={len(info.unexpected_keys)} "
                f"missing_first={list(info.missing_keys[:20])} "
                f"unexpected_first={list(info.unexpected_keys[:20])}",
                flush=True,
            )
            del state_dict

        # Update duplex config
        if duplex_config:
            self._duplex_config.update(duplex_config)
            fc_keys = set(self._fc_duplex_config)
            self._fc_duplex_config.update({k: v for k, v in duplex_config.items() if k in fc_keys})

        # Preload TTS vocoder
        self.init_token2wav(streaming=True)

        # Create DuplexCapability instance from the vendored duplex initializer.
        self.duplex = DuplexCapability.from_existing_model(
            model=self,
            device=device,
            **self._duplex_config,
        )
        self.fc_duplex = FcDuplexCapability(
            model=self,
            device=device,
            **self._fc_duplex_config,
        )

        self._unified_initialized = True

        # Default to Streaming mode
        self.set_mode(ProcessorMode.STREAMING)

        logger.info("Unified mode initialization complete")

    def set_mode(self, mode: ProcessorMode) -> None:
        """Set the current processor mode (millisecond-level switch).

        Args:
            mode: Target mode (CHAT / STREAMING / DUPLEX).
        """
        if mode == self._current_mode:
            return

        logger.info(f"Switching mode: {self._current_mode} -> {mode}")

        # Reset session state
        self.reset_session(reset_token2wav_cache=True)

        # Extra reset for duplex mode
        if mode == ProcessorMode.DUPLEX and hasattr(self, 'duplex') and self.duplex is not None:
            self.duplex._reset_streaming_state()
            self.duplex.decoder.reset()

        self._current_mode = mode

    def apply_torch_compile(
        self,
        mode: str = "default",
        dynamic: bool = True,
        skip_modules: Optional[List[str]] = None,
    ) -> "MiniCPMO":
        """Apply torch.compile to compute-intensive sub-modules.

        Must be called after init_unified().  DuplexCapability and other
        components access the same model instance by reference, so they
        automatically use the compiled versions after this call.

        Compile targets (compute-intensive sub-modules):
          - vpm: SiglipVisionTransformer (vision encoder)
          - llm.model: Qwen3Model backbone (LLM core; outer lm_head + generate
            logic is kept un-compiled)
          - resampler: vision resampler
          - tts.model: LlamaModel backbone (TTS core)

        NOT compiled:
          - apm (Whisper audio encoder): streaming-specific behavior + dynamic
            shapes, low compile benefit
          - tts.audio_tokenizer (Token2wav / CosyVoice2): external library, not
            a standard nn.Module
          - MiniCPMO outer wrapper: heavy Python control flow, low compile benefit

        Note: torch.compile only wraps the modules; actual Triton compilation is
        triggered on the first forward pass.  Call warmup_compile() afterwards to
        trigger compilation proactively.

        Args:
            mode: torch.compile mode.
                - "default": balanced compile time vs runtime speed (recommended)
                - "reduce-overhead": uses CUDA Graphs (static shapes only)
                - "max-autotune": maximum optimization (very long compile time)
            dynamic: Enable dynamic shape support (recommended True to avoid
                recompilation when shapes change).
            skip_modules: Module names to skip (e.g. ["llm.model"] for AWQ
                quantized LLM whose custom kernels are incompatible with compile).

        Returns:
            self (for method chaining).
        """
        import time as _time
        skip = set(skip_modules or [])
        logger.info(
            f"[torch.compile] Compiling sub-modules "
            f"(mode={mode}, dynamic={dynamic}, skip={skip or 'none'})"
        )
        t0 = _time.time()

        compile_kwargs = dict(mode=mode, dynamic=dynamic)
        compiled_modules: list = []
        skipped_modules: list = []

        # if hasattr(self, "vpm") and "vpm" not in skip:
        #     self.vpm = torch.compile(self.vpm, **compile_kwargs)
        #     compiled_modules.append("vpm")
        # elif "vpm" in skip:
        #     skipped_modules.append("vpm")

        if hasattr(self, "llm") and "llm.model" not in skip:
            self.llm.model = torch.compile(self.llm.model, **compile_kwargs)
            compiled_modules.append("llm.model")
        elif "llm.model" in skip:
            skipped_modules.append("llm.model")

        # if hasattr(self, "resampler") and "resampler" not in skip:
        #     self.resampler = torch.compile(self.resampler, **compile_kwargs)
        #     compiled_modules.append("resampler")
        # elif "resampler" in skip:
        #     skipped_modules.append("resampler")

        if hasattr(self, "tts") and hasattr(self.tts, "model") and "tts.model" not in skip:
            self.tts.model = torch.compile(self.tts.model, **compile_kwargs)
            compiled_modules.append("tts.model")
        elif "tts.model" in skip:
            skipped_modules.append("tts.model")

        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")

        elapsed = _time.time() - t0
        self._compiled = True
        self._compile_active = True
        logger.info(
            f"[torch.compile] Wrapping done ({elapsed:.2f}s), "
            f"compiled: {compiled_modules}"
            + (f", skipped: {skipped_modules}" if skipped_modules else "")
            + ". Actual compilation triggers on first forward."
        )
        return self

    def set_compile_enabled(self, enabled: bool) -> None:
        """Switch between compiled and eager execution for all compiled sub-modules.

        Only effective after apply_torch_compile() has been called.
        Compiled and eager modules share the same weights (zero copy),
        so switching is instant and costs no extra memory.
        """
        if not getattr(self, "_compiled", False):
            return
        if enabled == getattr(self, "_compile_active", True):
            return

        swapped: list = []

        if hasattr(self, "llm"):
            cur = self.llm.model
            if enabled:
                compiled = getattr(cur, "_compiled_ref", None)
                if compiled is not None:
                    self.llm.model = compiled
                    swapped.append("llm.model")
            else:
                orig = getattr(cur, "_orig_mod", None)
                if orig is not None:
                    orig._compiled_ref = cur
                    self.llm.model = orig
                    swapped.append("llm.model")

        if hasattr(self, "tts") and hasattr(self.tts, "model"):
            cur = self.tts.model
            if enabled:
                compiled = getattr(cur, "_compiled_ref", None)
                if compiled is not None:
                    self.tts.model = compiled
                    swapped.append("tts.model")
            else:
                orig = getattr(cur, "_orig_mod", None)
                if orig is not None:
                    orig._compiled_ref = cur
                    self.tts.model = orig
                    swapped.append("tts.model")

        self._compile_active = enabled
        logger.info(f"[torch.compile] {'enabled' if enabled else 'disabled'} → swapped {swapped}")

    def warmup_compile(
        self,
        warmup_video_path: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        max_warmup_chunks: int = 10,
        total_estimate_seconds: int = 400,
    ) -> None:
        """Trigger Triton kernel compilation via a real omni full-duplex session.

        Runs a complete duplex inference loop (prepare → prefill → generate →
        finalize) using an actual MP4 video, exercising the four compiled
        sub-modules (vpm / resampler / llm.model / tts.model) in their real
        execution context.  apm and token2wav are NOT compile targets; they
        simply run as part of the duplex pipeline.

        Per-unit timing breakdown is logged for each chunk.

        Args:
            warmup_video_path: MP4 video for warmup.  Defaults to
                ``assets/samples/compile.mp4``.
            ref_audio_path: Reference audio for TTS voice cloning.  Defaults to
                ``assets/ref_audio/ref_minicpm_signature.wav``.
            max_warmup_chunks: Maximum number of 1-second chunks to process.
        """
        if not getattr(self, "_compiled", False):
            logger.warning("[warmup] model not compiled, skipping warmup")
            return

        if self.duplex is None:
            logger.warning("[warmup] duplex not initialized, skipping warmup")
            return

        import sys
        import time as _time
        import threading

        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        if warmup_video_path is None:
            warmup_video_path = os.path.join(
                project_root, "assets", "samples", "compile.mp4"
            )
        if ref_audio_path is None:
            ref_audio_path = os.path.join(
                project_root, "assets", "ref_audio", "ref_minicpm_signature.wav"
            )

        if not os.path.isfile(warmup_video_path):
            logger.warning("[warmup] warmup video not found: %s, skipping", warmup_video_path)
            return
        if not os.path.isfile(ref_audio_path):
            logger.warning("[warmup] ref audio not found: %s, skipping", ref_audio_path)
            return

        # ── Persistent spinner ──
        _SPINNER = r"\|/-"
        _TOTAL_EST_S = total_estimate_seconds
        _G = "\033[32m"
        _B = "\033[1m"
        _R = "\033[0m"

        t_total = _time.time()
        _spin_idx = [0]
        _stage_info = ["(1/6) Initializing"]
        _lock = threading.Lock()
        _stop_evt = threading.Event()
        _out = sys.stderr

        def _render_spinner():
            elapsed = _time.time() - t_total
            remaining = max(0, _TOTAL_EST_S - elapsed)
            s = _SPINNER[_spin_idx[0] % 4]
            _spin_idx[0] += 1
            info = _stage_info[0]
            pct = min(100, int(elapsed / _TOTAL_EST_S * 100))
            bar_w = 20
            filled = int(bar_w * pct / 100)
            bar = "█" * filled + "░" * (bar_w - filled)
            return (
                f"\r{_G}{_B}[{s}] {bar} {pct:3d}% | {info} "
                f"| elapsed={elapsed:.0f}s remaining~{remaining:.0f}s{_R}\033[K"
            )

        def _spinner_loop():
            while not _stop_evt.wait(0.15):
                with _lock:
                    _out.write(_render_spinner())
                    _out.flush()

        def _set_stage(stage_idx: int, total: int, name: str, detail: str = ""):
            tag = f" {detail}" if detail else ""
            _stage_info[0] = f"({stage_idx}/{total}) {name}{tag}"

        def _log(msg: str):
            with _lock:
                _out.write(f"\r\033[K")
                _out.flush()
                logger.info(f"{_G}%s{_R}", msg)
                _out.write(_render_spinner())
                _out.flush()

        spinner_thread = threading.Thread(target=_spinner_loop, daemon=True)
        spinner_thread.start()

        _log(f"[warmup] starting omni duplex warmup (video={warmup_video_path}, "
             f"max_chunks={max_warmup_chunks}, est~{_TOTAL_EST_S}s)")

        # ── 1. Extract audio chunks and video frames from MP4 ──
        _set_stage(1, 6, "Extracting MP4")
        audio_chunks, frames = self._extract_mp4_chunks(
            warmup_video_path, max_chunks=max_warmup_chunks
        )
        _log(f"[warmup] (1/6) MP4 extracted: {len(audio_chunks)} chunks, {len(frames)} frames")
        if not audio_chunks:
            _stop_evt.set()
            with _lock:
                _out.write("\r\033[K")
                _out.flush()
            logger.warning("[warmup] no audio extracted from MP4, skipping")
            return

        # ── 2. Load reference audio ──
        _set_stage(2, 6, "Loading reference audio")
        import librosa as _librosa
        ref_audio, _ = _librosa.load(ref_audio_path, sr=16000, mono=True)
        _log("[warmup] (2/6) Reference audio loaded")

        # ── 3. Start duplex session ──
        _set_stage(3, 6, "Duplex prepare")
        self.duplex.prepare(
            prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
            suffix_system_prompt="<|audio_end|><|im_end|>",
            ref_audio=ref_audio,
            prompt_wav_path=ref_audio_path,
        )
        _log("[warmup] (3/6) Duplex prepare done")

        # ── 4. Per-chunk warmup (Triton compilation) ──
        tts_triggered = False
        num_chunks = min(len(audio_chunks), len(frames)) if frames else len(audio_chunks)

        for i in range(num_chunks):
            _set_stage(4, 6, "Triton compilation", f"unit {i}/{num_chunks}")
            frame_list = [frames[i]] if frames and i < len(frames) else None

            # ── Prefill ──
            t_pf = _time.time()
            try:
                prefill_result = self.duplex.streaming_prefill(
                    audio_waveform=audio_chunks[i],
                    frame_list=frame_list,
                    max_slice_nums=1,
                )
            except Exception as e:
                _log(f"[warmup] unit {i} prefill failed: {e}")
                break
            cost_prefill = _time.time() - t_pf

            pf_vp = prefill_result.get("cost_vision_process", 0) * 1000
            pf_ve = prefill_result.get("cost_vision_embed", 0) * 1000
            pf_vf = prefill_result.get("cost_vision_feed", 0) * 1000
            pf_ap = prefill_result.get("cost_audio_process", 0) * 1000
            pf_ae = prefill_result.get("cost_audio_embed", 0) * 1000
            pf_af = prefill_result.get("cost_audio_feed", 0) * 1000
            pf_all = cost_prefill * 1000

            # ── Generate ──
            t_gen = _time.time()
            try:
                gen_result = self.duplex.streaming_generate()
            except Exception as e:
                _log(f"[warmup] unit {i} generate failed: {e}")
                break
            cost_generate = _time.time() - t_gen

            is_listen = gen_result.get("is_listen", True)
            gen_llm = gen_result.get("cost_llm", 0) * 1000 if gen_result.get("cost_llm") else 0
            gen_tts_p = gen_result.get("cost_tts_prep", 0) * 1000 if gen_result.get("cost_tts_prep") else 0
            gen_tts = gen_result.get("cost_tts", 0) * 1000 if gen_result.get("cost_tts") else 0
            gen_t2w = gen_result.get("cost_token2wav", 0) * 1000 if gen_result.get("cost_token2wav") else 0
            gen_all = cost_generate * 1000
            decision = "LISTEN" if is_listen else "SPEAK"

            if not is_listen:
                tts_triggered = True
                text = gen_result.get("text", "")
                if text:
                    decision += f' "{text[:20]}"'

            elapsed = _time.time() - t_total
            remaining = max(0, _TOTAL_EST_S - elapsed)
            _log(
                f"[warmup] unit={i}/{num_chunks} | prefill: vis_proc={pf_vp:.0f}ms vis_emb={pf_ve:.0f}ms "
                f"vis_feed={pf_vf:.0f}ms aud_proc={pf_ap:.0f}ms aud_emb={pf_ae:.0f}ms "
                f"aud_feed={pf_af:.0f}ms total={pf_all:.0f}ms | generate: llm={gen_llm:.0f}ms "
                f"tts_prep={gen_tts_p:.0f}ms tts={gen_tts:.0f}ms token2wav={gen_t2w:.0f}ms "
                f"total={gen_all:.0f}ms | decision={decision} | elapsed={elapsed:.0f}s remaining~{remaining:.0f}s"
            )

            # ── Finalize ──
            try:
                self.duplex.finalize_unit()
            except Exception as e:
                _log(f"[warmup] unit {i} finalize failed: {e}")
                break

            if gen_result.get("end_of_turn", False):
                _log("[warmup] model emitted end_of_turn, stopping early")
                break

        # ── 5. TTS fallback if model stayed in LISTEN throughout ──
        if not tts_triggered and hasattr(self, "tts") and hasattr(self.tts, "model"):
            _set_stage(5, 6, "TTS fallback warmup")
            _log("[warmup] (5/6) TTS was not triggered during duplex, running fallback...")
            self._warmup_tts_fallback()
            _log("[warmup] (5/6) TTS fallback done")
        else:
            _log("[warmup] (5/6) TTS fallback skipped (already triggered)")

        # ── 6. Clean up duplex session state ──
        _set_stage(6, 6, "Cleanup")
        self.duplex._reset_streaming_state()
        self.duplex.decoder.reset()
        if hasattr(self.tts, "audio_tokenizer"):
            tokenizer = self.tts.audio_tokenizer
            for attr in ("stream_cache", "hift_cache_dict", "cache"):
                if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
                    setattr(tokenizer, attr, None)
        self.reset_session(reset_token2wav_cache=True)
        torch.cuda.empty_cache()

        # ── Stop spinner and print final line ──
        _stop_evt.set()
        spinner_thread.join(timeout=1)
        with _lock:
            _out.write("\r\033[K")
            _out.flush()

        total = _time.time() - t_total
        logger.info(
            "%s[warmup] ✓ omni duplex warmup complete (total=%.1fs, tts_triggered=%s)%s",
            _G, total, tts_triggered, _R,
        )

    def benchmark(
        self,
        video_paths: Optional[List[str]] = None,
        video_dir: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        system_prompt: str = "Streaming Omni Conversation.",
        max_chunks_per_video: int = 0,
    ) -> dict:
        """Run omni duplex benchmark, collecting per-module timing for LISTEN / SPEAK.

        Processes one or more MP4 videos through the full duplex pipeline
        (prepare → prefill → generate → finalize) and logs per-unit timing
        breakdown.  Summary statistics are printed at the end, grouped by
        decision type (LISTEN vs SPEAK).

        Args:
            video_paths: List of MP4 video file paths.
            video_dir: Directory containing MP4 videos (scanned for ``*.mp4``).
                Can be used together with *video_paths*; all paths are merged.
            ref_audio_path: Reference audio for TTS voice cloning.
            system_prompt: Content of the system prompt (wrapped in the
                standard ``<|im_start|>`` framing automatically).
            max_chunks_per_video: Maximum number of 1-second chunks to process
                per video.  ``0`` means process the entire video.

        Returns:
            Dict with ``units`` (per-unit records), counts, and elapsed time.
        """
        import time as _time

        if self.duplex is None:
            logger.warning("[bench] duplex not initialized, cannot benchmark")
            return {}

        # ── Resolve video paths ──
        resolved_videos: List[str] = []
        if video_dir and os.path.isdir(video_dir):
            for f in sorted(os.listdir(video_dir)):
                if f.lower().endswith(".mp4"):
                    resolved_videos.append(os.path.join(video_dir, f))
        if video_paths:
            for p in video_paths:
                if os.path.isfile(p):
                    resolved_videos.append(p)
                else:
                    logger.warning("[bench] video not found, skipping: %s", p)
        if not resolved_videos:
            logger.warning("[bench] no valid video files found, aborting")
            return {}

        # ── Resolve ref audio ──
        if ref_audio_path is None:
            project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            ref_audio_path = os.path.join(
                project_root, "assets", "ref_audio", "ref_minicpm_signature.wav"
            )
        if not os.path.isfile(ref_audio_path):
            logger.warning("[bench] ref audio not found: %s", ref_audio_path)
            return {}

        import librosa as _librosa
        ref_audio, _ = _librosa.load(ref_audio_path, sr=16000, mono=True)

        prefix_prompt = f"<|im_start|>system\n{system_prompt}\n<|audio_start|>"
        suffix_prompt = "<|audio_end|><|im_end|>"

        all_units: List[dict] = []
        t_total = _time.time()

        logger.info(
            "[bench] Starting benchmark: %d video(s), system_prompt=%r",
            len(resolved_videos), system_prompt,
        )

        for vid_idx, video_path in enumerate(resolved_videos):
            video_name = os.path.basename(video_path)
            max_c = max_chunks_per_video if max_chunks_per_video > 0 else 99999

            logger.info(
                "[bench] ── Video %d/%d: %s (max_chunks=%s) ──",
                vid_idx + 1, len(resolved_videos), video_name,
                max_chunks_per_video if max_chunks_per_video > 0 else "all",
            )

            # ── Extract audio chunks and video frames ──
            audio_chunks, frames = self._extract_mp4_chunks(
                video_path, max_chunks=max_c
            )
            if not audio_chunks:
                logger.warning("[bench] no audio extracted from %s, skipping", video_name)
                continue

            num_chunks = (
                min(len(audio_chunks), len(frames)) if frames else len(audio_chunks)
            )
            logger.info("[bench] Extracted %d chunks, %d frames", len(audio_chunks), len(frames))

            # ── Prepare duplex session ──
            self.duplex.prepare(
                prefix_system_prompt=prefix_prompt,
                suffix_system_prompt=suffix_prompt,
                ref_audio=ref_audio,
                prompt_wav_path=ref_audio_path,
            )

            # ── Per-chunk loop ──
            for i in range(num_chunks):
                frame_list = [frames[i]] if frames and i < len(frames) else None

                # Prefill
                t_pf = _time.time()
                try:
                    prefill_result = self.duplex.streaming_prefill(
                        audio_waveform=audio_chunks[i],
                        frame_list=frame_list,
                        max_slice_nums=1,
                    )
                except Exception as e:
                    logger.error("[bench] video=%s unit=%d prefill failed: %s", video_name, i, e)
                    break
                cost_prefill = (_time.time() - t_pf) * 1000

                pf = {
                    "vision_process": prefill_result.get("cost_vision_process", 0) * 1000,
                    "vision_embed": prefill_result.get("cost_vision_embed", 0) * 1000,
                    "vision_feed": prefill_result.get("cost_vision_feed", 0) * 1000,
                    "audio_process": prefill_result.get("cost_audio_process", 0) * 1000,
                    "audio_embed": prefill_result.get("cost_audio_embed", 0) * 1000,
                    "audio_feed": prefill_result.get("cost_audio_feed", 0) * 1000,
                    "total": cost_prefill,
                }

                # Generate
                t_gen = _time.time()
                try:
                    gen_result = self.duplex.streaming_generate()
                except Exception as e:
                    logger.error("[bench] video=%s unit=%d generate failed: %s", video_name, i, e)
                    break
                cost_generate = (_time.time() - t_gen) * 1000

                is_listen = gen_result.get("is_listen", True)
                gn = {
                    "llm": (gen_result.get("cost_llm") or 0) * 1000,
                    "tts_prep": (gen_result.get("cost_tts_prep") or 0) * 1000,
                    "tts": (gen_result.get("cost_tts") or 0) * 1000,
                    "token2wav": (gen_result.get("cost_token2wav") or 0) * 1000,
                    "total": cost_generate,
                }

                decision = "LISTEN" if is_listen else "SPEAK"
                unit_total = pf["total"] + gn["total"]
                text_snippet = ""
                if not is_listen:
                    text_snippet = gen_result.get("text", "")[:40]

                unit_record = {
                    "video": video_name,
                    "unit_idx": i,
                    "num_chunks": num_chunks,
                    "decision": decision,
                    "prefill": pf,
                    "generate": gn,
                    "unit_total": unit_total,
                }
                if text_snippet:
                    unit_record["text"] = text_snippet

                # Per-unit log
                decision_str = decision
                if text_snippet:
                    decision_str += f' "{text_snippet}"'

                logger.info(
                    "[bench] video=%s unit=%d/%d | %s | "
                    "prefill: vis_proc=%.0fms vis_emb=%.0fms vis_feed=%.0fms "
                    "aud_proc=%.0fms aud_emb=%.0fms aud_feed=%.0fms total=%.0fms | "
                    "generate: llm=%.0fms tts_prep=%.0fms tts=%.0fms token2wav=%.0fms "
                    "total=%.0fms | unit_total=%.0fms",
                    video_name, i, num_chunks, decision_str,
                    pf["vision_process"], pf["vision_embed"], pf["vision_feed"],
                    pf["audio_process"], pf["audio_embed"], pf["audio_feed"], pf["total"],
                    gn["llm"], gn["tts_prep"], gn["tts"], gn["token2wav"], gn["total"],
                    unit_total,
                )

                all_units.append(unit_record)

                # Finalize
                try:
                    self.duplex.finalize_unit()
                except Exception as e:
                    logger.error("[bench] video=%s unit=%d finalize failed: %s", video_name, i, e)
                    break

                if gen_result.get("end_of_turn", False):
                    logger.info("[bench] end_of_turn at unit %d, stopping video", i)
                    break

            # ── Cleanup after each video ──
            self.duplex._reset_streaming_state()
            self.duplex.decoder.reset()
            if hasattr(self.tts, "audio_tokenizer"):
                tokenizer = self.tts.audio_tokenizer
                for attr in ("stream_cache", "hift_cache_dict", "cache"):
                    if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
                        setattr(tokenizer, attr, None)
            self.reset_session(reset_token2wav_cache=True)
            torch.cuda.empty_cache()

        total_elapsed = _time.time() - t_total

        # ── Summary ──
        listen_units = [u for u in all_units if u["decision"] == "LISTEN"]
        speak_units = [u for u in all_units if u["decision"] == "SPEAK"]

        def _agg(records: list, key_path: str) -> dict:
            vals = []
            for r in records:
                v = r
                for k in key_path.split("."):
                    v = v[k]
                vals.append(v)
            if not vals:
                return {"avg": 0.0, "min": 0.0, "max": 0.0}
            return {
                "avg": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }

        def _print_group(label: str, records: list):
            if not records:
                logger.info("[bench] %s: (no units)", label)
                return
            n = len(records)
            logger.info("[bench] %s (n=%d):", label, n)

            pf_t = _agg(records, "prefill.total")
            logger.info(
                "[bench]   prefill:        avg=%.0fms  min=%.0fms  max=%.0fms",
                pf_t["avg"], pf_t["min"], pf_t["max"],
            )
            for key in ("vision_process", "vision_embed", "vision_feed",
                        "audio_process", "audio_embed", "audio_feed"):
                s = _agg(records, f"prefill.{key}")
                logger.info(
                    "[bench]     %-14s avg=%.0fms  min=%.0fms  max=%.0fms",
                    key + ":", s["avg"], s["min"], s["max"],
                )

            gn_t = _agg(records, "generate.total")
            logger.info(
                "[bench]   generate:       avg=%.0fms  min=%.0fms  max=%.0fms",
                gn_t["avg"], gn_t["min"], gn_t["max"],
            )
            for key in ("llm", "tts_prep", "tts", "token2wav"):
                s = _agg(records, f"generate.{key}")
                if s["avg"] > 0 or key == "llm":
                    logger.info(
                        "[bench]     %-14s avg=%.0fms  min=%.0fms  max=%.0fms",
                        key + ":", s["avg"], s["min"], s["max"],
                    )

            ut = _agg(records, "unit_total")
            logger.info(
                "[bench]   unit_total:     avg=%.0fms  min=%.0fms  max=%.0fms",
                ut["avg"], ut["min"], ut["max"],
            )

        logger.info("[bench] " + "=" * 60)
        logger.info("[bench] Benchmark Summary")
        logger.info("[bench] " + "=" * 60)
        logger.info(
            "[bench] Total: %d units (%d LISTEN, %d SPEAK) over %d video(s), elapsed=%.1fs",
            len(all_units), len(listen_units), len(speak_units),
            len(resolved_videos), total_elapsed,
        )
        logger.info("[bench]")
        _print_group("LISTEN", listen_units)
        logger.info("[bench]")
        _print_group("SPEAK", speak_units)
        logger.info("[bench] " + "=" * 60)

        def _build_group_stats(records: list) -> dict:
            if not records:
                return {}
            prefill_keys = ("vision_process", "vision_embed", "vision_feed",
                            "audio_process", "audio_embed", "audio_feed", "total")
            generate_keys = ("llm", "tts_prep", "tts", "token2wav", "total")
            return {
                "count": len(records),
                "prefill": {k: _agg(records, f"prefill.{k}") for k in prefill_keys},
                "generate": {k: _agg(records, f"generate.{k}") for k in generate_keys},
                "unit_total": _agg(records, "unit_total"),
            }

        return {
            "total_time": total_elapsed,
            "num_videos": len(resolved_videos),
            "num_units": len(all_units),
            "listen_count": len(listen_units),
            "speak_count": len(speak_units),
            "listen_stats": _build_group_stats(listen_units),
            "speak_stats": _build_group_stats(speak_units),
            "units": all_units,
        }

    def _extract_mp4_chunks(
        self,
        video_path: str,
        max_chunks: int = 10,
        sample_rate: int = 16000,
    ) -> tuple:
        """Extract 1-second audio chunks and corresponding video frames from MP4.

        Uses ffmpeg for both audio extraction and frame extraction (no cv2).

        Returns:
            (audio_chunks, frames): audio_chunks is list[np.ndarray] (16kHz mono),
            frames is list[PIL.Image].
        """
        import subprocess
        import tempfile
        from PIL import Image

        audio_chunks: list = []
        frames: list = []
        tmp_dir = tempfile.mkdtemp(prefix="warmup_")

        try:
            # ── Extract audio ──
            tmp_wav_path = os.path.join(tmp_dir, "audio.wav")
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-ar", str(sample_rate), "-ac", "1",
                    "-t", str(max_chunks),
                    "-f", "wav", "-y", tmp_wav_path,
                ],
                capture_output=True,
                check=True,
            )
            import librosa as _librosa
            audio, _ = _librosa.load(tmp_wav_path, sr=sample_rate, mono=True)

            chunk_size = sample_rate
            n_audio = min(max_chunks, len(audio) // chunk_size)
            for i in range(n_audio):
                audio_chunks.append(audio[i * chunk_size : (i + 1) * chunk_size])

            # ── Extract video frames (ffmpeg 1fps → JPEG) ──
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-t", str(max_chunks),
                    "-vf", "fps=1",
                    os.path.join(frames_dir, "frame_%04d.jpg"),
                ],
                capture_output=True,
                check=True,
            )
            frame_files = sorted(
                f for f in os.listdir(frames_dir) if f.endswith(".jpg")
            )
            for fname in frame_files[:n_audio]:
                frames.append(
                    Image.open(os.path.join(frames_dir, fname)).convert("RGB")
                )

        except Exception as e:
            logger.warning("[warmup] MP4 extraction failed: %s", e)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return audio_chunks, frames

    def _warmup_tts_fallback(self) -> None:
        """TTS sub-module fallback warmup (called when duplex stayed in LISTEN)."""
        import time as _time
        device = next(self.tts.model.parameters()).device
        tts_hidden = self.tts.config.hidden_size
        tts_dtype = self.tts.model.embed_tokens.weight.dtype

        with torch.no_grad():
            t0 = _time.time()
            prefill_len = 32
            dummy = torch.randn(1, prefill_len, tts_hidden, device=device, dtype=tts_dtype)
            pos = torch.arange(prefill_len, dtype=torch.long, device=device).unsqueeze(0)
            out = self.tts.model(inputs_embeds=dummy, position_ids=pos, use_cache=True)
            torch.cuda.synchronize(device)
            logger.info("[warmup]   tts prefill fallback done (%.1fs)", _time.time() - t0)

            t1 = _time.time()
            dec = torch.randn(1, 1, tts_hidden, device=device, dtype=tts_dtype)
            dec_pos = torch.tensor([[prefill_len]], dtype=torch.long, device=device)
            _ = self.tts.model(
                inputs_embeds=dec, position_ids=dec_pos,
                past_key_values=out.past_key_values, use_cache=True,
            )
            torch.cuda.synchronize(device)
            del out
            logger.info("[warmup]   tts decode fallback done (%.1fs)", _time.time() - t1)

    @property
    def current_mode(self) -> Optional[ProcessorMode]:
        """当前模式"""
        return self._current_mode

    @staticmethod
    def get_sys_prompt(ref_audio=None, mode="default", language="en", ref_audio_max_ms=None):
        if ref_audio is not None:
            if isinstance(ref_audio, str):
                if ref_audio == "assets/demo.wav":
                    import librosa

                    duration = ref_audio_max_ms / 1000.0 if ref_audio_max_ms else None
                    ref_audio, _ = librosa.load(ref_audio, sr=16000, mono=True, duration=duration)
                else:
                    import os

                    import librosa

                    if os.path.isfile(ref_audio) and os.path.exists(ref_audio):
                        duration = ref_audio_max_ms / 1000.0 if ref_audio_max_ms else None
                        ref_audio, _ = librosa.load(ref_audio, sr=16000, mono=True, duration=duration)
                    else:
                        logger.error(f"Could not find {ref_audio}")
                        ref_audio = None

            assert isinstance(ref_audio, np.ndarray), "ref_audio error"

        if mode == "omni":
            if language == "zh":
                sys_prompt = ""
                vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
                vc_prompt_suffix = (
                    "请用这种声音风格来为用户提供帮助。 请认真、高质量地回复用户的问题。 请用高自然度的方式和用户聊天。"
                )
            else:
                sys_prompt = ""
                vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                sys_msgs = {"role": "system", "content": [sys_prompt]}

            return sys_msgs
        elif mode == "audio_assistant":
            if language == "zh":
                vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
                vc_prompt_suffix = "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
            else:
                vc_prompt_prefix = "Use the voice in the audio prompt to synthesize new content."
                vc_prompt_suffix = "You are a helpful assistant with the above voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                logger.warning(
                    "Warning: ref_audio is None, speech generation will be performed based on the default voice."
                )
                sys_msgs = {"role": "system", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "audio_roleplay":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                print("Warning: ref_audio is None, speech generation will be performed based on the default voice.")
                sys_msgs = {"role": "system", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "voice_cloning":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio]}
            else:
                raise ValueError("ref_audio con't be None in voice_cloning mode.")

            return sys_msgs
        else:
            sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
            sys_msgs = {"role": "system", "content": [sys_prompt]}

            return sys_msgs

    @torch.inference_mode()
    def chat(
        self,
        image=None,
        msgs=None,
        tokenizer=None,  # deprecated
        processor=None,  # deprecated
        vision_hidden_states=None,
        max_new_tokens=4096,
        min_new_tokens=0,
        do_sample=True,
        sampling=None,  # deprecated, please use do_sample
        max_inp_length=8192,
        stream=False,
        stream_input=False,
        max_slice_nums=None,
        use_image_id=None,
        enable_thinking=False,
        use_tts_template=False,
        generate_audio=False,
        output_audio_path=None,
        output_tts_inputs_embeds_path=None,
        # add
        omni_mode=False,
        omni_input=None,  # deprecated, please use omni_mode
        teacher_forcing=False,
        return_prompt=False,
        tts_proj_layer=-1,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        merge_audio_from_same_content=True,
        tts_ref_audio: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if sampling is not None:
            do_sample = sampling
        if omni_input is not None:
            omni_mode = omni_input
        if tts_sampling_params is None:
            tts_sampling_params = TTSSamplingParams()

        should_return_waveform = bool(use_tts_template and generate_audio and not stream)
        temp_audio_path = None
        audio_path_for_base = output_audio_path
        if should_return_waveform and not audio_path_for_base:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="minicpmo_chat_", delete=False)
            temp_audio_path = tmp.name
            tmp.close()
            audio_path_for_base = temp_audio_path

        captured_stats = {}
        original_generate = self.generate
        sentinel = object()
        previous_generate_attr = self.__dict__.get("generate", sentinel)
        should_override_tts_prompt = tts_ref_audio is not None
        previous_tts_attr = (
            self.__dict__.get("_generate_speech_non_streaming", sentinel)
            if should_override_tts_prompt
            else sentinel
        )

        def _capture_generate(*args, **generate_kwargs):
            res, outputs = original_generate(*args, **generate_kwargs)
            input_ids = generate_kwargs.get("input_ids")
            sequences = getattr(outputs, "sequences", None)
            if input_ids is not None and sequences is not None:
                try:
                    captured_stats["input_tokens"] = int(input_ids[0].shape[0])
                    captured_stats["generated_tokens"] = int(sequences[0].shape[0])
                except Exception:
                    pass
            return res, outputs

        def _generate_speech_with_ref_audio(
            _self,
            outputs,
            tts_bound,
            tts_proj_layer,
            audio_prompt,
            output_tts_inputs_embeds_path=None,
            tts_sampling_params=TTSSamplingParams(),
        ):
            return BaseMiniCPMO._generate_speech_non_streaming(
                _self,
                outputs=outputs,
                tts_bound=tts_bound,
                tts_proj_layer=tts_proj_layer,
                audio_prompt=tts_ref_audio,
                output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
                tts_sampling_params=tts_sampling_params,
            )

        try:
            try:
                self.generate = _capture_generate
                if should_override_tts_prompt:
                    import types

                    self._generate_speech_non_streaming = types.MethodType(
                        _generate_speech_with_ref_audio, self
                    )
                result = BaseMiniCPMO.chat(
                    self,
                    image=image,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    processor=processor,
                    vision_hidden_states=vision_hidden_states,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=do_sample,
                    max_inp_length=max_inp_length,
                    stream=stream,
                    stream_input=stream_input,
                    max_slice_nums=max_slice_nums,
                    use_image_id=use_image_id,
                    enable_thinking=enable_thinking,
                    use_tts_template=use_tts_template,
                    generate_audio=generate_audio,
                    output_audio_path=audio_path_for_base,
                    output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
                    omni_mode=omni_mode,
                    teacher_forcing=teacher_forcing,
                    return_prompt=return_prompt,
                    tts_proj_layer=tts_proj_layer,
                    tts_sampling_params=tts_sampling_params,
                    merge_audio_from_same_content=merge_audio_from_same_content,
                    **kwargs,
                )
            finally:
                if previous_generate_attr is sentinel:
                    self.__dict__.pop("generate", None)
                else:
                    self.generate = previous_generate_attr
                if should_override_tts_prompt:
                    if previous_tts_attr is sentinel:
                        self.__dict__.pop("_generate_speech_non_streaming", None)
                    else:
                        self._generate_speech_non_streaming = previous_tts_attr
        except Exception:
            if temp_audio_path is not None and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    pass
            raise

        if captured_stats:
            self._last_chat_token_stats = captured_stats

        generated_waveform = None
        if (
            should_return_waveform
            and audio_path_for_base
            and os.path.exists(audio_path_for_base)
            and os.path.getsize(audio_path_for_base) > 0
        ):
            try:
                generated_waveform, _ = sf.read(audio_path_for_base, dtype="float32")
            except Exception:
                logger.exception("Failed to read generated chat audio from %s", audio_path_for_base)
                generated_waveform = None
            finally:
                if temp_audio_path is not None:
                    try:
                        os.unlink(temp_audio_path)
                    except OSError:
                        pass

        if not should_return_waveform or generated_waveform is None:
            return result

        if return_prompt:
            if isinstance(result, tuple):
                answer = result[0]
                prompt = result[1] if len(result) > 1 else None
            else:
                answer = result
                prompt = None
            return answer, prompt, generated_waveform

        answer = result[0] if isinstance(result, tuple) else result
        return answer, generated_waveform

    # for sliding window

    @torch.inference_mode()
    def non_streaming_prefill(
        self,
        session_id,
        msgs,
        image=None,
        omni_mode=False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template=False,
        enable_thinking=False,
        stream_input=False,
        max_inp_length=8192,
        merge_audio_from_same_content=True,
    ):
        """一次性 prefill 所有消息到 KV cache（非流式，复用 chat 的消息解析逻辑）

        与 streaming_prefill 的区别：
        - streaming_prefill 每次处理 1 条 msg，需要调用多次
        - non_streaming_prefill 一次处理所有 msgs，只调用一次

        两者都不加 generation_prompt，都设置好 KV cache 状态，
        之后统一用 streaming_generate() 或 non_streaming_generate() 做解码。

        Args:
            session_id: 会话 ID
            msgs: 消息列表 [{role, content}, ...]，content 可含 PIL.Image / np.ndarray / str
            image: 兼容 chat() 的 image 参数（一般传 None，图像集成到 msgs 中）
            omni_mode: 是否为 omni 模式（视频输入时为 True）
            max_slice_nums: HD 图像最大切片数
            use_image_id: 是否使用图像 ID
            use_tts_template: 是否使用 TTS 模板
            enable_thinking: 是否启用思考模式
            stream_input: 音频输入模式（False=完整音频）
            max_inp_length: 最大输入长度
            merge_audio_from_same_content: 是否合并同一 content 中的音频

        Returns:
            str: 构建的 prompt 字符串
        """
        assert session_id is not None, "session_id cannot be None"

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        # ── 1. 消息解析（复用 chat() 的逻辑） ──

        if isinstance(msgs, str):
            msgs = json.loads(msgs)

        copy_msgs = deepcopy(msgs)
        assert len(copy_msgs) > 0, "msgs is empty"

        if image is not None and isinstance(copy_msgs[0]["content"], str):
            copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

        images = []
        audios = []
        audio_parts = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["system", "user", "assistant"]
            if i == 0:
                assert role in ["user", "system"], "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("<image>./</image>")
                elif isinstance(c, np.ndarray):
                    audios.append(c)
                    audio_parts.append(i)
                    cur_msgs.append("<audio>./</audio>")
                    use_tts_template = True
                elif isinstance(c, str):
                    cur_msgs.append(c)

            if omni_mode or stream_input:
                msg["content"] = "".join(cur_msgs)
            else:
                msg["content"] = "\n".join(cur_msgs)

        prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=False,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )

        if not merge_audio_from_same_content:
            audio_parts = None

        # ── 2. Tokenize + 预处理 ──

        inputs = self.processor(
            [prompt],
            [images],
            [audios],
            [audio_parts] if audio_parts is not None else None,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            stream_input=stream_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        inputs.pop("image_sizes", None)

        # ── 3. Session 状态初始化（与 streaming_prefill 对齐） ──

        self.reset_session(reset_token2wav_cache=False)
        self.session_id = session_id
        self.init_streaming_processor()

        # ── 4. Embedding 计算 ──

        model_inputs = {
            "input_ids": inputs["input_ids"],
            "audio_features": inputs.get("audio_features"),
            "audio_feature_lens": inputs.get("audio_feature_lens"),
            "image_bound": inputs.get("image_bound"),
            "audio_bounds": inputs.get("audio_bounds"),
            "spk_bounds": inputs.get("spk_bounds"),
        }

        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs["pixel_values"]
            model_inputs["tgt_sizes"] = inputs.get("tgt_sizes")

        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        inputs_embeds = self.get_omni_embedding(
            model_inputs,
            input_embeddings=model_inputs["inputs_embeds"],
            chunk_length=self.config.audio_chunk_length,
        )

        # ── 5. KV Cache Prefill ──

        round_id = self._next_round_id
        self._pending_round_id = round_id
        seq_len = inputs_embeds.shape[1]
        self._enforce_text_window()
        cache_length = self._get_kv_cache_length()

        attention_mask = torch.ones(
            (1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device
        )

        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )

        self.llm_past_key_values = as_dynamic_cache(outputs["past_key_values"])
        self._register_chunk(
            seq_len,
            "user",
            round_id=round_id,
            input_ids=inputs["input_ids"],
            tokenizer=self.processor.tokenizer,
        )
        self._enforce_text_window()
        if self.force_rope_reindex:
            self._force_reindex_all_cache()

        logger.info(
            f"non_streaming_prefill done: session={session_id}, "
            f"prompt_len={seq_len}, kv_cache_len={self._get_kv_cache_length()}"
        )

        return prompt

    @torch.inference_mode()
    def non_streaming_generate(
        self,
        session_id,
        max_new_tokens=256,
        do_sample=True,
        min_new_tokens=0,
        generate_audio=False,
        use_tts_template=True,
        enable_thinking=False,
        tts_ref_audio=None,
        tts_sampling_params=None,
        output_audio_path=None,
        length_penalty=1.1,
        tts_proj_layer=-1,
    ):
        """基于已有 KV cache 做非流式 HF generate + 可选 TTS

        必须在 non_streaming_prefill() 之后调用。
        """
        assert self.llm_past_key_values is not None, \
            "KV cache is empty — call non_streaming_prefill() first"

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
        tokenizer = self.processor.tokenizer

        # 1. 构建 bos string（与 streaming_generate 对齐）
        bos_input = "".join([
            "<|im_end|>\n<|im_start|>assistant\n",
            "" if enable_thinking else self.think_str.replace("\\n", "\n"),
            "<|tts_bos|>" if use_tts_template else "",
        ])

        bos_input_ids = tokenizer.encode(bos_input)
        bos_input_ids = torch.tensor(
            bos_input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        bos_embeds = self.llm.get_input_embeddings()(bos_input_ids)

        # 2. bos prefill（注入 KV cache）
        cache_length = self._get_kv_cache_length()
        attention_mask = torch.ones(
            (1, cache_length + bos_embeds.shape[1]),
            dtype=torch.bool, device=self.device,
        )

        bos_outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=bos_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )
        self.llm_past_key_values = as_dynamic_cache(bos_outputs["past_key_values"])

        bos_seq_len = bos_embeds.shape[1]
        round_id = self._next_round_id
        self._pending_round_id = round_id
        self._register_chunk(
            bos_seq_len, "assistant", round_id=round_id,
            input_ids=bos_input_ids, tokenizer=tokenizer,
        )

        # 3. HF generate（基于 KV cache）
        generation_config = self.prepare_generation_config(
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=1,
            length_penalty=length_penalty,
        )
        generation_config.pop("max_new_tokens", None)

        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]

        cache_length_for_gen = self._get_kv_cache_length()
        gen_attention_mask = torch.ones(
            (1, cache_length_for_gen + 1),
            dtype=torch.bool, device=self.device,
        )

        last_logits = bos_outputs.logits[:, -1:, :]
        next_token = torch.argmax(last_logits, dim=-1)

        outputs = self.llm.generate(
            input_ids=next_token,
            past_key_values=self.llm_past_key_values,
            attention_mask=gen_attention_mask,
            pad_token_id=0,
            eos_token_id=terminators,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **generation_config,
        )

        # 4. 文本提取
        generated_ids = outputs.sequences[0]
        full_sequence = torch.cat([bos_input_ids[0], generated_ids])
        full_sequences = full_sequence.unsqueeze(0)
        outputs["full_sequences"] = full_sequences

        self._last_chat_token_stats = {
            "input_tokens": cache_length + bos_seq_len,
            "generated_tokens": len(generated_ids),
        }

        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )
        for term_str in self.terminators:
            text = text.replace(term_str, "")
        text = text.rstrip("<|tts_eos|>").strip()

        # 更新 KV cache 状态
        self.llm_past_key_values = as_dynamic_cache(outputs.past_key_values) \
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None \
            else self.llm_past_key_values
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = True

        # 5. TTS（可选）
        generated_waveform = None
        if use_tts_template and generate_audio:
            try:
                tts_bos_token = tokenizer.convert_tokens_to_ids("<|tts_bos|>")
                tts_eos_token = tokenizer.convert_tokens_to_ids("<|tts_eos|>")

                tts_bos_indices = []
                tts_eos_indices = []
                for i, x in enumerate(full_sequences[0]):
                    if x == tts_bos_token:
                        tts_bos_indices.append(i + 1)
                    elif x == tts_eos_token:
                        tts_eos_indices.append(i)

                tts_bos_idx = tts_bos_indices[-1] if tts_bos_indices else -1
                tts_eos_idx = tts_eos_indices[-1] if tts_eos_indices else None
                tts_bound = (tts_bos_idx, tts_eos_idx)

                _tts_audio_prompt = tts_ref_audio
                if _tts_audio_prompt is not None:
                    logger.info(f"[non_streaming_generate TTS] ref_audio: {len(_tts_audio_prompt)} samples")
                else:
                    logger.warning("[non_streaming_generate TTS] No ref audio")

                if tts_sampling_params is None:
                    tts_sampling_params = TTSSamplingParams()

                generated_waveform = self._generate_speech_non_streaming(
                    outputs=outputs,
                    tts_bound=tts_bound,
                    tts_proj_layer=tts_proj_layer,
                    audio_prompt=_tts_audio_prompt,
                    tts_sampling_params=tts_sampling_params,
                )
                if isinstance(generated_waveform, torch.Tensor):
                    generated_waveform = generated_waveform.cpu().numpy()

                if output_audio_path and generated_waveform is not None:
                    import soundfile as sf
                    sf.write(output_audio_path, generated_waveform, samplerate=24000)
            except:
                import traceback
                traceback.print_exc()
                generated_waveform = None

        logger.info(
            f"non_streaming_generate done: session={session_id}, "
            f"generated_tokens={len(generated_ids)}, "
            f"kv_cache_len={self._get_kv_cache_length()}, "
            f"has_audio={generated_waveform is not None}"
        )

        if generated_waveform is not None:
            return text, generated_waveform
        return text

    @torch.inference_mode()
    def streaming_prefill(
        self,
        session_id,
        msgs,
        tokenizer=None,  # deprecated
        omni_mode=True,
        max_slice_nums=None,
        use_tts_template=True,
        enable_thinking=False,
        is_last_chunk=False,  # for audio chunk, if is the last chunk, set to True
        stream_input=None,  # None=auto (is_not_system_prefill), False=完整音频, True=实时流式音频(双工)
        **kwargs,
    ):
        assert session_id is not None, "session_id cannot be None"
        self.is_first = self.session_id is None or session_id != self.session_id

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        images = []
        audios = []

        assert len(msgs) == 1
        copy_msgs = deepcopy(msgs)
        msg = copy_msgs[0]

        assert msg["role"] in ["system", "user", "assistant"]
        is_not_system_prefill = msg["role"] != "system"

        content = msg["content"]
        cur_msgs = []
        for j, c in enumerate(content):
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("<image>./</image>")
            elif isinstance(c, np.ndarray):
                audios.append(c)
                cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur_msgs.append(c)
            else:
                logger.error(f"Invalid content type: {c}, ignore it.")

        cur_contents = "".join(cur_msgs) if omni_mode else "\n".join(cur_msgs)

        if msg["role"] in ["system", "assistant"]:
            self.new_user_msg = True
            self.audio_past_key_values = None

        if self.is_first:
            self.reset_session(reset_token2wav_cache=False)
            self.session_id = session_id

            self.init_streaming_processor()

            if msg["role"] == "user":
                # 没有 system prefill，第一个 user turn 的第一个 segment
                # 不使用 apply_chat_template，手动构建 prompt 以避免自动添加 <|im_end|>
                prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False  # 标记后续 segments 不需要再添加 user 前缀
            else:
                # system 或 assistant prefill，使用 apply_chat_template
                msg["content"] = cur_contents
                prompt = self.processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            add_special_tokens = True  # add bos
        else:
            # 非首次 prefill
            if self.new_user_msg and msg["role"] == "user":
                # 新的 user turn 的第一个 segment
                if self.llm_generated:
                    # todo: when to set llm_generate_completed?
                    if self.llm_generate_completed:
                        prompt = "<|im_end|>\n<|im_start|>user\n" + cur_contents
                    else:
                        prompt = "<|tts_eos|><|im_end|>\n<|im_start|>user\n" + cur_contents
                else:
                    prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False
            else:
                # 同一个 turn 的后续 segments，直接使用内容
                prompt = cur_contents
            add_special_tokens = False

        # when first user audio prefill, ensure audio length satisfies FIRST_CHUNK_MS requirements
        if is_not_system_prefill and len(audios) > 0 and self.audio_chunk_idx == 0:
            assert len(audios) == 1, f"streaming mode only supports single audio, currently {len(audios)}"
            first_chunk_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
            if len(audios[0]) < first_chunk_samples:
                pad_len = first_chunk_samples - len(audios[0])
                audios[0] = np.concatenate([np.zeros(pad_len, dtype=audios[0].dtype), audios[0]])

        # stream_input: None=auto, False=完整音频, True=实时流式（双工）
        _stream_input = stream_input if stream_input is not None else is_not_system_prefill

        # online_streaming: 控制 processor 是否使用流式 mel 处理
        # 完整音频（stream_input=False）时不用流式 mel
        _online_streaming = is_not_system_prefill if _stream_input else False

        model_inputs = self.processor(
            [prompt],
            [images],
            [audios],
            max_slice_nums=1 if max_slice_nums is None else max_slice_nums,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            max_length=None,
            sampling_rate=16000,
            add_special_tokens=add_special_tokens,
            online_streaming=_online_streaming,
            audio_chunk_idx=self.audio_chunk_idx,
            is_last_chunk=is_last_chunk,
        ).to(self.device)

        # DEBUG: 打印 mel 特征的 checksum（用于诊断 rollback 不一致问题）
        if len(audios) > 0 and is_not_system_prefill and hasattr(self, "_debug_prefill") and self._debug_prefill:
            audio_feats = model_inputs.get("audio_features", None)
            if audio_feats is not None and hasattr(audio_feats, "sum"):
                mel_sum = audio_feats.sum().item()
                mel_shape = audio_feats.shape
                print(
                    f"[DEBUG prefill] audio_chunk_idx={self.audio_chunk_idx}, mel_sum={mel_sum:.6f}, mel_shape={mel_shape}"
                )
            else:
                print(f"[DEBUG prefill] audio_chunk_idx={self.audio_chunk_idx}, audio_feats type={type(audio_feats)}")

        if len(audios) > 0 and is_not_system_prefill:
            self.audio_chunk_idx += 1

        # 1. prepare input embeddings
        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        # get audio embedding with audio_past_key_values
        # todo: should pass chunk_length=self.config.audio_chunk_length ?
        inputs_embeds = self.get_omni_embedding(
            model_inputs, input_embeddings=model_inputs["inputs_embeds"], stream_input=_stream_input
        )

        # DEBUG: 打印 inputs_embeds 的 checksum
        if len(audios) > 0 and is_not_system_prefill and hasattr(self, "_debug_prefill") and self._debug_prefill:
            embed_sum = inputs_embeds.sum().item()
            embed_shape = inputs_embeds.shape
            print(f"[DEBUG prefill] inputs_embeds sum={embed_sum:.6f}, shape={embed_shape}")

        if self.is_first:
            self.audio_past_key_values = None  # clean audio_past_key_values after first prefill

        round_id = self._next_round_id
        self._pending_round_id = round_id
        chunk_type = "system" if msg["role"] == "system" else ("user" if msg["role"] == "user" else "assistant")
        seq_len = inputs_embeds.shape[1]
        self._enforce_text_window()
        cache_length = self._get_kv_cache_length()

        attention_mask = torch.ones((1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)

        # 2. do prefill
        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )

        self.llm_past_key_values = as_dynamic_cache(outputs["past_key_values"])
        self._register_chunk(
            seq_len,
            chunk_type,
            round_id=round_id,
            input_ids=model_inputs["input_ids"],
            tokenizer=self.processor.tokenizer,
        )
        self._enforce_text_window()
        if self.force_rope_reindex:
            self._force_reindex_all_cache()

        return prompt

    @torch.inference_mode()
    def streaming_generate(
        self,
        session_id,
        tokenizer=None,  # deprecated
        bos_input=None,
        generate_audio=True,
        audio_token_chunk_size=25,  # 25 token/s
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        max_new_tokens=256,
        fn="chunk_generate",
        enable_thinking=False,
        use_tts_template=True,
        do_sample=True,
        enable_speculative_snapshot=False,
        **kwargs,
    ):
        # 保存抢跑快照（在修改任何状态之前）
        # 用于 VAD 抢跑场景：如果抢跑失败，可调用 restore_speculative_snapshot() 恢复
        # enable_speculative_snapshot=True 时启用，False 时跳过（节省少量开销）
        if enable_speculative_snapshot:
            self._speculative_snapshot = self.save_speculative_snapshot()

        # reset buf
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = False
        self.audio_past_key_values = None

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        # reset current turn generated token IDs
        if hasattr(self, "_streaming_generated_token_ids"):
            del self._streaming_generated_token_ids
        # reset full generated text
        if hasattr(self, "_last_streaming_text"):
            del self._last_streaming_text

        cache = self._ensure_dynamic_cache()
        cache_length = self._get_kv_cache_length(cache)
        host_round_id = self._pending_round_id
        logger.info("streaming_generate kv cache length before= %s", cache_length)

        ## 单工情况每调用一次 streaming_generate 需要重新初始化 streaming_processor, 进入下一个 turn
        self.init_streaming_processor()

        # 1) llm generate token and hidden states per chunk=10, 2) tts generate audio token chunk per chunk=25, 3) yield 1 chunk audio token
        def audio_chunk_generator(
            bos_input,
            tokenizer,
            generate_audio,
            tts_sampling_params,
            max_new_tokens,
            do_sample,
            **kwargs,
        ):
            generate_chunk_size = 10

            if bos_input is None:
                bos_input = "".join(
                    [
                        "<|im_end|>\n<|im_start|>assistant\n",
                        "" if enable_thinking else self.think_str.replace("\\n", "\n"),
                        "<|tts_bos|>" if use_tts_template else "",
                    ]
                )

            bos_input_ids = tokenizer.encode(bos_input)
            bos_input_ids = torch.tensor(bos_input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            bos_input_embeds = self.llm.get_input_embeddings()(bos_input_ids)

            generation_inputs_embeds = bos_input_embeds
            generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)

            num_chunks_decode = (max_new_tokens + generate_chunk_size - 1) // generate_chunk_size

            conditions = []

            # generate chunk by chunk, each chunk has 10 tokens, each chunk takes last hidden states, and pass tokens to tts
            llm_streaming_generator = ChunkPrefillChunkGenerate(
                model=self.llm,
                tokenizer=tokenizer,
                terminators=["<|tts_eos|>", "<|im_end|>", "</s>"],
            )

            if generate_audio:
                logits_warpers, logits_processors = gen_logits(
                    num_code=self.tts.config.num_audio_tokens,
                    repetition_penalty=tts_sampling_params.repetition_penalty,
                    top_p=tts_sampling_params.top_p,
                    top_k=tts_sampling_params.top_k,
                )

                tts_streaming_generator = TTSStreamingGenerator(
                    model=self.tts,
                    temperature=tts_sampling_params.temperature,
                    eos_token=torch.tensor(
                        [self.tts.config.num_audio_tokens - 1],
                        dtype=torch.long,
                        device=self.tts.device,
                    ),
                    chunk_size=audio_token_chunk_size,  # s3tokenizer 1s = 25token
                    tts_last_turn_tokens=self.tts_last_turn_tokens,
                    logits_processors=logits_processors,
                    logits_warpers=logits_warpers,
                )

            # LLM chunk generate outer loop
            for chunk_idx in range(num_chunks_decode):
                is_first_generate_chunk = chunk_idx == 0

                output = llm_streaming_generator.chunk_generate(
                    inputs_embeds=generation_inputs_embeds,
                    past_key_values=self.llm_past_key_values,
                    is_first_generate_chunk=is_first_generate_chunk,
                    return_hidden_states=True,
                    chunk_size=generate_chunk_size + 1 * is_first_generate_chunk,
                    do_sample=do_sample,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.8),
                    top_k=kwargs.get("top_k", 100),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.02),
                    length_penalty=kwargs.get("length_penalty", 1.0),
                    all_input_ids=generated_ids,
                    suppress_forbidden_tokens=generate_audio,
                )

                if output.chunk_token_ids is None:
                    break

                if is_first_generate_chunk:
                    if generate_audio:
                        spk_emb = torch.empty(
                            (bos_input_embeds.shape[0], 0, bos_input_embeds.shape[2]),
                            dtype=bos_input_embeds.dtype,
                            device=bos_input_embeds.device,
                        )
                        tts_streaming_generator.spk_emb = spk_emb

                    if output.finished:
                        yield_chunk_token_ids = output.chunk_token_ids
                    else:
                        # the first chunk generated chunk_size + 1 tokens, we only take the first chunk_size tokens,
                        # the last token is not prefilled, and last hidden states is not obtained
                        yield_chunk_token_ids = output.chunk_token_ids[:, :-1]

                elif output.finished:
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], output.chunk_token_ids], dim=1)
                else:
                    # in the chunk that is not the first chunk, we need to add the token at the end of the previous chunk,
                    # it is not prefilled into the model to get last hidden states
                    # similarly, the last generated token of subsequent chunks is not prefilled, and last hidden states is not obtained,
                    # so it is not passed out
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], output.chunk_token_ids[:, :-1]], dim=1)

                if not generate_audio:
                    chunk_generated_text = tokenizer.decode(yield_chunk_token_ids[0])
                    yield yield_chunk_token_ids, output.finished
                else:
                    # TTS inner loop
                    # dense connection here is hardcoded to use text-hidden merged as condition
                    llm_embeds = self.tts.emb_text(yield_chunk_token_ids)
                    hidden_embeds = output.last_hidden_states
                    hidden_embeds = self.tts.projector_semantic(hidden_embeds)
                    if self.tts.config.normalize_projected_hidden:  # default should be opened
                        hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

                    tts_embeds = llm_embeds + hidden_embeds
                    conditions.append(tts_embeds)

                    # Store token IDs instead of decoded text to avoid UTF-8 multi-byte character truncation
                    if not hasattr(self, "_streaming_generated_token_ids"):
                        self._streaming_generated_token_ids = []
                    self._streaming_generated_token_ids.extend(yield_chunk_token_ids[0].tolist())

                    # there is buffer generated, each time exactly returns 25 audio tokens,
                    # the last audio chunk returns audio tokens of variable length, length [0, 25]
                    tts_generator = tts_streaming_generator.generate_with_buffer(
                        condition=tts_embeds, text_finished=output.finished
                    )

                    for audio_token_chunk, is_last_audio_chunk in tts_generator:
                        yield audio_token_chunk, is_last_audio_chunk

                generated_ids = torch.cat([generated_ids, output.chunk_token_ids], dim=1)
                generation_inputs_embeds = output.current_inputs_embeds
                self.llm_past_key_values = output.past_key_values

                if output.finished:
                    if generate_audio:
                        self.tts_last_turn_tokens = tts_streaming_generator.tts_last_turn_tokens
                    break

            # IMPORTANT: Flush remaining TTS buffer when LLM generation ends
            # This handles BOTH cases:
            # 1. LLM finished with terminator (output.finished=True) - buffer may still have tokens
            # 2. LLM hit max chunks limit (output.finished=False) - buffer definitely has tokens
            if generate_audio:
                if len(tts_streaming_generator._token_buffer) > 0:
                    batch = torch.cat(tts_streaming_generator._token_buffer, dim=1)
                    yield batch, True
                    tts_streaming_generator._token_buffer = []

            if generate_audio:
                if hasattr(self, "_streaming_generated_token_ids"):
                    try:
                        self._last_streaming_text = tokenizer.decode(self._streaming_generated_token_ids)
                        assistant_input_ids = self._encode_text(tokenizer=tokenizer, text=self._last_streaming_text)
                        self._finalize_round(
                            round_id=host_round_id, cache_before=cache_length, assistant_input_ids=assistant_input_ids
                        )
                    except Exception:
                        self._last_streaming_text = None
                else:
                    self._last_streaming_text = None

                yield None, None
            else:
                return

        # iter for generating text chunk and audio chunk
        audio_chunk_generator_iter = audio_chunk_generator(
            bos_input=bos_input,
            tokenizer=self.processor.tokenizer,
            generate_audio=generate_audio,
            tts_sampling_params=tts_sampling_params,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        if generate_audio:
            if self.tts.config.audio_tokenizer_type == "s3tokenizer_step_audio":
                self.tts.audio_tokenizer.stream_cache = torch_clone_recursive(self.token2wav_cache["flow_cache_base"])
                self.tts.audio_tokenizer.hift_cache_dict = torch_clone_recursive(
                    self.token2wav_cache["hift_cache_base"]
                )

                # pre-insert 3-5 prefix 4218 silence tokens, each token corresponds to 0.04s,
                # adding 5 tokens means introducing 0.2s of silence
                buffer = [4218] * 3
                pre_lookahead = 3
                CHUNK_SIZE = 25
                chunk_idx = 0
                prev_text_len = 0  # track text position for streaming text output
                for audio_token_chunk, is_last_audio_chunk in audio_chunk_generator_iter:
                    if audio_token_chunk is None:
                        break

                    buffer += audio_token_chunk.reshape(-1).tolist()

                    if len(buffer) >= CHUNK_SIZE + pre_lookahead:
                        waveform_chunk = self.tts.audio_tokenizer.stream(
                            buffer[: CHUNK_SIZE + pre_lookahead],
                            prompt_wav=None,
                            last_chunk=is_last_audio_chunk,
                            return_waveform=True,
                        )

                        waveform_chunk = torch.from_numpy(waveform_chunk)

                        # get new text chunk corresponding to this waveform
                        # Decode from accumulated token IDs to avoid UTF-8 multi-byte truncation
                        new_text = ""
                        if hasattr(self, "_streaming_generated_token_ids"):
                            current_text = self.processor.tokenizer.decode(self._streaming_generated_token_ids)
                            # Filter out trailing replacement characters (incomplete UTF-8 sequences)
                            safe_end = len(current_text)
                            while safe_end > 0 and current_text[safe_end - 1] == "\ufffd":
                                safe_end -= 1
                            safe_text = current_text[:safe_end]
                            new_text = safe_text[prev_text_len:]
                            prev_text_len = len(safe_text)

                        yield waveform_chunk, new_text

                        buffer = buffer[CHUNK_SIZE:]
                        chunk_idx += 1

                # flush rest
                if len(buffer) > 0:
                    waveform_chunk = self.tts.audio_tokenizer.stream(
                        buffer,
                        prompt_wav=None,
                        last_chunk=True,
                        return_waveform=True,
                    )

                    waveform_chunk = torch.from_numpy(waveform_chunk)

                    # get remaining new text for the final chunk
                    # Final chunk: decode all remaining text without filtering
                    new_text = ""
                    if hasattr(self, "_streaming_generated_token_ids"):
                        current_text = self.processor.tokenizer.decode(self._streaming_generated_token_ids)
                        new_text = current_text[prev_text_len:]
                        prev_text_len = len(current_text)

                    yield waveform_chunk, new_text

                # maybe the buffer is empty, and text is not empty, should we flush text without wave?
            else:
                raise NotImplementedError(f"not supported audio tokenizer: {self.tts.config.audio_tokenizer_type}")
        else:
            # For text-only generation, decode tokens and handle partial multi-byte characters
            yield from streaming_token_decoder(
                audio_chunk_generator_iter,
                self.processor.tokenizer,
                skip_special_tokens=False,
            )

    # ==================== Duplex 透传方法 ====================
    # 以下方法透传到 self.duplex，减少调用层级
    # 外部可以直接调用 model.duplex_prepare() 而不是 model.duplex.prepare()
    
    def duplex_prepare(
        self,
        prefix_system_prompt: Optional[str] = None,
        suffix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        context_previous_marker: str = "\n\nprevious: ",
    ):
        """准备双工会话（透传到 self.duplex.prepare）
        
        Args:
            prefix_system_prompt: system prompt 前缀
            suffix_system_prompt: system prompt 后缀
            ref_audio: 参考音频（16kHz numpy array）
            prompt_wav_path: TTS prompt 音频路径
            context_previous_marker: 上下文历史标记
            
        Returns:
            完整的 system prompt 字符串
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        return self.duplex.prepare(
            prefix_system_prompt=prefix_system_prompt,
            suffix_system_prompt=suffix_system_prompt,
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
            context_previous_marker=context_previous_marker,
        )
    
    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        max_slice_nums: int = 1,
    ):
        """预填充用户输入（透传到 self.duplex.streaming_prefill）
        
        Args:
            audio_waveform: 音频波形（16kHz numpy array）
            frame_list: 视频帧列表
            max_slice_nums: HD 图像切片数
            
        Returns:
            预填充结果 dict
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        return self.duplex.streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )
    
    def duplex_generate(
        self,
        decode_mode: str = "greedy",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        listen_prob_scale: Optional[float] = None,
        listen_top_k: int = 5,
        text_repetition_penalty: Optional[float] = None,
        text_repetition_window_size: Optional[int] = None,
        length_penalty: float = 1.1,
        force_listen_override: bool = False,
    ):
        """生成响应（透传到 self.duplex.streaming_generate）
        
        Args:
            decode_mode: 解码模式 ("greedy" 或 "sample")
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P 采样
            listen_prob_scale: Listen 概率缩放
            listen_top_k: Listen 判断的 top-k
            text_repetition_penalty: 文本重复惩罚
            text_repetition_window_size: 重复惩罚窗口大小
            length_penalty: 长度惩罚系数，>1.0 抑制 turn_eos 使输出更长
            force_listen_override: 前端 Force Listen 开关，强制本次生成为 listen
            
        Returns:
            生成结果 dict
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        return self.duplex.streaming_generate(
            decode_mode=decode_mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            listen_prob_scale=listen_prob_scale,
            listen_top_k=listen_top_k,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            length_penalty=length_penalty,
            force_listen_override=force_listen_override,
        )
    
    def duplex_finalize(self):
        """完成 streaming_generate 的延迟操作（透传到 self.duplex.finalize_unit）
        
        必须在 duplex_generate 之后、下一次 duplex_prefill 之前调用。
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        self.duplex.finalize_unit()

    def duplex_set_break(self):
        """设置打断信号（透传到 self.duplex.set_break_event）"""
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        self.duplex.set_break_event()
    
    def duplex_clear_break(self):
        """清除打断信号（透传到 self.duplex.clear_break_event）"""
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        self.duplex.clear_break_event()
    
    def duplex_stop(self):
        """停止当前会话（透传到 self.duplex.set_session_stop）"""
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        self.duplex.set_session_stop()
    
    def duplex_is_break_set(self) -> bool:
        """检查是否设置了打断（透传到 self.duplex.is_break_set）"""
        if self.duplex is None:
            return False
        return self.duplex.is_break_set()
    
    def duplex_is_stopped(self) -> bool:
        """检查会话是否已停止（透传到 self.duplex.is_session_stop_set）"""
        if self.duplex is None:
            return False
        return self.duplex.is_session_stop_set()

    # ==================== FC Duplex 透传方法 ====================

    def _require_fc_duplex(self) -> "FcDuplexCapability":
        if self.fc_duplex is None:
            raise RuntimeError("FC Duplex 未初始化，请先调用 init_unified()")
        return self.fc_duplex

    def fc_duplex_prepare(
        self,
        system_prompt: str,
        tools=None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        generate_audio: Optional[bool] = None,
    ) -> dict:
        return self._require_fc_duplex().prepare(
            system_prompt=system_prompt,
            tools=tools,
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
            generate_audio=generate_audio,
        )

    def fc_duplex_streaming_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        tool_responses=None,
        sample_rate: int = 16000,
        max_slice_nums: int = 1,
    ) -> dict:
        return self._require_fc_duplex().streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            tool_responses=tool_responses,
            sample_rate=sample_rate,
            max_slice_nums=max_slice_nums,
        )

    def fc_duplex_streaming_spoken_generate(
        self,
        max_tokens: int = 24,
        decode_mode: str = "greedy",
    ) -> dict:
        return self._require_fc_duplex().streaming_spoken_generate(
            max_tokens=max_tokens,
            decode_mode=decode_mode,
        )

    def fc_duplex_streaming_non_spoken_generate(
        self,
        decode_mode: str = "greedy",
        max_tokens: int = 1,
        close_reason: Optional[str] = None,
    ) -> dict:
        return self._require_fc_duplex().streaming_non_spoken_generate(
            decode_mode=decode_mode,
            max_tokens=max_tokens,
            close_reason=close_reason,
        )

    def fc_duplex_finalize_unit(self) -> dict:
        return self._require_fc_duplex().finalize_unit()

    def fc_duplex_decode_output_ids(self, output_ids=None, tools=None) -> dict:
        return self._require_fc_duplex().decode_output_ids(output_ids=output_ids, tools=tools)

    def fc_duplex_trace_snapshot(self, session_id=None, reason=None) -> dict:
        return self._require_fc_duplex().trace_snapshot(session_id=session_id, reason=reason)

    def fc_duplex_dump_trace(self, path, session_id=None, reason=None) -> dict:
        return self._require_fc_duplex().dump_trace(path, session_id=session_id, reason=reason)

    def fc_duplex_cleanup(self) -> None:
        self._require_fc_duplex().cleanup()
    
    def duplex_chat(
        self,
        user_audio: np.ndarray,
        system_prompt: str = "You are a helpful assistant.",
        ref_audio: Optional[np.ndarray] = None,
        ref_audio_path: Optional[str] = None,
        image_list: Optional[List] = None,
        chunk_ms: int = 1000,
        sample_rate: int = 16000,
        generate_audio: bool = True,
        decode_mode: str = "greedy",
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.8,
        force_listen_count: int = 3,
    ) -> dict:
        """双工离线推理（便捷方法）
        
        对完整音频进行离线双工对话，一站式处理。
        
        适用场景：
        - 离线批量处理音频文件
        - 单元测试
        - 演示场景
        
        注意：这不是实时双工会话，而是对完整音频的离线处理。
        实时双工请使用 duplex_prepare/duplex_prefill/duplex_generate 原语。
        
        Args:
            user_audio: 用户音频波形（16kHz numpy array）
            system_prompt: 系统提示文本
            ref_audio: 参考音频波形（16kHz numpy array，用于 TTS）
            ref_audio_path: 参考音频路径（用于 TTS，与 ref_audio 二选一）
            image_list: 图像列表（视频双工，每个 chunk 一张 PIL Image）
            chunk_ms: 每个 chunk 的时长（毫秒）
            sample_rate: 音频采样率
            generate_audio: 是否生成音频
            decode_mode: 解码模式 ("greedy" 或 "sampling")
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P 采样
            force_listen_count: 强制 listen 的 chunk 数
            
        Returns:
            dict: {
                "success": bool,
                "full_text": str,
                "chunks": List[dict],
                "audio_chunks": List[np.ndarray],
                "error": Optional[str],
            }
        
        示例：
            >>> result = model.duplex_chat(
            ...     user_audio=audio_16k,
            ...     system_prompt="你是一个友好的助手。",
            ...     ref_audio_path="/path/to/ref.wav",
            ... )
            >>> print(result["full_text"])
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        
        chunks = []
        full_text = ""
        audio_chunks = []
        
        try:
            # 准备会话
            self.duplex_prepare(
                prefix_system_prompt=system_prompt,
                ref_audio=ref_audio,
                prompt_wav_path=ref_audio_path,
            )
            
            # 配置 force_listen
            if hasattr(self.duplex, '_force_listen_count'):
                self.duplex._force_listen_count = force_listen_count
            
            # 分块处理音频
            chunk_samples = sample_rate * chunk_ms // 1000
            num_chunks = (len(user_audio) + chunk_samples - 1) // chunk_samples
            
            for i in range(num_chunks):
                # 获取音频块
                start_idx = i * chunk_samples
                end_idx = min(start_idx + chunk_samples, len(user_audio))
                audio_chunk = user_audio[start_idx:end_idx]
                
                # 补零到完整块
                if len(audio_chunk) < chunk_samples:
                    audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))
                
                # 获取图像帧（如果有）
                frame_list = None
                if image_list and i < len(image_list):
                    frame_list = [image_list[i]]
                
                # 预填充
                self.duplex_prefill(
                    audio_waveform=audio_chunk,
                    frame_list=frame_list,
                )
                
                # 生成
                result = self.duplex_generate(
                    decode_mode=decode_mode,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                
                # 记录结果
                chunk_result = {
                    "chunk_idx": i,
                    "is_listen": result.get("is_listen", True),
                    "text": result.get("text", ""),
                    "has_audio": result.get("audio") is not None,
                    "end_of_turn": result.get("end_of_turn", False),
                }
                chunks.append(chunk_result)
                
                if not chunk_result["is_listen"]:
                    full_text += chunk_result["text"]
                    if result.get("audio") is not None:
                        audio_chunks.append(result["audio"])
                
                if chunk_result["end_of_turn"]:
                    break
            
            # 停止会话
            self.duplex_stop()
            
            return {
                "success": True,
                "full_text": full_text,
                "chunks": chunks,
                "audio_chunks": audio_chunks,
                "error": None,
            }
            
        except Exception as e:
            logger.error(f"duplex_chat 失败: {e}")
            return {
                "success": False,
                "full_text": full_text,
                "chunks": chunks,
                "audio_chunks": audio_chunks,
                "error": str(e),
            }


class DuplexCapability(BaseMiniCPMODuplex):
    """双工能力组件 - 封装双工对话的全部逻辑
    
    继承 vendored MiniCPMODuplex，只保留 demo runtime 的增量逻辑。
    
    使用方式（推荐，使用透传方法）：
        model = MiniCPMO.from_pretrained(...)
        model.init_unified(...)
        
        model.duplex_prepare(...)
        model.duplex_prefill(...)
        result = model.duplex_generate()
        model.duplex_set_break()
    
    使用方式（直接访问）：
        model.duplex.prepare(...)
        model.duplex.streaming_prefill(...)
        result = model.duplex.streaming_generate(...)
    """

    _default_duplex_params = dict(BaseMiniCPMODuplex._default_duplex_params)
    _default_duplex_params.update(
        {
            "top_k": 20,
            "n_timesteps": 5,
            "basic_window_high_tokens": 4000,
            "basic_window_low_tokens": 3500,
        }
    )

    @classmethod
    def from_existing_model(
        cls,
        model: "MiniCPMO",
        device: Optional[str] = None,
        **kwargs,
    ) -> "DuplexCapability":
        instance = BaseMiniCPMODuplex.from_existing_model.__func__(
            cls,
            model,
            device=device,
            **kwargs,
        )
        logger.info("[DuplexCapability] 初始化完成")
        return instance

    def _reset_streaming_state(self):
        super()._reset_streaming_state()
        self._pending_finalize = None
        self._last_chunk_had_tts_pad = False

    def prepare(
        self,
        prefix_system_prompt: Optional[str] = None,
        suffix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        context_previous_marker: str = "\n\nprevious: ",
    ):
        self.clear_break_event()
        self.clear_session_stop()

        self._reset_streaming_state()
        self.decoder.reset()

        self.model.init_streaming_processor()

        if prompt_wav_path is not None and prompt_wav_path and self.generate_audio:
            self._init_token2wav_cache(prompt_wav_path)
            self._reset_token2wav_for_new_turn()

        # Prefill system prompt prefix (batch)
        if prefix_system_prompt:
            tokens = self.tokenizer.encode(prefix_system_prompt, add_special_tokens=False)
            if tokens:
                embeds = self.decoder.embed_tokens(tokens)
                self.decoder.feed(embeds)

        # Prefill reference audio
        if ref_audio is not None:
            data = self.processor.process_audio([ref_audio])
            embeds_nested = self.model.get_audio_embedding(data, chunk_length=self.model.config.audio_chunk_length)
            embeds = torch.cat([t for g in embeds_nested for t in g], dim=0) if embeds_nested else None
            if embeds is not None:
                self.decoder.feed(embeds)

        # 注册 system prompt 保护长度（滑窗时保护这部分不被移除）
        if prefix_system_prompt or suffix_system_prompt or ref_audio is not None:
            logger.info("[Duplex] prepare: registering system prompt protection")
            if self.decoder._window_config.sliding_window_mode == "context":
                # Context 保留模式：
                # 初始化时布局: [prefix] [suffix] [units...]
                # 首次滑窗后布局: [prefix] [context_previous_marker + content] [suffix] [units...]
                # 此时先注册 prefix 长度，再 feed suffix
                self._prefix_system_prompt = prefix_system_prompt
                self._suffix_system_prompt = suffix_system_prompt
                self._ref_audio = ref_audio

                # 获取 suffix token ids
                suffix_token_ids = []
                if suffix_system_prompt:
                    suffix_token_ids = self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)

                # 注册（此时 cache 只有 prefix，还没有 suffix，也没有 previous）
                self.decoder.register_system_prompt_with_context(
                    suffix_token_ids=suffix_token_ids,
                    context_previous_marker=context_previous_marker,  # 首次滑窗时动态添加
                )

                # 现在 feed suffix (batch)
                if suffix_token_ids:
                    suffix_embeds = self.decoder.embed_tokens(suffix_token_ids)
                    self.decoder.feed(suffix_embeds)

                logger.info(
                    "[Duplex] prepare: context-preserve mode, prefix=%d, suffix=%d tokens, marker='%s'",
                    self.decoder._preserve_prefix_length,
                    len(suffix_token_ids),
                    context_previous_marker.replace("\n", "\\n"),
                )
            else:
                # 非 context 保留模式：先 feed suffix，再注册总长度
                if suffix_system_prompt:
                    tokens = self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)
                    if tokens:
                        suffix_embeds = self.decoder.embed_tokens(tokens)
                        self.decoder.feed(suffix_embeds)
                self.decoder.register_system_prompt()

        if prefix_system_prompt or suffix_system_prompt:
            if ref_audio is not None:
                full_prompt = (prefix_system_prompt or "") + "[音频嵌入]" + (suffix_system_prompt or "")
            else:
                full_prompt = (prefix_system_prompt or "") + (suffix_system_prompt or "")

            return full_prompt

        return ""

    @torch.no_grad()
    def streaming_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[list] = None,
        max_slice_nums: Union[int, List[int]] = 1,
        batch_vision_feed: bool = False,
    ):
        """Streaming prefill - called once per second, processing audio/video data

        Args:
            audio_waveform: audio waveform data
            frame_list: image frame list
            max_slice_nums: maximum number of slices for HD image encoding (default 1, no slicing)
                           Can be an int (same for all images) or a list matching frame_list length
            batch_vision_feed: if True, batch all vision embeddings into a single feed call for better performance.
                              if False (default), feed each embedding individually (original behavior).

        Process:
            0. determine mode based on input: AUDIO / VISION / OMNI
            1. feed <unit> token
            2. get and feed image embed (if frame_list) - return pending logits in VISION MODE
            3. get and feed audio embed (if audio_waveform) - return pending logits in AUDIO/OMNI MODE

        Returns:
            dict with keys:
                - success: bool
                - cost_vision_process: float (image processing time)
                - cost_vision_embed: float (vision embedding time)
                - cost_vision_feed: float (vision feed time)
                - cost_audio_process: float (audio processing time)
                - cost_audio_embed: float (audio embedding time)
                - cost_audio_feed: float (audio feed time)
                - cost_all: float (total time)
        """
        # Fail-fast: 上一轮 finalize 未完成就进入下一轮 prefill
        if self.needs_finalize:
            raise RuntimeError(
                "streaming_prefill called before finalize_unit()! "
                "必须在 streaming_generate 之后调用 finalize_unit() 再进入下一轮 prefill。"
            )

        start_time = time.time()
        cost_vision_process = 0.0
        cost_vision_embed = 0.0
        cost_vision_feed = 0.0
        cost_audio_process = 0.0
        cost_audio_embed = 0.0
        cost_audio_feed = 0.0

        def _make_result(success, reasons=""):
            reason = reasons
            if isinstance(reasons, list):
                reason = "; ".join(reasons)

            return {
                "success": success,
                "reason": reason,
                "cost_vision_process": cost_vision_process,
                "cost_vision_embed": cost_vision_embed,
                "cost_vision_feed": cost_vision_feed,
                "cost_audio_process": cost_audio_process,
                "cost_audio_embed": cost_audio_embed,
                "cost_audio_feed": cost_audio_feed,
                "cost_all": time.time() - start_time,
            }

        if self.is_session_stop_set():
            return _make_result(False)
        # prefill 只检查 session_stop，不检查 break_event
        # Force Listen 通过 per-chunk force_listen_override 参数在 generate() 中处理

        has_frames = frame_list is not None and len(frame_list) > 0
        has_audio = audio_waveform is not None and len(audio_waveform) > 0

        if has_frames and has_audio:
            mode = "OMNI"
        elif has_frames:
            mode = "VISION"
        elif has_audio:
            mode = "AUDIO"
        else:
            return _make_result(False)

        self.pending_logits = None

        # 滑窗：记录 unit 开始位置
        logger.info(
            "[Duplex] streaming_prefill: mode=%s, has_frames=%s, has_audio=%s, starting unit",
            mode,
            has_frames,
            has_audio,
        )
        self.decoder.register_unit_start()

        # Schema tracking: 开始新的 unit，记录 prefill tokens
        self._current_unit_prefill_tokens = []

        # Step 1: Feed <unit> token
        self.decoder.feed(self.decoder.embed_token(self.unit_token_id))
        self._current_unit_prefill_tokens.append(self.unit_token_id)

        # Step 2: process image
        if has_frames:
            t0 = time.time()

            # Normalize max_slice_nums to a list matching frame_list length
            if isinstance(max_slice_nums, int):
                max_slice_nums_list = [max_slice_nums] * len(frame_list)
            else:
                max_slice_nums_list = list(max_slice_nums)
                if len(max_slice_nums_list) != len(frame_list):
                    raise ValueError(
                        f"max_slice_nums list length ({len(max_slice_nums_list)}) "
                        f"must match frame_list length ({len(frame_list)})"
                    )

            # Check if all max_slice_nums are the same (can use batch processing)
            all_same = len(set(max_slice_nums_list)) == 1

            if all_same:
                # All images use the same max_slice_nums, use batch processing
                processed_frames = self.processor.process_image(frame_list, max_slice_nums=max_slice_nums_list[0])
                if self.device:
                    processed_frames = processed_frames.to(self.device)
            else:
                # Different max_slice_nums per image, process individually and merge
                all_pixel_values = []
                all_tgt_sizes = []
                for frame, max_slices in zip(frame_list, max_slice_nums_list):
                    pf = self.processor.process_image([frame], max_slice_nums=max_slices)
                    if self.device:
                        pf = pf.to(self.device)
                    # pf["pixel_values"][0] is the list of slices for this image
                    all_pixel_values.extend(pf["pixel_values"][0])
                    # pf["tgt_sizes"][0] is the array of target sizes for this image's slices
                    if hasattr(pf["tgt_sizes"][0], "tolist"):
                        all_tgt_sizes.extend(pf["tgt_sizes"][0].tolist())
                    else:
                        all_tgt_sizes.extend(list(pf["tgt_sizes"][0]))

                # Reconstruct processed_frames with merged data
                processed_frames = {
                    "pixel_values": [all_pixel_values],
                    "tgt_sizes": [torch.tensor(all_tgt_sizes) if all_tgt_sizes else []],
                }

            cost_vision_process = time.time() - t0

            t0 = time.time()
            # Get vision embeddings for all images (each may have multiple slices)
            # vision_hidden_states is a list, one entry per input image
            # Each entry contains embeddings for [source_image, slice_1, slice_2, ...]
            vision_hidden_states = self.model.get_vision_embedding(processed_frames)
            cost_vision_embed = time.time() - t0

            if vision_hidden_states is not None and len(vision_hidden_states) > 0:
                t0 = time.time()

                # vision_hidden_states[0] contains ALL slices from ALL images (flattened)
                # Shape: [total_slices, 64, D] where total_slices = sum of slices across all images
                # We need to know how many slices each image has to correctly group them

                # Calculate slice counts for each image using get_sliced_grid (lightweight, no actual slicing)
                slice_counts = []  # e.g., [5, 9] means img1 has 5 slices (1 source + 4 HD), img2 has 9
                for frame_idx, frame in enumerate(frame_list):
                    max_slices = max_slice_nums_list[frame_idx]
                    if hasattr(frame, "size"):
                        # get_sliced_grid returns [M, N] grid or None if no slicing needed
                        # Total images = 1 (source) + M * N (HD slices)
                        grid = self.processor.image_processor.get_sliced_grid(
                            frame.size, max_slices, nerver_split=False
                        )
                        if grid is not None:
                            slice_counts.append(1 + grid[0] * grid[1])  # 1 source + M*N slices
                        else:
                            slice_counts.append(1)  # No slicing, only source image
                    else:
                        slice_counts.append(1)  # Default: single image, no slicing

                # Get the flattened embeddings tensor
                # vision_hidden_states is a list with one element (the batch)
                # vision_hidden_states[0] shape: [total_slices, 64, D]
                all_embeds = vision_hidden_states[0]

                # Collect all feed operations first, then execute
                # This allows us to identify the last token for VISION mode logits
                feed_operations = []  # List of (embed, is_last_for_vision_mode, token_id_or_none)

                embed_idx = 0  # Current index in all_embeds
                for img_idx, num_slices in enumerate(slice_counts):
                    if num_slices == 0:
                        continue

                    # First embedding is always the source image (downsampled overview)
                    # Feed <image> token
                    feed_operations.append(
                        (self.decoder.embed_token(self.image_start_token_id), False, self.image_start_token_id)
                    )
                    # Feed source image embedding (shape: [64, D]) - use None to indicate embedding
                    feed_operations.append((all_embeds[embed_idx], False, None))
                    # Feed </image> token
                    feed_operations.append(
                        (self.decoder.embed_token(self.image_end_token_id), False, self.image_end_token_id)
                    )
                    embed_idx += 1

                    # Remaining embeddings are HD slices (if num_slices > 1)
                    if num_slices > 1:
                        for slice_i in range(1, num_slices):
                            # Feed <slice> token
                            feed_operations.append(
                                (self.decoder.embed_token(self.slice_start_token_id), False, self.slice_start_token_id)
                            )
                            # Feed slice embedding (shape: [64, D])
                            feed_operations.append((all_embeds[embed_idx], False, None))
                            # Feed </slice> token
                            feed_operations.append(
                                (self.decoder.embed_token(self.slice_end_token_id), False, self.slice_end_token_id)
                            )
                            embed_idx += 1

                # Mark the last operation for VISION mode logits
                if feed_operations:
                    feed_operations[-1] = (feed_operations[-1][0], True, feed_operations[-1][2])

                # Execute feed operations
                if batch_vision_feed and feed_operations:
                    # Batch mode: concatenate all embeddings and feed at once
                    # This reduces LLM forward passes from N to 1
                    #
                    # NOTE: Batch mode may have slight numerical differences compared to for-loop mode
                    # due to floating-point precision in attention computation. This is expected behavior
                    # for causal attention with incremental vs batch computation.

                    all_embeds_list = []
                    for embed, is_last, token_id in feed_operations:
                        # Ensure all embeddings have shape [L, H]
                        if embed.dim() == 1:
                            embed = embed.unsqueeze(0)
                        all_embeds_list.append(embed)

                    # Concatenate all embeddings
                    # torch.cat requires consistent dtype; embeddings should already be same dtype
                    all_embeds_to_feed = torch.cat(all_embeds_list, dim=0)  # [total_L, H]

                    # Debug: log embedding info for first unit
                    if self.audio_chunk_idx == 0:
                        logger.info(
                            "[Batch Vision Feed] total_L=%d, dtype=%s, device=%s, sum=%.4f",
                            all_embeds_to_feed.shape[0],
                            all_embeds_to_feed.dtype,
                            all_embeds_to_feed.device,
                            all_embeds_to_feed.sum().item(),
                        )

                    if mode == "VISION":
                        # VISION mode needs logits from the last token
                        self.pending_logits, _ = self.decoder.feed(all_embeds_to_feed, return_logits=True)
                    else:
                        # OMNI mode: just feed, wait for audio to get logits
                        self.decoder.feed(all_embeds_to_feed)

                    # Schema tracking: record all token IDs and embedding markers
                    for embed, is_last, token_id in feed_operations:
                        if token_id is not None:
                            self._current_unit_prefill_tokens.append(token_id)
                        else:
                            embed_dim = embed.shape[0] if len(embed.shape) > 1 else 1
                            self._current_unit_prefill_tokens.append(("img", embed_dim))
                else:
                    # Original mode: feed each embedding individually

                    # Debug: log embedding info for first unit
                    if self.audio_chunk_idx == 0:
                        total_len = sum(e.shape[0] if len(e.shape) > 1 else 1 for e, _, _ in feed_operations)
                        embed_sum = sum(e.sum().item() for e, _, _ in feed_operations)
                        logger.info(
                            "[For-loop Vision Feed] total_L=%d, sum=%.4f",
                            total_len,
                            embed_sum,
                        )

                    for embed, is_last, token_id in feed_operations:
                        if mode == "VISION" and is_last:
                            # Get logits from the last token
                            self.pending_logits, _ = self.decoder.feed(embed, return_logits=True)
                        else:
                            self.decoder.feed(embed)
                        # Schema tracking: 记录 token ID 或 embedding 标记
                        if token_id is not None:
                            self._current_unit_prefill_tokens.append(token_id)
                        else:
                            # 用元组标记 image embedding: ("img", dim)
                            embed_dim = embed.shape[0] if len(embed.shape) > 1 else 1
                            self._current_unit_prefill_tokens.append(("img", embed_dim))
                # For OMNI MODE, no pending logits needed here (wait for audio)

                cost_vision_feed = time.time() - t0

        # Step 3: process audio (if any)
        if has_audio:
            # accumulate audio to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_waveform])

            # calculate required audio length
            if self.audio_chunk_idx == 0:
                required_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
                if len(self.audio_buffer) < required_samples:
                    padding_samples = required_samples - len(self.audio_buffer)
                    padding = np.zeros(padding_samples, dtype=np.float32)
                    self.audio_buffer = np.concatenate([padding, self.audio_buffer])
            else:
                required_samples = int(self.CHUNK_MS * self.SAMPLE_RATE / 1000)

            need_samples = self.processor.get_streaming_chunk_size()
            if len(self.audio_buffer) < need_samples:
                return _make_result(False, f"音频不足: 需要 {need_samples} 样本, 只有 {len(self.audio_buffer)}")

            audio_chunk = self.audio_buffer[:need_samples]

            t0 = time.time()
            batch_feature = self.processor.process_audio_streaming(
                audio_chunk,
                reset=False,
                return_batch_feature=True,
            )

            if batch_feature is None or batch_feature.audio_features.shape[-1] == 0:
                return _make_result(False, "流式音频处理返回空")

            # metadata
            batch_feature.chunk_idx = self.audio_chunk_idx
            batch_feature.use_extra_context = True
            batch_feature.prefix_extra_frames = 0 if self.audio_chunk_idx == 0 else 2
            batch_feature.suffix_extra_frames = 2

            batch_feature = batch_feature.to(self.device)
            cost_audio_process = time.time() - t0

            t0 = time.time()
            embeds_nested = self.model.get_audio_embedding_streaming(
                batch_feature,
                use_extra_context=batch_feature.use_extra_context,
                prefix_extra_frames=batch_feature.prefix_extra_frames,
                suffix_extra_frames=batch_feature.suffix_extra_frames,
            )
            audio_embeds = torch.cat([t for g in embeds_nested for t in g], dim=0)
            cost_audio_embed = time.time() - t0

            t0 = time.time()
            self.pending_logits, _ = self.decoder.feed(audio_embeds, return_logits=True)
            cost_audio_feed = time.time() - t0

            # Schema tracking: 用元组标记 audio embedding: ("audio", dim)
            embed_dim = audio_embeds.shape[0] if len(audio_embeds.shape) > 1 else 1
            self._current_unit_prefill_tokens.append(("audio", embed_dim))

            if self.audio_chunk_idx == 0:
                cfg = self.processor._streaming_mel_processor.get_config()
                consumed_ms = int(cfg.get("effective_first_chunk_ms", self.FIRST_CHUNK_MS))
                consumed_samples = int(consumed_ms * self.SAMPLE_RATE / 1000)
            else:
                consumed_samples = int(self.CHUNK_MS * self.SAMPLE_RATE / 1000)

            self.audio_buffer = self.audio_buffer[consumed_samples:]

            self.audio_chunk_idx += 1

        self.current_mode = mode

        # for VISION mode, need to manually increase chunk count (AUDIO and OMNI modes already increased in _process_audio_buffer)
        if mode == "VISION":
            self.audio_chunk_idx += 1

        # Schema tracking: 保存当前 unit 的 prefill tokens
        self.prefill_schema_tokens.append(self._current_unit_prefill_tokens)

        return _make_result(True)

    def _make_generate_result(
        self,
        start_time: float,
        is_listen: bool = True,
        text: str = "",
        audio_waveform=None,
        end_of_turn: bool = False,
        cost_llm: float = 0.0,
        cost_tts_prep: float = 0.0,
        cost_tts: float = 0.0,
        cost_token2wav: float = 0.0,
        n_tokens: int = 0,
        n_tts_tokens: int = 0,
    ) -> dict:
        """构造 streaming_generate 的标准返回 dict"""
        return {
            "is_listen": is_listen,
            "text": text,
            "audio_waveform": audio_waveform if audio_waveform is not None else self._generate_silence_waveform(),
            "end_of_turn": end_of_turn,
            "current_time": self.audio_chunk_idx,
            "cost_llm": cost_llm,
            "cost_tts_prep": cost_tts_prep,
            "cost_tts": cost_tts,
            "cost_token2wav": cost_token2wav,
            "cost_all": time.time() - start_time,
            "n_tokens": n_tokens,
            "n_tts_tokens": n_tts_tokens,
        }

    @property
    def needs_finalize(self) -> bool:
        """是否有待执行的 finalize（用于调用方检查）"""
        return getattr(self, "_pending_finalize", None) is not None

    @torch.no_grad()
    def streaming_generate(
        self,
        prompt_wav_path=None,
        max_new_speak_tokens_per_chunk=20,
        decode_mode: str = "sampling",
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        listen_prob_scale=1.0,
        listen_top_k=None,
        text_repetition_penalty=1.05,
        text_repetition_window_size=512,
        length_penalty=1.1,
        force_listen_override: bool = False,
    ):
        """生成响应。返回后必须调用 finalize_unit()（除非 needs_finalize 为 False）。

        调用方可以选择调度策略：
        - 模式 A（异步）: generate → 返回结果 → finalize（与网络传输重叠）
        - 模式 B（同步）: generate → finalize → 返回结果
        """
        start_time = time.time()

        if self.is_session_stop_set():
            self._pending_finalize = None  # 无需 finalize
            return self._make_generate_result(start_time, end_of_turn=True)

        # check if there are pending logits to process
        if not hasattr(self, "pending_logits") or self.pending_logits is None:
            self._pending_finalize = None  # 无需 finalize
            return self._make_generate_result(start_time)

        # use pending logits generated in streaming_prefill
        logits = self.pending_logits
        self.pending_logits = None

        # Force listen: initial N calls OR per-chunk force_listen_override from frontend
        force_listen = self._streaming_generate_count < self.force_listen_count or force_listen_override
        self._streaming_generate_count += 1
        if force_listen:
            _reason = "force_listen_override" if force_listen_override else f"call #{self._streaming_generate_count}"
            print(f"[Duplex] streaming_generate: force_listen=True ({_reason})")

        # Force Listen 前处理：如果模型正在说话，先补 <|turn_eos|> 关闭说话 turn
        # 这样 KV cache 序列合法：... <|turn_eos|> <|listen|> </unit>
        # 放在循环外，避免污染 res_ids / speak_count / total_hidden_in_unit
        if force_listen_override and not self.current_turn_ended:
            self.total_ids.append(self.turn_eos_token_id)
            logits, _ = self.decoder.feed(
                self.decoder.embed_token(self.turn_eos_token_id), return_logits=True
            )
            self.current_turn_ended = True
            self._reset_token2wav_for_new_turn()
            logger.info("[Duplex] force_listen: fed <|turn_eos|> to close speaking turn, reset TTS caches")

        total_hidden_in_unit = []
        total_ids_in_unit = []
        current_time = self.audio_chunk_idx
        is_listen = False
        end_of_turn = False

        # 如果上个 chunk 解码了 <|tts_pad|>，本 chunk 禁止再解码
        _tts_pad_suppressed = False
        if self._last_chunk_had_tts_pad and self.tts_pad_id not in self.decoder.forbidden_token_ids:
            self.decoder.forbidden_token_ids.append(self.tts_pad_id)
            _tts_pad_suppressed = True

        llm_start_time = time.time()
        _token_trace = []  # [DEBUG] 记录每个 token 的详细信息
        _pending_terminator_id = None  # 延迟 feed 的终止符，和 </unit> 合并
        _chunk_has_tts_pad = False

        for j in range(max_new_speak_tokens_per_chunk):
            if j == max_new_speak_tokens_per_chunk - 1:
                if self.ls_mode == "explicit":
                    # 不立即 feed，记录下来和 </unit> 合并
                    _pending_terminator_id = self.chunk_eos_token_id
                    self.total_ids.append(self.chunk_eos_token_id)
                    _tok_str = self.tokenizer.decode([self.chunk_eos_token_id])
                    _token_trace.append(f"  j={j} CHUNK_EOS id={self.chunk_eos_token_id} '{_tok_str}' (deferred)")
                    break

            t_step = time.time()
            if force_listen:
                last_id = torch.tensor([self.listen_token_id], dtype=torch.long, device=self.device)
                _decode_ms = 0.0
            else:
                last_id = self.decoder.decode(
                    logits=logits,
                    mode=decode_mode,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    listen_top_k=listen_top_k,
                    listen_prob_scale=listen_prob_scale,
                    text_repetition_penalty=text_repetition_penalty,
                    text_repetition_window_size=text_repetition_window_size,
                    length_penalty=length_penalty,
                )
                _decode_ms = (time.time() - t_step) * 1000

                # if current turn not ended, not allowed to listen (only check when not force_listen)
                if last_id.item() == self.listen_token_id and (not self.current_turn_ended):
                    last_id = torch.tensor([self.tts_bos_token_id], dtype=torch.long, device=self.device)

            self.total_ids.append(last_id.item())

            if last_id.item() == self.tts_pad_id:
                _chunk_has_tts_pad = True

            is_listen = last_id.item() == self.listen_token_id
            _tok_str = self.tokenizer.decode([last_id.item()])
            _is_special = last_id.item() in self.chunk_terminator_token_ids or last_id.item() in self.chunk_speak_token_ids

            # termination condition detection
            if last_id.item() in self.chunk_terminator_token_ids:
                # 不立即 feed 终止符，记录下来和 </unit> 合并（省一次 LLM forward）
                if self.ls_mode == "explicit":
                    _pending_terminator_id = last_id.item()
                _token_trace.append(f"  j={j} TERM id={last_id.item()} '{_tok_str}' decode={_decode_ms:.1f}ms (deferred)")
                break
            else:
                # normal speak
                self.current_turn_ended = False

                # 在 feed 之前检查字符长度，超限则不 feed、不记录，直接终止
                if j != 0:
                    _test_ids = total_ids_in_unit + [last_id.item()]
                    _chunk_text = self.tokenizer.decode(_test_ids, skip_special_tokens=True)
                    if len(_chunk_text) >= 28:
                        self.total_ids.pop()
                        if self.ls_mode == "explicit":
                            _pending_terminator_id = self.chunk_eos_token_id
                            self.total_ids.append(self.chunk_eos_token_id)
                        _kept_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=True) if total_ids_in_unit else ""
                        _token_trace.append(
                            f"  j={j} CHAR_LIMIT len={len(_chunk_text)}>=20, rejected token id={last_id.item()} '{_tok_str}', "
                            f"kept len={len(_kept_text)} text='{_kept_text}' (forced chunk_eos, not fed to KV)"
                        )
                        break

                if last_id.item() in self.chunk_speak_token_ids:
                    pass
                else:
                    self.res_ids.append(last_id.item())
                    self.speak_count += 1

                t_feed = time.time()
                logits, hidden = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)
                _feed_ms = (time.time() - t_feed) * 1000

                assert len(hidden.shape) == 3
                assert hidden.shape[0] == 1
                assert hidden.shape[1] == 1

                end_of_turn = last_id.item() in self.turn_terminator_token_ids

                if end_of_turn:
                    self.current_turn_ended = True

                _kind = "SPECIAL" if _is_special else "TEXT"
                _token_trace.append(f"  j={j} {_kind} id={last_id.item()} '{_tok_str}' decode={_decode_ms:.1f}ms feed={_feed_ms:.1f}ms")

                if j != 0:
                    total_hidden_in_unit.append([last_id.item(), hidden, end_of_turn])
                    total_ids_in_unit.append(last_id.item())

        # 恢复 forbidden list & 更新连续 tts_pad 状态
        if _tts_pad_suppressed:
            self.decoder.forbidden_token_ids.remove(self.tts_pad_id)
            assert self.tts_pad_id not in self.decoder.forbidden_token_ids
        self._last_chunk_had_tts_pad = _chunk_has_tts_pad

        # [DEBUG] 打印完整 token trace
        _trace_str = "\n".join(_token_trace)
        logger.info(
            f"[TokenTrace] t={current_time} is_listen={is_listen} "
            f"text_tokens={len(total_ids_in_unit)} total_steps={len(_token_trace)}"
            f" tts_pad={'suppressed' if _tts_pad_suppressed else 'allowed'}"
            f" had_tts_pad={_chunk_has_tts_pad}\n{_trace_str}"
        )

        # 计算生成的文本（用于滑窗 context 保留，过滤掉特殊 token）
        if os.environ.get("DEBUG_CHUNK_TEXT") == "1":
            generated_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=False) if total_ids_in_unit else ""
            generated_text += f"({len(total_ids_in_unit)},{len(generated_text)})|"
        else:
            generated_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=True) if total_ids_in_unit else ""

        # 存储 finalize 所需状态（延迟到 finalize_unit() 执行）
        input_type = self.current_mode.lower() if self.current_mode else "audio"
        self._pending_finalize = {
            "terminator_id": _pending_terminator_id,
            "total_ids_in_unit": total_ids_in_unit,
            "is_listen": is_listen,
            "generated_text": generated_text,
            "input_type": input_type,
        }

        llm_end_time = time.time()
        cost_llm = llm_end_time - llm_start_time

        if is_listen:
            self.total_hidden.append([])
            return self._make_generate_result(
                start_time, cost_llm=cost_llm,
                n_tokens=len(total_ids_in_unit),
            )

        # 如果 unit 中出现了 tts_pad_id，传空列表给 TTS
        tts_hidden_in_unit = [] if _chunk_has_tts_pad else total_hidden_in_unit
        tts_hidden_in_unit = total_hidden_in_unit

        self.total_hidden.append(total_hidden_in_unit)
        text = generated_text
        if _chunk_has_tts_pad:
            print(f"> speak (tts_pad): {text}, give an empty condition to duplex tts")
        else:
            print(f"> speak: {text}")

        if not self.generate_audio:
            return self._make_generate_result(
                start_time, is_listen=False, text=text,
                end_of_turn=end_of_turn, cost_llm=cost_llm,
                n_tokens=len(total_ids_in_unit),
            )

        # TTS generate
        tts_start_time = time.time()
        tts_prep_start_time = time.time()
        tts_condition = self._convert_results_to_tts_input(tts_hidden_in_unit)
        tts_prep_end_time = time.time()

        max_token_per_chunk = 25 + 1
        min_token_per_chunk = 25 + 1

        if end_of_turn:
            min_token_per_chunk = 0
        force_flush = True
        if self.tts_text_start_pos == 0:  # 这是turn的开始
            min_token_per_chunk = 0  # 可以允许解码<1s的音频
            # min_token_per_chunk = 10 + 1
            force_flush = True

        if self.tts_current_turn_start_time is None:
            self.tts_current_turn_start_time = current_time

        new_tokens, old_kv = self.model.tts.generate_chunk(
            inputs_embeds=tts_condition,
            temperature=self.tts_temperature,
            repetition_penalty=self.tts_repetition_penalty,
            eos_token=self.tts_eos_token,
            force_no_stop=False,
            max_new_token=max_token_per_chunk,
            min_new_tokens=min_token_per_chunk,
            past_key_values=self.tts_past_key_values,
            logits_processors=self.tts_logits_processors,
            text_start_pos=self.tts_text_start_pos,
        )

        tts_end_time = time.time()

        # 更新 TTS 状态（注意：token2wav 的重置必须在音频生成之后，否则会丢失 buffer 中的 tokens）
        if end_of_turn:
            self.tts_text_start_pos = 0
            self.tts_past_key_values = None
            self.tts_current_turn_start_time = None
            # 注意：_reset_token2wav_for_new_turn() 移到下面音频生成之后
        else:
            self.tts_past_key_values = old_kv
            self.tts_text_start_pos += tts_condition.shape[1] + new_tokens.shape[1]

        # Token2Wav 生成（必须在 reset 之前，否则 buffer 中倒数第二个 chunk 的 tokens 会丢失）
        token2wav_start_time = time.time()
        _buf_before = len(self.token2wav_buffer)
        audio_waveform = self._generate_waveform_from_tokens(
            new_tokens, prompt_wav_path, end_of_turn, force_flush=force_flush
        )
        _buf_after = len(self.token2wav_buffer)
        token2wav_end_time = time.time()

        # [DIAG] Token2Wav 诊断：buffer 状态 + 音频产出
        _wav_samples = len(audio_waveform) if audio_waveform is not None else 0
        _wav_dur_ms = (_wav_samples / 24000 * 1000) if _wav_samples > 0 else 0
        _wav_max = float(np.max(np.abs(audio_waveform))) if audio_waveform is not None and _wav_samples > 0 else 0.0
        logger.info(
            f"[Token2Wav] t={current_time} end_of_turn={end_of_turn} force_flush={force_flush} "
            f"tts_tokens={new_tokens.numel()} buf_before={_buf_before} buf_after={_buf_after} "
            f"wav={'None' if audio_waveform is None else f'{_wav_samples}samples/{_wav_dur_ms:.0f}ms'} "
            f"wav_max={_wav_max:.4f}"
        )

        # 在音频生成完成后再重置 token2wav 状态，确保 buffer 中的 tokens 都被处理
        if end_of_turn:
            self._reset_token2wav_for_new_turn()

        return self._make_generate_result(
            start_time, is_listen=False, text=text,
            audio_waveform=audio_waveform, end_of_turn=end_of_turn,
            cost_llm=cost_llm,
            cost_tts_prep=tts_prep_end_time - tts_prep_start_time,
            cost_tts=tts_end_time - tts_start_time,
            cost_token2wav=token2wav_end_time - token2wav_start_time,
            n_tokens=len(total_ids_in_unit),
            n_tts_tokens=new_tokens.numel(),
        )

    @torch.no_grad()
    def finalize_unit(self):
        """完成 streaming_generate 的延迟操作：feed 终止符 + </unit>，注册 unit 结束，执行滑窗。

        必须在 streaming_generate 之后、下一次 streaming_prefill 之前调用。
        设计为可异步调度：调用方可以先返回结果给前端，再在后台执行 finalize。
        """
        state = getattr(self, "_pending_finalize", None)
        if state is None:
            logger.warning("[Duplex] finalize_unit called but no pending finalize state")
            return

        t_start = time.time()

        # 1. 合并 feed：终止符（如有）+ </unit>
        unit_end_id = self.tokenizer.convert_tokens_to_ids("</unit>")
        terminator_id = state["terminator_id"]
        if terminator_id is not None:
            self.decoder.feed(self.decoder.embed_tokens([terminator_id, unit_end_id]))
        else:
            self.decoder.feed(self.decoder.embed_token(unit_end_id))
        self.total_ids.append(unit_end_id)

        # 2. 注册 unit 结束
        self.decoder.register_unit_end(
            input_type=state["input_type"],
            generated_tokens=state["total_ids_in_unit"],
            is_listen=state["is_listen"],
            generated_text=state["generated_text"],
        )

        # 3. 滑窗
        if self.decoder._window_config.sliding_window_mode == "context":
            self.decoder.enforce_window_with_context()
        elif self.decoder._window_config.sliding_window_mode == "basic":
            self.decoder.enforce_window()

        self._pending_finalize = None

        finalize_ms = (time.time() - t_start) * 1000
        logger.info(
            f"[Duplex] finalize_unit: {state['input_type']} is_listen={state['is_listen']} "
            f"finalize={finalize_ms:.0f}ms"
        )
