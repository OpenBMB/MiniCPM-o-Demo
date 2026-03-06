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
import types
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from threading import Thread
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from PIL import Image
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from transformers import LlamaConfig
from transformers import LlamaModel
from transformers import PreTrainedModel
from transformers import Qwen3ForCausalLM
from transformers import Qwen3PreTrainedModel
from transformers import TextIteratorStreamer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import EncoderDecoderCache
from transformers.cache_utils import StaticCache
from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import TopPLogitsWarper
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import ModelOutput
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from enum import Enum

# 相对导入（同目录）
from .configuration_minicpmo import MiniCPMOConfig
from .configuration_minicpmo import MiniCPMTTSConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .processing_minicpmo import MiniCPMOProcessor
from .utils import as_dynamic_cache
from .utils import ChunkPrefillChunkGenerate
from .utils import drop_tokens_from_cache
from .utils import DuplexWindowConfig
from .utils import get_kv_cache_length
from .utils import realign_rotary_suffix
from .utils import SpeculativeSnapshot
from .utils import streaming_token_decoder
from .utils import StreamingWindowConfig
from .utils import torch_clone_recursive
from .utils import TTSSamplingParams
from .utils import TTSStreamingGenerator
from .utils import StreamDecoder

logger = logging.getLogger(__name__)


class ProcessorMode(Enum):
    """处理器模式枚举"""
    CHAT = "chat"           # 单工对话（非流式 TTS）
    STREAMING = "streaming" # 流式对话（流式 TTS）
    DUPLEX = "duplex"       # 双工对话（流式 TTS + 双工组件）


class MiniCPMOPreTrainedModel(Qwen3PreTrainedModel):
    config_class = MiniCPMOConfig


class MiniCPMO(MiniCPMOPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.llm = Qwen3ForCausalLM(config)
        self.embed_dim = self.llm.config.hidden_size
        self.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.llm)  # patch llm

        # init vision module
        if self.config.init_vision:
            self.vpm = self.init_vision_module()
            self.vision_dim = self.vpm.embed_dim
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # init audio module
        if self.config.init_audio:
            self.apm = self.init_audio_module()
            audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
            self.audio_avg_pooler = nn.AvgPool1d(self.config.audio_pool_step, stride=self.config.audio_pool_step)
            self.audio_projection_layer = MultiModalProjector(in_dim=audio_output_dim, out_dim=self.embed_dim)
            self.audio_encoder_layer = -1

        # init tts module
        if self.config.init_tts:
            self.tts = self.init_speech_decoder()

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

        self.think_str = ""
        if self.llm.__class__.__name__ == "Qwen3ForCausalLM":
            self.think_str = "<think>\\n\\n</think>\\n\\n"

        self.default_tts_chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n"
            + self.think_str
            + "<|tts_bos|>' }}{% endif %}"
        )

        # for streaming
        self.reset_session(reset_token2wav_cache=True)

        # streaming audio processing constants
        self.SAMPLE_RATE = 16000
        self.CHUNK_MS = 1000  # regular chunk length (ms)
        self.FIRST_CHUNK_MS = 1035  # first chunk length (ms)
        self.CNN_REDUNDANCY_MS = 0  # CNN redundancy (ms)

        # for sliding window
        self.streaming_window_config = StreamingWindowConfig()
        self.streaming_require_system_prompt = True
        self.streaming_window_enabled = True
        self.force_rope_reindex = False  # RoPE reindex testing switch

        # ========== 统一模式支持（新增）==========
        self._current_mode: Optional[ProcessorMode] = None
        self._unified_initialized = False
        
        # 双 TTS 缓存（用于毫秒级切换）
        self._tts_streaming = None      # Token2wav (streaming TTS)
        self._tts_non_streaming = None  # CosyVoice2 (non-streaming TTS)
        
        # 双工能力组件（组合模式，通过 init_unified 初始化）
        self.duplex: Optional["DuplexCapability"] = None
        
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

    def _ensure_asset_dir(self, asset_subpath: str, model_dir: Optional[str] = None) -> str:
        """Ensure asset directory exists, downloading from HF if needed."""
        model_dir = model_dir or os.path.join(self.config._name_or_path, asset_subpath)
        if not os.path.exists(model_dir):
            from huggingface_hub import snapshot_download

            repo_dir = snapshot_download(
                repo_id="openbmb/MiniCPM-o-4_5",
                allow_patterns=[f"{asset_subpath}/**"],
            )
            model_dir = os.path.join(repo_dir, asset_subpath)
        assert os.path.exists(model_dir), f"Asset directory not found: {model_dir}"
        return model_dir

    def init_streaming_processor(self):
        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        if hasattr(self.processor, "set_streaming_mode"):
            self.processor.set_streaming_mode(
                mode="exact",
                chunk_ms=self.CHUNK_MS,
                first_chunk_ms=self.FIRST_CHUNK_MS,
                cnn_redundancy_ms=self.CNN_REDUNDANCY_MS,
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
            self.processor.reset_streaming()
            self.audio_chunk_idx = 0

    def reset_session(self, reset_token2wav_cache=True):
        self.llm_past_key_values = None
        self.audio_past_key_values = None
        self.tts_last_turn_tokens = None
        self.llm_generated = False  # last turn generated by llm or not
        self.llm_generate_completed = False
        self.new_user_msg = True

        self.session_id = None

        if reset_token2wav_cache:
            self.token2wav_cache = None

        # for sliding window
        self.streaming_text_preserve = 0
        self.streaming_position_offset = 0

        self._rope_inv_freq_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        self._next_round_id = 0
        self._pending_round_id = None

        self._omni_chunk_history: List[Dict[str, Union[str, int]]] = []
        self._round_history: List[Dict[str, Union[int, str, torch.Tensor, Optional[int]]]] = []

    def init_vision_module(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.config.vision_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True,
        )

    def init_audio_module(self):
        if self.config._attn_implementation == "eager":
            self.config.audio_config._attn_implementation = "eager"
        else:
            # using flash_attention_2 will cause: RuntimeError: cu_seqlens_q must have shape (batch_size + 1)
            self.config.audio_config._attn_implementation = "sdpa"

        return MiniCPMWhisperEncoder(self.config.audio_config)

    def init_speech_decoder(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.tts_config.attn_implementation = "flash_attention_2"
        else:
            self.config.tts_config.attn_implementation = "eager"

        return MiniCPMTTS(config=self.config.tts_config, audio_tokenizer=None)

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
            info = self.load_state_dict(state_dict, strict=False)
            logger.info(f"Weights loaded — missing: {len(info.missing_keys)}, unexpected: {len(info.unexpected_keys)}")
            if info.unexpected_keys:
                logger.warning(f"Unexpected keys: {info.unexpected_keys[:5]}...")
            del state_dict

        # Update duplex config
        if duplex_config:
            self._duplex_config.update(duplex_config)

        # Preload TTS vocoder
        self.init_token2wav(streaming=True)

        # Create DuplexCapability instance (composition pattern)
        self.duplex = DuplexCapability(
            model=self,
            device=device,
            **self._duplex_config,
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

        if hasattr(self, "vpm") and "vpm" not in skip:
            self.vpm = torch.compile(self.vpm, **compile_kwargs)
            compiled_modules.append("vpm")
        elif "vpm" in skip:
            skipped_modules.append("vpm")

        if hasattr(self, "llm") and "llm.model" not in skip:
            self.llm.model = torch.compile(self.llm.model, **compile_kwargs)
            compiled_modules.append("llm.model")
        elif "llm.model" in skip:
            skipped_modules.append("llm.model")

        if hasattr(self, "resampler") and "resampler" not in skip:
            self.resampler = torch.compile(self.resampler, **compile_kwargs)
            compiled_modules.append("resampler")
        elif "resampler" in skip:
            skipped_modules.append("resampler")

        if hasattr(self, "tts") and hasattr(self.tts, "model") and "tts.model" not in skip:
            self.tts.model = torch.compile(self.tts.model, **compile_kwargs)
            compiled_modules.append("tts.model")
        elif "tts.model" in skip:
            skipped_modules.append("tts.model")

        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")

        elapsed = _time.time() - t0
        self._compiled = True
        logger.info(
            f"[torch.compile] Wrapping done ({elapsed:.2f}s), "
            f"compiled: {compiled_modules}"
            + (f", skipped: {skipped_modules}" if skipped_modules else "")
            + ". Actual compilation triggers on first forward."
        )
        return self

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
    
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

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

    @staticmethod
    def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device
            num_lookhead:

        Returns:
            torch.Tensor: mask
        """
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
            ret[i, start:ending] = True
        return ret

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Computes the output length of the convolutional layers and the output length of the audio encoder"""
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_vision_embedding(self, data):
        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values,
                        patch_attention_mask=patch_attn_mask,
                        tgt_sizes=tgt_sizes,
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [
                            [
                                (224 // self.config.patch_size),
                                math.ceil(224 / self.config.patch_size),
                            ]
                        ]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)
        else:
            vision_hidden_states = data["vision_hidden_states"]

        return vision_hidden_states

    def get_vllm_embedding(self, data):
        vision_hidden_states = self.get_vision_embedding(data)

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def get_audio_embedding_streaming(
        self,
        data,
        use_extra_context=False,
        prefix_extra_frames=1,
        suffix_extra_frames=1,
        return_debug=False,
        cnn_min_length=None,
    ):
        """Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. It only supports batch_size=1 and is intended
        for streaming scenarios.

        Args:
            data (dict):
                - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                - **"audio_feature_lens"** (List[List[int]]): Lengths of each audio segment for each item in the batch.
            use_extra_context (bool): If True, assumes input contains extra frames for CNN context.
            prefix_extra_frames (int): Number of prefix extra frames.
            suffix_extra_frames (int): Number of suffix extra frames.
            return_debug (bool): Whether to return debug information.
            cnn_min_length (int): Minimum length for CNN input padding.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            assert batch_size == 1
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # whisper's past_key_values management (core)
            if self.audio_past_key_values is not None:
                cache_length = self.audio_past_key_values[0][0].shape[2]
                apm_max_len = self.apm.embed_positions.weight.shape[0]
                if cache_length + max_seq_len >= apm_max_len:
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )
                    self.audio_past_key_values = None

            # build attention mask (bidirectional attention, same as offline mode)
            batch_size, _, max_mel_seq_len = wavforms.shape
            current_seq_len = (max_mel_seq_len - 1) // 2 + 1
            # if use extra context, need to adjust sequence length
            if use_extra_context:
                # calculate actual sequence length after removing redundancy
                # conv2's stride=2, so the mapping from mel frames to output frames is ceil(x/2)
                prefix_to_remove = (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0
                suffix_to_remove = (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0
                current_seq_len = current_seq_len - prefix_to_remove - suffix_to_remove
            # calculate history length (if there is KV cache)
            if self.audio_past_key_values is not None:
                past_len = self.audio_past_key_values[0][0].shape[2]  # get history sequence length
                total_seq_len = past_len + current_seq_len
            else:
                past_len = 0
                total_seq_len = current_seq_len
            # create bidirectional attention mask (full attention)
            audio_attention_mask = torch.zeros(
                (batch_size, 1, current_seq_len, total_seq_len),
                dtype=self.apm.conv1.weight.dtype,
                device=wavforms.device,
            )

            debug_info = {} if return_debug else None

            # DEBUG: 打印输入和 past_key_values 状态
            if hasattr(self, "_debug_prefill") and self._debug_prefill:
                wavform_sum = wavforms.sum().item()
                past_kv_info = (
                    "None" if self.audio_past_key_values is None else f"len={self.audio_past_key_values[0][0].shape[2]}"
                )
                print(
                    f"[DEBUG audio_embed] wavforms sum={wavform_sum:.6f}, shape={wavforms.shape}, past_kv={past_kv_info}"
                )

            # Step 1: APM processing
            ret_apm = self.apm(
                wavforms,
                past_key_values=self.audio_past_key_values,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=audio_attention_mask,
                use_extra_context=use_extra_context,
                prefix_extra_frames=prefix_extra_frames,
                suffix_extra_frames=suffix_extra_frames,
                return_debug=return_debug,
                cnn_min_length=cnn_min_length,
            )
            if return_debug:
                audio_outputs, debug_info["apm_debug"] = ret_apm
            else:
                audio_outputs = ret_apm

            if hasattr(self, "audio_encoder_layer"):
                audio_states = audio_outputs.hidden_states[self.audio_encoder_layer]
            else:
                audio_states = audio_outputs.last_hidden_state

            # DEBUG: 打印 apm 输出的 checksum
            if hasattr(self, "_debug_prefill") and self._debug_prefill:
                apm_out_sum = audio_outputs.last_hidden_state.sum().item()
                audio_states_sum = audio_states.sum().item()
                print(f"[DEBUG audio_embed] apm_output sum={apm_out_sum:.6f}, audio_states sum={audio_states_sum:.6f}")

            if return_debug:
                debug_info["apm_output"] = audio_outputs.last_hidden_state.clone()
                debug_info["audio_states_before_proj"] = audio_states.clone()
                debug_info["attention_mask"] = audio_attention_mask.clone()
                debug_info["use_extra_context"] = use_extra_context
                debug_info["prefix_extra_frames"] = prefix_extra_frames
                debug_info["suffix_extra_frames"] = suffix_extra_frames

            self.audio_past_key_values = audio_outputs.past_key_values

            # Step 2: Projection
            audio_embeds = self.audio_projection_layer(audio_states)

            # DEBUG: 打印 projection 和 pooling 后的 checksum
            if hasattr(self, "_debug_prefill") and self._debug_prefill:
                proj_sum = audio_embeds.sum().item()
                print(f"[DEBUG audio_embed] after_projection sum={proj_sum:.6f}, shape={audio_embeds.shape}")

            if return_debug:
                debug_info["after_projection"] = audio_embeds.clone()

            # Step 3: Pooling
            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            # DEBUG: 打印 pooling 后的 checksum
            if hasattr(self, "_debug_prefill") and self._debug_prefill:
                pool_sum = audio_embeds.sum().item()
                print(f"[DEBUG audio_embed] after_pooling sum={pool_sum:.6f}, shape={audio_embeds.shape}")

            if return_debug:
                debug_info["after_pooling"] = audio_embeds.clone()

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)

            if return_debug:
                debug_info["final_embeddings"] = final_audio_embeds
                return final_audio_embeds, debug_info
            else:
                return final_audio_embeds
        else:
            return [] if not return_debug else ([], {})

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        dtype = self.apm.embed_positions.weight.dtype
        device = self.apm.embed_positions.weight.device

        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(
                    0,
                    max_seq_len,
                    dtype=audio_feature_lens.dtype,
                    device=audio_feature_lens.device,
                )
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
            )

            if chunk_length > 0:
                chunk_num_frame = int(chunk_length * 50)
                chunk_mask = self.subsequent_chunk_mask(
                    size=max_seq_len,
                    chunk_size=chunk_num_frame,
                    num_left_chunks=-1,
                    device=audio_attention_mask_.device,
                )
                audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_states = self.apm(
                wavforms, output_hidden_states=True, attention_mask=audio_attention_mask
            ).hidden_states[self.audio_encoder_layer]
            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        elif self.training and dummy:
            dummy_wavs = torch.zeros((1, 80, 100), device=device, dtype=dtype)
            audio_states = self.apm(dummy_wavs, output_hidden_states=True).hidden_states[self.audio_encoder_layer]

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)
            return [audio_embeds]
        else:
            return []

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        """
        Args:
            data:
            input_embeddings:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding or not

        Returns:
            final embeddings with audio feature
        """
        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(data)
        else:
            audio_embeddings = self.get_audio_embedding(data, chunk_length)

        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]

                if self.config.stream_input:
                    assert bs == 1, "audio stream_input mode only support batch size 1"
                    for i in range(bs):
                        audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                            device=input_embeddings.device, dtype=input_embeddings.dtype
                        )
                        audio_start_pos = 0
                        for bound in audio_bounds[i]:
                            audio_len = bound[1] - bound[0]
                            input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                                audio_start_pos : audio_start_pos + audio_len, :
                            ]
                            audio_start_pos += audio_len
                else:
                    for i in range(bs):
                        audio_embs = audio_embeddings[i]
                        bounds = audio_bounds[i]
                        for embs, bound in zip(audio_embs, bounds):
                            audio_indices = torch.arange(bound[0], bound[1], dtype=torch.long).to(
                                input_embeddings.device
                            )

                            if embs.shape[0] != len(audio_indices):
                                logger.error(f"Sample {i}:")
                                logger.error(f"  Bounds: {bound}, Indices Length: {len(audio_indices)}")
                                logger.error(f"  Embeddings Shape: {embs.shape}")
                                logger.error(
                                    f"  Input Embedding Shape at Indices: {input_embeddings[i, audio_indices].shape}"
                                )
                                raise ValueError(
                                    f"Shape mismatch: Trying to assign embeddings of shape {embs.shape} "
                                    f"to input indices of length {len(audio_indices)}"
                                )
                            input_embeddings[i, audio_indices] = embs.to(input_embeddings.dtype)
        elif self.training:
            for i in range(bs):
                # dummy audio_embedings
                input_embeddings += audio_embeddings[0].mean() * 0

        return input_embeddings

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        vllm_embedding = self.get_omni_embedding(
            data,
            input_embeddings=vllm_embedding,
            chunk_length=self.config.audio_chunk_length,
        )

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs,
        )

    def _decode(self, inputs_embeds, tokenizer, attention_mask, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        return outputs

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_config = {
            "inputs_embeds": inputs_embeds,
            "pad_token_id": 0,
            "eos_token_id": terminators,
            "streamer": streamer,
        }
        generation_config.update(kwargs)
        thread = Thread(target=self.llm.generate, kwargs=generation_config)
        thread.start()
        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result))
        return result_text

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=None,
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        **kwargs,
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                model_inputs,
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=self.config.audio_chunk_length,
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
                outputs = {}  # if stream return TextIteratorStreamer and output is empty
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)
                result = self._decode_text(outputs.sequences, tokenizer)

        return result, outputs

    def _build_streaming_mask(self, tts_tokens_len):
        tts_sequence_full_length = 1 + self.tts.streaming_text_reserved_len + 1
        streaming_attention_mask = torch.zeros(tts_sequence_full_length, dtype=torch.int8)
        streaming_attention_mask[0 : 1 + 1 + tts_tokens_len + 1] = 1
        streaming_attention_mask[-1] = 1
        return streaming_attention_mask

    def _generate_mel_spec(self, inputs, outputs, text, output_chunk_size=25, tts_max_new_tokens=2048):
        spk_embeds = self._get_last_spk_embeds(inputs, outputs)

        text = text.split("<|tts_bos|>")[-1]
        gen_text = text.split("<|tts_eos|>")[0]
        tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
        tts_inputs = self.tts_processor.text_tokenizer.encode(tts_text, add_special_tokens=False)
        tts_input_ids = torch.Tensor(tts_inputs).unsqueeze(0).to(self.device, dtype=torch.long)
        streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=626,
            top_p=self.tts.top_p,
            top_k=self.tts.top_k,
            repetition_penalty=self.tts.repetition_penalty,
        )

        condition_length = 1 + self.tts.streaming_text_reserved_len + 1

        dtype = self.tts.emb_text.weight.dtype
        emb = torch.zeros(1, condition_length, self.tts.num_vq, dtype=dtype, device=self.tts.device)
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]

        audio_input_ids = torch.zeros(
            1,
            condition_length,
            self.tts.num_vq,
            dtype=torch.long,
            device=self.tts.device,
        )

        eos_lab = False
        for chunk_idx in range(math.ceil(emb.shape[1] / self.tts.streaming_text_chunk_size)):
            if chunk_idx == 0:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 0
                end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + 1
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 1
                end = min(
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size + 1,
                    condition_length - 1,
                )

            if end - begin > 0:
                text_input_ids = tts_input_ids[:, begin:end]
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                if begin == 0:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        lm_spk_emb_last_hidden_states=spk_embeds,
                    )
                else:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    )

            outputs = self.tts.generate(
                input_ids=audio_input_ids,
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
                max_new_token=output_chunk_size,
                force_no_stop=self.force_no_stop,
                temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
            )
            audio_input_ids = outputs.audio_input_ids
            past_key_values = outputs.past_key_values

            if outputs.finished:
                logger.debug("Generation finished.")
                eos_lab = True
                break

        if not eos_lab:
            logger.debug("eos_lab False, Generation continue.")
            while True:
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )

                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > tts_max_new_tokens:
                    logger.debug(f"Generation length > {tts_max_new_tokens}, stopped.")
                    break

    @staticmethod
    def prepare_generation_config(do_sample, max_new_tokens=50, min_new_tokens=0, **kwargs):
        num_beams = kwargs.get("num_beams", 3)
        generation_config = {
            "num_beams": num_beams,
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.02,
        }

        if do_sample:
            generation_config.update(
                {
                    "top_p": 0.8,
                    "top_k": 100,
                    "temperature": 0.7,
                    "do_sample": True,
                    "repetition_penalty": 1.02,
                }
            )
        elif num_beams > 1:
            generation_config.update({"num_beams": num_beams, "repetition_penalty": 1.2, "do_sample": False})
        else:
            generation_config.update({"do_sample": False, "repetition_penalty": 1.02})

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config["max_new_tokens"] = max_new_tokens

        return generation_config

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
        # todo: deprecated
        if sampling is not None:
            do_sample = sampling
        if omni_input is not None:
            omni_mode = omni_input

        batched = isinstance(msgs[0], list)
        msgs_list = msgs
        images_list = image

        if not batched:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert do_sample or not stream, "if use stream mode, make sure do_sample=True"

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
                    elif isinstance(c, np.ndarray):  # audio
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

            prompts_lists.append(
                self.processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False if teacher_forcing else True,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        if not merge_audio_from_same_content:
            audio_parts_list = None

        inputs = self.processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            stream_input=stream_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        generation_config = self.prepare_generation_config(
            do_sample=do_sample, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, **kwargs
        )
        generation_config.pop("max_new_tokens", None)

        inputs.pop("image_sizes")

        # teacher_forcing = True => generate audio with given text
        with torch.inference_mode():
            res, outputs = self.generate(
                **inputs,
                tokenizer=self.processor.tokenizer,
                max_new_tokens=1 if teacher_forcing else max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config,
            )

        # spk bound and tts bound
        tts_bos_token = self.processor.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        tts_eos_token = self.processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>")

        # Combine input_ids and generated sequences to get complete sequence
        input_ids = inputs["input_ids"][0]
        generated_ids = outputs.sequences[0]
        # Combine by concatenating input_ids with the new tokens from generated sequence
        full_sequence = torch.cat([input_ids, generated_ids])
        # Update the sequences in outputs
        full_sequences = full_sequence.unsqueeze(0)

        outputs["full_sequences"] = full_sequences

        # 存储 token 统计，供 ChatView 等外部消费者读取
        # input_tokens: tokenizer 级别（含 audio/image 占位符，不含 embedding 展开）
        # generated_tokens: LLM 实际生成的 token 数
        self._last_chat_token_stats = {
            "input_tokens": len(input_ids),
            "generated_tokens": len(generated_ids),
        }

        tts_bos_indices = []
        tts_eos_indices = []
        for i, x in enumerate(full_sequences[0]):
            if x == tts_bos_token:
                tts_bos_indices.append(i + 1)  # tts_bos + 1 才是第一个tts的位置，这样方便直接给tts去slice hidden states
            elif x == tts_eos_token:
                if teacher_forcing and i == len(full_sequences[0]) - 1:
                    continue
                tts_eos_indices.append(i)

        tts_bos_idx = tts_bos_indices[-1] if tts_bos_indices else -1
        # Use None instead of -1 when no EOS token found, so that slice [start:None]
        # means "to the end" rather than [start:-1] which excludes the last element
        tts_eos_idx = tts_eos_indices[-1] if tts_eos_indices else None

        tts_bound = (tts_bos_idx, tts_eos_idx)

        answer = res[0]
        if answer is not None:
            answer = answer.rstrip("<|tts_eos|>")

        generated_waveform = None
        if use_tts_template and generate_audio:
            try:
                # TTS ref audio 优先级：
                # 1. tts_ref_audio（显式指定的 TTS 参考音频，与 LLM ref audio 分离）
                # 2. input_audios_list[0][0]（messages 中第一个音频，即 LLM ref audio）
                # 3. None（无参考音频）
                if tts_ref_audio is not None:
                    _tts_audio_prompt = tts_ref_audio
                    logger.info(f"[Chat TTS] Using separate tts_ref_audio: {len(tts_ref_audio)} samples ({len(tts_ref_audio)/16000:.1f}s)")
                elif len(input_audios_list) > 0 and len(input_audios_list[0]) > 0:
                    _tts_audio_prompt = input_audios_list[0][0]
                    logger.info(f"[Chat TTS] Using LLM ref audio from messages: {len(_tts_audio_prompt)} samples ({len(_tts_audio_prompt)/16000:.1f}s)")
                else:
                    _tts_audio_prompt = None
                    logger.warning("[Chat TTS] No ref audio available for TTS")

                generated_waveform = self._generate_speech_non_streaming(
                    outputs=outputs,
                    tts_bound=tts_bound,
                    tts_proj_layer=tts_proj_layer,
                    audio_prompt=_tts_audio_prompt,
                    output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
                    tts_sampling_params=tts_sampling_params,
                )
                # 统一为 numpy array
                if isinstance(generated_waveform, torch.Tensor):
                    generated_waveform = generated_waveform.cpu().numpy()

                # 如果指定了保存路径，也保存到文件
                if output_audio_path and generated_waveform is not None:
                    import soundfile as sf
                    sf.write(output_audio_path, generated_waveform, samplerate=24000)
                    logger.debug(f"audio saved to {output_audio_path}")
            except:
                import traceback
                traceback.print_exc()
                generated_waveform = None

        if return_prompt:
            return answer, prompts_lists[0], generated_waveform
        elif generated_waveform is not None:
            return answer, generated_waveform
        else:
            return answer

    @torch.inference_mode()
    def _generate_speech_non_streaming(
        self,
        outputs,
        tts_bound,
        tts_proj_layer,
        audio_prompt,
        output_tts_inputs_embeds_path=None,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
    ):
        last_hidden_states = [hs[tts_proj_layer] for hs in outputs.hidden_states]
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])
        
        # FIX: 某些 pt 权重可能导致 hidden_states 和 sequences 长度不一致
        # 以 full_sequences 为准（这是实际要用的 tokens）
        full_seq_len = len(outputs["full_sequences"][0])
        if last_hidden_states.shape[0] != full_seq_len:
            logger.warning(f"TTS: hidden_states({last_hidden_states.shape[0]}) != full_sequences({full_seq_len}), truncating")
            last_hidden_states = last_hidden_states[:full_seq_len]

        spk_embeds = (
            torch.ones([0, self.tts.config.hidden_size]).to(last_hidden_states.device).to(last_hidden_states.dtype)
        )

        if self.tts.condition_type == "hidden_text_merge":
            llm_tokens = outputs["full_sequences"][0][tts_bound[0] : tts_bound[1]]
            llm_tokens = torch.tensor(llm_tokens, device=self.tts.emb_text.weight.device, dtype=torch.long)
            llm_embeds = self.tts.emb_text(llm_tokens)  # make sure emb_text is compatible with llm vocab size

            hidden_embeds = last_hidden_states[tts_bound[0] : tts_bound[1]]
            hidden_embeds = self.tts.projector_semantic(hidden_embeds)

            if self.tts.config.normalize_projected_hidden:
                hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

            tts_embeds = llm_embeds + hidden_embeds
            if self.tts.interleaved:
                chunks = []
                cond_length = tts_embeds.shape[0]
                for i in range(0, cond_length, 10):
                    chunks.append(tts_embeds[i : i + 10])
                tts_embeds = chunks
        else:
            raise NotImplementedError

        audio_bos = [self.tts.audio_bos_token_id]
        audio_bos = torch.tensor(audio_bos, device=self.tts.emb_text.weight.device, dtype=torch.long)

        audio_bos_embeds = self.tts.emb_text(audio_bos)

        text_eos_embed = self.tts.emb_text(
            torch.tensor(
                [self.tts.config.text_eos_token_id],
                device=self.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        )

        if self.tts.interleaved:

            tts_embeds[-1] = torch.cat([tts_embeds[-1], text_eos_embed], dim=0)
            for i in range(len(tts_embeds)):
                tts_embeds[i] = torch.cat([tts_embeds[i], audio_bos_embeds], dim=0).unsqueeze(0)
            outputs = self.tts.interleaved_generate(
                spk_embeds=spk_embeds,
                conditions=tts_embeds,
                temperature=0.8,
                repetition_penalty=1.05,
                eos_token=torch.tensor(
                    [self.tts.config.num_audio_tokens - 1],
                    dtype=torch.long,
                    device=self.tts.device,
                ),
            )
        else:
            if self.tts.condition_type == "tts_token":
                inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds], dim=0).unsqueeze(
                    0
                )
            elif self.tts.condition_type == "tts_token_streaming":
                tts_embeds[1] = spk_embeds.squeeze(0)  # apply speaker embedding
                inputs_embeds = tts_embeds.unsqueeze(0)
            else:  # modern case
                inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds], dim=0).unsqueeze(
                    0
                )

            # save inputs_embeds to file
            if output_tts_inputs_embeds_path:
                torch.save(inputs_embeds, output_tts_inputs_embeds_path)

            outputs = self.tts.generate(
                inputs_embeds=inputs_embeds,
                sampling_params=tts_sampling_params,
                eos_token=torch.tensor(
                    [self.tts.config.num_audio_tokens - 1],
                    dtype=torch.long,
                    device=self.tts.device,
                ),
            )

        if self.tts.config.audio_tokenizer_type == "s3tokenizer":
            # ========== CosyVoice2 vocoder 路径 ==========
            generated_tokens = outputs.new_ids.squeeze(-1)
            reference_audio = audio_prompt
            if reference_audio is not None:
                logger.debug("use reference audio in data to generate waveform")
                prompt_speech_16k = torch.tensor(reference_audio).unsqueeze(0)

            if self.tts.config.s3_stream_generate:
                waveform_pred = self.tts.audio_tokenizer.inference_token2wav(
                    speech_tokens=generated_tokens,
                    prompt_speech_16k=prompt_speech_16k,
                    prompt_speech=None,
                    stream=True,
                    n_timesteps=self.tts.config.s3_stream_n_timesteps,
                    code_chunk_size=self.tts.config.s3_stream_chunk_size,
                    chunk_prelook_size=self.tts.config.s3_stream_prelook_size,
                    use_attn_idx=False,
                )
                return waveform_pred[0]
            else:
                for i, j in enumerate(
                    self.tts.audio_tokenizer.token2wav(
                        speech_token=generated_tokens,
                        speech_token_len=torch.tensor([generated_tokens.shape[1]], device=generated_tokens.device),
                        prompt_speech_16k=prompt_speech_16k,
                        stream=False,
                    )
                ):
                    waveform_pred = j["tts_speech"]
                    waveform_sample_rate = self.tts.audio_tokenizer.sample_rate  # 24000 here, not 16000 input.
                return waveform_pred[0]

        elif self.tts.config.audio_tokenizer_type == "s3tokenizer_step_audio":
            # ========== Token2Wav vocoder 路径（非流式批量转换）==========
            generated_tokens = outputs.new_ids.squeeze(-1)
            token_ids = generated_tokens.reshape(-1).tolist()

            if not token_ids:
                logger.warning("Token2Wav non-streaming: 无 audio tokens 可转换")
                return None

            # Token2Wav 需要文件路径作为 prompt，将 ref audio 写入临时文件
            reference_audio = audio_prompt
            prompt_wav_path = None
            temp_file = None

            if reference_audio is not None:
                logger.debug("use reference audio in data to generate waveform (Token2Wav)")
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="t2w_ref_")
                sf.write(temp_file.name, reference_audio, 16000)
                prompt_wav_path = temp_file.name

            try:
                # 初始化 Token2Wav stream cache
                # set_stream_cache() 返回 (flow_cache, hift_cache)，需手动设置到实例属性
                self.tts.audio_tokenizer.cache = None
                flow_cache, hift_cache = self.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)
                self.tts.audio_tokenizer.stream_cache = flow_cache
                self.tts.audio_tokenizer.hift_cache_dict = hift_cache

                # Token2Wav 的 flow 模型有固定大小的 attention cache buffer，
                # 必须分块喂入（与 streaming 路径一致），否则长文本会溢出。
                CHUNK_SIZE = 25
                pre_lookahead = 3
                waveform_chunks = []
                buffer = list(token_ids)
                pos = 0

                while pos + CHUNK_SIZE + pre_lookahead <= len(buffer):
                    chunk = buffer[pos : pos + CHUNK_SIZE + pre_lookahead]
                    is_last = (pos + CHUNK_SIZE + pre_lookahead >= len(buffer))
                    wav_chunk = self.tts.audio_tokenizer.stream(
                        chunk, prompt_wav=prompt_wav_path,
                        last_chunk=is_last, return_waveform=True,
                    )
                    if wav_chunk is not None:
                        waveform_chunks.append(wav_chunk.squeeze())
                    pos += CHUNK_SIZE

                # flush 剩余 tokens
                if pos < len(buffer):
                    remaining = buffer[pos:]
                    wav_chunk = self.tts.audio_tokenizer.stream(
                        remaining, prompt_wav=prompt_wav_path,
                        last_chunk=True, return_waveform=True,
                    )
                    if wav_chunk is not None:
                        waveform_chunks.append(wav_chunk.squeeze())

                if waveform_chunks:
                    waveform = np.concatenate(waveform_chunks)
                    logger.info(
                        f"Token2Wav non-streaming: {len(token_ids)} tokens → "
                        f"{len(waveform)} samples ({len(waveform)/24000:.2f}s), "
                        f"{len(waveform_chunks)} chunks"
                    )
                    return waveform
                else:
                    logger.warning("Token2Wav non-streaming: 所有 chunks 返回空")
                    return None
            finally:
                # 清理临时文件
                if temp_file is not None:
                    try:
                        os.unlink(temp_file.name)
                    except OSError:
                        pass
        else:
            raise NotImplementedError(
                f"不支持的 audio_tokenizer_type: {self.tts.config.audio_tokenizer_type}"
            )

    @torch.inference_mode()
    def init_token2wav_cache(self, prompt_speech_16k):
        if hasattr(self.tts.audio_tokenizer, "set_stream_cache"):
            self.tts.audio_tokenizer.cache = None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                prompt_wav_path = tmp_wav.name
                sf.write(prompt_wav_path, prompt_speech_16k, 16000)
                flow_cache_base, hift_cache_base = self.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)

            self.token2wav_cache = {
                "flow_cache_base": torch_clone_recursive(flow_cache_base),
                "hift_cache_base": torch_clone_recursive(hift_cache_base),
            }
        else:
            model_input = self.tts.audio_tokenizer.frontend.frontend_token2wav(
                speech_tokens=torch.zeros(1, 1, dtype=torch.long, device=self.tts.device),
                speech_16k=None,
                prompt_speech_16k=prompt_speech_16k,
                resample_rate=self.tts.audio_tokenizer.sample_rate,
                prompt_speech=None,
            )

            prompt_token = model_input["flow_prompt_speech_token"]
            prompt_feat = model_input["prompt_speech_feat"]
            embedding = model_input["flow_embedding"]

            if self.tts.audio_tokenizer.fp16:
                prompt_feat = prompt_feat.to(torch.half)
                embedding = embedding.to(torch.half)

            prepared_cache = self.tts.audio_tokenizer.model.prepare_cache_from_prompt(
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                n_timesteps=self.tts.config.s3_stream_n_timesteps,
                code_chunk_size=self.tts.config.s3_stream_chunk_size,
                chunk_prelook_size=self.tts.config.s3_stream_prelook_size,
                use_attn_idx=False,
            )

            self.token2wav_cache = prepared_cache

    # for sliding window
    def _ensure_dynamic_cache(self):
        cache = self.llm_past_key_values
        if cache is None:
            return None

        cache = as_dynamic_cache(cache)
        if isinstance(cache, DynamicCache):
            self.llm_past_key_values = cache
            return cache

        return None

    def _get_kv_cache_length(self, cache=None):
        cache = cache if cache is not None else self.llm_past_key_values
        return get_kv_cache_length(cache)

    # todo: not-used del?
    def _rebuild_cache_from_history(self):
        preserved_ids: List[torch.Tensor] = []
        for entry in self._omni_chunk_history:
            ids = entry.get("input_ids")
            if ids is None or not isinstance(ids, torch.Tensor) or ids.numel() == 0:
                continue
            preserved_ids.append(ids.to(self.device))
        if not preserved_ids:
            self.llm_past_key_values = None
            self.streaming_position_offset = 0
            self._rope_inv_freq_cache.clear()
            return

        concat_ids = torch.cat(preserved_ids, dim=1)
        attention_mask = torch.ones((1, concat_ids.shape[1]), dtype=torch.bool, device=self.device)
        outputs = self.llm(
            input_ids=concat_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        self.llm_past_key_values = outputs.past_key_values
        self.streaming_position_offset = 0
        self._rope_inv_freq_cache.clear()

    def _get_rope_theta(self) -> float:
        return float(getattr(self.llm.config, "rope_theta", 10000.0))

    def _realign_rotary_suffix(
        self,
        suffix_keys: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapper for realign_rotary_suffix using instance's rope_theta and cache."""
        return realign_rotary_suffix(
            suffix_keys,
            old_positions,
            new_positions,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )

    def _encode_text(self, tokenizer, text) -> Optional[torch.Tensor]:
        if tokenizer is None or not text:
            return None
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.to(self.device)

    @staticmethod
    def _safe_decode(tokenizer, input_ids):
        if tokenizer is None or input_ids is None:
            return None
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.cpu().tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            ids = input_ids
        try:
            return tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            return None

    def _finalize_round(
        self, round_id: Optional[int], cache_before: int, assistant_input_ids: Optional[torch.Tensor] = None
    ):
        if round_id is None:
            self._pending_round_id = None
            return
        cache_after = self._get_kv_cache_length()
        if assistant_input_ids is not None:
            assistant_len = assistant_input_ids.shape[1]
        else:
            assistant_len = max(cache_after - cache_before, 0)
        if assistant_len > 0:
            self._register_chunk(
                assistant_len,
                "assistant",
                round_id=round_id,
                input_ids=assistant_input_ids,
                tokenizer=self.processor.tokenizer if hasattr(self, "processor") else None,
            )
        logger.info(
            "Finalized round=%s cache len before=%s after=%s assistant_len=%s",
            round_id,
            cache_before,
            cache_after,
            assistant_len,
        )
        self._pending_round_id = None
        self._next_round_id += 1

    def _register_chunk(
        self,
        seq_len: int,
        chunk_type: str,
        *,
        round_id: int,
        input_ids=None,
        tokenizer=None,
    ) -> None:
        if seq_len <= 0:
            return
        entry = {"length": int(seq_len), "type": chunk_type, "round": round_id}
        if input_ids is not None:
            entry["input_ids"] = input_ids.clone().detach()
            entry["decoded"] = self._safe_decode(tokenizer, entry["input_ids"])
        else:
            entry["input_ids"] = None
            entry["decoded"] = None
        self._omni_chunk_history.append(entry)
        logger.info(
            "Registered chunk round=%s type=%s len=%s decoded=%s",
            round_id,
            chunk_type,
            entry["length"],
            entry["decoded"],
        )
        if chunk_type == "system":
            self.streaming_text_preserve = max(self.streaming_text_preserve, entry["length"])

    def _drop_tokens_from_cache(self, length: int, cache: DynamicCache) -> bool:
        """Drop tokens from cache using the utility function."""
        _, new_offset, success = drop_tokens_from_cache(
            cache=cache,
            length=length,
            preserve=self.streaming_text_preserve,
            position_offset=self.streaming_position_offset,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )
        if success:
            self.streaming_position_offset = new_offset
        return success

    def _drop_next_round(self, cache: DynamicCache) -> bool:
        seen_rounds = set()
        for entry in self._omni_chunk_history:
            round_id = entry.get("round")
            if round_id is None or round_id in seen_rounds:
                continue
            seen_rounds.add(round_id)
            round_entries = [e for e in self._omni_chunk_history if e.get("round") == round_id]
            if any(e.get("type") == "system" for e in round_entries):
                continue
            if self._drop_round(round_id, cache):
                return True
        return False

    def _drop_round(self, round_id: int, cache: DynamicCache) -> bool:
        entries = [e for e in self._omni_chunk_history if e.get("round") == round_id]
        if not entries:
            return False
        total_len = sum(e["length"] for e in entries)
        if total_len <= 0:
            for e in entries:
                self._omni_chunk_history.remove(e)
            return False
        if not self._drop_tokens_from_cache(total_len, cache):
            return False
        for e in entries:
            logger.info(
                "Dropped round=%s chunk type=%s len=%s decoded=%s",
                round_id,
                e["type"],
                e["length"],
                e.get("decoded"),
            )
            self._omni_chunk_history.remove(e)
        return True

    def _enforce_text_window(self) -> None:
        if not self.streaming_window_enabled:
            return
        cache = self._ensure_dynamic_cache()
        if cache is None:
            return
        high_limit = max(0, int(self.streaming_window_config.text_window_high_tokens))
        low_limit = max(0, int(self.streaming_window_config.text_window_low_tokens))
        if high_limit <= 0:
            return
        target = max(0, low_limit)
        total_len = self._get_kv_cache_length(cache)
        if total_len <= high_limit:
            return
        dropped_any = False
        while total_len > target:
            if not self._drop_next_round(cache):
                break
            dropped_any = True
            total_len = self._get_kv_cache_length(cache)

    # for sliding window

    # ============== 抢跑快照/恢复接口 ==============

    def _save_speculative_snapshot(self) -> SpeculativeSnapshot:
        """Internal method: save speculative snapshot.

        Called at the start of streaming_generate, saves to self._speculative_snapshot.

        Save strategy:
        - LLM KV Cache: only record length (restore by truncation, zero extra VRAM)
        - Audio KV Cache: deep clone (as generate sets it to None)
        - Mel processor: full state snapshot (including buffer)
        """
        # 1. 获取 LLM cache 信息
        llm_cache_length = self._get_kv_cache_length()
        llm_cache_checksum = None
        if self.llm_past_key_values is not None and hasattr(self.llm_past_key_values, "key_cache"):
            if len(self.llm_past_key_values.key_cache) > 0:
                llm_cache_checksum = self.llm_past_key_values.key_cache[0].sum().item()

        # 2. 获取 audio cache 长度并克隆 audio_past_key_values
        audio_cache_length = 0
        audio_cache_checksum = None
        audio_past_key_values_clone = None
        if self.audio_past_key_values is not None:
            # 处理 DynamicCache 格式（Whisper encoder 可能返回此格式）
            if isinstance(self.audio_past_key_values, DynamicCache):
                if hasattr(self.audio_past_key_values, "key_cache") and len(self.audio_past_key_values.key_cache) > 0:
                    audio_cache_length = self.audio_past_key_values.key_cache[0].shape[2]
                    audio_cache_checksum = self.audio_past_key_values.key_cache[0].sum().item()
                # 深度克隆 DynamicCache
                cloned_cache = DynamicCache()
                for k, v in zip(self.audio_past_key_values.key_cache, self.audio_past_key_values.value_cache):
                    cloned_cache.update(k.clone(), v.clone(), layer_idx=len(cloned_cache.key_cache))
                audio_past_key_values_clone = cloned_cache
                logger.debug(f"[Speculative] Cloned DynamicCache with length {audio_cache_length}")
            # 处理 EncoderDecoderCache 格式
            elif isinstance(self.audio_past_key_values, EncoderDecoderCache):
                self_attn_cache = self.audio_past_key_values.self_attention_cache
                if hasattr(self_attn_cache, "key_cache") and len(self_attn_cache.key_cache) > 0:
                    audio_cache_length = self_attn_cache.key_cache[0].shape[2]
                    audio_cache_checksum = self_attn_cache.key_cache[0].sum().item()
                # 深度克隆 EncoderDecoderCache
                cloned_self_attn = DynamicCache()
                if hasattr(self_attn_cache, "key_cache"):
                    for k, v in zip(self_attn_cache.key_cache, self_attn_cache.value_cache):
                        cloned_self_attn.update(k.clone(), v.clone(), layer_idx=len(cloned_self_attn.key_cache))
                cross_attn_cache = self.audio_past_key_values.cross_attention_cache
                cloned_cross_attn = DynamicCache()
                if hasattr(cross_attn_cache, "key_cache"):
                    for k, v in zip(cross_attn_cache.key_cache, cross_attn_cache.value_cache):
                        cloned_cross_attn.update(k.clone(), v.clone(), layer_idx=len(cloned_cross_attn.key_cache))
                audio_past_key_values_clone = EncoderDecoderCache(cloned_self_attn, cloned_cross_attn)
                logger.debug(f"[Speculative] Cloned EncoderDecoderCache with length {audio_cache_length}")
            # 处理 tuple 格式（兼容旧格式）
            elif isinstance(self.audio_past_key_values, tuple) and len(self.audio_past_key_values) > 0:
                audio_cache_length = self.audio_past_key_values[0][0].shape[2]
                audio_cache_checksum = self.audio_past_key_values[0][0].sum().item()
                # 深度克隆 audio_past_key_values（tuple of tuples of tensors）
                audio_past_key_values_clone = tuple(
                    tuple(t.clone() for t in layer_cache) for layer_cache in self.audio_past_key_values
                )

        # 3. 获取 mel processor 快照
        mel_processor_snapshot = None
        mel_buffer_checksum = None
        if hasattr(self, "processor") and self.processor is not None:
            mel_processor_snapshot = self.processor.get_streaming_snapshot()
            if mel_processor_snapshot:
                buf = mel_processor_snapshot.get("buffer")
                if buf is not None and len(buf) > 0:
                    mel_buffer_checksum = float(buf.sum())

        # 4. 保存 RNG 状态（关键：用于恢复后确保 dithering 等随机操作的确定性）
        rng_state_cpu = torch.get_rng_state()
        rng_state_cuda = None
        if torch.cuda.is_available() and self.device.type == "cuda":
            rng_state_cuda = torch.cuda.get_rng_state(self.device)

        # 5. 创建快照
        snapshot = SpeculativeSnapshot(
            llm_cache_length=llm_cache_length,
            audio_cache_length=audio_cache_length,
            new_user_msg=self.new_user_msg,
            llm_generated=self.llm_generated,
            llm_generate_completed=self.llm_generate_completed,
            next_round_id=self._next_round_id,
            pending_round_id=self._pending_round_id,
            omni_chunk_history_length=len(self._omni_chunk_history),
            tts_last_turn_tokens=self.tts_last_turn_tokens.clone() if self.tts_last_turn_tokens is not None else None,
            audio_chunk_idx=self.audio_chunk_idx,
            mel_processor_snapshot=mel_processor_snapshot,
            audio_past_key_values=audio_past_key_values_clone,
            timestamp=time.time(),
            # 调试字段
            llm_cache_checksum=llm_cache_checksum,
            audio_cache_checksum=audio_cache_checksum,
            mel_buffer_checksum=mel_buffer_checksum,
            # RNG 状态
            rng_state_cpu=rng_state_cpu,
            rng_state_cuda=rng_state_cuda,
        )

        logger.info("[Speculative] Saved snapshot: %s", snapshot.summary())
        logger.debug(
            "[Speculative] Snapshot checksums: llm=%.6f, audio=%.6f, mel_buf=%.6f",
            llm_cache_checksum or 0.0,
            audio_cache_checksum or 0.0,
            mel_buffer_checksum or 0.0,
        )

        return snapshot

    def restore_speculative_snapshot(self) -> bool:
        """Restore speculative snapshot - called when VAD speculation fails.

        Restores model state to before streaming_generate was called,
        allowing continued streaming_prefill for newly arrived audio.

        Notes:
        - Snapshot is saved when streaming_generate is called with enable_speculative_snapshot=True
        - This method uses the most recent snapshot for restoration
        - Snapshot is cleared after restore, cannot be called repeatedly

        Returns:
            bool: Whether restoration was successful
        """
        snapshot = getattr(self, "_speculative_snapshot", None)

        if snapshot is None:
            logger.warning("[Speculative] No snapshot to restore")
            return False

        try:
            # 记录恢复前的状态（用于日志对比）
            current_cache_length = self._get_kv_cache_length()
            current_history_length = len(self._omni_chunk_history)

            logger.info(
                "[Speculative] Restoring snapshot: target=%s",
                snapshot.summary(),
            )
            logger.info(
                "[Speculative] Current state before restore: llm_cache=%d, history_len=%d, "
                "audio_chunk_idx=%d, new_user_msg=%s, llm_generated=%s",
                current_cache_length,
                current_history_length,
                self.audio_chunk_idx,
                self.new_user_msg,
                self.llm_generated,
            )

            # 1. 裁剪 LLM KV Cache
            if current_cache_length > snapshot.llm_cache_length:
                self._truncate_llm_cache(snapshot.llm_cache_length)
                logger.debug(
                    "[Speculative] Truncated LLM cache: %d -> %d",
                    current_cache_length,
                    snapshot.llm_cache_length,
                )

            # 2. 恢复 Audio KV Cache（关键：从克隆的副本恢复）
            # 因为 streaming_generate 会将 audio_past_key_values 设为 None
            self.audio_past_key_values = snapshot.audio_past_key_values
            if snapshot.audio_past_key_values is not None:
                logger.debug(
                    "[Speculative] Restored audio cache: length=%d, checksum=%.6f",
                    snapshot.audio_cache_length,
                    snapshot.audio_cache_checksum or 0.0,
                )
            else:
                logger.debug("[Speculative] Audio cache restored to None")

            # 3. 恢复会话状态
            self.new_user_msg = snapshot.new_user_msg
            self.llm_generated = snapshot.llm_generated
            self.llm_generate_completed = snapshot.llm_generate_completed

            # 4. 恢复 Round 管理
            self._next_round_id = snapshot.next_round_id
            self._pending_round_id = snapshot.pending_round_id

            # 5. 截断 chunk 历史
            if current_history_length > snapshot.omni_chunk_history_length:
                self._omni_chunk_history = self._omni_chunk_history[: snapshot.omni_chunk_history_length]
                logger.debug(
                    "[Speculative] Truncated chunk history: %d -> %d",
                    current_history_length,
                    snapshot.omni_chunk_history_length,
                )

            # 6. 恢复 TTS 状态
            self.tts_last_turn_tokens = snapshot.tts_last_turn_tokens

            # 7. 恢复 streaming 处理器状态
            self.audio_chunk_idx = snapshot.audio_chunk_idx

            # 8. 恢复 mel processor 状态（关键！否则后续 prefill 会因帧数不匹配而失败）
            if (
                snapshot.mel_processor_snapshot is not None
                and hasattr(self, "processor")
                and self.processor is not None
            ):
                self.processor.restore_streaming_snapshot(snapshot.mel_processor_snapshot)
                mel_snap = snapshot.mel_processor_snapshot
                logger.info(
                    "[Speculative] Restored mel processor: buffer_len=%d, chunk_count=%d, "
                    "last_emitted_T=%d, total_samples=%d",
                    len(mel_snap.get("buffer", [])),
                    mel_snap.get("chunk_count", 0),
                    mel_snap.get("last_emitted_T", 0),
                    mel_snap.get("total_samples_processed", 0),
                )

            # 9. 恢复 RNG 状态（关键：确保 dithering 等随机操作的确定性）
            if snapshot.rng_state_cpu is not None:
                torch.set_rng_state(snapshot.rng_state_cpu)
                logger.debug("[Speculative] Restored CPU RNG state")
            if snapshot.rng_state_cuda is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(snapshot.rng_state_cuda, self.device)
                logger.debug("[Speculative] Restored CUDA RNG state")

            # 11. 清理生成过程中产生的临时状态
            if hasattr(self, "_streaming_generated_token_ids"):
                del self._streaming_generated_token_ids
            if hasattr(self, "_last_streaming_text"):
                del self._last_streaming_text

            # 12. 验证恢复后的状态
            restored_cache_length = self._get_kv_cache_length()
            if restored_cache_length != snapshot.llm_cache_length:
                logger.warning(
                    "[Speculative] LLM cache length mismatch after restore: expected=%d, actual=%d",
                    snapshot.llm_cache_length,
                    restored_cache_length,
                )

            # 验证 LLM cache checksum（如果有）
            if snapshot.llm_cache_checksum is not None and self.llm_past_key_values is not None:
                if hasattr(self.llm_past_key_values, "key_cache") and len(self.llm_past_key_values.key_cache) > 0:
                    current_checksum = self.llm_past_key_values.key_cache[0].sum().item()
                    if abs(current_checksum - snapshot.llm_cache_checksum) > 1e-3:
                        logger.warning(
                            "[Speculative] LLM cache checksum mismatch: expected=%.6f, actual=%.6f",
                            snapshot.llm_cache_checksum,
                            current_checksum,
                        )
                    else:
                        logger.debug(
                            "[Speculative] LLM cache checksum verified: %.6f",
                            current_checksum,
                        )

            # 11. 清除快照（只能恢复一次）
            self._speculative_snapshot = None

            logger.info(
                "[Speculative] Restore completed: llm_cache %d -> %d, elapsed=%.3fs",
                current_cache_length,
                snapshot.llm_cache_length,
                time.time() - snapshot.timestamp,
            )

            return True

        except Exception as e:
            import traceback

            logger.error("[Speculative] Failed to restore snapshot: %s", e)
            logger.error("[Speculative] Traceback: %s", traceback.format_exc())
            return False

    def has_speculative_snapshot(self) -> bool:
        return getattr(self, "_speculative_snapshot", None) is not None

    def clear_speculative_snapshot(self) -> None:
        if hasattr(self, "_speculative_snapshot"):
            self._speculative_snapshot = None

    def _truncate_llm_cache(self, target_length: int) -> None:
        if self.llm_past_key_values is None:
            return

        cache = self._ensure_dynamic_cache()
        if cache is None:
            return

        current_length = self._get_kv_cache_length(cache)
        if current_length <= target_length:
            return

        # 裁剪每一层的 cache
        for layer_idx in range(len(cache.key_cache)):
            if cache.key_cache[layer_idx].numel() > 0:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :target_length, :].contiguous()
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :target_length, :].contiguous()

        # 更新 cache 元数据
        cache.crop(target_length)
        cache._seen_tokens = target_length

        logger.debug("[Speculative] Truncated LLM cache: %d -> %d", current_length, target_length)

    # ============== 抢跑快照/恢复接口 结束 ==============

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
            self._speculative_snapshot = self._save_speculative_snapshot()

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

            # DEBUG: 打印生成开始时的状态
            _cache_len = self._get_kv_cache_length()
            _cache_sum = self.llm_past_key_values.key_cache[0].sum().item() if self.llm_past_key_values else 0
            # 检查 KV Cache 最后几个位置的值
            _k_last = (
                self.llm_past_key_values.key_cache[0][0, 0, -5:, :3].flatten().tolist()
                if self.llm_past_key_values
                else []
            )
            print(f"[DEBUG streaming_generate] cache_len={_cache_len}, cache_sum={_cache_sum:.6f}, k_last={_k_last}")

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

                # DEBUG: 打印第一个 chunk 生成的 token
                if chunk_idx == 0:
                    print(f"[DEBUG streaming_generate] first_chunk_tokens={output.chunk_token_ids.tolist()}")

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


class DuplexCapability:
    """双工能力组件 - 封装双工对话的全部逻辑
    
    采用组合模式，接受外部传入的 MiniCPMO 实例，不自己加载模型。
    
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
    
    def __init__(
        self,
        model: "MiniCPMO",
        generate_audio: bool = True,
        ls_mode: str = "explicit",
        device: str = "cuda",
        **kwargs,
    ):
        """初始化双工能力组件
        
        Args:
            model: 外部传入的 MiniCPMO 实例（已加载好的模型）
            generate_audio: 是否生成音频输出
            ls_mode: Listen/Speak 模式
            device: 设备
            **kwargs: 其他配置参数
        """
        # 使用外部传入的模型（不自己加载！）
        self.model = model
        self.device = device
        
        self.session_logs = []
        self.session_start_time = None
        self.log_file_path = None
        
        self.generate_audio = generate_audio
        self.ls_mode = ls_mode
        
        # 使用 model 的 processor 和 tokenizer
        if not hasattr(self.model, "processor") or self.model.processor is None:
            self.model.processor = MiniCPMOProcessor.from_pretrained(
                self.model.config._name_or_path, trust_remote_code=True
            )
        self.processor = self.model.processor
        self.tokenizer = self.processor.tokenizer
        
        self.break_event = threading.Event()
        self.session_stop_event = threading.Event()
        
        # llm generation_config
        self.max_new_speak_tokens_per_chunk = kwargs.get("max_new_speak_tokens_per_chunk", 20)
        self.text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.05)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_k = kwargs.get("top_k", 20)
        self.top_p = kwargs.get("top_p", 0.8)
        self.text_repetition_window_size = kwargs.get("text_repetition_window_size", 512)
        self.listen_prob_scale = kwargs.get("listen_prob_scale", 1.0)
        self.force_listen_count = kwargs.get("force_listen_count", 0)
        
        # tts generation_config
        tts_temp_value = kwargs.get("tts_temperature", 0.8)
        self.tts_temperature = torch.tensor([tts_temp_value], dtype=torch.float, device=self.device)
        self.tts_repetition_penalty = kwargs.get("tts_repetition_penalty", 1.05)
        
        # stream config
        self.CHUNK_MS = kwargs.get("chunk_ms", 1000)
        self.FIRST_CHUNK_MS = kwargs.get("first_chunk_ms", 1035)
        self.CNN_REDUNDANCY_MS = kwargs.get("cnn_redundancy_ms", 20)
        self.SAMPLE_RATE = kwargs.get("sample_rate", 16000)
        
        self.model.CHUNK_MS = self.CHUNK_MS
        self.model.FIRST_CHUNK_MS = self.FIRST_CHUNK_MS
        self.model.CNN_REDUNDANCY_MS = self.CNN_REDUNDANCY_MS
        self.model.SAMPLE_RATE = self.SAMPLE_RATE
        
        # special tokens
        self.unit_token_id = self.tokenizer.convert_tokens_to_ids("<unit>")
        self.image_start_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids("</image>")
        self.slice_start_token_id = self.tokenizer.convert_tokens_to_ids("<slice>")
        self.slice_end_token_id = self.tokenizer.convert_tokens_to_ids("</slice>")
        
        self.listen_token_id = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        self.speak_token_id = self.tokenizer.convert_tokens_to_ids("<|speak|>")
        self.tts_bos_token_id = self.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        self.tts_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|tts_eos|>")
        
        self.chunk_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        self.chunk_tts_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        self.turn_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|turn_eos|>")
        
        self.chunk_terminator_token_ids = [self.listen_token_id, self.chunk_eos_token_id, self.chunk_tts_eos_token_id]
        self.turn_terminator_token_ids = [self.turn_eos_token_id]
        self.chunk_speak_token_ids = [self.speak_token_id]
        
        self.tts_pad_id = self.tokenizer.convert_tokens_to_ids("<|tts_pad|>")
        bad_token_ids = getattr(self.tokenizer, "bad_token_ids", [])
        self.forbidden_token_ids = [self.tts_pad_id] + list(bad_token_ids)
        # self.forbidden_token_ids = [] + list(bad_token_ids)
        
        self.decoder = StreamDecoder(
            llm=self.model.llm, tokenizer=self.tokenizer, forbidden_token_ids=self.forbidden_token_ids
        )
        
        # 滑窗模式: "off" / "basic" / "context"
        sliding_window_mode = kwargs.get("sliding_window_mode", "off")
        
        # 不带 Context 的滑窗参数
        basic_window_high_tokens = kwargs.get("basic_window_high_tokens", 4000)
        basic_window_low_tokens = kwargs.get("basic_window_low_tokens", 3500)
        
        # 带 Context 的滑窗参数
        context_previous_max_tokens = kwargs.get("context_previous_max_tokens", 500)
        context_max_units = kwargs.get("context_max_units", 24)
        
        self.decoder.set_window_config(
            DuplexWindowConfig(
                sliding_window_mode=sliding_window_mode,
                basic_window_high_tokens=basic_window_high_tokens,
                basic_window_low_tokens=basic_window_low_tokens,
                context_previous_max_tokens=context_previous_max_tokens,
                context_max_units=context_max_units,
            )
        )
        # 根据 mode 设置滑窗开关
        window_enabled = sliding_window_mode != "off"
        self.decoder.set_window_enabled(window_enabled)
        logger.info(
            "[DuplexCapability] Sliding window: mode=%s, high=%d, low=%d, prev_max=%d, max_units=%d",
            sliding_window_mode,
            basic_window_high_tokens,
            basic_window_low_tokens,
            context_previous_max_tokens,
            context_max_units,
        )
        
        self.tts_logits_processors = None
        self.tts_eos_token = None
        if self.generate_audio:
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
        
        logger.info("[DuplexCapability] 初始化完成")

    def set_break_event(self):
        self.break_event.set()

    def clear_break_event(self):
        self.break_event.clear()

    def set_session_stop(self):
        self.session_stop_event.set()
        self.break_event.set()

    def clear_session_stop(self):
        self.session_stop_event.clear()

    def is_break_set(self) -> bool:
        return self.break_event.is_set()

    def is_session_stop_set(self) -> bool:
        return self.session_stop_event.is_set()

    def _init_token2wav_cache(self, prompt_wav_path: str):
        self.model.tts.audio_tokenizer.cache = None
        flow_cache, hift_cache = self.model.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)
        self.flow_cache_base = torch_clone_recursive(flow_cache)
        self.hift_cache_base = torch_clone_recursive(hift_cache)
        self.pre_lookahead = int(self.model.tts.audio_tokenizer.flow.pre_lookahead_len)
        self.token2wav_initialized = True

    def _reset_token2wav_for_new_turn(self):
        if self.token2wav_initialized:
            self.model.tts.audio_tokenizer.stream_cache = torch_clone_recursive(self.flow_cache_base)
            self.model.tts.audio_tokenizer.hift_cache_dict = torch_clone_recursive(self.hift_cache_base)
            self.token2wav_buffer = [4218] * 3  # silence token prefix

    def _reset_streaming_state(self):
        self.audio_chunk_idx = 0
        self.current_turn_ended = True
        self.speak_count = 0
        self.res_ids = []
        self.total_ids = []
        self.total_hidden = []

        # TTS state
        self.tts_text_start_pos = 0
        self.tts_past_key_values = None
        self.tts_current_turn_start_time = None

        # token2wav state
        self.token2wav_initialized = False
        self.token2wav_buffer = []
        self.flow_cache_base = None
        self.hift_cache_base = None

        # Audio prefill state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.pending_logits: Optional[torch.Tensor] = None
        self.current_mode: Optional[str] = None

        # Force listen state
        self._streaming_generate_count = 0

        # Deferred finalize state（必须清理，否则上一个 session 异常退出后
        # _pending_finalize 残留，导致新 session 的第一个 prefill 报错）
        self._pending_finalize = None

        # 禁止连续 chunk 解码 <|tts_pad|>
        self._last_chunk_had_tts_pad = False

        # Schema tracking: 记录完整的 prefill + generate token 序列
        # prefill_schema_tokens: 每个元素是一个 unit 的 prefill token 列表
        # 格式: [[unit0_prefill_tokens], [unit1_prefill_tokens], ...]
        self.prefill_schema_tokens = []
        self._current_unit_prefill_tokens = []

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

        self.session_start_time = time.time()

        self._reset_streaming_state()
        self.decoder.reset()

        self.model.init_streaming_processor()

        if prompt_wav_path is not None and prompt_wav_path and self.generate_audio:
            self._init_token2wav_cache(prompt_wav_path)
            self._reset_token2wav_for_new_turn()

        # Prefill system prompt prefix
        if prefix_system_prompt:
            tokens = self.tokenizer.encode(prefix_system_prompt, add_special_tokens=False)
            for token_id in tokens:
                self.decoder.feed(self.decoder.embed_token(token_id))

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

                # 现在 feed suffix
                for token_id in suffix_token_ids:
                    self.decoder.feed(self.decoder.embed_token(token_id))

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
                    for token_id in tokens:
                        self.decoder.feed(self.decoder.embed_token(token_id))
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
                    if len(_chunk_text) >= 22:
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

    def get_session_schema(self, include_embeddings: bool = True) -> str:
        """获取当前 session 的完整 schema（包含 prefill 和 generate 阶段）

        Args:
            include_embeddings: 是否包含 embedding 占位符 (如 [img_embed_64], [audio_embed_50])

        Returns:
            完整的 schema 字符串，每个 unit 的格式为:
            <unit><image>[img_embed_64]</image>[audio_embed_50]<|listen|或|speak|>生成内容</unit>
        """
        if not hasattr(self, "prefill_schema_tokens") or not hasattr(self, "total_ids"):
            return ""

        # 获取 </unit> token id 用于分割 generate tokens
        unit_end_token_id = self.tokenizer.convert_tokens_to_ids("</unit>")

        # 分割 generate tokens 为每个 unit
        generate_units = []
        current_unit = []
        for tid in self.total_ids:
            current_unit.append(tid)
            if tid == unit_end_token_id:
                generate_units.append(current_unit)
                current_unit = []

        # 构建完整 schema
        full_schema_parts = []
        num_units = max(len(self.prefill_schema_tokens), len(generate_units))

        for unit_idx in range(num_units):
            unit_schema = ""

            # Prefill 部分
            if unit_idx < len(self.prefill_schema_tokens):
                prefill_tokens = self.prefill_schema_tokens[unit_idx]
                for item in prefill_tokens:
                    if isinstance(item, tuple):
                        # 元组表示 embedding: ("img", dim) 或 ("audio", dim)
                        embed_type, embed_dim = item
                        if include_embeddings:
                            unit_schema += f"[{embed_type}_embed_{embed_dim}]"
                    else:
                        # 正常 token ID
                        unit_schema += self.tokenizer.decode([item], skip_special_tokens=False)

            # Generate 部分
            if unit_idx < len(generate_units):
                unit_schema += self.tokenizer.decode(generate_units[unit_idx], skip_special_tokens=False)

            full_schema_parts.append(unit_schema)

        return "".join(full_schema_parts)

    def get_unit_schemas(self, include_embeddings: bool = True) -> list:
        """获取每个 unit 的 schema 列表

        Returns:
            每个 unit 的 schema 字符串列表
        """
        if not hasattr(self, "prefill_schema_tokens") or not hasattr(self, "total_ids"):
            return []

        unit_end_token_id = self.tokenizer.convert_tokens_to_ids("</unit>")

        # 分割 generate tokens 为每个 unit
        generate_units = []
        current_unit = []
        for tid in self.total_ids:
            current_unit.append(tid)
            if tid == unit_end_token_id:
                generate_units.append(current_unit)
                current_unit = []

        # 构建每个 unit 的 schema
        unit_schemas = []
        num_units = max(len(self.prefill_schema_tokens), len(generate_units))

        for unit_idx in range(num_units):
            unit_schema = ""

            # Prefill 部分
            if unit_idx < len(self.prefill_schema_tokens):
                prefill_tokens = self.prefill_schema_tokens[unit_idx]
                for item in prefill_tokens:
                    if isinstance(item, tuple):
                        # 元组表示 embedding: ("img", dim) 或 ("audio", dim)
                        embed_type, embed_dim = item
                        if include_embeddings:
                            unit_schema += f"[{embed_type}_embed_{embed_dim}]"
                    else:
                        # 正常 token ID
                        unit_schema += self.tokenizer.decode([item], skip_special_tokens=False)

            # Generate 部分
            if unit_idx < len(generate_units):
                unit_schema += self.tokenizer.decode(generate_units[unit_idx], skip_special_tokens=False)

            unit_schemas.append(unit_schema)

        return unit_schemas

    def _convert_results_to_tts_input(self, results):
        """convert LLM hidden states to TTS input"""
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
        for hidden in results:
            llm_tokens.append(hidden[0])
            llm_hidden.append(hidden[1].squeeze(0))

        llm_tokens_tensor = torch.Tensor(llm_tokens).to(self.device, dtype=torch.long)
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

        tts_embeds = torch.cat([tts_embeds, audio_bos], dim=0)
        return tts_embeds.unsqueeze(0)

    def _generate_waveform_from_tokens(
        self,
        new_tokens: torch.Tensor,
        prompt_wav_path: Optional[str],
        is_last_chunk: bool = False,
        force_flush: bool = False,
    ) -> Optional[np.ndarray]:
        """从 audio tokens 生成波形"""
        if not self.token2wav_initialized:
            print("⚠️ token2wav 未初始化")
            return None

        CHUNK_SIZE = 25

        # 将新 tokens 添加到 buffer
        token_ids = torch.reshape(new_tokens, (-1,)).tolist()
        self.token2wav_buffer += token_ids

        # 检测是否有 chunk_eos token
        has_chunk_eos = any(tid in self.chunk_terminator_token_ids for tid in token_ids)

        self._log(
            "AUDIO",
            f"Token2Wav buffer size: {len(self.token2wav_buffer)}, new tokens: {len(token_ids)}, has_chunk_eos: {has_chunk_eos}",
        )

        pcm_bytes_list = []

        # process enough tokens
        # if there is chunk_eos, try to flush more content
        if has_chunk_eos or force_flush:
            # when there is chunk_eos, try to flush more content
            while len(self.token2wav_buffer) >= self.pre_lookahead + 5:  # at least keep some lookahead
                chunk_to_process = min(CHUNK_SIZE + self.pre_lookahead, len(self.token2wav_buffer))
                pcm_bytes = self.model.tts.audio_tokenizer.stream(
                    self.token2wav_buffer[:chunk_to_process],
                    prompt_wav=prompt_wav_path,
                )
                pcm_bytes_list.append(pcm_bytes)
                self.token2wav_buffer = self.token2wav_buffer[min(CHUNK_SIZE, chunk_to_process - self.pre_lookahead) :]
        else:
            while len(self.token2wav_buffer) >= CHUNK_SIZE + self.pre_lookahead:
                pcm_bytes = self.model.tts.audio_tokenizer.stream(
                    self.token2wav_buffer[: CHUNK_SIZE + self.pre_lookahead],
                    prompt_wav=prompt_wav_path,
                )
                pcm_bytes_list.append(pcm_bytes)
                self.token2wav_buffer = self.token2wav_buffer[CHUNK_SIZE:]

        # if is the last chunk, flush remaining tokens
        if is_last_chunk and len(self.token2wav_buffer) > 0:
            self._log("AUDIO", f"Flushing final {len(self.token2wav_buffer)} tokens")
            pcm_bytes = self.model.tts.audio_tokenizer.stream(
                self.token2wav_buffer,
                prompt_wav=prompt_wav_path,
                last_chunk=True,
            )
            pcm_bytes_list.append(pcm_bytes)
            self.token2wav_buffer = []

        if not pcm_bytes_list:
            return None

        # merge PCM and convert to numpy array (24kHz, int16 -> float32)
        all_pcm = b"".join(pcm_bytes_list)
        if len(all_pcm) == 0:
            self._log("AUDIO", "No audio bytes generated")
            return None

        pcm_np = np.frombuffer(all_pcm, dtype="<i2")
        audio_waveform = pcm_np.astype(np.float32) / 32768.0

        # left pad with zeros if audio is less than 1 second (24kHz), skip for last chunk
        min_samples = 24000  # 1 second at 24kHz
        if not is_last_chunk and len(audio_waveform) < min_samples:
            pad_length = min_samples - len(audio_waveform)
            audio_waveform = np.pad(audio_waveform, (pad_length, 0), mode="constant", constant_values=0)
            self._log("AUDIO", f"Left padded {pad_length} samples ({pad_length/24000:.3f}s) to reach 1s minimum")

        # record generated audio information
        duration_sec = len(audio_waveform) / 24000
        self._log(
            "AUDIO",
            f"Generated audio: {duration_sec:.2f}s @ 24kHz, {len(pcm_bytes_list)} chunks, remaining buffer: {len(self.token2wav_buffer)} tokens",
        )

        return audio_waveform

    @staticmethod
    def _generate_silence_waveform(duration_sec: float = 1.0) -> np.ndarray:
        """generate silence waveform (24kHz)"""
        sample_rate = 24000
        num_samples = int(duration_sec * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)

    def get_generated_text(self) -> str:
        return self.tokenizer.decode(self.res_ids)

    def get_current_time(self) -> int:
        return self.audio_chunk_idx

    def _log(self, event_type: str, message: str, data: Optional[dict] = None):
        if self.session_start_time is None:
            self.session_start_time = time.time()

        timestamp = time.time() - self.session_start_time
        log_entry = {"timestamp": timestamp, "type": event_type, "message": message}
        if data:
            log_entry["data"] = data

        self.session_logs.append(log_entry)

        prefix = f"[{timestamp:6.3f}s]"
        if event_type == "FEED":
            print(f"{prefix} [FEED] {message}")
        elif event_type == "DECODE":
            print(f"{prefix} [DECODE] {message}")
        elif event_type == "SYS":
            print(f"{prefix} [SYS] {message}")
        elif event_type == "AUDIO":
            print(f"{prefix} [AUDIO] {message}")
        else:
            print(f"{prefix} [{event_type}] {message}")


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        adaptive=False,
        max_size=(70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size, device="cpu"):
        if is_deepspeed_zero3_enabled():
            device = "cuda"
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_sizes=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        try:
            # compatible old transformers
            from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES

            self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
                layer_idx=layer_idx,
            )
        except:
            from transformers.models.whisper.modeling_whisper import WhisperAttention

            self.self_attn = WhisperAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
                layer_idx=layer_idx,
            )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference
class MiniCPMWhisperEncoder(WhisperEncoder):

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniCPMWhisperEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)]
        )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        use_extra_context: Optional[bool] = False,
        prefix_extra_frames: Optional[int] = 1,
        suffix_extra_frames: Optional[int] = 1,
        return_debug: Optional[bool] = False,
        cnn_min_length: Optional[int] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 初始化debug字典
        debug_info = {} if return_debug else None

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        # 可选：将短输入pad到最小长度以确保CNN计算一致性
        original_length = input_features.shape[2]
        padded_for_cnn = False
        if cnn_min_length is not None and original_length < cnn_min_length:
            padded_features = torch.zeros(
                input_features.shape[0],
                input_features.shape[1],
                cnn_min_length,
                dtype=input_features.dtype,
                device=input_features.device,
            )
            padded_features[:, :, :original_length] = input_features
            input_features = padded_features
            padded_for_cnn = True
            if return_debug:
                debug_info["padded_for_cnn"] = True
                debug_info["original_length"] = original_length
                debug_info["padded_length"] = cnn_min_length

        if return_debug:
            debug_info["input_features"] = input_features.clone()

        conv1_output = self.conv1(input_features)
        if return_debug:
            debug_info["after_conv1"] = conv1_output.clone()

        inputs_embeds = nn.functional.gelu(conv1_output)
        if return_debug:
            debug_info["after_gelu1"] = inputs_embeds.clone()

        conv2_output = self.conv2(inputs_embeds)
        if return_debug:
            debug_info["after_conv2"] = conv2_output.clone()

        inputs_embeds = nn.functional.gelu(conv2_output)
        if return_debug:
            debug_info["after_gelu2"] = inputs_embeds.clone()

        # 如果之前进行了padding，现在需要移除padding的影响
        if padded_for_cnn:
            # Conv1: stride=1, 输出长度=输入长度
            # Conv2: stride=2, 输出长度=(输入长度+1)//2
            actual_cnn_output_length = (original_length + 1) // 2
            inputs_embeds = inputs_embeds[:, :, :actual_cnn_output_length]
            if return_debug:
                debug_info["after_unpad"] = inputs_embeds.clone()

        # 如果使用额外上下文，CNN操作后需要移除冗余帧
        # conv2的stride=2，所以输入的冗余帧在输出中会变成一半（向上取整）
        if use_extra_context:
            # 输入有prefix_extra_frames前缀帧和suffix_extra_frames后缀帧
            # conv2 stride=2，输出长度 = ceil(输入长度 / 2)
            # 对于2帧冗余，输出是1帧（ceil(2/2) = 1）
            prefix_to_remove = (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0  # 向上取整
            suffix_to_remove = (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0  # 向上取整

            # 移除前后冗余帧 (batch, channels, time)
            if prefix_to_remove > 0:
                inputs_embeds = inputs_embeds[:, :, prefix_to_remove:]
            if 0 < suffix_to_remove < inputs_embeds.shape[2]:
                inputs_embeds = inputs_embeds[:, :, :-suffix_to_remove]

        if return_debug and use_extra_context:
            debug_info["after_redundancy_removal"] = inputs_embeds.clone()

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        if return_debug:
            debug_info["after_permute"] = inputs_embeds.clone()

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(DynamicCache.from_legacy_cache(past_key_values), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            else:
                pass
            past_key_values_length = past_key_values.self_attention_cache.get_usable_length(inputs_embeds.shape[1])
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning("seems the audio is longer than 30s. repeating the last part of the audio")
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        if return_debug:
            debug_info["positional_embedding"] = embed_pos.clone()
            debug_info["past_key_values_length"] = past_key_values_length

        hidden_states = inputs_embeds + embed_pos
        if return_debug:
            debug_info["after_add_pos_embed"] = hidden_states.clone()

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if return_debug:
            debug_info["after_dropout"] = hidden_states.clone()

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                        past_key_values,
                        use_cache,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if return_debug and idx < 3:  # 只记录前3层以节省内存
                    debug_info[f"after_layer_{idx}"] = hidden_states.clone()

            if use_cache:
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if return_debug:
            debug_info["after_layer_norm"] = hidden_states.clone()

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            result = tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            if return_debug:
                return result, debug_info
            return result
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )

        if return_debug:
            return result, debug_info
        return result


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_dim = config.llm_hidden_size
        self.out_dim = config.hidden_size
        self.intermediate_size = config.llm_intermediate_size
        self.gate_proj = nn.Linear(self.in_dim, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.in_dim, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_dim, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


@dataclass
class MiniCPMTTSGenerationOutput(ModelOutput):
    """
    Output class for MiniCPMTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.
    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_input_ids: Optional[torch.LongTensor] = None
    finished: bool = None


def make_streaming_chunk_mask_inference(
    tts_text_scope: List[int],
    tts_text_mask: torch.Tensor,
    streaming_audio_chunk_size: int = 50,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    max_sequence_length: int = 4096,
):
    """
    Example:
    Input sequence:
    [t1, t2, t3, t4, t5, [Ptts], a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ...]
    Output 4D causal mask:
    ------- text positions -------
    [0] <- here is [Stts]
    [0,   0] <- here is [spk_emb] * N
    [0,   0,   0]
    [0,   0,   0,    0]
    [0,   0,   0,    0,    0]
    ------- audio positions --------
    [0,    0, -inf, -inf, -inf, 0] <- here is [Ptts], [Ptts]'s last hidden state should predict the first audio token
                                v- here is [Ptts]
    [0,    0, -inf, -inf, -inf, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0, 0, 0] # end of first 1s audio chunk
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    # Create a complete attention mask for input embeds [batch_size, seq_len], without considering audio mask as audio is always at the end

    assert tts_text_mask.dtype == torch.int8

    padding_mask = torch.ones(max_sequence_length, dtype=torch.int8, device=device)
    padding_mask[tts_text_scope[0] : tts_text_scope[1]] = tts_text_mask

    # Initialize a standard upper triangular causal mask
    min_dtype = torch.finfo(dtype).min

    causal_mask = torch.full(
        (max_sequence_length, max_sequence_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if max_sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    else:
        raise ValueError("max_sequence_length of tts could not be 1.")

    # For each data sample
    audio_token_start = tts_text_scope[1]
    audio_duration = max_sequence_length - tts_text_scope[1]

    # Record which text chunk the current audio chunk can see up to
    text_pivot = 0
    num_valid_text_tokens = torch.sum(tts_text_mask).item() - 1  # [Ptts] excluded
    # How many audio chunks are in total, the num of buckets should be smaller as possible

    num_text_tokens_per_audio_chunk = 10

    # For each chunk of audio
    for chunk_idx in range(math.ceil(audio_duration / streaming_audio_chunk_size)):
        audio_chunk_start = audio_token_start + chunk_idx * streaming_audio_chunk_size
        audio_chunk_end = audio_token_start + (chunk_idx + 1) * streaming_audio_chunk_size
        # New text seen by this new audio chunk
        new_text_this_chunk = num_text_tokens_per_audio_chunk
        # The right bound of visible text tokens
        text_pivot = min(new_text_this_chunk + text_pivot, num_valid_text_tokens)
        # Mask all text chunks after the visible ones
        # -> [text_pivot, len(tts_text_scope)-1] excluding [Ptts]
        # print("audio_chunk_start-1", audio_chunk_start-1, "audio_chunk_end-1", audio_chunk_end-1, "tts_text_scope[0] + text_pivot", tts_text_scope[0] + text_pivot, "tts_text_scope[1] - 1", tts_text_scope[1] - 1)
        causal_mask[
            audio_chunk_start - 1 : audio_chunk_end - 1,
            # tts_text_scope[0] + text_pivot: tts_text_scope[1],
            tts_text_scope[0] + text_pivot : tts_text_scope[1] - 1,
        ] = min_dtype

    # Mask the padding parts in tts_text_masks (no position will attend to it)
    causal_mask[:, padding_mask == 0] = min_dtype

    # Add extra dimensions, [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


class MiniCPMTTS(PreTrainedModel):
    config_class = MiniCPMTTSConfig

    def __init__(self, config: MiniCPMTTSConfig, audio_tokenizer: None):
        super().__init__(config)

        self.use_llm_hidden_state = config.use_llm_hidden_state

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_min = config.streaming_text_chunk_min
        self.streaming_text_chunk_max = config.streaming_text_chunk_max
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        # streaming tts
        self.streaming_text_chunk_size = config.streaming_text_chunk_max
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        self.interleaved = config.interleaved
        self.attention_type = config.attention_type
        self.recomputed_chunks = config.recomputed_chunks
        
        # Two different window size concepts:
        # 1. chunk_window_size: number of chunks for sliding_recompute mode (default 2)
        # 2. token_window_size: number of tokens for sliding_window mode (default 300)
        self.chunk_window_size = config.window_size  # chunk-level window for sliding_recompute
        self.token_window_size = config.streaming_sliding_window_audio_window_size  # token-level window for sliding_window
        
        # Legacy aliases (for backward compatibility with existing code)
        self.window_size = self.chunk_window_size  # used in generate_streaming for sliding_recompute
        self.sliding_window_size = self.token_window_size  # used in TTSStreamingGenerator for sliding_window
        
        if self.attention_type == "sliding_recompute" and self.chunk_window_size <= self.recomputed_chunks:
            raise ValueError(
                f"sliding_recompute requires chunk_window_size > recomputed_chunks, "
                f"but got chunk_window_size={self.chunk_window_size} and recomputed_chunks={self.recomputed_chunks}"
            )

        if config.backbone_model == "llama":
            model_config = LlamaConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_hidden_layers=config.num_hidden_layers,
                num_key_value_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                attn_implementation=config.attn_implementation,
            )

            self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)

            model = LlamaModel(model_config)
            self.model = model
        else:
            raise ValueError(f"Unsupported backbone model: {config.backbone_model}")

        self.projector_spk = self.create_projector(config)
        self.projector_semantic = self.create_projector(config)

        self.audio_tokenizer = audio_tokenizer

        self.emb_code = nn.ModuleList(
            [nn.Embedding(config.num_audio_tokens, config.hidden_size) for _ in range(config.num_vq)]
        )

        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )

        self.condition_type = config.condition_type

        return

    @staticmethod
    def create_projector(config):
        if config.projector_type == "mlp":
            return MultiModalProjector(config.llm_dim, config.hidden_size)
        elif config.projector_type == "minicpm":
            return MiniCPMMLP(config)
        elif config.projector_type == "default":
            return nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        else:
            raise ValueError(f"Unsupported projector type: {config.projector_type}")

    # non-streaming
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        min_new_token=50,
        max_new_token=2048,
        show_tqdm=True,
        streaming=False,
        text_lengths=None,
        sampling_params: TTSSamplingParams = TTSSamplingParams(),
    ):
        temperature = torch.tensor(
            [sampling_params.temperature] * self.config.num_vq,
            dtype=torch.float,
            device=self.device,
        )
        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        condition_length = inputs_embeds.shape[1]
        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        if streaming:
            raise NotImplementedError("this kind of streaming is not supported yet")

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        past_key_values = None

        for t in range(max_new_token):
            audio_bos = False
            # If this is the first audio token, the case is special
            if t == 0:
                audio_bos = True
                inputs_embeds = inputs_embeds
                position_ids = torch.tensor(
                    list(range(0, condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                if streaming:
                    raise NotImplementedError("this kind of streaming is not supported yet")
                else:
                    causal_mask_4d = None

            else:
                code_emb = []
                for q in range(self.num_vq):
                    x = self.emb_code[q](new_tokens[:, t - 1 : t, q])
                    code_emb.append(x)

                inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor([condition_length + t - 1], dtype=torch.long, device=self.device).unsqueeze(
                    0
                )

                if streaming:
                    raise NotImplementedError("this kind of streaming is not supported yet")
                else:
                    causal_mask_4d = None

            if self.config.backbone_model == "llama":
                outputs: BaseModelOutputWithPast = self.model(
                    position_ids=position_ids,
                    cache_position=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask_4d,
                    use_cache=True,
                    output_attentions=False,
                    # return_dict=True,  # Add this to ensure returns dict with past_key_values
                )
            else:
                raise ValueError(f"Unsupported backbone model: {self.config.backbone_model}")

            del position_ids
            del inputs_embeds

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            logits = logits[:, -1].float()

            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))

            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens
                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

                del logits_token

            if t < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits

            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

            del scores

            idx_next = idx_next.view(-1, self.num_vq)

            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        genrated_input_ids = new_tokens[:, 0:t, :]

        return MiniCPMTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=None,  # for update purpose
            past_key_values=None,  # for update purpose
            past_input_ids=None,  # for update purpose
            finished=finish.all(),
        )

    # fake streaming
    @torch.inference_mode()
    def generate_mock_legacy_streaming(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        min_new_token=50,
        max_new_token=2048,
        show_tqdm=True,
        streaming=False,
        text_lengths=None,
        sampling_params: TTSSamplingParams = TTSSamplingParams(),
        valid_text_length=None,
    ):
        assert valid_text_length is not None, "valid_text_length should be not None"

        tts_text_scope = [0, inputs_embeds.shape[1]]
        tts_text_mask = torch.zeros(inputs_embeds.shape[1], dtype=torch.int8, device=inputs_embeds.device)
        tts_text_mask[0:valid_text_length] = 1
        tts_text_mask[-1] = 1  # [Ptts]

        streaming_mask_4d_full = make_streaming_chunk_mask_inference(
            tts_text_scope=tts_text_scope,
            tts_text_mask=tts_text_mask,
            dtype=torch.bfloat16,
            device=self.device,
            streaming_audio_chunk_size=50,
            max_sequence_length=4096,
        )

        temperature = torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.device)
        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        condition_length = inputs_embeds.shape[1]
        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        past_key_values = None

        for t in range(max_new_token):
            audio_bos = False
            if t == 0:
                audio_bos = True
                inputs_embeds = inputs_embeds
                position_ids = torch.tensor(
                    list(range(0, condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                causal_mask_4d = streaming_mask_4d_full[:, :, :condition_length, :condition_length]
            else:
                code_emb = []
                for q in range(self.num_vq):
                    x = self.emb_code[q](new_tokens[:, t - 1 : t, q])
                    code_emb.append(x)

                inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor([condition_length + t - 1], dtype=torch.long, device=self.device).unsqueeze(
                    0
                )

                causal_mask_4d = streaming_mask_4d_full[
                    :,
                    :,
                    condition_length + t : condition_length + t + 1,
                    : condition_length + t,
                ]

                # get length of past_key_values
                past_key_values_length = past_key_values[0][0].shape[2]

                assert causal_mask_4d.shape[-1] == (past_key_values_length + 1)

            if self.config.backbone_model == "llama":
                outputs: BaseModelOutputWithPast = self.model(
                    position_ids=position_ids,
                    cache_position=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask_4d,
                    use_cache=True,
                    output_attentions=False,
                    # return_dict=True,  # Add this to ensure returns dict with past_key_values
                )
            else:
                raise ValueError(f"Unsupported backbone model: {self.config.backbone_model}")

            del position_ids
            del inputs_embeds

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            logits = logits[:, -1].float()
            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))
            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

                del logits_token

            if t < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits
            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

            del scores

            idx_next = idx_next.view(-1, self.num_vq)
            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        genrated_input_ids = new_tokens[:, 0:t, :]

        return MiniCPMTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=None,  # for update purpose
            past_key_values=None,  # for update purpose
            past_input_ids=None,  # for update purpose
            finished=finish.all(),
        )

    # non-streaming, interleave
    @torch.inference_mode()
    def generate_chunk(
        self,
        inputs_embeds: torch.Tensor,
        temperature: torch.Tensor,
        repetition_penalty: float,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        max_new_token=500,
        min_new_tokens=0,
        past_key_values=None,
        logits_processors=None,
        text_start_pos=None,
    ):
        """For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
        |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
        where the last position is the audio BOS token.
        So, the first iteration in generation directly forward the model with inputs_embeds, and
        the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
        """
        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens, repetition_penalty=repetition_penalty
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        condition_length = inputs_embeds.shape[1]

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        for t in range(max_new_token):
            audio_bos = False

            # If this is the first audio token, the case is special
            if t == 0:
                audio_bos = True
                inputs_embeds_ = inputs_embeds
                position_ids = torch.tensor(
                    list(range(text_start_pos, text_start_pos + condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
            else:
                # Generate the following audio tokens, it is applicable to all other cases, including second and the following calling of `generate`
                inputs_embeds_ = self.emb_code[0](new_tokens[:, t - 1 : t, 0])

                position_ids = torch.tensor(
                    [text_start_pos + condition_length + t - 1],  # 把上一个token prefill进去
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

            outputs: BaseModelOutputWithPast = self.model(
                position_ids=position_ids,
                # cache_position=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds_,
                use_cache=True,
                output_attentions=False,
                # return_dict=True,  # Add this to ensure returns dict with past_key_values
            )

            del position_ids
            del inputs_embeds_

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            logits = logits[:, -1].float()
            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))

            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                del logits_token

            if force_no_stop or t < min_new_tokens:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)
            del logits

            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)
            del scores

            idx_next = idx_next.view(-1, self.num_vq)

            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        # 最新生成的那个token不在此次返回的范围内，如果是eos token，不返回，如果是其他正常的token，也不返回。正常！
        genrated_input_ids = new_tokens[:, 0:t, :]

        return genrated_input_ids, past_key_values

    @torch.inference_mode()
    def interleaved_generate(
        self,
        spk_embeds: torch.Tensor,
        conditions: List[torch.Tensor],
        temperature: torch.Tensor,
        repetition_penalty: float,
        eos_token: Union[int, torch.Tensor],
        **kwargs,
    ):
        """
        For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
        |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
        where the last position is the audio BOS token.
        So, the first iteration in generation directly forward the model with inputs_embeds, and the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
        """
        temperature = torch.tensor([temperature], dtype=torch.float, device=self.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=repetition_penalty,
        )

        eos_token = eos_token.to(conditions[0].device)

        num_chunks = len(conditions)
        text_start_pos = 0
        last_window_size = 0
        past_key_values = None

        for idx in range(num_chunks):
            condition = conditions[idx].to(conditions[0].device)
            if self.attention_type == "sliding_recompute":
                recomputed_conditions = []

                if (
                    idx >= self.window_size
                    and (idx - self.recomputed_chunks) % (self.window_size - self.recomputed_chunks) == 0
                ):
                    for i in range(self.recomputed_chunks):
                        recomputed_conditions.append(conditions[idx - self.recomputed_chunks + i])
                        recomputed_conditions.append(
                            self.emb_code[0](generated_tokens[-self.recomputed_chunks + i][:, :, 0])
                        )
                    recomputed_conditions.append(condition)
                    condition = torch.cat(recomputed_conditions, dim=1)

                    text_start_pos = 0
                    new_tokens, old_kv = self.generate_chunk(
                        inputs_embeds=condition,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        eos_token=eos_token,
                        force_no_stop=False,
                        max_new_token=500,
                        past_key_values=None,
                        logits_processors=logits_processors,
                        text_start_pos=text_start_pos,
                    )

                else:
                    new_tokens, old_kv = self.generate_chunk(
                        inputs_embeds=condition,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        eos_token=eos_token,
                        force_no_stop=False,
                        max_new_token=500,
                        past_key_values=past_key_values,
                        logits_processors=logits_processors,
                        text_start_pos=text_start_pos,
                    )
            else:
                new_tokens, old_kv = self.generate_chunk(
                    inputs_embeds=condition,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token=eos_token,
                    force_no_stop=False,
                    max_new_token=500,
                    past_key_values=past_key_values,
                    logits_processors=logits_processors,
                    text_start_pos=text_start_pos,
                )

            past_key_values = []
            if self.attention_type == "sliding_window" and idx >= 1:
                for layer_idx in range(len(old_kv)):
                    past_key_values.append(
                        (
                            old_kv[layer_idx][0][:, :, last_window_size:, :],
                            old_kv[layer_idx][1][:, :, last_window_size:, :],
                        )
                    )
            else:
                past_key_values = old_kv

            last_window_size = condition.shape[1] + new_tokens.shape[1]
            text_start_pos += last_window_size

            if idx == 0:
                generated_tokens = [new_tokens]
            else:
                generated_tokens.append(new_tokens)

        return MiniCPMTTSGenerationOutput(new_ids=torch.cat(generated_tokens, dim=1), finished=True)


class CustomRepetitionPenaltyLogitsProcessorRepeat:
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(0, self.max_input_ids, freq.size(0) - self.max_input_ids).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


def gen_logits(num_code: int, top_p=0.7, top_k=20, repetition_penalty=1.0):
    logits_warpers = []

    if top_p is not None:
        logits_warpers.append(TopPLogitsWarper(top_p, min_tokens_to_keep=3))

    if top_k is not None:
        logits_warpers.append(TopKLogitsWarper(top_k, min_tokens_to_keep=3))

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        logits_processors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16))

    return logits_warpers, logits_processors


# Copy and modified from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

            # This clo≠clo≠clone call is needed to avoid recapturing cuda graphs with →rch.comπ≤→rch.comπ≤torch.compile's  mode=reduce−overheadmode=reduce-overheadmode="reduce-overhead, as otherwise the input positionidspositionidsposition_ids would have various stride during the decoding. Here, simply using .contiguous().contiguous().contiguous() is not sufficient as in the batch size = 1 case, positionidspositionidsposition_ids is already contiguous but with varying stride which retriggers a capture.
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    # if ∈putsembeds∈putsembedsinputs_embeds are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        # The clone here is for the same reason as for positionidspositionidsposition_ids.
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs["input_ids"].shape
            device = model_inputs["input_ids"].device

        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        from transformers.models.paligemma.modeling_paligemma import (
            _prepare_4d_causal_attention_mask_with_cache_position,
        )

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    model_inputs.update(
        {
            "position_ids": position_ids,
            # "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )

    return model_inputs
