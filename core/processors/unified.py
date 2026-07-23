"""统一处理器 - 一次加载，支持 Chat/Streaming/Duplex 热切换

本模块提供统一的多模式处理器，解决了传统架构中模型无法共享的问题。

核心优势：
=========

1. **一次加载**：模型只加载一次，节省显存和启动时间
2. **毫秒级切换**：Chat/Streaming/Duplex 模式切换 < 1ms
3. **类型安全**：每个模式返回专用的 View，API 清晰
4. **资源共享**：所有模式共享同一个模型实例

架构设计：
=========

```
UnifiedProcessor（统一入口）
├── model: MiniCPMO（统一模型，支持三种模式）
│   └── duplex: DuplexCapability（双工能力组件）
│
├── set_chat_mode() → ChatView
│   └── chat(request) → ChatResponse
│
│
├── set_half_duplex_mode() → HalfDuplexView
│   ├── prefill(request) → str
│   ├── generate(...) → Generator[StreamingChunk]
│   └── rollback() → RollbackResult
│
└── set_duplex_mode() → DuplexView
    ├── prepare(...) → str
    ├── prefill(...) → dict
    ├── generate(...) → DuplexGenerateResult
    └── set_break() / stop()
```

与传统架构对比：
==============

**传统架构**（问题）：
```
ChatProcessor      → 加载 MiniCPMO
StreamingProcessor → 加载 MiniCPMO（重复！）
DuplexProcessor    → 加载 MiniCPMODuplex（独立！不能共享）

问题：
- 显存浪费（多份模型）
- 切换慢（需重新加载）
- 代码重复
```

**统一架构**（本模块）：
```
UnifiedProcessor → 加载一次 MiniCPMO
                 → init_unified() 初始化所有模式
                 → set_xxx_mode() 毫秒级切换

优势：
- 显存节省（一份模型）
- 切换快（< 1ms）
- 代码复用
```

使用示例：
=========

```python
from core.processors.unified import UnifiedProcessor

# 创建统一处理器（一次加载）
processor = UnifiedProcessor(
    model_path="/path/to/base_model",  # HuggingFace 格式基础模型
    pt_path="/path/to/custom_weights.pt",  # 可选：覆盖权重
    ref_audio_path="/path/to/ref.wav",
)

# ========== Chat 模式 ==========
chat = processor.set_chat_mode()
response = chat.chat(ChatRequest(
    messages=[Message(role=Role.USER, content="你好")]
))
print(response.content)

# ========== Half-Duplex 模式（毫秒级切换）==========
half_duplex = processor.set_half_duplex_mode()
half_duplex.prefill(StreamingRequest(
    session_id="user_001",
    messages=[Message(role=Role.USER, content="讲个故事")],
    is_last_chunk=True
))
for chunk in half_duplex.generate(session_id="user_001"):
    print(chunk.text_delta, end="", flush=True)

# ========== Duplex 模式（毫秒级切换）==========
duplex = processor.set_duplex_mode()
duplex.prepare(system_prompt_text="你是一个友好的助手。")

for audio_chunk in audio_stream:
    duplex.prefill(audio_waveform=audio_chunk)
    result = duplex.generate()
    if not result.is_listen:
        print(result.text)
        play_audio(result.audio_data)
```
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Generator, List, TYPE_CHECKING
from pathlib import Path
import json
import os
import time
import logging
import base64
import hashlib

import numpy as np
import torch

from core.capabilities import ProcessorMode
from core.processors.base import BaseProcessor, MiniCPMOProcessorMixin
from core.schemas import (
    # Chat
    ChatRequest, ChatResponse,
    # Streaming
    StreamingRequest, StreamingChunk, StreamingResponse, RollbackResult,
    # Duplex
    DuplexConfig, DuplexGenerateResult, DuplexOfflineInput, DuplexOfflineOutput,
    # Common
    Message, Role,
)
from core.schemas.fc_duplex import (
    FcClosedSpan,
    FcDecodedToolCall,
    FcDecodedUnit,
    FcDecodeOutputRequest,
    FcDecodeOutputResult,
    FcDuplexConfig,
    FcDuplexAudioArtifact,
    FcDuplexComparisonResult,
    FcDuplexOfflineInput,
    FcDuplexOfflineOutput,
    FcDuplexPrepareRequest,
    FcDuplexPrepareResult,
    FcDuplexPrefillRequest,
    FcDuplexPrefillResult,
    FcDuplexStepResult,
    FcDuplexTrainDataRequest,
    FcDuplexTrainDataResult,
    FcDuplexUnitInfo,
    FcFinalizeUnitRequest,
    FcGenerationProtocolOutput,
    FcGenerationStreamTerminationResult,
    FcGenerationTextDeltaOutput,
    FcGenerationTextPendingOutput,
    FcGenerationWarning,
    FcNonSpokenGenerateRequest,
    FcNonSpokenGenerateResult,
    FcSpokenGenerateRequest,
    FcSpokenGenerateResult,
    FcTokenStreamDiff,
    FcToolResponse,
    FcViewGenerationStep,
    NonSpokenStepGenerationFlag,
)

if TYPE_CHECKING:
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO, ProcessorMode as ModelProcessorMode


logger = logging.getLogger(__name__)


# ============================================================
# View 类：各模式的专用接口
# ============================================================

class ChatView(MiniCPMOProcessorMixin):
    """Chat 模式视图
    
    提供 Chat 模式专用的 API。
    
    特性：
    - 无状态（每次完整 prefill）
    - 支持多模态（文本、图像、音频）
    - 支持 TTS 输出
    
    示例：
        >>> chat = processor.set_chat_mode()
        >>> response = chat.chat(request)
        >>> print(response.content)
    """
    
    def __init__(self, model: "MiniCPMO", ref_audio_path: Optional[str] = None):
        self._model = model
        self.ref_audio_path = ref_audio_path
        self._ref_audio_cache = None
        self._session_id = None
    
    def prefill(
        self,
        session_id: str,
        msgs,
        omni_mode: bool = False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template: bool = False,
        enable_thinking: bool = False,
        max_inp_length: int = 8192,
    ) -> str:
        """Prefill 所有消息到 KV cache（不含 generation prompt）"""
        self._session_id = session_id
        prompt = self._model.non_streaming_prefill(
            session_id=session_id,
            msgs=msgs,
            omni_mode=omni_mode,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
            max_inp_length=max_inp_length,
        )
        return prompt
    
    def generate(
        self,
        session_id: str,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        generate_audio: bool = False,
        use_tts_template: bool = True,
        enable_thinking: bool = False,
        tts_ref_audio=None,
        tts_sampling_params=None,
        output_audio_path=None,
        length_penalty: float = 1.1,
    ):
        """基于已有 KV cache 做非流式 generate + 可选 TTS"""
        result = self._model.non_streaming_generate(
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            generate_audio=generate_audio,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
            tts_ref_audio=tts_ref_audio,
            tts_sampling_params=tts_sampling_params,
            output_audio_path=output_audio_path,
            length_penalty=length_penalty,
        )
        return result
    
    def streaming_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        length_penalty: float = 1.1,
    ):
        """基于已有 KV cache 做流式 generate（yield StreamingChunk）"""
        import base64
        start_time = time.time()
        chunk_index = 0
        
        try:
            iter_gen = self._model.streaming_generate(
                session_id=session_id,
                do_sample=do_sample,
                generate_audio=generate_audio,
                max_new_tokens=max_new_tokens,
                use_tts_template=True,
                length_penalty=length_penalty,
            )
            
            for item in iter_gen:
                if item is None:
                    continue
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                    
                item1, item2 = item[0], item[1]
                
                if generate_audio:
                    if item1 is None and item2 is None:
                        continue
                    waveform_chunk = item1
                    text_value = item2 if item2 and isinstance(item2, str) else None
                    audio_data = None
                    if waveform_chunk is not None and hasattr(waveform_chunk, 'cpu'):
                        audio_np = waveform_chunk.cpu().numpy().astype(np.float32)
                        audio_bytes = audio_np.tobytes()
                        audio_data = base64.b64encode(audio_bytes).decode('utf-8')
                else:
                    text_value = item1 if item1 and isinstance(item1, str) else None
                    audio_data = None
                
                from core.schemas.streaming import StreamingChunk
                yield StreamingChunk(
                    chunk_index=chunk_index,
                    text_delta=text_value,
                    audio_data=audio_data,
                    audio_sample_rate=24000,
                    is_final=False,
                )
                chunk_index += 1
                
        except Exception as e:
            logger.error(f"ChatView streaming_generate error: {e}", exc_info=True)
            raise
    
    @property
    def kv_cache_length(self) -> int:
        """当前 KV cache 长度"""
        return self._model._get_kv_cache_length()
    
    def chat(
        self,
        request: ChatRequest,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        generate_audio: Optional[bool] = None,
    ) -> ChatResponse:
        """执行 Chat 推理
        
        Args:
            request: Chat 请求
            max_new_tokens: 最大生成 token 数
            do_sample: 是否采样
            generate_audio: 是否生成音频（None 时从 request.tts.enabled 读取）
            
        Returns:
            ChatResponse
        """
        start_time = time.time()
        
        try:
            return self._chat_impl(request, max_new_tokens, do_sample, generate_audio, start_time)
        except Exception as e:
            logger.error(f"Chat 推理失败: {e}")
            return ChatResponse(
                success=False,
                error=str(e),
                text="",
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    def _chat_impl(
        self,
        request: ChatRequest,
        max_new_tokens: int,
        do_sample: bool,
        generate_audio: Optional[bool],
        start_time: float,
    ) -> ChatResponse:
        """Chat 推理实现（内部方法）"""
        
        # 确定 TTS 参数
        tts_config = request.tts if hasattr(request, 'tts') and request.tts else None
        tts_enabled = tts_config.enabled if tts_config else False
        
        # 如果 TTS 启用但没指定 ref_audio，使用 ChatView 的默认 ref_audio
        if tts_config and tts_enabled and not tts_config.ref_audio_path and not tts_config.ref_audio_data:
            if self.ref_audio_path:
                tts_config = tts_config.model_copy(update={"ref_audio_path": self.ref_audio_path})
        
        # 如果未显式指定 generate_audio，从 tts.enabled 读取
        if generate_audio is None:
            generate_audio = tts_enabled
        
        use_tts_template = request.use_tts_template if hasattr(request, 'use_tts_template') else False
        use_tts_template = use_tts_template or generate_audio
        
        output_audio_path = None
        tts_sampling_params = None
        
        if generate_audio and tts_config:
            output_audio_path = tts_config.output_path
            
            # 构建 TTS 采样参数
            if tts_config.sampling:
                from MiniCPMO45.utils import TTSSamplingParams as ModelTTSSamplingParams
                tts_sampling_params = ModelTTSSamplingParams(
                    top_p=tts_config.sampling.top_p,
                    min_p=tts_config.sampling.min_p,
                    top_k=tts_config.sampling.top_k,
                    repetition_penalty=tts_config.sampling.repetition_penalty,
                    temperature=tts_config.sampling.temperature,
                    win_size=tts_config.sampling.win_size,
                    tau_r=tts_config.sampling.tau_r,
                )
        
        # 转换消息格式
        msgs = self._convert_messages_to_model_format(
            request.messages,
            tts_config=tts_config,
        )
        
        # 解析 TTS ref audio（独立于 LLM ref audio）
        # 当用户在 tts_config 中提供了 ref_audio_data 或 ref_audio_path 时，
        # 将其解析为 ndarray 传给 model.chat()，用于 TTS vocoder 初始化。
        # 这样即使 messages 中 system prompt 的 audio（LLM ref audio）是另一个音频，
        # TTS 也能使用独立的参考音频。
        tts_ref_audio: Optional[np.ndarray] = None
        if tts_config and generate_audio:
            if tts_config.ref_audio_data:
                import base64 as b64_mod
                tts_ref_bytes = b64_mod.b64decode(tts_config.ref_audio_data)
                tts_ref_audio = np.frombuffer(tts_ref_bytes, dtype=np.float32)
                logger.info(f"Chat TTS ref audio from tts_config.ref_audio_data: {len(tts_ref_audio)} samples ({len(tts_ref_audio)/16000:.1f}s)")
            elif tts_config.ref_audio_path:
                import librosa
                tts_ref_audio, _ = librosa.load(tts_config.ref_audio_path, sr=16000, mono=True)
                logger.info(f"Chat TTS ref audio from tts_config.ref_audio_path: {tts_config.ref_audio_path}")
        
        # 调用模型
        with torch.no_grad():
            result = self._model.chat(
                msgs=msgs,
                sampling=do_sample,
                max_new_tokens=max_new_tokens,
                stream=False,
                # TTS 参数
                use_tts_template=use_tts_template,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                tts_sampling_params=tts_sampling_params,
                tts_ref_audio=tts_ref_audio,
                # 高级参数
                omni_mode=request.omni_mode if hasattr(request, 'omni_mode') else False,
                enable_thinking=request.enable_thinking if hasattr(request, 'enable_thinking') else False,
                return_prompt=request.return_prompt if hasattr(request, 'return_prompt') else False,
                # 图像参数
                max_slice_nums=request.image.max_slice_nums if hasattr(request, 'image') and request.image else None,
                use_image_id=request.image.use_image_id if hasattr(request, 'image') and request.image else False,
            )
        
        # 处理返回值
        # 模型 chat() 返回值（modeling_minicpmo_unified.py）：
        #   - 无音频: answer (str)
        #   - 有音频: (answer, waveform_np)
        #   - return_prompt + 无音频: (answer, prompt)
        #   - return_prompt + 有音频: (answer, prompt, waveform_np)
        return_prompt_flag = request.return_prompt if hasattr(request, 'return_prompt') else False
        
        text_content = None
        prompt = None
        waveform = None
        
        if isinstance(result, tuple):
            if return_prompt_flag:
                if len(result) == 3:
                    text_content, prompt, waveform = result
                else:
                    text_content, prompt = result
            else:
                if len(result) == 2:
                    text_content, waveform = result
                else:
                    text_content = result[0]
        else:
            text_content = result
        
        # 将 waveform numpy array 转为 base64 WAV
        audio_base64 = None
        if waveform is not None:
            try:
                import io
                import soundfile as sf_lib
                buf = io.BytesIO()
                sf_lib.write(buf, waveform, 24000, format="WAV")
                audio_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                logger.info(f"TTS 音频生成成功: {len(waveform)} samples, {len(waveform)/24000:.1f}s")
            except Exception as e:
                logger.error(f"TTS 音频编码失败: {e}")
        
        duration_ms = (time.time() - start_time) * 1000
        
        # 读取模型存储的 token 统计（model.chat() 内部设置）
        chat_token_stats = getattr(self._model, '_last_chat_token_stats', {})
        input_tokens = chat_token_stats.get('input_tokens', 0)
        generated_tokens = chat_token_stats.get('generated_tokens', 0)
        
        return ChatResponse(
            text=text_content or "",
            audio_data=audio_base64,
            audio_path=tts_config.output_path if (tts_config and tts_config.output_path) else None,
            audio_sample_rate=24000,
            duration_ms=duration_ms,
            prompt=prompt,
            tokens_generated=generated_tokens,
            token_stats={
                "cached_tokens": 0,  # Chat 无状态，无缓存
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": input_tokens + generated_tokens,
            },
        )


class HalfDuplexView(MiniCPMOProcessorMixin):
    """Half-Duplex 模式视图
    
    提供 Half-Duplex 模式专用的 API。
    
    特性：
    - 有状态（session_id + KV Cache 复用）
    - 流式返回（边生成边返回）
    - 支持回溯（speculative_snapshot）
    - 独占 Worker（会话期间）
    
    示例：
        >>> half_duplex = processor.set_half_duplex_mode()
        >>> half_duplex.prefill(request)
        >>> for chunk in half_duplex.generate(session_id):
        ...     print(chunk.text_delta, end="")
    """
    
    def __init__(self, model: "MiniCPMO", ref_audio_path: Optional[str] = None):
        self._model = model
        self.ref_audio_path = ref_audio_path
        self._ref_audio_cache = None
    
    def init_ref_audio(self, ref_audio_path: Optional[str] = None) -> None:
        """初始化参考音频（用于 TTS，从文件路径）
        
        Args:
            ref_audio_path: 参考音频路径
        """
        path = ref_audio_path or self.ref_audio_path
        if path:
            ref_audio = self._load_ref_audio(path)
            self._model.init_token2wav_cache(prompt_speech_16k=ref_audio)
            logger.info(f"已初始化参考音频: {path}")
    
    def init_ref_audio_from_data(self, ref_audio: np.ndarray) -> None:
        """初始化参考音频（用于 TTS，从 ndarray 数据）
        
        用于前端直接上传 base64 ref audio 的场景，
        无需落盘为文件，直接用 ndarray 初始化 TTS cache。
        
        Args:
            ref_audio: 16kHz mono float32 音频 ndarray
        """
        self._model.init_token2wav_cache(prompt_speech_16k=ref_audio)
        logger.info(f"已初始化参考音频 (from data, {len(ref_audio)} samples, {len(ref_audio)/16000:.1f}s)")
    
    def reset_session(self, session_id: str) -> None:
        """重置会话
        
        Args:
            session_id: 会话 ID
        """
        logger.info(f"重置会话: {session_id}")
        self._model.reset_session()
    
    def prefill(self, request: StreamingRequest) -> str:
        """流式预填充（支持多条消息逐条 prefill）
        
        模型的 streaming_prefill 要求每次只处理一条消息（assert len(msgs)==1）。
        本方法将 request.messages 拆分为逐条调用，is_last_chunk 仅在最后一条
        且 request.is_last_chunk=True 时设为 True。
        
        Args:
            request: 流式请求（可包含多条消息）
            
        Returns:
            最后一条消息的 prompt 文本
        """
        prompt = ""
        num_messages = len(request.messages)
        
        for i, msg in enumerate(request.messages):
            content = self._convert_content_to_model_format(msg.content)
            if len(content) == 1 and isinstance(content[0], str):
                content = content[0]
            msgs = [{
                "role": msg.role.value,
                "content": content
            }]
            
            # is_last_chunk 仅在最后一条消息且 request 标记为 last 时为 True
            is_last = request.is_last_chunk and (i == num_messages - 1)
            
            max_slice = request.image.max_slice_nums if hasattr(request, 'image') and request.image else None
            result = self._model.streaming_prefill(
                session_id=request.session_id,
                msgs=msgs,
                omni_mode=request.omni_mode,
                max_slice_nums=max_slice,
                use_tts_template=request.use_tts_template,
                enable_thinking=request.enable_thinking,
                is_last_chunk=is_last,
                stream_input=False,
            )
            if result:
                prompt = result
        
        return prompt
    
    def non_streaming_prefill(
        self,
        session_id: str,
        msgs,
        omni_mode: bool = False,
        max_slice_nums=None,
        use_tts_template: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        """非流式预填充：一次性 prefill 所有消息到 KV cache"""
        prompt = self._model.non_streaming_prefill(
            session_id=session_id,
            msgs=msgs,
            omni_mode=omni_mode,
            max_slice_nums=max_slice_nums,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )
        return prompt
    
    def generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        enable_speculative_snapshot: bool = False,
        length_penalty: float = 1.1,
    ) -> Generator[StreamingChunk, None, None]:
        """流式生成
        
        Args:
            session_id: 会话 ID
            generate_audio: 是否生成音频
            max_new_tokens: 最大生成 token 数
            do_sample: 是否采样
            enable_speculative_snapshot: 是否启用回溯快照
            length_penalty: 长度惩罚系数（>1.0 抑制 EOS，输出更长；=1.0 不惩罚）
            
        Yields:
            StreamingChunk
        """
        start_time = time.time()
        chunk_index = 0
        
        try:
            iter_gen = self._model.streaming_generate(
                session_id=session_id,
                do_sample=do_sample,
                generate_audio=generate_audio,
                max_new_tokens=max_new_tokens,
                use_tts_template=True,
                enable_speculative_snapshot=enable_speculative_snapshot,
                length_penalty=length_penalty,
            )
            
            for item in iter_gen:
                if item is None:
                    continue
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                    
                item1, item2 = item[0], item[1]
                chunk_start = time.time()
                
                if generate_audio:
                    if item1 is None and item2 is None:
                        continue
                    
                    waveform_chunk = item1
                    text_value = item2 if item2 and isinstance(item2, str) else None
                    
                    audio_data = None
                    if waveform_chunk is not None and hasattr(waveform_chunk, 'cpu'):
                        audio_np = waveform_chunk.cpu().numpy().astype(np.float32)
                        audio_bytes = audio_np.tobytes()
                        audio_data = base64.b64encode(audio_bytes).decode('utf-8')
                else:
                    text_value = item1 if item1 and isinstance(item1, str) else None
                    audio_data = None
                
                chunk_duration = (time.time() - chunk_start) * 1000
                
                yield StreamingChunk(
                    chunk_index=chunk_index,
                    text_delta=text_value,
                    audio_data=audio_data,
                    audio_sample_rate=24000,
                    is_final=False,
                    duration_ms=chunk_duration,
                )
                
                chunk_index += 1
            
            # 最终块
            total_duration = (time.time() - start_time) * 1000
            yield StreamingChunk(
                chunk_index=chunk_index,
                text_delta=None,
                audio_data=None,
                is_final=True,
                duration_ms=total_duration,
            )
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield StreamingChunk(
                chunk_index=chunk_index,
                text_delta=None,
                audio_data=None,
                is_final=True,
                duration_ms=(time.time() - start_time) * 1000,
            )
            raise
    
    def can_rollback(self) -> bool:
        """检查是否可以回溯"""
        return self._model.has_speculative_snapshot()
    
    def rollback(self) -> RollbackResult:
        """回溯到上一个快照点"""
        if not self._model.has_speculative_snapshot():
            return RollbackResult(
                success=False,
                reason="没有可用的快照"
            )
        
        try:
            success = self._model.restore_speculative_snapshot()
            if success:
                return RollbackResult(
                    success=True,
                    restored_position="已恢复到 streaming_generate 调用前"
                )
            else:
                return RollbackResult(success=False, reason="恢复失败")
        except Exception as e:
            return RollbackResult(success=False, reason=str(e))
    
    def clear_rollback_point(self) -> None:
        """清除回溯点"""
        self._model.clear_speculative_snapshot()
    
    def complete_turn(
        self,
        session_id: str,
        messages: List[Message],
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        output_audio_path: Optional[str] = None,
        length_penalty: float = 1.1,
    ) -> StreamingResponse:
        """完成一轮对话（便捷方法）
        
        封装 prefill + generate 流程，自动累加增量文本和音频。
        适用于不需要实时流式输出的场景。
        
        Args:
            session_id: 会话 ID
            messages: 消息列表（可以包含多条，会逐条 prefill）
            generate_audio: 是否生成音频
            max_new_tokens: 最大生成 token 数
            output_audio_path: 可选，自动保存音频的路径
            
        Returns:
            StreamingResponse: 包含完整文本和音频的响应
            
        示例：
            >>> half_duplex = processor.set_half_duplex_mode()
            >>> half_duplex.reset_session("user_001")
            >>> half_duplex.init_ref_audio("/path/to/ref.wav")
            >>> 
            >>> response = streaming.complete_turn(
            ...     session_id="user_001",
            ...     messages=[
            ...         Message(role=Role.SYSTEM, content="你是一个友好的助手。"),
            ...         Message(role=Role.USER, content="你好，介绍一下你自己。"),
            ...     ],
            ...     generate_audio=True,
            ...     output_audio_path="/tmp/output.wav"
            ... )
            >>> print(response.full_text)
            >>> print(f"音频时长: {response.audio_duration_ms}ms")
        """
        # StreamingRequest, StreamingResponse, Role, Message 已在顶层导入
        
        start_time = time.time()
        
        # 逐条 prefill 消息
        for i, msg in enumerate(messages):
            is_last = (i == len(messages) - 1)
            self.prefill(StreamingRequest(
                session_id=session_id,
                messages=[msg],
                use_tts_template=True,
                is_last_chunk=is_last,
            ))
        
        # 生成并累加结果
        full_text = ""
        audio_chunks: List[np.ndarray] = []
        chunk_count = 0
        
        for chunk in self.generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        ):
            chunk_count += 1
            
            # 累加增量文本
            if chunk.text_delta:
                full_text += chunk.text_delta
            
            # 收集音频块
            if chunk.audio_data:
                audio_bytes = base64.b64decode(chunk.audio_data)
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                if audio_np.size > 0:
                    audio_chunks.append(audio_np)
        
        # 合并音频
        audio_data_base64 = None
        audio_duration_ms = None
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_duration_ms = len(full_audio) / 24000 * 1000
            
            # 保存音频文件（如果指定）
            if output_audio_path:
                import soundfile as sf
                sf.write(output_audio_path, full_audio, 24000)
                logger.info(f"音频已保存: {output_audio_path}")
            
            # 转为 Base64
            audio_data_base64 = base64.b64encode(full_audio.tobytes()).decode('utf-8')
        
        total_duration_ms = (time.time() - start_time) * 1000
        
        return StreamingResponse(
            success=True,
            session_id=session_id,
            full_text=full_text,
            audio_path=output_audio_path,
            audio_data=audio_data_base64,
            audio_sample_rate=24000,
            audio_duration_ms=audio_duration_ms,
            total_chunks=chunk_count,
            total_duration_ms=total_duration_ms,
        )


class DuplexView:
    """Duplex 模式视图
    
    提供 Duplex 模式专用的 API。
    
    特性：
    - 全双工实时对话
    - 支持打断
    - Listen/Speak 状态管理
    
    示例：
        >>> duplex = processor.set_duplex_mode()
        >>> duplex.prepare(system_prompt_text="你是助手")
        >>> duplex.prefill(audio_waveform=chunk)
        >>> result = duplex.generate()
    """
    
    def __init__(
        self, 
        model: "MiniCPMO",
        ref_audio_path: Optional[str] = None,
        config: Optional[DuplexConfig] = None,
    ):
        self._model = model
        self.ref_audio_path = ref_audio_path
        self.config = config or DuplexConfig()
        self._ref_audio_cache: Optional[np.ndarray] = None
    
    def _load_ref_audio(self, path: Optional[str] = None) -> np.ndarray:
        """加载参考音频"""
        import librosa
        
        audio_path = path or self.ref_audio_path
        if audio_path is None:
            raise ValueError("未提供参考音频路径")
        
        if self._ref_audio_cache is not None and path is None:
            return self._ref_audio_cache
        
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        if path is None:
            self._ref_audio_cache = audio
        
        return audio
    
    def prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
    ) -> str:
        """准备双工会话
        
        Args:
            system_prompt_text: 系统提示文本
            ref_audio_path: 参考音频路径
            prompt_wav_path: TTS prompt 音频路径
            
        Returns:
            完整的 system prompt 字符串
        """
        if system_prompt_text is None:
            system_prompt_text = "Streaming Omni Conversation."
        
        prefix_system_prompt = f"<|im_start|>system\n{system_prompt_text}\n<|audio_start|>"
        suffix_system_prompt = "<|audio_end|><|im_end|>"
        
        # 加载参考音频
        ref_audio = None
        if ref_audio_path or self.ref_audio_path:
            ref_audio = self._load_ref_audio(ref_audio_path)
        
        # 调用透传方法
        prompt = self._model.duplex_prepare(
            prefix_system_prompt=prefix_system_prompt,
            suffix_system_prompt=suffix_system_prompt,
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path or ref_audio_path or self.ref_audio_path,
        )
        
        logger.info(f"双工会话准备完成")
        return prompt
    
    def prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        audio_path: Optional[str] = None,
        frame_list: Optional[List[np.ndarray]] = None,
        max_slice_nums: int = 1,
    ) -> dict:
        """预填充用户音频
        
        Args:
            audio_waveform: 音频波形数据（16kHz mono）
            audio_path: 音频文件路径
            frame_list: 图像帧列表
            max_slice_nums: HD 图像切片数
            
        Returns:
            预填充结果 dict
        """
        import librosa
        
        if audio_path and audio_waveform is None:
            audio_waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        result = self._model.duplex_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )
        
        return result
    
    def generate(self, force_listen: bool = False) -> DuplexGenerateResult:
        """生成响应
        
        Args:
            force_listen: 前端 Force Listen 开关，强制本次生成为 listen
            
        Returns:
            DuplexGenerateResult
        """
        result = self._model.duplex_generate(
            decode_mode=self.config.decode_mode,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            listen_prob_scale=self.config.listen_prob_scale,
            listen_top_k=self.config.listen_top_k,
            text_repetition_penalty=self.config.text_repetition_penalty,
            text_repetition_window_size=self.config.text_repetition_window_size,
            length_penalty=self.config.length_penalty,
            force_listen_override=force_listen,
        )
        
        # 转换音频
        audio_data = None
        if result.get("audio_waveform") is not None:
            waveform = result["audio_waveform"]
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            audio_bytes = waveform.astype(np.float32).tobytes()
            audio_data = base64.b64encode(audio_bytes).decode('utf-8')
        
        return DuplexGenerateResult(
            is_listen=result.get("is_listen", True),
            text=result.get("text", ""),
            audio_data=audio_data,
            end_of_turn=result.get("end_of_turn", False),
            current_time=result.get("current_time", 0),
            cost_llm_ms=result.get("cost_llm", 0) * 1000 if result.get("cost_llm") else None,
            cost_tts_prep_ms=result.get("cost_tts_prep", 0) * 1000 if result.get("cost_tts_prep") else None,
            cost_tts_ms=result.get("cost_tts", 0) * 1000 if result.get("cost_tts") else None,
            cost_token2wav_ms=result.get("cost_token2wav", 0) * 1000 if result.get("cost_token2wav") else None,
            cost_all_ms=result.get("cost_all", 0) * 1000 if result.get("cost_all") else None,
            n_tokens=result.get("n_tokens"),
            n_tts_tokens=result.get("n_tts_tokens"),
        )
    
    def finalize(self) -> None:
        """完成 generate 的延迟操作（feed 终止符 + </unit>，滑窗维护）
        
        必须在 generate() 之后、下一次 prefill() 之前调用。
        可异步调度：先返回结果给前端，再在后台执行 finalize。
        """
        self._model.duplex_finalize()

    def set_break(self) -> None:
        """设置打断信号"""
        self._model.duplex_set_break()
        logger.info("设置打断信号")
    
    def clear_break(self) -> None:
        """清除打断信号"""
        self._model.duplex_clear_break()
    
    def stop(self) -> None:
        """停止当前会话"""
        self._model.duplex_stop()
        logger.info("会话已停止")
    
    def is_break_set(self) -> bool:
        """检查是否设置了打断"""
        return self._model.duplex_is_break_set()
    
    def is_stopped(self) -> bool:
        """检查会话是否已停止"""
        return self._model.duplex_is_stopped()
    
    def cleanup(self) -> None:
        """清理 Duplex 会话资源，释放 GPU 显存
        
        在会话结束后调用（stop 之后），释放所有 Duplex 相关的 GPU 资源，
        使显存恢复到模型刚加载时的状态。
        
        释放的资源包括：
        - Duplex KV cache（decoder.cache）— ~660 MB
        - TTS audio_tokenizer live caches（stream_cache, hift_cache_dict）— ~820 MB
        - 模型级 session 状态（token2wav_cache 等）— ~66 MB
        
        注意：
        - 此方法不调用 gc.collect() 和 torch.cuda.empty_cache()，
          调用方应在此方法之后自行调用以确保显存回收到 CUDA driver。
        - cleanup 后可以正常启动新的 Duplex session（prepare 会重新初始化所有状态）。
        - cleanup 后也可以正常切换到 Chat/Streaming 模式。
        """
        model = self._model
        
        # Step 1: 重置 DuplexCapability 状态（释放 KV cache、TTS past_key_values 等）
        if hasattr(model, 'duplex') and model.duplex is not None:
            model.duplex._reset_streaming_state()
            model.duplex.decoder.reset()
        
        # Step 2: 清理 TTS audio_tokenizer 的 live caches（最大泄漏源）
        if hasattr(model, 'tts') and hasattr(model.tts, 'audio_tokenizer'):
            tokenizer = model.tts.audio_tokenizer
            for attr in ('stream_cache', 'hift_cache_dict', 'cache'):
                if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
                    setattr(tokenizer, attr, None)
        
        # Step 3: 重置模型级 session 状态
        model.reset_session(reset_token2wav_cache=True)
        
        logger.info("Duplex 会话资源已清理")
    
    def offline_inference(self, task_input: "DuplexOfflineInput") -> "DuplexOfflineOutput":
        """离线推理（便捷方法）
        
        对完整音频文件进行离线推理，一站式处理。
        
        适用场景：
        - 单元测试
        - 离线批量处理
        - 演示场景
        
        注意：这不是实时双工会话，而是对完整音频文件的离线处理。
        实时双工请直接使用 prepare/prefill/generate 原语。
        
        Args:
            task_input: 离线推理输入
            
        Returns:
            离线推理输出
        
        示例：
            >>> output = duplex.offline_inference(DuplexOfflineInput(
            ...     system_prompt="你是一个友好的助手。",
            ...     user_audio_path="/path/to/audio.wav",
            ...     ref_audio_path="/path/to/ref.wav"
            ... ))
            >>> print(output.full_text)
        """
        from core.schemas.duplex import DuplexOfflineInput, DuplexOfflineOutput, DuplexChunkResult
        import librosa
        
        start_time = time.time()
        chunks = []
        full_text = ""
        audio_chunks = []
        
        try:
            # 准备会话
            self.prepare(
                system_prompt_text=task_input.system_prompt,
                ref_audio_path=task_input.ref_audio_path,
            )
            
            # 加载用户音频并分块
            if task_input.user_audio_path:
                user_audio, _ = librosa.load(
                    task_input.user_audio_path, 
                    sr=task_input.config.sample_rate, 
                    mono=True
                )
                chunk_samples = task_input.config.sample_rate * task_input.config.chunk_ms // 1000
                num_chunks = (len(user_audio) + chunk_samples - 1) // chunk_samples
                
                for i in range(num_chunks):
                    chunk_start = time.time()
                    
                    # 获取音频块
                    start_idx = i * chunk_samples
                    end_idx = min(start_idx + chunk_samples, len(user_audio))
                    audio_chunk = user_audio[start_idx:end_idx]
                    
                    # 如果不足一个块，补零
                    if len(audio_chunk) < chunk_samples:
                        audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))
                    
                    # 获取图像帧（如果有）
                    # [CRITICAL] 必须传 PIL Image，不能是 numpy array（否则内存激增 18GB）
                    frame_list = None
                    if task_input.image_paths and i < len(task_input.image_paths):
                        from PIL import Image
                        frame = Image.open(task_input.image_paths[i]).convert("RGB")
                        frame_list = [frame]  # PIL Image, NOT np.array(frame)
                    
                    # 预填充
                    self.prefill(audio_waveform=audio_chunk, frame_list=frame_list)
                    
                    # 生成
                    result = self.generate()

                    self.finalize()

                    chunk_elapsed = (time.time() - chunk_start) * 1000
                    
                    # 记录结果
                    chunks.append(DuplexChunkResult(
                        chunk_idx=i,
                        phase="user",
                        is_listen=result.is_listen,
                        text=result.text,
                        has_audio=result.audio_data is not None,
                        audio_data=result.audio_data,  # 保存音频数据
                        end_of_turn=result.end_of_turn,
                        elapsed_ms=chunk_elapsed,
                    ))
                    
                    if not result.is_listen:
                        full_text += result.text
                        if result.audio_data:
                            audio_chunks.append(result.audio_data)
                    
                    if result.end_of_turn:
                        break
            
            # 停止会话
            self.stop()
            
            total_duration = (time.time() - start_time) * 1000
            
            return DuplexOfflineOutput(
                success=True,
                full_text=full_text,
                total_chunks=len(chunks),
                audio_duration_s=len(audio_chunks) * 0.5,  # 估算
                total_duration_ms=total_duration,
                chunks=chunks,
            )
            
        except Exception as e:
            logger.error(f"离线推理失败: {e}")
            return DuplexOfflineOutput(
                success=False,
                error=str(e),
                total_duration_ms=(time.time() - start_time) * 1000,
            )


class ToolCallIdGenerator:
    """Simple unique tool call id generator."""

    def __init__(self, prefix: str = "fc_call"):
        self.prefix = prefix
        self._next = 1

    def next_id(self) -> str:
        value = f"{self.prefix}_{self._next:06d}"
        self._next += 1
        return value


class FixedToolCallIdGenerator:
    """Deterministic generator for offline train/infer consistency tests."""

    def __init__(self, ids: Iterable[str]):
        self._ids = list(ids)
        self._index = 0

    def next_id(self) -> str:
        if self._index >= len(self._ids):
            raise ValueError("fixed tool_call_id generator exhausted")
        value = self._ids[self._index]
        self._index += 1
        return value


@dataclass
class ToolCallState:
    id: str
    tool_call: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None
    wire: Optional[str] = None
    started_sent: bool = False
    response_received: bool = False


class ToolCallStateManager:
    """View-level tool call id and response state manager."""

    def __init__(self, id_generator: Optional[Any] = None):
        self.id_generator = id_generator or ToolCallIdGenerator()
        self._states: Dict[str, ToolCallState] = {}
        self._pending_started: List[str] = []
        self._pending_error_responses: List[Dict[str, Any]] = []

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        calls = []
        for state in self._states.values():
            if state.tool_call is not None and not state.parse_error:
                calls.append({
                    "tool_call_id": state.id,
                    **state.tool_call,
                })
        return calls

    @property
    def has_state(self) -> bool:
        """Whether a valid tool call is still waiting for its external result."""

        return bool(
            self._pending_started
            or self._pending_error_responses
        ) or any(
            state.tool_call is not None
            and not state.parse_error
            and not state.response_received
            for state in self._states.values()
        )

    def _next_unique_id(self) -> str:
        call_id = self.id_generator.next_id()
        if call_id in self._states:
            raise ValueError(f"duplicate generated tool_call_id: {call_id}")
        return call_id

    def register_tool_call(self, tool_call: Dict[str, Any], wire: Optional[str] = None) -> str:
        call_id = self._next_unique_id()
        self._states[call_id] = ToolCallState(id=call_id, tool_call=tool_call, wire=wire)
        self._pending_started.append(call_id)
        return call_id

    def register_parse_error(self, error: str, wire: Optional[str] = None) -> str:
        call_id = self._next_unique_id()
        self._states[call_id] = ToolCallState(id=call_id, parse_error=error, wire=wire)
        self._pending_started.append(call_id)
        self._pending_error_responses.append({
            "type": "tool_response",
            "call_id": call_id,
            "content": f"工具调用解析失败，无法执行该工具调用。错误信息：{error}",
        })
        return call_id

    def consume_pending_started_events(self) -> List[Dict[str, Any]]:
        events = []
        while self._pending_started:
            call_id = self._pending_started.pop(0)
            state = self._states[call_id]
            state.started_sent = True
            events.append({"type": "tool_started", "call_id": call_id})
        return events

    def consume_pending_error_responses(self) -> List[Dict[str, Any]]:
        events = list(self._pending_error_responses)
        self._pending_error_responses.clear()
        return events

    def validate_and_mark_responses(self, tool_responses) -> List[Dict[str, Any]]:
        if not tool_responses:
            return []
        converted = []
        for response in tool_responses:
            if hasattr(response, "model_dump"):
                item = response.model_dump()
            elif isinstance(response, dict):
                item = dict(response)
            else:
                call_id, content = response
                item = {"call_id": call_id, "content": content}
            call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id")
            if call_id not in self._states:
                raise ValueError(f"unknown tool_call_id: {call_id}")
            state = self._states[call_id]
            if state.response_received:
                raise ValueError(f"duplicate tool_response for tool_call_id: {call_id}")
            state.response_received = True
            item["call_id"] = call_id
            item.setdefault("type", "tool_response")
            converted.append(item)
        return converted

    def restore_completed_sequence(self, tool_call_count: int) -> None:
        """Advance deterministic internal IDs after replayed completed calls."""

        if tool_call_count < 0:
            raise ValueError(f"tool_call_count 必须 >= 0: {tool_call_count}")
        if self._states:
            raise RuntimeError("cannot restore tool-call sequence with active states")
        for _ in range(tool_call_count):
            self.id_generator.next_id()


@dataclass
class _FcTextStreamState:
    """View-owned ordinary text decoder state for one continuous stream."""

    stream_id: str
    kind: str
    decoder: Any
    pending_token_count: int = 0
    pending_token_ids: List[int] = field(default_factory=list)
    emitted_parts: List[str] = field(default_factory=list)


class FcDuplexView:
    """FC slot Duplex 模式视图。"""

    def __init__(self, model: "MiniCPMO", config: Optional[FcDuplexConfig] = None):
        self._model = model
        self.config = config or FcDuplexConfig()
        self.tool_call_manager = ToolCallStateManager()
        self._ref_audio_cache: Dict[str, np.ndarray] = {}
        self._protocol_tokenizer: Any = None
        self._stream_seq = 0
        self._non_spoken_text_stream: Optional[_FcTextStreamState] = None
        self._spoken_text_stream: Optional[_FcTextStreamState] = None
        self._closed_non_spoken_texts: List[tuple[str, str, bool]] = []
        self._next_non_spoken_continuation_kind: Optional[str] = None
        self._non_spoken_aggregate_kind: Optional[str] = None
        self._non_spoken_aggregate_parts: List[str] = []
        self._non_spoken_aggregate_incomplete = False
        self._last_closed_spoken_text: Optional[str] = None
        self._resume_text_roundtrip_valid = True
        self._resume_text_roundtrip_error: Optional[Dict[str, Any]] = None
        self._resume_ref_audio_sha256: Optional[str] = None
        self._resume_prompt_wav_sha256: Optional[str] = None

    @staticmethod
    def _audio_from_base64(audio_data: Optional[str]) -> Optional[np.ndarray]:
        if not audio_data:
            return None
        audio_bytes = base64.b64decode(audio_data)
        return np.frombuffer(audio_bytes, dtype=np.float32)

    @staticmethod
    def _non_spoken_generation_flag(data: dict) -> NonSpokenStepGenerationFlag:
        if not data.get("terminated", False):
            return NonSpokenStepGenerationFlag.continue_non_spoken_generation
        if data.get("close_reason") == "no_action":
            return NonSpokenStepGenerationFlag.no_action
        return NonSpokenStepGenerationFlag.non_spoken_slot_eos

    def _ensure_protocol_tokenizer(self) -> Any:
        """Return the model capability's SDK tokenizer without loading another target."""

        if self._protocol_tokenizer is not None:
            return self._protocol_tokenizer
        capability = getattr(self._model, "fc_duplex", None)
        if capability is None:
            raise RuntimeError("FC duplex capability is not initialized")
        self._protocol_tokenizer = capability.protocol_tokenizer
        return self._protocol_tokenizer

    def _new_text_stream(self, kind: str) -> _FcTextStreamState:
        """Create an isolated SDK decoder for one semantic text stream."""

        self._stream_seq += 1
        tokenizer = self._ensure_protocol_tokenizer()
        return _FcTextStreamState(
            stream_id=f"{kind}_{self._stream_seq}",
            kind=kind,
            decoder=tokenizer.create_ordinary_text_decode_stream(),
        )

    def _semantic_key(self, token_id: int) -> Optional[str]:
        """Resolve a protocol structural token ID to its stable semantic key."""

        tokenizer = self._ensure_protocol_tokenizer()
        info = tokenizer.token_info(token_id)
        return info.semantic_key

    def _terminate_text_stream(
        self,
        *,
        track: str,
        reason: str,
        record_closed_span: bool = False,
    ) -> Optional[FcGenerationWarning]:
        """Destroy one semantic decoder and report an incomplete BPE boundary."""

        stream_attr = (
            "_spoken_text_stream" if track == "spoken" else "_non_spoken_text_stream"
        )
        stream: Optional[_FcTextStreamState] = getattr(self, stream_attr)
        if stream is None:
            return None
        has_pending = bool(stream.pending_token_count)
        if track == "non_spoken":
            self._non_spoken_aggregate_incomplete = (
                self._non_spoken_aggregate_incomplete or has_pending
            )
            if record_closed_span:
                self._closed_non_spoken_texts.append(
                    (
                        self._non_spoken_aggregate_kind or stream.kind,
                        "".join(self._non_spoken_aggregate_parts),
                        self._non_spoken_aggregate_incomplete,
                    )
                )
                self._non_spoken_aggregate_kind = None
                self._non_spoken_aggregate_parts = []
                self._non_spoken_aggregate_incomplete = False
        else:
            self._last_closed_spoken_text = "".join(stream.emitted_parts)
        warning = None
        if has_pending:
            self._resume_text_roundtrip_valid = False
            self._resume_text_roundtrip_error = {
                "status": "unavailable",
                "reason": "pending_text_delta",
                "stream_id": stream.stream_id,
            }
            warning = FcGenerationWarning(
                code="incomplete_bpe_at_stream_end",
                stream_id=stream.stream_id,
                track=track,
                reason=reason,
                message="文本边界包含未完成 BPE，公共 API 历史无法保证精确复现",
            )
        setattr(self, stream_attr, None)
        return warning

    def _decode_generation_steps(
        self,
        token_ids: List[int],
        *,
        track: str,
    ) -> tuple[
        List[FcViewGenerationStep],
        str,
        List[FcGenerationWarning],
        Optional[str],
    ]:
        """Convert raw generated IDs into resumable per-token View steps."""

        if track not in {"spoken", "non_spoken"}:
            raise ValueError(f"unsupported FC generation track: {track}")
        tokenizer = self._ensure_protocol_tokenizer()
        stream_attr = (
            "_spoken_text_stream" if track == "spoken" else "_non_spoken_text_stream"
        )
        stream: Optional[_FcTextStreamState] = getattr(self, stream_attr)
        steps: List[FcViewGenerationStep] = []
        emitted_parts: List[str] = []
        warnings: List[FcGenerationWarning] = []
        span_started: Optional[str] = None
        start_keys = {
            "non_spoken": {
                "think_start": "think",
                "tool_call_start": "tool_call",
            },
            "spoken": {
                "speak": "spoken",
            },
        }
        end_keys = {
            "non_spoken": {
                "think_end": "think",
                "tool_call_end": "tool_call",
            },
            "spoken": {
                "spoken_turn_eos": "spoken",
            },
        }

        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if tokenizer.is_ordinary_token_id(token_id):
                if stream is None:
                    if (
                        track == "non_spoken"
                        and self._next_non_spoken_continuation_kind is not None
                    ):
                        stream = self._new_text_stream(
                            self._next_non_spoken_continuation_kind
                        )
                        span_started = stream.kind
                        if self._non_spoken_aggregate_kind not in {
                            None,
                            stream.kind,
                        }:
                            raise RuntimeError(
                                "non-spoken continuation changed aggregate kind: "
                                f"{self._non_spoken_aggregate_kind} -> {stream.kind}"
                            )
                        self._non_spoken_aggregate_kind = stream.kind
                        self._next_non_spoken_continuation_kind = None
                        setattr(self, stream_attr, stream)
                    else:
                        raise RuntimeError(
                            f"FC {track} ordinary token arrived before stream opener: "
                            f"{token_id}"
                        )
                stream.pending_token_count += 1
                stream.pending_token_ids.append(token_id)
                text_delta = stream.decoder.step(token_id)
                if text_delta is None:
                    output = FcGenerationTextPendingOutput()
                else:
                    recovered_ids = tokenizer.encode_ordinary(text_delta)
                    if recovered_ids != stream.pending_token_ids:
                        self._resume_text_roundtrip_valid = False
                        self._resume_text_roundtrip_error = {
                            "status": "unavailable",
                            "reason": "text_delta_roundtrip_mismatch",
                            "stream_id": stream.stream_id,
                        }
                    output = FcGenerationTextDeltaOutput(
                        text=text_delta,
                        source_step_count=stream.pending_token_count,
                    )
                    stream.pending_token_count = 0
                    stream.pending_token_ids.clear()
                    stream.emitted_parts.append(text_delta)
                    if track == "non_spoken":
                        self._non_spoken_aggregate_parts.append(text_delta)
                    emitted_parts.append(text_delta)
                steps.append(
                    FcViewGenerationStep(
                        token_id=token_id,
                        stream_id=stream.stream_id,
                        track=track,
                        output=output,
                    )
                )
                continue

            semantic_key = self._semantic_key(token_id)
            if semantic_key is None:
                raise RuntimeError(
                    f"FC generated non-ordinary token has no semantic key: {token_id}"
                )
            stream_kind = start_keys[track].get(semantic_key)
            if stream_kind is not None:
                if stream is not None:
                    if track != "spoken" or stream.kind != stream_kind:
                        raise RuntimeError(
                            f"FC {track} stream opened before previous stream closed: "
                            f"{stream.stream_id} -> {stream_kind}"
                        )
                else:
                    stream = self._new_text_stream(stream_kind)
                    if track == "non_spoken":
                        span_started = stream_kind
                        if self._non_spoken_aggregate_kind not in {
                            None,
                            stream_kind,
                        }:
                            raise RuntimeError(
                                "non-spoken opener changed aggregate kind: "
                                f"{self._non_spoken_aggregate_kind} -> {stream_kind}"
                            )
                        self._non_spoken_aggregate_kind = stream_kind
                        self._next_non_spoken_continuation_kind = None
                    setattr(self, stream_attr, stream)
            if semantic_key == "listen" and self._spoken_text_stream is not None:
                raise RuntimeError(
                    "listen before spoken_turn_eos: "
                    f"active_stream={self._spoken_text_stream.stream_id}"
                )
            stream_id = (
                stream.stream_id
                if stream is not None
                else f"{track}_protocol"
            )
            steps.append(
                FcViewGenerationStep(
                    token_id=token_id,
                    stream_id=stream_id,
                    track=track,
                    output=FcGenerationProtocolOutput(semantic_key=semantic_key),
                )
            )
            expected_stream_kind = end_keys[track].get(semantic_key)
            if expected_stream_kind is not None:
                if stream is None or stream.kind != expected_stream_kind:
                    raise RuntimeError(
                        f"FC {track} end token without matching stream: "
                        f"end={semantic_key}, active={getattr(stream, 'kind', None)}"
                    )
                warning = self._terminate_text_stream(
                    track=track,
                    reason=semantic_key,
                    record_closed_span=track == "non_spoken",
                )
                if warning is not None:
                    warnings.append(warning)
                stream = None
            if (
                track == "non_spoken"
                and semantic_key
                in {
                    "no_action",
                    "non_spoken_eos",
                    "non_spoken_hold",
                    "non_spoken_abort",
                }
            ):
                self._next_non_spoken_continuation_kind = None
                self._non_spoken_aggregate_kind = None
                self._non_spoken_aggregate_parts = []
                self._non_spoken_aggregate_incomplete = False

        return steps, "".join(emitted_parts), warnings, span_started

    def _step_result(self, data: dict) -> FcNonSpokenGenerateResult:
        spans = [FcClosedSpan(**span) for span in data.get("closed_spans", []) or []]
        generation_steps, text_delta, warnings, span_started = self._decode_generation_steps(
            list(data.get("token_ids", []) or []),
            track="non_spoken",
        )
        for span in spans:
            if not self._closed_non_spoken_texts:
                raise RuntimeError(
                    f"closed span has no matching View text stream: {span.type}"
                )
            stream_kind, emitted_text, incomplete_boundary = (
                self._closed_non_spoken_texts.pop(0)
            )
            expected_text = span.text if span.type == "think" else span.wire
            if (
                stream_kind != span.type
                or (
                    not incomplete_boundary
                    and emitted_text != (expected_text or "")
                )
            ):
                raise RuntimeError(
                    "FC closed span text mismatch: "
                    f"stream={stream_kind}/{emitted_text!r}, "
                    f"span={span.type}/{expected_text!r}"
                )
        return FcNonSpokenGenerateResult(
            token_ids=data.get("token_ids", []),
            terminated=data.get("terminated", False),
            close_reason=data.get("close_reason"),
            closed_spans=spans,
            text=data.get("text", ""),
            text_delta=text_delta,
            span_started=span_started,
            generation_steps=generation_steps,
            warnings=warnings,
            audio_waveform=data.get("audio_waveform"),
            audio_sample_rate=data.get("audio_sample_rate"),
            n_tts_tokens=data.get("n_tts_tokens", 0),
            generation_flag=self._non_spoken_generation_flag(data),
            metadata={
                k: v
                for k, v in data.items()
                if k not in {
                    "token_ids",
                    "terminated",
                    "close_reason",
                    "closed_spans",
                    "text",
                    "audio_waveform",
                    "audio_sample_rate",
                    "n_tts_tokens",
                    "generation_flag",
                }
            },
        )

    @staticmethod
    def _prepare_result(data: dict) -> FcDuplexPrepareResult:
        resize_info = data.get("resize_info") or {}
        return FcDuplexPrepareResult(
            prefill_ids=data.get("prefill_ids", []),
            output_render=data.get("output_render", ""),
            resized=bool(resize_info.get("resized", False)),
            old_vocab_size=resize_info.get("old_vocab"),
            new_vocab_size=resize_info.get("new_vocab"),
            required_vocab_size=resize_info.get("need"),
            generate_audio=bool(data.get("generate_audio", False)),
            has_ref_audio=bool(data.get("has_ref_audio", False)),
            prompt_wav_path=data.get("prompt_wav_path"),
        )

    @staticmethod
    def _prefill_result(data: dict) -> FcDuplexPrefillResult:
        return FcDuplexPrefillResult(
            unit_index=data.get("unit", 0),
            n_audio_placeholders=data.get("n_audio", 0),
            has_input_event=data.get("has_event", False),
            is_listen=data.get("is_listen"),
            is_speaking=data.get("is_speaking", False),
            inserted_token_ids=data.get("inserted_token_ids", []),
            tool_events=list(data.get("tool_events") or []),
        )

    def _spoken_result(self, data: dict) -> FcSpokenGenerateResult:
        audio_waveform = data.get("audio_waveform")
        generation_steps, spoken_text_delta, warnings, _ = self._decode_generation_steps(
            list(data.get("spoken_ids", []) or []),
            track="spoken",
        )
        if bool(data.get("spoken_turn_eos", False)):
            tokenizer = self._ensure_protocol_tokenizer()
            spoken_slot_eos_id = tokenizer.token_to_id("<|spoken_slot_eos|>")
            spoken_slot_stream_id = next(
                (
                    step.stream_id
                    for step in reversed(generation_steps)
                    if (
                        isinstance(step.output, FcGenerationProtocolOutput)
                        and step.output.semantic_key == "spoken_turn_eos"
                    )
                ),
                "spoken_protocol",
            )
            generation_steps.append(
                FcViewGenerationStep(
                    token_id=spoken_slot_eos_id,
                    stream_id=spoken_slot_stream_id,
                    track="spoken",
                    output=FcGenerationProtocolOutput(
                        semantic_key="spoken_slot_eos"
                    ),
                )
            )
        return FcSpokenGenerateResult(
            is_listen=bool(data.get("is_listen", False)),
            is_speaking=bool(data.get("is_speaking", False)),
            spoken_token_ids=data.get("spoken_ids", []),
            spoken_text=data.get("spoken_text", data.get("text", "")),
            spoken_text_delta=spoken_text_delta,
            spoken_full_text=(
                self._last_closed_spoken_text
                if bool(data.get("spoken_turn_eos", False))
                else None
            ),
            generation_steps=generation_steps,
            warnings=warnings,
            spoken_turn_eos=bool(data.get("spoken_turn_eos", False)),
            audio_waveform=audio_waveform,
            audio_sample_rate=data.get("audio_sample_rate"),
            n_audio_samples=int(len(audio_waveform)) if audio_waveform is not None else 0,
            n_tts_tokens=data.get("n_tts_tokens", 0),
            cost_llm=data.get("cost_llm", 0.0),
            cost_tts_prep=data.get("cost_tts_prep", 0.0),
            cost_tts=data.get("cost_tts", 0.0),
            cost_token2wav=data.get("cost_token2wav", 0.0),
        )

    @staticmethod
    def _unit_info(data: dict) -> FcDuplexUnitInfo:
        spans = [FcClosedSpan(**span) for span in data.get("closed_spans", []) or []]
        return FcDuplexUnitInfo(
            unit=data.get("unit", 0),
            n_audio=data.get("n_audio", 0),
            has_event=data.get("has_event", False),
            is_listen=data.get("is_listen"),
            is_speaking=data.get("is_speaking", False),
            spoken_ids=data.get("spoken_ids", []),
            non_spoken_ids=data.get("non_spoken_ids", []),
            non_spoken_terminator=data.get("non_spoken_terminator"),
            closed_spans=spans,
            audio_sample_rate=data.get("audio_sample_rate"),
            n_audio_samples=data.get("n_audio_samples", 0),
        )

    @staticmethod
    def _decode_result(data: dict) -> FcDecodeOutputResult:
        units = [
            FcDecodedUnit(
                unit_index=index,
                is_listen=unit.get("is_listen"),
                spoken_text=unit.get("spoken_text", ""),
                non_spoken_terminator=unit.get("non_spoken_terminator"),
                raw_non_spoken=unit.get("raw_non_spoken", ""),
            )
            for index, unit in enumerate(data.get("units", []) or [])
        ]
        tool_calls = [FcDuplexView._decoded_tool_call(call) for call in data.get("tool_calls", []) or []]
        return FcDecodeOutputResult(
            units=units,
            spoken_text=data.get("spoken_text", ""),
            think_text=data.get("think_text", ""),
            tool_calls=tool_calls,
            output_ids=data.get("output_ids", []),
            output_render=data.get("output_render", ""),
        )

    @staticmethod
    def _decoded_tool_call(call: Dict[str, Any]) -> FcDecodedToolCall:
        return FcDecodedToolCall(
            tool_call_id=call.get("tool_call_id"),
            name=call.get("name"),
            arguments=call.get("arguments"),
            error=call.get("error"),
            wire=call.get("wire"),
        )

    @staticmethod
    def _tool_calls_semantic(tool_calls: List[FcDecodedToolCall]) -> List[Dict[str, Any]]:
        return [
            {"name": call.name, "arguments": call.arguments, "error": call.error}
            for call in (tool_calls or [])
        ]

    @staticmethod
    def _first_diff(gt: str, pred: str) -> Optional[FcTokenStreamDiff]:
        if gt == pred:
            return None
        n_chars = min(len(gt), len(pred))
        index = next((i for i in range(n_chars) if gt[i] != pred[i]), n_chars)
        return FcTokenStreamDiff(
            index=index,
            gt_context=gt[max(0, index - 80): index + 160],
            pred_context=pred[max(0, index - 80): index + 160],
        )

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _save_audio_artifact(output_dir: Path, waveforms: List[Any], sample_rate: int = 24000) -> FcDuplexAudioArtifact:
        import soundfile as sf

        audio_dir = output_dir / "pred_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        unit_audio_paths = []
        arrays = []
        for index, waveform in enumerate(waveforms or []):
            if waveform is None:
                continue
            array = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if array.size == 0:
                continue
            unit_path = audio_dir / f"pred_audio_unit_{index:03d}.wav"
            sf.write(unit_path, array, sample_rate)
            unit_audio_paths.append(str(unit_path))
            arrays.append(array)

        full_audio_path = None
        if arrays:
            full_audio_path = audio_dir / "pred_audio_full.wav"
            sf.write(full_audio_path, np.concatenate(arrays), sample_rate)

        return FcDuplexAudioArtifact(
            sample_rate=sample_rate,
            unit_audio_paths=unit_audio_paths,
            full_audio_path=str(full_audio_path) if full_audio_path else None,
            n_audio_units=len(unit_audio_paths),
        )

    def prepare(self, request: FcDuplexPrepareRequest, tool_call_id_generator: Optional[Any] = None) -> FcDuplexPrepareResult:
        import librosa

        self.tool_call_manager = ToolCallStateManager(tool_call_id_generator)
        ref_audio = None
        if request.ref_audio_path:
            if request.ref_audio_path not in self._ref_audio_cache:
                self._ref_audio_cache[request.ref_audio_path], _ = librosa.load(
                    request.ref_audio_path,
                    sr=16000,
                    mono=True,
                )
            ref_audio = self._ref_audio_cache[request.ref_audio_path]
        self._resume_ref_audio_sha256 = (
            hashlib.sha256(
                np.asarray(ref_audio, dtype=np.float32).tobytes()
            ).hexdigest()
            if ref_audio is not None
            else None
        )
        effective_prompt_wav_path = (
            request.prompt_wav_path or request.ref_audio_path
        )
        self._resume_prompt_wav_sha256 = None
        if effective_prompt_wav_path:
            prompt_path = Path(effective_prompt_wav_path)
            if prompt_path.is_file():
                self._resume_prompt_wav_sha256 = hashlib.sha256(
                    prompt_path.read_bytes()
                ).hexdigest()
        result = self._model.fc_duplex_prepare(
            system_prompt=request.system_prompt,
            tools=request.tools,
            ref_audio=ref_audio,
            prompt_wav_path=request.prompt_wav_path or request.ref_audio_path,
            generate_audio=request.generate_audio,
        )
        self._protocol_tokenizer = None
        self._stream_seq = 0
        self._non_spoken_text_stream = None
        self._spoken_text_stream = None
        self._closed_non_spoken_texts = []
        self._next_non_spoken_continuation_kind = None
        self._non_spoken_aggregate_kind = None
        self._non_spoken_aggregate_parts = []
        self._non_spoken_aggregate_incomplete = False
        self._last_closed_spoken_text = None
        self._resume_text_roundtrip_valid = True
        self._resume_text_roundtrip_error = None
        self._ensure_protocol_tokenizer()
        return self._prepare_result(result)

    def streaming_prefill(self, request: FcDuplexPrefillRequest) -> FcDuplexPrefillResult:
        import librosa

        audio_waveform = self._audio_from_base64(request.audio_data)
        if request.audio_path and audio_waveform is None:
            audio_waveform, _ = librosa.load(request.audio_path, sr=request.sample_rate, mono=True)
        tool_events = []
        tool_events.extend(self.tool_call_manager.consume_pending_started_events())
        tool_events.extend(self.tool_call_manager.consume_pending_error_responses())
        tool_events.extend(self.tool_call_manager.validate_and_mark_responses(request.tool_responses))
        result = self._model.fc_duplex_streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=request.frame_list,
            tool_responses=tool_events or None,
            sample_rate=request.sample_rate,
        )
        result = dict(result)
        result["tool_events"] = list(tool_events)
        return self._prefill_result(result)

    def streaming_spoken_generate(self, request: FcSpokenGenerateRequest) -> FcSpokenGenerateResult:
        result = self._model.fc_duplex_streaming_spoken_generate(
            max_tokens=request.max_tokens,
            decode_mode=request.decode_mode,
        )
        return self._spoken_result(result)

    def streaming_non_spoken_generate(self, request: FcNonSpokenGenerateRequest) -> FcNonSpokenGenerateResult:
        result = self._model.fc_duplex_streaming_non_spoken_generate(
            decode_mode=request.decode_mode,
            max_tokens=request.max_tokens,
            close_reason=request.close_reason,
        )
        self._attach_tool_call_ids(result)
        return self._step_result(result)

    def terminate_non_spoken_text_stream(
        self,
        reason: str,
    ) -> FcGenerationStreamTerminationResult:
        """Terminate the View decoder at an externally-triggered slot close."""

        semantic_key_by_reason = {
            "budget_reached": "non_spoken_budget_reached",
            "eos": "non_spoken_eos",
            "no_action": "no_action",
            "hold": "non_spoken_hold",
            "abort": "non_spoken_abort",
        }
        display_name_by_reason = {
            "budget_reached": "<|non_spoken_budget_reached|>",
            "eos": "<|non_spoken_eos|>",
            "no_action": "<|no_action|>",
            "hold": "<|non_spoken_hold|>",
            "abort": "<|non_spoken_abort|>",
        }
        if reason not in semantic_key_by_reason:
            raise ValueError(f"unsupported non-spoken stream close reason: {reason}")
        stream_id = (
            self._non_spoken_text_stream.stream_id
            if self._non_spoken_text_stream is not None
            else "non_spoken_protocol"
        )
        if (
            reason == "budget_reached"
            and self._non_spoken_text_stream is not None
        ):
            self._next_non_spoken_continuation_kind = (
                self._non_spoken_text_stream.kind
            )
        warning = self._terminate_text_stream(
            track="non_spoken",
            reason=reason,
        )
        tokenizer = self._ensure_protocol_tokenizer()
        step = FcViewGenerationStep(
            token_id=tokenizer.token_to_id(display_name_by_reason[reason]),
            stream_id=stream_id,
            track="non_spoken",
            output=FcGenerationProtocolOutput(
                semantic_key=semantic_key_by_reason[reason],
                deferred_model_feed=reason == "budget_reached",
            ),
        )
        return FcGenerationStreamTerminationResult(
            generation_steps=[step],
            warnings=[warning] if warning is not None else [],
        )

    def _attach_tool_call_ids(self, result: dict) -> None:
        for span in result.get("closed_spans", []) or []:
            if span.get("type") != "tool_call":
                continue
            tool_call = span.get("tool_call")
            wire = span.get("wire")
            error = span.get("error")
            if isinstance(tool_call, dict) and tool_call.get("error"):
                error = tool_call.get("error")
            if error or not isinstance(tool_call, dict) or not tool_call.get("name"):
                call_id = self.tool_call_manager.register_parse_error(
                    error or "tool call parse failed",
                    wire=wire,
                )
                span["tool_call_id"] = call_id
                span["error"] = error or "tool call parse failed"
                continue
            call_id = self.tool_call_manager.register_tool_call(tool_call, wire=wire)
            span["tool_call_id"] = call_id
            tool_call["tool_call_id"] = call_id

    def finalize_unit(self, request: Optional[FcFinalizeUnitRequest] = None) -> FcDuplexUnitInfo:
        del request
        return self._unit_info(self._model.fc_duplex_finalize_unit())

    def resume_boundary_status(self) -> Dict[str, Any]:
        """Return whether the current Unit boundary is reconstructable from public text."""

        if self.tool_call_manager.has_state:
            return {
                "status": "unavailable",
                "reason": "pending_tool_result",
            }
        if not self._resume_text_roundtrip_valid:
            return dict(
                self._resume_text_roundtrip_error
                or {
                    "status": "unavailable",
                    "reason": "text_delta_roundtrip_mismatch",
                }
            )
        if self._next_non_spoken_continuation_kind is not None:
            return {
                "status": "unavailable",
                "reason": "unsupported_open_span",
                "stream_kind": self._next_non_spoken_continuation_kind,
            }
        for stream in (self._non_spoken_text_stream, self._spoken_text_stream):
            if stream is None:
                continue
            if stream.pending_token_count:
                return {
                    "status": "unavailable",
                    "reason": "pending_text_delta",
                    "stream_id": stream.stream_id,
                }
            return {
                "status": "unavailable",
                "reason": (
                    "unsupported_spoken_turn_state"
                    if stream.kind == "spoken"
                    else "unsupported_open_span"
                ),
                "stream_id": stream.stream_id,
            }
        capability = getattr(self._model, "fc_duplex", None)
        capability_status = getattr(capability, "resume_boundary_status", None)
        if capability_status is not None:
            return dict(capability_status())
        return {"status": "available"}

    def resume_identity(self) -> Dict[str, Any]:
        """Return the model/tokenizer identity required for stateless resume."""

        tokenizer = self._ensure_protocol_tokenizer()
        fingerprint = tokenizer.fingerprint
        model_config = getattr(self._model, "config", None)
        model_name = str(
            getattr(model_config, "_name_or_path", None)
            or getattr(self._model, "name_or_path", None)
            or "unknown"
        )
        return {
            "protocol_version": "fc-duplex-semantic-v2",
            "model": model_name,
            "tokenizer_target": tokenizer.target,
            "tokenizer_fingerprint": {
                "vocab_hash": fingerprint.vocab_hash,
                "merges_hash": fingerprint.merges_hash,
            },
            "ref_audio_sha256": self._resume_ref_audio_sha256,
            "prompt_wav_sha256": self._resume_prompt_wav_sha256,
        }

    def replay_completed_unit(
        self,
        *,
        audio_data: Optional[str],
        frame_list: Optional[List[Any]],
        tool_responses: Optional[List[Any]],
        sample_rate: int,
        spoken_token_ids: List[int],
        non_spoken_token_ids: List[int],
        deferred_non_spoken_close: bool,
    ) -> FcDuplexUnitInfo:
        """Deterministically feed one historical Unit without sampling outputs."""

        audio_waveform = self._audio_from_base64(audio_data)
        result = self._model.fc_duplex_replay_completed_unit(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            tool_responses=tool_responses,
            sample_rate=sample_rate,
            spoken_token_ids=spoken_token_ids,
            non_spoken_token_ids=non_spoken_token_ids,
            deferred_non_spoken_close=deferred_non_spoken_close,
        )
        return self._unit_info(result)

    def restore_generation_stream_sequence(self, next_stream_sequence: int) -> None:
        """Advance future stream IDs after stateless replay without restoring open streams."""

        if next_stream_sequence < 1:
            raise ValueError(
                f"next_stream_sequence 必须 >= 1: {next_stream_sequence}"
            )
        self._stream_seq = next_stream_sequence - 1

    def restore_tool_call_sequence(self, tool_call_count: int) -> None:
        """Advance internal tool-call IDs after stateless history replay."""

        self.tool_call_manager.restore_completed_sequence(tool_call_count)

    def decode_output(
        self,
        request: Optional[FcDecodeOutputRequest] = None,
        output_ids: Optional[List[int]] = None,
        tools=None,
    ) -> FcDecodeOutputResult:
        if request is not None:
            output_ids = request.output_ids
            tools = request.tools
        return self._decode_result(self._model.fc_duplex_decode_output_ids(output_ids=output_ids, tools=tools))

    def trace_snapshot(self, *, session_id: Optional[str] = None, reason: Optional[str] = None) -> dict:
        return self._model.fc_duplex_trace_snapshot(session_id=session_id, reason=reason)

    def dump_trace(self, path: str, *, session_id: Optional[str] = None, reason: Optional[str] = None) -> dict:
        return self._model.fc_duplex_dump_trace(path=path, session_id=session_id, reason=reason)

    def cleanup(self) -> None:
        self._model.fc_duplex_cleanup()
        self._protocol_tokenizer = None
        self._stream_seq = 0
        self._non_spoken_text_stream = None
        self._spoken_text_stream = None
        self._closed_non_spoken_texts = []
        self._next_non_spoken_continuation_kind = None
        self._non_spoken_aggregate_kind = None
        self._non_spoken_aggregate_parts = []
        self._non_spoken_aggregate_incomplete = False
        self._last_closed_spoken_text = None
        self._resume_text_roundtrip_valid = True
        self._resume_text_roundtrip_error = None
        self._resume_ref_audio_sha256 = None
        self._resume_prompt_wav_sha256 = None

    def offline_inference(
        self,
        task_input: FcDuplexOfflineInput,
        non_spoken_budget_per_unit: Optional[int] = None,
    ) -> FcDuplexOfflineOutput:
        import librosa

        start_time = time.time()
        debug_budget_override = non_spoken_budget_per_unit
        units_info = []

        try:
            id_generator = FixedToolCallIdGenerator(task_input.tool_call_ids) if task_input.tool_call_ids else None
            self.prepare(
                FcDuplexPrepareRequest(
                    system_prompt=task_input.system_prompt,
                    tools=task_input.tools,
                    ref_audio_path=task_input.ref_audio_path,
                    prompt_wav_path=task_input.prompt_wav_path,
                    generate_audio=task_input.generate_audio,
                ),
                tool_call_id_generator=id_generator,
            )

            sample_rate = task_input.config.sample_rate
            samples_per_unit = max(1, int(round(task_input.config.unit_sec * sample_rate)))
            if task_input.unit_audio_chunks is not None:
                chunks = [
                    np.asarray(chunk, dtype=np.float32).reshape(-1)
                    for chunk in task_input.unit_audio_chunks
                ]
            else:
                audio = self._audio_from_base64(task_input.audio_data)
                if task_input.user_audio_path and audio is None:
                    audio, _ = librosa.load(task_input.user_audio_path, sr=sample_rate, mono=True)
                if audio is None:
                    audio = np.zeros(samples_per_unit, dtype=np.float32)
                chunks = [
                    audio[i:i + samples_per_unit]
                    for i in range(0, len(audio), samples_per_unit)
                ] or [np.zeros(samples_per_unit, dtype=np.float32)]
            n_audio_units = len(chunks)
            total_units = max(1, n_audio_units + task_input.config.extra_response_units)
            silence = np.zeros(samples_per_unit, dtype=np.float32)
            scheduled_tool_responses: Dict[int, List[FcToolResponse]] = {}
            audio_waveforms = []

            for unit_idx in range(total_units):
                chunk = chunks[unit_idx] if unit_idx < n_audio_units else silence
                if len(chunk) < samples_per_unit:
                    chunk = np.pad(chunk, (0, samples_per_unit - len(chunk)))

                responses = list(scheduled_tool_responses.pop(unit_idx, []))
                responses.extend(task_input.tool_responses_by_unit.get(unit_idx) or [])
                self.streaming_prefill(FcDuplexPrefillRequest(
                    audio_data=base64.b64encode(chunk.astype(np.float32).tobytes()).decode("utf-8"),
                    tool_responses=responses or None,
                    sample_rate=sample_rate,
                ))
                spoken_step = self.streaming_spoken_generate(FcSpokenGenerateRequest(
                    max_tokens=task_input.config.max_spoken_tokens,
                    decode_mode=task_input.config.decode_mode,
                ))
                if spoken_step.audio_waveform is not None:
                    audio_waveforms.append(spoken_step.audio_waveform)

                has_budget_source = True
                if debug_budget_override is not None:
                    unit_budget = debug_budget_override
                elif spoken_step.is_speaking and task_input.non_spoken_budgets_while_speaking:
                    unit_budget = task_input.non_spoken_budgets_while_speaking[min(
                        unit_idx,
                        len(task_input.non_spoken_budgets_while_speaking) - 1,
                    )]
                elif (not spoken_step.is_speaking) and task_input.non_spoken_budgets_while_listening:
                    unit_budget = task_input.non_spoken_budgets_while_listening[min(
                        unit_idx,
                        len(task_input.non_spoken_budgets_while_listening) - 1,
                    )]
                else:
                    unit_budget = task_input.config.non_spoken_budget_per_unit
                    has_budget_source = unit_budget is not None
                if not has_budget_source:
                    raise RuntimeError(
                        "offline FC duplex non-spoken budget must come from "
                        "TrainingData UnitPolicy or an explicit override"
                    )

                terminated = False
                steps = 0
                while unit_budget is None or steps < unit_budget:
                    steps += 1
                    step = self.streaming_non_spoken_generate(FcNonSpokenGenerateRequest(
                        max_tokens=1,
                        decode_mode=task_input.config.decode_mode,
                    ))
                    for span in step.closed_spans:
                        if span.type != "tool_call" or not span.tool_call_id:
                            continue
                        if span.tool_call_id in task_input.tool_responses_by_call_id:
                            response_unit = unit_idx + 2
                            scheduled_tool_responses.setdefault(response_unit, []).append(
                                FcToolResponse(
                                    call_id=span.tool_call_id,
                                    content=task_input.tool_responses_by_call_id[span.tool_call_id],
                                )
                            )
                    if step.terminated:
                        terminated = True
                        break
                if not terminated:
                    self.streaming_non_spoken_generate(FcNonSpokenGenerateRequest(
                        max_tokens=0,
                        decode_mode=task_input.config.decode_mode,
                        close_reason="budget_reached",
                    ))
                units_info.append(self.finalize_unit())

            decoded = self.decode_output(FcDecodeOutputRequest(tools=task_input.tools))
            return FcDuplexOfflineOutput(
                success=True,
                output_ids=decoded.output_ids,
                output_render=decoded.output_render,
                spoken_text=decoded.spoken_text,
                think_text=decoded.think_text,
                tool_calls=self.tool_call_manager.tool_calls or decoded.tool_calls,
                units_info=units_info,
                audio_waveforms=audio_waveforms,
                total_units=len(units_info),
                n_audio_units=n_audio_units,
                total_duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as exc:
            logger.exception("FC duplex offline inference failed")
            return FcDuplexOfflineOutput(
                success=False,
                error=str(exc),
                total_duration_ms=(time.time() - start_time) * 1000,
            )

    def _load_train_data_structure(self, request: FcDuplexTrainDataRequest) -> tuple:
        if request.train_data_path:
            source_path = Path(request.train_data_path)
            structure = json.loads(source_path.read_text(encoding="utf-8"))
            sample_id = source_path.stem
            data_root = Path(request.data_root) if request.data_root else source_path.parent
            return structure, source_path, sample_id, data_root

        if request.train_data is None:
            raise ValueError("train_data_path or train_data must be provided")

        source_path = None
        train_data = request.train_data
        if isinstance(train_data, dict):
            structure = train_data
        elif hasattr(train_data, "model_dump"):
            structure = train_data.model_dump()
        elif hasattr(train_data, "structure"):
            structure = train_data.structure
        else:
            raise TypeError(f"unsupported train_data type: {type(train_data)!r}")

        sample_id = str(structure.get("data_id") or "train_data").split(":")[0]
        if request.data_root is None:
            raise ValueError("data_root is required when train_data_path is not provided")
        return structure, source_path, sample_id, Path(request.data_root)

    @staticmethod
    def _extract_train_tools(structure: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        from minicpm_o5_sdk import OpenAIToolDefinition

        raw_tools = structure.get("system", {}).get("tools") or []
        return [OpenAIToolDefinition.model_validate(tool).model_dump() for tool in raw_tools] or None

    @staticmethod
    def _extract_train_system_prompt(structure: Dict[str, Any]) -> str:
        return "\n".join(
            segment["text"]
            for segment in structure.get("system", {}).get("segments", [])
            if segment.get("kind") == "text"
        )

    @staticmethod
    def _extract_train_tool_call_ids(structure: Dict[str, Any]) -> List[str]:
        tool_call_ids = []
        segments = ((structure.get("tracks") or {}).get("ai_non_spoken") or {}).get("segments") or []
        for segment in segments:
            content = segment.get("content") or {}
            if content.get("kind") == "tool_call" and content.get("tool_call_id"):
                tool_call_ids.append(content["tool_call_id"])
        return tool_call_ids

    @staticmethod
    def _extract_train_tool_responses(structure: Dict[str, Any]) -> Dict[str, Any]:
        responses = {}
        segments = ((structure.get("tracks") or {}).get("input_event") or {}).get("segments") or []
        for segment in segments:
            event = segment.get("event") or {}
            if event.get("kind") != "tool_response":
                continue
            call_id = event.get("tool_call_id")
            if not call_id:
                continue
            text_parts = [
                item.get("text", "")
                for item in event.get("contents", [])
                if item.get("kind") == "text"
            ]
            responses[call_id] = "".join(text_parts)
        return responses

    @staticmethod
    def _load_sdk_train_data(structure: Dict[str, Any], data_root: Path):
        from minicpm_o5_sdk import O5DuplexTrainingData, O5TokenizerID

        training_data = O5DuplexTrainingData.load_structure(
            structure,
            data_root=data_root,
        )
        tokenized_result = training_data.tokenize(tokenizer_id=O5TokenizerID.O45_FC)
        return training_data, tokenized_result

    @staticmethod
    def _resolve_train_media_path(data_root: Path, file_path: Optional[str]) -> Optional[Path]:
        if not file_path:
            return None
        path = Path(file_path)
        if path.is_absolute():
            return path
        return data_root / path

    @classmethod
    def _extract_system_ref_audio_path(cls, structure: Dict[str, Any], data_root: Path) -> Optional[str]:
        for segment in structure.get("system", {}).get("segments", []) or []:
            if segment.get("kind") != "audio":
                continue
            audio = segment.get("audio") or {}
            path = cls._resolve_train_media_path(data_root, audio.get("file_path"))
            if path is not None and path.exists():
                return str(path)
        return None

    @staticmethod
    def _build_unit_audio_chunks_from_arrangement(training_data: Any, arrangement: Any, config: FcDuplexConfig) -> List[np.ndarray]:
        from minicpm_o5_sdk.protocols.duplex.tokenization.span import compute_total_unit_count

        sample_rate = config.sample_rate
        unit_sec = float(getattr(arrangement.unit_policy, "unit_sec", config.unit_sec))
        samples_per_unit = max(1, int(round(unit_sec * sample_rate)))
        total_units = max(1, compute_total_unit_count(arrangement))
        chunks = [
            np.zeros(samples_per_unit, dtype=np.float32)
            for _ in range(total_units)
        ]

        user_audio_track = getattr(getattr(arrangement, "tracks", None), "user_audio", None)
        if user_audio_track is None:
            return chunks

        source_segments = getattr(getattr(training_data, "tracks", None), "user_audio", None)
        source_segments = getattr(source_segments, "segments", []) if source_segments is not None else []
        for index, arranged_segment in enumerate(user_audio_track.segments):
            if index >= len(source_segments):
                continue
            waveform = source_segments[index].audio.get_tensor().detach().cpu().numpy().astype(np.float32)
            timeline = arranged_segment.timeline
            segment_start_sec = float(timeline.timeline_start_sec)
            segment_end_sec = float(timeline.timeline_end_sec)
            start_unit = max(0, int(timeline.start_unit_index))
            end_unit = min(total_units, int(timeline.end_unit_index_exclusive))
            for unit_index in range(start_unit, end_unit):
                unit_start_sec = unit_index * unit_sec
                unit_end_sec = (unit_index + 1) * unit_sec
                overlap_start_sec = max(unit_start_sec, segment_start_sec)
                overlap_end_sec = min(unit_end_sec, segment_end_sec)
                if overlap_end_sec <= overlap_start_sec:
                    continue

                chunk_start = int(round((overlap_start_sec - unit_start_sec) * sample_rate))
                chunk_end = int(round((overlap_end_sec - unit_start_sec) * sample_rate))
                audio_start = int(round((overlap_start_sec - segment_start_sec) * sample_rate))
                audio_end = audio_start + max(0, chunk_end - chunk_start)

                chunk_start = max(0, min(samples_per_unit, chunk_start))
                chunk_end = max(chunk_start, min(samples_per_unit, chunk_end))
                audio_start = max(0, min(len(waveform), audio_start))
                audio_end = max(audio_start, min(len(waveform), audio_end))
                n = min(chunk_end - chunk_start, audio_end - audio_start)
                if n > 0:
                    chunks[unit_index][chunk_start:chunk_start + n] += waveform[audio_start:audio_start + n]
        return chunks

    @staticmethod
    def _sdk_budget_to_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError("budget must be an integer or O5NoBudgetLimit, not bool")
        if isinstance(value, int):
            return int(value)
        if value.__class__.__name__ == "O5NoBudgetLimit":
            return None
        if isinstance(value, dict) and value.get("__class__") == "O5NoBudgetLimit":
            return None
        raise TypeError(f"unsupported SDK budget type: {type(value).__name__}")

    @classmethod
    def _build_non_spoken_budget_lists_from_arrangement(cls, arrangement: Any) -> tuple[List[Optional[int]], List[Optional[int]]]:
        from minicpm_o5_sdk.protocols.duplex.tokenization.span import compute_total_unit_count

        total_units = max(1, compute_total_unit_count(arrangement))
        listening = [
            cls._sdk_budget_to_int_or_none(
                arrangement.get_non_spoken_budget_while_listening(unit_index)
            )
            for unit_index in range(total_units)
        ]
        speaking = [
            cls._sdk_budget_to_int_or_none(
                arrangement.get_non_spoken_budget_while_speaking(unit_index)
            )
            for unit_index in range(total_units)
        ]
        return listening, speaking

    def offline_inference_from_train_data(self, request: FcDuplexTrainDataRequest) -> FcDuplexTrainDataResult:
        start_time = time.time()
        try:
            structure, source_path, sample_id, data_root = self._load_train_data_structure(request)
            training_data, tokenized_result = self._load_sdk_train_data(structure, data_root)
            arrangement = tokenized_result.arrangement
            tokenized_data = tokenized_result.tokenized_data
            unit_audio_chunks = self._build_unit_audio_chunks_from_arrangement(
                training_data,
                arrangement,
                request.config,
            )
            budget_listening, budget_speaking = self._build_non_spoken_budget_lists_from_arrangement(arrangement)

            user_audio_path = None
            user_audio_track = (structure.get("tracks") or {}).get("user_audio") or {}
            user_audio_segments = user_audio_track.get("segments") or []
            if user_audio_segments:
                user_audio_path = self._resolve_train_media_path(
                    data_root,
                    ((user_audio_segments[0].get("audio") or {}).get("file_path")),
                )

            tools = self._extract_train_tools(structure)
            system_prompt = self._extract_train_system_prompt(structure)
            tool_call_ids = self._extract_train_tool_call_ids(structure) if request.use_train_tool_call_ids else []
            tool_responses_by_call_id = (
                self._extract_train_tool_responses(structure)
                if request.inject_train_tool_responses
                else {}
            )

            gt_output_ids = list(tokenized_data.input_ids)
            gt_decoded = self.decode_output(FcDecodeOutputRequest(output_ids=gt_output_ids, tools=tools))

            system_ref_audio_path = self._extract_system_ref_audio_path(structure, data_root)
            ref_audio_path = request.ref_audio_path or system_ref_audio_path or (str(user_audio_path) if user_audio_path else None)
            prompt_wav_path = request.prompt_wav_path or ref_audio_path
            pred = self.offline_inference(
                FcDuplexOfflineInput(
                    system_prompt=system_prompt,
                    tools=tools,
                    user_audio_path=str(user_audio_path) if user_audio_path else None,
                    unit_audio_chunks=unit_audio_chunks,
                    ref_audio_path=ref_audio_path,
                    prompt_wav_path=prompt_wav_path if request.generate_audio else None,
                    generate_audio=request.generate_audio,
                    tool_call_ids=tool_call_ids or None,
                    tool_responses_by_call_id=tool_responses_by_call_id,
                    non_spoken_budgets_while_listening=budget_listening,
                    non_spoken_budgets_while_speaking=budget_speaking,
                    config=request.config,
                ),
                non_spoken_budget_per_unit=request.non_spoken_budget_per_unit,
            )

            pred_tool_calls = [
                call if isinstance(call, FcDecodedToolCall) else self._decoded_tool_call(call)
                for call in pred.tool_calls
            ]
            comparison = FcDuplexComparisonResult(
                token_ids_exact=gt_output_ids == pred.output_ids,
                rendered_token_stream_exact=gt_decoded.output_render == pred.output_render,
                spoken_text_exact=gt_decoded.spoken_text == pred.spoken_text,
                think_text_exact=gt_decoded.think_text == pred.think_text,
                tool_calls_semantic_exact=self._tool_calls_semantic(gt_decoded.tool_calls) == self._tool_calls_semantic(pred_tool_calls),
                tool_call_ids_exact=tool_call_ids == [call.tool_call_id for call in pred_tool_calls],
                first_rendered_token_stream_diff=self._first_diff(gt_decoded.output_render, pred.output_render),
            )

            artifact_dir = Path(request.output_artifact_dir) if request.output_artifact_dir else None
            audio_artifact = None
            if artifact_dir is not None:
                self._write_json(artifact_dir / "source.json", structure)
                self._write_text(artifact_dir / "gt_token_stream.txt", gt_decoded.output_render)
                self._write_text(artifact_dir / "pred_token_stream.txt", pred.output_render)
                self._write_json(artifact_dir / "units_info.json", [unit.model_dump() for unit in pred.units_info])
                if request.generate_audio and pred.success:
                    audio_artifact = self._save_audio_artifact(artifact_dir, pred.audio_waveforms)

            result = FcDuplexTrainDataResult(
                sample_id=sample_id,
                success=pred.success,
                error=pred.error,
                source_path=str(source_path) if source_path else None,
                user_audio_path=str(user_audio_path) if user_audio_path else None,
                gt_output_ids=gt_output_ids,
                pred_output_ids=pred.output_ids,
                gt_output_render=gt_decoded.output_render,
                pred_output_render=pred.output_render,
                gt_spoken_text=gt_decoded.spoken_text,
                pred_spoken_text=pred.spoken_text,
                gt_think_text=gt_decoded.think_text,
                pred_think_text=pred.think_text,
                gt_tool_calls=gt_decoded.tool_calls,
                pred_tool_calls=pred_tool_calls,
                tool_call_ids=tool_call_ids,
                tool_responses_by_call_id=tool_responses_by_call_id,
                units_info=pred.units_info,
                comparison=comparison,
                audio_artifact=audio_artifact,
                total_duration_ms=(time.time() - start_time) * 1000,
            )
            if artifact_dir is not None:
                self._write_json(artifact_dir / "comparison.json", result)
            return result
        except Exception as exc:
            logger.exception("FC duplex train-data offline inference failed")
            return FcDuplexTrainDataResult(
                sample_id=Path(request.train_data_path).stem if request.train_data_path else "",
                success=False,
                error=str(exc),
                source_path=request.train_data_path,
                total_duration_ms=(time.time() - start_time) * 1000,
            )


# ============================================================
# UnifiedProcessor：统一入口
# ============================================================

class UnifiedProcessor(BaseProcessor):
    """Unified processor — load once, hot-switch between Chat/Streaming/Duplex.

    Key features:
    - Model loaded once, shared across all modes
    - Mode switching < 1ms
    - Each mode returns a dedicated View with type-safe API

    Usage:
        >>> processor = UnifiedProcessor(model_path=..., pt_path=...)
        >>>
        >>> # Chat mode
        >>> chat = processor.set_chat_mode()
        >>> response = chat.chat(request)
        >>>
        >>> # Half-Duplex mode
        >>> half_duplex = processor.set_half_duplex_mode()
        >>> half_duplex.prefill(request)
        >>> for chunk in half_duplex.generate(session_id):
        ...     print(chunk.text_delta, end="")
        >>>
        >>> # Duplex mode
        >>> duplex = processor.set_duplex_mode()
        >>> duplex.prepare(...)
        >>> result = duplex.generate()

    Attributes:
        model_path: Base model path (HuggingFace format directory).
        pt_path: Optional extra .pt weights path (overrides base model weights).
        device: Target device.
        ref_audio_path: Default reference audio path.
        model: MiniCPMO unified model instance.
    """

    def __init__(
        self,
        model_path: str,
        pt_path: Optional[str] = None,
        device: str = "cuda",
        ref_audio_path: Optional[str] = None,
        duplex_config: Optional[DuplexConfig] = None,
        preload_both_tts: bool = True,
        compile: bool = False,
        chat_vocoder: str = "token2wav",
        attn_implementation: str = "auto",
    ):
        """Initialize the unified processor.

        Args:
            model_path: Base model path (HuggingFace format directory).
            pt_path: Optional extra .pt weights path (overrides base weights).
            device: Target device.
            ref_audio_path: Default reference audio path for TTS voice cloning.
                If None, TTS requests fail-fast when the client also omits it.
            duplex_config: Duplex configuration.
            preload_both_tts: Whether to preload both TTS vocoders (recommended True).
            compile: Whether to apply torch.compile to core sub-modules.
            chat_vocoder: Chat mode vocoder ("token2wav" or "cosyvoice2").
            attn_implementation: Attention implementation
                ("auto" / "flash_attention_2" / "sdpa" / "eager").
        """
        self.pt_path = pt_path
        self.ref_audio_path = ref_audio_path
        self.duplex_config = duplex_config or DuplexConfig()
        self.preload_both_tts = preload_both_tts
        self.compile = compile
        self.chat_vocoder = chat_vocoder
        self.attn_implementation = attn_implementation

        # View instances (lazily created)
        self._chat_view: Optional[ChatView] = None
        self._half_duplex_view: Optional[HalfDuplexView] = None
        self._duplex_view: Optional[DuplexView] = None
        self._fc_duplex_view: Optional[FcDuplexView] = None

        # Current mode
        self._current_mode: Optional[ProcessorMode] = None

        super().__init__(model_path=model_path, device=device)

    @property
    def mode(self) -> ProcessorMode:
        """Current processor mode."""
        return self._current_mode or ProcessorMode.HALF_DUPLEX

    def _resolve_attn_implementation(self) -> str:
        """Resolve the actual attention implementation to use.

        When configured as "auto", auto-detects the environment:
        - flash-attn available -> flash_attention_2
        - flash-attn unavailable -> sdpa

        When configured explicitly, uses the value directly (fail-fast if unavailable).

        Returns:
            The resolved attn_implementation string.
        """
        configured = self.attn_implementation

        if configured != "auto":
            if configured == "flash_attention_2":
                try:
                    from transformers.utils import is_flash_attn_2_available
                    if not is_flash_attn_2_available():
                        raise RuntimeError(
                            "config.json specifies attn_implementation='flash_attention_2', "
                            "but flash-attn is not installed or unavailable.\n"
                            "Solutions:\n"
                            "  1. Install flash-attn: MAX_JOBS=16 pip install 'flash-attn>=2.6' --no-build-isolation\n"
                            "  2. Or set to 'auto'/'sdpa' to use PyTorch built-in SDPA"
                        )
                except ImportError:
                    raise RuntimeError(
                        "config.json specifies attn_implementation='flash_attention_2', "
                        "but transformers.utils.is_flash_attn_2_available is not available."
                    )
            logger.info(f"[Attention] Using user-specified: {configured}")
            return configured

        # Auto mode: detect flash-attn availability
        try:
            from transformers.utils import is_flash_attn_2_available
            flash_available = is_flash_attn_2_available()
        except ImportError:
            flash_available = False

        if flash_available:
            try:
                import flash_attn
                flash_version = flash_attn.__version__
            except (ImportError, AttributeError):
                flash_version = "unknown"
            logger.info(
                f"[Attention] auto -> flash_attention_2 "
                f"(flash-attn {flash_version} available, best performance)"
            )
            return "flash_attention_2"
        else:
            logger.info(
                "[Attention] auto -> sdpa "
                "(flash-attn unavailable, using PyTorch built-in SDPA. "
                "For flash_attention_2, install: "
                "MAX_JOBS=16 pip install 'flash-attn>=2.6' --no-build-isolation)"
            )
            return "sdpa"

    def _is_quantized_model(self, model_path: str) -> bool:
        """Check if the model at *model_path* uses quantization (AWQ / GPTQ / BnB).

        Reads ``config.json`` in the model directory and looks for a
        ``quantization_config`` section with a ``quant_method``.
        """
        config_file = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_file):
            return False
        try:
            import json as _json
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = _json.load(f)
            qcfg = cfg.get("quantization_config")
            return bool(qcfg and qcfg.get("quant_method"))
        except Exception:
            return False

    def _load_model(self) -> None:
        """Load the unified model.

        Supports both full-precision (bf16) and quantized (AWQ) model weights.
        Quantization metadata (including ``modules_to_not_convert``) is read
        from the model's own ``config.json``.  When quantization is detected
        the loader skips ``.bfloat16()`` and ``torch.compile``.
        """
        logger.info(f"Loading unified model: {self.model_path}")
        if self.pt_path:
            logger.info(f"Extra weights: {self.pt_path}")
        start = time.time()

        from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO, ProcessorMode as ModelProcessorMode

        # Resolve attention implementation (auto-detect when set to "auto")
        resolved_attn = self._resolve_attn_implementation()

        is_quantized = self._is_quantized_model(self.model_path)
        if is_quantized:
            logger.info("Quantized model detected")

        # Load base model
        self.model = MiniCPMO.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            _attn_implementation=resolved_attn,
        )

        if is_quantized:
            # AWQ/GPTQ: integer qweight/qzeros must NOT be cast to bfloat16.
            # Non-quantized sub-modules (vpm, apm, tts, resampler) are already
            # stored in the correct dtype by the checkpoint.
            self.model.eval()
            logger.info(
                "Quantized model detected — skipping .bfloat16() cast "
                "(quantized layers use integer weights)"
            )
        else:
            self.model.bfloat16().eval()

        if self.device == "cuda":
            self.model.cuda()

        load_time = time.time() - start
        logger.info(
            f"Base model loaded in {load_time:.1f}s, "
            f"attn_implementation={resolved_attn}, quantized={is_quantized}"
        )

        # Unified initialization (supports all three modes)
        logger.info("Initializing unified mode...")
        init_start = time.time()

        self.model.init_unified(
            pt_path=self.pt_path,
            preload_both_tts=self.preload_both_tts,
            duplex_config={
                "generate_audio": self.duplex_config.generate_audio,
                "ls_mode": self.duplex_config.ls_mode,
                "max_new_speak_tokens_per_chunk": self.duplex_config.max_new_speak_tokens_per_chunk,
                "temperature": self.duplex_config.temperature,
                "top_k": self.duplex_config.top_k,
                "top_p": self.duplex_config.top_p,
                "force_listen_count": self.duplex_config.force_listen_count,
            },
            device=self.device,
            chat_vocoder=self.chat_vocoder,
        )

        init_time = time.time() - init_start
        logger.info(f"Unified mode initialization done in {init_time:.1f}s")

        # torch.compile acceleration + warmup (optional)
        if self.compile:
            compile_start = time.time()
            # AWQ: skip llm.model (custom INT4 kernels incompatible with compile),
            # but still compile vpm / resampler / tts.model (all float, full benefit).
            skip = ["llm.model"] if is_quantized else None
            self.model.apply_torch_compile(mode="default", dynamic=True, skip_modules=skip)
            self.model.warmup_compile(ref_audio_path=self.ref_audio_path)
            compile_time = time.time() - compile_start
            logger.info(f"torch.compile + warmup done in {compile_time:.1f}s")

        # Create View instances
        self._chat_view = ChatView(self.model, self.ref_audio_path)
        self._half_duplex_view = HalfDuplexView(self.model, self.ref_audio_path)
        self._duplex_view = DuplexView(self.model, self.ref_audio_path, self.duplex_config)
        self._fc_duplex_view = FcDuplexView(self.model)

        total_time = time.time() - start
        logger.info(f"UnifiedProcessor initialization complete in {total_time:.1f}s")

    def _release_resources(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ==================== Mode Switching ====================

    def _sync_compile_state(self, want_compiled: bool) -> None:
        """Enable/disable torch.compile based on target mode."""
        if self.compile and self.model is not None:
            self.model.set_compile_enabled(want_compiled)

    def set_chat_mode(self) -> ChatView:
        """Switch to Chat mode.

        Returns:
            ChatView instance.
        """
        from MiniCPMO45.modeling_minicpmo_unified import ProcessorMode as ModelProcessorMode

        if self._current_mode != ProcessorMode.CHAT:
            start = time.time()
            self._sync_compile_state(False)
            self.model.set_mode(ModelProcessorMode.CHAT)
            self._current_mode = ProcessorMode.CHAT
            logger.info(f"Switched to CHAT mode in {(time.time()-start)*1000:.1f}ms")

        return self._chat_view

    def set_half_duplex_mode(self) -> HalfDuplexView:
        """Switch to Half-Duplex mode.

        Returns:
            HalfDuplexView instance.
        """
        from MiniCPMO45.modeling_minicpmo_unified import ProcessorMode as ModelProcessorMode

        if self._current_mode != ProcessorMode.HALF_DUPLEX:
            start = time.time()
            self._sync_compile_state(False)
            self.model.set_mode(ModelProcessorMode.STREAMING)
            self._current_mode = ProcessorMode.HALF_DUPLEX
            logger.info(f"Switched to HALF_DUPLEX mode in {(time.time()-start)*1000:.1f}ms")

        return self._half_duplex_view

    def set_duplex_mode(self) -> DuplexView:
        """Switch to Duplex mode.

        Returns:
            DuplexView instance.
        """
        from MiniCPMO45.modeling_minicpmo_unified import ProcessorMode as ModelProcessorMode

        if self._current_mode != ProcessorMode.DUPLEX:
            start = time.time()
            self._sync_compile_state(True)
            self.model.set_mode(ModelProcessorMode.DUPLEX)
            self._current_mode = ProcessorMode.DUPLEX
            logger.info(f"Switched to DUPLEX mode in {(time.time()-start)*1000:.1f}ms")

        return self._duplex_view

    def set_fc_duplex_mode(self) -> FcDuplexView:
        """Return FC slot Duplex view.

        FC duplex uses its own runtime on the same model instance and does not
        reuse the old ``ProcessorMode.DUPLEX`` switching path.
        """
        self._sync_compile_state(True)
        return self._fc_duplex_view

    # ==================== KV Cache State ====================

    @property
    def kv_cache_length(self) -> int:
        """Total token count in the LLM KV cache.

        Returns the number of tokens processed in the backbone LLM's KV cache,
        including system prompt + all history turns + currently generated tokens.

        Notes:
        - Half-Duplex mode: reads model.llm_past_key_values
        - Duplex mode: reads model.duplex.decoder.cache (separate KV cache)
        - Chat mode: only valid during a chat() call
        - Returns 0 when KV cache is empty or model is not loaded
        """
        if self.model is None:
            return 0
        # Duplex mode uses the DuplexCapability's internal decoder cache
        if (self._current_mode == ProcessorMode.DUPLEX
                and hasattr(self.model, 'duplex')
                and self.model.duplex is not None
                and hasattr(self.model.duplex, 'decoder')):
            length = self.model.duplex.decoder.get_cache_length()
            if length == 0:
                decoder = self.model.duplex.decoder
                cache_type = type(decoder.cache).__name__ if decoder.cache is not None else "None"
                logger.warning(
                    f"[kv_cache_length] Duplex decoder.get_cache_length() returned 0: "
                    f"cache_type={cache_type}, cache is None={decoder.cache is None}"
                )
            return length
        if self._current_mode == ProcessorMode.DUPLEX:
            logger.warning(
                f"[kv_cache_length] Mode is DUPLEX but conditions not met: "
                f"has_duplex={hasattr(self.model, 'duplex')}, "
                f"duplex_is_none={getattr(self.model, 'duplex', None) is None}, "
                f"has_decoder={hasattr(getattr(self.model, 'duplex', None) or object(), 'decoder')}"
            )
        return self.model._get_kv_cache_length()

    # ==================== Convenience Properties ====================

    @property
    def chat(self) -> ChatView:
        """Chat view (does not switch mode, only returns the view)."""
        return self._chat_view

    @property
    def half_duplex(self) -> HalfDuplexView:
        """Half-Duplex view (does not switch mode, only returns the view)."""
        return self._half_duplex_view

    @property
    def duplex(self) -> DuplexView:
        """Duplex view (does not switch mode, only returns the view)."""
        return self._duplex_view

    @property
    def fc_duplex(self) -> FcDuplexView:
        """FC slot Duplex view (does not switch old Duplex mode)."""
        return self._fc_duplex_view
