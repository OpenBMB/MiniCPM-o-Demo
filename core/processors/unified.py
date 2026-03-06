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

from typing import Optional, Generator, List, TYPE_CHECKING
import os
import time
import logging
import base64

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
    DuplexConfig, DuplexGenerateResult,
    # Common
    Message, Role,
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

    def set_chat_mode(self) -> ChatView:
        """Switch to Chat mode.

        Returns:
            ChatView instance.
        """
        from MiniCPMO45.modeling_minicpmo_unified import ProcessorMode as ModelProcessorMode

        if self._current_mode != ProcessorMode.CHAT:
            start = time.time()
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
            self.model.set_mode(ModelProcessorMode.DUPLEX)
            self._current_mode = ProcessorMode.DUPLEX
            logger.info(f"Switched to DUPLEX mode in {(time.time()-start)*1000:.1f}ms")

        return self._duplex_view

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
