"""双工对话（Duplex）Schema 定义

本模块定义双工对话模式的请求和响应格式。

双工对话特点：
============

1. **全双工**：用户和模型可以同时说话
2. **实时交互**：低延迟，支持打断
3. **独占资源**：需要独占一个 Worker（GPU）
4. **状态复杂**：listen/speak 状态切换

**[CRITICAL] 与单工/流式的本质区别**：

- 单工/流式：使用 MiniCPMO 类
- 双工：使用 **MiniCPMODuplex** 类（完全不同的实现）
- 不能在同一个模型实例上切换单工和双工

工作流程：
=========

```
┌─────────────────────────────────────────────────────────────────────┐
│                       双工对话流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. prepare()      准备会话（设置 system prompt + 参考音频）         │
│        ↓                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  2. 循环处理（每秒执行）                                      │   │
│  │        ↓                                                     │   │
│  │     prefill()    预填充用户音频（1秒一个 chunk）              │   │
│  │        ↓                                                     │   │
│  │     generate()   生成响应（返回 listen/speak 状态）           │   │
│  │        ↓                                                     │   │
│  │     [处理结果]   播放音频 / 显示文本 / 检查 end_of_turn      │   │
│  │                                                              │   │
│  │     [可选] set_break()   用户打断（模型停止说话）             │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│        ↓                                                            │
│  3. stop()         停止会话                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Listen/Speak 状态：
==================

双工模式下，模型有两种状态：

- **Listen（听）**：模型在听用户说话，不产生输出
- **Speak（说）**：模型在说话，产生文本和音频

状态由模型自主决定（基于语义判断），但可以通过 force_listen_count 强制前 N 次为 listen。

**force_listen_count 的意义**：

这是一个"启动保护期"，确保模型在开始时先听用户说完一段话，而不是立即开始回复。

- 默认值：3（即前 3 秒强制 listen）
- 设置为 0：模型立即自主判断
- 真实场景中，我们不知道用户会说多长时间，所以这是一个固定的配置值

**[CRITICAL] System Prompt 格式**：

双工模式的 system prompt 必须使用特殊 token 格式，否则输出会乱码：

```
prefix: <|im_start|>system\\n{你的提示文本}\\n<|audio_start|>
suffix: <|audio_end|><|im_end|>
```

DuplexProcessor 会自动处理这个格式，调用者只需提供纯文本即可。

使用示例：
=========

```python
from core.schemas.duplex import (
    DuplexConfig,
    DuplexOfflineInput,
    DuplexOfflineOutput,
    DuplexGenerateResult,
)

# 1. 创建离线推理输入（用于测试和离线服务）
task_input = DuplexOfflineInput(
    system_prompt="你是一个友好的助手。",
    user_audio_path="/path/to/user_audio.wav",
    ref_audio_path="/path/to/reference.wav",
    config=DuplexConfig(force_listen_count=3)
)

# 2. 手动控制（更细粒度）
processor.prepare(
    system_prompt_text="你是一个友好的助手。",
    ref_audio_path="/path/to/reference.wav"
)

for audio_chunk in audio_stream:  # 每秒一个
    processor.prefill(audio_waveform=audio_chunk)
    result = processor.generate()
    
    if result.is_listen:
        print("[Listening...]")
    else:
        print(f"[Speaking] {result.text}")
        play_audio(result.audio_data)
    
    if result.end_of_turn:
        break

processor.stop()
```

打断机制：
=========

用户可以随时打断模型的发言：

```python
# 检测到用户按下打断按钮或开始说话
processor.set_break()

# 后续的 generate() 会立即返回 listen 状态
result = processor.generate()
assert result.is_listen == True

# 继续预填充用户的新输入
processor.prefill(new_audio_chunk)
```
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from core.schemas.common import TTSSamplingParams


# =============================================================================
# 双工配置
# =============================================================================

class DuplexConfig(BaseModel):
    """双工对话配置
    
    控制双工模式的行为和参数。
    
    **核心参数**：
    
    - generate_audio: 是否生成音频（通常为 True）
    - force_listen_count: 强制 listen 的次数（启动保护期）
    
    **LLM 生成参数**：
    
    - max_new_speak_tokens_per_chunk: 每个 chunk 最大 speak tokens
    - temperature, top_k, top_p: 采样参数
    - text_repetition_penalty: 文本重复惩罚
    
    **高级参数**：
    
    - ls_mode: Listen/Speak 模式（"explicit" 或 "implicit"）
    - decode_mode: 解码模式（"sampling" 或 "greedy"）
    - listen_prob_scale: listen 概率缩放
    
    **force_listen_count 详解**：
    
    这是一个关键参数，控制模型在对话开始时的行为：
    
    - 作用：前 N 次 streaming_generate() 强制返回 listen 状态
    - 目的：确保模型先听完用户说的一段话，再开始回复
    - 默认值：3（即前 3 秒）
    
    为什么不根据音频长度动态设置？
    - 真实场景中，我们不知道用户会说多长时间
    - 这是一个固定的"启动保护期"配置
    - 保护期后，模型自主判断何时开始说话
    
    Attributes:
        generate_audio: 是否生成音频
        ls_mode: Listen/Speak 模式
        force_listen_count: 强制 listen 次数
        max_new_speak_tokens_per_chunk: 每 chunk 最大 speak tokens
        decode_mode: 解码模式
        temperature: 采样温度
        top_k: Top-K 采样
        top_p: Top-P 采样
        text_repetition_penalty: 文本重复惩罚
        text_repetition_window_size: 重复检测窗口大小
        listen_prob_scale: listen 概率缩放
        tts_temperature: TTS 温度
        chunk_ms: 每个 chunk 的毫秒数
        sample_rate: 音频采样率
    """
    # 核心参数
    generate_audio: bool = Field(
        True, 
        description="是否生成音频"
    )
    ls_mode: str = Field(
        "explicit", 
        description="Listen/Speak 模式（explicit/implicit）"
    )
    
    # [CRITICAL] 启动保护期
    force_listen_count: int = Field(
        3, 
        ge=0, 
        description="强制 listen 的次数（固定配置，默认 3）"
    )
    
    # LLM 生成参数
    max_new_speak_tokens_per_chunk: int = Field(
        20, 
        ge=1,
        description="每个 chunk 最大 speak tokens"
    )
    decode_mode: str = Field(
        "sampling",
        description="解码模式（'sampling' 或 'greedy'）"
    )
    temperature: float = Field(
        0.7, 
        ge=0.0, 
        le=2.0, 
        description="采样温度"
    )
    top_k: int = Field(
        20, 
        ge=0, 
        description="Top-K 采样"
    )
    top_p: float = Field(
        0.8, 
        ge=0.0, 
        le=1.0, 
        description="Top-P 采样"
    )
    text_repetition_penalty: float = Field(
        1.05, 
        ge=1.0,
        description="文本重复惩罚"
    )
    text_repetition_window_size: int = Field(
        512,
        ge=1,
        description="重复检测窗口大小"
    )
    length_penalty: float = Field(
        1.1,
        ge=0.1,
        le=5.0,
        description="长度惩罚系数。>1.0 抑制 turn_eos token 使模型当前 turn 输出更长，=1.0 不惩罚，<1.0 鼓励更早结束"
    )
    
    # Listen 相关
    listen_prob_scale: float = Field(
        1.0,
        ge=0.0,
        description="listen 概率缩放（越大越倾向于 listen）"
    )
    listen_top_k: Optional[int] = Field(
        None,
        description="listen 时的 Top-K（None 使用默认值）"
    )
    
    # TTS 参数
    tts_temperature: float = Field(
        0.8, 
        ge=0.0,
        le=2.0,
        description="TTS 温度"
    )
    
    # 流参数
    chunk_ms: int = Field(
        1000, 
        ge=100,
        description="每个 chunk 的毫秒数（默认 1 秒）"
    )
    sample_rate: int = Field(
        16000, 
        description="输入音频采样率（必须 16000）"
    )


# =============================================================================
# 准备请求（初始化会话）
# =============================================================================

class DuplexPrepareRequest(BaseModel):
    """双工会话准备请求
    
    用于初始化双工会话，设置 system prompt 和参考音频。
    
    **System Prompt 格式说明**：
    
    双工模式需要特殊的 token 格式包装 system prompt：
    - prefix_system_prompt: `<|im_start|>system\\n{text}\\n<|audio_start|>`
    - suffix_system_prompt: `<|audio_end|><|im_end|>`
    
    **推荐**：使用 `DuplexProcessor.prepare(system_prompt_text=...)` 自动处理格式。
    
    Attributes:
        prefix_system_prompt: 系统提示前缀（包含特殊 token）
        suffix_system_prompt: 系统提示后缀（包含特殊 token）
        ref_audio_path: 参考音频路径（用于 TTS 音色）
        prompt_wav_path: TTS prompt 音频路径
    
    示例：
        ```python
        # 手动构建（需要处理 token 格式）
        request = DuplexPrepareRequest(
            prefix_system_prompt="<|im_start|>system\\n你是助手\\n<|audio_start|>",
            suffix_system_prompt="<|audio_end|><|im_end|>",
            ref_audio_path="/path/to/ref.wav"
        )
        
        # 推荐：使用 DuplexProcessor.prepare()
        processor.prepare(system_prompt_text="你是助手")  # 自动处理格式
        ```
    """
    prefix_system_prompt: Optional[str] = Field(
        None, 
        description="系统提示前缀（包含特殊 token）"
    )
    suffix_system_prompt: Optional[str] = Field(
        None, 
        description="系统提示后缀（包含特殊 token）"
    )
    ref_audio_path: Optional[str] = Field(
        None, 
        description="参考音频路径（用于 TTS 音色）"
    )
    prompt_wav_path: Optional[str] = Field(
        None, 
        description="TTS prompt 音频路径"
    )


# =============================================================================
# 预填充请求
# =============================================================================

class DuplexPrefillRequest(BaseModel):
    """双工预填充请求（每秒调用一次）
    
    用于向模型预填充用户的音频（和可选的图像帧）。
    
    **调用频率**：每秒调用一次，传入 1 秒的音频数据
    
    **音频格式**：
    
    - 采样率：16kHz
    - 声道：单声道（mono）
    - 格式：float32 numpy array 或文件路径
    
    **图像帧（可选）**：
    
    用于视频双工场景，每次 prefill 可以传入当前帧：
    - frame_list: 图像文件路径列表
    - max_slice_nums: HD 图像切片数
    
    Attributes:
        audio_waveform: 音频波形 Base64 数据
        audio_path: 音频文件路径
        frame_list: 图像帧路径列表
        max_slice_nums: HD 图像切片数
    
    示例：
        ```python
        # 从 numpy 数组
        import numpy as np
        import base64
        audio = np.zeros(16000, dtype=np.float32)  # 1秒
        data = base64.b64encode(audio.tobytes()).decode()
        request = DuplexPrefillRequest(audio_waveform=data)
        
        # 从文件
        request = DuplexPrefillRequest(audio_path="/path/to/chunk.wav")
        
        # 带图像帧
        request = DuplexPrefillRequest(
            audio_path="/path/to/chunk.wav",
            frame_list=["/path/to/frame.png"]
        )
        ```
    """
    audio_waveform: Optional[str] = Field(
        None, 
        description="音频波形 Base64 数据（16kHz mono float32）"
    )
    audio_path: Optional[str] = Field(
        None, 
        description="音频文件路径"
    )
    frame_list: Optional[List[str]] = Field(
        None, 
        description="图像帧路径列表"
    )
    max_slice_nums: int = Field(
        1, 
        ge=1,
        description="HD 图像切片数"
    )


# =============================================================================
# 生成结果
# =============================================================================

class DuplexGenerateResult(BaseModel):
    """双工生成结果（每次 generate 返回）
    
    每次调用 processor.generate() 返回的结果。
    
    **核心字段**：
    
    - is_listen: 模型当前是否在听（True=听，False=说）
    - text: 生成的文本（is_listen=True 时为空）
    - audio_data: 生成的音频（is_listen=True 时可能是静音或 None）
    - end_of_turn: 当前轮是否结束
    
    **状态说明**：
    
    1. is_listen=True, end_of_turn=False
       - 模型在听，等待用户说更多
       - 继续 prefill + generate
       
    2. is_listen=False, end_of_turn=False
       - 模型在说，但还没说完
       - 继续 prefill + generate，播放音频
       
    3. is_listen=False, end_of_turn=True
       - 模型说完了这一轮
       - 可以等待用户新输入，或结束会话
       
    4. is_listen=True, end_of_turn=True
       - 被打断或停止
       - 检查是否调用了 set_break() 或 stop()
    
    **性能指标**：
    
    - cost_llm_ms: LLM 推理耗时
    - cost_tts_prep_ms: TTS 准备耗时
    - cost_tts_ms: TTS CODEC 生成耗时
    - cost_token2wav_ms: Token2Wav 转换耗时
    - cost_all_ms: 总耗时（streaming_generate 内部）
    
    Attributes:
        is_listen: 是否处于 listen 状态
        text: 生成的文本
        audio_data: 音频波形 Base64 数据（24kHz）
        end_of_turn: 是否结束当前轮
        current_time: 当前时间索引（chunk 数）
        cost_llm_ms: LLM 耗时
        cost_tts_prep_ms: TTS 准备耗时
        cost_tts_ms: TTS CODEC 耗时
        cost_token2wav_ms: Token2Wav 耗时
        cost_all_ms: 总耗时
        n_tokens: LLM 生成的 token 数
        n_tts_tokens: TTS 生成的 token 数
    """
    is_listen: bool = Field(
        ..., 
        description="是否处于 listen 状态"
    )
    text: str = Field(
        "", 
        description="生成的文本（listen 时为空）"
    )
    audio_data: Optional[str] = Field(
        None, 
        description="音频波形 Base64 数据（24kHz float32）"
    )
    end_of_turn: bool = Field(
        False, 
        description="是否结束当前轮"
    )
    current_time: int = Field(
        0, 
        description="当前时间索引（chunk 数）"
    )
    
    # 性能指标
    cost_llm_ms: Optional[float] = Field(
        None, 
        description="LLM 耗时（毫秒）"
    )
    cost_tts_prep_ms: Optional[float] = Field(
        None,
        description="TTS 准备耗时（毫秒）"
    )
    cost_tts_ms: Optional[float] = Field(
        None, 
        description="TTS CODEC 耗时（毫秒）"
    )
    cost_token2wav_ms: Optional[float] = Field(
        None,
        description="Token2Wav 转换耗时（毫秒）"
    )
    cost_all_ms: Optional[float] = Field(
        None, 
        description="总耗时（毫秒）— streaming_generate 内部"
    )
    n_tokens: Optional[int] = Field(
        None,
        description="LLM 生成的 token 数"
    )
    n_tts_tokens: Optional[int] = Field(
        None,
        description="TTS 生成的 token 数"
    )
    usage: Optional[Dict[str, int]] = Field(
        None,
        description="本次 generate 产生的 output token delta"
    )
    server_send_ts: Optional[float] = Field(
        None,
        description="服务端发送此 result 的时间戳（time.time()），用于全链路时延分析"
    )


# =============================================================================
# 离线推理输入/输出（用于测试和离线服务）
# =============================================================================

class DuplexOfflineInput(BaseModel):
    """双工离线推理输入
    
    用于离线推理场景，包含完整的音频文件和配置信息。
    
    适用场景：
    - 单元测试
    - 离线批量处理
    - 演示场景
    
    注意：这不是实时双工会话，而是对完整音频文件的离线处理。
    实时双工请直接使用 DuplexProcessor 的 prepare/prefill/generate 原语。
    
    Attributes:
        system_prompt: 系统提示文本（会自动添加特殊 token）
        user_audio_path: 用户音频文件路径
        ref_audio_path: 参考音频路径（TTS 音色）
        image_paths: 图像路径列表（视频双工，每个 chunk 一张）
        config: 双工配置
    
    示例：
        ```python
        task_input = DuplexOfflineInput(
            system_prompt="你是一个友好的助手，请简短回复。",
            user_audio_path="/path/to/user_audio.wav",
            ref_audio_path="/path/to/reference.wav",
            config=DuplexConfig(force_listen_count=3)
        )
        output = processor.offline_inference(task_input)
        ```
    """
    # 系统配置
    system_prompt: str = Field(
        "You are a helpful assistant.",
        description="系统提示文本（会自动添加特殊 token 格式）"
    )
    
    # 音频输入
    user_audio_path: Optional[str] = Field(
        None, 
        description="用户音频文件路径"
    )
    ref_audio_path: Optional[str] = Field(
        None, 
        description="参考音频路径（用于 TTS 音色）"
    )
    
    # 图像输入（可选，视频双工）
    image_paths: Optional[List[str]] = Field(
        None, 
        description="图像路径列表（每个 chunk 对应一张）"
    )
    
    # 双工配置
    config: DuplexConfig = Field(
        default_factory=DuplexConfig, 
        description="双工配置"
    )


class DuplexChunkResult(BaseModel):
    """双工会话中单个 chunk 的结果
    
    用于记录和展示每个 chunk 的详细信息。
    
    Attributes:
        chunk_idx: chunk 索引（从 0 开始）
        phase: 阶段（"user" 用户输入 / "response" 模型响应）
        is_listen: 是否处于 listen 状态
        text: 生成的文本
        has_audio: 是否有音频输出
        audio_data: 音频数据（Base64 编码，可选）
        end_of_turn: 是否结束当前轮
        elapsed_ms: 本 chunk 耗时
    """
    chunk_idx: int = Field(
        ..., 
        description="chunk 索引（从 0 开始）"
    )
    phase: str = Field(
        ..., 
        description="阶段：user（用户输入）或 response（模型响应）"
    )
    is_listen: bool = Field(
        ..., 
        description="是否处于 listen 状态"
    )
    text: str = Field(
        "", 
        description="生成的文本"
    )
    has_audio: bool = Field(
        False, 
        description="是否有音频输出"
    )
    audio_data: Optional[str] = Field(
        None,
        description="音频数据（Base64 编码的 float32 PCM，24kHz）"
    )
    end_of_turn: bool = Field(
        False, 
        description="是否结束当前轮"
    )
    elapsed_ms: float = Field(
        0.0, 
        description="本 chunk 耗时（毫秒）"
    )


class DuplexOfflineOutput(BaseModel):
    """双工离线推理输出
    
    DuplexOfflineInput 的输出，包含完整的推理结果。
    
    **汇总信息**：
    
    - full_text: 模型输出的完整文本
    - total_chunks: 总 chunk 数
    - audio_duration_s: 输出音频时长
    - total_duration_ms: 总耗时
    
    **分块详情**：
    
    - chunks: 每个 chunk 的详细结果列表
    
    Attributes:
        success: 是否成功
        error: 错误信息
        full_text: 完整输出文本
        total_chunks: 总 chunk 数
        audio_duration_s: 输出音频时长（秒）
        total_duration_ms: 总耗时（毫秒）
        chunks: 分块结果列表
    """
    success: bool = Field(
        ..., 
        description="是否成功"
    )
    error: Optional[str] = Field(
        None, 
        description="错误信息"
    )
    
    # 汇总结果
    full_text: str = Field(
        "", 
        description="完整的输出文本"
    )
    total_chunks: int = Field(
        0, 
        description="总 chunk 数"
    )
    audio_duration_s: float = Field(
        0.0, 
        description="输出音频时长（秒）"
    )
    total_duration_ms: float = Field(
        0.0, 
        description="总耗时（毫秒）"
    )
    
    # 分块结果
    chunks: List[DuplexChunkResult] = Field(
        default_factory=list, 
        description="每个 chunk 的结果"
    )

