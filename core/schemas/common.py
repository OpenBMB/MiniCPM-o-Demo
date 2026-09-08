"""MiniCPMO45 通用类型定义

本模块定义所有模式共享的基础类型，是整个 Schema 体系的基石。

核心概念：
=========

1. **消息角色（Role）**
   - SYSTEM: 系统指令，定义模型行为
   - USER: 用户输入（文本/图像/音频）
   - ASSISTANT: 模型输出

2. **内容类型（ContentItem）**
   模型支持多模态输入，每种模态有对应的 Schema：
   - TextContent: 文本内容
   - ImageContent: 图像内容（base64 编码）
   - AudioContent: 音频内容（base64 PCM float32，必须 16kHz 单声道）
   - VideoContent: 视频内容（base64 编码的视频文件，自动提取帧和音频）

3. **消息结构（Message）**
   ```python
   # 纯文本
   Message(role=Role.USER, content="你好")
   
   # 多模态（图像+文本）
   Message(role=Role.USER, content=[
       ImageContent(data="<base64 encoded image>"),
       TextContent(text="描述这张图片")
   ])
   ```

4. **TTS 相关配置**
   - TTSMode: TTS 模式枚举（CRITICAL: 必须用 AUDIO_ASSISTANT 才能使用参考音频）
   - TTSSamplingParams: TTS 采样参数（temperature, top_p 等）
   - TTSConfig: TTS 完整配置

5. **图像处理配置（ImageConfig）**
   - max_slice_nums: HD 图像切片数，影响高分辨率图像的理解质量

使用示例：
=========

```python
from core.schemas.common import (
    Message, Role,
    TextContent, ImageContent, AudioContent, VideoContent,
    TTSConfig, TTSMode, TTSSamplingParams,
)

# 构建多模态消息
msg = Message(
    role=Role.USER,
    content=[
        ImageContent(data="<base64 encoded image>"),
        TextContent(text="这是什么？")
    ]
)

# 配置 TTS
tts = TTSConfig(
    enabled=True,
    mode=TTSMode.AUDIO_ASSISTANT,  # 必须！否则参考音频无效
    ref_audio_path="/path/to/ref.wav",
    sampling=TTSSamplingParams(temperature=0.8)
)
```

注意事项：
=========

1. **音频格式**：AudioContent 必须是 16kHz 单声道，这是模型硬性要求
2. **TTS 模式**：mode="default" 会忽略 ref_audio！必须使用 AUDIO_ASSISTANT 或 OMNI
3. **图像切片**：高分辨率图像建议设置 max_slice_nums > 1
"""

from enum import Enum
from typing import List, Optional, Union, Literal, Dict, Any
import base64

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# 基础枚举类型
# =============================================================================

class Role(str, Enum):
    """消息角色
    
    MiniCPMO 对话系统中的三种角色：
    
    - SYSTEM: 系统指令
      - 用途：定义模型的行为、人设、输出格式
      - 位置：必须是对话的第一条消息（如果有的话）
      - TTS 模式下会自动注入包含参考音频的 system prompt
      
    - USER: 用户输入
      - 用途：用户的问题、指令、上传的媒体文件
      - 支持：纯文本、图像、音频、或它们的组合
      
    - ASSISTANT: 模型输出
      - 用途：模型的回复（用于多轮对话的历史记录）
      - 在多轮对话中，需要将之前的模型回复添加到消息列表
    
    示例：
        >>> from core.schemas.common import Role
        >>> Role.USER.value
        'user'
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    # Tool-calling roles (OpenAI-compatible function calling). "tool" carries a
    # tool's result back into the conversation; assistant messages may also
    # carry "tool_calls". Required for the chat template's <tools> block.
    TOOL = "tool"


class TTSMode(str, Enum):
    """TTS 模式
    
    **[CRITICAL] mode="default" 会忽略 ref_audio！**
    
    要使用参考音频克隆声音，必须使用 AUDIO_ASSISTANT 或 OMNI 模式。
    这是模型的内部行为，不是 bug。
    
    模式说明：
    
    - DEFAULT: 默认模式
      - 特点：不使用参考音频，使用模型默认声音
      - 用途：不需要特定音色时使用
      - ⚠️ 即使提供了 ref_audio 也会被忽略！
      
    - AUDIO_ASSISTANT: 音频助手模式（推荐）
      - 特点：使用参考音频的声音特征
      - 用途：克隆特定声音、保持对话音色一致
      - 参数：需要提供 ref_audio（16kHz 单声道 WAV）
      
    - OMNI: Omni 模式
      - 特点：支持视频输入（图像帧 + 音频）
      - 用途：视频理解、实时视频对话
      - 参数：需要提供 ref_audio
      
    - AUDIO_ROLEPLAY: 角色扮演模式
      - 特点：模型扮演特定角色
      - 用途：角色扮演对话
      
    - VOICE_CLONING: 声音克隆模式
      - 特点：更强的声音克隆能力
      - 用途：高质量声音复制
    
    使用建议：
        - 普通 TTS：使用 AUDIO_ASSISTANT
        - 视频对话：使用 OMNI
        - 不需要特定声音：使用 DEFAULT
    """
    DEFAULT = "default"
    AUDIO_ASSISTANT = "audio_assistant"
    OMNI = "omni"
    AUDIO_ROLEPLAY = "audio_roleplay"
    VOICE_CLONING = "voice_cloning"


class ContentType(str, Enum):
    """内容类型标识
    
    用于区分不同类型的内容，主要用于序列化和反序列化。
    
    - TEXT: 文本内容
    - IMAGE: 图像内容
    - AUDIO: 音频内容
    - VIDEO: 视频内容
    """
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


# =============================================================================
# 多模态内容类型
# =============================================================================

class TextContent(BaseModel):
    """文本内容
    
    表示对话中的纯文本部分。
    
    Attributes:
        type: 固定为 "text"，用于类型识别
        text: 文本内容字符串
    
    示例：
        >>> text = TextContent(text="描述这张图片")
        >>> text.model_dump()
        {'type': 'text', 'text': '描述这张图片'}
    """
    type: Literal["text"] = "text"
    text: str = Field(..., description="文本内容")


class ImageContent(BaseModel):
    """图像内容
    
    表示对话中的图像输入，通过 Base64 编码提供：
    
    ```python
    ImageContent(data="iVBORw0KGgo...")  # Base64 字符串
    ```
    
    Attributes:
        type: 固定为 "image"
        data: Base64 编码的图像数据
    
    支持的图像格式：
        - PNG, JPEG, GIF, WebP
        - 建议使用 RGB 模式（会自动转换）
    
    示例：
        >>> import base64
        >>> with open("image.png", "rb") as f:
        ...     data = base64.b64encode(f.read()).decode()
        >>> img = ImageContent(data=data)
    """
    type: Literal["image"] = "image"
    data: str = Field(..., description="Base64 编码的图像数据")


class AudioContent(BaseModel):
    """音频内容
    
    表示对话中的音频输入（用户语音）。
    
    **[CRITICAL] 音频格式要求**：
    - 采样率：必须是 16000 Hz（16kHz）
    - 声道：必须是单声道（mono）
    - 格式：PCM float32
    
    这是模型的硬性要求，不符合格式的音频会导致推理错误或效果下降。
    
    提供方式（Base64 编码的 PCM 数据）：
    
    ```python
    # audio_array 是 np.float32 数组
    data = base64.b64encode(audio_array.tobytes()).decode()
    AudioContent(data=data)
    ```
    
    Attributes:
        type: 固定为 "audio"
        data: Base64 编码的 PCM 数据（float32，16kHz，mono）
        sample_rate: 采样率（必须为 16000）
    
    示例：
        >>> import numpy as np
        >>> import base64
        >>> audio_array = np.zeros(16000, dtype=np.float32)  # 1秒静音
        >>> data = base64.b64encode(audio_array.tobytes()).decode()
        >>> audio = AudioContent(data=data)
    """
    type: Literal["audio"] = "audio"
    data: str = Field(
        ..., 
        description="Base64 编码的 PCM 数据（float32，16kHz，mono）"
    )
    sample_rate: int = Field(16000, description="采样率（必须为 16000）")
    
    @field_validator("sample_rate")
    @classmethod
    def check_sample_rate(cls, v: int) -> int:
        """验证采样率必须为 16000"""
        if v != 16000:
            raise ValueError(f"采样率必须为 16000，当前为 {v}")
        return v


class VideoContent(BaseModel):
    """视频内容
    
    表示对话中的视频输入，通过 Base64 编码提供整个视频文件。
    
    处理流程：
    - Processor 层会将 base64 视频解码为临时文件
    - 使用 ``get_video_frame_audio_segments`` 提取帧和音频片段
    - 按交错顺序（frame, audio, [stacked_frame], ...）送入模型
    
    Attributes:
        type: 固定为 "video"
        data: Base64 编码的视频文件数据（MP4 等常见格式）
        stack_frames: 高刷帧率模式的帧数，1 为标准（每秒 1 帧），
                      >1 时额外提取中间帧拼接为 stacked frame
    
    示例：
        >>> import base64
        >>> with open("video.mp4", "rb") as f:
        ...     data = base64.b64encode(f.read()).decode()
        >>> video = VideoContent(data=data, stack_frames=1)
    """
    type: Literal["video"] = "video"
    data: str = Field(..., description="Base64 编码的视频文件数据")
    stack_frames: int = Field(
        1,
        ge=1,
        description="高刷帧率模式帧数，1=标准（每秒 1 帧）"
    )


# 内容项的联合类型
ContentItem = Union[TextContent, ImageContent, AudioContent, VideoContent]
"""多模态内容项

可以是以下任意一种：
- TextContent: 文本
- ImageContent: 图像
- AudioContent: 音频
- VideoContent: 视频

用于 Message.content 字段的类型注解。
"""


# =============================================================================
# 消息类型
# =============================================================================

class Message(BaseModel):
    """对话消息
    
    MiniCPMO 对话系统的基本单元。每条消息包含角色和内容。
    
    **内容格式**：
    
    1. 纯文本（简写）：
       ```python
       Message(role=Role.USER, content="你好")
       ```
       
    2. 多模态内容：
       ```python
       Message(role=Role.USER, content=[
           ImageContent(data="<base64 encoded image>"),
           TextContent(text="这是什么？")
       ])
       ```
       
    3. 音频输入：
       ```python
       Message(role=Role.USER, content=[
           AudioContent(data="<base64 encoded PCM float32>"),
           TextContent(text="请复述用户说的话")
       ])
       ```
    
    Attributes:
        role: 消息角色（system/user/assistant）
        content: 消息内容，可以是字符串或 ContentItem 列表
    
    多轮对话示例：
        ```python
        messages = [
            Message(role=Role.USER, content="我叫小明"),
            Message(role=Role.ASSISTANT, content="你好小明！"),
            Message(role=Role.USER, content="我叫什么名字？"),
        ]
        ```
    
    注意事项：
        - 如果有 SYSTEM 消息，必须放在第一条
        - USER 和 ASSISTANT 消息应该交替出现
        - 最后一条消息通常是 USER（等待模型回复）
    """
    role: Role = Field(..., description="消息角色")
    content: Union[str, List[ContentItem]] = Field(
        ..., 
        description="消息内容（字符串或多模态列表）"
    )
    # OpenAI-compatible function calling. Populated on:
    #   - assistant messages that the model emitted as tool calls
    #   - (tool results live in the message whose role is "tool")
    # Each entry: {"id": str, "type": "function",
    #              "function": {"name": str, "arguments": str}}
    # When present, convert_to_model_msgs forwards it so the chat template's
    # <tools> block can render the assistant tool_call + <tool_response> turns.
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        """保持字符串格式不变"""
        return v


# =============================================================================
# TTS 采样参数
# =============================================================================

class TTSSamplingParams(BaseModel):
    """TTS 采样参数
    
    控制 TTS（文本转语音）生成的采样策略。这些参数直接影响：
    - 语音的自然度
    - 音色的稳定性
    - 韵律的变化
    
    **参数说明**：
    
    - top_p (0.85): Top-P 采样阈值
      - 较高值：更多样化的语音
      - 较低值：更稳定的语音
      - 推荐范围：0.7-0.95
      
    - min_p (0.01): 最小概率阈值
      - 过滤掉概率太低的 token
      - 通常保持默认值
      
    - top_k (25): Top-K 采样数量
      - 每步只考虑概率最高的 K 个 token
      - 较小值：更稳定但可能单调
      - 较大值：更丰富但可能不稳定
      
    - repetition_penalty (1.05): 重复惩罚
      - > 1.0：减少重复
      - = 1.0：不惩罚
      - 推荐：1.05-1.2
      
    - temperature (0.8): 采样温度
      - 较高值：更随机、更有表现力
      - 较低值：更确定、更平稳
      - 推荐范围：0.6-1.0
      
    - win_size (16): 滑动窗口大小
      - 用于重复检测的窗口
      - 通常保持默认值
      
    - tau_r (0.1): 温度调节参数
      - 用于动态调整采样策略
      - 通常保持默认值
    
    **预设配置**：
    
    - 稳定语音（适合朗读）:
      ```python
      TTSSamplingParams(temperature=0.6, top_p=0.8, top_k=15)
      ```
      
    - 自然语音（适合对话）:
      ```python
      TTSSamplingParams(temperature=0.8, top_p=0.85, top_k=25)
      ```
      
    - 富有表现力（适合演绎）:
      ```python
      TTSSamplingParams(temperature=1.0, top_p=0.9, top_k=50)
      ```
    
    注意：这些参数对应模型内部的 `utils.TTSSamplingParams`
    """
    top_p: float = Field(
        0.85, 
        ge=0.0, le=1.0, 
        description="Top-P 采样阈值，控制采样多样性"
    )
    min_p: float = Field(
        0.01, 
        ge=0.0, le=1.0, 
        description="最小概率阈值"
    )
    top_k: int = Field(
        25, 
        ge=0, 
        description="Top-K 采样数量"
    )
    repetition_penalty: float = Field(
        1.05, 
        ge=1.0, 
        description="重复惩罚系数"
    )
    temperature: float = Field(
        0.8, 
        ge=0.0, le=2.0, 
        description="采样温度"
    )
    win_size: int = Field(
        16, 
        ge=1, 
        description="重复检测滑动窗口大小"
    )
    tau_r: float = Field(
        0.1, 
        ge=0.0, 
        description="温度调节参数"
    )


# =============================================================================
# TTS 配置
# =============================================================================

class TTSConfig(BaseModel):
    """TTS（文本转语音）配置
    
    控制模型是否输出语音以及语音的特性。
    
    **[CRITICAL] 使用参考音频的条件**：
    
    1. enabled = True
    2. mode = AUDIO_ASSISTANT 或 OMNI（不能是 DEFAULT！）
    3. 提供 ref_audio_path 或 ref_audio_data
    
    如果 mode=DEFAULT，即使提供了参考音频也会被忽略！
    
    **基本用法**：
    
    ```python
    # 不输出语音
    tts = TTSConfig(enabled=False)
    
    # 输出语音（使用参考音频的声音）
    tts = TTSConfig(
        enabled=True,
        mode=TTSMode.AUDIO_ASSISTANT,
        ref_audio_path="/path/to/reference.wav",
        output_path="/path/to/output.wav"
    )
    ```
    
    **参考音频要求**：
    
    - 格式：16kHz 单声道 WAV
    - 长度：建议 3-10 秒
    - 内容：清晰的语音，无背景噪音
    - 可以通过 ref_audio_max_ms 限制使用的长度
    
    Attributes:
        enabled: 是否启用 TTS 输出
        mode: TTS 模式（CRITICAL: 影响参考音频是否生效）
        ref_audio_path: 参考音频文件路径
        ref_audio_data: 参考音频 Base64 数据
        ref_audio_max_ms: 参考音频最大使用长度（毫秒）
        output_path: 输出音频保存路径
        language: 语言（"en" 或 "zh"）
        sampling: TTS 采样参数
    
    验证规则：
        - 当 enabled=True 且 mode 不是 DEFAULT 时，必须提供参考音频
    """
    enabled: bool = Field(False, description="是否启用 TTS 输出")
    mode: TTSMode = Field(
        TTSMode.AUDIO_ASSISTANT, 
        description="TTS 模式（推荐 AUDIO_ASSISTANT）"
    )
    ref_audio_path: Optional[str] = Field(
        None,
        description="参考音频路径（16kHz mono WAV）"
    )
    ref_audio_data: Optional[str] = Field(
        None,
        description="参考音频 Base64 数据"
    )
    ref_audio_max_ms: Optional[int] = Field(
        None,
        ge=1000,
        description="参考音频最大使用长度（毫秒），建议 3000-10000"
    )
    output_path: Optional[str] = Field(
        None,
        description="输出音频保存路径"
    )
    language: str = Field(
        "en",
        description="语言（'en' 英语 / 'zh' 中文）"
    )
    sampling: TTSSamplingParams = Field(
        default_factory=TTSSamplingParams,
        description="TTS 采样参数"
    )
    
    @model_validator(mode="after")
    def check_ref_audio_when_enabled(self) -> "TTSConfig":
        """当启用 TTS 且非 DEFAULT 模式时，检查参考音频

        注意：允许 ref_audio 为空，Processor 层会在运行时
        使用 Worker 的默认 ref_audio 补上。
        """
        # 不再强制报错，改为由 Processor 层处理
        return self


# =============================================================================
# 图像处理配置
# =============================================================================

class ImageConfig(BaseModel):
    """图像处理配置
    
    控制模型如何处理输入图像，特别是高分辨率图像。
    
    **HD 图像切片（max_slice_nums）**：
    
    MiniCPMO 使用"切片"策略处理高分辨率图像：
    - 将大图切分为多个小块
    - 分别处理每个小块
    - 融合结果
    
    切片数量影响：
    - 更多切片：更精细的理解，但更慢、更占显存
    - 更少切片：更快，但可能丢失细节
    
    推荐设置：
    - 普通图像：max_slice_nums = 1 或 None（自动）
    - 高分辨率图像（需要细节）：max_slice_nums = 4-9
    - 极高分辨率：max_slice_nums = 9-16
    
    **图像 ID（use_image_id）**：
    
    在多图像场景下，为每张图像分配唯一 ID，便于模型区分。
    
    Attributes:
        max_slice_nums: HD 图像最大切片数
        use_image_id: 是否使用图像 ID（多图像场景）
    
    示例：
        ```python
        # 普通图像
        config = ImageConfig()  # 使用默认值
        
        # 高分辨率图像，需要精细理解
        config = ImageConfig(max_slice_nums=9)
        
        # 多图像场景
        config = ImageConfig(use_image_id=True)
        ```
    """
    max_slice_nums: Optional[int] = Field(
        None,
        ge=1,
        le=16,
        description="HD 图像最大切片数（None 为自动，1-16）"
    )
    use_image_id: bool = Field(
        False,
        description="是否使用图像 ID（多图像场景）"
    )


# =============================================================================
# 生成参数
# =============================================================================

class GenerationConfig(BaseModel):
    """LLM 生成参数配置
    
    控制文本生成的策略，适用于所有模式（单工/流式/双工）。
    
    **采样策略**：
    
    - do_sample=True（默认）：使用采样
      - 更自然、更多样
      - 适合对话、创作
      
    - do_sample=False：贪婪解码
      - 更确定、可复现
      - 适合问答、代码生成
    
    **参数说明**：
    
    - max_new_tokens (512): 最大生成 token 数
      - 控制回复长度上限
      - 太小可能截断回复
      - 太大可能浪费时间
      
    - min_new_tokens (0): 最小生成 token 数
      - 强制生成至少 N 个 token
      - 用于避免过短回复
      
    - temperature (0.7): 采样温度
      - 0.0-0.5：更确定
      - 0.5-1.0：正常
      - 1.0+：更随机
      
    - top_p (0.8): Top-P（核采样）
      - 只从累积概率达到 P 的 token 中采样
      - 推荐：0.7-0.95
      
    - top_k (100): Top-K 采样
      - 只从概率最高的 K 个 token 中采样
      - 0 表示禁用
      
    - max_inp_length (8192): 最大输入长度
      - 输入超过此长度会被截断
      - 模型上下文窗口限制
    
    Attributes:
        max_new_tokens: 最大生成 token 数
        min_new_tokens: 最小生成 token 数
        do_sample: 是否采样（False 为贪婪解码）
        temperature: 采样温度
        top_p: Top-P 采样
        top_k: Top-K 采样
        max_inp_length: 最大输入长度
    
    示例：
        ```python
        # 确定性回答（问答场景）
        config = GenerationConfig(do_sample=False)
        
        # 创意写作
        config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=1024
        )
        
        # 短回复
        config = GenerationConfig(max_new_tokens=50)
        ```
    """
    max_new_tokens: int = Field(
        512, 
        ge=1, 
        le=4096, 
        description="最大生成 token 数"
    )
    min_new_tokens: int = Field(
        0,
        ge=0,
        description="最小生成 token 数"
    )
    do_sample: bool = Field(
        True, 
        description="是否采样（False 为贪婪解码）"
    )
    temperature: float = Field(
        0.7, 
        ge=0.0, 
        le=2.0, 
        description="采样温度"
    )
    top_p: float = Field(
        0.8, 
        ge=0.0, 
        le=1.0, 
        description="Top-P 采样"
    )
    top_k: int = Field(
        100, 
        ge=0, 
        description="Top-K 采样（0 禁用）"
    )
    length_penalty: float = Field(
        1.1,
        ge=0.1,
        le=5.0,
        description="长度惩罚系数。>1.0 抑制 EOS token 使输出更长更详细，=1.0 不惩罚，<1.0 鼓励更早结束"
    )
    max_inp_length: int = Field(
        8192,
        ge=1,
        description="最大输入长度（token 数）"
    )
