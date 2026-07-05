"""处理器基类

本模块定义所有处理器的抽象基类，确保一致的接口和行为。

设计理念：
=========

1. **统一接口**：所有处理器继承自 BaseProcessor，提供一致的生命周期
2. **能力声明**：每个处理器声明自己的能力，便于 Gateway 路由
3. **资源管理**：统一的模型加载、释放逻辑
4. **配置管理**：支持从配置创建、运行时调整

继承结构：
=========

```
BaseProcessor (抽象基类)
├── ChatProcessor      - 单工对话
├── StreamingProcessor - 流式对话
└── DuplexProcessor    - 双工对话
```

生命周期：
=========

```
__init__()          创建处理器，加载模型
     ↓
[使用方法]          chat/streaming_generate/etc.
     ↓
__del__()           释放资源，清理显存
```

使用示例：
=========

```python
from core.processors import ChatProcessor
from core.capabilities import ProcessorMode

# 创建处理器
processor = ChatProcessor(model_path="/path/to/model")

# 查询能力
print(processor.mode)  # ProcessorMode.CHAT
print(processor.capabilities.supports_rollback)  # True

# 使用处理器
response = processor.chat(request)

# 释放资源（或让 Python GC 处理）
del processor
```
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from core.capabilities import ProcessorMode, ProcessorCapabilities, CAPABILITIES


logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """处理器抽象基类
    
    所有 MiniCPMO45 处理器的基类，定义统一的接口和行为。
    
    **子类必须实现**：
    
    1. `mode` 属性：返回 ProcessorMode
    2. `_load_model()` 方法：加载模型
    3. `_release_resources()` 方法：释放资源
    
    **子类可选实现**：
    
    - `_validate_config()`: 验证配置
    - `is_ready()`: 检查是否就绪
    
    **共享行为**：
    
    - 能力声明（通过 capabilities 属性）
    - 资源管理（__del__ 自动释放）
    - 日志记录
    
    Attributes:
        model_path: 模型路径
        device: 运行设备（cuda/cpu）
        model: 加载的模型实例（子类定义具体类型）
    
    示例：
        ```python
        class MyProcessor(BaseProcessor):
            @property
            def mode(self) -> ProcessorMode:
                return ProcessorMode.CHAT
            
            def _load_model(self) -> None:
                self.model = load_my_model(self.model_path)
            
            def _release_resources(self) -> None:
                del self.model
        ```
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        """初始化处理器
        
        Args:
            model_path: 模型路径
            device: 运行设备，默认 "cuda"
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        
        self._is_initialized = False
        
        # 加载模型
        self._load_model()
        self._is_initialized = True
        
        logger.info(
            f"{self.__class__.__name__} 初始化完成: "
            f"mode={self.mode.name}, device={self.device}"
        )
    
    @property
    @abstractmethod
    def mode(self) -> ProcessorMode:
        """处理器模式
        
        子类必须实现此属性，返回对应的 ProcessorMode。
        
        Returns:
            ProcessorMode 枚举值
        """
        pass
    
    @property
    def capabilities(self) -> ProcessorCapabilities:
        """处理器能力
        
        根据 mode 返回对应的能力声明。
        
        Returns:
            ProcessorCapabilities 对象
        """
        return CAPABILITIES[self.mode]
    
    @abstractmethod
    def _load_model(self) -> None:
        """加载模型
        
        子类必须实现此方法，完成模型加载。
        
        应该：
        - 加载模型到 self.model
        - 初始化必要的状态
        - 记录日志
        
        Raises:
            Exception: 模型加载失败
        """
        pass
    
    @abstractmethod
    def _release_resources(self) -> None:
        """释放资源
        
        子类必须实现此方法，完成资源释放。
        
        应该：
        - 删除模型对象
        - 清理 GPU 显存
        - 释放其他资源（文件句柄等）
        """
        pass
    
    def is_ready(self) -> bool:
        """检查处理器是否就绪
        
        Returns:
            True 如果处理器已初始化且模型已加载
        """
        return self._is_initialized and self.model is not None
    
    def __del__(self):
        """析构函数，释放资源"""
        if self._is_initialized:
            try:
                self._release_resources()
                logger.info(f"{self.__class__.__name__} 资源已释放")
            except Exception as e:
                logger.error(f"{self.__class__.__name__} 释放资源失败: {e}")
    
    def __repr__(self) -> str:
        """返回处理器的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"mode={self.mode.name}, "
            f"device={self.device}, "
            f"ready={self.is_ready()}"
            f")"
        )


class MiniCPMOProcessorMixin:
    """MiniCPMO 处理器混入类
    
    提供 MiniCPMO 类（非 Duplex）共享的功能：
    
    - 参考音频加载和缓存
    - 内容格式转换（Schema → 模型格式）
    - TTS 模式切换
    
    ChatProcessor 和 StreamingProcessor 都使用此混入。
    
    **为什么使用 Mixin？**
    
    Chat 和 Streaming 都使用 MiniCPMO 类，有很多共享逻辑：
    - 加载同一个模型
    - 相同的内容转换逻辑
    - 相同的参考音频处理
    
    但它们有不同的调用方法（chat vs streaming_generate），
    所以用 Mixin 共享公共逻辑，而不是继承。
    """
    
    # 默认参考音频路径（子类可覆盖）
    DEFAULT_REF_AUDIO: Optional[str] = None
    
    def _load_ref_audio(
        self, 
        path: Optional[str] = None,
        cache: bool = True
    ):
        """加载参考音频
        
        Args:
            path: 音频路径，None 则使用默认路径
            cache: 是否缓存（默认缓存）
            
        Returns:
            16kHz mono 音频数组 (np.ndarray)
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 未提供路径且无默认路径
        """
        import librosa
        import numpy as np
        
        audio_path = path or self.DEFAULT_REF_AUDIO
        if audio_path is None:
            raise ValueError("未提供参考音频路径")
        
        # 检查缓存
        if cache and hasattr(self, '_ref_audio_cache') and self._ref_audio_cache is not None:
            cache_path, cache_audio = self._ref_audio_cache
            if cache_path == audio_path:
                return cache_audio
        
        # 加载音频
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # 缓存
        if cache:
            self._ref_audio_cache = (audio_path, audio)
        
        return audio
    
    def _convert_content_to_model_format(self, content):
        """将 Schema 内容转换为模型格式
        
        Args:
            content: Message.content（字符串或 ContentItem 列表）
            
        Returns:
            模型可接受的内容格式（列表）
        """
        from core.schemas import TextContent, ImageContent, AudioContent, VideoContent
        import numpy as np
        from PIL import Image
        import base64
        import io
        import tempfile
        import os
        
        if isinstance(content, str):
            return [content]
        
        result = []
        for item in content:
            if isinstance(item, TextContent):
                result.append(item.text)
            elif isinstance(item, ImageContent):
                img_bytes = base64.b64decode(item.data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                result.append(img)
            elif isinstance(item, AudioContent):
                audio_bytes = base64.b64decode(item.data)
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                result.append(audio)
            elif isinstance(item, VideoContent):
                video_bytes = base64.b64decode(item.data)
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(video_bytes)
                    tmp_path = tmp.name
                try:
                    from minicpmo.utils import get_video_frame_audio_segments
                    video_frames, audio_segments, stacked_frames = \
                        get_video_frame_audio_segments(tmp_path, stack_frames=item.stack_frames)
                    for i in range(len(video_frames)):
                        result.append(video_frames[i])
                        if audio_segments is not None:
                            result.append(audio_segments[i])
                        if stacked_frames is not None and stacked_frames[i] is not None:
                            result.append(stacked_frames[i])
                finally:
                    os.unlink(tmp_path)
            else:
                raise ValueError(f"未知的内容类型: {type(item)}")
        
        return result
    
    def _convert_messages_to_model_format(
        self,
        messages,
        tts_config=None,
    ):
        """将 Schema 消息列表转换为模型格式
        
        Args:
            messages: Message 列表
            tts_config: TTS 配置（如果启用，可能需要自动添加 system prompt）
            
        Returns:
            模型可接受的消息格式（dict 列表）
            
        Note:
            当用户的 messages 已包含 system 角色消息时，跳过自动构造 system prompt。
            用户自己管理 system prompt（含 ref audio），代码不应重复添加。
        """
        from core.schemas import TTSMode, Role
        
        result = []
        
        # 检查用户是否已提供 system 消息（若有，则跳过自动构造）
        has_user_system_msg = any(
            msg.role == Role.SYSTEM for msg in messages
        )
        
        # 仅当 TTS 启用 + 非 DEFAULT 模式 + 用户未自行提供 system 消息时，自动构造
        if (tts_config and tts_config.enabled 
                and tts_config.mode != TTSMode.DEFAULT
                and not has_user_system_msg):
            ref_audio = self._resolve_ref_audio(tts_config)
            
            # 获取 system prompt（包含参考音频）
            from MiniCPMO45.modeling_minicpmo import MiniCPMO
            sys_msg = MiniCPMO.get_sys_prompt(
                ref_audio=ref_audio,
                mode=tts_config.mode.value,
                language=tts_config.language,
                ref_audio_max_ms=tts_config.ref_audio_max_ms,
            )
            result.append(sys_msg)
        
        # 转换用户消息
        for msg in messages:
            content = self._convert_content_to_model_format(msg.content)
            # 如果只有一个文本元素，简化为字符串
            if len(content) == 1 and isinstance(content[0], str):
                content = content[0]
            result.append({
                "role": msg.role.value,
                "content": content
            })
        
        return result
    
    def _resolve_ref_audio(self, tts_config) -> "np.ndarray":
        """从 tts_config 解析参考音频（支持 path 和 base64 data 两种来源）
        
        优先级：ref_audio_path > ref_audio_data > DEFAULT_REF_AUDIO
        
        Args:
            tts_config: TTSConfig 实例
            
        Returns:
            16kHz mono 音频数组 (np.ndarray)
            
        Raises:
            ValueError: 所有来源均无可用的参考音频
        """
        import numpy as np
        
        # 1. 优先使用文件路径
        if tts_config.ref_audio_path:
            return self._load_ref_audio(tts_config.ref_audio_path)
        
        # 2. 使用 base64 数据
        if tts_config.ref_audio_data:
            import base64
            audio_bytes = base64.b64decode(tts_config.ref_audio_data)
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            return audio
        
        # 3. 使用默认路径
        return self._load_ref_audio(None)
    
    def _init_tts_mode(self, streaming: bool = False) -> None:
        """初始化/切换 TTS 模式
        
        Args:
            streaming: True 为流式模式，False 为非流式模式
        """
        if hasattr(self, 'model') and self.model is not None:
            self.model.init_tts(streaming=streaming)
            logger.info(f"TTS 模式切换: streaming={streaming}")
