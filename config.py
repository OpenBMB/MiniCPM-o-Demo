"""MiniCPMO45 服务配置

所有端口、路径、超时、前端默认值等配置集中管理。
Worker 和 Gateway 统一读取此文件。

配置来源优先级（高 → 低）：
    1. CLI 参数（worker.py / gateway.py 的 argparse）
    2. config.json（与本文件同级目录，gitignored）
    3. Pydantic 默认值（本文件中定义）

首次部署时，复制 config.example.json 为 config.json 并修改 model_path：
    cp config.example.json config.json
    # 编辑 config.json 中的 model.model_path

使用方式：
    from config import get_config
    config = get_config()
    print(config.model.model_path)
    print(config.audio.playback_delay_ms)
"""

import json
import logging
import os
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "config.example.json")


# ============ 配置子模型 ============


class ModelConfig(BaseModel):
    """模型加载配置"""

    model_path: str = Field(
        description="基础模型路径（HuggingFace 格式目录）。必填，无默认值。",
    )
    pt_path: Optional[str] = Field(
        default=None,
        description="额外权重路径（.pt 文件，可选）。为 null 时不加载额外权重。",
    )
    attn_implementation: str = Field(
        default="auto",
        description=(
            "Attention 实现方式。"
            "'auto'（默认）= 自动检测，优先 flash_attention_2，不可用时降级到 sdpa；"
            "'flash_attention_2' = 强制使用 Flash Attention 2（需安装 flash-attn 包）；"
            "'sdpa' = 强制使用 PyTorch SDPA（无额外依赖）；"
            "'eager' = 朴素实现（仅 debug 用）。"
        ),
        pattern="^(auto|flash_attention_2|sdpa|eager)$",
    )


class AudioConfig(BaseModel):
    """音频相关配置"""

    ref_audio_path: Optional[str] = Field(
        default="assets/ref_audio/ref_minicpm_signature.wav",
        description="默认参考音频路径（TTS 声音克隆，相对于 minicpmo45_service/）",
    )
    playback_delay_ms: int = Field(
        default=200,
        description="前端收到首个 SPEAK chunk 后延迟多少 ms 开始播放（吸收网络/推理抖动）",
        ge=0,
        le=2000,
    )
    chat_vocoder: str = Field(
        default="token2wav",
        description=(
            "Chat（非流式）模式使用的 vocoder。"
            "'token2wav' = Step Audio Token2Wav（轻量，默认）；"
            "'cosyvoice2' = CosyVoice2-0.5B（需额外依赖和模型文件）。"
            "Streaming/Duplex 始终使用 token2wav。"
            "当设为 'token2wav' 时不会加载 CosyVoice2，节省 ~0.5GB 显存和依赖。"
        ),
        pattern="^(token2wav|cosyvoice2)$",
    )


class ServiceSectionConfig(BaseModel):
    """服务部署配置"""

    gateway_port: int = Field(
        default=8006,
        description="Gateway 端口",
    )
    worker_base_port: int = Field(
        default=22400,
        description="Worker 起始端口（Worker 0 = 22400, Worker 1 = 22401, ...）",
    )
    max_queue_size: int = Field(
        default=1000,
        description="最大排队请求数",
    )
    eta_chat_s: float = Field(
        default=15.0,
        description="Chat 预估耗时基准（秒），Admin 可动态调整",
    )
    eta_half_duplex_s: float = Field(
        default=180.0,
        description="Half-Duplex 预估耗时基准（秒），Admin 可动态调整",
    )
    eta_audio_duplex_s: float = Field(
        default=120.0,
        description="Audio Duplex 预估耗时基准（秒），Admin 可动态调整",
    )
    eta_omni_duplex_s: float = Field(
        default=90.0,
        description="Omni Duplex 预估耗时基准（秒），Admin 可动态调整",
    )
    eta_ema_alpha: float = Field(
        default=0.3,
        description="ETA 动态 EMA 平滑系数（0-1，越大越敏感）",
    )
    eta_ema_min_samples: int = Field(
        default=3,
        description="EMA 生效最少样本数（不足时使用基准值）",
    )
    request_timeout: float = Field(
        default=300.0,
        description="请求超时时间（秒）",
    )
    compile: bool = Field(
        default=False,
        description="是否对核心子模块应用 torch.compile 加速（首次推理触发编译）",
    )
    data_dir: str = Field(
        default="data",
        description="数据目录（相对于项目根目录）",
    )


class RecordingConfig(BaseModel):
    """Session 录制配置"""

    enabled: bool = Field(
        default=True,
        description="是否开启自动录制",
    )
    session_retention_days: int = Field(
        default=-1,
        description="Session 保留天数（-1 = 不清理，>0 = 超过天数后删除）",
    )
    max_storage_gb: float = Field(
        default=-1,
        description="录制数据总容量上限 (GB)（-1 = 不限制，>0 = 超过后按时间 LRU 删除）",
    )


class DuplexSectionConfig(BaseModel):
    """双工对话配置"""

    pause_timeout: float = Field(
        default=60.0,
        description="Duplex 暂停超时（秒），超时后释放 Worker",
    )


# ============ 顶层配置 ============


class ServiceConfig(BaseModel):
    """MiniCPMO45 服务完整配置

    从 config.json 加载，所有字段（除 model.model_path）均有默认值。
    用户只需在 config.json 中写需要覆盖的字段。
    """

    model: ModelConfig = Field(
        description="模型加载配置",
    )
    audio: AudioConfig = Field(
        default_factory=AudioConfig,
        description="音频相关配置",
    )
    service: ServiceSectionConfig = Field(
        default_factory=ServiceSectionConfig,
        description="服务部署配置",
    )
    duplex: DuplexSectionConfig = Field(
        default_factory=DuplexSectionConfig,
        description="双工对话配置",
    )
    recording: RecordingConfig = Field(
        default_factory=RecordingConfig,
        description="Session 录制配置",
    )

    # ========== 便捷属性（兼容旧代码） ==========

    @property
    def gateway_port(self) -> int:
        return self.service.gateway_port

    @property
    def worker_base_port(self) -> int:
        return self.service.worker_base_port

    @property
    def max_queue_size(self) -> int:
        return self.service.max_queue_size

    @property
    def request_timeout(self) -> float:
        return self.service.request_timeout

    @property
    def eta_chat_s(self) -> float:
        return self.service.eta_chat_s

    @property
    def eta_half_duplex_s(self) -> float:
        return self.service.eta_half_duplex_s

    @property
    def eta_audio_duplex_s(self) -> float:
        return self.service.eta_audio_duplex_s

    @property
    def eta_omni_duplex_s(self) -> float:
        return self.service.eta_omni_duplex_s

    @property
    def eta_ema_alpha(self) -> float:
        return self.service.eta_ema_alpha

    @property
    def eta_ema_min_samples(self) -> int:
        return self.service.eta_ema_min_samples

    @property
    def compile(self) -> bool:
        return self.service.compile

    @property
    def data_dir(self) -> str:
        return self.service.data_dir

    @property
    def ref_audio_path(self) -> Optional[str]:
        return self.audio.ref_audio_path

    @property
    def chat_vocoder(self) -> str:
        return self.audio.chat_vocoder

    @property
    def attn_implementation(self) -> str:
        return self.model.attn_implementation

    @property
    def duplex_pause_timeout(self) -> float:
        return self.duplex.pause_timeout

    @property
    def playback_delay_ms(self) -> int:
        return self.audio.playback_delay_ms

    # ========== 派生方法 ==========

    def worker_port(self, worker_index: int) -> int:
        """获取指定 Worker 的端口"""
        return self.worker_base_port + worker_index

    def worker_addresses(self, num_workers: int) -> List[str]:
        """生成 Worker 地址列表"""
        return [f"localhost:{self.worker_port(i)}" for i in range(num_workers)]

    def frontend_defaults(self) -> dict:
        """返回前端页面需要的默认配置（供 /api/frontend_defaults 使用）"""
        return {
            "playback_delay_ms": self.playback_delay_ms,
        }


# ============ 加载逻辑 ============


def load_config(path: str = _CONFIG_PATH) -> ServiceConfig:
    """从 config.json 加载服务配置

    config.json 支持部分覆盖：只需写需要修改的字段，其余走 Pydantic 默认值。
    最小配置只需 {"model": {"model_path": "/path/to/model"}}

    Args:
        path: config.json 的路径

    Returns:
        ServiceConfig 实例

    Raises:
        FileNotFoundError: 配置文件不存在（提示用户从 example 复制）
        ValueError: 配置文件格式错误
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"配置文件不存在: {path}\n"
            f"请从示例文件创建：\n"
            f"  cp {_EXAMPLE_PATH} {path}\n"
            f"然后修改 model.model_path 为实际模型路径。\n"
            f"\n"
            f"最小配置：\n"
            f'{{"model": {{"model_path": "/path/to/your/model"}}}}'
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 检查必填字段：model.model_path
    model_section = data.get("model")
    if not model_section or not model_section.get("model_path"):
        raise ValueError(
            f"config.json 缺少必填字段 model.model_path\n"
            f"请编辑 {path}，设置模型路径：\n"
            f'\n'
            f'{{"model": {{"model_path": "/path/to/your/model"}}}}'
        )

    config = ServiceConfig(**data)
    logger.info(
        f"配置已加载: model={config.model.model_path}, "
        f"attn_implementation={config.attn_implementation}, "
        f"gateway_port={config.gateway_port}, "
        f"playback_delay_ms={config.playback_delay_ms}, "
        f"chat_vocoder={config.chat_vocoder}"
    )
    return config


# ============ 全局单例 ============

_config: Optional[ServiceConfig] = None


def get_config() -> ServiceConfig:
    """获取全局配置（单例）

    首次调用时从 config.json 加载并缓存。
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config
