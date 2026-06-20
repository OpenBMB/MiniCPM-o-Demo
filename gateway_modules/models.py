"""Gateway 数据模型

定义 Gateway 层的 Pydantic 请求/响应模型、Worker 状态、队列模型等。
"""

from typing import Optional, List
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field


# ============ Worker 状态（Gateway 视角） ============

class GatewayWorkerStatus(str, Enum):
    """Worker 状态（Gateway 追踪）"""
    IDLE = "idle"
    BUSY_CHAT = "busy_chat"
    BUSY_HALF_DUPLEX = "busy_half_duplex"
    DUPLEX_ACTIVE = "duplex_active"
    DUPLEX_PAUSED = "duplex_paused"
    LOADING = "loading"
    ERROR = "error"
    OFFLINE = "offline"


class WorkerInfo(BaseModel):
    """Worker 信息（Gateway 视角）"""
    worker_id: str
    host: str
    port: int
    gpu_id: int
    status: GatewayWorkerStatus
    current_ticket_id: Optional[str] = None
    total_requests: int = 0
    avg_inference_time_ms: float = 0.0
    last_heartbeat: Optional[datetime] = None
    current_request_type: Optional[str] = None
    task_started_at: Optional[datetime] = None
    capabilities: List[str] = Field(default_factory=list)


# ============ 队列模型 ============

class QueueTicket(BaseModel):
    """排队凭证

    每个等待中的请求持有一个 ticket，包含位置和预估等待时间。
    位置和 ETA 随队列变化动态更新。
    """
    ticket_id: str
    request_type: str  # "chat" | "half_duplex_audio" | "audio_duplex" | "omni_duplex"
    enqueued_at: datetime = Field(default_factory=datetime.now)
    position: int = 0  # 1-based，0 表示已分配
    estimated_wait_s: float = 0.0
    cancelled: bool = False


class QueueTicketSummary(BaseModel):
    """队列项摘要（供 API 和 Admin 展示）"""
    ticket_id: str
    request_type: str
    position: int
    estimated_wait_s: float
    enqueued_at: datetime
    wait_elapsed_s: float = 0.0  # 已等待时间


class RunningTaskInfo(BaseModel):
    """正在运行的任务信息（用于 ETA 估算和 Admin 展示）"""
    worker_id: str
    request_type: str
    started_at: datetime
    elapsed_s: float
    estimated_remaining_s: float


class QueueStatus(BaseModel):
    """队列快照（供前端/Admin 使用）"""
    queue_length: int
    max_queue_size: int
    items: List[QueueTicketSummary] = []
    running_tasks: List[RunningTaskInfo] = []


# ============ 服务状态 ============

class ServiceStatus(BaseModel):
    """服务全局状态"""
    gateway_healthy: bool = True
    total_workers: int = 0
    idle_workers: int = 0
    busy_workers: int = 0
    duplex_workers: int = 0
    loading_workers: int = 0
    error_workers: int = 0
    offline_workers: int = 0
    queue_length: int = 0
    max_queue_size: int = 1000
    running_tasks: List[RunningTaskInfo] = []


# ============ ETA 配置（运行时可调） ============

class EtaConfig(BaseModel):
    """ETA 基准配置（Admin 可调）"""
    eta_chat_s: float = Field(default=15.0, description="Chat 预估耗时（秒）")
    eta_half_duplex_s: float = Field(default=180.0, description="Half-Duplex 预估耗时（秒）")
    eta_audio_duplex_s: float = Field(default=120.0, description="Audio Duplex 预估耗时（秒）")
    eta_omni_duplex_s: float = Field(default=90.0, description="Omni Duplex 预估耗时（秒）")
    ema_alpha: Optional[float] = Field(default=None, description="EMA 平滑系数（0-1，越大越敏感），None 表示不修改")


class EtaStatus(BaseModel):
    """ETA 状态（含 EMA 动态值）"""
    config: EtaConfig
    ema_alpha: float = 0.3
    ema_chat_s: Optional[float] = None
    ema_half_duplex_s: Optional[float] = None
    ema_audio_duplex_s: Optional[float] = None
    ema_omni_duplex_s: Optional[float] = None
    ema_chat_samples: int = 0
    ema_half_duplex_samples: int = 0
    ema_audio_duplex_samples: int = 0
    ema_omni_duplex_samples: int = 0


# ============ 响应模型 ============

class WorkersResponse(BaseModel):
    """Worker 列表响应"""
    total: int
    workers: List[WorkerInfo]
