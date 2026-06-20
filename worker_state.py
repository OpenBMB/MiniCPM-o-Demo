"""Shared worker state models used by worker hosts and backend implementations."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class WorkerStatus(str, Enum):
    """Worker 状态"""
    LOADING = "loading"        # 正在加载模型
    IDLE = "idle"              # 空闲（可接受新请求）
    BUSY_CHAT = "busy_chat"    # 正在处理 Chat 请求
    BUSY_HALF_DUPLEX = "busy_half_duplex"  # 正在处理 Half-Duplex 请求
    DUPLEX_ACTIVE = "duplex_active"    # Duplex 活跃中
    DUPLEX_PAUSED = "duplex_paused"    # Duplex 暂停中
    ERROR = "error"            # 异常状态


class WorkerState(BaseModel):
    """Worker 运行时状态"""
    status: WorkerStatus = WorkerStatus.LOADING
    current_ticket_id: Optional[str] = None
    duplex_pause_time: Optional[float] = None  # Duplex 暂停的时间戳
    total_requests: int = 0
    total_inference_time_ms: float = 0.0
    last_activity: Optional[str] = None

    @property
    def is_idle(self) -> bool:
        return self.status == WorkerStatus.IDLE

    @property
    def is_busy(self) -> bool:
        return self.status in (
            WorkerStatus.BUSY_CHAT,
            WorkerStatus.BUSY_HALF_DUPLEX,
            WorkerStatus.DUPLEX_ACTIVE,
            WorkerStatus.DUPLEX_PAUSED,
        )
