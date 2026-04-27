"""APP 注册表

管理前端 APP 的启用/禁用状态，支持通过 Admin API 动态切换。
纯内存实现，服务重启后恢复默认（全部启用）。
"""

import threading
from typing import List, Optional

from pydantic import BaseModel, Field


class AppInfo(BaseModel):
    """单个 APP 的信息"""
    app_id: str = Field(description="APP 唯一标识，如 turnbased / omni / audio_duplex")
    name: str = Field(description="APP 显示名称")
    route: str = Field(description="APP 页面路由，如 /turnbased")
    enabled: bool = Field(default=True, description="是否启用")


class AppPublicInfo(BaseModel):
    """返回给前端的 APP 信息（不含 enabled 字段）"""
    app_id: str
    name: str
    route: str


class AppToggleRequest(BaseModel):
    """切换 APP 启用状态的请求"""
    enabled: bool


class AppsPublicResponse(BaseModel):
    """前端 API 响应：当前启用的 APP 列表"""
    apps: List[AppPublicInfo]


class AppsAdminResponse(BaseModel):
    """Admin API 响应：所有 APP 列表（含 enabled 状态）"""
    apps: List[AppInfo]


# 默认 APP 定义（有序）
_DEFAULT_APPS = [
    AppInfo(app_id="turnbased", name="Turn-based Chat", route="/turnbased", enabled=True),
    AppInfo(app_id="half_duplex_audio", name="Half-Duplex Audio", route="/half_duplex", enabled=True),
    AppInfo(app_id="omni", name="Omni Full-Duplex", route="/omni", enabled=True),
    AppInfo(app_id="audio_duplex", name="Audio Full-Duplex", route="/audio_duplex", enabled=True),
    AppInfo(app_id="realtime", name="Realtime API", route="/realtime", enabled=True),
]


class AppRegistry:
    """APP 注册表

    线程安全的内存注册表，管理 APP 启用/禁用状态。
    Gateway 启动时初始化，默认全部 APP 启用。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._apps: dict[str, AppInfo] = {
            a.app_id: a.model_copy() for a in _DEFAULT_APPS
        }

    def is_enabled(self, app_id: str) -> bool:
        with self._lock:
            app = self._apps.get(app_id)
            return app.enabled if app else False

    def set_enabled(self, app_id: str, enabled: bool) -> Optional[AppInfo]:
        """设置 APP 启用状态，返回更新后的 AppInfo；app_id 不存在返回 None"""
        with self._lock:
            app = self._apps.get(app_id)
            if app is None:
                return None
            app.enabled = enabled
            return app.model_copy()

    def get_enabled_apps(self) -> List[AppPublicInfo]:
        """返回启用的 APP 列表（保持注册顺序）"""
        with self._lock:
            return [
                AppPublicInfo(app_id=a.app_id, name=a.name, route=a.route)
                for a in self._apps.values()
                if a.enabled
            ]

    def get_all_apps(self) -> List[AppInfo]:
        """返回所有 APP 列表（Admin 用）"""
        with self._lock:
            return [a.model_copy() for a in self._apps.values()]
