"""Runtime manager for worker-local sessions."""

from __future__ import annotations

from typing import Dict

from core.runtime.backends import DuplexRuntimeBackend
from core.runtime.duplex import DuplexSessionRuntime


class RuntimeManager:
    """Own worker-local runtime instances by session id.

    This is intentionally worker-local.  Cross-worker routing/leases remain a
    gateway/scheduler concern; this manager only keeps the hot runtime objects
    living inside one worker process.
    """

    def __init__(self) -> None:
        self._duplex: Dict[str, DuplexSessionRuntime] = {}

    def create_duplex(
        self,
        session_id: str,
        backend: DuplexRuntimeBackend,
    ) -> DuplexSessionRuntime:
        if session_id in self._duplex:
            raise RuntimeError(f"duplex runtime already exists for session: {session_id}")
        runtime = DuplexSessionRuntime(backend)
        self._duplex[session_id] = runtime
        return runtime

    def get_duplex(self, session_id: str) -> DuplexSessionRuntime:
        return self._duplex[session_id]

    def forget_duplex(self, session_id: str) -> None:
        """Drop a runtime that has already been closed elsewhere."""
        self._duplex.pop(session_id, None)

    async def close_duplex(self, session_id: str) -> None:
        runtime = self._duplex.pop(session_id, None)
        if runtime is not None:
            await runtime.close()

    async def close_all(self) -> None:
        session_ids = list(self._duplex)
        for session_id in session_ids:
            await self.close_duplex(session_id)

