"""display_object_on_board service for the standalone board prototype.

Two responsibilities:

1. Frontend-facing: produce a `DisplayObjectOnBoardToolResult` that the board
   page can render (image_url, source_url, title, elapsed_ms, error). When the
   live image backend cannot return an image (network / search source down),
   fall back to a deterministic SVG placeholder so the UI path is still
   exercisable.

2. Model-facing: produce a `tool_response_content` JSON string that matches the
   `display_object_on_board` training schema exactly:

       {"status": "displayed", "name": "<name>", "reason": "<reason>"}

   This is what the model saw during training (see
   `display_object_on_board_midtrain/.../delivery_train_data/*.json` →
   `tracks.input_event.segments[*].event.contents[0].text`). Any divergence
   (different keys, dict-instead-of-string, missing fields) makes the model's
   next prefill see OOD content and can derail the run, so we keep it strict.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from urllib.parse import quote

from audio_duplex_board.schemas import BoardImageResult
from audio_duplex_board.tools.display_object_on_board.live_image_search_backend import (
    search_live_image,
)
from audio_duplex_board.tools.display_object_on_board.schemas import (
    DisplayObjectOnBoardToolResult,
)


DEFAULT_DOWNLOAD_DIR = (
    Path(__file__).resolve().parent / "live_image_downloads"
)


# 推理时没有训练阶段的离线标注 reason，给一个固定且与训练用语风格接近的通用 reason。
# 训练数据典型 reason 例如 "该片段在 user_content 中逐字且唯一出现，属于符合展示规则的具体可见物。"。
# 此通用 reason 不暴露任何运行时元数据；同时保留 name 字段供模型回看。
DEFAULT_RUNTIME_REASON = "运行时识别为符合展示规则的具体可见物，已展示到画板。"


class DisplayObjectOnBoardService:
    """Small service wrapper around display_object_on_board image lookup."""

    def __init__(
        self,
        *,
        download_dir: Path | None = None,
        runtime_reason: str = DEFAULT_RUNTIME_REASON,
    ) -> None:
        """Args:
            download_dir: 实时搜图下载落地目录，默认到本模块下 `live_image_downloads/`。
            runtime_reason: 推理时填到 tool response JSON 的 reason 字段，默认是固定通用句。
        """

        self._download_dir = download_dir or DEFAULT_DOWNLOAD_DIR
        self._download_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_reason = runtime_reason

    def search(self, query: str) -> DisplayObjectOnBoardToolResult:
        """Return a board image result for `query`.

        Args:
            query: 模型 tool call 中的 `name` 字段，既是搜图 query，也是 board 上展示的 query。

        Returns:
            前端 + 模型双视图的工具结果。
        """

        started_at = time.perf_counter()
        safe_query = (query or "").strip() or "object"
        live_error: str | None = None
        try:
            live = search_live_image(
                query_text=safe_query,
                download_dir=self._download_dir,
            )
        except Exception as exc:  # 网络 / 搜索源失败时降级，不让 board 卡死
            live = None
            live_error = f"{type(exc).__name__}: {exc}"

        if live is not None:
            image_url = live.asset_url
            source_url = live.source_url
            title = live.title or safe_query
            elapsed_ms = live.elapsed_ms
            error = None
        else:
            image_url = _placeholder_svg_data_url(safe_query)
            source_url = None
            title = safe_query
            elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
            error = live_error

        return DisplayObjectOnBoardToolResult(
            query=safe_query,
            image_url=image_url,
            source_url=source_url,
            title=title,
            elapsed_ms=elapsed_ms,
            error=error,
            # 训练严格格式：JSON 字符串，schema = {status, name, reason}
            tool_response_content=json.dumps(
                {
                    "status": "displayed",
                    "name": safe_query,
                    "reason": self._runtime_reason,
                },
                ensure_ascii=False,
            ),
        )


def board_image_result_from_tool_result(
    tool_result: DisplayObjectOnBoardToolResult,
    *,
    tool_call_id: str | None,
) -> BoardImageResult:
    """Convert tool result to the BoardImageResult schema used by frontend events."""

    return BoardImageResult(
        query=tool_result.query,
        asset_id=f"tool:{tool_call_id or tool_result.query}",
        image_url=tool_result.image_url,
        source_url=tool_result.source_url,
        title=tool_result.title or tool_result.query,
        elapsed_ms=tool_result.elapsed_ms,
        error=tool_result.error,
    )


def _placeholder_svg_data_url(query: str) -> str:
    """Build a deterministic SVG data URL used when live search fails."""

    escaped = html.escape(query or "object")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop stop-color="#172033" offset="0"/>
      <stop stop-color="#334155" offset="1"/>
    </linearGradient>
  </defs>
  <rect width="640" height="420" rx="32" fill="url(#g)"/>
  <circle cx="110" cy="100" r="44" fill="#38bdf8" opacity="0.85"/>
  <circle cx="540" cy="330" r="72" fill="#f59e0b" opacity="0.70"/>
  <text x="320" y="205" fill="#f8fafc" font-family="sans-serif" font-size="42" font-weight="700" text-anchor="middle">{escaped}</text>
  <text x="320" y="258" fill="#cbd5e1" font-family="sans-serif" font-size="22" text-anchor="middle">display_object_on_board (placeholder)</text>
</svg>"""
    return "data:image/svg+xml;charset=utf-8," + quote(svg)
