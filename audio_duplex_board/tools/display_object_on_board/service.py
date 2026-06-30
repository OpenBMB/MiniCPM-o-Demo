"""display_object_on_board service for the standalone board prototype.

The first prototype uses a deterministic SVG placeholder so the UI and event
protocol can be developed without depending on external image search
availability. A real search backend can replace `search()` later without
changing the board session API.
"""

from __future__ import annotations

import html
import importlib.util
import time
from pathlib import Path
from urllib.parse import quote

from audio_duplex_board.schemas import BoardImageResult
from audio_duplex_board.tools.display_object_on_board.schemas import (
    DisplayObjectOnBoardToolResult,
)


class DisplayObjectOnBoardService:
    """Small service wrapper around display_object_on_board image lookup."""

    def search(self, query: str) -> DisplayObjectOnBoardToolResult:
        """Return a board image result for `query`.

        Args:
            query: Object name from the model tool call.

        Returns:
            Tool result containing a frontend image URL and a model-facing tool
            response payload. The model-facing payload intentionally stays small
            and training-format friendly.
        """

        started_at = time.perf_counter()
        safe_query = query.strip()
        live_result = _try_live_search(safe_query)
        if live_result is not None:
            return DisplayObjectOnBoardToolResult(
                query=safe_query,
                image_url=live_result.asset_url,
                source_url=live_result.source_url,
                title=live_result.title,
                elapsed_ms=live_result.elapsed_ms,
                tool_response_content={
                    "status": "success",
                    "displayed_object_name": safe_query,
                },
            )
        image = BoardImageResult(
            query=safe_query,
            asset_id=f"placeholder:{safe_query}",
            image_url=_placeholder_svg_data_url(safe_query),
            title=safe_query,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
        return DisplayObjectOnBoardToolResult(
            query=safe_query,
            image_url=image.image_url,
            source_url=image.source_url,
            title=image.title,
            elapsed_ms=image.elapsed_ms,
            tool_response_content={
                "status": "success",
                "displayed_object_name": safe_query,
            },
        )


def _placeholder_svg_data_url(query: str) -> str:
    """Build a deterministic SVG data URL for placeholder board cards."""

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
  <text x="320" y="258" fill="#cbd5e1" font-family="sans-serif" font-size="22" text-anchor="middle">display_object_on_board</text>
</svg>"""
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def _try_live_search(query: str):
    """Try the swy-dev live image backend when available.

    The standalone prototype should still run without swy-dev internet helpers,
    so any import / network / download failure falls back to placeholder cards.
    """

    backend_path = Path(
        "/user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_dataset/"
        "display_object_on_board_midtrain/display_object_tool/live_image_search_backend.py"
    )
    if not backend_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_audio_duplex_board_live_image", backend_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        download_dir = Path(__file__).resolve().parent / "live_image_downloads"
        return module.search_live_image(query, download_dir=download_dir)
    except Exception as exc:
        print(f"[audio_duplex_board] live image fallback query={query!r}: {exc}", flush=True)
        return None
