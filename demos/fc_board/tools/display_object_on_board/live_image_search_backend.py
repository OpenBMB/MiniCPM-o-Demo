"""实时网页图片搜索后端（FC board demo 本地副本）。

本文件是从 swy-dev 仓库
`omni_agent_research/minicpm_o5_dataset/display_object_on_board_midtrain/display_object_tool/live_image_search_backend.py`
复制而来，使 FC board demo 不再依赖 swy-dev 绝对路径 import。两份代码应保持同步；
后续应在两边都改并同步提交。

本模块为 `display_object_on_board` 提供"实时搜图"能力：调用方传入 `query_text`，
后端实时请求图片搜索源，下载排名靠前的候选并返回相关性较高、下载成功的一张图片。
不缓存搜索结果，每次都重新发起请求。

当前实现优先使用 360 图片搜索，因为它对中文细粒度商品词覆盖较好；当 360 候选全部下载
失败时再使用 Bing 图片搜索兜底。选择策略不是"谁先下载成功返回谁"，而是先按标题相关性
排序，并发下载排名靠前的候选，在短宽限期内返回排名最高的成功候选。

原 backend 还有 COCO 真实图片 / Iconify SVG 两种后端和验收 HTML；那些不属于 demo
MVP 范围，因此本副本只保留实时搜图能力。
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import ssl
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field


DEFAULT_DOWNLOAD_DIR = Path(__file__).resolve().parent / "live_image_downloads"
DEFAULT_MAX_DOWNLOAD_CANDIDATES = 5
DEFAULT_SEARCH_TIMEOUT_SECONDS = 2.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 2.0
DEFAULT_SELECTION_GRACE_SECONDS = 0.75


class LiveImageCandidate(BaseModel):
    """实时图片搜索候选。

    参数:
        image_url: 可下载图片 URL，可能是原图 URL，也可能是搜索源代理缩略图 URL。
        title: 搜索结果标题，用于相关性排序和人工验收。
        source: 搜索来源，例如 `360` 或 `bing`。
        rank: 搜索源返回后的重排名次，从 0 开始。
        relevance_score: 标题相关性分数，越高越优先。
    """

    image_url: str = Field(..., min_length=1)
    title: str = ""
    source: str = Field(..., min_length=1)
    rank: int = 0
    relevance_score: int = 0


class LiveImageSearchResult(BaseModel):
    """实时图片后端返回结果。

    参数:
        query_text: 本次搜索 query。
        asset_id: 本地资产 ID。
        asset_type: 资产类型，固定为 `image`。
        asset_url: 服务前端可访问的相对 URL。
        local_path: 本地下载文件路径。
        source_url: 图片原始 URL。
        source: 搜索来源。
        title: 搜索结果标题。
        relevance_score: 标题相关性分数。
        elapsed_ms: 本次搜索和下载总耗时。
        candidates: 排序后的候选列表，用于前端展示和调试。
    """

    query_text: str
    asset_id: str
    asset_type: str = "image"
    asset_url: str
    local_path: Path
    source_url: str
    source: str
    title: str
    relevance_score: int
    elapsed_ms: float
    candidates: list[LiveImageCandidate] = Field(default_factory=list)


class DownloadedLiveImage(BaseModel):
    """已下载的实时图片候选。

    参数:
        candidate: 对应搜索候选。
        file_path: 本地下载文件路径。
        elapsed_seconds: 单个候选下载耗时。
    """

    candidate: LiveImageCandidate
    file_path: Path
    elapsed_seconds: float


def normalize_keyword(keyword: str) -> str:
    """标准化搜索关键词。

    参数:
        keyword: 用户输入或工具传入的搜索词。

    返回:
        去掉首尾空白并压缩内部空白后的关键词。
    """

    return re.sub(r"\s+", " ", keyword.strip())


def build_browser_headers(referer: str) -> dict[str, str]:
    """构造搜索和下载用 HTTP headers。

    参数:
        referer: HTTP Referer。

    返回:
        模拟普通浏览器访问的 headers。
    """

    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": referer,
    }


def decode_search_text(raw_text: str) -> str:
    """解码搜索页里常见的 HTML/JSON 转义文本。

    参数:
        raw_text: 搜索页中提取到的原始字符串。

    返回:
        尽量还原后的可读字符串。
    """

    unescaped_text = html.unescape(raw_text).replace("\\/", "/")
    return unescaped_text.encode("utf-8").decode("unicode_escape", errors="ignore")


def score_candidate_relevance(keyword: str, title: str) -> int:
    """根据关键词和标题计算轻量相关性分数。

    参数:
        keyword: 搜索关键词。
        title: 搜索结果标题。

    返回:
        分数越高表示标题越像用户想要的结果。
    """

    normalized_keyword = normalize_keyword(keyword).lower()
    normalized_title = title.lower()
    score = 0

    if normalized_keyword and normalized_keyword in normalized_title:
        score += 100

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized_keyword)
    bigrams = {"".join(chinese_chars[index : index + 2]) for index in range(len(chinese_chars) - 1)}
    for token in bigrams:
        if token in normalized_title:
            score += 8

    for token in re.findall(r"[a-zA-Z0-9]+", normalized_keyword):
        if token in normalized_title:
            score += 5

    return score


def rank_candidates(keyword: str, candidates: list[LiveImageCandidate]) -> list[LiveImageCandidate]:
    """按标题相关性和标题共识对候选排序。

    参数:
        keyword: 搜索关键词。
        candidates: 搜索源返回的原始候选。

    返回:
        已重排并写入 `rank` / `relevance_score` 的候选列表。
    """

    title_counts: dict[str, int] = {}
    for candidate in candidates:
        title_counts[candidate.title] = title_counts.get(candidate.title, 0) + 1

    scored_candidates: list[tuple[int, int, LiveImageCandidate]] = []
    for index, candidate in enumerate(candidates):
        base_score = score_candidate_relevance(keyword, candidate.title)
        consensus_boost = min(title_counts.get(candidate.title, 1) - 1, 3) * 2
        scored_candidates.append((base_score + consensus_boost, -index, candidate))

    ranked: list[LiveImageCandidate] = []
    for rank, (score, _negative_index, candidate) in enumerate(sorted(scored_candidates, reverse=True)):
        ranked.append(
            LiveImageCandidate(
                image_url=candidate.image_url,
                title=candidate.title,
                source=candidate.source,
                rank=rank,
                relevance_score=score,
            )
        )
    return ranked


def search_360_candidates(keyword: str, timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS) -> list[LiveImageCandidate]:
    """从 360 图片搜索提取候选图片。

    参数:
        keyword: 搜索关键词。
        timeout_seconds: 搜索页面请求超时时间。

    返回:
        已按标题相关性排序的候选列表。
    """

    search_url = f"https://image.so.com/i?q={quote(keyword)}"
    request = Request(search_url, headers=build_browser_headers("https://image.so.com/"))
    with urlopen(request, timeout=timeout_seconds) as response:
        page_html = response.read().decode("utf-8", errors="ignore")

    candidates: list[LiveImageCandidate] = []
    seen_urls: set[str] = set()
    result_blocks = page_html.split('{"id":')
    for result_block in result_blocks[1:]:
        raw_title_match = re.search(r'"title"\s*:\s*"(.*?)"', result_block)
        if raw_title_match is None:
            continue
        raw_url_matches = re.findall(r'"(?:img|thumb|_thumb)"\s*:\s*"(.*?)"', result_block)
        if not raw_url_matches:
            continue

        title = decode_search_text(raw_title_match.group(1))
        for raw_url in raw_url_matches:
            image_url = decode_search_text(raw_url)
            if not image_url.startswith(("http://", "https://")):
                continue
            if image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            candidates.append(LiveImageCandidate(image_url=image_url, title=title, source="360"))

    return rank_candidates(keyword, candidates)


def search_bing_candidates(keyword: str, timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS) -> list[LiveImageCandidate]:
    """从 Bing 图片搜索提取候选图片。

    参数:
        keyword: 搜索关键词。
        timeout_seconds: 搜索页面请求超时时间。

    返回:
        已按标题相关性排序的候选列表。
    """

    search_url = f"https://www.bing.com/images/search?q={quote(keyword)}&first=1"
    request = Request(search_url, headers=build_browser_headers("https://www.bing.com/"))
    with urlopen(request, timeout=timeout_seconds) as response:
        page_html = response.read().decode("utf-8", errors="ignore")

    candidates: list[LiveImageCandidate] = []
    seen_urls: set[str] = set()
    metadata_blocks = re.findall(r'm=\\"(\{.*?\})\\"', page_html)
    for block in metadata_blocks:
        try:
            payload = json.loads(html.unescape(block))
        except json.JSONDecodeError:
            continue
        image_url = str(payload.get("murl") or "")
        if not image_url.startswith(("http://", "https://")):
            continue
        if image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        candidates.append(
            LiveImageCandidate(
                image_url=image_url,
                title=str(payload.get("t") or ""),
                source="bing",
            )
        )

    if not candidates:
        raw_urls = re.findall(r'"murl"\s*:\s*"(.*?)"', page_html)
        raw_urls.extend(re.findall(r"murl&quot;:&quot;(.*?)&quot;", page_html))
        for raw_url in raw_urls:
            image_url = decode_search_text(raw_url)
            if not image_url.startswith(("http://", "https://")):
                continue
            if image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            candidates.append(LiveImageCandidate(image_url=image_url, title="", source="bing"))

    return rank_candidates(keyword, candidates)


def guess_suffix(content_type: str, image_url: str, image_bytes: bytes) -> str:
    """根据响应类型、URL 和文件头推测图片扩展名。

    参数:
        content_type: HTTP 响应 Content-Type。
        image_url: 图片 URL。
        image_bytes: 图片二进制内容。

    返回:
        图片扩展名。
    """

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if b"ftypavif" in image_bytes[:32]:
        return ".avif"

    lowered_content_type = content_type.lower()
    lowered_url = image_url.lower()
    if "avif" in lowered_content_type or lowered_url.endswith(".avif"):
        return ".avif"
    if "png" in lowered_content_type or lowered_url.endswith(".png"):
        return ".png"
    if "webp" in lowered_content_type or lowered_url.endswith(".webp"):
        return ".webp"
    if "gif" in lowered_content_type or lowered_url.endswith(".gif"):
        return ".gif"
    return ".jpg"


def download_referer_for_candidate(candidate: LiveImageCandidate) -> str:
    """根据候选来源选择下载 Referer。

    参数:
        candidate: 图片候选项。

    返回:
        更贴近来源站点的 Referer。
    """

    if candidate.source == "360":
        return "https://image.so.com/"
    if candidate.source == "bing":
        return "https://www.bing.com/"
    return "https://www.baidu.com/"


def download_candidate(
    candidate: LiveImageCandidate,
    download_dir: Path,
    keyword: str,
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> DownloadedLiveImage:
    """下载单个实时图片候选。

    参数:
        candidate: 图片候选项。
        download_dir: 图片下载目录。
        keyword: 搜索关键词。
        timeout_seconds: 下载超时时间。

    返回:
        下载成功后的本地图片信息。
    """

    started_at = time.perf_counter()
    download_dir.mkdir(parents=True, exist_ok=True)
    request = Request(candidate.image_url, headers=build_browser_headers(download_referer_for_candidate(candidate)))
    with urlopen(request, timeout=timeout_seconds, context=ssl._create_unverified_context()) as response:
        content_type = response.headers.get("Content-Type", "")
        image_bytes = response.read()

    if len(image_bytes) < 2048:
        raise ValueError("图片响应过小，可能不是有效图片")
    if not content_type.lower().startswith("image/") and b"<html" in image_bytes[:512].lower():
        raise ValueError(f"响应不是图片: {content_type}")

    keyword_hash = hashlib.sha1(keyword.encode("utf-8")).hexdigest()[:12]
    url_hash = hashlib.sha1(candidate.image_url.encode("utf-8")).hexdigest()[:12]
    request_hash = hashlib.sha1(f"{time.time_ns()}:{candidate.image_url}".encode("utf-8")).hexdigest()[:8]
    suffix = guess_suffix(content_type, candidate.image_url, image_bytes)
    file_path = download_dir / f"{keyword_hash}_{url_hash}_{request_hash}{suffix}"
    file_path.write_bytes(image_bytes)

    return DownloadedLiveImage(
        candidate=candidate,
        file_path=file_path,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def fetch_best_downloaded_image(
    *,
    keyword: str,
    download_dir: Path,
    candidates: list[LiveImageCandidate],
    started_at: float,
    max_download_candidates: int = DEFAULT_MAX_DOWNLOAD_CANDIDATES,
    selection_grace_seconds: float = DEFAULT_SELECTION_GRACE_SECONDS,
) -> DownloadedLiveImage:
    """从候选列表中并发下载并返回排名最高的成功图片。

    参数:
        keyword: 搜索关键词。
        download_dir: 图片下载目录。
        candidates: 已经按相关性排序的候选图片。
        started_at: 整次搜索开始时间。
        max_download_candidates: 最多并发尝试的候选数量。
        selection_grace_seconds: 低排名候选先成功后等待高排名候选的宽限时间。

    返回:
        下载成功且排名最高的图片。
    """

    selected_candidates = candidates[:max_download_candidates]
    if not selected_candidates:
        raise RuntimeError("没有可下载候选图片")

    errors: list[str] = []
    best_success: tuple[int, DownloadedLiveImage] | None = None
    pending_ranks = set(range(len(selected_candidates)))
    executor = ThreadPoolExecutor(max_workers=len(selected_candidates))
    try:
        future_to_candidate = {
            executor.submit(download_candidate, candidate, download_dir, keyword): (rank, candidate)
            for rank, candidate in enumerate(selected_candidates)
        }
        pending_futures: set[Future[DownloadedLiveImage]] = set(future_to_candidate.keys())
        best_success_at: float | None = None
        while pending_futures:
            done_futures, pending_futures = wait(
                pending_futures,
                timeout=0.05,
                return_when=FIRST_COMPLETED,
            )
            if not done_futures:
                if best_success is not None and best_success_at is not None:
                    if time.perf_counter() - best_success_at >= selection_grace_seconds:
                        return best_success[1]
                continue

            for future in done_futures:
                rank, candidate = future_to_candidate[future]
                pending_ranks.discard(rank)
                try:
                    image = future.result()
                    if best_success is None or rank < best_success[0]:
                        best_success = (rank, image)
                        best_success_at = time.perf_counter()
                    if all(pending_rank > best_success[0] for pending_rank in pending_ranks):
                        return best_success[1]
                except Exception as exc:
                    errors.append(f"{candidate.image_url}: {type(exc).__name__}: {exc}")

            if best_success is not None and best_success_at is not None:
                if time.perf_counter() - best_success_at >= selection_grace_seconds:
                    return best_success[1]
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if best_success is not None:
        return best_success[1]
    raise RuntimeError("候选图片全部下载失败: " + " | ".join(errors[:3]))


def search_live_image(
    query_text: str,
    *,
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
    max_download_candidates: int = DEFAULT_MAX_DOWNLOAD_CANDIDATES,
) -> LiveImageSearchResult:
    """实时搜索并下载一张图片。

    参数:
        query_text: 图片搜索 query。
        download_dir: 本地下载目录。
        max_download_candidates: 最多并发尝试的候选数量。

    返回:
        实时搜索后下载成功的图片结果。
    """

    normalized_query = normalize_keyword(query_text)
    if not normalized_query:
        raise ValueError("搜索 query 不能为空")

    started_at = time.perf_counter()
    primary_candidates = search_360_candidates(normalized_query)
    candidates_for_response = primary_candidates
    if primary_candidates:
        try:
            downloaded = fetch_best_downloaded_image(
                keyword=normalized_query,
                download_dir=download_dir,
                candidates=primary_candidates,
                started_at=started_at,
                max_download_candidates=max_download_candidates,
            )
            return build_live_result(normalized_query, downloaded, candidates_for_response, started_at)
        except Exception as exc:
            print(f"[live primary failed] query={normalized_query!r}: {exc}")

    fallback_candidates = search_bing_candidates(normalized_query)
    candidates_for_response = fallback_candidates
    if fallback_candidates:
        downloaded = fetch_best_downloaded_image(
            keyword=normalized_query,
            download_dir=download_dir,
            candidates=fallback_candidates,
            started_at=started_at,
            max_download_candidates=max_download_candidates,
        )
        return build_live_result(normalized_query, downloaded, candidates_for_response, started_at)

    raise RuntimeError("没有搜索到实时图片候选")


def build_live_result(
    query_text: str,
    downloaded: DownloadedLiveImage,
    candidates: list[LiveImageCandidate],
    started_at: float,
) -> LiveImageSearchResult:
    """构造实时图片搜索结果。

    参数:
        query_text: 搜索 query。
        downloaded: 下载成功的候选图片。
        candidates: 当前搜索源排序后的候选列表。
        started_at: 整次搜索开始时间。

    返回:
        `LiveImageSearchResult`。
    """

    file_name = downloaded.file_path.name
    asset_id = f"live-image:{downloaded.candidate.source}:{hashlib.sha1(file_name.encode('utf-8')).hexdigest()[:12]}"
    return LiveImageSearchResult(
        query_text=query_text,
        asset_id=asset_id,
        asset_url=f"/live-image-downloads/{quote(file_name)}",
        local_path=downloaded.file_path,
        source_url=downloaded.candidate.image_url,
        source=downloaded.candidate.source,
        title=downloaded.candidate.title,
        relevance_score=downloaded.candidate.relevance_score,
        elapsed_ms=round((time.perf_counter() - started_at) * 1000, 3),
        candidates=candidates[:12],
    )
