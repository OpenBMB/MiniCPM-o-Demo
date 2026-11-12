"""Browser connection to a MiniCPM-o vLLM-Omni server."""

import argparse
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException


def _realtime_url(value):
    url = urlsplit(value)
    if url.scheme not in {"ws", "wss"} or not url.netloc or url.path != "/v1/realtime":
        raise argparse.ArgumentTypeError("expected ws(s)://host:port/v1/realtime")
    return value


def add_arguments(parser):
    parser.add_argument(
        "--vllm-omni-url", type=_realtime_url,
        help="Browser-accessible MiniCPM vLLM-Omni WebSocket URL",
    )
    parser.add_argument(
        "--vllm-omni-model", default="openbmb/MiniCPM-o-4_5",
        help="vLLM-Omni served model name",
    )


def configure(args, app, app_registry):
    """Enable the desktop duplex pages and return their connection settings."""
    if not args.vllm_omni_url:
        return None

    url = urlsplit(args.vllm_omni_url)
    health_url = urlunsplit(("https" if url.scheme == "wss" else "http", url.netloc, "/health", "", ""))

    async def health():
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(health_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="vLLM-Omni is unavailable") from exc
        return {"status": "healthy"}

    app.add_api_route("/api/vllm_omni/health", health)
    app_registry.set_enabled("turnbased", False)
    app_registry.set_enabled("half_duplex_audio", False)
    return {"url": args.vllm_omni_url, "model": args.vllm_omni_model}
