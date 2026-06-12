#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/opt/llama.cpp-omni/bin/llama-omni-server}"
GGUF_MODEL="${GGUF_MODEL:-/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf}"
BACKEND_BIND_HOST="${BACKEND_BIND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-22500}"
WORKER_PORT="${WORKER_PORT:-22400}"
GPU_ID="${GPU_ID:-0}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1200}"
CHECK_MODEL_LAYOUT="${CHECK_MODEL_LAYOUT:-1}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
LOG_DIR="${LOG_DIR:-/app/logs}"
LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/llama-omni-server.log}"

cd /app
mkdir -p "$LOG_DIR"

require_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "[entrypoint] missing required path: $path" >&2
        exit 1
    fi
}

require_path "$LLAMA_SERVER_BIN"
require_path "$GGUF_MODEL"

if [ "$CHECK_MODEL_LAYOUT" = "1" ]; then
    model_root="$(dirname "$GGUF_MODEL")"
    require_path "${model_root}/vision/MiniCPM-o-4_5-vision-F16.gguf"
    require_path "${model_root}/audio/MiniCPM-o-4_5-audio-F16.gguf"
    require_path "${model_root}/tts/MiniCPM-o-4_5-tts-F16.gguf"
    require_path "${model_root}/tts/MiniCPM-o-4_5-projector-F16.gguf"
    require_path "${model_root}/token2wav-gguf"
fi

echo "=================================================="
echo "  C++ worker-backend bundle"
echo "  llama server = $LLAMA_SERVER_BIN"
echo "  GGUF_MODEL   = $GGUF_MODEL"
echo "  backend      = ${BACKEND_BIND_HOST}:${BACKEND_PORT}"
echo "  worker       = 0.0.0.0:${WORKER_PORT} -> ${BACKEND_URL}"
echo "  gpu-id       = ${GPU_ID}"
echo "  n-gpu-layers = ${N_GPU_LAYERS}"
echo "=================================================="

backend_pid=""
worker_pid=""
tail_pid=""

cleanup() {
    echo "[entrypoint] stopping child processes..."
    [ -n "$worker_pid" ] && kill "$worker_pid" 2>/dev/null || true
    [ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null || true
    [ -n "$tail_pid" ] && kill "$tail_pid" 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

: > "$LLAMA_LOG_FILE"
tail -n +1 -F "$LLAMA_LOG_FILE" &
tail_pid=$!

llama_args=(
    -m "$GGUF_MODEL"
    -ngl "$N_GPU_LAYERS"
    --host "$BACKEND_BIND_HOST"
    --port "$BACKEND_PORT"
)

if [ -n "${LLAMA_SERVER_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_args=( $LLAMA_SERVER_EXTRA_ARGS )
    llama_args+=( "${extra_args[@]}" )
fi

echo "[entrypoint] starting llama server..."
"$LLAMA_SERVER_BIN" "${llama_args[@]}" >> "$LLAMA_LOG_FILE" 2>&1 &
backend_pid=$!

echo "[entrypoint] waiting for backend /health..."
max_retries=$((READY_TIMEOUT_S / 2))
if [ "$max_retries" -lt 1 ]; then
    max_retries=1
fi

for i in $(seq 1 "$max_retries"); do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[entrypoint] llama server exited while loading" >&2
        tail -50 "$LLAMA_LOG_FILE" >&2 || true
        cleanup
    fi
    if curl -sf "${BACKEND_URL}/health" >/dev/null 2>&1; then
        echo "[entrypoint] backend ready after ~$((i * 2))s"
        break
    fi
    if [ "$i" -eq "$max_retries" ]; then
        echo "[entrypoint] backend did not become ready within ${READY_TIMEOUT_S}s" >&2
        tail -80 "$LLAMA_LOG_FILE" >&2 || true
        cleanup
    fi
    sleep 2
done

echo "[entrypoint] starting worker..."
python worker.py \
    --host 0.0.0.0 \
    --port "$WORKER_PORT" \
    --gpu-id "$GPU_ID" \
    --backend-server-url "$BACKEND_URL" &
worker_pid=$!

sleep 3
if curl -sf "http://127.0.0.1:${WORKER_PORT}/health" >/dev/null 2>&1; then
    echo "[entrypoint] worker ready"
else
    echo "[entrypoint] worker health check is not ready yet"
fi

echo "[entrypoint] running. backend pid=${backend_pid} worker pid=${worker_pid}"

while true; do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[entrypoint] llama server exited" >&2
        cleanup
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        echo "[entrypoint] worker exited" >&2
        cleanup
    fi
    sleep 5
done
