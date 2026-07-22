#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5}"
PT_PATH="${PT_PATH:-/user/weihongliang/o45_fc_assets/checkpoints/minicpm-v_100.pt}"
REF_AUDIO_PATH="${REF_AUDIO_PATH:-/user/weihongliang/o45_fc_assets/training/delivery_train_data/media/system_reference/HTRef06.wav}"
SDK_SRC="${SDK_SRC:-/user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_sdk/src}"
PYTHON="${PYTHON:-/user/sunweiyue/lib/swy-dev/.venv/minicpm_o5_demo/bin/python}"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-22500}"
WORKER_HOST="${WORKER_HOST:-127.0.0.1}"
WORKER_PORT="${WORKER_PORT:-23510}"
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8009}"
GPU_ID="${GPU_ID:-0}"

LOG_DIR="${LOG_DIR:-/user/weihongliang/o45_fc_api_cctl_logs}"
ENABLE_FRP="${ENABLE_FRP:-1}"
FRPC_BIN="${FRPC_BIN:-/user/weihongliang/frp_0.65.0_linux_amd64/frpc}"
FRPC_CONFIG="${FRPC_CONFIG:-/user/weihongliang/frp_0.65.0_linux_amd64/frpc_o5_cctl_8009_8444.toml}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${SDK_SRC}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
WORKER_ENDPOINT="${WORKER_HOST}:${WORKER_PORT}"

if [ ! -x "${PYTHON}" ]; then
    echo "[start] missing python: ${PYTHON}" >&2
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[start] missing model dir: ${MODEL_PATH}" >&2
    exit 1
fi
if [ ! -f "${PT_PATH}" ]; then
    echo "[start] missing pt file: ${PT_PATH}" >&2
    exit 1
fi

backend_pid=""
worker_pid=""
gateway_pid=""
frpc_pid=""

cleanup() {
    echo "[start] cleanup"
    [ -n "${frpc_pid}" ] && kill "${frpc_pid}" 2>/dev/null || true
    [ -n "${gateway_pid}" ] && kill "${gateway_pid}" 2>/dev/null || true
    [ -n "${worker_pid}" ] && kill "${worker_pid}" 2>/dev/null || true
    [ -n "${backend_pid}" ] && kill "${backend_pid}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_http() {
    local url="$1"
    local timeout_s="$2"
    local name="$3"
    local elapsed=0
    until curl -sf "${url}" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "${elapsed}" -ge "${timeout_s}" ]; then
            echo "[start] timeout waiting for ${name}: ${url}" >&2
            return 1
        fi
    done
    echo "[start] ${name} ready after ${elapsed}s"
}

wait_https() {
    local url="$1"
    local timeout_s="$2"
    local name="$3"
    local elapsed=0
    until curl -skf "${url}" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "${elapsed}" -ge "${timeout_s}" ]; then
            echo "[start] timeout waiting for ${name}: ${url}" >&2
            return 1
        fi
    done
    echo "[start] ${name} ready after ${elapsed}s"
}

echo "[start] project=${PROJECT_DIR}"
echo "[start] model=${MODEL_PATH}"
echo "[start] pt=${PT_PATH}"
echo "[start] backend=${BACKEND_URL} worker=${WORKER_ENDPOINT} gateway=https://${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "[start] logs=${LOG_DIR}"

"${PYTHON}" -m py_backend.server \
    --host "${BACKEND_HOST}" \
    --port "${BACKEND_PORT}" \
    --model-path "${MODEL_PATH}" \
    --pt-path "${PT_PATH}" \
    --ref-audio-path "${REF_AUDIO_PATH}" \
    --gpu-id "${GPU_ID}" \
    > "${LOG_DIR}/backend.log" 2>&1 &
backend_pid=$!
wait_http "${BACKEND_URL}/health" 900 "backend"

"${PYTHON}" worker.py \
    --host "${WORKER_HOST}" \
    --port "${WORKER_PORT}" \
    --gpu-id "${GPU_ID}" \
    --backend-server-url "${BACKEND_URL}" \
    > "${LOG_DIR}/worker.log" 2>&1 &
worker_pid=$!
wait_http "http://${WORKER_ENDPOINT}/health" 120 "worker"

"${PYTHON}" gateway.py \
    --host "${GATEWAY_HOST}" \
    --port "${GATEWAY_PORT}" \
    --workers "${WORKER_ENDPOINT}" \
    --https \
    --ssl-certfile certs/cert.pem \
    --ssl-keyfile certs/key.pem \
    > "${LOG_DIR}/gateway.log" 2>&1 &
gateway_pid=$!
wait_https "https://127.0.0.1:${GATEWAY_PORT}/health" 120 "gateway"
echo "[start] service ready: https://127.0.0.1:${GATEWAY_PORT}/fc_board"

if [ "${ENABLE_FRP}" = "1" ]; then
    if [ ! -x "${FRPC_BIN}" ]; then
        echo "[start] missing frpc: ${FRPC_BIN}" >&2
        exit 1
    fi
    if [ ! -f "${FRPC_CONFIG}" ]; then
        echo "[start] missing frpc config: ${FRPC_CONFIG}" >&2
        exit 1
    fi
    "${FRPC_BIN}" -c "${FRPC_CONFIG}" > "${LOG_DIR}/frpc.log" 2>&1 &
    frpc_pid=$!
    sleep 3
    if ! kill -0 "${frpc_pid}" 2>/dev/null; then
        echo "[start] frpc exited during startup" >&2
        tail -n 80 "${LOG_DIR}/frpc.log" >&2 || true
        exit 1
    fi
    echo "[start] frpc ready: ${FRPC_CONFIG}"
fi

while true; do
    if [ -n "${frpc_pid}" ] && ! kill -0 "${frpc_pid}" 2>/dev/null; then
        echo "[start] frpc exited" >&2
        tail -n 80 "${LOG_DIR}/frpc.log" >&2 || true
        exit 1
    fi
    if ! kill -0 "${gateway_pid}" 2>/dev/null; then
        echo "[start] gateway exited" >&2
        tail -n 80 "${LOG_DIR}/gateway.log" >&2 || true
        exit 1
    fi
    if ! kill -0 "${worker_pid}" 2>/dev/null; then
        echo "[start] worker exited" >&2
        tail -n 80 "${LOG_DIR}/worker.log" >&2 || true
        exit 1
    fi
    if ! kill -0 "${backend_pid}" 2>/dev/null; then
        echo "[start] backend exited" >&2
        tail -n 80 "${LOG_DIR}/backend.log" >&2 || true
        exit 1
    fi
    sleep 5
done
