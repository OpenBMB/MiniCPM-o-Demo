#!/usr/bin/env bash
#
# worker + backend bundle 容器入口（新链路）。
#
# 容器内拉起两个进程：
#   1. py_backend.server  —— 加载模型，独占本容器可见的 GPU（容器内恒为 cuda:0）
#   2. worker.py          —— 纯转发，--backend-server-url 指向 localhost 的 backend
#
# 健康判定以 backend 的 /health 为准（模型真加载好），而非 worker
#（worker 一进转发模式就报 model_loaded=true，不反映模型状态）。
#
# 环境变量：
#   MODEL_PATH        模型权重路径（容器内，通常是挂载点）。默认 /models/MiniCPM-o-4_5
#   BACKEND_PORT      backend 协议端口。默认 22500
#   WORKER_PORT       worker 转发端口（gateway 连这个）。默认 22400
#   GPU_ID            backend --gpu-id（容器内通常 0，由 --gpus 决定看到哪张卡）。默认 0

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/MiniCPM-o-4_5}"
BACKEND_PORT="${BACKEND_PORT:-22500}"
WORKER_PORT="${WORKER_PORT:-22400}"
GPU_ID="${GPU_ID:-0}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"

cd /app

# ---- config.json：项目要求存在（除 model_path 外字段走默认）----
# model_path 我们用命令行 --model-path 显式覆盖，但 config.json 仍需存在以提供其它默认值。
if [ ! -f /app/config.json ]; then
    cp /app/config.example.json /app/config.json
    echo "[entrypoint] config.json 不存在，已从 config.example.json 生成"
fi

# ---- 校验模型挂载 ----
if [ ! -d "$MODEL_PATH" ]; then
    echo "[entrypoint] 错误：模型目录不存在：$MODEL_PATH" >&2
    echo "[entrypoint] 请用 -v <宿主机权重>:$MODEL_PATH 挂载模型" >&2
    exit 1
fi

echo "=================================================="
echo "  worker + backend bundle"
echo "  MODEL_PATH   = $MODEL_PATH"
echo "  backend      = 127.0.0.1:$BACKEND_PORT  (gpu-id=$GPU_ID)"
echo "  worker       = 0.0.0.0:$WORKER_PORT  -> $BACKEND_URL"
echo "=================================================="

mkdir -p /app/data /app/tmp /app/torch_compile_cache

backend_pid=""
worker_pid=""

cleanup() {
    echo "[entrypoint] 收到终止信号，关闭子进程..."
    [ -n "$worker_pid" ] && kill "$worker_pid" 2>/dev/null || true
    [ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# ---- 1. 启动 backend（加载模型，耗时 30-90s）----
echo "[entrypoint] 启动 py_backend.server ..."
python -m py_backend.server \
    --host 0.0.0.0 --port "$BACKEND_PORT" \
    --gpu-id "$GPU_ID" \
    --model-path "$MODEL_PATH" &
backend_pid=$!

# ---- 2. 等 backend 把模型加载好 ----
echo "[entrypoint] 等待 backend 加载模型..."
for i in $(seq 1 300); do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[entrypoint] 错误：backend 进程已退出（模型加载失败）" >&2
        exit 1
    fi
    if curl -sf "${BACKEND_URL}/health" >/dev/null 2>&1; then
        echo "[entrypoint] backend 就绪（/health OK，用时 ~$((i*2))s）"
        break
    fi
    sleep 2
    if [ "$i" -eq 300 ]; then
        echo "[entrypoint] 错误：backend 600s 内未就绪" >&2
        exit 1
    fi
done

# ---- 3. 启动 worker（纯转发，指向 localhost backend）----
echo "[entrypoint] 启动 worker（转发模式）..."
python worker.py \
    --host 0.0.0.0 --port "$WORKER_PORT" \
    --gpu-id "$GPU_ID" \
    --backend-server-url "$BACKEND_URL" &
worker_pid=$!

sleep 3
if curl -sf "http://127.0.0.1:${WORKER_PORT}/health" >/dev/null 2>&1; then
    echo "[entrypoint] worker 就绪（/health OK）"
else
    echo "[entrypoint] 警告：worker /health 暂未就绪，继续运行"
fi

echo "[entrypoint] bundle 运行中。backend pid=$backend_pid worker pid=$worker_pid"

# ---- 4. 守护：任一进程退出则整容器退出（让 docker restart 能整体拉起）----
while true; do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[entrypoint] backend 进程退出，终止容器" >&2
        cleanup
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        echo "[entrypoint] worker 进程退出，终止容器" >&2
        cleanup
    fi
    sleep 5
done
