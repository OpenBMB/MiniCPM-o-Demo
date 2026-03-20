#!/bin/bash
# Docker entrypoint for MiniCPM-o 4.5 Service
#
# Workspace mode (recommended):
#   Mount a single directory to /workspace containing models, config, etc.
#   docker run --gpus all -p 8006:8006 -v /path/to/my-workspace:/workspace minicpm-o-demo
#
#   Expected workspace layout:
#     my-workspace/
#     ├── models/MiniCPM-o-4_5/   # Model weights (required)
#     ├── config.json              # Custom config (optional)
#     ├── certs/                   # TLS certs (optional, for HTTPS)
#     ├── data/                    # Auto-created: persistent session data
#     └── torch_compile_cache/     # Auto-created: compilation cache
#
# Environment Variables:
#   GATEWAY_PORT       Gateway port (default: from config.json, fallback 8006)
#   WORKER_BASE_PORT   Worker base port (default: from config.json, fallback 22400)
#   GATEWAY_PROTO      "https" or "http" (default: http)
#   NUM_GPUS_OVERRIDE  Override GPU count (default: auto-detect)

set -e

# ============ Workspace setup ============

WORKSPACE="/workspace"

if [ -d "$WORKSPACE" ]; then
    echo "[Workspace] Detected /workspace mount"

    # config.json: use workspace version if present, else auto-create
    if [ -f "$WORKSPACE/config.json" ]; then
        ln -sf "$WORKSPACE/config.json" /app/config.json
        echo "[Workspace] Using config.json from workspace"
    else
        echo "[Workspace] config.json not found, auto-creating from template"
        cp /app/config.example.json "$WORKSPACE/config.json"
        # If local model exists, patch model_path to use workspace path
        if [ -d "$WORKSPACE/models/MiniCPM-o-4_5" ]; then
            python -c "
import json
p = '$WORKSPACE/config.json'
with open(p) as f: c = json.load(f)
c['model']['model_path'] = '/workspace/models/MiniCPM-o-4_5'
with open(p, 'w') as f: json.dump(c, f, indent=4, ensure_ascii=False)
"
            echo "[Workspace] Set model_path to /workspace/models/MiniCPM-o-4_5 (local model detected)"
        fi
        ln -sf "$WORKSPACE/config.json" /app/config.json
    fi

    # certs/: symlink if present, else auto-generate self-signed certs
    if [ -d "$WORKSPACE/certs" ] && [ -f "$WORKSPACE/certs/cert.pem" ] && [ -f "$WORKSPACE/certs/key.pem" ]; then
        ln -sfn "$WORKSPACE/certs" /app/certs
        echo "[Workspace] Using certs/ from workspace"
    else
        mkdir -p "$WORKSPACE/certs"
        openssl req -x509 -newkey rsa:2048 \
            -keyout "$WORKSPACE/certs/key.pem" \
            -out "$WORKSPACE/certs/cert.pem" \
            -days 365 -nodes \
            -subj "/CN=minicpm-o-demo" \
            2>/dev/null
        ln -sfn "$WORKSPACE/certs" /app/certs
        echo "[Workspace] Auto-generated self-signed TLS certs (valid 365 days)"
    fi

    # data/: persist to workspace
    mkdir -p "$WORKSPACE/data"
    rm -rf /app/data
    ln -sfn "$WORKSPACE/data" /app/data
    echo "[Workspace] data/ -> workspace (persistent)"

    # torch_compile_cache/: persist to workspace
    mkdir -p "$WORKSPACE/torch_compile_cache"
    rm -rf /app/torch_compile_cache
    ln -sfn "$WORKSPACE/torch_compile_cache" /app/torch_compile_cache
    echo "[Workspace] torch_compile_cache/ -> workspace (persistent)"

    echo ""
fi

# Fallback: ensure certs exist even without workspace mount
if [ ! -f /app/certs/cert.pem ] || [ ! -f /app/certs/key.pem ]; then
    mkdir -p /app/certs
    openssl req -x509 -newkey rsa:2048 \
        -keyout /app/certs/key.pem \
        -out /app/certs/cert.pem \
        -days 365 -nodes \
        -subj "/CN=minicpm-o-demo" \
        2>/dev/null
    echo "[Certs] Auto-generated self-signed TLS certs (valid 365 days)"
fi

export TORCHINDUCTOR_CACHE_DIR=/app/torch_compile_cache

GATEWAY_PROTO="${GATEWAY_PROTO:-http}"
GATEWAY_EXTRA_ARGS=""
if [ "$GATEWAY_PROTO" = "http" ]; then
    GATEWAY_EXTRA_ARGS="--http"
fi

# ============ Read config ============

GATEWAY_PORT="${GATEWAY_PORT:-$(python -c "from config import get_config; print(get_config().gateway_port)" 2>/dev/null || echo 8006)}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-$(python -c "from config import get_config; print(get_config().worker_base_port)" 2>/dev/null || echo 22400)}"

# ============ Detect GPUs ============

if [ -n "$NUM_GPUS_OVERRIDE" ]; then
    NUM_GPUS=$NUM_GPUS_OVERRIDE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_LIST=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
else
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
fi

echo "=================================================="
echo "  MiniCPM-o 4.5 Service (Docker)"
echo "=================================================="
echo "  GPUs:    $GPU_LIST ($NUM_GPUS)"
echo "  Gateway: ${GATEWAY_PROTO}://0.0.0.0:$GATEWAY_PORT"
echo "  Workers: localhost:$WORKER_BASE_PORT ~ localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1))"
echo "=================================================="

mkdir -p tmp data

# ============ Start Workers ============

WORKER_ADDRS=""
GPU_IDX=0

cleanup() {
    echo "Shutting down..."
    kill $(cat tmp/*.pid 2>/dev/null) 2>/dev/null
    wait
    exit 0
}
trap cleanup SIGTERM SIGINT

for GPU_ID in $(echo "$GPU_LIST" | tr ',' ' '); do
    WORKER_PORT=$((WORKER_BASE_PORT + GPU_IDX))
    echo "[Worker $GPU_IDX] Starting on GPU $GPU_ID, port $WORKER_PORT..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python worker.py \
        --port $WORKER_PORT \
        --gpu-id $GPU_ID \
        --worker-index $GPU_IDX \
        > "tmp/worker_${GPU_IDX}.log" 2>&1 &

    echo $! > "tmp/worker_${GPU_IDX}.pid"

    if [ -z "$WORKER_ADDRS" ]; then
        WORKER_ADDRS="localhost:$WORKER_PORT"
    else
        WORKER_ADDRS="$WORKER_ADDRS,localhost:$WORKER_PORT"
    fi

    GPU_IDX=$((GPU_IDX + 1))
done

echo ""
echo "Waiting for Workers to load models (~30-90s)..."

sleep 5
for i in $(seq 0 $((NUM_GPUS - 1))); do
    WORKER_PORT=$((WORKER_BASE_PORT + i))
    MAX_RETRIES=300
    RETRY=0

    while [ $RETRY -lt $MAX_RETRIES ]; do
        if curl -sf "http://localhost:$WORKER_PORT/health" 2>/dev/null \
            | python -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('model_loaded') else 1)" 2>/dev/null; then
            echo "[Worker $i] Ready (port $WORKER_PORT)"
            break
        fi
        RETRY=$((RETRY + 1))
        sleep 2
    done

    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo "[Worker $i] FAILED to start! Log:"
        tail -50 "tmp/worker_${i}.log" 2>/dev/null || true
        exit 1
    fi
done

# ============ Start Gateway ============

echo ""
echo "[Gateway] Starting on port $GATEWAY_PORT..."

python gateway.py \
    --port $GATEWAY_PORT \
    --workers "$WORKER_ADDRS" \
    $GATEWAY_EXTRA_ARGS \
    > "tmp/gateway.log" 2>&1 &

echo $! > "tmp/gateway.pid"

sleep 2

CURL_FLAGS=""
if [ "$GATEWAY_PROTO" = "https" ]; then
    CURL_FLAGS="-k"
fi

if curl -sf $CURL_FLAGS "${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/health" 2>/dev/null \
    | python -c "import sys,json; d=json.load(sys.stdin); exit(0)" 2>/dev/null; then
    echo "[Gateway] Ready"
else
    echo "[Gateway] May still be starting. Check tmp/gateway.log"
fi

echo ""
echo "=================================================="
echo "  Service is running!"
echo "  Chat Demo:  ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT"
echo "  Admin:      ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/admin"
echo "  API Docs:   ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/docs"
echo "  Workers:    $WORKER_ADDRS"
echo "=================================================="

# Keep container alive and forward signals
wait
