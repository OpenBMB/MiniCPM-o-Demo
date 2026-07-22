#!/usr/bin/env bash
set -euo pipefail

# Real-GPU regression for the cross-Unit stream lifecycle:
# 1) replay the recorded Unit-25 session that previously triggered deferred
#    close and nested-stream RuntimeErrors;
# 2) require a later checkpoint to recover to available;
# 3) run stateless live→resume trace equivalence.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUT="${OUT:-/user/sunweiyue/lib/swy-dev/tmp/fc_stream_lifecycle_gpu_validation}"
PYTHON="${PYTHON:-/user/sunweiyue/lib/swy-dev/.venv/minicpm_o5_demo/bin/python}"
SDK_SRC="${SDK_SRC:-/user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_sdk/src}"
RECORDED_SESSION_DIR="${RECORDED_SESSION_DIR:-${PROJECT_DIR}/data/sessions/sess_fc820332f0c7}"

mkdir -p "${OUT}"
cd "${PROJECT_DIR}"

ENABLE_FRP=0 \
LOG_DIR="${OUT}/service_logs" \
FC_DUPLEX_TRACE_DIR="${OUT}/service_logs/fc_traces" \
bash scripts/start_o45_fc_api_cctl_service.sh >"${OUT}/service.log" 2>&1 &
SERVICE_PID=$!

cleanup() {
    kill "${SERVICE_PID}" 2>/dev/null || true
    wait "${SERVICE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 600); do
    if curl -skf https://127.0.0.1:8009/health >/dev/null; then
        break
    fi
    if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
        echo "[validation] service exited before ready" >&2
        tail -n 200 "${OUT}/service.log" >&2
        exit 2
    fi
    sleep 2
done
curl -skf https://127.0.0.1:8009/health >/dev/null

PYTHONPATH=".:${SDK_SRC}" "${PYTHON}" scripts/fc_duplex_replay_session.py \
    --session-dir "${RECORDED_SESSION_DIR}" \
    --base-url https://127.0.0.1:8009 \
    --output "${OUT}/recorded_replay.jsonl" \
    --timing fast \
    --between-idle-rounds 2 \
    --close-idle-rounds 8 \
    --insecure \
    --skip-recorded-tool-results \
    >"${OUT}/recorded_replay.log" 2>&1

"${PYTHON}" - "${OUT}/recorded_replay.jsonl" <<'PY'
from pathlib import Path
import json
import sys

events = [
    json.loads(line)["frame"]
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
checkpoints = [
    event for event in events
    if event.get("type") == "response.unit.committed"
]
deferred_positions = [
    index for index, event in enumerate(checkpoints)
    if event.get("resume", {}).get("reason") == "unsupported_deferred_close"
]
if not deferred_positions:
    raise RuntimeError("recorded replay did not exercise deferred close")
first_deferred = deferred_positions[0]
if not any(
    event.get("resume", {}).get("status") == "available"
    for event in checkpoints[first_deferred + 1:]
):
    raise RuntimeError("checkpoint never recovered after deferred close")
if any(
    event.get("type") == "session.closed"
    and event.get("reason") == "backend_error"
    for event in events
):
    raise RuntimeError("recorded replay hit backend_error")
if "\ufffd" in json.dumps(events, ensure_ascii=False):
    raise RuntimeError("public API leaked U+FFFD")
print(json.dumps({
    "recorded_replay_ok": True,
    "checkpoint_count": len(checkpoints),
    "first_deferred_unit": checkpoints[first_deferred].get("unit_index"),
    "later_available": True,
}, ensure_ascii=False))
PY

PYTHONPATH=".:${SDK_SRC}" "${PYTHON}" scripts/fc_duplex_resume_smoke.py \
    --base-url https://127.0.0.1:8009 \
    --insecure \
    --max-units 8 \
    --trace-dir "${OUT}/service_logs/fc_traces" \
    | tee "${OUT}/resume_smoke.log"

echo "FC_STREAM_LIFECYCLE_GPU_VALIDATION_PASS"
