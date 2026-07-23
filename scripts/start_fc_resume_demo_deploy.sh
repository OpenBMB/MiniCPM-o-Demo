#!/usr/bin/env bash
set -euo pipefail

# Long-running Cybertron deploy entry for the public FC resumable API demo.
# It derives a one-proxy frpc config from SWY's authenticated local config
# without printing or copying the auth token into this repository.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-/user/sunweiyue/lib/swy-dev/tmp/fc_resume_demo_deploy}"
SOURCE_FRPC_CONFIG="${SOURCE_FRPC_CONFIG:-/user/sunweiyue/lib/swy-dev/frpc_manager/frpc.toml}"
FRPC_BIN="${FRPC_BIN:-/user/sunweiyue/lib/swy-dev/frpc_manager/frpc}"
REMOTE_PORT="${REMOTE_PORT:-7001}"
DEMO_FRPC_CONFIG="${DEMO_FRPC_CONFIG:-/tmp/fc_resume_demo_frpc.toml}"

for required_name in \
    CHECKPOINT_PROFILE_ID \
    MODEL_PATH \
    PT_PATH \
    FC_DUPLEX_NON_SPOKEN_SCHEDULING \
    FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING \
    FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING; do
    if [ -z "${!required_name:-}" ]; then
        echo "[deploy] missing Checkpoint Profile field: ${required_name}" >&2
        exit 1
    fi
done

if [ ! -f "${SOURCE_FRPC_CONFIG}" ]; then
    echo "[deploy] missing source frpc config: ${SOURCE_FRPC_CONFIG}" >&2
    exit 1
fi
if [ ! -x "${FRPC_BIN}" ]; then
    echo "[deploy] missing frpc binary: ${FRPC_BIN}" >&2
    exit 1
fi

echo "[deploy] checkpoint_profile_id=${CHECKPOINT_PROFILE_ID}"
echo "[deploy] non_spoken_scheduling=${FC_DUPLEX_NON_SPOKEN_SCHEDULING}"
echo "[deploy] non_spoken_budget_while_listening=${FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING}"
echo "[deploy] non_spoken_budget_while_speaking=${FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING}"

python - "${SOURCE_FRPC_CONFIG}" "${DEMO_FRPC_CONFIG}" "${REMOTE_PORT}" <<'PY'
from pathlib import Path
import sys

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
remote_port = int(sys.argv[3])
source = source_path.read_text(encoding="utf-8")
global_config = source.split("[[proxies]]", maxsplit=1)[0].rstrip()
proxy = f"""

[[proxies]]
name = "swy-fc-resume-demo"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8009
remotePort = {remote_port}
"""
target_path.write_text(global_config + proxy, encoding="utf-8")
PY

cd "${PROJECT_DIR}"
exec env \
    ENABLE_FRP=1 \
    LOG_DIR="${LOG_DIR}" \
    FC_DUPLEX_TRACE_DIR="${LOG_DIR}/fc_traces" \
    FRPC_BIN="${FRPC_BIN}" \
    FRPC_CONFIG="${DEMO_FRPC_CONFIG}" \
    bash scripts/start_o45_fc_api_cctl_service.sh
