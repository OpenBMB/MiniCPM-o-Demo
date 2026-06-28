#!/bin/bash
# MiniCPMO45 Service One-Click Environment Installation Script
#
# Usage:
#   cd minicpmo45_service
#   bash install.sh
#
# Features:
#   1. Create a Python 3.10 virtual environment
#   2. Install PyTorch + core dependencies
#   3. Attempt to install Flash Attention 2 (auto-skip on failure, fallback to SDPA)
#   4. Verify installation results
#
# Environment Variables (optional):
#   PYTHON=python3.11        Specify Python interpreter (default: python3.10)
#   SKIP_FLASH_ATTN=1        Skip Flash Attention installation
#   MAX_JOBS=8               Flash Attention compilation parallelism (default: nproc)

set -e  # Exit on error (flash-attn section handled separately)

# ============ Configuration ============

VENV_DIR=".venv/base"
PIP="${VENV_DIR}/bin/pip"
PYTHON_BIN="${VENV_DIR}/bin/python"
PYTHON="${PYTHON:-python3.10}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 8)}"
FLASH_ATTN_VERSION=">=2.7.1,<=2.8.2"  # Officially recommended version range

# ============ Colored Output ============

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============ Step 1: Create Virtual Environment ============

info "Step 1/4: Creating virtual environment (${VENV_DIR})"

if [ -d "${VENV_DIR}" ]; then
    warn "Virtual environment already exists: ${VENV_DIR}, skipping creation"
else
    if ! command -v "${PYTHON}" &> /dev/null; then
        error "${PYTHON} not found. Please install Python 3.10+ or specify the path via PYTHON=python3.x"
        exit 1
    fi

    PYTHON_VERSION=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "Using Python ${PYTHON_VERSION} (${PYTHON})"

    "${PYTHON}" -m venv "${VENV_DIR}"
    info "Virtual environment created successfully"
fi

${PIP} install --upgrade pip -q

# ============ Step 2: Install PyTorch ============

info "Step 2/4: Installing PyTorch + torchaudio"

# Check if already installed (skip redundant installation)
if ${PYTHON_BIN} -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.8"; then
    TORCH_VER=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)")
    CUDA_VER=$(${PYTHON_BIN} -c "import torch; print(torch.version.cuda)")
    info "PyTorch already installed: ${TORCH_VER} (CUDA ${CUDA_VER}), skipping"
else
    ${PIP} install "torch==2.8.0" "torchaudio==2.8.0"
    TORCH_VER=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)")
    CUDA_VER=$(${PYTHON_BIN} -c "import torch; print(torch.version.cuda)")
    info "PyTorch installed successfully: ${TORCH_VER} (CUDA ${CUDA_VER})"
fi

# ============ Step 3: Install Core Dependencies ============

info "Step 3/4: Installing core dependencies (requirements.txt)"
${PIP} install -r requirements.txt
info "Core dependencies installed successfully"

# ============ Step 4: Install Flash Attention 2 (Not Recommended) ============

# info "Step 4/4: Installing Flash Attention 2 (optional, auto-skip on failure)"

# if [ "${SKIP_FLASH_ATTN}" = "1" ]; then
#     warn "SKIP_FLASH_ATTN=1, skipping Flash Attention installation"
#     warn "Inference will use PyTorch SDPA (approximately 5-15% slower)"
# else
#     # Check if already installed
#     if ${PYTHON_BIN} -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null; then
#         FA_VER=$(${PYTHON_BIN} -c "import flash_attn; print(flash_attn.__version__)")
#         info "Flash Attention already installed: ${FA_VER}, skipping"
#     else
#         info "Attempting to install flash-attn${FLASH_ATTN_VERSION} (MAX_JOBS=${MAX_JOBS})..."
#         info "This may take several minutes (compiling CUDA kernels)..."

#         set +e  # Temporarily disable errexit to allow failure
#         MAX_JOBS=${MAX_JOBS} ${PIP} install "flash-attn${FLASH_ATTN_VERSION}" --no-build-isolation 2>&1
#         FLASH_EXIT_CODE=$?
#         set -e  # Restore errexit

#         if [ ${FLASH_EXIT_CODE} -eq 0 ]; then
#             FA_VER=$(${PYTHON_BIN} -c "import flash_attn; print(flash_attn.__version__)")
#             info "Flash Attention installed successfully: ${FA_VER}"
#         else
#             warn "=========================================="
#             warn "Flash Attention installation failed (exit code: ${FLASH_EXIT_CODE})"
#             warn "This does not affect service operation — inference will automatically use PyTorch SDPA"
#             warn "Performance difference: SDPA is approximately 5-15% slower than Flash Attention"
#             warn ""
#             warn "Common causes:"
#             warn "  - CUDA toolkit version mismatch with PyTorch"
#             warn "  - Unsupported GPU architecture (requires SM80+, e.g. A100/H100)"
#             warn "  - Missing compilation toolchain (gcc/g++/nvcc)"
#             warn ""
#             warn "To retry manually:"
#             warn "  MAX_JOBS=${MAX_JOBS} ${PIP} install \"flash-attn${FLASH_ATTN_VERSION}\" --no-build-isolation"
#             warn "=========================================="
#         fi
#     fi
# fi

# ============ Installation Summary ============

echo ""
echo "============================================"
info "Installation complete! Environment summary:"
echo "============================================"

${PYTHON_BIN} -c "
import torch
print(f'  Python:       {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda}')
print(f'  GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

try:
    import flash_attn
    print(f'  Flash Attn:   {flash_attn.__version__} ✓')
    attn_backend = 'flash_attention_2'
except ImportError:
    print(f'  Flash Attn:   Not installed (will use SDPA)')
    attn_backend = 'sdpa'

import transformers
print(f'  Transformers: {transformers.__version__}')
print()
print(f'  Attention Backend: {attn_backend}')
"

echo ""
info "Next steps:"
echo "  1. Configure model path:"
echo "     cp config.example.json config.json"
echo "     # Edit config.json and set model.model_path"
echo ""
echo "  2. Start the service:"
echo "     bash start_all.sh"
echo "============================================"
