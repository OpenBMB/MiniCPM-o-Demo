FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.8.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG DEBIAN_FRONTEND=noninteractive

# ============ System dependencies ============

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        ffmpeg \
        openssl \
        curl \
        git \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ============ PyTorch (CUDA 12.8) ============

RUN pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"

# ============ Python dependencies ============

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============ Project files ============

COPY . .
RUN chmod +x docker-entrypoint.sh

# Ensure config.json exists (user should mount their own)
RUN if [ ! -f config.json ]; then cp config.example.json config.json; fi

# ============ Runtime directories ============

RUN mkdir -p tmp data torch_compile_cache

# ============ Environment ============

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TORCHINDUCTOR_CACHE_DIR=/app/torch_compile_cache

VOLUME /workspace
EXPOSE 8006

ENTRYPOINT ["./docker-entrypoint.sh"]
