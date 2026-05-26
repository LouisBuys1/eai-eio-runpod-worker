FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EAI_EIO_OUTPUT_MODE=auto \
    EAI_EIO_MODEL_CACHE_DIR=/runpod-volume/eai-eio-models \
    EAI_EIO_PERSISTENT_MODEL_CACHE_DIR=/runpod-volume/eai-eio-models \
    EAI_EIO_RUNPOD_CACHED_MODEL_HUB=/runpod-volume/huggingface-cache/hub \
    HF_HOME=/runpod-volume/eai-eio-hf \
    HF_HUB_CACHE=/runpod-volume/eai-eio-models \
    HF_XET_CACHE=/runpod-volume/eai-eio-xet \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      git \
      python3 \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 \
    && python3 -m pip install .

CMD ["python3", "-m", "eai_eio_runpod_sulphur_worker.worker"]
