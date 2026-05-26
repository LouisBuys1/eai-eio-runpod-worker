ARG EAI_EIO_WORKER_BASE_IMAGE=ghcr.io/louisbuys1/eai-eio-runpod-sulphur:c45d04d6c082d4343521906148f018090089a8b6
FROM ${EAI_EIO_WORKER_BASE_IMAGE}

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src

RUN python3 -m pip install --no-deps --force-reinstall .

CMD ["python3", "-m", "eai_eio_runpod_sulphur_worker.worker"]
