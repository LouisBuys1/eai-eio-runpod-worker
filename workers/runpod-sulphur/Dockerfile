ARG EAI_EIO_WORKER_BASE_IMAGE=ghcr.io/louisbuys1/eai-eio-runpod-sulphur:0b738424d1378524784caa1f5be0271d393a296b
FROM ${EAI_EIO_WORKER_BASE_IMAGE}

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src

RUN python3 -m pip install --no-deps --force-reinstall .

CMD ["python3", "-m", "eai_eio_runpod_sulphur_worker.worker"]
