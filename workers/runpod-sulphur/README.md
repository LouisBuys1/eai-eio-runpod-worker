# EAI-EIO RunPod Sulphur Worker

RunPod Serverless worker for EAI-EIO cloud video generation. This worker implements the desktop app contract in `docs/runpod-sulphur-worker.md` and keeps cloud Sulphur/LTX execution separate from the local Apple MLX provider.

## Runtime

- GPU: NVIDIA CUDA, recommended 48 GB+ VRAM for 1280×720 clips.
- Model: defaults to `diffusers/LTX-2.3-Distilled-Diffusers`; override per request or with `EAI_EIO_DEFAULT_MODEL_ID`.
- GPU target metadata: EAI-EIO sends `gpu_type_id` from the Settings dropdown for logging and cost estimates. Configure the RunPod endpoint itself with the matching GPU pool.
- Output: uploads to S3-compatible storage when configured, otherwise returns base64 video bytes for smoke tests.

## Environment

| Variable | Purpose |
| --- | --- |
| `EAI_EIO_DEFAULT_MODEL_ID` | Default Hugging Face model ID. |
| `EAI_EIO_MODEL_CACHE_DIR` | Optional model cache path. |
| `EAI_EIO_PERSISTENT_MODEL_CACHE_DIR` | Preferred cache path when `/runpod-volume` is mounted. |
| `EAI_EIO_RUNPOD_CACHED_MODEL_HUB` | RunPod cached-model Hugging Face hub path. |
| `EAI_EIO_OUTPUT_MODE` | `auto`, `s3`, or `base64`. Default: `auto`. |
| `S3_ENDPOINT_URL` | Optional S3/R2 endpoint URL. |
| `S3_BUCKET` | Bucket for generated videos. |
| `S3_PUBLIC_BASE_URL` | Optional public base URL for returned `video_url`. |
| `AWS_ACCESS_KEY_ID` | S3/R2 access key. |
| `AWS_SECRET_ACCESS_KEY` | S3/R2 secret key. |
| `AWS_REGION` | S3 region. Default: `auto`. |

## Local Smoke Test

```bash
python -m eai_eio_runpod_sulphur_worker.local_smoke --prompt "A lighthouse at dusk"
```

Without GPU dependencies, run contract tests:

```bash
python -m unittest discover workers/runpod-sulphur/tests
```

## Build

EAI-EIO uses `ghcr.io/louisbuys1/eai-eio-runpod-sulphur:latest` as the default advanced endpoint image. The repository workflow publishes it when worker files change.

```bash
cd workers/runpod-sulphur
docker build -t ghcr.io/louisbuys1/eai-eio-runpod-sulphur:latest .
```

RunPod Flash auto mode does not require this Docker image. For a permanent endpoint, push the image to a registry, then create a RunPod Serverless endpoint using that image. Paste the endpoint ID into EAI-EIO Settings → API Keys → RunPod Sulphur.
