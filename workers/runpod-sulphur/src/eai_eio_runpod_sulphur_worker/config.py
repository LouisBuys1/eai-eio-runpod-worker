"""Environment configuration for the RunPod worker."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_MODEL_ID = "diffusers/LTX-2.3-Distilled-Diffusers"


@dataclass(frozen=True)
class WorkerConfig:
    default_model_id: str
    model_cache_dir: Path
    output_dir: Path
    output_mode: str
    s3_endpoint_url: str
    s3_bucket: str
    s3_public_base_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str

    @property
    def s3_configured(self) -> bool:
        return bool(
            self.s3_bucket
            and self.aws_access_key_id
            and self.aws_secret_access_key
        )


def load_config() -> WorkerConfig:
    return WorkerConfig(
        default_model_id=os.environ.get("EAI_EIO_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID,
        model_cache_dir=Path(os.environ.get("EAI_EIO_MODEL_CACHE_DIR", "/workspace/models")).expanduser(),
        output_dir=Path(os.environ.get("EAI_EIO_OUTPUT_DIR", "/workspace/outputs")).expanduser(),
        output_mode=os.environ.get("EAI_EIO_OUTPUT_MODE", "auto").strip().lower() or "auto",
        s3_endpoint_url=os.environ.get("S3_ENDPOINT_URL", "").strip(),
        s3_bucket=os.environ.get("S3_BUCKET", "").strip(),
        s3_public_base_url=os.environ.get("S3_PUBLIC_BASE_URL", "").strip().rstrip("/"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "").strip(),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip(),
        aws_region=os.environ.get("AWS_REGION", "auto").strip() or "auto",
    )
