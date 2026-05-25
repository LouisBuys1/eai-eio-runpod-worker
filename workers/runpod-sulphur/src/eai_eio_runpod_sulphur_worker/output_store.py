"""Output publishing for generated videos."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict
import base64
import mimetypes
import uuid

from eai_eio_runpod_sulphur_worker.config import WorkerConfig


class PublishedVideo(TypedDict, total=False):
    video_url: str
    video_base64: str


def publish_video(video_path: Path, *, config: WorkerConfig) -> PublishedVideo:
    if config.output_mode not in ("auto", "s3", "base64"):
        raise ValueError("EAI_EIO_OUTPUT_MODE must be auto, s3, or base64.")

    if config.output_mode in ("auto", "s3") and config.s3_configured:
        return {"video_url": upload_video_to_s3(video_path, config=config)}

    if config.output_mode == "s3":
        raise RuntimeError("S3 output mode is selected but S3_BUCKET/AWS credentials are not configured.")

    return {"video_base64": base64.b64encode(video_path.read_bytes()).decode("ascii")}


def upload_video_to_s3(video_path: Path, *, config: WorkerConfig) -> str:
    import boto3

    key = f"eai-eio/runpod/{uuid.uuid4().hex}/{video_path.name}"
    client = boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint_url or None,
        region_name=config.aws_region,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
    )
    content_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    client.upload_file(
        str(video_path),
        config.s3_bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    if config.s3_public_base_url:
        return f"{config.s3_public_base_url}/{key}"
    if config.s3_endpoint_url:
        return f"{config.s3_endpoint_url.rstrip('/')}/{config.s3_bucket}/{key}"
    return f"https://{config.s3_bucket}.s3.{config.aws_region}.amazonaws.com/{key}"
