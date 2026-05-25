"""RunPod job contract validation and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast
import base64
import binascii
import tempfile
import time
import uuid

from eai_eio_runpod_sulphur_worker.config import WorkerConfig, load_config
from eai_eio_runpod_sulphur_worker.output_store import publish_video

VideoMode = Literal["text_to_video", "image_to_video", "first_last_frame_to_video", "multi_keyframe_to_video"]
WORKER_CONTRACT_VERSION = "eai-eio-runpod-sulphur-v3"
MODEL_DIMENSION_MULTIPLE = 32
MIN_MODEL_DIMENSION = 64


class WorkerOutput(TypedDict, total=False):
    video_url: str
    video_base64: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ImagePayload:
    filename: str
    mime_type: str
    base64_data: str


@dataclass(frozen=True)
class WorkerInput:
    prompt: str
    model_id: str
    gpu_type_id: str | None
    duration_seconds: float
    fps: int
    width: int
    height: int
    requested_width: int
    requested_height: int
    mode: VideoMode
    contract_version: str | None = None
    seed: int | None = None
    image: ImagePayload | None = None
    first_frame: ImagePayload | None = None
    middle_frame: ImagePayload | None = None
    last_frame: ImagePayload | None = None


class VideoGenerator(Protocol):
    def generate(
        self,
        request: WorkerInput,
        image_path: Path | None,
        output_path: Path,
        *,
        first_frame_path: Path | None = None,
        middle_frame_path: Path | None = None,
        last_frame_path: Path | None = None,
    ) -> dict[str, Any]:
        ...


def handle_job(job: dict[str, Any], generator: VideoGenerator | None = None, config: WorkerConfig | None = None) -> WorkerOutput:
    started_at = time.perf_counter()
    resolved_config = config or load_config()
    request = parse_worker_input(job.get("input"), resolved_config)
    resolved_config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_config.output_dir / f"eai_eio_runpod_{uuid.uuid4().hex}.mp4"

    with tempfile.TemporaryDirectory(prefix="eai-eio-input-") as temp_dir:
        image_path = write_image_payload(request.image, Path(temp_dir), "image") if request.image else None
        first_frame_path = write_image_payload(request.first_frame, Path(temp_dir), "first-frame") if request.first_frame else None
        middle_frame_path = write_image_payload(request.middle_frame, Path(temp_dir), "middle-frame") if request.middle_frame else None
        last_frame_path = write_image_payload(request.last_frame, Path(temp_dir), "last-frame") if request.last_frame else None
        resolved_generator = generator or _load_default_generator()
        generation_metadata = resolved_generator.generate(
            request,
            image_path,
            output_path,
            first_frame_path=first_frame_path,
            middle_frame_path=middle_frame_path,
            last_frame_path=last_frame_path,
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Video generator finished without writing a non-empty mp4.")

    published = publish_video(output_path, config=resolved_config)
    return {
        **published,
        "metadata": {
            **generation_metadata,
            "model_id": request.model_id,
            "gpu_type_id": request.gpu_type_id,
            "mode": request.mode,
            "duration_seconds": request.duration_seconds,
            "fps": request.fps,
            "width": request.width,
            "height": request.height,
            "requested_width": request.requested_width,
            "requested_height": request.requested_height,
            "dimension_adjusted": request.width != request.requested_width or request.height != request.requested_height,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "output_mode": "video_url" if "video_url" in published else "video_base64",
            "first_frame_control": request.first_frame is not None,
            "middle_frame_control": request.middle_frame is not None,
            "last_frame_control": request.last_frame is not None,
            "multi_keyframe_control": request.first_frame is not None and request.middle_frame is not None and request.last_frame is not None,
            "worker_contract_version": WORKER_CONTRACT_VERSION,
            "expected_contract_version": request.contract_version,
            "contract_version_match": request.contract_version in (None, WORKER_CONTRACT_VERSION),
        },
    }


def parse_worker_input(raw_input: object, config: WorkerConfig) -> WorkerInput:
    if not isinstance(raw_input, dict):
        raise ValueError("RunPod job input must be an object.")
    payload = cast(dict[str, Any], raw_input)

    prompt = _required_string(payload, "prompt")
    model_id = _optional_string(payload, "model_id") or config.default_model_id
    gpu_type_id = _optional_string(payload, "gpu_type_id")
    duration_seconds = _float_range(payload.get("duration_seconds", 6), "duration_seconds", minimum=1, maximum=120)
    fps = int(_float_range(payload.get("fps", 24), "fps", minimum=1, maximum=120))
    input_width = int(_float_range(payload.get("width", 1280), "width", minimum=64, maximum=8192))
    input_height = int(_float_range(payload.get("height", 720), "height", minimum=64, maximum=8192))
    requested_width = int(_float_range(payload.get("requested_width", input_width), "requested_width", minimum=64, maximum=8192))
    requested_height = int(_float_range(payload.get("requested_height", input_height), "requested_height", minimum=64, maximum=8192))
    width, height = normalize_model_dimensions(input_width, input_height)
    mode_value = _optional_string(payload, "mode") or "text_to_video"
    contract_version = _optional_string(payload, "contract_version")
    if mode_value not in ("text_to_video", "image_to_video", "first_last_frame_to_video", "multi_keyframe_to_video"):
        raise ValueError("mode must be text_to_video, image_to_video, first_last_frame_to_video, or multi_keyframe_to_video.")

    seed_value = payload.get("seed")
    seed = None if seed_value is None else int(_float_range(seed_value, "seed", minimum=0, maximum=2_147_483_647))
    image = parse_image_payload(payload.get("image")) if payload.get("image") is not None else None
    first_frame = parse_image_payload(payload.get("first_frame")) if payload.get("first_frame") is not None else None
    middle_frame = parse_image_payload(payload.get("middle_frame")) if payload.get("middle_frame") is not None else None
    last_frame = parse_image_payload(payload.get("last_frame")) if payload.get("last_frame") is not None else None
    mode = cast(VideoMode, mode_value)

    if mode == "image_to_video" and image is None:
        raise ValueError("image_to_video mode requires an image payload.")
    if mode == "first_last_frame_to_video" and (first_frame is None or last_frame is None):
        raise ValueError("first_last_frame_to_video mode requires first_frame and last_frame payloads.")
    if mode == "multi_keyframe_to_video" and (first_frame is None or middle_frame is None or last_frame is None):
        raise ValueError("multi_keyframe_to_video mode requires first_frame, middle_frame, and last_frame payloads.")
    if mode == "text_to_video" and image is not None:
        mode = "image_to_video"
    if first_frame is not None and middle_frame is not None and last_frame is not None:
        mode = "multi_keyframe_to_video"
    elif first_frame is not None and last_frame is not None:
        mode = "first_last_frame_to_video"

    return WorkerInput(
        prompt=prompt,
        model_id=model_id,
        gpu_type_id=gpu_type_id,
        duration_seconds=duration_seconds,
        fps=fps,
        width=width,
        height=height,
        requested_width=requested_width,
        requested_height=requested_height,
        mode=mode,
        contract_version=contract_version,
        seed=seed,
        image=image,
        first_frame=first_frame,
        middle_frame=middle_frame,
        last_frame=last_frame,
    )


def parse_image_payload(raw_image: object) -> ImagePayload:
    if not isinstance(raw_image, dict):
        raise ValueError("image must be an object.")
    image_payload = cast(dict[str, Any], raw_image)
    filename = _optional_string(image_payload, "filename") or "reference.png"
    mime_type = _optional_string(image_payload, "mime_type") or "application/octet-stream"
    base64_data = _required_string(image_payload, "base64")
    try:
        base64.b64decode(base64_data, validate=True)
    except binascii.Error as exc:
        raise ValueError("image.base64 is not valid base64.") from exc
    return ImagePayload(filename=Path(filename).name, mime_type=mime_type, base64_data=base64_data)


def write_image_payload(image: ImagePayload, directory: Path, prefix: str = "image") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{prefix}-{image.filename}"
    output_path.write_bytes(base64.b64decode(image.base64_data))
    return output_path


def normalize_model_dimensions(width: int, height: int) -> tuple[int, int]:
    return normalize_model_dimension(width), normalize_model_dimension(height)


def normalize_model_dimension(value: int) -> int:
    aligned = (value // MODEL_DIMENSION_MULTIPLE) * MODEL_DIMENSION_MULTIPLE
    return max(MIN_MODEL_DIMENSION, aligned)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required.")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value.strip() or None


def _float_range(value: object, key: str, *, minimum: float, maximum: float) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number.")
    parsed = float(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}.")
    return parsed


def _load_default_generator() -> VideoGenerator:
    from eai_eio_runpod_sulphur_worker.ltx_diffusers import LtxDiffusersGenerator

    return LtxDiffusersGenerator()
