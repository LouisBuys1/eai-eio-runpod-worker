"""LTX/Sulphur generation through Diffusers."""

from __future__ import annotations

from inspect import signature
from pathlib import Path
from typing import Any
import os
import shutil

from eai_eio_runpod_sulphur_worker.config import load_config
from eai_eio_runpod_sulphur_worker.contract import ProgressCallback, WorkerInput

MIN_MODEL_CACHE_FREE_BYTES = 80 * 1024 * 1024 * 1024


class LtxDiffusersGenerator:
    def __init__(self) -> None:
        self._pipe_by_model: dict[tuple[str, str], Any] = {}

    def generate(
        self,
        request: WorkerInput,
        image_path: Path | None,
        output_path: Path,
        *,
        first_frame_path: Path | None = None,
        middle_frame_path: Path | None = None,
        last_frame_path: Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        pipeline_kind = self._pipeline_kind(image_path, first_frame_path, middle_frame_path, last_frame_path)
        self._emit_progress(progress, "loading_pipeline", {"pipeline_kind": pipeline_kind, "model_id": request.model_id})
        pipe, load_metadata = self._load_pipeline(request.model_id, pipeline_kind)
        self._emit_progress(progress, "building_request", {"pipeline_kind": pipeline_kind})
        call_kwargs = self._build_call_kwargs(request, image_path, first_frame_path, middle_frame_path, last_frame_path, pipe=pipe)
        self._emit_progress(progress, "running_inference", {"pipeline_kind": pipeline_kind, "frame_count": call_kwargs.get("num_frames")})
        result = pipe(**call_kwargs)
        frames = self._extract_frames(result)
        self._emit_progress(progress, "exporting_video", {"frame_count": len(frames) if hasattr(frames, "__len__") else None})
        self._export_video(frames, output_path, fps=request.fps)
        return {
            "backend": "diffusers",
            **load_metadata,
            "call_kwargs": sorted(call_kwargs.keys()),
            "pipeline_kind": pipeline_kind,
            "frame_count": len(frames) if hasattr(frames, "__len__") else None,
            "first_frame_control": first_frame_path is not None,
            "middle_frame_control": middle_frame_path is not None,
            "last_frame_control": last_frame_path is not None,
            "multi_keyframe_control": first_frame_path is not None and middle_frame_path is not None and last_frame_path is not None,
        }

    def _load_pipeline(self, model_id: str, pipeline_kind: str) -> tuple[Any, dict[str, Any]]:
        cache_key = (model_id, pipeline_kind)
        cached = self._pipe_by_model.get(cache_key)
        if cached is not None:
            return cached, {"model_cache_hit": True, "model_id": model_id}

        try:
            import torch
            _patch_torch_dynamo_compat()
            try:
                from diffusers import LTX2ImageToVideoPipeline
            except Exception:
                try:
                    from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import LTX2ImageToVideoPipeline
                except Exception:
                    LTX2ImageToVideoPipeline = None
            try:
                from diffusers import LTX2Pipeline
            except Exception:
                try:
                    from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
                except Exception:
                    LTX2Pipeline = None
            try:
                from diffusers import LTX2ConditionPipeline
            except Exception:
                LTX2ConditionPipeline = None
        except Exception as exc:
            raise RuntimeError(
                "Diffusers/Torch dependencies are not installed. Build the RunPod Docker image before running real generation."
            ) from exc

        config = load_config()
        model_cache_dir = self._select_model_cache_dir(config.model_cache_dir, config.persistent_model_cache_dir)
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(model_cache_dir / "hf-home"))
        os.environ.setdefault("HF_HUB_CACHE", str(model_cache_dir))
        os.environ.setdefault("HF_XET_CACHE", str(model_cache_dir / "xet"))
        local_cached_snapshot = self._resolve_cached_snapshot(model_id, model_cache_dir)
        runpod_cached_snapshot = self._resolve_cached_snapshot(model_id, config.runpod_cached_model_hub)
        model_ref = local_cached_snapshot or runpod_cached_snapshot
        cache_free_bytes = self._free_bytes(model_cache_dir)
        if model_ref is None:
            self._ensure_model_cache_has_room(
                model_cache_dir,
                cache_free_bytes,
                has_existing_artifacts=self._has_model_cache_artifacts(model_cache_dir, model_id),
            )
        dtype = torch.bfloat16 if os.environ.get("EAI_EIO_DTYPE", "bfloat16") == "bfloat16" else torch.float16
        pipeline_class = self._pipeline_class_for_model(
            model_id,
            pipeline_kind,
            LTX2Pipeline,
            LTX2ImageToVideoPipeline,
            LTX2ConditionPipeline,
        )
        if pipeline_class is None:
            from diffusers import DiffusionPipeline

            pipeline_class = DiffusionPipeline
        from_pretrained_kwargs: dict[str, Any] = {
            "cache_dir": str(model_cache_dir),
        }
        device_map = os.environ.get("EAI_EIO_DEVICE_MAP", "").strip()
        if device_map:
            from_pretrained_kwargs["device_map"] = device_map
        try:
            pipe = pipeline_class.from_pretrained(str(model_ref or model_id), torch_dtype=dtype, **from_pretrained_kwargs)
        except TypeError:
            pipe = pipeline_class.from_pretrained(str(model_ref or model_id), dtype=dtype, **from_pretrained_kwargs)

        device = os.environ.get("EAI_EIO_DEVICE", "cuda")
        vae = getattr(pipe, "vae", None)
        if os.environ.get("EAI_EIO_ENABLE_VAE_TILING", "1") != "0" and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        if not device_map:
            if os.environ.get("EAI_EIO_ENABLE_CPU_OFFLOAD", "1") != "0" and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload(device=device)
            elif hasattr(pipe, "to"):
                pipe.to(device)
        self._pipe_by_model[cache_key] = pipe
        return pipe, {
            "model_cache_hit": False,
            "model_id": model_id,
            "model_ref": str(model_ref or model_id),
            "model_cache_dir": str(model_cache_dir),
            "model_cache_free_bytes_at_start": cache_free_bytes,
            "local_cached_model_hit": local_cached_snapshot is not None,
            "runpod_cached_model_hit": runpod_cached_snapshot is not None,
        }

    def _select_model_cache_dir(self, requested_cache_dir: Path, persistent_cache_dir: Path) -> Path:
        if str(requested_cache_dir).startswith("/runpod-volume"):
            return requested_cache_dir
        runpod_volume = Path("/runpod-volume")
        if runpod_volume.exists():
            return persistent_cache_dir
        return requested_cache_dir

    def _resolve_cached_snapshot(self, model_id: str, cached_model_hub: Path) -> Path | None:
        if "/" not in model_id or Path(model_id).exists():
            return None
        snapshots_dir = cached_model_hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
        if not snapshots_dir.exists():
            return None
        ref_path = snapshots_dir.parent / "refs" / "main"
        if ref_path.exists():
            revision = ref_path.read_text(encoding="utf-8").strip()
            ref_snapshot = snapshots_dir / revision
            if self._looks_like_diffusers_snapshot(ref_snapshot):
                return ref_snapshot
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        valid_snapshots = [path for path in snapshots if self._looks_like_diffusers_snapshot(path)]
        if not valid_snapshots:
            return None
        return max(valid_snapshots, key=lambda path: path.stat().st_mtime)

    def _looks_like_diffusers_snapshot(self, path: Path) -> bool:
        if not path.is_dir() or not (path / "model_index.json").exists():
            return False

        component_config_groups = [
            (path / "transformer" / "config.json",),
            (path / "vae" / "config.json",),
            (path / "scheduler" / "scheduler_config.json", path / "scheduler" / "config.json"),
            (path / "text_encoder" / "config.json",),
            (path / "tokenizer" / "tokenizer_config.json", path / "tokenizer" / "vocab.json"),
        ]
        return all(any(candidate.exists() for candidate in group) for group in component_config_groups)

    def _has_model_cache_artifacts(self, model_cache_dir: Path, model_id: str) -> bool:
        if "/" not in model_id:
            return False
        return (model_cache_dir / f"models--{model_id.replace('/', '--')}").exists()

    def _free_bytes(self, path: Path) -> int | None:
        try:
            return shutil.disk_usage(str(path)).free
        except Exception:
            return None

    def _ensure_model_cache_has_room(self, model_cache_dir: Path, free_bytes: int | None, *, has_existing_artifacts: bool) -> None:
        default_min_free_gb = str(MIN_MODEL_CACHE_FREE_BYTES / (1024 ** 3))
        min_free_bytes = int(
            float(os.environ.get("EAI_EIO_MIN_MODEL_CACHE_FREE_GB", default_min_free_gb))
            * 1024
            * 1024
            * 1024
        )
        if free_bytes is None or free_bytes >= min_free_bytes:
            return
        if has_existing_artifacts:
            return
        raise RuntimeError(
            f"Model cache path {model_cache_dir} has {free_bytes / (1024 ** 3):.1f} GiB free; "
            f"at least {min_free_bytes / (1024 ** 3):.0f} GiB is required before downloading LTX-2.3."
        )

    def _build_call_kwargs(
        self,
        request: WorkerInput,
        image_path: Path | None,
        first_frame_path: Path | None,
        middle_frame_path: Path | None,
        last_frame_path: Path | None,
        *,
        pipe: Any,
    ) -> dict[str, Any]:
        num_frames = _compute_frame_count(request.duration_seconds, request.fps)
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "num_frames": num_frames,
            "frame_rate": float(request.fps),
            "fps": request.fps,
        }
        self._apply_model_call_defaults(kwargs, request.model_id)
        if request.seed is not None:
            kwargs["generator"] = self._make_torch_generator(request.seed)
        if first_frame_path is not None or middle_frame_path is not None or last_frame_path is not None:
            if first_frame_path is None or last_frame_path is None:
                raise RuntimeError("Strict first/last-frame generation requires both first_frame and last_frame.")
            kwargs.update(self._build_condition_frame_kwargs(pipe, first_frame_path, middle_frame_path, last_frame_path, num_frames))
        elif image_path is not None:
            from diffusers.utils import load_image

            kwargs["image"] = load_image(str(image_path))
        return self._filter_supported_kwargs(pipe, kwargs)

    def _emit_progress(self, progress: ProgressCallback | None, phase: str, metadata: dict[str, Any]) -> None:
        if progress is None:
            return
        progress(phase, metadata)

    def _build_condition_frame_kwargs(
        self,
        pipe: Any,
        first_frame_path: Path,
        middle_frame_path: Path | None,
        last_frame_path: Path,
        num_frames: int,
    ) -> dict[str, Any]:
        from diffusers.utils import load_image

        condition_frames: list[tuple[Any, int, float]] = [
            (load_image(str(first_frame_path)), 0, 1.0),
        ]
        if middle_frame_path is not None:
            condition_frames.append((load_image(str(middle_frame_path)), _middle_frame_index(num_frames), 0.92))
        condition_frames.append((load_image(str(last_frame_path)), -1, 1.0))

        supported = self._supported_call_kwargs(pipe)

        if "conditions" in supported or "*" in supported:
            try:
                from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

                return {
                    "conditions": [
                        LTX2VideoCondition(frames=image, index=index, strength=strength)
                        for image, index, strength in condition_frames
                    ],
                }
            except Exception:
                pass

        if ("image" in supported or "*" in supported) and ("frame_index" in supported or "*" in supported):
            return {
                "image": [image for image, _, _ in condition_frames],
                "frame_index": [index for _, index, _ in condition_frames],
                "strength": [strength for _, _, strength in condition_frames],
            }

        raise RuntimeError("Selected Diffusers pipeline does not expose multi-keyframe conditioning controls.")

    def _filter_supported_kwargs(self, pipe: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        supported = self._supported_call_kwargs(pipe)
        if "*" in supported:
            return kwargs
        return {key: value for key, value in kwargs.items() if key in supported}

    def _supported_call_kwargs(self, pipe: Any) -> set[str]:
        if pipe is None or not hasattr(pipe, "__call__"):
            return {"*"}
        call_signature = signature(pipe.__call__)
        if any(parameter.kind == parameter.VAR_KEYWORD for parameter in call_signature.parameters.values()):
            return {"*"}
        return set(call_signature.parameters)

    def _make_torch_generator(self, seed: int) -> Any:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Torch is required for seeded generation.") from exc
        return torch.Generator(device=os.environ.get("EAI_EIO_DEVICE", "cuda")).manual_seed(seed)

    def _extract_frames(self, result: Any) -> Any:
        if isinstance(result, tuple) and result:
            result = result[0]
        shape = getattr(result, "shape", None)
        if shape is not None and len(shape) == 5:
            return result[0]
        if isinstance(result, list) and result:
            return result[0] if isinstance(result[0], list) else result
        frames = getattr(result, "frames", None)
        if isinstance(frames, list) and frames:
            return frames[0]
        if isinstance(result, dict):
            frames_value = result.get("frames")
            if isinstance(frames_value, list) and frames_value:
                return frames_value[0]
        raise RuntimeError("Diffusers pipeline did not return frames.")

    def _export_video(self, frames: Any, output_path: Path, *, fps: int) -> None:
        from diffusers.utils import export_to_video

        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(output_path), fps=fps)

    def _pipeline_kind(
        self,
        image_path: Path | None,
        first_frame_path: Path | None,
        middle_frame_path: Path | None,
        last_frame_path: Path | None,
    ) -> str:
        if first_frame_path is not None or middle_frame_path is not None or last_frame_path is not None:
            return "condition"
        if image_path is not None:
            return "image"
        return "text"

    def _pipeline_class_for_model(
        self,
        model_id: str,
        pipeline_kind: str,
        ltx2_pipeline: Any,
        ltx2_image_to_video_pipeline: Any,
        ltx2_condition_pipeline: Any,
    ) -> Any:
        if "ltx-2" not in model_id.lower():
            return None
        if pipeline_kind == "condition":
            return ltx2_condition_pipeline or ltx2_pipeline
        if pipeline_kind == "image":
            return ltx2_image_to_video_pipeline
        return ltx2_pipeline

    def _apply_model_call_defaults(self, kwargs: dict[str, Any], model_id: str) -> None:
        normalized_model_id = model_id.lower()
        if "ltx-2" not in normalized_model_id:
            return

        kwargs.setdefault("output_type", os.environ.get("EAI_EIO_OUTPUT_TYPE", "np"))
        kwargs.setdefault("return_dict", False)

        if "distilled" not in normalized_model_id:
            return

        kwargs.setdefault("num_inference_steps", int(os.environ.get("EAI_EIO_NUM_INFERENCE_STEPS", "8")))
        kwargs.setdefault("guidance_scale", float(os.environ.get("EAI_EIO_GUIDANCE_SCALE", "1.0")))
        try:
            from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES

            kwargs.setdefault("negative_prompt", os.environ.get("EAI_EIO_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT))
            kwargs.setdefault("sigmas", DISTILLED_SIGMA_VALUES)
        except Exception:
            pass


def _compute_frame_count(duration_seconds: float, fps: int) -> int:
    raw_frame_count = max(1, round(duration_seconds * fps))
    remainder = (raw_frame_count - 1) % 8
    return raw_frame_count if remainder == 0 else raw_frame_count + (8 - remainder)


def _middle_frame_index(num_frames: int) -> int:
    latent_frame_count = _latent_frame_count(num_frames)
    if latent_frame_count <= 1:
        return 0
    if latent_frame_count <= 2:
        return 1
    return max(1, min(latent_frame_count - 2, latent_frame_count // 2))


def _latent_frame_count(num_frames: int) -> int:
    return max(1, ((max(1, num_frames) - 1) // 8) + 1)


def _identity_decorator(fn: Any = None, *args: Any, **kwargs: Any) -> Any:
    if fn is None:
        return lambda inner: inner
    return fn


def _patch_torch_dynamo_compat() -> None:
    try:
        import torch._dynamo as torch_dynamo

        for name in ("assume_constant_result", "mark_static_address"):
            if not hasattr(torch_dynamo, name):
                setattr(torch_dynamo, name, _identity_decorator)
    except Exception:
        pass
