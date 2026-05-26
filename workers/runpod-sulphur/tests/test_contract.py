from __future__ import annotations

import base64
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eai_eio_runpod_sulphur_worker.config import WorkerConfig
from eai_eio_runpod_sulphur_worker.contract import WORKER_CONTRACT_VERSION, WorkerInput, handle_job, parse_worker_input


class FakeGenerator:
    def __init__(self) -> None:
        self.last_request: WorkerInput | None = None
        self.last_image_path: Path | None = None
        self.last_image_bytes: bytes | None = None
        self.last_first_frame_path: Path | None = None
        self.last_middle_frame_path: Path | None = None
        self.last_last_frame_path: Path | None = None
        self.last_first_frame_bytes: bytes | None = None
        self.last_middle_frame_bytes: bytes | None = None
        self.last_last_frame_bytes: bytes | None = None

    def generate(
        self,
        request: WorkerInput,
        image_path: Path | None,
        output_path: Path,
        *,
        first_frame_path: Path | None = None,
        middle_frame_path: Path | None = None,
        last_frame_path: Path | None = None,
        progress=None,
    ) -> dict[str, object]:
        self.last_request = request
        self.last_image_path = image_path
        self.last_image_bytes = image_path.read_bytes() if image_path else None
        self.last_first_frame_path = first_frame_path
        self.last_middle_frame_path = middle_frame_path
        self.last_last_frame_path = last_frame_path
        self.last_first_frame_bytes = first_frame_path.read_bytes() if first_frame_path else None
        self.last_middle_frame_bytes = middle_frame_path.read_bytes() if middle_frame_path else None
        self.last_last_frame_bytes = last_frame_path.read_bytes() if last_frame_path else None
        output_path.write_bytes(b"fake-mp4")
        return {"backend": "fake"}


def make_config(output_dir: Path) -> WorkerConfig:
    return WorkerConfig(
        default_model_id="diffusers/LTX-2.3-Distilled-Diffusers",
        model_cache_dir=output_dir / "models",
        persistent_model_cache_dir=output_dir / "persistent-models",
        runpod_cached_model_hub=output_dir / "runpod-cache" / "hub",
        output_dir=output_dir,
        output_mode="base64",
        s3_endpoint_url="",
        s3_bucket="",
        s3_public_base_url="",
        aws_access_key_id="",
        aws_secret_access_key="",
        aws_region="auto",
    )


class WorkerContractTests(unittest.TestCase):
    def test_parse_text_to_video_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request = parse_worker_input({
                "prompt": "A lighthouse at dusk",
                "gpu_type_id": "AMPERE_48",
            }, make_config(Path(temp_dir)))

        self.assertEqual(request.prompt, "A lighthouse at dusk")
        self.assertEqual(request.model_id, "diffusers/LTX-2.3-Distilled-Diffusers")
        self.assertEqual(request.gpu_type_id, "AMPERE_48")
        self.assertEqual(request.duration_seconds, 6)
        self.assertEqual(request.fps, 24)
        self.assertEqual(request.width, 1280)
        self.assertEqual(request.height, 704)
        self.assertEqual(request.requested_width, 1280)
        self.assertEqual(request.requested_height, 720)
        self.assertEqual(request.mode, "text_to_video")

    def test_parse_normalizes_model_dimensions_to_ltx_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request = parse_worker_input({
                "prompt": "A lighthouse at dusk",
                "width": 1920,
                "height": 1080,
            }, make_config(Path(temp_dir)))

        self.assertEqual(request.width, 1920)
        self.assertEqual(request.height, 1056)
        self.assertEqual(request.requested_width, 1920)
        self.assertEqual(request.requested_height, 1080)

    def test_parse_preserves_original_requested_dimensions_when_payload_is_pre_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request = parse_worker_input({
                "prompt": "A lighthouse at dusk",
                "width": 1920,
                "height": 1056,
                "requested_width": 1920,
                "requested_height": 1080,
            }, make_config(Path(temp_dir)))

        self.assertEqual(request.width, 1920)
        self.assertEqual(request.height, 1056)
        self.assertEqual(request.requested_width, 1920)
        self.assertEqual(request.requested_height, 1080)

    def test_parse_image_payload_switches_to_image_to_video(self) -> None:
        encoded = base64.b64encode(b"fake-image").decode("ascii")
        with tempfile.TemporaryDirectory() as temp_dir:
            request = parse_worker_input({
                "prompt": "Animate this image",
                "image": {
                    "filename": "../reference.png",
                    "mime_type": "image/png",
                    "base64": encoded,
                },
            }, make_config(Path(temp_dir)))

        self.assertEqual(request.mode, "image_to_video")
        self.assertIsNotNone(request.image)
        self.assertEqual(request.image.filename if request.image else "", "reference.png")

    def test_rejects_missing_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                parse_worker_input({}, make_config(Path(temp_dir)))

    def test_handle_job_returns_base64_video_and_metadata(self) -> None:
        fake_generator = FakeGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = handle_job({
                "input": {
                    "prompt": "A lighthouse at dusk",
                    "duration_seconds": 2,
                    "fps": 12,
                "width": 512,
                "height": 288,
                "gpu_type_id": "BLACKWELL_180",
                "contract_version": WORKER_CONTRACT_VERSION,
            }
        }, generator=fake_generator, config=make_config(Path(temp_dir)))

        self.assertEqual(base64.b64decode(output["video_base64"]), b"fake-mp4")
        self.assertEqual(output["metadata"]["backend"], "fake")
        self.assertEqual(output["metadata"]["gpu_type_id"], "BLACKWELL_180")
        self.assertEqual(output["metadata"]["mode"], "text_to_video")
        self.assertEqual(output["metadata"]["width"], 512)
        self.assertEqual(output["metadata"]["height"], 288)
        self.assertEqual(output["metadata"]["requested_width"], 512)
        self.assertEqual(output["metadata"]["requested_height"], 288)
        self.assertFalse(output["metadata"]["dimension_adjusted"])
        self.assertEqual(output["metadata"]["worker_contract_version"], WORKER_CONTRACT_VERSION)
        self.assertEqual(output["metadata"]["expected_contract_version"], WORKER_CONTRACT_VERSION)
        self.assertTrue(output["metadata"]["contract_version_match"])
        self.assertEqual(fake_generator.last_request.prompt if fake_generator.last_request else "", "A lighthouse at dusk")

    def test_handle_job_reports_contract_mismatch(self) -> None:
        fake_generator = FakeGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = handle_job({
                "input": {
                    "prompt": "A lighthouse at dusk",
                    "contract_version": "unexpected-version",
                }
            }, generator=fake_generator, config=make_config(Path(temp_dir)))

        self.assertEqual(output["metadata"]["worker_contract_version"], WORKER_CONTRACT_VERSION)
        self.assertEqual(output["metadata"]["expected_contract_version"], "unexpected-version")
        self.assertFalse(output["metadata"]["contract_version_match"])

    def test_handle_job_decodes_image_for_generator(self) -> None:
        fake_generator = FakeGenerator()
        encoded = base64.b64encode(b"fake-image").decode("ascii")
        with tempfile.TemporaryDirectory() as temp_dir:
            handle_job({
                "input": {
                    "prompt": "Animate this image",
                    "mode": "image_to_video",
                    "image": {
                        "filename": "reference.png",
                        "mime_type": "image/png",
                        "base64": encoded,
                    },
                }
            }, generator=fake_generator, config=make_config(Path(temp_dir)))

        self.assertIsNotNone(fake_generator.last_image_path)
        self.assertEqual(fake_generator.last_image_bytes, b"fake-image")

    def test_first_last_frame_mode_requires_and_decodes_both_frames(self) -> None:
        fake_generator = FakeGenerator()
        first = base64.b64encode(b"first-frame").decode("ascii")
        last = base64.b64encode(b"last-frame").decode("ascii")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = handle_job({
                "input": {
                    "prompt": "Loop this motion",
                    "mode": "first_last_frame_to_video",
                    "first_frame": {
                        "filename": "first.png",
                        "mime_type": "image/png",
                        "base64": first,
                    },
                    "last_frame": {
                        "filename": "last.png",
                        "mime_type": "image/png",
                        "base64": last,
                    },
                }
            }, generator=fake_generator, config=make_config(Path(temp_dir)))

        self.assertEqual(fake_generator.last_request.mode if fake_generator.last_request else "", "first_last_frame_to_video")
        self.assertIsNotNone(fake_generator.last_first_frame_path)
        self.assertIsNotNone(fake_generator.last_last_frame_path)
        self.assertEqual(fake_generator.last_first_frame_bytes, b"first-frame")
        self.assertEqual(fake_generator.last_last_frame_bytes, b"last-frame")
        self.assertTrue(output["metadata"]["first_frame_control"])
        self.assertTrue(output["metadata"]["last_frame_control"])
        self.assertFalse(output["metadata"]["multi_keyframe_control"])

    def test_multi_keyframe_mode_requires_and_decodes_first_middle_last_frames(self) -> None:
        fake_generator = FakeGenerator()
        first = base64.b64encode(b"first-frame").decode("ascii")
        middle = base64.b64encode(b"middle-frame").decode("ascii")
        last = base64.b64encode(b"last-frame").decode("ascii")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = handle_job({
                "input": {
                    "prompt": "Loop this motion",
                    "mode": "multi_keyframe_to_video",
                    "first_frame": {
                        "filename": "first.png",
                        "mime_type": "image/png",
                        "base64": first,
                    },
                    "middle_frame": {
                        "filename": "middle.png",
                        "mime_type": "image/png",
                        "base64": middle,
                    },
                    "last_frame": {
                        "filename": "last.png",
                        "mime_type": "image/png",
                        "base64": last,
                    },
                }
            }, generator=fake_generator, config=make_config(Path(temp_dir)))

        self.assertEqual(fake_generator.last_request.mode if fake_generator.last_request else "", "multi_keyframe_to_video")
        self.assertIsNotNone(fake_generator.last_first_frame_path)
        self.assertIsNotNone(fake_generator.last_middle_frame_path)
        self.assertIsNotNone(fake_generator.last_last_frame_path)
        self.assertEqual(fake_generator.last_first_frame_bytes, b"first-frame")
        self.assertEqual(fake_generator.last_middle_frame_bytes, b"middle-frame")
        self.assertEqual(fake_generator.last_last_frame_bytes, b"last-frame")
        self.assertTrue(output["metadata"]["first_frame_control"])
        self.assertTrue(output["metadata"]["middle_frame_control"])
        self.assertTrue(output["metadata"]["last_frame_control"])
        self.assertTrue(output["metadata"]["multi_keyframe_control"])

    def test_first_last_frame_mode_rejects_missing_last_frame(self) -> None:
        first = base64.b64encode(b"first-frame").decode("ascii")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                parse_worker_input({
                    "prompt": "Loop this motion",
                    "mode": "first_last_frame_to_video",
                    "first_frame": {
                        "filename": "first.png",
                        "mime_type": "image/png",
                        "base64": first,
                    },
                }, make_config(Path(temp_dir)))


if __name__ == "__main__":
    unittest.main()
