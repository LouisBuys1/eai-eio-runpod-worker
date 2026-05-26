from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eai_eio_runpod_sulphur_worker.ltx_diffusers import LtxDiffusersGenerator, _compute_frame_count, _middle_frame_index


def write_minimal_diffusers_snapshot(snapshot: Path) -> None:
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    for component in ("transformer", "vae", "text_encoder"):
        (snapshot / component).mkdir()
        (snapshot / component / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "scheduler").mkdir()
    (snapshot / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer").mkdir()
    (snapshot / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class LtxDiffusersTests(unittest.TestCase):
    def test_frame_count_is_ltx_time_grid_aligned(self) -> None:
        self.assertEqual(_compute_frame_count(6, 24), 145)
        self.assertEqual(_compute_frame_count(2, 12), 25)
        self.assertEqual(_compute_frame_count(1, 8), 9)

    def test_middle_frame_index_prefers_interior_ltx_condition_slot(self) -> None:
        self.assertEqual(_middle_frame_index(145), 9)
        self.assertEqual(_middle_frame_index(121), 8)
        self.assertEqual(_middle_frame_index(25), 2)
        self.assertEqual(_middle_frame_index(9), 1)

    def test_resolves_runpod_cached_model_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "hub"
            model_root = cache_root / "models--diffusers--LTX-2.3-Distilled-Diffusers"
            snapshot = model_root / "snapshots" / "abc123"
            write_minimal_diffusers_snapshot(snapshot)
            (model_root / "refs").mkdir()
            (model_root / "refs" / "main").write_text("abc123", encoding="utf-8")

            resolved = LtxDiffusersGenerator()._resolve_cached_snapshot(
                "diffusers/LTX-2.3-Distilled-Diffusers",
                cache_root,
            )

        self.assertEqual(resolved, snapshot)

    def test_ignores_partial_cached_model_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "hub"
            model_root = cache_root / "models--diffusers--LTX-2.3-Distilled-Diffusers"
            snapshot = model_root / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
            (model_root / "refs").mkdir()
            (model_root / "refs" / "main").write_text("abc123", encoding="utf-8")

            resolved = LtxDiffusersGenerator()._resolve_cached_snapshot(
                "diffusers/LTX-2.3-Distilled-Diffusers",
                cache_root,
            )

        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
