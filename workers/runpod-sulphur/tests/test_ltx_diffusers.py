from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eai_eio_runpod_sulphur_worker.ltx_diffusers import LtxDiffusersGenerator, _compute_frame_count, _middle_frame_index


class LtxDiffusersTests(unittest.TestCase):
    def test_frame_count_is_ltx_time_grid_aligned(self) -> None:
        self.assertEqual(_compute_frame_count(6, 24), 145)
        self.assertEqual(_compute_frame_count(2, 12), 25)
        self.assertEqual(_compute_frame_count(1, 8), 9)

    def test_middle_frame_index_prefers_interior_ltx_condition_slot(self) -> None:
        self.assertEqual(_middle_frame_index(145), 72)
        self.assertEqual(_middle_frame_index(25), 16)
        self.assertEqual(_middle_frame_index(9), 4)

    def test_resolves_runpod_cached_model_snapshot(self) -> None:
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

        self.assertEqual(resolved, snapshot)


if __name__ == "__main__":
    unittest.main()
