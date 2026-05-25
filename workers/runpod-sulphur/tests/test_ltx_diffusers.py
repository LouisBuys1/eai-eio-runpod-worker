from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eai_eio_runpod_sulphur_worker.ltx_diffusers import _compute_frame_count, _middle_frame_index


class LtxDiffusersTests(unittest.TestCase):
    def test_frame_count_is_ltx_time_grid_aligned(self) -> None:
        self.assertEqual(_compute_frame_count(6, 24), 145)
        self.assertEqual(_compute_frame_count(2, 12), 25)
        self.assertEqual(_compute_frame_count(1, 8), 9)

    def test_middle_frame_index_prefers_interior_ltx_condition_slot(self) -> None:
        self.assertEqual(_middle_frame_index(145), 72)
        self.assertEqual(_middle_frame_index(25), 16)
        self.assertEqual(_middle_frame_index(9), 4)


if __name__ == "__main__":
    unittest.main()
