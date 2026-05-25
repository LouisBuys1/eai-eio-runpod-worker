from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eai_eio_runpod_sulphur_worker.worker import runpod_handler


class WorkerEntrypointTests(unittest.TestCase):
    def test_runpod_handler_returns_structured_error_for_invalid_input(self) -> None:
        output = runpod_handler({"input": {}})

        self.assertEqual(output["detail"], "ValueError")
        self.assertIn("prompt is required", output["error"])
        self.assertIn("ValueError", output["traceback_tail"])

    def test_runpod_handler_returns_contract_output(self) -> None:
        with patch("eai_eio_runpod_sulphur_worker.worker.handle_job", return_value={
            "video_url": "https://example.test/output.mp4",
            "metadata": {"backend": "fake"},
        }):
            output = runpod_handler({"input": {"prompt": "A lighthouse"}})

        self.assertEqual(output["video_url"], "https://example.test/output.mp4")
        self.assertEqual(output["metadata"]["backend"], "fake")


if __name__ == "__main__":
    unittest.main()
