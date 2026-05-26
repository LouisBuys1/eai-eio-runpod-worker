"""RunPod entrypoint."""

from __future__ import annotations

from typing import Any, cast
import json
import traceback

from eai_eio_runpod_sulphur_worker.contract import handle_job


def runpod_handler(job: dict[str, Any]) -> dict[str, Any]:
    def progress(phase: str, metadata: dict[str, Any]) -> None:
        try:
            import runpod

            payload = {"phase": phase, **metadata}
            runpod.serverless.progress_update(job, json.dumps(payload, separators=(",", ":")))
        except Exception:
            pass

    try:
        return cast(dict[str, Any], handle_job(job, progress=progress))
    except Exception as exc:
        progress("failed", {"error": str(exc) or exc.__class__.__name__})
        return {
            "error": str(exc) or exc.__class__.__name__,
            "detail": exc.__class__.__name__,
            "traceback_tail": traceback.format_exc(limit=4),
        }


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": runpod_handler})


if __name__ == "__main__":
    main()
