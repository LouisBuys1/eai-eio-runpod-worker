"""RunPod entrypoint."""

from __future__ import annotations

from typing import Any, cast
import traceback

from eai_eio_runpod_sulphur_worker.contract import handle_job


def runpod_handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], handle_job(job))
    except Exception as exc:
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
