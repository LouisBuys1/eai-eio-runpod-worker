"""Local smoke runner for the RunPod worker contract."""

from __future__ import annotations

import argparse

from eai_eio_runpod_sulphur_worker.contract import handle_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local EAI-EIO RunPod Sulphur worker job.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--duration", type=float, default=6)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    output = handle_job({
        "input": {
            "task": "sulphur_video",
            "prompt": args.prompt,
            "model_id": args.model_id,
            "duration_seconds": args.duration,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "mode": "text_to_video",
            "seed": args.seed,
        }
    })
    print(output)


if __name__ == "__main__":
    main()
