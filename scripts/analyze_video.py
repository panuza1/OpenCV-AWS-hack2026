#!/usr/bin/env python3
"""Extract an orange-ball track from a video for prediction and evaluation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physicseye.tracker import VideoError, track_video, write_track


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--fps", type=float, help="explicit FPS when media metadata is unavailable")
    args = parser.parse_args(argv)

    try:
        observations = track_video(args.video, args.fps)
        output_stem = args.output_dir / f"{args.video.stem}.track"
        json_path, csv_path = write_track(observations, output_stem)
    except (VideoError, OSError) as error:
        parser.exit(2, f"error: {error}\n")

    valid_frames = sum(observation["valid"] for observation in observations)
    print(
        json.dumps(
            {
                "video": str(args.video),
                "frames": len(observations),
                "valid_frames": valid_frames,
                "json": str(json_path),
                "csv": str(csv_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

