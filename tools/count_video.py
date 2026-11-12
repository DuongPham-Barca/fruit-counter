#!/usr/bin/env python
"""Run a trained detector, tracker, and counter on a video."""

import argparse
import json
from pathlib import Path

from fruit_counter.frcnn_detector import FasterRCNNDetector
from fruit_counter.pipeline import count_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/count"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--line", type=float, default=0.5)
    parser.add_argument("--direction", choices=("ltr", "rtl"), default="ltr")
    parser.add_argument("--min-score", type=float, default=0.1)
    args = parser.parse_args()
    detector = FasterRCNNDetector(args.checkpoint, args.device, args.min_score)
    result = count_video(
        args.input,
        args.output,
        args.width,
        args.line,
        args.direction,
        detector=detector,
        method=f"faster_rcnn:{args.checkpoint}",
    )
    print(json.dumps({"counts": result["counts"], "total": result["total"]}))


if __name__ == "__main__":
    main()
