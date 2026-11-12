"""Run a detector, tracker, line counter, and video output on a video file."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from .demo_detector import detect_demo_fruit
from .tracker import Detection, LineCounter
from .video import encode_video, iter_video_frames, probe_video, scaled_size


def _annotate(frame: np.ndarray, counter: LineCounter, frame_index: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_x = round(counter.line_x)
    # Counts drawn per frame are provisional; summary.json uses final class votes.
    draw.line((line_x, 0, line_x, height), fill=(255, 255, 255), width=2)
    for track in counter.tracks.values():
        if not track.confirmed or track.last_frame < frame_index:
            continue
        color = (255, 165, 0) if track.label == "orange" else (255, 50, 70)
        draw.rectangle(track.bbox, outline=color, width=2)
        draw.text((track.bbox[0], max(0, track.bbox[1] - 12)),
                  f"{track.label} #{track.id}", fill=(255, 255, 255))
    total = sum(counter.counts.values())
    draw.rectangle((0, 0, min(width, 258), 18), fill=(0, 0, 0))
    draw.text((4, 2),
              f"orange {counter.counts['orange']}  apple {counter.counts['apple']}  total {total}",
              fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def count_video(
    input_path: Path,
    output_dir: Path,
    width: int = 320,
    line_ratio: float = 0.5,
    direction: str = "ltr",
    detector: Callable[[np.ndarray], list[Detection]] | None = None,
    method: str = "synthetic_color_demo",
) -> dict:
    if not 0 < line_ratio < 1:
        raise ValueError("line_ratio must be between 0 and 1")
    info = probe_video(input_path)
    size = scaled_size(info, width)
    detector = detector or detect_demo_fruit
    counter = LineCounter(line_x=size[0] * line_ratio, direction=direction)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0

    def processed_frames():
        nonlocal frame_count
        for frame_count, frame in enumerate(iter_video_frames(input_path, size), start=1):
            counter.update(detector(frame), frame_count - 1, info.fps)
            yield _annotate(frame, counter, frame_count - 1)

    encode_video(output_dir / "annotated.mp4", processed_frames(), size, info.fps)
    if frame_count == 0:
        raise ValueError(f"No frames in {input_path}")
    counter.finalize()
    events = [asdict(event) for event in counter.events]
    summary = {
        "input": str(input_path),
        "method": method,
        "frames": frame_count,
        "fps": info.fps,
        "line_ratio": line_ratio,
        "direction": direction,
        "counts": counter.counts,
        "total": sum(counter.counts.values()),
        "events": events,
    }
    if method == "synthetic_color_demo":
        summary["warning"] = "Color thresholds are only calibrated for the generated demo video."
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "seconds", "track_id", "label"])
        writer.writeheader()
        writer.writerows(events)
    return summary
