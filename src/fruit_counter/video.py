"""Read and write video files through the installed FFmpeg tools."""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float


def probe_video(path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream in {path}")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = float(Fraction(rate)) if rate else 0
    if fps <= 0:
        raise ValueError(f"Cannot determine frame rate of {path}")
    return VideoInfo(int(stream["width"]), int(stream["height"]), fps)


def scaled_size(info: VideoInfo, width: int) -> tuple[int, int]:
    if width < 2 or width % 2:
        raise ValueError("width must be a positive even number")
    height = max(2, round(info.height * width / info.width / 2) * 2)
    return width, height


def _read_exact(pipe, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = pipe.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_video_frames(path: Path, size: tuple[int, int]) -> Iterator[np.ndarray]:
    width, height = size
    process = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-map", "0:v:0", "-an", "-vf", f"scale={width}:{height}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    frame_size = width * height * 3
    try:
        while True:
            raw = _read_exact(process.stdout, frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("FFmpeg returned an incomplete video frame")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"Cannot decode video: {error.strip()}")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()


def encode_video(
    path: Path,
    frames: Iterable[np.ndarray],
    size: tuple[int, int],
    fps: float,
) -> None:
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "mpeg4",
            "-q:v", "3", str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None
    try:
        for frame in frames:
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError("frames must be RGB uint8 arrays matching the video size")
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"Cannot encode video: {error.strip()}")
    except Exception:
        if not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        process.stderr.close()
