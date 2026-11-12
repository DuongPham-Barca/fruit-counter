"""Generate a small synthetic conveyor clip for pipeline checks."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .video import encode_video


def make_demo(path: Path, frames: int = 80, fps: float = 20) -> None:
    width, height = 320, 180

    def images():
        for index in range(frames):
            image = Image.new("RGB", (width, height), (35, 40, 42))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 25, width, 160), fill=(55, 58, 58))
            orange_x = 25 + 3 * index
            apple_x = -30 + 3 * index
            for x, y, color in (
                (orange_x, 72, (245, 135, 25)),
                (apple_x, 120, (215, 30, 45)),
            ):
                draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=color)
            yield np.asarray(image, dtype=np.uint8)

    encode_video(path, images(), (width, height), fps)
