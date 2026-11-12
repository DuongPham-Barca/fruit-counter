"""Color detector for the generated demo only; not a trained fruit model."""

import numpy as np

from .tracker import Detection


def _components(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=np.bool_)
    boxes = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [(int(start_y), int(start_x))]
        area = 0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        while stack:
            y, x = stack.pop()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for neighbor_y, neighbor_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and mask[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))
        if area >= min_area:
            boxes.append((min_x, min_y, max_x + 1, max_y + 1))
    return boxes


def detect_demo_fruit(frame: np.ndarray, min_area: int = 40) -> list[Detection]:
    """Detect the deliberately saturated colors used by the synthetic fixture."""
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    masks = {
        "orange": (red > 175) & (green > 75) & (green < 190) & (blue < 95),
        "apple": (red > 145) & (green < 95) & (blue < 115),
    }
    return [
        Detection(label, bbox)
        for label, mask in masks.items()
        for bbox in _components(mask, min_area)
    ]
