"""YOLO-format dataset, augmentation, and label parsing.

Expected layout:

    dataset/train/images + dataset/train/labels
    dataset/val/images   + dataset/val/labels
    dataset/test/images  + dataset/test/labels

Each label line is: class_id center_x center_y width height, normalized to 0..1.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from .labels import CLASSES, model_label

if TYPE_CHECKING:
    from .train import TrainConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

__all__ = [
    "IMAGE_SUFFIXES",
    "YoloDetectionDataset",
    "build_loaders",
    "collate",
    "eval_transforms",
    "read_yolo_labels",
    "seed_worker",
    "train_transforms",
]


def read_yolo_labels(path: Path, width: int, height: int) -> tuple[list, list]:
    """Convert normalized YOLO labels to absolute XYXY boxes for Faster R-CNN."""
    boxes, labels = [], []
    if not path.exists():
        return boxes, labels
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 'class cx cy w h'")
        class_id = int(parts[0])
        cx, cy, box_width, box_height = (float(value) for value in parts[1:])
        if not all(
            0 <= value <= 1 for value in (cx, cy, box_width, box_height)
        ):
            raise ValueError(f"{path}:{line_number}: coordinates must be in 0..1")
        x1 = max(0.0, (cx - box_width / 2) * width)
        y1 = max(0.0, (cy - box_height / 2) * height)
        x2 = min(float(width), (cx + box_width / 2) * width)
        y2 = min(float(height), (cy + box_height / 2) * height)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append((x1, y1, x2, y2))
        labels.append(model_label(class_id))
    return boxes, labels


def train_transforms() -> v2.Compose:
    """Augment training images and their bounding boxes together."""
    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomApply([
            v2.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.7, 1.3)
            )
        ], p=0.5),
        v2.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02
        ),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_size=2),
        v2.ToDtype(torch.float32, scale=True),
    ])


def eval_transforms() -> v2.Compose:
    """Validation/test only convert pixels to float tensors; no random changes."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


class YoloDetectionDataset(Dataset):
    """Load RGB images and Faster R-CNN targets from one dataset split."""

    def __init__(
        self, root: Path, split: str, transforms: v2.Compose | None = None
    ) -> None:
        self.image_dir = Path(root) / split / "images"
        self.label_dir = Path(root) / split / "labels"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory {self.image_dir}")
        self.images = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise ValueError(f"No images in {self.image_dir}")
        self.transforms = transforms or eval_transforms()

    def __len__(self) -> int:
        return len(self.images)

    def target(self, index: int, size: tuple[int, int]) -> tuple[list, list]:
        width, height = size
        label_path = self.label_dir / f"{self.images[index].stem}.txt"
        return read_yolo_labels(label_path, width, height)

    def __getitem__(self, index: int):
        with Image.open(self.images[index]) as file:
            image = file.convert("RGB")
        boxes, labels = self.target(index, image.size)
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                format="XYXY",
                canvas_size=(image.height, image.width),
            ),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": index,
        }
        image, target = self.transforms(image, target)
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)
        return image, target

    def class_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(CLASSES, 0)
        for index, path in enumerate(self.images):
            with Image.open(path) as file:
                _, labels = self.target(index, file.size)
            for label in labels:
                counts[CLASSES[label - 1]] += 1
        return counts


def collate(batch):
    """Detection models consume lists because each image has a different box count."""
    return tuple(zip(*batch))


def seed_worker(_worker_id: int) -> None:
    """Give each loader worker deterministic NumPy RNG state."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def build_loaders(
    config: "TrainConfig",
    device: torch.device,
    generator: torch.Generator,
):
    """Build deterministic train and validation datasets/loaders."""
    train_set = YoloDetectionDataset(
        Path(config.data),
        config.train_split,
        train_transforms(),
    )
    val_set = YoloDetectionDataset(
        Path(config.data),
        config.val_split,
        eval_transforms(),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=len(train_set) > config.batch_size,
        persistent_workers=config.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=min(config.workers, 2),
        collate_fn=collate,
    )
    return train_set, val_set, train_loader, val_loader
