"""Build deterministic train and validation loaders."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..engine.config import TrainConfig
from ..dataset import YoloDetectionDataset, collate, eval_transforms, train_transforms


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_loaders(config: TrainConfig, device: torch.device, generator: torch.Generator):
    train_set = YoloDetectionDataset(Path(config.data), config.train_split, train_transforms())
    val_set = YoloDetectionDataset(Path(config.data), config.val_split, eval_transforms())
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=device.type == "cuda",
        drop_last=len(train_set) > config.batch_size,
        persistent_workers=config.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, config.workers),
        collate_fn=collate,
    )
    return train_set, val_set, train_loader, val_loader
