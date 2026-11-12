"""Public training API.

For the easiest reading order, start at the repository-level ``train_model.py``.
The per-batch forward/backward/optimizer steps live in ``engine/loop.py`` and
checkpoint/logging orchestration lives in ``engine/trainer.py``.
"""

from .engine.checkpoint import load_checkpoint, save_checkpoint
from .engine.config import TrainConfig
from .engine.schedule import lr_factor
from .engine.trainer import set_seed, train

__all__ = [
    "TrainConfig",
    "load_checkpoint",
    "lr_factor",
    "save_checkpoint",
    "set_seed",
    "train",
]
