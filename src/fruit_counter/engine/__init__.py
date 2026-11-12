"""Training and evaluation runtime for fruit detection models."""

from .config import TrainConfig


def train(*args, **kwargs):
    """Load the trainer lazily so inference does not import training dependencies."""
    from .trainer import train as run_training

    return run_training(*args, **kwargs)

__all__ = ["TrainConfig", "train"]
