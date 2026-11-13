"""Training configuration, runtime helpers, and public training API."""

import json
import math
import platform
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision


@dataclass
class TrainConfig:
    data: str
    output: str
    backbone: str = "resnet50"
    epochs: int = 100
    batch_size: int = 2
    accumulate: int = 4
    lr: float | None = None
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.01
    max_grad_norm: float = 10.0
    workers: int = 4
    seed: int = 0
    device: str = "cuda"
    amp: bool = True
    eval_every: int = 1
    save_every: int = 0
    log_every: int = 20
    vis_images: int = 4
    max_bad_batches: int = 20
    limit_batches: int | None = None
    train_split: str = "train"
    val_split: str = "val"

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.accumulate

    @property
    def base_lr(self) -> float:
        return self.lr if self.lr is not None else 0.02 * self.effective_batch / 16

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1 or self.accumulate < 1:
            raise ValueError("batch_size and accumulate must be positive")
        if self.workers < 0:
            raise ValueError("workers cannot be negative")
        if self.eval_every < 1 or self.log_every < 1:
            raise ValueError("eval_every and log_every must be positive")
        if self.save_every < 0:
            raise ValueError("save_every cannot be negative")
        if self.base_lr <= 0:
            raise ValueError("learning rate must be positive")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between 0 and 1")


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    logs: Path
    evaluation: Path

    @classmethod
    def create(cls, root: Path) -> "RunPaths":
        paths = cls(
            root=root,
            checkpoints=root / "checkpoints",
            logs=root / "logs",
            evaluation=root / "evaluation",
        )
        for path in (paths.root, paths.checkpoints, paths.logs, paths.evaluation):
            path.mkdir(parents=True, exist_ok=True)
        return paths


def lr_factor(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> float:
    """Linear warmup followed by cosine decay to a minimum LR ratio."""
    if step < warmup_steps:
        return 0.001 + (1 - 0.001) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, progress))
    )


def parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Exclude normalization scales and biases from weight decay."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(path: Path, **state) -> None:
    """Atomically save a training or inference checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


def capture_rng_state(generator: torch.Generator) -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "data_loader": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None, generator: torch.Generator) -> None:
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(state["torch"])
    generator.set_state(state["data_loader"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def write_run_metadata(
    paths: RunPaths,
    train_config: TrainConfig,
    model_config: dict,
    resolved_config: dict | None,
    config_source: Path | None,
) -> None:
    """Record immutable environment and configuration details for a run."""
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "train": asdict(train_config),
        "model": model_config,
        "pretrained_weights": None,
    }
    metadata_path = paths.root / "run_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (paths.root / f"resume_metadata_{timestamp}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    if resolved_config is not None and not (paths.root / "resolved_config.json").exists():
        (paths.root / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2) + "\n",
            encoding="utf-8",
        )
    if config_source is not None and not (paths.root / "source_config.py").exists():
        shutil.copy2(config_source, paths.root / "source_config.py")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(*args, **kwargs):
    """Load training code only when a training run starts."""
    from .engine import train as run_training

    return run_training(*args, **kwargs)

__all__ = [
    "TrainConfig",
    "load_checkpoint",
    "lr_factor",
    "save_checkpoint",
    "set_seed",
    "train",
]
