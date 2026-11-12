"""Typed runtime configuration for detector training."""

from dataclasses import dataclass


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
