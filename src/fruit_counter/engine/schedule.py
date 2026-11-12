"""Optimizer parameter groups and learning-rate schedules."""

import math

import torch


def lr_factor(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> float:
    """Linear warmup followed by cosine decay to 1% of the base rate."""
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
