"""One-epoch detector training loop."""

import time
from dataclasses import dataclass

import torch

from .config import TrainConfig
from .schedule import lr_factor


@dataclass
class EpochResult:
    step: int
    loss: dict[str, float]
    seconds: float
    lr: float
    gpu_peak_memory_gb: float | None = None


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    config: TrainConfig,
    epoch: int,
    step: int,
    batches_per_epoch: int,
    warmup_steps: int,
    total_steps: int,
    writer,
    training_started: float,
) -> EpochResult:
    model.train()
    started = time.time()
    epoch_sums, window_sums, window_batches = {}, {}, 0
    finite_batches, bad_batches, window_started = 0, 0, time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (images, targets) in enumerate(loader):
        if batch_index >= batches_per_epoch:
            break
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)}
            for target in targets
        ]
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            bad_batches += 1
            writer.add_scalar("train/skipped_batches", bad_batches, step)
            optimizer.zero_grad(set_to_none=True)
            if bad_batches >= config.max_bad_batches:
                raise RuntimeError(
                    f"{bad_batches} consecutive non-finite losses at epoch {epoch}; "
                    "try a lower learning rate or disable AMP"
                )
            continue
        bad_batches = 0
        finite_batches += 1
        scaler.scale(loss / config.accumulate).backward()
        loss_values = {**losses, "loss_total": loss}
        for name, value in loss_values.items():
            scalar = value.item()
            epoch_sums[name] = epoch_sums.get(name, 0.0) + scalar
            window_sums[name] = window_sums.get(name, 0.0) + scalar
        window_batches += 1

        last_batch = batch_index + 1 == batches_per_epoch
        if finite_batches % config.accumulate != 0 and not last_batch:
            continue
        lr = config.base_lr * lr_factor(
            step,
            warmup_steps,
            total_steps,
            config.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % config.log_every == 0 and window_batches:
            elapsed = time.time() - window_started
            for name, value in window_sums.items():
                writer.add_scalar(f"train/{name}", value / window_batches, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar(
                "train/images_per_second",
                window_batches * config.batch_size / max(elapsed, 1e-6),
                step,
            )
            if device.type == "cuda":
                writer.add_scalar(
                    "train/gpu_peak_memory_gb",
                    torch.cuda.max_memory_allocated(device) / 1e9,
                    step,
                )
            remaining = (
                (time.time() - training_started)
                / max(step, 1)
                * max(total_steps - step, 0)
            )
            print(
                f"epoch {epoch + 1}/{config.epochs} step {step}/{total_steps} "
                f"loss {window_sums['loss_total'] / window_batches:.4f} "
                f"lr {lr:.5f} eta {_format_seconds(remaining)}",
                flush=True,
            )
            window_sums, window_batches, window_started = {}, 0, time.time()

    denominator = max(1, finite_batches)
    peak = None
    if device.type == "cuda":
        peak = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    return EpochResult(
        step=step,
        loss={name: value / denominator for name, value in epoch_sums.items()},
        seconds=round(time.time() - started, 1),
        lr=optimizer.param_groups[0]["lr"],
        gpu_peak_memory_gb=peak,
    )
