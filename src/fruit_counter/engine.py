"""Faster R-CNN training: one epoch, validation, checkpoints and TensorBoard."""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes

from .dataset import build_loaders
from .evaluate import evaluate_loss, evaluate_model
from .labels import CLASSES
from .model import ModelConfig, build_model
from .train import (
    RunPaths,
    TrainConfig,
    capture_rng_state,
    load_checkpoint,
    lr_factor,
    parameter_groups,
    restore_rng_state,
    save_checkpoint,
    set_seed,
    write_run_metadata,
)

COLORS = {1: (255, 150, 0), 2: (230, 30, 60)}


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
    """Run forward pass, loss, backward pass and optimizer update for one epoch."""
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
        for name, value in {**losses, "loss_total": loss}.items():
            scalar = value.item()
            epoch_sums[name] = epoch_sums.get(name, 0.0) + scalar
            window_sums[name] = window_sums.get(name, 0.0) + scalar
        window_batches += 1

        last_batch = batch_index + 1 == batches_per_epoch
        if finite_batches % config.accumulate != 0 and not last_batch:
            continue
        lr = config.base_lr * lr_factor(step, warmup_steps, total_steps, config.min_lr_ratio)
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
            writer.add_scalar("train/images_per_second", window_batches * config.batch_size / max(elapsed, 1e-6), step)
            if device.type == "cuda":
                writer.add_scalar("train/gpu_peak_memory_gb", torch.cuda.max_memory_allocated(device) / 1e9, step)
            remaining = (time.time() - training_started) / max(step, 1) * max(total_steps - step, 0)
            print(
                f"epoch {epoch + 1}/{config.epochs} step {step}/{total_steps} "
                f"loss {window_sums['loss_total'] / window_batches:.4f} lr {lr:.5f} "
                f"eta {_format_seconds(remaining)}",
                flush=True,
            )
            window_sums, window_batches, window_started = {}, 0, time.time()

    peak = round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else None
    return EpochResult(
        step=step,
        loss={name: value / max(1, finite_batches) for name, value in epoch_sums.items()},
        seconds=round(time.time() - started, 1),
        lr=optimizer.param_groups[0]["lr"],
        gpu_peak_memory_gb=peak,
    )


def _log_confusion_matrix(writer, confusion: dict, step: int) -> None:
    labels, matrix = confusion["labels"], np.asarray(confusion["matrix"], dtype=int)
    cell, label_width, header_height = 92, 112, 52
    image = Image.new("RGB", (label_width + cell * len(labels), header_height + cell * len(labels)), "white")
    draw = ImageDraw.Draw(image)
    maximum = max(1, int(matrix.max()))
    draw.text((4, 4), "GT / Pred", fill="black")
    for index, label in enumerate(labels):
        draw.text((label_width + index * cell + 5, 18), label, fill="black")
        draw.text((5, header_height + index * cell + 36), label, fill="black")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = int(matrix[row, column])
            intensity = int(235 - 180 * value / maximum)
            left, top = label_width + column * cell, header_height + row * cell
            draw.rectangle((left, top, left + cell, top + cell), fill=(intensity, intensity, 255), outline=(80, 80, 80))
            draw.text((left + 38, top + 36), str(value), fill="black")
    writer.add_image("val/confusion_matrix", torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1), step)


@torch.inference_mode()
def _log_predictions(writer, model, dataset, count: int, device, step: int) -> None:
    model.eval()
    for index in range(min(count, len(dataset))):
        image, target = dataset[index]
        output = model([image.to(device)])[0]
        keep = output["scores"] >= 0.5
        canvas = (image * 255).to(torch.uint8)
        if len(target["boxes"]):
            canvas = draw_bounding_boxes(canvas, target["boxes"], colors="white", width=1)
        boxes = output["boxes"][keep].float().cpu()
        if len(boxes):
            labels, scores = output["labels"][keep].cpu().tolist(), output["scores"][keep].float().cpu().tolist()
            canvas = draw_bounding_boxes(
                canvas, boxes, width=2,
                labels=[f"{CLASSES[label - 1]} {score:.2f}" for label, score in zip(labels, scores)],
                colors=[COLORS[label] for label in labels],
            )
        writer.add_image(f"val_predictions/{index}", canvas, step)


def _resume_path(path: Path, paths: RunPaths) -> Path:
    return paths.checkpoints / "last.pt" if str(path) == "auto" else path


def _validate_resume_config(previous: dict | None, current: TrainConfig) -> None:
    if not previous:
        return
    immutable = ("data", "batch_size", "accumulate", "lr", "momentum", "weight_decay", "warmup_steps", "min_lr_ratio", "max_grad_norm", "train_split", "val_split", "seed")
    changed = [name for name in immutable if name in previous and previous[name] != getattr(current, name)]
    if changed:
        details = ", ".join(f"{name}: {previous[name]!r} -> {getattr(current, name)!r}" for name in changed)
        raise ValueError(f"Resume would change training state: {details}")


def train(
    config: TrainConfig,
    model_config: ModelConfig | None = None,
    resume: Path | None = None,
    *,
    resolved_config: dict | None = None,
    config_source: Path | None = None,
) -> dict:
    """Train, validate, save checkpoints and return a short run summary."""
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use a CPU config or override runtime.device")

    paths = RunPaths.create(Path(config.output))
    if resume is None and (paths.checkpoints / "last.pt").exists():
        raise ValueError(f"{paths.checkpoints / 'last.pt'} already exists; use --resume auto or choose another output directory")
    resume = _resume_path(resume, paths) if resume is not None else None
    state = load_checkpoint(resume, map_location="cpu") if resume else None
    if state is not None:
        _validate_resume_config(state.get("train_config"), config)
        saved_model_config = ModelConfig.from_dict(state["model_config"])
        if model_config is not None and model_config != saved_model_config:
            raise ValueError("Resume config does not match the checkpoint model config")
        model_config = saved_model_config
    model_config = model_config or ModelConfig(backbone=config.backbone)

    set_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.SGD(parameter_groups(model, config.weight_decay), lr=config.base_lr, momentum=config.momentum)
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")
    train_set, val_set, train_loader, val_loader = build_loaders(config, device, generator)
    batches_per_epoch = min(len(train_loader), config.limit_batches) if config.limit_batches else len(train_loader)
    steps_per_epoch = math.ceil(batches_per_epoch / config.accumulate)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = min(config.warmup_steps, max(1, total_steps // 10))

    start_epoch, step, best_map, best_val_loss = 0, 0, -1.0, float("inf")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step, best_map = state["epoch"] + 1, state["step"], state["best_map"]
        best_val_loss = state.get("best_val_loss", float("inf"))
        restore_rng_state(state.get("rng_state"), generator)
        print(f"resumed from {resume}: epoch {start_epoch}, step {step}", flush=True)
        del state

    write_run_metadata(paths, config, model_config.to_dict(), resolved_config, config_source)
    writer = SummaryWriter(str(paths.logs / "tensorboard"), purge_step=step if step else None)
    if step == 0:
        writer.add_text("config", json.dumps(resolved_config or asdict(config), indent=2), 0)
    log = (paths.logs / "metrics.jsonl").open("a", encoding="utf-8")

    def make_state(epoch: int) -> dict:
        return {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "model_config": model_config.to_dict(), "train_config": asdict(config), "classes": list(CLASSES),
            "epoch": epoch, "step": step, "best_map": best_map, "best_val_loss": best_val_loss,
            "rng_state": capture_rng_state(generator),
        }

    print(f"training {model_config.backbone}: {len(train_set)} train / {len(val_set)} val images", flush=True)
    metrics, epoch, started = {}, start_epoch, time.time()
    try:
        for epoch in range(start_epoch, config.epochs):
            result = train_one_epoch(model, train_loader, optimizer, scaler, device, config, epoch, step, batches_per_epoch, warmup_steps, total_steps, writer, started)
            step = result.step
            metrics = {"epoch": epoch, "step": step, "lr": result.lr, "seconds": result.seconds, "loss": result.loss}
            if result.gpu_peak_memory_gb is not None:
                metrics["gpu_peak_memory_gb"] = result.gpu_peak_memory_gb
            for name, value in result.loss.items():
                writer.add_scalar(f"epoch/{name}", value, epoch + 1)

            if (epoch + 1) % config.eval_every == 0 or epoch + 1 == config.epochs:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                metrics["val_loss"] = evaluate_loss(model, val_loader, device, config.amp)
                metrics["val"] = evaluate_model(model, val_loader, device, config.amp)
                for name, value in metrics["val_loss"].items():
                    writer.add_scalar(f"val/{name}", value, epoch + 1)
                writer.add_scalar("val/mAP50", metrics["val"]["map"], epoch + 1)
                writer.add_scalar("val/precision_macro", metrics["val"]["precision"], epoch + 1)
                writer.add_scalar("val/recall_macro", metrics["val"]["recall"], epoch + 1)
                for name, item in metrics["val"]["classes"].items():
                    for metric in ("ap", "precision", "recall", "max_recall"):
                        writer.add_scalar(f"val/{'AP50' if metric == 'ap' else metric}_{name}", item[metric], epoch + 1)
                _log_confusion_matrix(writer, metrics["val"]["confusion_matrix"], epoch + 1)
                (paths.evaluation / f"val_epoch_{epoch + 1:03d}.json").write_text(json.dumps({"loss": metrics["val_loss"], "metrics": metrics["val"]}, indent=2) + "\n", encoding="utf-8")
                if config.vis_images:
                    _log_predictions(writer, model, val_set, config.vis_images, device, epoch + 1)

            val_map = metrics.get("val", {}).get("map")
            val_loss = metrics.get("val_loss", {}).get("loss_total")
            improved = val_map is not None and not math.isnan(val_map) and val_map > best_map
            loss_improved = val_loss is not None and math.isfinite(val_loss) and val_loss < best_val_loss
            if improved:
                best_map = val_map
                save_checkpoint(paths.checkpoints / "best_map50.pt", **{key: value for key, value in make_state(epoch).items() if key not in ("optimizer", "scaler", "rng_state")})
            if loss_improved:
                best_val_loss = val_loss
                save_checkpoint(paths.checkpoints / "best_val_loss.pt", **{key: value for key, value in make_state(epoch).items() if key not in ("optimizer", "scaler", "rng_state")})
            checkpoint = make_state(epoch)
            save_checkpoint(paths.checkpoints / "last.pt", **checkpoint)
            if config.save_every and (epoch + 1) % config.save_every == 0:
                save_checkpoint(paths.checkpoints / f"epoch_{epoch + 1:03d}.pt", **checkpoint)
            metrics["best_map"] = best_map
            metrics["best_val_loss"] = best_val_loss if math.isfinite(best_val_loss) else None
            writer.add_scalar("val/best_mAP50", max(best_map, 0.0), epoch + 1)
            if math.isfinite(best_val_loss):
                writer.add_scalar("val/best_loss", best_val_loss, epoch + 1)
            writer.flush()
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(f"epoch {epoch + 1} done: loss {result.loss.get('loss_total', float('nan')):.4f}", flush=True)
    except KeyboardInterrupt:
        save_checkpoint(paths.checkpoints / "last.pt", **make_state(epoch - 1))
        raise
    finally:
        writer.close()
        log.close()
    return {"best_map": best_map, "best_val_loss": best_val_loss, "last": metrics, "work_dir": str(paths.root), "best": str(paths.checkpoints / "best_map50.pt")}
