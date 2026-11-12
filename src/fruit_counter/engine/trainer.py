"""Orchestrate detector training, evaluation, logging, and checkpoints."""

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..data.loaders import build_loaders
from ..labels import CLASSES
from ..model import ModelConfig, build_model
from .artifacts import RunPaths, write_run_metadata
from .checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from .config import TrainConfig
from .evaluator import evaluate_loss, evaluate_model
from .loop import train_one_epoch
from .schedule import parameter_groups
from .visualization import log_confusion_matrix, log_predictions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resume_path(path: Path, paths: RunPaths) -> Path:
    if str(path) == "auto":
        return paths.checkpoints / "last.pt"
    return path


def _validate_resume_config(previous: dict | None, current: TrainConfig) -> None:
    if not previous:
        return
    immutable = (
        "data",
        "batch_size",
        "accumulate",
        "lr",
        "momentum",
        "weight_decay",
        "warmup_steps",
        "min_lr_ratio",
        "max_grad_norm",
        "train_split",
        "val_split",
        "seed",
    )
    changed = [
        name
        for name in immutable
        if name in previous and previous[name] != getattr(current, name)
    ]
    if changed:
        details = ", ".join(
            f"{name}: {previous[name]!r} -> {getattr(current, name)!r}"
            for name in changed
        )
        raise ValueError(f"Resume would change training state: {details}")


def train(
    config: TrainConfig,
    model_config: ModelConfig | None = None,
    resume: Path | None = None,
    *,
    resolved_config: dict | None = None,
    config_source: Path | None = None,
) -> dict:
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use a CPU config or override runtime.device")

    paths = RunPaths.create(Path(config.output))
    if resume is None and (paths.checkpoints / "last.pt").exists():
        raise ValueError(
            f"{paths.checkpoints / 'last.pt'} already exists; use --resume auto "
            "or choose another output directory"
        )
    resume = _resume_path(resume, paths) if resume is not None else None
    state = load_checkpoint(resume, map_location="cpu") if resume else None
    if state is not None:
        _validate_resume_config(state.get("train_config"), config)
        checkpoint_model_config = ModelConfig.from_dict(state["model_config"])
        if model_config is not None and model_config != checkpoint_model_config:
            raise ValueError("Resume config does not match the checkpoint model config")
        model_config = checkpoint_model_config
    model_config = model_config or ModelConfig(backbone=config.backbone)

    set_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.SGD(
        parameter_groups(model, config.weight_decay),
        lr=config.base_lr,
        momentum=config.momentum,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.amp and device.type == "cuda",
    )
    train_set, val_set, train_loader, val_loader = build_loaders(
        config,
        device,
        generator,
    )

    batches_per_epoch = len(train_loader)
    if config.limit_batches:
        batches_per_epoch = min(batches_per_epoch, config.limit_batches)
    steps_per_epoch = math.ceil(batches_per_epoch / config.accumulate)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = min(config.warmup_steps, max(1, total_steps // 10))

    start_epoch, step, best_map, best_val_loss = 0, 0, -1.0, float("inf")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        step = state["step"]
        best_map = state["best_map"]
        best_val_loss = state.get("best_val_loss", float("inf"))
        restore_rng_state(state.get("rng_state"), generator)
        print(
            f"resumed from {resume}: epoch {start_epoch}, step {step}, "
            f"best mAP50 {best_map:.4f}, best val loss {best_val_loss:.4f}",
            flush=True,
        )
        del state

    write_run_metadata(
        paths,
        config,
        model_config.to_dict(),
        resolved_config,
        config_source,
    )
    writer = SummaryWriter(
        str(paths.logs / "tensorboard"),
        purge_step=step if step else None,
    )
    if step == 0:
        writer.add_text(
            "config",
            json.dumps(resolved_config or asdict(config), indent=2),
            0,
        )
    log = (paths.logs / "metrics.jsonl").open("a", encoding="utf-8")

    def make_state(epoch: int) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model_config": model_config.to_dict(),
            "train_config": asdict(config),
            "classes": list(CLASSES),
            "epoch": epoch,
            "step": step,
            "best_map": best_map,
            "best_val_loss": best_val_loss,
            "rng_state": capture_rng_state(generator),
        }

    print(
        f"training {model_config.backbone} from random init: "
        f"{len(train_set)} train / {len(val_set)} val images\n"
        f"work dir: {paths.root}\n"
        f"{steps_per_epoch} steps x {config.epochs} epochs, lr {config.base_lr:g}",
        flush=True,
    )
    metrics = {}
    epoch = start_epoch
    training_started = time.time()
    try:
        for epoch in range(start_epoch, config.epochs):
            result = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                config=config,
                epoch=epoch,
                step=step,
                batches_per_epoch=batches_per_epoch,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                writer=writer,
                training_started=training_started,
            )
            step = result.step
            metrics = {
                "epoch": epoch,
                "step": step,
                "lr": result.lr,
                "seconds": result.seconds,
                "loss": result.loss,
            }
            if result.gpu_peak_memory_gb is not None:
                metrics["gpu_peak_memory_gb"] = result.gpu_peak_memory_gb
            for name, value in result.loss.items():
                writer.add_scalar(f"epoch/{name}", value, epoch + 1)

            if (epoch + 1) % config.eval_every == 0 or epoch + 1 == config.epochs:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                metrics["val_loss"] = evaluate_loss(
                    model, val_loader, device, config.amp
                )
                metrics["val"] = evaluate_model(model, val_loader, device, config.amp)
                val_loss = metrics["val_loss"]
                val = metrics["val"]
                for loss_name, value in val_loss.items():
                    writer.add_scalar(f"val/{loss_name}", value, epoch + 1)
                writer.add_scalar("val/mAP50", val["map"], epoch + 1)
                writer.add_scalar("val/precision_macro", val["precision"], epoch + 1)
                writer.add_scalar("val/recall_macro", val["recall"], epoch + 1)
                for name, item in val["classes"].items():
                    writer.add_scalar(f"val/AP50_{name}", item["ap"], epoch + 1)
                    writer.add_scalar(
                        f"val/precision_{name}", item["precision"], epoch + 1
                    )
                    writer.add_scalar(
                        f"val/recall_{name}", item["recall"], epoch + 1
                    )
                    writer.add_scalar(
                        f"val/max_recall_{name}",
                        item["max_recall"],
                        epoch + 1,
                    )
                log_confusion_matrix(writer, val["confusion_matrix"], epoch + 1)
                (paths.evaluation / f"val_epoch_{epoch + 1:03d}.json").write_text(
                    json.dumps({"loss": val_loss, "metrics": val}, indent=2) + "\n",
                    encoding="utf-8",
                )
                if config.vis_images:
                    log_predictions(
                        writer,
                        model,
                        val_set,
                        config.vis_images,
                        device,
                        epoch + 1,
                    )

            val_map = metrics.get("val", {}).get("map")
            improved = (
                val_map is not None
                and not math.isnan(val_map)
                and val_map > best_map
            )
            val_loss_total = metrics.get("val_loss", {}).get("loss_total")
            loss_improved = (
                val_loss_total is not None
                and math.isfinite(val_loss_total)
                and val_loss_total < best_val_loss
            )
            if improved:
                best_map = val_map
                inference = {
                    key: value
                    for key, value in make_state(epoch).items()
                    if key not in ("optimizer", "scaler", "rng_state")
                }
                save_checkpoint(paths.checkpoints / "best_map50.pt", **inference)
            if loss_improved:
                best_val_loss = val_loss_total
                inference = {
                    key: value
                    for key, value in make_state(epoch).items()
                    if key not in ("optimizer", "scaler", "rng_state")
                }
                save_checkpoint(paths.checkpoints / "best_val_loss.pt", **inference)
            checkpoint = make_state(epoch)
            save_checkpoint(paths.checkpoints / "last.pt", **checkpoint)
            if config.save_every and (epoch + 1) % config.save_every == 0:
                save_checkpoint(
                    paths.checkpoints / f"epoch_{epoch + 1:03d}.pt",
                    **checkpoint,
                )
            metrics["best_map"] = best_map
            metrics["best_val_loss"] = (
                best_val_loss if math.isfinite(best_val_loss) else None
            )
            writer.add_scalar("val/best_mAP50", max(best_map, 0.0), epoch + 1)
            if math.isfinite(best_val_loss):
                writer.add_scalar("val/best_loss", best_val_loss, epoch + 1)
            writer.flush()
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            summary = (
                f"epoch {epoch + 1} done in {result.seconds:.0f}s, "
                f"loss {result.loss.get('loss_total', float('nan')):.4f}"
            )
            if val_map is not None:
                summary += (
                    f", val loss {metrics['val_loss']['loss_total']:.4f}"
                    f", val mAP50 {val_map:.4f}"
                )
                if improved:
                    summary += " (new best)"
                if loss_improved:
                    summary += " (new best val loss)"
            print(summary, flush=True)
    except KeyboardInterrupt:
        save_checkpoint(paths.checkpoints / "last.pt", **make_state(epoch - 1))
        print(
            f"\ninterrupted: saved {paths.checkpoints / 'last.pt'}",
            flush=True,
        )
        raise
    finally:
        writer.close()
        log.close()
    return {
        "best_map": best_map,
        "best_val_loss": best_val_loss,
        "last": metrics,
        "work_dir": str(paths.root),
        "best": str(paths.checkpoints / "best_map50.pt"),
    }
