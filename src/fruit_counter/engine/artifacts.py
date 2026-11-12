"""Run-directory layout and immutable experiment metadata."""

import json
import platform
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision

from .config import TrainConfig


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


def write_run_metadata(
    paths: RunPaths,
    train_config: TrainConfig,
    model_config: dict,
    resolved_config: dict | None,
    config_source: Path | None,
) -> None:
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
