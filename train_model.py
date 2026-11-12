#!/usr/bin/env python
"""Readable training entrypoint for the fruit detector.

Edit the configuration block below or pass command-line arguments. The CNN itself is
declared layer by layer in src/fruit_counter/model.py; this file keeps the training
flow in the same order used by the brain-tumor notebook.
"""

import argparse
import json
from pathlib import Path

from fruit_counter.model import ModelConfig
from fruit_counter.train import TrainConfig, train


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path("data/datasets/coco-fruit")
OUTPUT_DIR = Path("outputs/frcnn_r50_coco_scratch")
BACKBONE = "resnet50"
EPOCHS = 100
BATCH_SIZE = 2
ACCUMULATE_GRAD_BATCHES = 4
LEARNING_RATE = 0.01
NUM_WORKERS = 4
DEVICE = "cuda"
USE_AMP = True
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN fruit detector")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--backbone", choices=("resnet18", "resnet34", "resnet50"), default=BACKBONE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accumulate", type=int, default=ACCUMULATE_GRAD_BATCHES)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ========================================================
    # 1. MODEL
    # ResNet CNN -> FPN -> RPN -> RoI box head.
    # All weights are randomly initialized; no pretrained model.
    # ========================================================
    model_config = ModelConfig(backbone=args.backbone)

    # ========================================================
    # 2. DATA + OPTIMIZER + TRAINING SETTINGS
    # Effective batch = batch_size * gradient accumulation.
    # ========================================================
    train_config = TrainConfig(
        data=str(args.data),
        output=str(args.output),
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate=args.accumulate,
        lr=args.lr,
        workers=args.workers,
        seed=SEED,
        device=args.device,
        amp=USE_AMP and not args.no_amp,
    )

    print("=" * 64)
    print("FRUIT DETECTOR TRAINING")
    print(f"Model          : Faster R-CNN + {args.backbone} + FPN")
    print("Classes        : background, orange, apple")
    print(f"Dataset        : {args.data}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch          : {args.batch_size} x {args.accumulate} accumulation")
    print(f"Learning rate  : {args.lr}")
    print("Optimizer      : SGD (momentum=0.9, weight_decay=1e-4)")
    print("Scheduler      : Linear warmup + cosine decay")
    print(f"Device / AMP   : {args.device} / {train_config.amp}")
    print("RPN loss       : BCEWithLogits + SmoothL1")
    print("Class loss     : CrossEntropy (Softmax during inference)")
    print("Box loss       : SmoothL1")
    print("Validation     : loss + mAP50 + precision/recall + confusion matrix")
    print("=" * 64)

    # ========================================================
    # 3. TRAIN -> VALIDATE -> SAVE BEST/LAST CHECKPOINTS
    # ========================================================
    result = train(
        train_config,
        model_config,
        Path(args.resume) if args.resume else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
