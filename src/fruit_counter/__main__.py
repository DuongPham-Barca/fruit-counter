"""Command line interface: prepare data, train, evaluate, and count fruit."""

import argparse
import json
import os
from pathlib import Path

# Reduces CUDA memory fragmentation, which matters on a 4 GB GPU. Set before torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _add_count(commands) -> None:
    count = commands.add_parser("count", help="count fruit crossing a line in a video")
    count.add_argument("--input", type=Path, required=True)
    count.add_argument("--output", type=Path, default=Path("runs/count"))
    count.add_argument(
        "--detector",
        choices=("fruit-counter-net", "color-demo"),
        default="fruit-counter-net",
    )
    count.add_argument("--weights", type=Path, help="FruitCounterNet checkpoint (best_map50.pt)")
    count.add_argument("--device", default="cuda")
    count.add_argument("--min-score", type=float, default=0.1,
                       help="lowest detection score passed to the tracker")
    count.add_argument("--width", type=int, default=640, help="processing width in pixels")
    count.add_argument("--line", type=float, default=0.5, help="count line at this fraction of the width")
    count.add_argument("--direction", choices=("ltr", "rtl"), default="ltr")


def _add_train(commands) -> None:
    train = commands.add_parser("train", help="train FruitCounterNet from random initialization")
    train.add_argument("--data", type=Path, required=True, help="YOLO-format dataset root")
    train.add_argument("--output", type=Path, default=Path("runs/train"))
    train.add_argument("--backbone", choices=("resnet18", "resnet34", "resnet50"), default="resnet50")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--accumulate", type=int, default=4)
    train.add_argument("--lr", type=float)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--eval-every", type=int, default=1, help="evaluate on val every N epochs")
    train.add_argument("--save-every", type=int, default=0,
                       help="also keep epoch_NNN.pt every N epochs (0: last and best only)")
    train.add_argument("--log-every", type=int, default=20, help="TensorBoard scalars every N steps")
    train.add_argument("--vis-images", type=int, default=4,
                       help="val images drawn with predictions in TensorBoard")
    train.add_argument("--min-size", type=int, nargs="+",
                       help="shorter image side(s) in px; training samples one, inference uses the last "
                            "(default 416 448 480 512)")
    train.add_argument("--max-size", type=int, help="longer image side cap in px (default 912)")
    train.add_argument("--limit-batches", type=int, help="batches per epoch, for smoke tests")
    train.add_argument("--resume", type=Path, help="continue from a last.pt checkpoint")


def _add_evaluate(commands) -> None:
    evaluate = commands.add_parser("evaluate", help="report per-class AP@0.5 on a split")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--device", default="cuda")


def _add_prepare(commands) -> None:
    prepare = commands.add_parser("prepare-coco", help="build the COCO orange/apple dataset")
    prepare.add_argument("--output", type=Path, default=Path("data/datasets/coco-fruit"))
    prepare.add_argument("--cache", type=Path, default=Path("data/cache/coco"))
    prepare.add_argument("--val-fraction", type=float, default=0.1)


def _run(args) -> object:
    if args.command == "demo":
        from .demo import make_demo

        make_demo(args.output)
        return str(args.output)
    if args.command == "prepare-coco":
        from .prepare_coco import prepare

        return prepare(args.output, args.cache, args.val_fraction)
    if args.command == "train":
        from .model import ModelConfig
        from .train import TrainConfig, train

        model_config = ModelConfig(backbone=args.backbone)
        if args.min_size:
            model_config.min_size = tuple(args.min_size)
        if args.max_size:
            model_config.max_size = args.max_size
        config = TrainConfig(
            data=str(args.data), output=str(args.output), backbone=args.backbone,
            epochs=args.epochs, batch_size=args.batch_size, accumulate=args.accumulate,
            lr=args.lr, workers=args.workers, seed=args.seed, device=args.device,
            amp=not args.no_amp, eval_every=args.eval_every, save_every=args.save_every,
            log_every=args.log_every, vis_images=args.vis_images, limit_batches=args.limit_batches,
        )
        result = train(config, model_config, resume=args.resume)
        return {"best_map": result["best_map"], "best": result["best"]}
    if args.command == "evaluate":
        from torch.utils.data import DataLoader

        from .dataset import YoloDetectionDataset, collate
        from .evaluate import evaluate_model
        from .fruit_counter_net import FruitCounterNetDetector

        detector = FruitCounterNetDetector(args.weights, args.device, min_score=0.05)
        loader = DataLoader(YoloDetectionDataset(args.data, args.split), batch_size=2,
                            collate_fn=collate)
        return evaluate_model(detector.model, loader, detector.device)

    from .pipeline import count_video

    if args.detector == "color-demo":
        result = count_video(args.input, args.output, args.width, args.line, args.direction)
    else:
        if args.weights is None:
            raise ValueError("--weights is required with --detector fruit-counter-net")
        from .fruit_counter_net import FruitCounterNetDetector

        detector = FruitCounterNetDetector(args.weights, args.device, min_score=args.min_score)
        result = count_video(args.input, args.output, args.width, args.line, args.direction,
                             detector=detector, method=f"fruit_counter_net:{args.weights}")
    return {"counts": result["counts"], "total": result["total"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Count oranges and apples in conveyor videos")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="create a synthetic conveyor video")
    demo.add_argument("--output", type=Path, default=Path("runs/demo.mp4"))
    _add_prepare(commands)
    _add_train(commands)
    _add_evaluate(commands)
    _add_count(commands)
    args = parser.parse_args()
    try:
        result = _run(args)
    except KeyboardInterrupt:
        parser.exit(130, "stopped\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(result if isinstance(result, str) else json.dumps(result))


if __name__ == "__main__":
    main()
