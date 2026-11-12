#!/usr/bin/env python
"""Evaluate a trained checkpoint without requiring an experiment config file."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fruit_counter.dataset import YoloDetectionDataset, collate
from fruit_counter.evaluate import evaluate_model
from fruit_counter.frcnn_detector import FasterRCNNDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fruit detector")
    parser.add_argument("--data", type=Path, default=Path("data/datasets/coco-fruit"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    detector = FasterRCNNDetector(args.weights, args.device, min_score=0.05)
    dataset = YoloDetectionDataset(args.data, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    metrics = evaluate_model(
        detector.model, loader, torch.device(args.device), amp=True
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
