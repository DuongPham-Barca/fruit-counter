"""Validation loss, AP, precision, recall, and confusion-matrix evaluation."""

from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import box_iou

from .labels import CLASSES

__all__ = [
    "DetectionEvaluator",
    "average_precision",
    "evaluate_loss",
    "evaluate_model",
]


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the all-point interpolated precision-recall curve."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.nonzero(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1])
    )


class DetectionEvaluator:
    """Accumulate detection metrics at fixed IoU and confidence thresholds.

    The confusion matrix uses rows as ground-truth classes and columns as predicted
    classes. Index 0 is background, so missed objects appear in column 0 and false
    positive detections appear in row 0.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self._scores = defaultdict(list)
        self._matches = defaultdict(list)
        self._ground_truth = defaultdict(int)
        size = len(CLASSES) + 1
        self._confusion = np.zeros((size, size), dtype=np.int64)

    def _update_confusion(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        true_boxes: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> None:
        keep = pred_scores >= self.score_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        used_truth = torch.zeros(len(true_boxes), dtype=torch.bool)
        ious = (
            box_iou(pred_boxes, true_boxes)
            if len(pred_boxes) and len(true_boxes)
            else None
        )
        for pred_index, pred_label in enumerate(pred_labels):
            matched_truth = None
            if ious is not None:
                candidates = ious[pred_index].clone()
                candidates[used_truth] = -1
                best = int(torch.argmax(candidates))
                if candidates[best] >= self.iou_threshold:
                    matched_truth = best
            if matched_truth is None:
                self._confusion[0, int(pred_label)] += 1
            else:
                used_truth[matched_truth] = True
                self._confusion[
                    int(true_labels[matched_truth]), int(pred_label)
                ] += 1

        for truth_index, true_label in enumerate(true_labels):
            if not used_truth[truth_index]:
                self._confusion[int(true_label), 0] += 1

    def add(self, prediction: dict, target: dict) -> None:
        pred_boxes = prediction["boxes"].detach().float().cpu()
        pred_scores = prediction["scores"].detach().float().cpu()
        pred_labels = prediction["labels"].detach().cpu()
        true_boxes = target["boxes"].detach().float().cpu()
        true_labels = target["labels"].detach().cpu()

        self._update_confusion(
            pred_boxes, pred_scores, pred_labels, true_boxes, true_labels
        )

        # AP uses all model predictions, ranked globally by confidence.
        for label in range(1, len(CLASSES) + 1):
            truth = true_boxes[true_labels == label]
            self._ground_truth[label] += len(truth)
            keep = pred_labels == label
            boxes, scores = pred_boxes[keep], pred_scores[keep]
            order = torch.argsort(scores, descending=True)
            boxes, scores = boxes[order], scores[order]
            used = torch.zeros(len(truth), dtype=torch.bool)
            ious = box_iou(boxes, truth) if len(truth) and len(boxes) else None
            for index in range(len(boxes)):
                matched = False
                if ious is not None:
                    candidates = ious[index].clone()
                    candidates[used] = -1
                    best = int(torch.argmax(candidates))
                    if candidates[best] >= self.iou_threshold:
                        used[best] = True
                        matched = True
                self._scores[label].append(float(scores[index]))
                self._matches[label].append(matched)

    def summary(self) -> dict:
        per_class = {}
        macro_precision, macro_recall = [], []
        for label, name in enumerate(CLASSES, start=1):
            total = self._ground_truth[label]
            scores = np.asarray(self._scores[label])
            matches = np.asarray(self._matches[label], dtype=bool)
            matches = matches[np.argsort(-scores, kind="stable")]
            true_positive_curve = np.cumsum(matches)
            false_positive_curve = np.cumsum(~matches)
            if total == 0:
                ap = max_recall = float("nan")
            else:
                recall_curve = true_positive_curve / total
                precision_curve = true_positive_curve / np.maximum(
                    true_positive_curve + false_positive_curve, 1
                )
                ap = (
                    average_precision(recall_curve, precision_curve)
                    if len(matches)
                    else 0.0
                )
                max_recall = float(recall_curve[-1]) if len(matches) else 0.0

            true_positive = int(self._confusion[label, label])
            false_positive = int(self._confusion[:, label].sum() - true_positive)
            false_negative = int(self._confusion[label, :].sum() - true_positive)
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            macro_precision.append(precision)
            macro_recall.append(recall)
            per_class[name] = {
                "ap": ap,
                "ground_truth": int(total),
                "predictions": int(len(matches)),
                "max_recall": max_recall,
                "precision": precision,
                "recall": recall,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }

        values = [
            item["ap"]
            for item in per_class.values()
            if not np.isnan(item["ap"])
        ]
        return {
            "iou_threshold": self.iou_threshold,
            "score_threshold": self.score_threshold,
            "map": float(np.mean(values)) if values else float("nan"),
            "precision": float(np.mean(macro_precision)),
            "recall": float(np.mean(macro_recall)),
            "classes": per_class,
            "confusion_matrix": {
                "labels": ["background", *CLASSES],
                "rows": "ground_truth",
                "columns": "prediction",
                "matrix": self._confusion.tolist(),
            },
        }


@torch.inference_mode()
def evaluate_model(model, loader, device: torch.device, amp: bool = True) -> dict:
    """Run inference and calculate detection-quality metrics."""
    model.eval()
    evaluator = DetectionEvaluator()
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        with torch.autocast(
            device.type, enabled=amp and device.type == "cuda"
        ):
            outputs = model(images)
        for output, target in zip(outputs, targets):
            evaluator.add(output, target)
    return evaluator.summary()


@torch.no_grad()
def evaluate_loss(model, loader, device: torch.device, amp: bool = True) -> dict:
    """Calculate validation detection losses without changing training RNG state.

    TorchVision detection models return losses only in training mode. This project
    uses GroupNorm and no Dropout, so switching to training mode here does not update
    running normalization statistics. ``fork_rng`` keeps validation sampling from
    changing the following training epoch.
    """
    was_training = model.training
    devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    totals = defaultdict(float)
    batches = 0
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(0)
            model.train()
            for images, targets in loader:
                images = [image.to(device, non_blocking=True) for image in images]
                targets = [
                    {
                        "boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device),
                    }
                    for target in targets
                ]
                with torch.autocast(
                    device.type, enabled=amp and device.type == "cuda"
                ):
                    losses = model(images, targets)
                    total_loss = sum(losses.values())
                for name, value in {**losses, "loss_total": total_loss}.items():
                    totals[name] += float(value)
                batches += 1
    finally:
        model.train(was_training)
    return {name: value / max(batches, 1) for name, value in totals.items()}
