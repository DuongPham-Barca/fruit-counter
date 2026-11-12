"""Backward-compatible imports for the evaluation engine."""

from .engine.evaluator import (
    DetectionEvaluator,
    average_precision,
    evaluate_loss,
    evaluate_model,
)

__all__ = [
    "DetectionEvaluator",
    "average_precision",
    "evaluate_loss",
    "evaluate_model",
]
