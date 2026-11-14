"""Class names shared by datasets, the detector, and the counter."""

CLASSES = ("orange", "apple")
"""Dataset class ids 0 and 1. The detector reserves label 0 for background."""


def model_label(class_id: int) -> int:
    """Convert a dataset class id to a FruitCounterNet label (background is 0)."""
    if not 0 <= class_id < len(CLASSES):
        raise ValueError(f"class id {class_id} is outside 0..{len(CLASSES) - 1}")
    return class_id + 1


def class_name(label: int) -> str:
    """Convert a FruitCounterNet label back to a class name."""
    if not 1 <= label <= len(CLASSES):
        raise ValueError(f"label {label} is not a fruit class")
    return CLASSES[label - 1]
