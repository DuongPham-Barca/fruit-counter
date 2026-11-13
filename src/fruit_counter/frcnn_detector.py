"""Load the trained Faster R-CNN and convert frame predictions into detections."""

from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

from .labels import class_name
from .model import ModelConfig, build_model
from .train import load_checkpoint
from .tracker import Detection


def merge_cross_class(output: dict, iou: float) -> dict:
    """Suppress duplicate orange/apple boxes for the same physical fruit."""
    boxes, scores = output["boxes"].float(), output["scores"].float()
    keep = nms(boxes, scores, iou)
    return {
        "boxes": boxes[keep],
        "labels": output["labels"][keep],
        "scores": scores[keep],
    }


class FasterRCNNDetector:
    """Callable detector used by the video counting pipeline."""

    def __init__(
        self,
        weights: Path,
        device: str = "cuda",
        min_score: float = 0.1,
        amp: bool = True,
        cross_class_iou: float = 0.6,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; pass --device cpu")

        checkpoint = load_checkpoint(Path(weights), map_location="cpu")
        config = ModelConfig.from_dict(checkpoint["model_config"])
        config.box_score_thresh = min_score
        self.model = build_model(config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.amp = amp and self.device.type == "cuda"
        self.cross_class_iou = cross_class_iou

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> list[Detection]:
        image = (
            torch.from_numpy(frame)
            .to(self.device)
            .permute(2, 0, 1)
            .float()
            / 255
        )
        with torch.autocast(self.device.type, enabled=self.amp):
            output = self.model([image])[0]
        output = merge_cross_class(output, self.cross_class_iou)
        return [
            Detection(
                class_name(int(label)),
                tuple(float(value) for value in box),
                float(score),
            )
            for box, label, score in zip(
                output["boxes"].float().cpu(),
                output["labels"].cpu(),
                output["scores"].float().cpu(),
            )
        ]
