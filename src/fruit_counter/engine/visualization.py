"""TensorBoard visualizations for validation predictions and metrics."""

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import draw_bounding_boxes

from ..labels import CLASSES

COLORS = {1: (255, 150, 0), 2: (230, 30, 60)}


def log_confusion_matrix(writer, confusion: dict, step: int) -> None:
    """Render a labeled detection confusion matrix into TensorBoard."""
    labels = confusion["labels"]
    matrix = np.asarray(confusion["matrix"], dtype=int)
    cell, label_width, header_height = 92, 112, 52
    width = label_width + cell * len(labels)
    height = header_height + cell * len(labels)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    maximum = max(1, int(matrix.max()))

    draw.text((4, 4), "GT / Pred", fill="black")
    for index, label in enumerate(labels):
        draw.text(
            (label_width + index * cell + 5, 18),
            label,
            fill="black",
        )
        draw.text((5, header_height + index * cell + 36), label, fill="black")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = int(matrix[row, column])
            intensity = int(235 - 180 * value / maximum)
            left = label_width + column * cell
            top = header_height + row * cell
            draw.rectangle(
                (left, top, left + cell, top + cell),
                fill=(intensity, intensity, 255),
                outline=(80, 80, 80),
            )
            draw.text((left + 38, top + 36), str(value), fill="black")

    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    writer.add_image("val/confusion_matrix", tensor, step)


@torch.inference_mode()
def log_predictions(
    writer,
    model,
    dataset,
    count: int,
    device,
    step: int,
    min_score: float = 0.5,
) -> None:
    model.eval()
    for index in range(min(count, len(dataset))):
        image, target = dataset[index]
        output = model([image.to(device)])[0]
        keep = output["scores"] >= min_score
        canvas = (image * 255).to(torch.uint8)
        if len(target["boxes"]):
            canvas = draw_bounding_boxes(canvas, target["boxes"], colors="white", width=1)
        boxes = output["boxes"][keep].float().cpu()
        if len(boxes):
            labels = output["labels"][keep].cpu().tolist()
            scores = output["scores"][keep].float().cpu().tolist()
            canvas = draw_bounding_boxes(
                canvas,
                boxes,
                width=2,
                labels=[
                    f"{CLASSES[label - 1]} {score:.2f}"
                    for label, score in zip(labels, scores)
                ],
                colors=[COLORS[label] for label in labels],
            )
        writer.add_image(f"val_predictions/{index}", canvas, step)
