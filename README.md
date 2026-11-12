# Fruit Counter

Full Vietnamese project documentation: [TAI_LIEU_DU_AN.md](TAI_LIEU_DU_AN.md).

Train a Faster R-CNN detector from random initialization, track oranges and apples in
prerecorded conveyor video, and count confirmed tracks crossing a configured line.

Pipeline: **Faster R-CNN → ByteTrack-style tracker → line-crossing counter**.

## Project layout

```text
train_model.py                   readable training entrypoint and settings
evaluate_model.py                evaluate one trained checkpoint
src/fruit_counter/
├── model.py                     explicit CNN, RPN and box-head layers
├── dataset.py                   YOLO parsing and augmentation
├── train.py                     public training API
├── evaluate.py                  AP@0.5 metrics
├── tracker.py                   tracking and line crossing
├── pipeline.py                  end-to-end video pipeline
├── video.py                     FFmpeg I/O
└── engine/                      checkpoint/logging implementation details
tools/                           dataset preparation and video-count utilities
tests/                           unit and integration tests
```

Start with `model.py`, `dataset.py`, and `train_model.py`. They expose the same
Dataset → CNN → loss/optimizer → train/validate flow used by the brain-tumor project.
The `engine/` package only contains reusable checkpoint and logging details.

## Setup

Python 3.10+, FFmpeg/FFprobe and a CUDA build of PyTorch are recommended.

```sh
source .venv/bin/activate
python -m pip install -e . --no-build-isolation
python -c "import torch; print(torch.cuda.is_available())"
```

## Dataset

Data uses YOLO text layout:

```text
<dataset>/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each label is `class cx cy width height`, normalized to 0..1. Class 0 is orange
and class 1 is apple. A missing label file represents a background image.

Build the initial COCO-derived dataset:

```sh
PYTHONPATH=src python tools/prepare_coco.py --output data/datasets/coco-fruit
```

COCO is only a starting point. Accuracy for the real application must be measured on
conveyor footage split by recording session, not randomly by individual frames.

## Model architecture

The complete architecture is declared in `src/fruit_counter/model.py`; convolution,
GroupNorm, ReLU, residual connections, RPN outputs and classification outputs are
visible directly in project source rather than hidden inside a TorchVision ResNet
factory.

```text
RGB image
  → ResNet: Conv2d → GroupNorm → ReLU → residual stages
  → FPN multi-scale features
  → RPN: Conv2d → ReLU → objectness logits + box deltas
  → RoIAlign
  → 4 × (Conv2d → GroupNorm → ReLU)
  → Linear → ReLU
  → class logits (CrossEntropy / inference Softmax) + box deltas (SmoothL1)
```

RPN objectness uses BCE-with-logits, which includes the sigmoid operation internally.
The orange/apple/background classes are mutually exclusive, so the final class head
uses Softmax rather than independent sigmoid outputs.

Every detector weight starts from random initialization. No pretrained backbone or
detector checkpoint is downloaded.

## Train

Open the configuration block at the top of `train_model.py`, then run:

```sh
PYTHONPATH=src python train_model.py
```

Settings can also be overridden from the command line:

```sh
PYTHONPATH=src python train_model.py --epochs 50 --workers 2 --backbone resnet34
```

Use a custom work directory:

```sh
PYTHONPATH=src python train_model.py --output outputs/experiment_001
```

Monitor training:

```sh
tensorboard --logdir outputs/frcnn_r50_coco_scratch/logs/tensorboard
```

TensorBoard records training loss components, validation loss components, mAP50,
per-class precision/recall, macro precision/recall, a background/orange/apple confusion
matrix, learning rate, gradient norm, GPU memory, speed, and annotated validation images.
Precision, recall, and the confusion matrix use confidence ≥ 0.5 and IoU ≥ 0.5.

Resume after Ctrl+C. Omitting the checkpoint value selects
`checkpoints/last.pt` inside the work directory:

```sh
PYTHONPATH=src python train_model.py --resume
```

Resume restores model, optimizer, AMP scaler, Python/NumPy/PyTorch RNG and data-loader
generator state. Changes to data, batch size, accumulation, optimizer or seed are
rejected instead of silently changing the experiment.

## Run artifacts

```text
outputs/frcnn_r50_coco_scratch/
├── run_metadata.json
├── checkpoints/
│   ├── best_map50.pt
│   ├── best_val_loss.pt
│   ├── last.pt
│   └── epoch_NNN.pt
├── logs/
│   ├── metrics.jsonl
│   └── tensorboard/
└── evaluation/
    └── val_epoch_NNN.json
```

Run settings and dependency versions are recorded in `run_metadata.json`. Resume
metadata is stored separately, so the original run information is not overwritten.

## Evaluate

Evaluate the best checkpoint on the held-out test split:

```sh
PYTHONPATH=src python evaluate_model.py \
  --weights outputs/frcnn_r50_coco_scratch/checkpoints/best_map50.pt \
  --split test
```

The current evaluator reports AP@0.5 and maximum recall by class. A COCO
mAP@[0.50:0.95] evaluator should be added before publishing benchmark results.

## Count a video

```sh
PYTHONPATH=src python tools/count_video.py video.mp4 \
  outputs/frcnn_r50_coco_scratch/checkpoints/best_map50.pt \
  --output outputs/count
```

The output contains `annotated.mp4`, `summary.json`, and `events.csv`.

The synthetic demo remains available for checking the video/tracker path without a
trained model:

```sh
fruit-counter demo --output runs/demo.mp4
fruit-counter count --detector color-demo --input runs/demo.mp4 --output runs/demo-count
```

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests cover config inheritance and overrides, random model initialization, training and
resume artifacts, YOLO parsing and augmentation, AP calculation, tracker association,
line crossing, and the end-to-end synthetic video pipeline.
