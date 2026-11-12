"""Explicit Faster R-CNN architecture for orange/apple detection.

The project intentionally declares every trainable CNN block here so the model can be
read in the same order as a textbook diagram:

    image -> ResNet CNN -> FPN -> RPN -> RoIAlign -> box CNN -> class/box outputs

The final heads return logits. Faster R-CNN applies BCE-with-logits to RPN objectness,
cross-entropy to the mutually exclusive background/orange/apple classes, and softmax
during inference. Therefore an explicit sigmoid does not belong in the class head.
"""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .labels import CLASSES


def group_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    """GroupNorm is stable for the small batches used by object detection."""
    selected = min(groups, channels)
    if channels % selected:
        raise ValueError(f"group_norm_groups={groups} does not divide {channels} channels")
    return nn.GroupNorm(selected, channels)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet-18/34 block: two 3x3 convolutions with a residual connection."""

    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int, groups: int) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.norm1 = group_norm(channels, groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, out_channels)
        self.norm2 = group_norm(out_channels, groups)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                group_norm(out_channels, groups),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        x = self.relu(self.norm1(self.conv1(inputs)))
        x = self.norm2(self.conv2(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x + identity)


class Bottleneck(nn.Module):
    """ResNet-50 block: 1x1 -> 3x3 -> 1x1 convolutions plus a residual path."""

    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int, groups: int) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = conv1x1(in_channels, channels)
        self.norm1 = group_norm(channels, groups)
        self.conv2 = conv3x3(channels, channels, stride)
        self.norm2 = group_norm(channels, groups)
        self.conv3 = conv1x1(channels, out_channels)
        self.norm3 = group_norm(out_channels, groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                group_norm(out_channels, groups),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        x = self.relu(self.norm1(self.conv1(inputs)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x + identity)


BACKBONES = {
    "resnet18": (BasicBlock, (2, 2, 2, 2)),
    "resnet34": (BasicBlock, (3, 4, 6, 3)),
    "resnet50": (Bottleneck, (3, 4, 6, 3)),
}


class ResNetBackbone(nn.Module):
    """Feature-only ResNet with explicitly declared Conv -> GroupNorm -> ReLU layers."""

    def __init__(self, block, layer_depths: tuple[int, ...], groups: int = 32) -> None:
        super().__init__()
        self.groups = groups
        self.in_channels = 64

        # Stem: RGB image -> 64 feature channels.
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.norm1 = group_norm(64, groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four residual stages. Spatial size halves at the start of stages 2-4.
        self.layer1 = self._make_layer(block, 64, layer_depths[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layer_depths[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_depths[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_depths[3], stride=2)
        self.out_channels_by_stage = [
            64 * block.expansion,
            128 * block.expansion,
            256 * block.expansion,
            512 * block.expansion,
        ]
        self._initialize_weights()

    def _make_layer(self, block, channels: int, depth: int, stride: int) -> nn.Sequential:
        layers = [block(self.in_channels, channels, stride, self.groups)]
        self.in_channels = channels * block.expansion
        layers.extend(
            block(self.in_channels, channels, 1, self.groups)
            for _ in range(1, depth)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Start every residual branch close to the identity mapping.
        for module in self.modules():
            if isinstance(module, BasicBlock):
                nn.init.zeros_(module.norm2.weight)
            elif isinstance(module, Bottleneck):
                nn.init.zeros_(module.norm3.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.relu(self.norm1(self.conv1(inputs)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class FruitRPNHead(nn.Module):
    """Region Proposal Network: Conv -> ReLU -> objectness and box logits."""

    def __init__(self, channels: int, anchors_per_location: int, depth: int = 2) -> None:
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.extend([
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
        self.conv = nn.Sequential(*blocks)
        self.objectness_logits = nn.Conv2d(
            channels, anchors_per_location, kernel_size=1
        )
        self.box_deltas = nn.Conv2d(
            channels, anchors_per_location * 4, kernel_size=1
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, feature_maps: list[Tensor]):
        objectness, box_regression = [], []
        for feature in feature_maps:
            x = self.conv(feature)
            objectness.append(self.objectness_logits(x))
            box_regression.append(self.box_deltas(x))
        return objectness, box_regression


class FruitBoxHead(nn.Module):
    """Four Conv -> GroupNorm -> ReLU blocks followed by Linear -> ReLU."""

    def __init__(self, channels: int, roi_size: int, groups: int) -> None:
        super().__init__()
        blocks = []
        for _ in range(4):
            blocks.extend([
                nn.Conv2d(channels, 256, kernel_size=3, padding=1, bias=False),
                group_norm(256, groups),
                nn.ReLU(inplace=True),
            ])
            channels = 256
        self.convolutions = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Linear(256 * roi_size * roi_size, 1024)
        self.relu = nn.ReLU(inplace=True)
        for module in self.convolutions.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

    def forward(self, roi_features: Tensor) -> Tensor:
        x = self.convolutions(roi_features)
        x = self.flatten(x)
        return self.relu(self.fully_connected(x))


class FruitPredictor(nn.Module):
    """Output raw logits for three classes and class-specific box deltas."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.class_logits = nn.Linear(in_features, num_classes)
        self.box_deltas = nn.Linear(in_features, num_classes * 4)

    def forward(self, features: Tensor):
        features = features.flatten(start_dim=1)
        return self.class_logits(features), self.box_deltas(features)


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    num_classes: int = len(CLASSES) + 1
    min_size: tuple[int, ...] = (416, 448, 480, 512)
    max_size: int = 912
    box_batch_size_per_image: int = 256
    anchor_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    aspect_ratios: tuple[float, ...] = (0.75, 1.0, 1.33)
    group_norm_groups: int = 32
    fpn_channels: int = 256
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    detections_per_image: int = 100

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        values = dict(values)
        for key in ("min_size", "anchor_sizes", "aspect_ratios"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


def build_backbone(config: ModelConfig) -> BackboneWithFPN:
    """Build explicit ResNet stages and connect their outputs to an FPN."""
    if config.backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(BACKBONES)}")
    block, depths = BACKBONES[config.backbone]
    body = ResNetBackbone(block, depths, config.group_norm_groups)
    return BackboneWithFPN(
        body,
        return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
        in_channels_list=body.out_channels_by_stage,
        out_channels=config.fpn_channels,
        extra_blocks=LastLevelMaxPool(),
        norm_layer=lambda channels: group_norm(channels, config.group_norm_groups),
    )


def build_model(config: ModelConfig | None = None) -> FasterRCNN:
    """Assemble the complete Faster R-CNN detector from explicit CNN components."""
    config = config or ModelConfig()
    if len(config.anchor_sizes) != 5:
        raise ValueError("anchor_sizes needs one size per FPN level (5 values)")

    backbone = build_backbone(config)
    anchors = AnchorGenerator(
        sizes=tuple((size,) for size in config.anchor_sizes),
        aspect_ratios=(tuple(config.aspect_ratios),) * len(config.anchor_sizes),
    )
    roi_size = 7
    return FasterRCNN(
        backbone,
        min_size=config.min_size,
        max_size=config.max_size,
        rpn_anchor_generator=anchors,
        rpn_head=FruitRPNHead(
            backbone.out_channels,
            anchors.num_anchors_per_location()[0],
            depth=2,
        ),
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=roi_size,
            sampling_ratio=2,
        ),
        box_head=FruitBoxHead(
            backbone.out_channels, roi_size, config.group_norm_groups
        ),
        box_predictor=FruitPredictor(1024, config.num_classes),
        box_score_thresh=config.box_score_thresh,
        box_nms_thresh=config.box_nms_thresh,
        box_detections_per_img=config.detections_per_image,
        box_batch_size_per_image=config.box_batch_size_per_image,
    )
