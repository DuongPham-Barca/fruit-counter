import unittest
from unittest import mock

import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from fruit_counter.frcnn_detector import merge_cross_class
from fruit_counter.model import (
    FruitBoxHead,
    FruitPredictor,
    FruitRPNHead,
    ModelConfig,
    build_model,
)


def small_config(**overrides) -> ModelConfig:
    values = {"backbone": "resnet18", "min_size": (128,), "max_size": 160,
              "anchor_sizes": (8, 16, 32, 64, 128)}
    values.update(overrides)
    return ModelConfig(**values)


def no_download(*args, **kwargs):
    raise AssertionError("a pretrained weight download was attempted")


class ModelTests(unittest.TestCase):
    def test_architecture_uses_explicit_project_cnn_heads(self):
        model = build_model(small_config())
        self.assertIsInstance(model.backbone.body, torch.nn.Module)
        self.assertIsInstance(model.rpn.head, FruitRPNHead)
        self.assertIsInstance(model.roi_heads.box_head, FruitBoxHead)
        self.assertIsInstance(model.roi_heads.box_predictor, FruitPredictor)
        self.assertTrue(any(isinstance(m, nn.ReLU) for m in model.modules()))

    def test_builds_from_random_init_without_downloads_or_frozen_layers(self):
        for backbone in ("resnet18", "resnet50"):
            with mock.patch("torch.hub.load_state_dict_from_url", no_download), \
                 mock.patch("torchvision.models._api.load_state_dict_from_url", no_download):
                model = build_model(small_config(backbone=backbone))
            modules = list(model.modules())
            self.assertFalse(any(isinstance(m, (nn.BatchNorm2d, FrozenBatchNorm2d)) for m in modules))
            self.assertTrue(any(isinstance(m, nn.GroupNorm) for m in modules))
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_two_models_with_different_seeds_differ(self):
        torch.manual_seed(0)
        first = build_model(small_config()).state_dict()
        torch.manual_seed(1)
        second = build_model(small_config()).state_dict()
        name = "backbone.body.conv1.weight"
        self.assertFalse(torch.equal(first[name], second[name]))

    def test_training_and_inference_outputs(self):
        torch.manual_seed(0)
        model = build_model(small_config())
        image = torch.rand(3, 120, 160)
        target = {"boxes": torch.tensor([[20.0, 20, 60, 60]]), "labels": torch.tensor([2])}
        model.train()
        losses = model([image], [target])
        self.assertEqual(
            set(losses),
            {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"},
        )
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        empty = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}
        self.assertTrue(all(torch.isfinite(v) for v in model([image], [empty]).values()))
        model.eval()
        with torch.no_grad():
            output = model([image])[0]
        self.assertEqual(set(output), {"boxes", "labels", "scores"})

    def test_overfits_one_image(self):
        torch.manual_seed(0)
        model = build_model(small_config())
        image = torch.zeros(3, 128, 160)
        image[0, 30:70, 30:70] = 1.0
        image[1, 30:70, 30:70] = 0.55
        target = {"boxes": torch.tensor([[30.0, 30, 70, 70]]), "labels": torch.tensor([1])}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()
        history = []
        for _ in range(40):
            loss = sum(model([image], [target]).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        self.assertLess(sum(history[-5:]) / 5, 0.5 * sum(history[:5]) / 5)

    def test_config_round_trip_and_validation(self):
        config = small_config()
        self.assertEqual(ModelConfig.from_dict(config.to_dict()), config)
        with self.assertRaises(ValueError):
            build_model(small_config(backbone="vgg"))
        with self.assertRaises(ValueError):
            build_model(small_config(anchor_sizes=(16, 32)))


    def test_cross_class_duplicates_are_merged(self):
        output = {
            "boxes": torch.tensor([[0.0, 0, 10, 10], [0.5, 0, 10, 10], [30, 30, 40, 40]]),
            "labels": torch.tensor([1, 2, 2]),
            "scores": torch.tensor([0.7, 0.9, 0.8]),
        }
        merged = merge_cross_class(output, 0.6)
        self.assertEqual(merged["labels"].tolist(), [2, 2])
        self.assertEqual(merged["scores"].tolist(), [0.8999999761581421, 0.800000011920929])


if __name__ == "__main__":
    unittest.main()
