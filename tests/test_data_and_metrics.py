import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fruit_counter.dataset import YoloDetectionDataset, read_yolo_labels, train_transforms
from fruit_counter.evaluate import DetectionEvaluator, average_precision
from fruit_counter.labels import class_name, model_label
from fruit_counter.prepare_coco import select_images, yolo_lines
from fruit_counter.train import lr_factor


def write_dataset(root: Path, labels: dict[str, str]) -> None:
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    for stem, text in labels.items():
        Image.new("RGB", (200, 100), (40, 40, 40)).save(root / "train" / "images" / f"{stem}.jpg")
        if text is not None:
            (root / "train" / "labels" / f"{stem}.txt").write_text(text)


def prediction(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class DatasetTests(unittest.TestCase):
    def test_reads_yolo_boxes_as_absolute_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.txt"
            path.write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.2 0.2\n\n")
            boxes, labels = read_yolo_labels(path, 200, 100)
        self.assertEqual(labels, [1, 2])
        np.testing.assert_allclose(boxes[0], (80, 30, 120, 70))
        np.testing.assert_allclose(boxes[1], (0, 0, 40, 20))

    def test_rejects_malformed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.txt"
            for text in ("0 0.5 0.5 0.2\n", "0 1.5 0.5 0.2 0.2\n", "7 0.5 0.5 0.2 0.2\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_yolo_labels(path, 100, 100)

    def test_dataset_items_and_background_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, {"fruit": "1 0.5 0.5 0.2 0.4\n", "empty": None})
            dataset = YoloDetectionDataset(root, "train")
            self.assertEqual(dataset.class_counts(), {"orange": 0, "apple": 1})
            images = {path.stem: i for i, path in enumerate(dataset.images)}
            image, target = dataset[images["fruit"]]
            self.assertEqual(tuple(image.shape), (3, 100, 200))
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(target["labels"].tolist(), [2])
            _, empty = dataset[images["empty"]]
            self.assertEqual(tuple(empty["boxes"].shape), (0, 4))

    def test_training_augmentation_keeps_boxes_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, {"fruit": "0 0.5 0.5 0.2 0.4\n1 0.95 0.5 0.1 0.2\n"})
            dataset = YoloDetectionDataset(root, "train", train_transforms())
            torch.manual_seed(0)
            for _ in range(30):
                image, target = dataset[0]
                boxes = target["boxes"]
                self.assertEqual(len(boxes), len(target["labels"]))
                if len(boxes):
                    self.assertTrue(torch.all(boxes[:, 2] > boxes[:, 0]))
                    self.assertTrue(torch.all(boxes[:, 0] >= 0))
                    self.assertTrue(torch.all(boxes[:, 2] <= image.shape[2]))

    def test_label_mapping(self):
        self.assertEqual(model_label(0), 1)
        self.assertEqual(class_name(2), "apple")
        with self.assertRaises(ValueError):
            class_name(0)


class CocoPreparationTests(unittest.TestCase):
    def test_selects_fruit_images_and_skips_crowds(self):
        coco = {
            "categories": [{"id": 53, "name": "apple"}, {"id": 55, "name": "orange"}, {"id": 1, "name": "person"}],
            "images": [{"id": i} for i in (1, 2, 3)],
            "annotations": [
                {"image_id": 1, "category_id": 55, "bbox": [0, 0, 10, 10], "iscrowd": 0},
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0},
                {"image_id": 2, "category_id": 53, "bbox": [0, 0, 10, 10], "iscrowd": 0},
                {"image_id": 2, "category_id": 53, "bbox": [0, 0, 50, 50], "iscrowd": 1},
                {"image_id": 3, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0},
            ],
        }
        selected = select_images(coco)
        self.assertEqual(set(selected), {1})
        self.assertEqual(selected[1]["boxes"], [(0, 0, 0, 10, 10)])

    def test_yolo_lines(self):
        self.assertEqual(yolo_lines([(1, 10, 20, 40, 20), (0, 0, 0, 0.5, 9)], 100, 50),
                         ["1 0.300000 0.600000 0.400000 0.400000"])


class MetricTests(unittest.TestCase):
    def test_perfect_and_missing_predictions(self):
        evaluator = DetectionEvaluator()
        truth = {"boxes": torch.tensor([[0.0, 0, 10, 10], [20, 20, 30, 30]]), "labels": torch.tensor([1, 2])}
        evaluator.add(prediction([[0, 0, 10, 10]], [0.9], [1]), truth)
        result = evaluator.summary()
        self.assertAlmostEqual(result["classes"]["orange"]["ap"], 1.0)
        self.assertEqual(result["classes"]["apple"]["ap"], 0.0)
        self.assertAlmostEqual(result["map"], 0.5)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertEqual(
            result["confusion_matrix"]["matrix"],
            [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
        )

    def test_precision_recall_and_confusion_include_wrong_and_extra_classes(self):
        evaluator = DetectionEvaluator(score_threshold=0.5)
        truth = {
            "boxes": torch.tensor([[0.0, 0, 10, 10], [20.0, 20, 30, 30]]),
            "labels": torch.tensor([1, 2]),
        }
        evaluator.add(
            prediction(
                [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                [0.9, 0.8, 0.7],
                [1, 1, 2],
            ),
            truth,
        )
        result = evaluator.summary()
        self.assertEqual(
            result["confusion_matrix"]["matrix"],
            [[0, 0, 1], [0, 1, 0], [0, 1, 0]],
        )
        self.assertAlmostEqual(result["classes"]["orange"]["precision"], 0.5)
        self.assertAlmostEqual(result["classes"]["orange"]["recall"], 1.0)
        self.assertEqual(result["classes"]["apple"]["precision"], 0.0)
        self.assertEqual(result["classes"]["apple"]["recall"], 0.0)

    def test_duplicate_and_wrong_class_are_false_positives(self):
        evaluator = DetectionEvaluator()
        truth = {"boxes": torch.tensor([[0.0, 0, 10, 10]]), "labels": torch.tensor([1])}
        evaluator.add(prediction([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]],
                                 [0.9, 0.8, 0.95], [1, 1, 2]), truth)
        result = evaluator.summary()["classes"]
        self.assertAlmostEqual(result["orange"]["ap"], 1.0)
        self.assertEqual(result["orange"]["predictions"], 2)
        self.assertTrue(math.isnan(result["apple"]["ap"]))

    def test_ranking_affects_average_precision(self):
        self.assertAlmostEqual(average_precision(np.array([0.5, 0.5, 1.0]), np.array([0.5, 1 / 3, 0.5])), 0.5)

    def test_learning_rate_schedule(self):
        self.assertLess(lr_factor(0, 10, 100), 0.01)
        self.assertAlmostEqual(lr_factor(10, 10, 100), 1.0)
        self.assertAlmostEqual(lr_factor(100, 10, 100), 0.01)


if __name__ == "__main__":
    unittest.main()
