import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from fruit_counter.model import ModelConfig
from fruit_counter.train import TrainConfig, load_checkpoint, train


def write_dataset(root: Path) -> None:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for index in range(4):
            image = Image.new("RGB", (160, 120), (50, 50, 50))
            ImageDraw.Draw(image).ellipse((40, 30, 80, 70), fill=(240, 130, 20))
            image.save(root / split / "images" / f"{index}.jpg")
            (root / split / "labels" / f"{index}.txt").write_text("0 0.375 0.416667 0.25 0.333333\n")


class TrainingRunTests(unittest.TestCase):
    def test_writes_checkpoints_and_tensorboard_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root / "data")
            output = root / "run"
            model_config = ModelConfig(backbone="resnet18", min_size=(96,), max_size=128,
                                       anchor_sizes=(8, 16, 32, 64, 128))
            config = TrainConfig(data=str(root / "data"), output=str(output), backbone="resnet18",
                                 epochs=1, batch_size=2, accumulate=1, workers=0, device="cpu",
                                 tb_every=1, vis_images=1, save_every=1)
            train(config, model_config)

            for name in (
                "run_metadata.json",
                "logs/metrics.jsonl",
                "checkpoints/last.pt",
                "checkpoints/epoch_001.pt",
                "evaluation/val_epoch_001.json",
            ):
                self.assertTrue((output / name).exists(), name)
            self.assertTrue(
                list((output / "logs" / "tensorboard").glob("events.out.tfevents.*"))
            )
            events = EventAccumulator(str(output / "logs" / "tensorboard"))
            events.Reload()
            scalar_tags = events.Tags()["scalars"]
            self.assertIn("val/loss_total", scalar_tags)
            self.assertIn("val/best_loss", scalar_tags)
            self.assertIn("val/precision_orange", scalar_tags)
            self.assertIn("val/recall_apple", scalar_tags)
            self.assertIn("val/confusion_matrix", events.Tags()["images"])
            validation = json.loads(
                (output / "evaluation" / "val_epoch_001.json").read_text()
            )
            self.assertIn("loss_total", validation["loss"])
            self.assertIn("precision", validation["metrics"])
            self.assertIn("confusion_matrix", validation["metrics"])
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertIsNone(metadata["pretrained_weights"])
            last = load_checkpoint(output / "checkpoints" / "last.pt")
            self.assertEqual((last["epoch"], last["step"]), (0, 2))
            self.assertIn("optimizer", last)
            self.assertIn("rng_state", last)
            if (output / "checkpoints" / "best_map50.pt").exists():
                self.assertNotIn(
                    "optimizer",
                    load_checkpoint(output / "checkpoints" / "best_map50.pt"),
                )
            self.assertTrue((output / "checkpoints" / "best_val_loss.pt").exists())
            self.assertNotIn(
                "optimizer",
                load_checkpoint(output / "checkpoints" / "best_val_loss.pt"),
            )

            with self.assertRaises(ValueError):
                train(config, model_config)

            config.epochs = 2
            train(config, resume=output / "checkpoints" / "last.pt")
            last = load_checkpoint(output / "checkpoints" / "last.pt")
            self.assertEqual((last["epoch"], last["step"]), (1, 4))
            self.assertEqual(list(last["model_config"]["min_size"]), [96])
            self.assertEqual(
                len((output / "logs" / "metrics.jsonl").read_text().splitlines()),
                2,
            )


if __name__ == "__main__":
    unittest.main()
