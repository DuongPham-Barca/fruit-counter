import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fruit_counter.demo import make_demo
from fruit_counter.pipeline import count_video
from fruit_counter.tracker import Detection
from fruit_counter.video import probe_video


class FixtureDetector:
    """Return the known boxes drawn by ``make_demo`` without production color logic."""

    def __init__(self) -> None:
        self.frame = 0

    def __call__(self, _frame) -> list[Detection]:
        detections = []
        for label, x, y in (
            ("orange", 25 + 3 * self.frame, 72),
            ("apple", -30 + 3 * self.frame, 120),
        ):
            if x + 14 >= 0 and x - 14 < 320:
                detections.append(
                    Detection(label, (max(0, x - 14), y - 14, min(320, x + 15), y + 15))
                )
        self.frame += 1
        return detections


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class PipelineTests(unittest.TestCase):
    def test_generated_video_counts_one_of_each_fruit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo.mp4"
            output = root / "result"
            make_demo(source)
            summary = count_video(
                source,
                output,
                width=320,
                detector=FixtureDetector(),
                method="test_fixture",
            )
            self.assertEqual(summary["counts"], {"orange": 1, "apple": 1})
            self.assertEqual(summary["total"], 2)
            self.assertEqual(len(summary["events"]), 2)
            self.assertEqual(json.loads((output / "summary.json").read_text())["total"], 2)
            self.assertTrue((output / "events.csv").exists())
            self.assertEqual(probe_video(output / "annotated.mp4").width, 320)


if __name__ == "__main__":
    unittest.main()
