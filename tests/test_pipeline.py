import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fruit_counter.demo import make_demo
from fruit_counter.pipeline import count_video
from fruit_counter.video import probe_video


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class PipelineTests(unittest.TestCase):
    def test_generated_video_counts_one_of_each_fruit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo.mp4"
            output = root / "result"
            make_demo(source)
            summary = count_video(source, output, width=320)
            self.assertEqual(summary["counts"], {"orange": 1, "apple": 1})
            self.assertEqual(summary["total"], 2)
            self.assertEqual(len(summary["events"]), 2)
            self.assertEqual(json.loads((output / "summary.json").read_text())["total"], 2)
            self.assertTrue((output / "events.csv").exists())
            self.assertEqual(probe_video(output / "annotated.mp4").width, 320)


if __name__ == "__main__":
    unittest.main()
