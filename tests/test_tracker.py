import unittest

from fruit_counter.tracker import (
    Detection,
    KalmanBoxFilter,
    LineCounter,
    Track,
    iou_matrix,
    match,
)


def at(x: float, label: str = "orange", score: float = 0.9, y: float = 15) -> Detection:
    return Detection(label, (x - 10, y - 10, x + 10, y + 10), score)


def run(counter: LineCounter, frames: list[list[Detection]]) -> LineCounter:
    for index, detections in enumerate(frames):
        counter.update(detections, index, fps=10)
    counter.finalize()
    return counter


class LineCounterTests(unittest.TestCase):
    def test_crossing_is_counted_once_even_when_track_returns(self):
        xs = [20, 30, 40, 50, 60, 70, 60, 50, 40, 50, 60]
        counter = run(LineCounter(line_x=55), [[at(x)] for x in xs])
        self.assertEqual(counter.counts, {"orange": 1, "apple": 0})
        self.assertEqual(len(counter.events), 1)

    def test_label_flicker_keeps_one_track_and_uses_majority_vote(self):
        labels = ["apple", "apple", "orange", "apple", "apple", "orange", "apple", "apple"]
        frames = [[at(20 + 8 * i, label)] for i, label in enumerate(labels)]
        counter = run(LineCounter(line_x=55), frames)
        self.assertEqual(counter.counts, {"orange": 0, "apple": 1})
        self.assertEqual({event.track_id for event in counter.events}, {1})

    def test_short_occlusion_keeps_identity_through_kalman_prediction(self):
        frames = [[at(20 + 6 * i)] for i in range(4)]
        frames += [[] for _ in range(3)]
        frames += [[at(20 + 6 * i)] for i in range(7, 14)]
        counter = run(LineCounter(line_x=55), frames)
        self.assertEqual(counter.counts["orange"], 1)
        self.assertEqual(counter.events[0].track_id, 1)

    def test_low_score_detection_keeps_track_alive(self):
        scores = [0.9, 0.9, 0.3, 0.3, 0.3, 0.9, 0.9]
        frames = [[at(30 + 6 * i, score=s)] for i, s in enumerate(scores)]
        counter = run(LineCounter(line_x=55, max_missing=0), frames)
        self.assertEqual(counter.counts["orange"], 1)
        self.assertEqual(counter.events[0].track_id, 1)

    def test_reentry_after_counting_is_not_counted_again(self):
        frames = [[at(30 + 10 * i)] for i in range(5)]
        frames += [[] for _ in range(5)]
        frames += [[at(75 + 5 * i)] for i in range(4)]
        counter = run(LineCounter(line_x=55, max_missing=2), frames)
        self.assertEqual(counter.counts["orange"], 1)

    def test_single_spurious_detection_is_not_counted(self):
        counter = run(LineCounter(line_x=55), [[at(50)], [], [at(60)], []])
        self.assertEqual(counter.counts["orange"], 0)

    def test_two_neighbouring_fruit_are_counted_separately(self):
        frames = [[at(10 + 8 * i, "orange", y=15), at(10 + 8 * i, "apple", y=45)] for i in range(10)]
        counter = run(LineCounter(line_x=55), frames)
        self.assertEqual(counter.counts, {"orange": 1, "apple": 1})

    def test_right_to_left_direction(self):
        frames = [[at(90 - 8 * i)] for i in range(10)]
        self.assertEqual(run(LineCounter(line_x=55, direction="rtl"), frames).counts["orange"], 1)
        self.assertEqual(run(LineCounter(line_x=55, direction="ltr"), frames).counts["orange"], 0)

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            LineCounter(line_x=1, direction="up")
        with self.assertRaises(ValueError):
            LineCounter(line_x=1, low_score=0.8, high_score=0.5)

    def test_iou_matrix(self):
        ious = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (5, 0, 15, 10), (20, 20, 30, 30)])
        self.assertAlmostEqual(ious[0, 0], 1.0)
        self.assertAlmostEqual(ious[0, 1], 1 / 3)
        self.assertEqual(ious[0, 2], 0.0)

    def test_matching_gates_invalid_pairs_before_assignment(self):
        track_boxes = [
            (15.1214, 65.9554, 55.1197, 90.7413),
            (28.6789, 70.7304, 46.4318, 124.1424),
        ]
        detection_boxes = [
            (9.1398, 70.2066, 48.1908, 90.7999),
            (22.0322, 31.5024, 64.1528, 82.0295),
        ]
        tracks = [
            Track(i + 1, box, box, 0, 0, KalmanBoxFilter(box), {"orange": 1})
            for i, box in enumerate(track_boxes)
        ]
        detections = [Detection("orange", box) for box in detection_boxes]
        pairs, unmatched_tracks, unmatched_detections = match(tracks, detections, 0.2)
        self.assertEqual(set(pairs), {(0, 1), (1, 0)})
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [])


if __name__ == "__main__":
    unittest.main()
