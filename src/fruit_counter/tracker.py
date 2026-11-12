"""Track detections across frames (ByteTrack-style) and count line crossings.

Per frame:
1. Every track predicts its box with a constant-velocity Kalman filter.
2. High-score detections are matched to all tracks with the Hungarian
   algorithm. The score of a pair is the larger IoU of the detection with the
   predicted box or with the last observed box, so fruit that stops or reverses
   suddenly is not lost. Matching ignores class labels, so a fruit whose predicted class
   flickers between frames keeps one ID.
3. Low-score detections are matched to tracks still unmatched that were seen in
   the previous frame, which keeps partly occluded fruit tracked (Zhang et al.,
   "ByteTrack", 2022).
4. Unmatched high-score detections start tentative tracks; a track is confirmed
   after ``min_hits`` consecutive matches and dropped after ``max_missing``
   frames without one.

Each track's class is a confidence-weighted vote over all its detections. A
confirmed track is counted once, when its center has moved from the side of the
line where it started to the other side in the configured direction.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: tuple[float, float, float, float]
    score: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.bbox
        return ((left + right) / 2, (top + bottom) / 2)


@dataclass(frozen=True)
class CountEvent:
    frame: int
    seconds: float
    track_id: int
    label: str


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over (cx, cy, w, h) and their velocities."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        cx, cy, w, h = _xyxy_to_cxcywh(bbox)
        self.mean = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        scale = max(w, h)
        self.covariance = np.diag(
            np.square([scale / 10, scale / 10, scale / 10, scale / 10,
                       scale / 2, scale / 2, scale / 20, scale / 20])
        )
        self.transition = np.eye(8)
        self.transition[:4, 4:] = np.eye(4)
        self.observation = np.eye(4, 8)

    def _scale(self) -> float:
        return max(1.0, float(max(self.mean[2], self.mean[3])))

    def predict(self) -> None:
        scale = self._scale()
        noise = np.diag(np.square([scale / 20] * 4 + [scale / 40, scale / 40, scale / 160, scale / 160]))
        self.mean = self.transition @ self.mean
        self.mean[2:4] = np.maximum(self.mean[2:4], 1.0)
        self.covariance = self.transition @ self.covariance @ self.transition.T + noise

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        scale = self._scale()
        noise = np.diag(np.square([scale / 20] * 4))
        innovation = np.array(_xyxy_to_cxcywh(bbox)) - self.observation @ self.mean
        system = self.observation @ self.covariance @ self.observation.T + noise
        gain = self.covariance @ self.observation.T @ np.linalg.inv(system)
        self.mean = self.mean + gain @ innovation
        self.covariance = (np.eye(8) - gain @ self.observation) @ self.covariance

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        cx, cy, w, h = self.mean[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


@dataclass
class Track:
    id: int
    bbox: tuple[float, float, float, float]
    observed: tuple[float, float, float, float]
    start_x: float
    last_frame: int
    kalman: KalmanBoxFilter
    votes: dict[str, float] = field(default_factory=dict)
    hits: int = 1
    confirmed: bool = False
    counted: bool = False

    @property
    def label(self) -> str:
        return max(self.votes, key=self.votes.get)

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.bbox
        return ((left + right) / 2, (top + bottom) / 2)

    def vote(self, detection: Detection) -> None:
        self.votes[detection.label] = self.votes.get(detection.label, 0.0) + detection.score


def _xyxy_to_cxcywh(bbox) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    return ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)


def iou_matrix(first: list, second: list) -> np.ndarray:
    if not first or not second:
        return np.zeros((len(first), len(second)))
    a = np.asarray(first, dtype=float)[:, None, :]
    b = np.asarray(second, dtype=float)[None, :, :]
    width = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    height = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    intersection = width * height
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def match(tracks: list[Track], detections: list[Detection], min_iou: float):
    """Hungarian assignment on 1 - IoU; pairs below ``min_iou`` are rejected."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    boxes = [d.bbox for d in detections]
    ious = np.maximum(
        iou_matrix([t.kalman.bbox for t in tracks], boxes),
        iou_matrix([t.observed for t in tracks], boxes),
    )
    cost = 1 - ious
    # This penalty makes assignment maximize the number of valid pairs before
    # minimizing IoU cost. Gating only afterwards can discard a valid pairing.
    cost[ious < min_iou] = max(cost.shape) + 1
    rows, columns = linear_sum_assignment(cost)
    pairs = [(r, c) for r, c in zip(rows, columns) if ious[r, c] >= min_iou]
    matched_rows = {r for r, _ in pairs}
    matched_columns = {c for _, c in pairs}
    return (
        pairs,
        [r for r in range(len(tracks)) if r not in matched_rows],
        [c for c in range(len(detections)) if c not in matched_columns],
    )


class LineCounter:
    def __init__(
        self,
        line_x: float,
        direction: str = "ltr",
        high_score: float = 0.5,
        low_score: float = 0.1,
        new_track_score: float = 0.6,
        min_iou: float = 0.2,
        min_hits: int = 2,
        max_missing: int = 30,
        labels: tuple[str, ...] = ("orange", "apple"),
    ) -> None:
        if direction not in {"ltr", "rtl"}:
            raise ValueError("direction must be 'ltr' or 'rtl'")
        if not 0 <= low_score <= high_score <= new_track_score <= 1:
            raise ValueError("expected 0 <= low_score <= high_score <= new_track_score <= 1")
        self.line_x = line_x
        self.direction = direction
        self.high_score = high_score
        self.low_score = low_score
        self.new_track_score = new_track_score
        self.min_iou = min_iou
        self.min_hits = min_hits
        self.max_missing = max_missing
        self.tracks: dict[int, Track] = {}
        self.events: list[CountEvent] = []
        self.counts = dict.fromkeys(labels, 0)
        self._next_id = 1
        self._counted_tracks: dict[int, Track] = {}

    def _before_line(self, x: float) -> bool:
        return x < self.line_x if self.direction == "ltr" else x > self.line_x

    def _apply(self, track: Track, detection: Detection, frame_index: int) -> None:
        track.kalman.update(detection.bbox)
        track.bbox = track.kalman.bbox
        track.observed = detection.bbox
        track.last_frame = frame_index
        track.hits += 1
        track.vote(detection)
        if track.hits >= self.min_hits:
            track.confirmed = True

    def update(self, detections: list[Detection], frame_index: int, fps: float) -> list[CountEvent]:
        if fps <= 0:
            raise ValueError("fps must be positive")
        for track in self.tracks.values():
            track.kalman.predict()
            track.bbox = track.kalman.bbox

        high = [d for d in detections if d.score >= self.high_score]
        low = [d for d in detections if self.low_score <= d.score < self.high_score]
        tracks = list(self.tracks.values())

        pairs, unmatched_tracks, unmatched_high = match(tracks, high, self.min_iou)
        for row, column in pairs:
            self._apply(tracks[row], high[column], frame_index)

        recent = [tracks[r] for r in unmatched_tracks if tracks[r].last_frame == frame_index - 1]
        low_pairs, _, _ = match(recent, low, max(self.min_iou, 0.5))
        for row, column in low_pairs:
            self._apply(recent[row], low[column], frame_index)

        for column in unmatched_high:
            detection = high[column]
            if detection.score < self.new_track_score:
                continue
            track = Track(
                id=self._next_id,
                bbox=detection.bbox,
                observed=detection.bbox,
                start_x=detection.center[0],
                last_frame=frame_index,
                kalman=KalmanBoxFilter(detection.bbox),
                confirmed=self.min_hits <= 1,
            )
            track.vote(detection)
            self.tracks[track.id] = track
            self._next_id += 1

        for track_id, track in list(self.tracks.items()):
            missing = frame_index - track.last_frame
            if missing > self.max_missing or (missing > 0 and not track.confirmed):
                del self.tracks[track_id]

        new_events = []
        for track in self.tracks.values():
            if track.counted or not track.confirmed or track.last_frame != frame_index:
                continue
            if self._before_line(track.start_x) and not self._before_line(track.center[0]):
                track.counted = True
                self._counted_tracks[track.id] = track
                event = CountEvent(frame_index, frame_index / fps, track.id, track.label)
                self.events.append(event)
                new_events.append(event)
                self.counts[track.label] = self.counts.get(track.label, 0) + 1
        return new_events

    def finalize(self) -> None:
        """Relabel past events with each track's final class vote."""
        final = {track_id: track.label for track_id, track in self._counted_tracks.items()}
        self.events = [
            CountEvent(e.frame, e.seconds, e.track_id, final.get(e.track_id, e.label))
            for e in self.events
        ]
        self.counts = dict.fromkeys(self.counts, 0)
        for event in self.events:
            self.counts[event.label] = self.counts.get(event.label, 0) + 1
