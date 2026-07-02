"""
Ball detection and tracking.

Squash ball tracking is the hardest perception module: the ball is small, dark,
fast, motion-blurred, and competes with player-limb motion and glass reflections.
A single frame-difference yields dozens of false candidates (mostly limbs), so the
real signal is *trajectory continuity* across many frames, not per-frame detection.

This module provides:

    BallDetector            - interface; swap MotionBallDetector for a trained
                              TrackNet later without changing the tracker.
    MotionBallDetector      - 3-frame-difference candidates, masked by player boxes.
    BallTracker             - links candidates over time with a constant-velocity
                              motion model and keeps the most ball-like trajectory.

The detector deliberately favours *recall* (keep noisy candidates); the tracker
supplies *precision* by demanding smooth, fast, physically-plausible motion. The
candidate stream also doubles as the bootstrap source for annotating a training
set for a future TrackNet model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BallCandidate:
    frame_index: int
    timestamp: float
    x: float          # image pixels
    y: float
    area: float
    score: float = 0.0  # detector confidence (higher = more ball-like)


@dataclass
class BallPoint:
    frame_index: int
    timestamp: float
    x: float
    y: float
    interpolated: bool = False  # filled by the motion model across a gap


# ----------------------------------------------------------------------------
# Detector interface + classical motion detector
# ----------------------------------------------------------------------------
# The interface is window-level: given a list of consecutive BGR frames, return
# one candidate list per frame. This is the seam where a trained TrackNet
# (perception/tracknet.py) drops in to replace the classical detector without the
# tracker or annotation engine changing — TrackNet needs colour + batching, which
# a per-frame-gray interface could not express.
PlayerBoxes = List[Tuple[float, float, float, float]]


class BallDetector(ABC):
    @abstractmethod
    def detect_window(
        self,
        frames_bgr: List[np.ndarray],
        start_frame_index: int,
        fps: float,
        player_boxes_per_frame: Optional[List[PlayerBoxes]] = None,
    ) -> List[List[BallCandidate]]:
        """Return per-frame ball candidates for a window of consecutive frames."""
        raise NotImplementedError


def _point_in_boxes(
    x: float, y: float, boxes: PlayerBoxes, pad: float
) -> bool:
    for (x1, y1, x2, y2) in boxes:
        if (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad):
            return True
    return False


class MotionBallDetector(BallDetector):
    """Frame-difference ball candidate detector.

    Uses a 3-frame difference (AND of consecutive abs-diffs) which isolates
    objects that are moving *and* small, suppressing slow background change.
    Candidates overlapping player boxes are dropped, since limb motion is the
    dominant false-positive source. Candidates are scored by how ball-like their
    size and roundness are. First/last frames of the window yield no candidates
    (no triplet available).
    """

    def __init__(
        self,
        diff_threshold: int = 16,
        min_area: float = 2.0,
        max_area: float = 350.0,
        max_aspect: float = 2.6,
        player_box_pad: float = 8.0,
        ideal_area: float = 25.0,
    ):
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_aspect = max_aspect
        self.player_box_pad = player_box_pad
        self.ideal_area = ideal_area

    def _detect_triplet(
        self, prev_gray, cur_gray, next_gray, frame_index, timestamp, player_boxes
    ) -> List[BallCandidate]:
        import cv2

        d1 = cv2.absdiff(cur_gray, prev_gray)
        d2 = cv2.absdiff(next_gray, cur_gray)
        motion = cv2.bitwise_and(d1, d2)
        _, th = cv2.threshold(motion, self.diff_threshold, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: List[BallCandidate] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect > self.max_aspect:
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            if player_boxes and _point_in_boxes(cx, cy, player_boxes, self.player_box_pad):
                continue
            roundness = 1.0 / aspect
            size_fit = np.exp(-abs(area - self.ideal_area) / self.ideal_area)
            out.append(
                BallCandidate(
                    frame_index=frame_index, timestamp=timestamp,
                    x=cx, y=cy, area=float(area),
                    score=float(0.5 * roundness + 0.5 * size_fit),
                )
            )
        return out

    def detect_window(
        self,
        frames_bgr,
        start_frame_index,
        fps,
        player_boxes_per_frame=None,
    ) -> List[List[BallCandidate]]:
        import cv2

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
        per_frame: List[List[BallCandidate]] = []
        for i in range(len(frames_bgr)):
            if i == 0 or i == len(frames_bgr) - 1:
                per_frame.append([])
                continue
            boxes = (
                player_boxes_per_frame[i]
                if player_boxes_per_frame and i < len(player_boxes_per_frame)
                else None
            )
            per_frame.append(
                self._detect_triplet(
                    grays[i - 1], grays[i], grays[i + 1],
                    start_frame_index + i, (start_frame_index + i) / fps, boxes,
                )
            )
        return per_frame


# ----------------------------------------------------------------------------
# Trajectory linking
# ----------------------------------------------------------------------------
@dataclass
class _Track:
    points: List[BallCandidate] = field(default_factory=list)
    vx: float = 0.0
    vy: float = 0.0
    misses: int = 0

    @property
    def last(self) -> BallCandidate:
        return self.points[-1]

    def predict(self) -> Tuple[float, float]:
        p = self.last
        return p.x + self.vx, p.y + self.vy


class BallTracker:
    """Links per-frame candidates into trajectories via constant-velocity gating.

    For each frame we extend existing tracks to the nearest candidate within a
    velocity-consistent search radius, coast (predict) across short misses, and
    start new tracks from unused candidates. The winning ball trajectory is the
    track that is long, fast and smooth — properties limb-motion tracks lack.
    """

    def __init__(
        self,
        search_radius: float = 60.0,
        max_misses: int = 6,
        min_track_len: int = 6,
        min_avg_speed: float = 3.0,   # px/frame; the ball actually moves
    ):
        self.search_radius = search_radius
        self.max_misses = max_misses
        self.min_track_len = min_track_len
        self.min_avg_speed = min_avg_speed

    def _update_velocity(self, tr: _Track):
        if len(tr.points) >= 2:
            a, b = tr.points[-2], tr.points[-1]
            # blend new velocity with old for a little smoothing
            nvx, nvy = b.x - a.x, b.y - a.y
            tr.vx = 0.5 * tr.vx + 0.5 * nvx
            tr.vy = 0.5 * tr.vy + 0.5 * nvy

    def build_tracks(self, per_frame: List[List[BallCandidate]]) -> List[_Track]:
        active: List[_Track] = []
        finished: List[_Track] = []

        for cands in per_frame:
            used = [False] * len(cands)
            # extend existing tracks
            for tr in active:
                px, py = tr.predict()
                best_j, best_d = -1, self.search_radius
                # tracks with velocity get a tighter, motion-consistent gate
                gate = self.search_radius if abs(tr.vx) + abs(tr.vy) < 1 else self.search_radius * 0.8
                for j, c in enumerate(cands):
                    if used[j]:
                        continue
                    d = np.hypot(c.x - px, c.y - py)
                    if d < min(best_d, gate):
                        best_d, best_j = d, j
                if best_j >= 0:
                    tr.points.append(cands[best_j])
                    used[best_j] = True
                    tr.misses = 0
                    self._update_velocity(tr)
                else:
                    tr.misses += 1

            # retire stale tracks
            still: List[_Track] = []
            for tr in active:
                if tr.misses > self.max_misses:
                    finished.append(tr)
                else:
                    still.append(tr)
            active = still

            # seed new tracks from unused candidates
            for j, c in enumerate(cands):
                if not used[j]:
                    active.append(_Track(points=[c]))

        finished.extend(active)
        return finished

    def _track_quality(self, tr: _Track) -> float:
        n = len(tr.points)
        if n < self.min_track_len:
            return -1.0
        xs = np.array([p.x for p in tr.points])
        ys = np.array([p.y for p in tr.points])
        step = np.hypot(np.diff(xs), np.diff(ys))
        avg_speed = float(step.mean())
        if avg_speed < self.min_avg_speed:
            return -1.0
        # Net displacement vs. path length: a real ball *travels* (high net
        # displacement), whereas limb/racket jitter accumulates path length while
        # staying put. straightness in [0,1] rewards genuine flight.
        path_len = float(step.sum())
        net_disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        # bounding-box diagonal: ball flight covers real spatial extent
        extent = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        straightness = net_disp / (path_len + 1e-6)
        # smoothness: low acceleration variance relative to speed
        acc = np.diff(step)
        smoothness = 1.0 / (1.0 + (np.std(acc) / (avg_speed + 1e-6)))
        # Combined: must move far and fast, fairly straight and smooth.
        return (
            extent
            * (0.4 + 0.6 * straightness)
            * (0.5 + 0.5 * smoothness)
            * (1.0 + 0.05 * avg_speed)
        )

    def best_trajectory(self, per_frame: List[List[BallCandidate]]) -> List[BallPoint]:
        tracks = self.build_tracks(per_frame)
        scored = [(self._track_quality(t), t) for t in tracks]
        scored = [(q, t) for q, t in scored if q > 0]
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        return [
            BallPoint(p.frame_index, p.timestamp, p.x, p.y) for p in best.points
        ]

    def all_trajectories(
        self, per_frame: List[List[BallCandidate]]
    ) -> List[List[BallPoint]]:
        """All plausible ball tracks (useful for debugging / annotation seeding)."""
        tracks = self.build_tracks(per_frame)
        out = []
        for t in tracks:
            if self._track_quality(t) > 0:
                out.append(
                    [BallPoint(p.frame_index, p.timestamp, p.x, p.y) for p in t.points]
                )
        return out


import os

# Trained TrackNet weights live here, one model per camera SETUP ("phone" =
# the user's close-up behind-glass games, "broadcast" = wide pro-broadcast angle).
# The two setups are visually too different for one small model, so we route to a
# specialised model per setup and fall back to the combined/classical detector.
WEIGHTS_DIR = "perception/weights"


def setup_weights_path(setup: str) -> str:
    return os.path.join(WEIGHTS_DIR, f"tracknet_{setup}.pt")


def get_ball_detector(setup: str = "phone") -> "BallDetector":
    """Return the best ball detector for a camera setup.

    Tries the setup-specific model first (e.g. tracknet_phone.pt), then the legacy
    combined model, then the classical fallback. Routing by setup keeps each model
    specialised so phone and broadcast footage don't degrade each other.
    """
    candidates = [
        setup_weights_path(setup),
        os.path.join(WEIGHTS_DIR, "tracknet.pt"),  # legacy combined
    ]
    for wp in candidates:
        if os.path.exists(wp):
            try:
                from .tracknet import TrackNetBallDetector

                det = TrackNetBallDetector(weights_path=wp)
                if det.available:
                    return det
            except Exception:
                pass
    return MotionBallDetector()
