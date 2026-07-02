"""
Perception pipeline: video -> structured, real-world movement data.

This is the orchestration layer that turns tracking + court homography into the
metrics the rest of the app consumes. Crucially it replaces the previous
``np.random`` placeholder movement data with measured player trajectories in
court metres, plus derived analytics:

    * per-player trajectory (court metres over time)
    * total distance travelled (metres)
    * average / peak speed (m/s)
    * court coverage as a fraction of floor area actually occupied
    * T-dominance: share of time spent near the T, and mean distance to the T
    * time spent in front / mid / back thirds

The output ``movement_data`` list is shape-compatible with what the frontend and
server already expect ({player, x, y, time}) but x/y are now **normalised real
court positions** (0..1 across width / length) instead of random noise.

Kinematics methodology follows Baclig, Ergezinger, Mei, Gül, Adeeb & Westover,
"A Deep Learning and Computer Vision Based Multi-Player Tracker for Squash",
Appl. Sci. 2020, 10, 8793 (CC BY 4.0): inverse-perspective court mapping, a
5th-order moving-average filter on foot coordinates, and T-relative tactical
metrics (distance, T-dominance, % left/behind the T, filtered speeds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .court import (
    COURT_LENGTH,
    COURT_WIDTH,
    HALF_COURT_X,
    SHORT_LINE_Y,
    CourtModel,
    CourtCalibration,
)
from .tracking import (
    PlayerTracker,
    assign_stable_player_labels,
    build_appearance_references,
    _hist_similarity,
)

# A player is "on the T" when within this radius (metres) of the T point.
T_RADIUS = 1.5
# Ignore implausible single-frame jumps above this speed (m/s) as tracking noise.
# Baclig et al. (2020) likewise exclude speeds >9 m/s as foot-detection noise.
MAX_PLAUSIBLE_SPEED = 9.0
# Below this speed (m/s) a player is effectively stationary (waiting / set for a
# shot). Baclig et al. report stats excluding <1 m/s for realistic movement speed.
STANDING_SPEED = 1.0


def _moving_average_filter(pts: np.ndarray, half_window: int = 2) -> np.ndarray:
    """5th-order moving-average smoothing of court coordinates.

    Implements Baclig et al. (2020) Eq. 2: Coordinate_t = (1/5) * sum_{j=-2..2}
    Coordinate_{t+j}. Foot detection jitters frame-to-frame (the foot node lands
    higher up the ankle or lower on the floor), which inflates distance and speed;
    smoothing cut their distance error from ~19.85% to ~3.73% vs ground truth.
    Window is clamped at the trajectory ends.
    """
    n = len(pts)
    if n < 2 * half_window + 1:
        return pts
    out = np.empty_like(pts, dtype=float)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        out[i] = pts[lo:hi].mean(axis=0)
    return out


@dataclass
class PlayerTrack:
    label: str
    times: List[float] = field(default_factory=list)
    court_xy: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class PerceptionResult:
    movement_data: List[Dict[str, float]]
    player_metrics: Dict[str, Dict[str, float]]
    fps_sampled: float
    frames_processed: int

    def to_dict(self) -> Dict:
        return {
            "movement_data": self.movement_data,
            "player_metrics": self.player_metrics,
            "fps_sampled": self.fps_sampled,
            "frames_processed": self.frames_processed,
        }


def _court_polygon(calib: CourtCalibration) -> np.ndarray:
    return calib.image_points()


def _label_detections_by_appearance(fd, ref1, ref2, fallback_map):
    """Assign each frame's detections to player1/player2 by shirt colour.

    With two detections, pick the global 2-to-2 pairing with higher total colour
    similarity (so the players never both collapse to one label). Falls back to the
    ByteTrack-id labelling when colour signatures are unavailable.
    """
    dets = fd.detections
    if ref1 is None or ref2 is None or any(d.torso_hist is None for d in dets):
        return [(d, fallback_map.get(d.track_id, "player1")) for d in dets]

    if len(dets) >= 2:
        a, b = dets[0], dets[1]
        # pairing 1: a->p1, b->p2 ; pairing 2: a->p2, b->p1
        s1 = _hist_similarity(a.torso_hist, ref1) + _hist_similarity(b.torso_hist, ref2)
        s2 = _hist_similarity(a.torso_hist, ref2) + _hist_similarity(b.torso_hist, ref1)
        if s1 >= s2:
            return [(a, "player1"), (b, "player2")] + [(d, fallback_map.get(d.track_id, "player1")) for d in dets[2:]]
        return [(a, "player2"), (b, "player1")] + [(d, fallback_map.get(d.track_id, "player1")) for d in dets[2:]]

    out = []
    for d in dets:
        label = "player1" if _hist_similarity(d.torso_hist, ref1) >= _hist_similarity(d.torso_hist, ref2) else "player2"
        out.append((d, label))
    return out


def _build_tracks(frames, court: CourtModel) -> Dict[str, PlayerTrack]:
    # Appearance-based player identification (Baclig et al.): re-identify by shirt
    # colour, robust to ByteTrack id fragmentation under occlusion.
    fallback_map = assign_stable_player_labels(frames)
    ref1, ref2 = build_appearance_references(frames)

    tracks: Dict[str, PlayerTrack] = {}
    for fd in frames:
        best: Dict[str, Tuple[float, Tuple[float, float]]] = {}
        for det, label in _label_detections_by_appearance(fd, ref1, ref2, fallback_map):
            cx, cy = court.to_court(det.foot_point)
            if not court.in_bounds((cx, cy)):
                continue
            if label not in best or det.area > best[label][0]:
                best[label] = (det.area, (cx, cy))
        for label, (_area, cxy) in best.items():
            tr = tracks.setdefault(label, PlayerTrack(label=label))
            tr.times.append(fd.timestamp)
            tr.court_xy.append(cxy)
    return tracks


def _metrics_for_track(tr: PlayerTrack) -> Dict[str, float]:
    """Per-player kinematics, following the validated methodology of Baclig et al.
    (2020), "A Deep Learning and Computer Vision Based Multi-Player Tracker for
    Squash" (Appl. Sci. 10, 8793, CC BY): smooth the foot coordinates, then derive
    distance, speed, T-dominance and T-relative tactical positioning."""
    if len(tr.court_xy) < 2:
        return {
            "distance_m": 0.0, "distance_unfiltered_m": 0.0,
            "avg_speed_ms": 0.0, "avg_speed_active_ms": 0.0, "peak_speed_ms": 0.0,
            "court_coverage_pct": 0.0, "t_dominance_pct": 0.0, "mean_dist_to_t_m": 0.0,
            "pct_left_of_t": 0.0, "pct_behind_t": 0.0, "pct_time_moving": 0.0,
            "front_pct": 0.0, "mid_pct": 0.0, "back_pct": 0.0,
            "samples": len(tr.court_xy),
        }

    raw = np.array(tr.court_xy)
    times = np.array(tr.times)
    # Smooth foot coordinates (Baclig et al. Eq. 2) before any cumulative stat.
    pts = _moving_average_filter(raw)

    dt = np.diff(times)
    dt[dt <= 0] = np.nan

    def _speeds(coords):
        seg = np.diff(coords, axis=0)
        seg_dist = np.hypot(seg[:, 0], seg[:, 1])
        return seg_dist, seg_dist / dt

    seg_dist, speeds = _speeds(pts)
    _, raw_speeds = _speeds(raw)
    valid = speeds <= MAX_PLAUSIBLE_SPEED          # exclude >9 m/s noise
    distance = float(np.nansum(seg_dist[valid]))
    raw_seg, _ = _speeds(raw)
    distance_unfiltered = float(np.nansum(raw_seg[raw_speeds <= MAX_PLAUSIBLE_SPEED]))

    avg_speed = float(np.nanmean(speeds[valid])) if valid.any() else 0.0
    peak_speed = float(np.nanmax(speeds[valid])) if valid.any() else 0.0
    # Active speed = exclude standing (<1 m/s) as well — realistic moving speed.
    active = valid & (speeds >= STANDING_SPEED)
    avg_speed_active = float(np.nanmean(speeds[active])) if active.any() else 0.0
    pct_time_moving = (float(np.sum(active)) / float(np.sum(valid)) * 100.0) if valid.any() else 0.0

    # Court coverage: fraction of a coarse grid of floor cells the player visited.
    grid_w, grid_h = 8, 12  # ~0.8m x ~0.8m cells
    cells = set()
    for x, y in pts:
        gx = min(grid_w - 1, max(0, int(x / COURT_WIDTH * grid_w)))
        gy = min(grid_h - 1, max(0, int(y / COURT_LENGTH * grid_h)))
        cells.add((gx, gy))
    coverage = len(cells) / float(grid_w * grid_h) * 100.0

    # T-dominance + average radial distance to the T.
    dists_to_t = np.array([CourtModel.distance_to_t((x, y)) for x, y in pts])
    t_dominance = float(np.mean(dists_to_t <= T_RADIUS)) * 100.0
    mean_dist_to_t = float(np.mean(dists_to_t))

    # T-relative tactical positioning (Baclig et al. §2.3). Origin at the T:
    #   left of T  = court x < half-court line (backhand side for a right-hander)
    #   behind T   = court y past the short line (toward the back wall)
    n = len(pts)
    pct_left_of_t = float(np.mean(pts[:, 0] < HALF_COURT_X)) * 100.0
    pct_behind_t = float(np.mean(pts[:, 1] > SHORT_LINE_Y)) * 100.0

    # Depth distribution.
    zones = [CourtModel.depth_zone((x, y)) for x, y in pts]
    front_pct = zones.count("front") / n * 100.0
    mid_pct = zones.count("mid") / n * 100.0
    back_pct = zones.count("back") / n * 100.0

    return {
        "distance_m": round(distance, 1),
        "distance_unfiltered_m": round(distance_unfiltered, 1),
        "avg_speed_ms": round(avg_speed, 2),
        "avg_speed_active_ms": round(avg_speed_active, 2),
        "peak_speed_ms": round(peak_speed, 2),
        "pct_time_moving": round(pct_time_moving, 1),
        "court_coverage_pct": round(coverage, 1),
        "t_dominance_pct": round(t_dominance, 1),
        "mean_dist_to_t_m": round(mean_dist_to_t, 2),
        "pct_left_of_t": round(pct_left_of_t, 1),
        "pct_behind_t": round(pct_behind_t, 1),
        "front_pct": round(front_pct, 1),
        "mid_pct": round(mid_pct, 1),
        "back_pct": round(back_pct, 1),
        "samples": int(n),
    }


def analyze_movement(
    video_path: str,
    calibration: CourtCalibration,
    sample_every: int = 3,
    model_name: str = "yolo11n.pt",
    device: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> PerceptionResult:
    """Full perception pass over a video given a court calibration.

    Returns measured, court-grounded movement data and per-player metrics.
    """
    court = CourtModel(calibration)
    tracker = PlayerTracker(
        model_name=model_name,
        court_polygon=_court_polygon(calibration),
        device=device,
    )
    frames = tracker.track_video(
        video_path, sample_every=sample_every, max_frames=max_frames
    )
    tracks = _build_tracks(frames, court)

    movement_data: List[Dict[str, float]] = []
    player_metrics: Dict[str, Dict[str, float]] = {}
    for label, tr in tracks.items():
        for (x, y), t in zip(tr.court_xy, tr.times):
            movement_data.append(
                {
                    "player": label,
                    "x": round(x / COURT_WIDTH, 4),   # normalised 0..1 across
                    "y": round(y / COURT_LENGTH, 4),  # normalised 0..1 depth
                    "court_x_m": round(x, 2),
                    "court_y_m": round(y, 2),
                    "time": round(t, 2),
                }
            )
        player_metrics[label] = _metrics_for_track(tr)

    fps_sampled = 0.0
    if len(frames) >= 2:
        span = frames[-1].timestamp - frames[0].timestamp
        if span > 0:
            fps_sampled = (len(frames) - 1) / span

    return PerceptionResult(
        movement_data=movement_data,
        player_metrics=player_metrics,
        fps_sampled=round(fps_sampled, 2),
        frames_processed=len(frames),
    )
