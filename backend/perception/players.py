"""
Player detection and tracking for squash video analysis.

Uses YOLOv8 (person class) to detect both players per frame, then a
centroid tracker to maintain consistent Player 1 / Player 2 identity across
the video. With a court calibration, each player's feet position is mapped
to real court metres so we can compute T-control, service-box presence, and
court-zone heatmaps.

Phase 2 of the squash analysis plan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
YOLO_WEIGHTS = os.path.join(WEIGHTS_DIR, "yolov8n.pt")

# ─── data classes ────────────────────────────────────────────────────────────

@dataclass
class PlayerBox:
    """One detected player bounding box in a single frame."""
    frame_index: int
    player_id: int            # 1 or 2 (consistent across frames via tracker)
    x1: float; y1: float      # top-left pixel
    x2: float; y2: float      # bottom-right pixel
    conf: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def feet_x(self) -> float:
        """Horizontal centre of the bounding box — best x estimate for feet."""
        return self.cx

    @property
    def feet_y(self) -> float:
        """Bottom of the bounding box — closest point to the court floor."""
        return self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class PlayerFrame:
    """All detected players in one frame."""
    frame_index: int
    timestamp: float
    players: List[PlayerBox] = field(default_factory=list)

    def get(self, player_id: int) -> Optional[PlayerBox]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None


# ─── centroid tracker ─────────────────────────────────────────────────────────

class _CentroidTracker:
    """Assigns consistent IDs across frames by nearest-centroid matching.

    In squash the camera is fixed, so identity rarely flips. We use a simple
    greedy nearest-centroid match with a max-distance gate. Identity is
    initialised on the first frame with two players: the player with the
    lower y-pixel (further from camera, closer to front wall) becomes P1,
    the one with higher y (closer to camera) becomes P2.
    """

    def __init__(self, max_dist: float = 200.0, max_miss: int = 10):
        self.max_dist = max_dist
        self.max_miss = max_miss
        self._tracks: Dict[int, np.ndarray] = {}   # id -> last centroid
        self._miss: Dict[int, int] = {}
        self._next_id = 1

    def _assign(self, boxes: List[Tuple[float, float]]) -> List[int]:
        """Match a list of (cx, cy) centroids to existing tracks. Returns IDs."""
        if not self._tracks:
            ids = []
            # Bootstrap: sort by cy ascending (smaller y = closer to front wall = P1)
            sorted_boxes = sorted(enumerate(boxes), key=lambda x: x[1][1])
            for rank, (orig_idx, _) in enumerate(sorted_boxes):
                pid = rank + 1   # 1, 2
                self._tracks[pid] = np.array(boxes[orig_idx])
                self._miss[pid] = 0
                ids.append(None)  # fill later
            id_list = [None] * len(boxes)
            for rank, (orig_idx, _) in enumerate(sorted(enumerate(boxes), key=lambda x: x[1][1])):
                id_list[orig_idx] = rank + 1
            return id_list

        centroids = np.array(boxes, dtype=float)
        track_ids = list(self._tracks.keys())
        track_pts = np.array([self._tracks[k] for k in track_ids])

        assigned = [-1] * len(boxes)
        used_tracks: set = set()

        # Greedy nearest match
        dists = np.linalg.norm(
            centroids[:, None, :] - track_pts[None, :, :], axis=2
        )  # (n_boxes, n_tracks)

        for _ in range(min(len(boxes), len(track_ids))):
            if dists.size == 0:
                break
            r, c = np.unravel_index(np.argmin(dists), dists.shape)
            if dists[r, c] > self.max_dist:
                break
            assigned[r] = track_ids[c]
            used_tracks.add(c)
            dists[r, :] = np.inf
            dists[:, c] = np.inf

        # Update matched tracks
        for bi, pid in enumerate(assigned):
            if pid != -1:
                self._tracks[pid] = centroids[bi]
                self._miss[pid] = 0

        # Age unmatched tracks
        for ci, pid in enumerate(track_ids):
            if ci not in used_tracks:
                self._miss[pid] = self._miss.get(pid, 0) + 1
                if self._miss[pid] > self.max_miss:
                    del self._tracks[pid]
                    del self._miss[pid]

        # New tracks for unmatched detections (only if <2 tracks exist)
        for bi, pid in enumerate(assigned):
            if pid == -1 and len(self._tracks) < 2:
                new_id = max(self._tracks.keys(), default=0) + 1
                # Clamp to 1 or 2
                new_id = 1 if 1 not in self._tracks else 2
                self._tracks[new_id] = centroids[bi]
                self._miss[new_id] = 0
                assigned[bi] = new_id

        return assigned


# ─── detector ─────────────────────────────────────────────────────────────────

class PlayerDetector:
    """Detects and tracks both squash players across a video span."""

    def __init__(self, conf: float = 0.35):
        from ultralytics import YOLO
        self._model = YOLO(YOLO_WEIGHTS)
        self._conf = conf

    def detect_frames(
        self,
        frames: List[np.ndarray],
        start_frame_index: int,
        fps: float,
        ref_sigs: Optional[Dict[int, List[float]]] = None,
        on_court=None,
    ) -> List[PlayerFrame]:
        """Run player detection on a list of BGR frames.

        ``ref_sigs`` {1: sig, 2: sig} — when given, each detection is assigned to
        the named player whose shirt-colour signature it best matches (identity
        is locked by appearance, so the two players never swap). Otherwise a
        position-based centroid tracker assigns IDs.

        ``on_court`` — optional predicate (feet_x, feet_y) -> bool to drop
        spectators/bystanders before assignment.

        Returns one PlayerFrame per input frame.
        """
        from .identity import color_signature, signature_similarity

        tracker = _CentroidTracker() if not ref_sigs else None
        results_list: List[PlayerFrame] = []

        for i, frame in enumerate(frames):
            fi = start_frame_index + i
            ts = fi / fps
            result = self._model(
                frame, classes=[0], conf=self._conf, verbose=False, device=""
            )[0]

            boxes_raw: List[Tuple] = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                feet_y = y2
                if on_court is not None and not on_court(cx, feet_y):
                    continue  # spectator / off-court person
                boxes_raw.append((x1, y1, x2, y2, conf, cx, cy))

            if ref_sigs:
                # Appearance-locked assignment: keep the best box per named player.
                # Consider more candidates (4) since colour, not position, decides.
                cands = sorted(boxes_raw, key=lambda b: -b[4])[:4]
                pf = PlayerFrame(fi, ts)
                best_for = {1: (None, -1.0), 2: (None, -1.0)}
                for b in cands:
                    sig = color_signature(frame, (b[0], b[1], b[2], b[3]))
                    for pid in (1, 2):
                        sim = signature_similarity(sig, ref_sigs.get(pid, []))
                        if sim > best_for[pid][1]:
                            best_for[pid] = (b, sim)
                used = set()
                for pid in (1, 2):
                    b, sim = best_for[pid]
                    if b is None or id(b) in used or sim <= 0:
                        continue
                    used.add(id(b))
                    pf.players.append(PlayerBox(fi, pid, b[0], b[1], b[2], b[3], b[4]))
                results_list.append(pf)
                continue

            # Position-based fallback (no reference signatures)
            boxes_raw = sorted(boxes_raw, key=lambda b: -b[4])[:2]
            if not boxes_raw:
                results_list.append(PlayerFrame(fi, ts))
                continue
            centroids = [(b[5], b[6]) for b in boxes_raw]
            ids = tracker._assign(centroids)
            pf = PlayerFrame(fi, ts)
            for idx, (x1, y1, x2, y2, conf, cx, cy) in enumerate(boxes_raw):
                pid = ids[idx] if idx < len(ids) and ids[idx] is not None else idx + 1
                if 1 <= pid <= 2:
                    pf.players.append(PlayerBox(fi, pid, x1, y1, x2, y2, conf))
            results_list.append(pf)

        return results_list


# ─── court-aware stats ─────────────────────────────────────────────────────────

def compute_player_court_stats(
    player_frames: List[PlayerFrame],
    calibration,          # CourtCalibration (optional, can be None)
    fps: float,
) -> Dict:
    """Summarise player positions across a span.

    Returns per-player stats:
      - time_in_zone: {front/mid/back -> seconds}
      - t_control_pct: % of frames within 1.5m of the T
      - avg_court_x, avg_court_y: mean position in metres
      - in_service_box_frames: frames where feet are in either service box
    """
    from .court import CourtModel, SHORT_LINE_Y, HALF_COURT_X, SERVICE_BOX, COURT_WIDTH, T_POINT

    stats: Dict[str, Dict] = {
        "1": {"frames": 0, "t_frames": 0, "service_box_frames": 0,
              "zone": {"front": 0, "mid": 0, "back": 0},
              "court_x_sum": 0.0, "court_y_sum": 0.0},
        "2": {"frames": 0, "t_frames": 0, "service_box_frames": 0,
              "zone": {"front": 0, "mid": 0, "back": 0},
              "court_x_sum": 0.0, "court_y_sum": 0.0},
    }

    model = CourtModel(calibration) if calibration else None

    for pf in player_frames:
        for pb in pf.players:
            pid = str(pb.player_id)
            if pid not in stats:
                continue
            s = stats[pid]
            s["frames"] += 1

            if model is None:
                continue

            cx, cy = model.to_court((pb.feet_x, pb.feet_y))
            if not (0 <= cx <= COURT_WIDTH and 0 <= cy <= 9.75):
                continue

            s["court_x_sum"] += cx
            s["court_y_sum"] += cy

            # Zone
            z = model.depth_zone((cx, cy))
            s["zone"][z] += 1

            # T control (within 1.5m of T)
            if model.distance_to_t((cx, cy)) <= 1.5:
                s["t_frames"] += 1

            # Service box — left or right, behind short line
            in_left = (0 <= cx <= SERVICE_BOX and SHORT_LINE_Y <= cy <= SHORT_LINE_Y + SERVICE_BOX)
            in_right = (COURT_WIDTH - SERVICE_BOX <= cx <= COURT_WIDTH and SHORT_LINE_Y <= cy <= SHORT_LINE_Y + SERVICE_BOX)
            if in_left or in_right:
                s["service_box_frames"] += 1

    out: Dict[str, Dict] = {}
    for pid, s in stats.items():
        f = max(1, s["frames"])
        out[pid] = {
            "frames_detected": s["frames"],
            "t_control_pct": round(s["t_frames"] / f * 100, 1),
            "service_box_pct": round(s["service_box_frames"] / f * 100, 1),
            "avg_court_x": round(s["court_x_sum"] / f, 2) if model else None,
            "avg_court_y": round(s["court_y_sum"] / f, 2) if model else None,
            "zone_pct": {
                z: round(s["zone"][z] / f * 100, 1)
                for z in ("front", "mid", "back")
            },
        }
    return out


# ─── court control (tactical movement analysis) ───────────────────────────────

# 3×3 tactical grid in court metres.
def _zone_name(cx: float, cy: float) -> str:
    from .court import COURT_WIDTH, COURT_LENGTH
    col = "left" if cx < COURT_WIDTH / 3 else ("right" if cx > 2 * COURT_WIDTH / 3 else "center")
    row = "front" if cy < COURT_LENGTH / 3 else ("back" if cy > 2 * COURT_LENGTH / 3 else "mid")
    return f"{row}-{col}"


ALL_ZONES = [f"{r}-{c}" for r in ("front", "mid", "back") for c in ("left", "center", "right")]


def compute_court_control(
    player_frames: List[PlayerFrame],
    calibration,
    fps: float,
    grid: int = 10,
) -> Dict:
    """Rich tactical court-control metrics per player + a head-to-head comparison.

    Requires a court calibration to produce court-metre metrics. Without it,
    returns frames_detected only (positions stay in pixel space, not comparable).

    Per player:
      t_control_pct       % of frames within 1.5 m of the T
      avg_dist_from_t_m   mean distance from the T (lower = more dominant)
      total_distance_m    total court distance travelled (work rate)
      court_coverage_pct  % of a grid×grid floor grid the player visited
      zone_pct            occupancy % across the 9 tactical zones
      dominant_zone       most-occupied zone
      back_corner_pct     time in back-left + back-right (on the defensive)
      front_pct/mid/back  depth distribution
    """
    from .court import CourtModel, COURT_WIDTH, COURT_LENGTH

    if not calibration:
        # No homography → can't map to metres. Report detection counts only.
        counts = {"1": 0, "2": 0}
        for pf in player_frames:
            for pb in pf.players:
                counts[str(pb.player_id)] = counts.get(str(pb.player_id), 0) + 1
        return {
            "calibrated": False,
            "players": {pid: {"frames_detected": n} for pid, n in counts.items()},
            "comparison": {},
            "insights": ["Calibrate the court to unlock T-control, coverage, and tactical zone analysis."],
        }

    model = CourtModel(calibration)

    acc: Dict[str, Dict] = {}
    for pid in ("1", "2"):
        acc[pid] = {
            "frames": 0, "t_frames": 0, "dist_from_t_sum": 0.0,
            "total_dist": 0.0, "last_xy": None,
            "zones": {z: 0 for z in ALL_ZONES},
            "grid_cells": set(),
        }

    for pf in player_frames:
        for pb in pf.players:
            pid = str(pb.player_id)
            if pid not in acc:
                continue
            cx, cy = model.to_court((pb.feet_x, pb.feet_y))
            if not (0 <= cx <= COURT_WIDTH and 0 <= cy <= COURT_LENGTH):
                continue
            a = acc[pid]
            a["frames"] += 1

            d_t = model.distance_to_t((cx, cy))
            a["dist_from_t_sum"] += d_t
            if d_t <= 1.5:
                a["t_frames"] += 1

            a["zones"][_zone_name(cx, cy)] += 1
            gx = min(grid - 1, int(cx / COURT_WIDTH * grid))
            gy = min(grid - 1, int(cy / COURT_LENGTH * grid))
            a["grid_cells"].add((gx, gy))

            # Distance travelled (cap per-frame jump to suppress detection noise)
            if a["last_xy"] is not None:
                step = float(np.hypot(cx - a["last_xy"][0], cy - a["last_xy"][1]))
                if step < 1.5:   # >1.5 m between frames = a tracking jump, skip
                    a["total_dist"] += step
            a["last_xy"] = (cx, cy)

    players_out: Dict[str, Dict] = {}
    for pid, a in acc.items():
        f = max(1, a["frames"])
        zone_pct = {z: round(a["zones"][z] / f * 100, 1) for z in ALL_ZONES}
        dominant = max(a["zones"], key=lambda z: a["zones"][z]) if a["frames"] else None
        back_corner = round((a["zones"]["back-left"] + a["zones"]["back-right"]) / f * 100, 1)
        front_pct = round(sum(a["zones"][f"front-{c}"] for c in ("left", "center", "right")) / f * 100, 1)
        mid_pct = round(sum(a["zones"][f"mid-{c}"] for c in ("left", "center", "right")) / f * 100, 1)
        back_pct = round(sum(a["zones"][f"back-{c}"] for c in ("left", "center", "right")) / f * 100, 1)
        players_out[pid] = {
            "frames_detected": a["frames"],
            "t_control_pct": round(a["t_frames"] / f * 100, 1),
            "avg_dist_from_t_m": round(a["dist_from_t_sum"] / f, 2),
            "total_distance_m": round(a["total_dist"], 1),
            "court_coverage_pct": round(len(a["grid_cells"]) / (grid * grid) * 100, 1),
            "zone_pct": zone_pct,
            "dominant_zone": dominant,
            "back_corner_pct": back_corner,
            "depth_pct": {"front": front_pct, "mid": mid_pct, "back": back_pct},
        }

    comparison, insights = _court_control_insights(players_out)
    return {
        "calibrated": True,
        "players": players_out,
        "comparison": comparison,
        "insights": insights,
    }


def _court_control_insights(players: Dict[str, Dict]) -> Tuple[Dict, List[str]]:
    """Derive head-to-head comparison + plain-English tactical insights."""
    p1, p2 = players.get("1"), players.get("2")
    if not p1 or not p2 or p1["frames_detected"] < 10 or p2["frames_detected"] < 10:
        return {}, ["Not enough player detections for a reliable comparison."]

    comparison: Dict = {}
    insights: List[str] = []

    # T-control
    t1, t2 = p1["t_control_pct"], p2["t_control_pct"]
    comparison["t_control"] = {"1": t1, "2": t2,
                               "leader": "1" if t1 > t2 else "2"}
    if abs(t1 - t2) >= 8:
        leader = "Player 1" if t1 > t2 else "Player 2"
        other = "Player 2" if t1 > t2 else "Player 1"
        insights.append(
            f"{leader} controls the T ({max(t1,t2)}% vs {min(t1,t2)}%) — "
            f"on the attack while {other} retrieves."
        )
    else:
        insights.append(f"T-control is contested ({t1}% vs {t2}%) — an even, positional battle.")

    # Who gets pushed to the back corners
    b1, b2 = p1["back_corner_pct"], p2["back_corner_pct"]
    if abs(b1 - b2) >= 8:
        pushed = "Player 1" if b1 > b2 else "Player 2"
        insights.append(
            f"{pushed} is pushed into the back corners more "
            f"({max(b1,b2)}% vs {min(b1,b2)}%) — pressure them deep, then attack short."
        )

    # Work rate (distance covered)
    d1, d2 = p1["total_distance_m"], p2["total_distance_m"]
    comparison["distance"] = {"1": d1, "2": d2}
    if max(d1, d2) > 0 and abs(d1 - d2) / max(d1, d2) >= 0.15:
        worker = "Player 1" if d1 > d2 else "Player 2"
        insights.append(
            f"{worker} covered more ground ({max(d1,d2)} m vs {min(d1,d2)} m) — "
            f"being run around the court; keep moving them."
        )

    # Court coverage (territory)
    c1, c2 = p1["court_coverage_pct"], p2["court_coverage_pct"]
    comparison["coverage"] = {"1": c1, "2": c2}

    # Dominant zones
    comparison["dominant_zone"] = {"1": p1["dominant_zone"], "2": p2["dominant_zone"]}

    return comparison, insights


def get_player_detector() -> PlayerDetector:
    return PlayerDetector()
