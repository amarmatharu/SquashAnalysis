"""
Rally segmentation — when does each rally start and end.

No training: this is *logic* on top of the ball trajectory. A rally is a stretch
of sustained ball motion (the ball flying between players); between rallies the
ball is stationary (on the floor) or absent (players walking, picking it up). So
we detect frames where the ball is *moving fast*, group those into intervals
(merging brief gaps), and drop blips too short to be a rally.

Memory-safe: the video is read in chunks so a long span never holds all frames at
once — only the tiny per-frame ball positions are kept.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .ball import get_ball_detector, BallTracker
from .annotation import PROC_MAX_WIDTH


def _stream_ball_positions(video_path: str, start_f: int, n_frames: int,
                           detector, fps: float, chunk: int = 120) -> Dict[int, tuple]:
    """Detect ball positions across a span and keep only those on a *coherent*
    trajectory (a smooth, fast arc — i.e. an actual ball in flight), discarding
    scattered per-frame false positives. Reads in chunks (bounded memory).
    Returns {frame_index: (x, y)} for in-flight frames only."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    positions: Dict[int, tuple] = {}
    read = 0
    fi = start_f
    # Overlap chunks slightly so arcs aren't cut at boundaries.
    overlap = 8
    while read < n_frames:
        frames = []
        for _ in range(min(chunk, n_frames - read)):
            ret, f = cap.read()
            if not ret:
                break
            nh, nw = f.shape[:2]
            if nw > PROC_MAX_WIDTH:
                f = cv2.resize(f, (PROC_MAX_WIDTH, int(round(nh * PROC_MAX_WIDTH / nw))))
            frames.append(f)
        if len(frames) < 5:
            break
        per = detector.detect_window(frames, fi, fps)
        # Keep only frames on a physically coherent ball arc — filters false
        # positives from player limbs that would otherwise keep the "in-play"
        # signal continuously high between points.
        for traj in BallTracker().all_trajectories(per):
            for p in traj:
                positions[p.frame_index] = (p.x, p.y, 1.0)
        advance = max(1, len(frames) - overlap)
        fi += advance
        read += advance
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    cap.release()
    return positions


def extract_rally_clip(video_path: str, start_t: float, end_t: float,
                       out_path: str, buffer_s: float = 0.4) -> bool:
    """Write a clip from (start_t - buffer) to (end_t + buffer) to out_path.
    Small buffer: audio rally boundaries already include serve lead / settle tail,
    so a tight clip avoids bleeding the reset of the next rally into this one.
    Returns True on success."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_start = max(0, int((start_t - buffer_s) * fps))
    clip_end = min(total - 1, int((end_t + buffer_s) * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    ret, first = cap.read()
    if not ret:
        cap.release()
        return False

    nh, nw = first.shape[:2]
    if nw > PROC_MAX_WIDTH:
        scale = PROC_MAX_WIDTH / nw
        nw, nh = PROC_MAX_WIDTH, int(nh * scale)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (nw, nh))
    if not writer.isOpened():
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (nw, nh))

    def _write(f):
        if f.shape[1] != nw or f.shape[0] != nh:
            f = cv2.resize(f, (nw, nh))
        writer.write(f)

    _write(first)
    for _ in range(clip_end - clip_start):
        ret, f = cap.read()
        if not ret:
            break
        _write(f)

    cap.release()
    writer.release()
    return True


def _count_direction_changes(pts: List[tuple], min_turn_deg: float = 90.0,
                             min_seg_px: float = 12.0, min_spacing: int = 6) -> int:
    """Approximate shot count: sharp ball-path reversals = contacts.

    The raw path is jittery, so we (1) smooth it, (2) require a real reversal
    (>=90°), (3) require meaningful travel either side (not micro-noise), and
    (4) space counted shots apart — otherwise jitter inflates the count wildly.
    """
    if len(pts) < 5:
        return 0
    arr = np.array(pts, dtype=float)
    # 5-point moving-average smoothing
    k = 2
    sm = np.array([arr[max(0, i - k):i + k + 1].mean(axis=0) for i in range(len(arr))])
    n = 0
    last = -min_spacing
    for i in range(1, len(sm) - 1):
        v1 = sm[i] - sm[i - 1]
        v2 = sm[i + 1] - sm[i]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < min_seg_px or l2 < min_seg_px:
            continue
        cos = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
        if np.degrees(np.arccos(cos)) >= min_turn_deg and i - last >= min_spacing:
            n += 1
            last = i
    return n


def segment_rallies(
    video_path: str,
    start_s: float = 0.0,
    duration_s: float = 60.0,
    setup: str = "phone",
    min_speed_px_s: float = 250.0,
    gap_s: float = 3.0,
    min_rally_s: float = 1.5,
) -> Dict:
    """Segment a video span into rallies from sustained ball motion.

    ``min_speed_px_s`` is how fast the ball must move to count as "in play";
    ``gap_s`` merges brief detection gaps *within* a rally but should NOT span
    the inter-point pause (players pick up ball, server bounces it ~3-10s);
    ``min_rally_s`` drops blips too short to be a real rally.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    start_f = int(start_s * fps)
    n_frames = min(int(duration_s * fps), max(0, total - start_f))

    detector = get_ball_detector(setup)
    positions = _stream_ball_positions(video_path, start_f, n_frames, detector, fps)
    det = sorted(positions.items())  # [(fi, (x,y,score)), ...]

    # Mark times where the ball is moving fast (in play).
    active = []  # (time_s, x, y)
    for k in range(1, len(det)):
        f0, (x0, y0, _) = det[k - 1]
        f1, (x1, y1, _) = det[k]
        dt = (f1 - f0) / fps
        if dt <= 0 or dt > 0.5:   # gap too large to be continuous flight
            continue
        speed = float(np.hypot(x1 - x0, y1 - y0) / dt)
        if speed >= min_speed_px_s:
            active.append((f1 / fps, x1, y1))

    # Group active samples into rallies, merging gaps shorter than gap_s.
    rallies = []
    cur: Optional[Dict] = None
    for t, x, y in active:
        if cur and t - cur["_last"] <= gap_s:
            cur["_last"] = t
            cur["pts"].append((x, y))
        else:
            if cur:
                rallies.append(cur)
            cur = {"start_t": t, "_last": t, "pts": [(x, y)]}
    if cur:
        rallies.append(cur)

    out = []
    for i, r in enumerate(r for r in rallies if r["_last"] - r["start_t"] >= min_rally_s):
        shots = _count_direction_changes(r["pts"])
        out.append({
            "rally_id": i + 1,
            "start_t": round(r["start_t"], 2),
            "end_t": round(r["_last"], 2),
            "duration_s": round(r["_last"] - r["start_t"], 2),
            "shots": shots,
            "ball_samples": len(r["pts"]),
        })

    active_s = sum(r["duration_s"] for r in out)
    span = n_frames / fps if fps else 0
    return {
        "setup": setup, "fps": fps,
        "span_s": round(span, 1),
        "num_rallies": len(out),
        "active_play_s": round(active_s, 1),
        "active_play_pct": round(active_s / span * 100, 1) if span else 0,
        "ball_detections": len(positions),
        "rallies": out,
    }
