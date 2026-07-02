"""
Ball-trace rendering — produce an annotated video with a glowing comet trail
following the ball's trajectory.

Runs the setup-specific trained ball model over a window, takes the per-frame
detection, optionally smooths it (reject physically-implausible jumps so the trail
never flickers onto a false positive), then draws a fading yellow trail + a bright
current-ball marker on each frame and encodes an mp4.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np

from .ball import get_ball_detector
from .annotation import PROC_MAX_WIDTH


def _smooth_positions(positions: Dict[int, tuple], fps: float,
                      max_speed_px: float = 80.0) -> Dict[int, tuple]:
    """Drop detections that jump implausibly far from the recent path (likely
    false positives), keeping the trail coherent. Keyed by frame index in window."""
    out: Dict[int, tuple] = {}
    last = None
    for i in sorted(positions):
        x, y, sc = positions[i]
        if last is not None:
            li, (lx, ly) = last
            gap = max(1, i - li)
            if np.hypot(x - lx, y - ly) > max_speed_px * gap and sc < 0.9:
                continue  # implausible jump and not super-confident → skip
        out[i] = (x, y, sc)
        last = (i, (x, y))
    return out


def trace_ball_video(
    video_path: str,
    out_path: str,
    start_s: float = 0.0,
    duration_s: float = 8.0,
    setup: str = "phone",
    smooth: bool = True,
    trail_len: int = 14,
    slow_factor: float = 2.0,
) -> Dict:
    """Render a ball-trace video for a window and write it to ``out_path``."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = int(start_s * fps)
    n_frames = int(duration_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(n_frames):
        ret, f = cap.read()
        if not ret:
            break
        nh, nw = f.shape[:2]
        if nw > PROC_MAX_WIDTH:
            f = cv2.resize(f, (PROC_MAX_WIDTH, int(round(nh * PROC_MAX_WIDTH / nw))))
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        return {"error": "not enough frames", "frame_count": len(frames)}

    detector = get_ball_detector(setup)
    per = detector.detect_window(frames, start_f, fps)
    positions = {i: (int(c[0].x), int(c[0].y), float(c[0].score))
                 for i, c in enumerate(per) if c}
    if smooth:
        positions = _smooth_positions(positions, fps)

    H, W = frames[0].shape[:2]
    # Try a browser-friendly H.264 codec first, fall back to mp4v.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"),
                             max(1.0, fps / slow_factor), (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps / slow_factor), (W, H))

    trail: deque = deque(maxlen=trail_len)
    for i, f in enumerate(frames):
        vis = f.copy()
        if i in positions:
            x, y, _ = positions[i]
            trail.append((x, y))
        n = len(trail)
        # fading comet (older = dimmer + smaller)
        for k, (tx, ty) in enumerate(trail):
            a = (k + 1) / n
            overlay = vis.copy()
            cv2.circle(overlay, (tx, ty), int(2 + 5 * a), (0, int(255 * a), int(255 * a)), -1)
            vis = cv2.addWeighted(overlay, a * 0.7, vis, 1 - a * 0.7, 0)
        for k in range(1, n):
            cv2.line(vis, trail[k - 1], trail[k], (0, 230, 230), 2)
        # current ball marker
        if i in positions:
            x, y, _ = positions[i]
            cv2.circle(vis, (x, y), 10, (0, 255, 255), 2)
            cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
        writer.write(vis)
    writer.release()

    return {
        "frame_count": len(frames),
        "detections": len(positions),
        "detection_rate": round(len(positions) / max(1, len(frames)), 3),
        "setup": setup,
        "smoothed": smooth,
        "out_path": out_path,
    }
