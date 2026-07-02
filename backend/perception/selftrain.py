"""
Self-training (active learning) — the scaling engine for the ball model.

The trained model runs over unlabelled footage and proposes ball trajectories;
the BallTracker's physics scoring (smooth, fast, large-displacement arcs) filters
those proposals down to the ones that *move like a ball*. A human then approves or
rejects each proposed track in bulk:

    approve -> added as ground-truth ball points (label="ball", source="selftrain")
    reject  -> kept as a HARD NEGATIVE   (label="not_ball")  — teaches the model
               what isn't the ball (the body/glass false positives it makes)

This is supervised self-training: the machine labels at scale, the human only
judges — far cheaper than frame-by-frame, and the verification step stops the
model from teaching itself its own mistakes (the reflection-arc trap).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import base64

from .ball import BallTracker, get_ball_detector
from .annotation import _crop_b64, PROC_MAX_WIDTH


def _arc_overlay_b64(frame, points, max_width: int = 720) -> str:
    """Draw the proposed ball arc on a real frame so a reviewer can judge it in
    context (is the path on the court like a ball, or on a body / the glass?)."""
    import cv2

    vis = frame.copy()
    pts = [(int(p["x"]), int(p["y"])) for p in points]
    for k in range(1, len(pts)):
        cv2.line(vis, pts[k - 1], pts[k], (0, 255, 0), 2)
    for i, p in enumerate(pts):
        # start green, end red — shows direction of travel
        color = (0, 0, 255) if i == len(pts) - 1 else (60, 220, 60)
        cv2.circle(vis, p, 4, color, -1)
    if pts:
        cv2.circle(vis, pts[0], 9, (0, 255, 255), 2)  # ring the start
    h, w = vis.shape[:2]
    if w > max_width:
        vis = cv2.resize(vis, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8") if ok else None


def mine_ball_tracks(
    video_path: str,
    start_s: float = 0.0,
    duration_s: float = 20.0,
    min_quality: float = 80.0,
    max_tracks: int = 12,
    setup: str = "phone",
) -> Dict:
    """Run the trained ball model over a window and propose physics-valid arcs.

    Uses ``get_ball_detector(setup)`` — the model specialised for this camera setup
    (phone vs broadcast). Tracks below ``min_quality`` — i.e. that don't move like a
    ball — are discarded; the rest are returned for human review.
    """
    import cv2

    detector = get_ball_detector(setup)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = int(start_s * fps)
    n_frames = int(duration_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames: List[np.ndarray] = []
    scale = 1.0  # proc px -> native px (high-res clips are processed downscaled)
    for _ in range(n_frames):
        ret, f = cap.read()
        if not ret:
            break
        nh, nw = f.shape[:2]
        if nw > PROC_MAX_WIDTH:
            f = cv2.resize(f, (PROC_MAX_WIDTH, int(round(nh * PROC_MAX_WIDTH / nw))))
            scale = nw / float(f.shape[1])
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        return {"fps": fps, "detector": type(detector).__name__,
                "num_proposals": 0, "proposals": []}

    # Model detections per frame → link into trajectories → physics gate.
    per_frame = detector.detect_window(frames, start_f, fps)
    tracker = BallTracker()
    raw_tracks = tracker.build_tracks(per_frame)
    scored = [(tracker._track_quality(t), t) for t in raw_tracks]
    scored = [(q, t) for q, t in scored if q >= min_quality]
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:max_tracks]

    proposals = []
    for tid, (q, t) in enumerate(scored):
        pts = []          # stored points in NATIVE coords (for training)
        proc_pts = []     # proc coords, for drawing crops/overlay
        scores = []
        for p in t.points:
            li = p.frame_index - start_f
            frame = frames[li] if 0 <= li < len(frames) else None
            crop = _crop_b64(frame, p.x, p.y) if frame is not None else None  # proc coords
            pts.append({
                "frame_index": p.frame_index,
                "timestamp": round(p.timestamp, 3),
                "x": round(p.x * scale, 1), "y": round(p.y * scale, 1),       # native
                "crop_b64": crop,
            })
            proc_pts.append({"x": p.x, "y": p.y})
            scores.append(getattr(p, "score", 0.0) or 0.0)
        # Context overlay: draw the arc on the (proc) frame at the track's midpoint.
        mid = t.points[len(t.points) // 2]
        mli = mid.frame_index - start_f
        overlay = (
            _arc_overlay_b64(frames[mli], proc_pts)
            if 0 <= mli < len(frames) else None
        )
        proposals.append({
            "track_id": tid,
            "quality": round(float(q), 1),
            "mean_confidence": round(float(np.mean(scores)) if scores else 0.0, 3),
            "num_points": len(pts),
            "overlay_b64": overlay,
            "points": pts,
        })

    return {
        "fps": fps,
        "detector": type(detector).__name__,
        "start_s": start_s,
        "duration_s": duration_s,
        "num_proposals": len(proposals),
        "proposals": proposals,
    }


def scan_video_for_arcs(
    video_path: str,
    n_windows: int = 4,
    window_s: float = 8.0,
    min_quality: float = 80.0,
    max_per_window: int = 3,
    setup: str = "phone",
) -> Dict:
    """Mine several windows spread across a video to actually catch rallies.

    A single fixed window often lands between rallies and finds nothing; scanning
    evenly-spaced windows (skipping the very start/end) gives the model real play
    to propose from. Returns the combined proposals.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = total / fps if fps else 0
    if dur < window_s * 2:
        return mine_ball_tracks(video_path, 0, min(window_s, dur), min_quality, setup=setup)

    # evenly spaced starts across the middle 80% of the video
    usable = dur * 0.8
    starts = [dur * 0.1 + usable * (k / max(1, n_windows - 1)) for k in range(n_windows)]

    all_props = []
    detector_name = None
    for s in starts:
        r = mine_ball_tracks(video_path, s, window_s, min_quality, max_tracks=max_per_window, setup=setup)
        detector_name = r.get("detector")
        all_props.extend(r.get("proposals", []))
    # renumber + keep best overall
    all_props.sort(key=lambda p: p["quality"], reverse=True)
    for i, p in enumerate(all_props):
        p["track_id"] = i
    return {"detector": detector_name, "num_proposals": len(all_props),
            "proposals": all_props}
