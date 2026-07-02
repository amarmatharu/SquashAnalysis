"""
Phase 3 — Event annotation on top of ball-based rally segmentation.

Rally *boundaries* still come from the ball trajectory (Phase 2 approach —
ball-only with BallTracker arc filtering). What Phase 3 adds on top:

  1. Tin detection  — ball crosses below the tin line during a rally
                      → other player automatically wins the point
  2. Serve detection — who is standing in the service box at the start of
                       each rally → who is the server
  3. Court-zone tagging — where on court is the ball during each rally

Phase 3 DOES NOT try to use player motion for boundary detection. That
requires a better ball model or floor-bounce detection (Phase 4+).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── tin hit detector ─────────────────────────────────────────────────────────

def _detect_tin_hits(
    per_frame_ball,        # list of per-frame candidate lists (absolute indexed)
    start_frame: int,
    fps: float,
    calibration,           # CourtCalibration in PROCESSED-frame pixels
    margin_px: float = 14.0,
    min_speed_px_s: float = 150.0,
) -> List[float]:
    """Return timestamps (s) of likely tin hits.

    A tin hit = ball crosses below the tin line while moving fast (in flight).
    We detect the TRANSITION (not below → below) to avoid repeated triggers.

    ``calibration`` must already be in processed-frame pixel space — the same
    space the ball candidates live in (max PROC_MAX_WIDTH wide).
    """
    if calibration is None or calibration.tin_left is None:
        return []

    tin_l = (calibration.tin_left[0], calibration.tin_left[1])
    tin_r = (calibration.tin_right[0], calibration.tin_right[1])

    # Sanity gate: the tin sits on the FRONT WALL, which in the image is the
    # line through front_left/front_right. The tin must be at or just above that
    # line (smaller y). If the marked tin line is well BELOW the front wall, it
    # was mis-placed in mid-court — refuse to emit (meaningless) hits.
    front_y = min(calibration.front_left[1], calibration.front_right[1])
    tin_y_avg = (tin_l[1] + tin_r[1]) / 2
    # Allow the tin to be a little below the floor corner (perspective), but not
    # more than ~15% of frame height into the court.
    frame_h_guess = max(calibration.back_left[1], calibration.back_right[1], 1)
    if tin_y_avg > front_y + 0.15 * frame_h_guess:
        return []

    def tin_y_at(px_x: float) -> float:
        x0, y0 = tin_l
        x1, y1 = tin_r
        if abs(x1 - x0) < 1:
            return float(y0)
        t = np.clip((px_x - x0) / (x1 - x0), 0.0, 1.0)
        return float(y0 + t * (y1 - y0))

    # Ball speeds for "in-flight" filter
    pts = []
    for local_fi, cands in enumerate(per_frame_ball):
        if cands:
            pts.append((start_frame + local_fi, cands[0].x, cands[0].y))

    speeds: Dict[int, float] = {}
    for k in range(1, len(pts)):
        f0, x0, y0 = pts[k - 1]
        f1, x1, y1 = pts[k]
        gap = f1 - f0
        if 1 <= gap <= 8:
            spd = float(np.hypot(x1 - x0, y1 - y0) / (gap / fps))
            for fi in range(f0, f1 + 1):
                speeds[fi] = max(speeds.get(fi, 0), spd)

    hits: List[float] = []
    prev_below = False

    for local_fi, cands in enumerate(per_frame_ball):
        if not cands:
            prev_below = False
            continue
        fi = start_frame + local_fi
        bx, by = cands[0].x, cands[0].y
        ty = tin_y_at(bx)
        below = by >= (ty - margin_px)
        if below and not prev_below and speeds.get(fi, 0) > min_speed_px_s:
            ts = round(fi / fps, 2)
            if not hits or ts - hits[-1] > 0.5:
                hits.append(ts)
        prev_below = below

    return hits


# ─── serve detector ───────────────────────────────────────────────────────────

def _detect_server(
    rally_start_t: float,
    player_frames,
    calibration,
    fps: float,
    window_s: float = 2.0,
) -> Optional[int]:
    """Return which player (1 or 2) is in the service box at the rally start.

    Looks at the first `window_s` seconds of the rally. If the calibration has
    a floor homography, maps player feet to court coordinates and checks against
    the service box regions. Returns 1 or 2, or None if indeterminate.
    """
    if calibration is None or not player_frames:
        return None

    try:
        from .court import CourtModel, SHORT_LINE_Y, SERVICE_BOX, COURT_WIDTH
        model = CourtModel(calibration)
    except Exception:
        return None

    start_f = int(rally_start_t * fps)
    end_f = int((rally_start_t + window_s) * fps)

    votes: Dict[int, int] = {}
    for pf in player_frames:
        if not (start_f <= pf.frame_index <= end_f):
            continue
        for pb in pf.players:
            cx, cy = model.to_court((pb.feet_x, pb.feet_y))
            in_left = (0 <= cx <= SERVICE_BOX and SHORT_LINE_Y <= cy <= SHORT_LINE_Y + SERVICE_BOX)
            in_right = (COURT_WIDTH - SERVICE_BOX <= cx <= COURT_WIDTH and
                        SHORT_LINE_Y <= cy <= SHORT_LINE_Y + SERVICE_BOX)
            if in_left or in_right:
                votes[pb.player_id] = votes.get(pb.player_id, 0) + 1

    if not votes:
        return None
    return max(votes, key=lambda k: votes[k])


# ─── direction change counter (shared with rallies.py) ────────────────────────

def _count_direction_changes(pts: List[tuple], min_turn_deg: float = 90.0,
                              min_seg_px: float = 12.0, min_spacing: int = 6) -> int:
    if len(pts) < 5:
        return 0
    arr = np.array(pts, dtype=float)
    k = 2
    sm = np.array([arr[max(0, i - k):i + k + 1].mean(axis=0) for i in range(len(arr))])
    n, last = 0, -min_spacing
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


# ─── main entry point ─────────────────────────────────────────────────────────

def segment_rallies_v2(
    video_path: str,
    start_s: float = 0.0,
    duration_s: float = 60.0,
    setup: str = "phone",
    calibration=None,
    chunk: int = 90,
    ball_speed_thresh: float = 200.0,
    gap_s: float = 3.0,
    min_rally_s: float = 1.5,
    # player_slow_thresh and gap_s_moving kept for signature compat but unused
    player_slow_thresh: float = 4.0,
    gap_s_paused: float = 2.5,
    gap_s_moving: float = 5.0,
) -> Dict:
    """
    Event-annotated rally segmentation.

    Boundaries: ball-only (BallTracker arcs, gap_s=3.0) — same reliable logic
                as v1 but with cleaner arc filtering.
    Events added on top:
      • tin_hits: timestamps where ball crossed the tin line during this rally
      • end_reason: 'tin' if a tin hit ended it, else 'unknown'
      • server: which player (1|2) was in the service box at rally start
    """
    import cv2
    from .ball import get_ball_detector, BallTracker
    from .annotation import PROC_MAX_WIDTH

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    start_f = int(start_s * fps)
    n_frames = min(int(duration_s * fps), max(0, total - start_f))
    proc_w = min(native_w, PROC_MAX_WIDTH)
    proc_h = int(native_h * proc_w / native_w) if native_w > PROC_MAX_WIDTH else native_h

    # The ball/player detectors run on PROCESSED frames (resized to proc_w), but
    # the calibration corners are in NATIVE pixels. Scale the calibration down to
    # processed-frame space so tin/serve checks share the detectors' coordinates.
    proc_calib = None
    if calibration is not None:
        scale = proc_w / native_w if native_w else 1.0
        from .court import CourtCalibration

        def _sc(pt):
            return (pt[0] * scale, pt[1] * scale) if pt is not None else None

        proc_calib = CourtCalibration(
            front_left=_sc(calibration.front_left),
            front_right=_sc(calibration.front_right),
            back_right=_sc(calibration.back_right),
            back_left=_sc(calibration.back_left),
            tin_left=_sc(calibration.tin_left),
            tin_right=_sc(calibration.tin_right),
        )

    ball_det = get_ball_detector(setup)

    # ── Pass 1: collect ball candidates ───────────────────────────────────────
    all_ball_per_frame: List = []

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    read, fi = 0, start_f

    while read < n_frames:
        frames = []
        for _ in range(min(chunk, n_frames - read)):
            ret, f = cap.read()
            if not ret:
                break
            h, w = f.shape[:2]
            if w > PROC_MAX_WIDTH:
                f = cv2.resize(f, (PROC_MAX_WIDTH, int(h * PROC_MAX_WIDTH / w)))
            frames.append(f)
        if len(frames) < 3:
            break
        all_ball_per_frame.extend(ball_det.detect_window(frames, fi, fps))
        fi += len(frames)
        read += len(frames)
    cap.release()

    # ── Pass 2: BallTracker arc filter → coherent ball speeds ─────────────────
    ball_speeds: Dict[int, float] = {}
    for traj in BallTracker().all_trajectories(all_ball_per_frame):
        for k in range(1, len(traj)):
            a, b = traj[k - 1], traj[k]
            gap = b.frame_index - a.frame_index
            if 1 <= gap <= 8:
                dt = gap / fps
                spd = float(np.hypot(b.x - a.x, b.y - a.y) / dt)
                for f2 in range(a.frame_index, b.frame_index + 1):
                    ball_speeds[f2] = max(ball_speeds.get(f2, 0), spd)

    # ── Pass 3: ball-only state machine ───────────────────────────────────────
    gap_thresh = int(gap_s * fps)
    IN_RALLY = False
    rally_start_f: Optional[int] = None
    gap_frames = 0
    raw_rallies: List[Dict] = []

    for f in range(start_f, start_f + n_frames):
        if ball_speeds.get(f, 0) >= ball_speed_thresh:
            gap_frames = 0
            if not IN_RALLY:
                IN_RALLY = True
                rally_start_f = f
        else:
            if IN_RALLY:
                gap_frames += 1
                if gap_frames >= gap_thresh:
                    raw_rallies.append({
                        "start_f": rally_start_f,
                        "end_f": f - gap_thresh,
                    })
                    IN_RALLY = False
                    gap_frames = 0

    if IN_RALLY and rally_start_f is not None:
        raw_rallies.append({"start_f": rally_start_f, "end_f": start_f + n_frames - 1})

    # ── Pass 4: tin hits ───────────────────────────────────────────────────────
    all_tin_hits = _detect_tin_hits(
        all_ball_per_frame, start_f, fps, proc_calib
    )

    # ── Pass 5: player detection at rally boundaries (serve detection) ─────────
    # Run player detection only over the first 3s of each rally (cheap).
    all_player_frames: List = []
    if proc_calib is not None and raw_rallies:
        from .players import get_player_detector
        pdet = get_player_detector()
        cap2 = cv2.VideoCapture(video_path)
        for r in raw_rallies:
            win_end = min(r["start_f"] + int(3.0 * fps), r["end_f"])
            win_len = win_end - r["start_f"]
            if win_len < 2:
                continue
            cap2.set(cv2.CAP_PROP_POS_FRAMES, r["start_f"])
            frames = []
            for _ in range(win_len):
                ret, f = cap2.read()
                if not ret:
                    break
                h, w = f.shape[:2]
                if w > PROC_MAX_WIDTH:
                    f = cv2.resize(f, (PROC_MAX_WIDTH, int(h * PROC_MAX_WIDTH / w)))
                frames.append(f)
            if frames:
                pf_list = pdet.detect_frames(frames, r["start_f"], fps)
                all_player_frames.extend(pf_list)
        cap2.release()

    # ── Pass 6: assemble output ────────────────────────────────────────────────
    rallies_out: List[Dict] = []
    for r in raw_rallies:
        dur = (r["end_f"] - r["start_f"]) / fps
        if dur < min_rally_s:
            continue

        start_t = round(r["start_f"] / fps, 2)
        end_t = round(r["end_f"] / fps, 2)

        # Ball positions within this rally for shot counting
        pts_in_rally = []
        for local_fi, cands in enumerate(all_ball_per_frame):
            fi2 = start_f + local_fi
            if cands and r["start_f"] <= fi2 <= r["end_f"]:
                pts_in_rally.append((cands[0].x, cands[0].y))
        shots = _count_direction_changes(pts_in_rally)

        # Tin hits within this rally
        tin_hits = [t for t in all_tin_hits if start_t <= t <= end_t]
        end_reason = "tin" if tin_hits else "unknown"

        # Server (who is in the service box at rally start)
        server = _detect_server(start_t, all_player_frames, proc_calib, fps)

        rallies_out.append({
            "rally_id": len(rallies_out) + 1,
            "start_t": start_t,
            "end_t": end_t,
            "duration_s": round(end_t - start_t, 2),
            "shots": shots,
            "ball_samples": len(pts_in_rally),
            "end_reason": end_reason,
            "tin_hits": tin_hits,
            "server": server,
        })

    active_s = sum(r["duration_s"] for r in rallies_out)
    span = n_frames / fps if fps else 0

    return {
        "setup": setup,
        "fps": round(fps, 1),
        "span_s": round(span, 1),
        "num_rallies": len(rallies_out),
        "active_play_s": round(active_s, 1),
        "active_play_pct": round(active_s / span * 100, 1) if span else 0,
        "ball_detections": sum(1 for c in all_ball_per_frame if c),
        "tin_hits_total": len(all_tin_hits),
        "rallies": rallies_out,
        "method": "dual_signal_v2",
    }
