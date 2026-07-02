"""
Shot pattern & error zone analysis (Phase 5, tactical layer).

For each rally we:
  1. detect shot CONTACTS (ball-path direction reversals) — timing in pixel space,
     which is robust to the floor-homography parallax problem (the ball in flight
     is above the floor, so its mapped court position is unreliable).
  2. attribute each contact to the nearer PLAYER (the striker) and record the
     striker's COURT-ZONE — player feet are on the floor, so this maps correctly.
  3. combine with manually-tagged rally outcomes to find ERROR ZONES: where the
     player who lost the point was positioned at the end of the rally.

Outputs feed the "how to beat this opponent" reasoning: shot-origin tendencies,
the zones where each player loses points, and rally-length-by-winner.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _zone_name(cx: float, cy: float, court_w: float, court_l: float) -> str:
    col = "left" if cx < court_w / 3 else ("right" if cx > 2 * court_w / 3 else "center")
    row = "front" if cy < court_l / 3 else ("back" if cy > 2 * court_l / 3 else "mid")
    return f"{row}-{col}"


ALL_ZONES = [f"{r}-{c}" for r in ("front", "mid", "back") for c in ("left", "center", "right")]


def _detect_contacts_pixel(ball_pts: List[Tuple[int, float, float]],
                           min_turn_deg: float = 60.0,
                           min_seg_px: float = 10.0,
                           min_spacing_frames: int = 5) -> List[int]:
    """Detect shot contacts as sharp direction reversals in the pixel-space ball
    path. Returns the frame indices of contacts.

    ``ball_pts`` is a list of (frame_index, x_px, y_px), time-ordered.
    """
    if len(ball_pts) < 5:
        return []

    # Light smoothing
    idx = [p[0] for p in ball_pts]
    xy = np.array([(p[1], p[2]) for p in ball_pts], dtype=float)
    k = 1
    sm = np.array([xy[max(0, i - k):i + k + 1].mean(axis=0) for i in range(len(xy))])

    contacts: List[int] = []
    last = -min_spacing_frames
    for i in range(1, len(sm) - 1):
        v1 = sm[i] - sm[i - 1]
        v2 = sm[i + 1] - sm[i]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < min_seg_px or l2 < min_seg_px:
            continue
        cos = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
        turn = np.degrees(np.arccos(cos))
        if turn >= min_turn_deg and (idx[i] - last) >= min_spacing_frames:
            contacts.append(idx[i])
            last = idx[i]
    return contacts


def analyze_shot_patterns(
    video_path: str,
    rally_windows: List[Tuple[float, float]],   # (start_s, end_s) per rally
    outcomes: Dict[str, str],                    # rally_id(str) -> outcome
    calibration,                                  # CourtCalibration (native px) or None
    setup: str = "phone",
    ref_sigs=None,                                # {1,2 -> colour sig} locks identity
    max_seconds: float = 240.0,
    rally_ids: Optional[List[int]] = None,        # real rally_id per window; keys `outcomes`
) -> Dict:
    """Run ball+player detection over rally windows and produce shot/error patterns."""
    import cv2
    from .ball import get_ball_detector, BallTracker
    from .players import get_player_detector
    from .court import CourtModel, CourtCalibration, COURT_WIDTH, COURT_LENGTH
    from .annotation import PROC_MAX_WIDTH
    from .identity import _on_court_filter

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not calibration:
        return {"calibrated": False,
                "insights": ["Calibrate the court to unlock shot-origin and error-zone analysis."]}

    # Processed-space calibration (detectors run on resized frames)
    proc_w = min(native_w, PROC_MAX_WIDTH)
    scale = proc_w / native_w if native_w else 1.0
    on_court = _on_court_filter(calibration, native_w, native_h, proc_w)

    def _sc(pt):
        return (pt[0] * scale, pt[1] * scale) if pt is not None else None

    proc_calib = CourtCalibration(
        front_left=_sc(calibration.front_left), front_right=_sc(calibration.front_right),
        back_right=_sc(calibration.back_right), back_left=_sc(calibration.back_left),
    )
    model = CourtModel(proc_calib)

    ball_det = get_ball_detector(setup)
    player_det = get_player_detector()

    # Per-player shot-origin zone tallies + per-rally records
    shot_origin = {"1": {z: 0 for z in ALL_ZONES}, "2": {z: 0 for z in ALL_ZONES}}
    total_shots = {"1": 0, "2": 0}
    rally_records: List[Dict] = []

    processed_s = 0.0
    cap = cv2.VideoCapture(video_path)

    for wi, (ws, we) in enumerate(rally_windows):
        # Look up outcomes by the REAL rally_id — not list position. The tagging
        # UI and scouting score key outcomes by rally_id, and ids are NOT 1..N
        # after any merge/split, so positional lookup misattributes error zones.
        ri = rally_ids[wi] if rally_ids is not None else wi + 1
        if processed_s >= max_seconds:
            break
        dur = min(we - ws, max_seconds - processed_s)
        if dur <= 0.3:
            continue
        start_f = int(ws * fps)
        n = int(dur * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        frames = []
        for _ in range(n):
            ret, f = cap.read()
            if not ret:
                break
            h, w = f.shape[:2]
            if w > PROC_MAX_WIDTH:
                f = cv2.resize(f, (PROC_MAX_WIDTH, int(h * PROC_MAX_WIDTH / w)))
            frames.append(f)
        processed_s += dur
        if len(frames) < 5:
            continue

        # Ball: keep coherent arc points only (suppress false positives)
        per_ball = ball_det.detect_window(frames, start_f, fps)
        ball_pts: List[Tuple[int, float, float]] = []
        for traj in BallTracker().all_trajectories(per_ball):
            for p in traj:
                ball_pts.append((p.frame_index, p.x, p.y))
        ball_pts.sort()

        # Players: per-frame court positions (identity locked by colour if available)
        player_frames = player_det.detect_frames(frames, start_f, fps, ref_sigs, on_court)
        pf_by_index = {pf.frame_index: pf for pf in player_frames}

        # Shot contacts (frame indices)
        contacts = _detect_contacts_pixel(ball_pts)

        # Attribute each contact to nearest player; record striker zone
        rally_shots = []
        for cf in contacts:
            # Ball pixel position at contact
            bp = min(ball_pts, key=lambda p: abs(p[0] - cf))
            bx, by = bp[1], bp[2]
            pf = pf_by_index.get(cf) or _nearest_pf(player_frames, cf)
            if not pf:
                continue
            # Nearest player to the ball (in pixel space — striker)
            striker, best = None, 1e9
            for pb in pf.players:
                d = np.hypot(pb.cx - bx, pb.cy - by)
                if d < best:
                    best, striker = d, pb
            if striker is None:
                continue
            # Striker court zone (feet on floor → valid mapping)
            cx, cy = model.to_court((striker.feet_x, striker.feet_y))
            if not (0 <= cx <= COURT_WIDTH and 0 <= cy <= COURT_LENGTH):
                continue
            zone = _zone_name(cx, cy, COURT_WIDTH, COURT_LENGTH)
            pid = str(striker.player_id)
            shot_origin[pid][zone] += 1
            total_shots[pid] += 1
            rally_shots.append({"frame": cf, "striker": pid, "zone": zone,
                                "court_xy": [round(cx, 2), round(cy, 2)]})

        # Rally-end positions per player (last ~1.5s) for error-zone analysis
        end_window_start = start_f + len(frames) - int(1.5 * fps)
        end_zone = {"1": None, "2": None}
        for pf in player_frames:
            if pf.frame_index < end_window_start:
                continue
            for pb in pf.players:
                cx, cy = model.to_court((pb.feet_x, pb.feet_y))
                if 0 <= cx <= COURT_WIDTH and 0 <= cy <= COURT_LENGTH:
                    end_zone[str(pb.player_id)] = _zone_name(cx, cy, COURT_WIDTH, COURT_LENGTH)

        rally_records.append({
            "rally_id": ri,
            "outcome": outcomes.get(str(ri)),
            "shot_count": len(rally_shots),
            "shots": rally_shots,
            "end_zone": end_zone,
        })

    cap.release()

    return _aggregate(shot_origin, total_shots, rally_records, processed_s)


def _nearest_pf(player_frames, target_fi):
    if not player_frames:
        return None
    return min(player_frames, key=lambda pf: abs(pf.frame_index - target_fi))


def _aggregate(shot_origin, total_shots, rally_records, processed_s) -> Dict:
    """Build shot-origin %, error zones (by outcome), and tactical insights."""
    # Shot-origin percentages per player
    origin_pct = {}
    for pid in ("1", "2"):
        tot = max(1, total_shots[pid])
        origin_pct[pid] = {z: round(shot_origin[pid][z] / tot * 100, 1) for z in ALL_ZONES}

    # Error zones: where the LOSER was positioned when they lost the point.
    # outcome p1 / stroke_p1 → P2 lost; p2 / stroke_p2 → P1 lost. let/warmup ignored.
    error_zones = {"1": {z: 0 for z in ALL_ZONES}, "2": {z: 0 for z in ALL_ZONES}}
    points_lost = {"1": 0, "2": 0}
    rally_len_by_winner = {"1": [], "2": []}

    for r in rally_records:
        o = r.get("outcome")
        if not o or o in ("let", "warmup"):
            continue
        if o in ("p1", "stroke_p1"):
            winner, loser = "1", "2"
        elif o in ("p2", "stroke_p2"):
            winner, loser = "2", "1"
        else:
            continue
        rally_len_by_winner[winner].append(r["shot_count"])
        lz = r["end_zone"].get(loser)
        if lz:
            error_zones[loser][lz] += 1
            points_lost[loser] += 1

    # Insights
    insights: List[str] = []
    for pid in ("1", "2"):
        # Favourite attacking zone
        if total_shots[pid] >= 8:
            fav = max(origin_pct[pid], key=lambda z: origin_pct[pid][z])
            if origin_pct[pid][fav] >= 25:
                insights.append(
                    f"Player {pid} plays most shots from {fav.replace('-', ' ')} "
                    f"({origin_pct[pid][fav]}%)."
                )
        # Error zone
        if points_lost[pid] >= 3:
            ez = max(error_zones[pid], key=lambda z: error_zones[pid][z])
            n = error_zones[pid][ez]
            if n >= 2:
                insights.append(
                    f"Player {pid} lost {n}/{points_lost[pid]} points positioned in "
                    f"{ez.replace('-', ' ')} — target that zone to pressure them."
                )

    # Rally length tendency
    for pid in ("1", "2"):
        lens = rally_len_by_winner[pid]
        if len(lens) >= 3:
            avg = float(np.mean(lens))
            style = "short, attacking" if avg <= 6 else ("long, grinding" if avg >= 12 else "balanced")
            insights.append(f"Player {pid} wins with {style} rallies (avg {avg:.0f} shots).")

    if not insights:
        insights.append("Tag rally outcomes (Rallies tab) to unlock error-zone and win-pattern analysis.")

    return {
        "calibrated": True,
        "active_play_s": round(processed_s, 1),
        "total_shots": total_shots,
        "shot_origin_pct": origin_pct,
        "error_zones": error_zones,
        "points_lost": points_lost,
        "avg_rally_len_by_winner": {
            pid: round(float(np.mean(v)), 1) if v else None
            for pid, v in rally_len_by_winner.items()
        },
        "rally_records": rally_records,
        "insights": insights,
    }
