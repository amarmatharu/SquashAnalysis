"""
The integration keystone — build the canonical Rally Timeline by running the
whole stack (Layers 1-6) on one real rally.

  ball 2D (TrackNet) ─┐
  players (id+court) ─┼─► 3D events (events3d) ─► shots (shots3d) ─► outcome (squash_brain)
  pose (hand)        ─┘                                    └─► RallyTimeline (source of truth)

Output is the single structured object the architecture defines; analytics, the
rules engine and the LLM all read it, never pixels. Every field carries a
confidence so downstream layers can weight perception that is bounded by the 2D
ball detector (the known weak link — improves via the flywheel).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .court3d import Court3D
from .events3d import reconstruct_rally_3d
from .shots3d import classify_shot


def _hampel_clean(track, win: int = 3, k: float = 3.5):
    """Reject isolated spatial outliers (false-positive spikes) while keeping real
    turning points. A turning point sits between its neighbours; a spike jumps away
    from them and is flagged by a windowed median/MAD test."""
    if len(track) < 2 * win + 1:
        return track
    arr = np.array([(u, v) for _, u, v in track], float)
    keep = []
    for i in range(len(track)):
        lo, hi = max(0, i - win), min(len(arr), i + win + 1)
        med = np.median(arr[lo:hi], axis=0)
        mad = np.median(np.abs(arr[lo:hi] - med), axis=0) + 1e-6
        dev = np.abs(arr[i] - med) / (1.4826 * mad)
        if np.all(dev < k):
            keep.append(track[i])
    return keep


def _ball_track_2d(frames, start_f, fps, ball_det,
                   min_traj_len: int = 6) -> List[Tuple[float, float, float]]:
    """Single coherent 2D ball track [(t,u,v)] over the rally.

    A ball is ONE object, so at any moment there is one ball position. We rank the
    tracker's candidate trajectories by quality and greedily select the best ones
    that DON'T overlap in time — building one clean path instead of the union of
    all competing arcs (which is what triggered the false contacts). Short
    fragments + isolated spikes are dropped.
    """
    from .ball import BallTracker
    tracker = BallTracker()
    per = ball_det.detect_window(frames, start_f, fps)
    tracks = tracker.build_tracks(per)

    scored = []
    for t in tracks:
        q = tracker._track_quality(t)
        if q > 0 and len(t.points) >= min_traj_len:
            xs = [p.x for p in t.points]; ys = [p.y for p in t.points]
            if (max(xs) - min(xs)) + (max(ys) - min(ys)) >= 20:  # real displacement
                scored.append((q, t))
    scored.sort(key=lambda x: -x[0])

    used = set()                      # frame indices already claimed by a better track
    track = []
    for _, t in scored:
        span = [p.frame_index for p in t.points]
        overlap = sum(1 for fi in span if fi in used)
        if overlap > 0.3 * len(span):  # competes in time with a better track → skip
            continue
        for p in t.points:
            used.add(p.frame_index)
            track.append((p.frame_index / fps, p.x, p.y))
    track.sort()
    return _hampel_clean(track)


def _court3d_on_court(court3d, margin=0.4):
    """Predicate(feet_x, feet_y) -> on the court, via 3D floor back-projection.
    Filters spectators behind/beside the court."""
    from .court import COURT_WIDTH, COURT_LENGTH

    def pred(fx, fy):
        gp = court3d.ground_to_court(fx, fy)
        if gp is None:
            return False
        x, y = gp
        return -margin <= x <= COURT_WIDTH + margin and -margin <= y <= COURT_LENGTH + margin
    return pred


def _players_by_t(frames, start_f, fps, player_det, ref_sigs, on_court, court3d):
    """List of (t, {pid: (court_x, court_y)}) from identity-locked player detection."""
    pfs = player_det.detect_frames(frames, start_f, fps, ref_sigs, on_court)
    out = []
    for pf in pfs:
        d = {}
        for pb in pf.players:
            gp = court3d.ground_to_court(pb.feet_x, pb.feet_y)
            if gp:
                d[pb.player_id] = gp
        out.append((pf.frame_index / fps, d))
    return out


def build_rally_timeline_3d(
    video_path: str,
    rally_id: int,
    start_s: float,
    end_s: float,
    court3d: Court3D,
    setup: str = "phone",
    ref_sigs=None,
    use_pose: bool = True,
) -> Dict:
    """Run the full stack on one rally → a RallyTimeline dict."""
    import cv2
    from .ball import get_ball_detector
    from .players import get_player_detector
    from .annotation import PROC_MAX_WIDTH

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(start_s * fps)
    n = int((end_s - start_s) * fps)
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
    cap.release()
    if len(frames) < 6:
        return {"rally_id": rally_id, "error": "too few frames"}

    on_court = _court3d_on_court(court3d)   # filter spectators via 3D floor bounds

    ball_det = get_ball_detector(setup)
    player_det = get_player_detector()

    track = _ball_track_2d(frames, start_f, fps, ball_det)
    players_t = _players_by_t(frames, start_f, fps, player_det, ref_sigs, on_court, court3d)

    rb = reconstruct_rally_3d(track, court3d, players_t)

    # Pose at racket-contact frames (for hand)
    poses_at = {}
    if use_pose:
        from .pose import get_pose_detector
        pdet = get_pose_detector()
        for c in rb.contacts:
            if c.kind == "racket":
                fi = int(round(c.t * fps)) - start_f
                if 0 <= fi < len(frames):
                    ps = pdet.detect(frames[fi], int(c.t * fps))
                    # nearest pose to the contact pixel
                    if ps:
                        best = min(ps, key=lambda p: (np.hypot(
                            (p.box[0] + p.box[2]) / 2 - c.u, (p.box[1] + p.box[3]) / 2 - c.v)))
                        poses_at[round(c.t, 3)] = best

    # Assemble shots: each racket contact begins a shot; its arc chain runs to the
    # next contact. Classify + determine last-shot result.
    racket_idx = [i for i, c in enumerate(rb.contacts) if c.kind == "racket"]
    shots = []
    last_result = "unknown"
    last_striker = None
    for k, ci in enumerate(racket_idx):
        c = rb.contacts[ci]
        # arc segments belonging to this shot: from this contact to the next contact
        seg_arcs = [a for a in rb.arcs if a["t_start"] >= c.t - 1e-3 and
                    (k + 1 >= len(racket_idx) or a["t_end"] <= rb.contacts[racket_idx[k + 1]].t + 1e-3)]
        was_volley = False  # (volley detection needs the incoming segment's bounces; left for refinement)
        pose = poses_at.get(round(c.t, 3))
        info = classify_shot(seg_arcs, was_volley=was_volley, pose=pose)
        # determine what the ball did after this shot
        result = "good"
        for a in seg_arcs:
            fw = a.get("front_wall_hit")
            if fw:
                result = {"down_tin": "down_tin", "out_top": "out", "good": "good"}.get(
                    fw["classification"], "good")
                break
        shots.append({
            "shot_id": k + 1, "t_contact": round(c.t, 3), "striker": c.striker,
            "type": info["type"], "hand": info["hand"], "quality": info["quality"],
            "result": result, "confidence": round(min(info["confidence"], c.confidence or 0.3), 2),
        })
        last_result, last_striker = result, c.striker

    # Rally outcome via the Squash Brain (perception path; may be low-confidence)
    from squash_brain import determine_rally_outcome
    outcome = determine_rally_outcome(last_striker, last_result if last_result in
                                      ("down_tin", "out", "not_up", "winner", "good") else "good")

    ball_events = []
    for c in rb.contacts:
        ball_events.append({"t": round(c.t, 3), "kind": c.kind,
                            "striker": c.striker, "confidence": round(c.confidence, 2)})

    return {
        "rally_id": rally_id,
        "start_t": round(start_s, 2), "end_t": round(end_s, 2),
        "n_ball_samples": len(track),
        "n_contacts": len(rb.contacts),
        "mean_consistency_px": rb.mean_consistency_px,
        "shots": shots,
        "ball_events": ball_events,
        "outcome": {"winner": outcome.winner, "reason": outcome.reason,
                    "striker": outcome.striker, "confidence": outcome.confidence,
                    "source": "perception"},
    }
