"""
Layer 4 — 3D ball events through a rally.

Ties Layer 1 (Court3D) and Layer 2 (anchored arc reconstruction) together:

  1. detect CONTACTS in the 2D ball track (sharp direction/speed changes =
     racket hit, wall bounce, or floor bounce).
  2. classify each contact's SURFACE. A single contact pixel is ambiguous, so we
     pick the surface assignment by PHYSICAL CONSISTENCY: the correct surfaces are
     the ones for which the anchored 3D arcs on both sides actually reproject onto
     the observed 2D points (low consistency error). Racket contacts are anchored
     to the striker's position instead of a wall.
  3. chain the anchored segments into one 3D ball path + a typed event list, each
     with a confidence from its reprojection consistency.

This is the keystone that produces the ball half of the Rally Timeline. Its
accuracy is bounded by the 2D ball detector (the known weak link); every event
carries a confidence so downstream layers can weight it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .court import COURT_WIDTH, COURT_LENGTH
from .court3d import Court3D
from .ball3d import fit_anchored_segment, Ball3DArc

# Candidate non-racket surfaces a bounce can be on.
WALL_SURFACES = ["front_wall", "floor", "left_wall", "right_wall", "back_wall"]


@dataclass
class Contact:
    t: float
    u: float
    v: float
    kind: str = "unknown"        # racket | front_wall | floor | left_wall | right_wall | back_wall
    striker: Optional[int] = None
    confidence: float = 0.0


@dataclass
class RallyBall3D:
    contacts: List[Contact] = field(default_factory=list)
    arcs: List[dict] = field(default_factory=list)   # per-segment 3D + queries
    mean_consistency_px: float = 0.0


def detect_contacts_2d(track: List[Tuple[float, float, float]],
                       min_turn_deg: float = 55.0, min_win_px: float = 6.0,
                       win: int = 3, min_gap_s: float = 0.06) -> List[int]:
    """Indices in `track` where the ball direction reverses sharply (a contact).

    Direction is measured over a `win`-frame window on each side, not single
    steps — robust to high frame-rates and far-from-camera events where per-frame
    motion is only a few pixels but the reversal is real. The local maximum turn
    within a burst is kept (one contact per reversal).
    """
    if len(track) < 2 * win + 1:
        return []
    pts = np.array([(u, v) for _, u, v in track], float)
    ts = np.array([t for t, _, _ in track])
    k = 1
    sm = np.array([pts[max(0, i - k):i + k + 1].mean(0) for i in range(len(pts))])

    # Median sample spacing → a temporal-gap guard. A "reversal" that straddles a
    # gap (missing frames) is a jump between track fragments, NOT a real contact.
    dts = np.diff(ts)
    med_dt = float(np.median(dts)) if len(dts) else 1.0 / 30
    max_span = med_dt * (2 * win) * 2.5   # allow 2.5× the ideal contiguous span

    turns = np.zeros(len(sm))
    for i in range(win, len(sm) - win):
        if (ts[i + win] - ts[i - win]) > max_span:
            continue  # window spans a time gap → fragment boundary, skip
        v1 = sm[i] - sm[i - win]; v2 = sm[i + win] - sm[i]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < min_win_px or l2 < min_win_px:
            continue
        cos = np.clip(v1 @ v2 / (l1 * l2), -1, 1)
        turns[i] = np.degrees(np.arccos(cos))

    idx, last_t = [], -1e9
    for i in range(win, len(sm) - win):
        if turns[i] < min_turn_deg:
            continue
        # local max within the window (one contact per reversal burst)
        lo, hi = max(0, i - win), min(len(sm), i + win + 1)
        if turns[i] < turns[lo:hi].max() - 1e-6:
            continue
        if (ts[i] - last_t) >= min_gap_s:
            idx.append(i); last_t = ts[i]
    return idx


def _player_near(u, v, players_at_t, court3d, reach_m=1.6) -> Optional[int]:
    """If a player is within racket reach of the contact (in court metres on the
    floor projection), return their id — it's a racket contact."""
    if not players_at_t:
        return None
    gp = court3d.ground_to_court(u, v)
    if gp is None:
        return None
    best, bid = 1e9, None
    for pid, (px, py) in players_at_t.items():
        d = np.hypot(gp[0] - px, gp[1] - py)
        if d < best:
            best, bid = d, pid
    return bid if best <= reach_m else None


def _surface_consistency(seg_obs, court3d, start_surf, end_surf) -> float:
    """Reprojection consistency (px) of an anchored arc with the given surfaces;
    np.inf if not solvable."""
    res = fit_anchored_segment(seg_obs, court3d, start_surf, end_surf)
    if res is None:
        return np.inf
    return res[0].reproj_err_px


def reconstruct_rally_3d(
    track: List[Tuple[float, float, float]],   # (t, u, v) 2D ball track, time-ordered
    court3d: Court3D,
    players_by_t: Optional[List[Tuple[float, Dict[int, Tuple[float, float]]]]] = None,
    racket_anchor: str = "floor",   # racket contacts anchored via floor-projected reach point
) -> RallyBall3D:
    """Reconstruct the 3D ball path + typed events for one rally."""
    out = RallyBall3D()
    if len(track) < 6:
        return out
    track = sorted(track, key=lambda o: o[0])

    cidx = detect_contacts_2d(track)
    # Always treat the first and last sample as segment boundaries.
    bounds = [0] + cidx + [len(track) - 1]
    bounds = sorted(set(bounds))

    def players_at(t):
        if not players_by_t:
            return {}
        return min(players_by_t, key=lambda x: abs(x[0] - t))[1]

    # Build contacts with a first-pass kind (racket by player proximity)
    contacts: List[Contact] = []
    for bi in bounds:
        t, u, v = track[bi]
        pid = _player_near(u, v, players_at(t), court3d)
        contacts.append(Contact(t=t, u=u, v=v,
                                kind=("racket" if pid else "unknown"), striker=pid))

    # For each segment between consecutive contacts, choose the wall surfaces of
    # its endpoints (when 'unknown') by minimising anchored-arc consistency.
    seg_consistencies = []
    for s in range(len(bounds) - 1):
        i0, i1 = bounds[s], bounds[s + 1]
        seg_obs = track[i0:i1 + 1]
        if len(seg_obs) < 3:
            continue
        c0, c1 = contacts[s], contacts[s + 1]

        cand0 = ["racket"] if c0.kind == "racket" else WALL_SURFACES
        cand1 = ["racket"] if c1.kind == "racket" else WALL_SURFACES

        best = (np.inf, None, None)
        for s0 in cand0:
            for s1 in cand1:
                # racket endpoints are anchored to floor-reach point (approx); a
                # racket↔racket "segment" with no wall is implausible unless short
                a0 = racket_anchor if s0 == "racket" else s0
                a1 = racket_anchor if s1 == "racket" else s1
                err = _surface_consistency(seg_obs, court3d, a0, a1)
                if err < best[0]:
                    best = (err, s0, s1)

        err, s0, s1 = best
        if s0 is None:
            continue
        # Commit surfaces (don't overwrite a confident racket label)
        if contacts[s].kind == "unknown":
            contacts[s].kind = s0
        if contacts[s + 1].kind == "unknown":
            contacts[s + 1].kind = s1

        a0 = racket_anchor if contacts[s].kind == "racket" else contacts[s].kind
        a1 = racket_anchor if contacts[s + 1].kind == "racket" else contacts[s + 1].kind
        res = fit_anchored_segment(seg_obs, court3d, a0, a1)
        if res is None:
            continue
        arc, info = res
        conf = float(np.exp(-arc.reproj_err_px / 12.0))  # 0..1, 12px → ~0.37
        seg_consistencies.append(arc.reproj_err_px)
        out.arcs.append({
            "t_start": round(arc.t0, 3), "t_end": round(arc.t_end, 3),
            "start_surface": contacts[s].kind, "end_surface": contacts[s + 1].kind,
            "P_start": info["P_start"], "P_end": info["P_end"],
            "front_wall_hit": arc.front_wall_hit(),
            "floor_bounces": arc.floor_bounces(),
            "consistency_px": arc.reproj_err_px,
            "confidence": round(conf, 2),
        })
        contacts[s].confidence = max(contacts[s].confidence, conf)
        contacts[s + 1].confidence = max(contacts[s + 1].confidence, conf)

    out.contacts = contacts
    out.mean_consistency_px = round(float(np.mean(seg_consistencies)), 2) if seg_consistencies else 0.0
    return out
