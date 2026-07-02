"""
Shot-event detection and rally timeline assembly (M2).

A *shot* is a ball–racket contact. With a ball trajectory (in court metres) and
the players' court positions, a contact shows up as a **direction reversal** in
the ball's path co-located with a player: the ball decelerates into the racket,
changes heading sharply, and accelerates away. We detect those reversal points,
attribute each to the nearer player (the striker), then group shots separated by
gaps into rallies — producing the structured timeline the analytics and reasoning
layers consume.

Honesty note: event quality is bounded by ball-track quality. On the classical
detector the ball track is noisy, so events are rough; they sharpen automatically
once a trained TrackNet supplies a clean trajectory (same code path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BallSample:
    frame_index: int
    t: float
    court_x: float
    court_y: float


@dataclass
class PlayerSample:
    t: float
    court_x: float
    court_y: float


@dataclass
class ShotEvent:
    frame_index: int
    t: float
    striker: str                 # player1 | player2 | unknown
    ball_court_xy: Tuple[float, float]
    turn_angle_deg: float        # heading change at contact (sharper = more certain)
    incoming_speed_ms: float
    outgoing_speed_ms: float
    confidence: float


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def detect_shot_events(
    ball: List[BallSample],
    players: Dict[str, List[PlayerSample]],
    min_turn_deg: float = 55.0,
    min_separation_s: float = 0.10,
) -> List[ShotEvent]:
    """Find ball–racket contacts as co-located direction reversals.

    ``ball`` must be time-ordered. ``players`` maps player label -> samples. A
    contact is a local heading change above ``min_turn_deg``; consecutive
    detections within ``min_separation_s`` are de-duplicated (keep the sharpest).
    The default separation only collapses adjacent-frame duplicates (~0.03s at
    30fps); genuine shots in a rally are much further apart and survive.
    """
    if len(ball) < 3:
        return []

    events: List[ShotEvent] = []
    for i in range(1, len(ball) - 1):
        a, b, c = ball[i - 1], ball[i], ball[i + 1]
        v_in = np.array([b.court_x - a.court_x, b.court_y - a.court_y])
        v_out = np.array([c.court_x - b.court_x, c.court_y - b.court_y])
        turn = _angle_between(v_in, v_out)
        if turn < min_turn_deg:
            continue

        dt_in = max(1e-3, b.t - a.t)
        dt_out = max(1e-3, c.t - b.t)
        spd_in = float(np.linalg.norm(v_in) / dt_in)
        spd_out = float(np.linalg.norm(v_out) / dt_out)

        # Striker = nearest player at this time (by court distance).
        striker, best_d = "unknown", 1e9
        for label, samples in players.items():
            ps = _nearest_player_sample(samples, b.t)
            if ps is None:
                continue
            d = np.hypot(ps.court_x - b.court_x, ps.court_y - b.court_y)
            if d < best_d:
                best_d, striker = d, label

        # Confidence: sharper turn + striker actually near the ball.
        turn_conf = min(1.0, (turn - min_turn_deg) / (180.0 - min_turn_deg) + 0.3)
        prox_conf = float(np.exp(-best_d / 2.0)) if best_d < 1e8 else 0.0
        conf = round(0.6 * turn_conf + 0.4 * prox_conf, 3)

        events.append(
            ShotEvent(
                frame_index=b.frame_index, t=round(b.t, 3), striker=striker,
                ball_court_xy=(round(b.court_x, 2), round(b.court_y, 2)),
                turn_angle_deg=round(turn, 1),
                incoming_speed_ms=round(spd_in, 2),
                outgoing_speed_ms=round(spd_out, 2),
                confidence=conf,
            )
        )

    # De-duplicate bursts: keep the sharpest turn within the separation window.
    deduped: List[ShotEvent] = []
    for ev in events:
        if deduped and (ev.t - deduped[-1].t) < min_separation_s:
            if ev.turn_angle_deg > deduped[-1].turn_angle_deg:
                deduped[-1] = ev
        else:
            deduped.append(ev)
    return deduped


def _nearest_player_sample(samples: List[PlayerSample], t: float) -> Optional[PlayerSample]:
    if not samples:
        return None
    return min(samples, key=lambda s: abs(s.t - t))


def build_timeline(
    events: List[ShotEvent],
    ball: List[BallSample],
    rally_gap_s: float = 1.5,
) -> Dict:
    """Group shots into rallies (split on time gaps) and assemble the M2 schema."""
    rallies: List[Dict] = []
    cur: List[ShotEvent] = []

    def flush(rally_id: int):
        if not cur:
            return None
        shots = [
            {
                "shot_id": j + 1,
                "frame_index": e.frame_index,
                "t_contact": e.t,
                "striker": e.striker,
                "ball_court_xy": list(e.ball_court_xy),
                "turn_angle_deg": e.turn_angle_deg,
                "incoming_speed_ms": e.incoming_speed_ms,
                "outgoing_speed_ms": e.outgoing_speed_ms,
                "confidence": e.confidence,
                # shot_type intentionally left null: needs the trained classifier (M4).
                "shot_type": None,
            }
            for j, e in enumerate(cur)
        ]
        return {
            "rally_id": rally_id,
            "start_t": cur[0].t,
            "end_t": cur[-1].t,
            "shot_count": len(cur),
            "shots": shots,
        }

    for e in events:
        if cur and (e.t - cur[-1].t) > rally_gap_s:
            r = flush(len(rallies) + 1)
            if r:
                rallies.append(r)
            cur = []
        cur.append(e)
    r = flush(len(rallies) + 1)
    if r:
        rallies.append(r)

    return {
        "ball_track": [
            {"frame_index": s.frame_index, "t": round(s.t, 3),
             "court_x_m": round(s.court_x, 2), "court_y_m": round(s.court_y, 2)}
            for s in ball
        ],
        "rallies": rallies,
        "total_shots": sum(r["shot_count"] for r in rallies),
        "total_rallies": len(rallies),
    }
