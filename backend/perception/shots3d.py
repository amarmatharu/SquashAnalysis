"""
Layer 5 — shot classification (scaffold) from the 3D reconstruction.

Given the 3D arcs of one shot (the chain of segments from a racket contact until
the next contact / rally end), plus optional striker pose, classify:
  • type    : drive · cross-court drive · drop · kill · boast · lob · volley · serve
  • hand    : forehand / backhand (from pose)
  • quality : front-wall height over tin, side-wall tightness, depth
  • target  : where it landed (court zone)

These are PRINCIPLED GEOMETRIC heuristics, not a trained model — they use the real
3D arc (side-wall-before-front-wall ⇒ boast, low-front-wall+short ⇒ drop, etc.).
The interface is the contract the trained classifier (from flywheel labels) will
later implement. Every result carries a confidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .court import COURT_WIDTH, COURT_LENGTH
from .court3d import TIN_HEIGHT, FRONT_OUT_HEIGHT

FRONT_THIRD = COURT_LENGTH / 3
BACK_THIRD = 2 * COURT_LENGTH / 3


def _zone(x: float, y: float) -> str:
    col = "left" if x < COURT_WIDTH / 3 else ("right" if x > 2 * COURT_WIDTH / 3 else "center")
    row = "front" if y < FRONT_THIRD else ("back" if y > BACK_THIRD else "mid")
    return f"{row}-{col}"


def classify_shot(
    shot_arcs: List[Dict],          # consecutive segment dicts from events3d, this shot
    was_volley: bool = False,
    pose=None,                      # PlayerPose of the striker (optional)
    is_serve: bool = False,
) -> Dict:
    """Classify one shot from its 3D segment chain."""
    if not shot_arcs:
        return {"type": "unknown", "hand": None, "quality": {},
                "target_zone": None, "confidence": 0.0}

    surfaces = [a.get("end_surface") for a in shot_arcs]
    touched_side_before_front = False
    seen_front = False
    for s in surfaces:
        if s in ("left_wall", "right_wall") and not seen_front:
            touched_side_before_front = True
        if s == "front_wall":
            seen_front = True

    # First front-wall contact + its height
    front_hit = None
    for a in shot_arcs:
        if a.get("front_wall_hit"):
            front_hit = a["front_wall_hit"]
            break
    fw_height = front_hit["height_m"] if front_hit else None

    # Landing = first in-bounds floor bounce AFTER the front wall
    landing = None
    passed_front = False
    for a in shot_arcs:
        if a.get("end_surface") == "front_wall":
            passed_front = True
        if passed_front:
            for fb in a.get("floor_bounces", []):
                if fb.get("in_bounds"):
                    landing = fb
                    break
        if landing:
            break

    # Contact x (start of shot) for straight vs cross-court
    contact_x = shot_arcs[0].get("P_start", [None])[0]

    # ── decide type ─────────────────────────────────────────────────────────
    conf = 0.5
    if is_serve:
        shot_type = "serve"; conf = 0.9
    elif touched_side_before_front:
        shot_type = "boast"; conf = 0.7
    elif fw_height is not None and fw_height >= 3.2:
        shot_type = "lob"; conf = 0.6
    elif landing is not None and landing["y_m"] < FRONT_THIRD:
        # lands short. drop (soft) vs kill (hard) is a PACE distinction — needs
        # ball speed, which this scaffold may not have. Default to drop; only call
        # it a kill when an explicit high outgoing speed is supplied.
        out_speed = shot_arcs[0].get("out_speed_ms")
        if out_speed is not None and out_speed >= 18.0 and (fw_height or 9) <= TIN_HEIGHT + 0.4:
            shot_type = "kill"; conf = 0.55
        else:
            shot_type = "drop"; conf = 0.6
    else:
        # lands mid/back → a drive; straight or cross-court by x change
        shot_type = "drive"
        if landing is not None and contact_x is not None:
            crossed = (contact_x < COURT_WIDTH / 2) != (landing["x_m"] < COURT_WIDTH / 2)
            if crossed and abs(landing["x_m"] - contact_x) > COURT_WIDTH / 3:
                shot_type = "cross-court drive"
        conf = 0.55
    if was_volley and shot_type in ("drive", "drop", "boast", "lob"):
        shot_type = "volley " + shot_type
        conf *= 0.9

    # ── hand from pose ──────────────────────────────────────────────────────
    hand = None
    if pose is not None:
        side = pose.swing_side()
        if side:
            # right-handed assumption: racket on the right side of body ⇒ forehand
            hand = "forehand" if side == "right_side" else "backhand"

    # ── quality ─────────────────────────────────────────────────────────────
    quality = {}
    if fw_height is not None:
        quality["front_wall_height_m"] = fw_height
        quality["over_tin_m"] = round(fw_height - TIN_HEIGHT, 2)
    if landing is not None:
        quality["landing_zone"] = _zone(landing["x_m"], landing["y_m"])
        side_dist = min(landing["x_m"], COURT_WIDTH - landing["x_m"])
        quality["side_wall_tightness_m"] = round(side_dist, 2)
        quality["depth_m"] = round(landing["y_m"], 2)

    return {
        "type": shot_type,
        "hand": hand,
        "quality": quality,
        "target_zone": quality.get("landing_zone"),
        "confidence": round(conf, 2),
    }
