"""
Phase 6 — Scouting / reasoning layer.

Takes the structured perception outputs (rally outcomes, court control, shot
patterns, error zones) and turns them into a coached, plain-English scouting
report: each player's strengths, weaknesses, and a game plan to beat them.

Two modes:
  • Deterministic  — pure rules over the data. Always available, no API key.
  • LLM narrative  — feeds the same structured facts to an LLM for a polished,
    coach-style writeup. Used when an LLM key is configured.

The LLM is a REASONING layer over facts the perception spine produced; it is
never asked to perceive from pixels here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

ZONE_LABEL = {
    "front-left": "front-left", "front-center": "front (short)", "front-right": "front-right",
    "mid-left": "mid-left", "mid-center": "the T area", "mid-right": "mid-right",
    "back-left": "back-left corner", "back-center": "back (deep)", "back-right": "back-right corner",
}


def _score_from_outcomes(rallies: List[Dict], outcomes: Dict[str, str]) -> Dict:
    """Tally the score from manually-tagged rally outcomes (squash: first to 11)."""
    p1 = p2 = lets = 0
    for r in rallies:
        o = outcomes.get(str(r.get("rally_id")))
        if o in ("p1", "stroke_p1"):
            p1 += 1
        elif o in ("p2", "stroke_p2"):
            p2 += 1
        elif o == "let":
            lets += 1
    return {"p1": p1, "p2": p2, "lets": lets, "tagged": p1 + p2 + lets}


def build_findings(
    match: Dict,
    rally_result: Optional[Dict],
    outcomes: Dict[str, str],
    court_control: Optional[Dict],
    shot_patterns: Optional[Dict],
) -> Dict:
    """Distil all analyses into a compact 'facts' dict, per player + match-level."""
    p1_name = match.get("player1_name") or "Player 1"
    p2_name = match.get("player2_name") or "Player 2"

    rallies = (rally_result or {}).get("rallies", [])
    score = _score_from_outcomes(rallies, outcomes)

    facts: Dict = {
        "player_names": {"1": p1_name, "2": p2_name},
        "score": score,
        "num_rallies": len(rallies),
        "data_available": {
            "court_control": bool(court_control and court_control.get("calibrated")),
            "shot_patterns": bool(shot_patterns and shot_patterns.get("calibrated")),
            "outcomes_tagged": score["tagged"],
        },
        "players": {"1": {}, "2": {}},
    }

    # Court control facts
    if court_control and court_control.get("calibrated"):
        cc = court_control.get("players", {})
        for pid in ("1", "2"):
            p = cc.get(pid, {})
            facts["players"][pid].update({
                "t_control_pct": p.get("t_control_pct"),
                "avg_dist_from_t_m": p.get("avg_dist_from_t_m"),
                "distance_covered_m": p.get("total_distance_m"),
                "court_coverage_pct": p.get("court_coverage_pct"),
                "back_corner_pct": p.get("back_corner_pct"),
                "dominant_zone": p.get("dominant_zone"),
                "depth_pct": p.get("depth_pct"),
            })

    # Shot pattern facts
    if shot_patterns and shot_patterns.get("calibrated"):
        origin = shot_patterns.get("shot_origin_pct", {})
        errors = shot_patterns.get("error_zones", {})
        lost = shot_patterns.get("points_lost", {})
        winlen = shot_patterns.get("avg_rally_len_by_winner", {})
        for pid in ("1", "2"):
            o = origin.get(pid, {})
            fav = max(o, key=lambda z: o[z]) if o else None
            ez = errors.get(pid, {})
            worst = max(ez, key=lambda z: ez[z]) if ez and any(ez.values()) else None
            facts["players"][pid].update({
                "favourite_shot_zone": fav,
                "favourite_shot_zone_pct": o.get(fav) if fav else None,
                "total_shots": shot_patterns.get("total_shots", {}).get(pid),
                "error_zone": worst,
                "error_zone_count": ez.get(worst) if worst else 0,
                "points_lost": lost.get(pid, 0),
                "win_rally_len": winlen.get(pid),
            })

    return facts


def build_deterministic_report(facts: Dict) -> Dict:
    """Rule-based scouting report — works with no LLM key."""
    names = facts["player_names"]
    report: Dict = {"mode": "deterministic", "players": {}, "summary": "", "matchup": []}

    # Match-level summary
    sc = facts["score"]
    if sc["tagged"] >= 1:
        lets_note = f" ({sc['lets']} lets)" if sc.get("lets") else ""
        base = f"{names['1']} {sc['p1']} – {sc['p2']} {names['2']}{lets_note}"
        if sc["p1"] > sc["p2"]:
            report["summary"] = f"Score on tagged rallies: {base}. {names['1']} is ahead."
        elif sc["p2"] > sc["p1"]:
            report["summary"] = f"Score on tagged rallies: {base}. {names['2']} is ahead."
        else:
            report["summary"] = f"Score level: {base}."
    else:
        report["summary"] = ("No rally outcomes tagged yet — tag them in the Rallies tab to unlock "
                             "score, error zones and win-pattern reasoning.")

    for pid in ("1", "2"):
        p = facts["players"][pid]
        name = names[pid]
        strengths: List[str] = []
        weaknesses: List[str] = []
        gameplan: List[str] = []

        tc = p.get("t_control_pct")
        if tc is not None:
            if tc >= 55:
                strengths.append(f"Strong T-control ({tc}%) — dictates the rally from the centre.")
            elif tc <= 45:
                weaknesses.append(f"Weak T-control ({tc}%) — often out of position.")
                gameplan.append("Take the T early and hold it; make them retrieve from the corners.")

        bc = p.get("back_corner_pct")
        if bc is not None and bc >= 25:
            weaknesses.append(f"Spends {bc}% of play pinned in the back corners.")
            gameplan.append("Pressure them deep with length, then drop short to stretch them.")

        ez = p.get("error_zone")
        if ez and p.get("error_zone_count", 0) >= 2:
            weaknesses.append(
                f"Loses points mostly from {ZONE_LABEL.get(ez, ez)} "
                f"({p['error_zone_count']} of {p.get('points_lost')} points lost there)."
            )
            gameplan.append(f"Work the ball into {ZONE_LABEL.get(ez, ez)} — that's where they break down.")

        fav = p.get("favourite_shot_zone")
        if fav and p.get("favourite_shot_zone_pct", 0) >= 30:
            strengths.append(
                f"Plays {p['favourite_shot_zone_pct']}% of shots from {ZONE_LABEL.get(fav, fav)} — "
                f"comfortable, predictable origin."
            )
            gameplan.append(f"Deny them {ZONE_LABEL.get(fav, fav)}; vary pace to pull them off it.")

        wl = p.get("win_rally_len")
        if wl is not None:
            if wl <= 6:
                strengths.append(f"Wins with short, attacking rallies (avg {wl} shots) — dangerous early.")
                gameplan.append("Slow the rally down; extend exchanges to deny quick winners.")
            elif wl >= 12:
                strengths.append(f"Wins long, grinding rallies (avg {wl} shots) — high engine.")
                gameplan.append("Shorten rallies; take it in early to avoid a war of attrition.")

        dist = p.get("distance_covered_m")
        if dist is not None:
            other = facts["players"]["2" if pid == "1" else "1"].get("distance_covered_m")
            if other and dist > other * 1.15:
                weaknesses.append(f"Covering more ground ({dist}m vs {other}m) — being run around.")
                gameplan.append("Keep moving them corner to corner; fitness may tell late in the game.")

        if not strengths:
            strengths.append("Not enough tagged data for a confident strength read.")
        if not weaknesses:
            weaknesses.append("No clear weakness surfaced yet — tag more rallies for sharper reads.")

        report["players"][pid] = {
            "name": name,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "gameplan_against": gameplan or ["Tag rally outcomes to generate a targeted game plan."],
        }

    return report


def build_llm_prompt(facts: Dict) -> str:
    """Build the prompt that asks an LLM to write the coached scouting narrative."""
    import json
    names = facts["player_names"]
    return f"""You are an elite squash coach writing a scouting report from match-tracking data.

The data below was produced by a computer-vision system that tracked the players
and ball, mapped positions to real court coordinates, and recorded rally outcomes
a human tagged. Court zones use: front/mid/back (depth) × left/center/right.
"mid-center" = the T area. Squash scoring is first to 11, win by 2.

PLAYERS: Player 1 = {names['1']}, Player 2 = {names['2']}

STRUCTURED DATA (JSON):
{json.dumps(facts, indent=2)}

Write a concise, practical scouting report with these sections in markdown:

## Match Summary
Two to three sentences: who is winning and the overall pattern of the match.

## {names['1']}
- **Strengths:** bullet points grounded in the data (cite the numbers).
- **Weaknesses:** bullet points grounded in the data.

## {names['2']}
- **Strengths:** ...
- **Weaknesses:** ...

## Game Plan to Beat {names['1']}
3-5 specific, actionable tactics a coach would give, each tied to a weakness above.

## Game Plan to Beat {names['2']}
3-5 specific, actionable tactics.

Rules:
- Only use the data provided. If a metric is null/missing, don't invent it; note
  that more tagged rallies are needed.
- Be specific and tactical (squash terms: length, drop, boast, volley, nick,
  holding the T, dying length). Avoid generic filler.
- Keep the whole report under ~400 words."""
