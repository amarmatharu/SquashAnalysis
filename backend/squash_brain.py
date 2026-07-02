"""
Layer 6 — the Squash Brain (deterministic rules + scoring engine).

This is the REFEREE, not perception and not the coach. It takes the structured
events (who hit the last shot, what the ball did) and applies the official WSF
rules to decide the rally outcome, then maintains the score, serve, games and
match under modern PAR scoring.

Two parts, kept separate:
  1. determine_rally_outcome(...) — from the ball's end-event + last striker,
     decide the winner and WHY (down/out/winner). Interference (let/stroke) is a
     human/override input, since it needs interference perception we don't yet do.
  2. ScoreEngine — PAR: first to 11, win by 2; best of 5 games; serve alternates
     by box on each point the server wins and passes to the receiver on a hand-out.

Both the perception pipeline AND manual tagging feed the SAME engine — the engine
never looks at pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────── 1. Rally outcome ────────────────────────────────

# Ball end-events the perception layer can emit for the LAST shot of a rally.
#   "good"      — last shot hit the front wall above the tin, below the out-line
#   "down_tin"  — hit the tin (below it) → striker error
#   "not_up"    — bounced twice / hit floor before front wall → striker error
#   "out"       — above the out-line / on the lines / ceiling → striker error
#   "winner"    — last good shot was not returned (opponent let it double-bounce)
BALL_ERRORS = {"down_tin", "not_up", "out"}


@dataclass
class RallyOutcome:
    winner: Optional[int]          # 1 or 2, or None if undecided/let
    reason: str                    # down_tin | not_up | out | winner | stroke | let | manual
    striker: Optional[int]         # who hit the last shot
    confidence: float = 1.0
    source: str = "rules"          # rules | manual | override


def determine_rally_outcome(
    last_striker: Optional[int],
    last_shot_result: str,          # one of the ball end-events above
    interference: Optional[str] = None,   # None | "let" | "stroke"
    obstructed_player: Optional[int] = None,
    confidence: float = 1.0,
) -> RallyOutcome:
    """Apply the rules to one rally's terminal facts.

    The striker is the player who hit the LAST shot. If that shot was an error
    (tin/out/not-up) the OPPONENT wins; if it was a winner the STRIKER wins.
    Interference overrides: a stroke awards the point to the obstructed player; a
    let replays (no winner).
    """
    if interference == "let":
        return RallyOutcome(winner=None, reason="let", striker=last_striker,
                            confidence=confidence, source="rules")
    if interference == "stroke":
        return RallyOutcome(winner=obstructed_player, reason="stroke",
                            striker=last_striker, confidence=confidence, source="rules")

    if last_striker not in (1, 2):
        return RallyOutcome(winner=None, reason="unknown", striker=last_striker,
                            confidence=0.0)
    opponent = 2 if last_striker == 1 else 1

    if last_shot_result in BALL_ERRORS:
        return RallyOutcome(winner=opponent, reason=last_shot_result,
                            striker=last_striker, confidence=confidence)
    if last_shot_result == "winner":
        return RallyOutcome(winner=last_striker, reason="winner",
                            striker=last_striker, confidence=confidence)
    if last_shot_result == "good":
        # A "good" last shot with no recorded return is ambiguous → treat as a
        # winner but low confidence (perception may have missed a return).
        return RallyOutcome(winner=last_striker, reason="winner",
                            striker=last_striker, confidence=min(confidence, 0.4))
    return RallyOutcome(winner=None, reason="unknown", striker=last_striker, confidence=0.0)


def outcome_from_manual_tag(tag: str) -> RallyOutcome:
    """Map a manual outcome tag (the source of truth on phone footage) to a
    RallyOutcome, so manual tags drive the SAME scoring engine."""
    m = {
        "p1": RallyOutcome(1, "manual", None, 1.0, "manual"),
        "p2": RallyOutcome(2, "manual", None, 1.0, "manual"),
        "stroke_p1": RallyOutcome(1, "stroke", None, 1.0, "manual"),
        "stroke_p2": RallyOutcome(2, "stroke", None, 1.0, "manual"),
        "let": RallyOutcome(None, "let", None, 1.0, "manual"),
        "warmup": RallyOutcome(None, "warmup", None, 1.0, "manual"),
    }
    return m.get(tag, RallyOutcome(None, "unknown", None, 0.0, "manual"))


# ─────────────────────────── 2. Scoring engine ───────────────────────────────

@dataclass
class GameState:
    p1: int = 0
    p2: int = 0
    finished: bool = False
    winner: Optional[int] = None


@dataclass
class ScoreEngine:
    """PAR scoring: first to `target` (11), win by 2; best of `best_of` (5) games.

    Serve: the server keeps serving while winning points (alternating L/R box);
    on losing a rally the serve passes to the opponent (hand-out). The receiver of
    a new serve may choose a box; we model that as 'right' by default.
    """
    target: int = 11
    best_of: int = 5
    server: int = 1
    serve_box: str = "R"            # "L" | "R"
    games: List[GameState] = field(default_factory=lambda: [GameState()])
    games_won: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    match_over: bool = False
    match_winner: Optional[int] = None
    history: List[Dict] = field(default_factory=list)

    @property
    def current(self) -> GameState:
        return self.games[-1]

    def _game_target_met(self, g: GameState) -> Optional[int]:
        hi, lo = max(g.p1, g.p2), min(g.p1, g.p2)
        if hi >= self.target and hi - lo >= 2:
            return 1 if g.p1 > g.p2 else 2
        return None

    def award(self, winner: Optional[int], reason: str = "") -> Dict:
        """Apply one rally result. winner None = let/warmup (no score change)."""
        if self.match_over or winner not in (1, 2):
            self.history.append({"winner": winner, "reason": reason,
                                 "p1": self.current.p1, "p2": self.current.p2,
                                 "no_score": winner not in (1, 2)})
            return self.state()

        g = self.current
        # Serve / box update (PAR: scorer serves next; box alternates if server held)
        if winner == self.server:
            self.serve_box = "L" if self.serve_box == "R" else "R"
        else:
            self.server = winner
            self.serve_box = "R"

        if winner == 1:
            g.p1 += 1
        else:
            g.p2 += 1

        gw = self._game_target_met(g)
        if gw is not None:
            g.finished = True
            g.winner = gw
            self.games_won[gw] += 1
            need = self.best_of // 2 + 1
            if self.games_won[gw] >= need:
                self.match_over = True
                self.match_winner = gw
            else:
                # new game; loser of previous game serves first (simplification: winner serves)
                self.games.append(GameState())
                self.server = gw
                self.serve_box = "R"

        self.history.append({"winner": winner, "reason": reason,
                             "p1": g.p1, "p2": g.p2,
                             "game_won_by": gw, "server": self.server})
        return self.state()

    def state(self) -> Dict:
        return {
            "current_game": {"p1": self.current.p1, "p2": self.current.p2},
            "games_won": dict(self.games_won),
            "server": self.server, "serve_box": self.serve_box,
            "match_over": self.match_over, "match_winner": self.match_winner,
            "games": [{"p1": g.p1, "p2": g.p2, "winner": g.winner} for g in self.games],
        }


def score_match(rally_outcomes: List[RallyOutcome], first_server: int = 1,
                target: int = 11, best_of: int = 5) -> Dict:
    """Run a whole match's rally outcomes through the engine → final state +
    a per-rally running score (the scoreboard timeline)."""
    eng = ScoreEngine(target=target, best_of=best_of, server=first_server)
    running = []
    for o in rally_outcomes:
        st = eng.award(o.winner, o.reason)
        running.append({
            "winner": o.winner, "reason": o.reason,
            "p1": st["current_game"]["p1"], "p2": st["current_game"]["p2"],
            "games_won": st["games_won"], "server": st["server"],
            "match_over": st["match_over"],
        })
    return {"final": eng.state(), "running": running}
