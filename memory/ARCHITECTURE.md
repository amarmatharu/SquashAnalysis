# SquashSense — System Architecture (the canonical blueprint)

**Status:** living document. Created 2026-06-27.
**Purpose:** the single architectural reference for what a complete squash
analyzer is, the models it needs, how they compose, how it scales/self-improves,
and the order to build it. Everything we build is measured against this.

> Related: [PRD.md](PRD.md) (product) · [ML_ROADMAP.md](ML_ROADMAP.md) (data→model path).
> This doc supersedes both as the *system* blueprint they hang off.

---

## 0. The principle and the one hard constraint

**Principle:** a squash analyzer is a **perception system** with a **rules engine**
and a **reasoning layer** on top. The LLM is the reasoning layer, *never* the
perception layer, and *never* the referee.

**The constraint that shapes everything — monocular 3D:**
Squash is a **3D game** (the ball interacts with the front wall, two side walls,
back wall, tin, and out-lines). A phone is a **single 2D camera with no depth.**
Almost every hard decision — tin or not, out or not, second bounce, tightness to a
side wall — is a **3D question asked of 2D data.** The architecture must recover 3D
from 2D using **court geometry + ball physics as priors**, and must attach a
**confidence** to every derived fact. A design that ignores this can never truly
analyze the game; it can only show surface stats.

---

## 1. Domain model — what a squash match *is*

```
Match ─► best of 5 Games ─► PAR scoring, first to 11, win by 2
Game  ─► sequence of Rallies
Rally ─► Serve (from a service box, onto front wall above the service line and
         below the out-line, rebounding into the opposite back quarter)
         └─► Exchange of Shots (alternating strikers)
              └─► End event:
                   DOWN  = ball hit the tin, bounced twice, or not-up
                   OUT   = above/on the out-line, or hit the ceiling
                   LET   = interference → replay, no point
                   STROKE= interference that cost a winning shot → point awarded
Shot  ─► one ball–racket contact, described by:
         type (drive · drop · boast · lob · volley · kill · nick · serve)
         × hand (forehand / backhand)
         × quality (height over tin, tightness to side wall, length/depth)
         × target (where it was aimed / landed)
Positions ─► both players' court location + the T (control), continuously
Interference ─► access to the ball, swing room, clearing effort (let/stroke logic)
```

Every italicised noun above is something the system must **perceive** or **decide**.

### The rulebook the engine must encode (WSF, modern PAR)
- **Scoring:** point-a-rally to 11, win by 2 (12–10, 13–11…); match = best of 5.
- **Serve:** alternate boxes after winning a point; ball must hit the front wall
  between the service line and the out-line and land in the opposite back quarter;
  one hand-out (loss of serve on losing the rally).
- **Good return:** before bouncing twice, the ball must reach the front wall above
  the tin and below the out-line (may use side/back walls en route).
- **Down / Out / Not-up:** the three ways to lose a rally on the ball.
- **Interference:** let (replay), stroke (point to the obstructed striker), or
  no-let — decided by whether the obstructed player could reach and make a good
  return, and whether the opponent made every effort to clear.

---

## 2. The understanding stack (the models, bottom-up by dependency)

Each layer consumes the ones below it. Build order is bottom-up.

| # | Layer | What it produces | Model / system | Kind |
|---|-------|------------------|----------------|------|
| 1 | **Court** | Full 3D court frame: 6 surfaces, tin, out-lines, boxes, T; camera pose | **Court line/keypoint detector → PnP camera calibration.** Standard dimensions are the prior, so a few lines give full 3D. Target: *automatic*, not manual clicks | ML + geometry |
| 2 | **Ball** | 2D track → **3D trajectory** with wall/floor bounce points | TrackNet (have) + **physics trajectory fitter** (ballistic arcs between surface contacts, constrained by the 3D court) | ML + physics |
| 3 | **Players** | Detection, tracking, **identity** (done), **pose/skeleton** | YOLO + colour re-ID (done) + **pose model** (swing, lunge, reach) | ML |
| 4 | **Events** | Physical atoms: ball–racket contact, ball–wall, ball–floor bounce, with 3D location + confidence | Derived from (2)+(3): velocity discontinuities + surface-plane intersections | Deterministic over perception |
| 5 | **Shots** | type × hand × quality × target, per contact | **Shot classifier** trained on labelled clips (ball arc + striker pose + contact location + outgoing vector) | ML (needs data) |
| 6 | **Squash Brain — rules engine** | rally boundaries, serve validity, **who won the point and WHY**, running score, let/stroke | **Deterministic finite-state machine** encoding WSF rules; consumes events+positions; emits outcome **with confidence** | Deterministic logic |
| 7 | **Strategy + Reasoning** | tendencies, strengths/weaknesses, "how to beat X", coaching narrative | Pattern mining over many rallies + **LLM** over the structured timeline | Analytics + LLM |

**Critical separation:** the "squash brain" is **two** distinct things —
(a) the **rules engine** (deterministic officiating: events → outcome) and
(b) the **strategic knowledge** (the coach). Keep them apart. The referee is logic;
the coach is analytics + LLM.

---

## 3. The single source of truth — the Rally Timeline

Everything perceived is written into one structured object per rally. Analytics,
the rules engine, and the LLM all read *only* this — never pixels.

```
RallyTimeline {
  rally_id, game_id, start_t, end_t,
  server: player_id, serve_box: L|R,
  shots: [ {
     t_contact, striker: player_id, hand, shot_type, quality{...},
     contact_court_xyz, outgoing_vector, confidence
  } ],
  ball_events: [ {t, kind: racket|frontwall|sidewall|backwall|floor, court_xyz, confidence} ],
  player_tracks: [ {player_id, t, court_xy, pose?} ],
  end: { reason: down_tin|down_2bounce|out|not_up|stroke|let,
         winner: player_id|null, confidence },
  score_after: {p1, p2}
}
```

Every field carries (or rolls up) a **confidence**. Low-confidence fields are what
the data flywheel (§5) surfaces for human confirmation.

---

## 4. Confidence & product tiers (how we live with monocular 3D)

Every derived fact is `(value, confidence)`. The product degrades gracefully:

- **Phone tier (today):** approximate 3D from one camera. Outcomes that hinge on
  fine 3D (tin/out) are **low-confidence → human confirms** (the tagging UI). We do
  NOT fake certainty. *This is the correct current behaviour, validated: a 2D tin
  line produced 314 false hits, so auto-tin is gated off.*
- **Pro tier (later):** fixed, well-placed (or multi-) camera → high-confidence
  automatic officiating. Same pipeline; more confidence, less human confirmation.

The architecture is identical across tiers; only confidence and the amount of human
confirmation change.

---

## 5. The data flywheel — how it scales & self-improves

```
video ─► perception(1–3) ─► events(4) ─► shots(5) ─► rules engine(6) ─► insight(7)
   ▲                                          │
   │                                          ▼
   └── retrain models ◄─ labelled data ◄─ human confirms LOW-CONFIDENCE items
                                         (the tagging UIs ARE the labelling tool)
```

- The system flags its **least-confident** calls (blurry ball, ambiguous tin,
  unclear shot) for a ~5-second human confirm → those become **training labels**.
- Models (ball, court, shot) retrain on a schedule against a **held-out eval set**;
  accuracy is tracked over time. This is "self-learns by watching squash."
- **Active learning:** spend human attention only where confidence is low (highest
  value labels), not on everything.
- **Drift guard (known risk):** human-confirmed labels must outrank model-generated
  pseudo-labels; cap self-generated labels per training run. (Confirmed earlier:
  self-training degraded the ball model vs manual-only.)
- **Versioning:** every model has a version + eval score; never ship a regression.

This flywheel — not any single model — is the moat. 1000s of matches → genuinely
good squash-specialised models.

---

## 6. Component data-flow (one diagram)

```
                ┌─────────────── Court (1) ── 3D calibration ───────────┐
                │                                                        ▼
 frames ─► Ball 2D (TrackNet) ─► Ball 3D fitter (2) ─► ball_events ─┐    │
        └─► Players (3) detect+ID+pose ─► player_tracks ────────────┤    │
                                                                    ▼    ▼
                                                       Events (4) = contacts/bounces in 3D
                                                                    │
                                                                    ▼
                                                       Shots (5) classify each contact
                                                                    │
                                                                    ▼
                                                  RALLY TIMELINE (source of truth)
                                                                    │
                                   ┌────────────────────────────────┤
                                   ▼                                 ▼
                       Rules engine (6)                     Strategy/analytics (7a)
                  (serve→exchange→end, score,                (tendencies, zones,
                   let/stroke) + confidence)                  win/loss patterns)
                                   └───────────────┬─────────────────┘
                                                   ▼
                                            LLM reasoning (7b)
                                       scouting report / how-to-beat-X
```

---

## 7. Current state vs this blueprint (honest)

| Layer | State | Gap to target |
|-------|-------|---------------|
| 1 Court | 🟢 **3D calibration (PnP) built + validated (cm-accurate on synthetic); self-check via reprojection error + projected-line overlay in the calibration UI** | auto line-detection still manual (needs data) |
| 2 Ball | 🟢 **3D reconstruction method built + validated.** KEY FINDING: free monocular ballistic fit is ILL-POSED for fast flat shots (1.5px reprojection yet 3.6m 3D error — depth ambiguity). SOLUTION: **surface-anchored** reconstruction (anchor each contact to its court plane → depth fixed by calibration) recovers to ~2cm and reads front-wall height correctly. Needs contact+surface detection wired in next | full pipeline integration; fast-shot reliability still better on pro tier (2nd cam / higher fps) |
| 3 Players | 🟢 detect + colour re-ID + naming + **pose (yolov8-pose, validated on real frame)** — wrist/swing-side/stance | wire pose into contact detection + biomechanics stats |
| 4 Events | 🟢 **3D event reconstruction built + validated on synthetic** (perception/events3d.py): windowed contact detection (robust to far/high-fps reversals), **surface classification by physical consistency** (picks the surface for which the anchored arcs actually reproject), chains 3D segments → typed events + tin/out + bounces, each with confidence. Validated: floor→front_wall→floor recovered exactly, tin/out correct. **Bounded by real ball+contact detection** | wire to real ball detector; robustness via flywheel/pro-tier |
| 5 Shots | 🟡 **geometric classifier scaffold built + validated** (shots3d.py): drive/cross/drop/kill/boast/lob/volley from the 3D arc (side-wall-before-front ⇒ boast, low+short ⇒ drop, etc.) + hand from pose + quality. Defines the interface the trained model will fill | replace heuristics with trained model once flywheel has ~500 shot labels |
| 6 Squash Brain | 🟢 **deterministic rules+scoring engine built + unit-tested** (squash_brain.py): rally outcome from last-striker + ball end-event (tin/out/not-up/winner/stroke/let), PAR-11 win-by-2, serve alternation/hand-out, best-of-5 match. Manual tags feed the SAME engine. All tests pass | wire engine into UI scoreboard; feed it perception outcomes once L4 runs on real video |
| 7 Strategy+LLM | 🟢 court-control + shot-patterns + Claude scout; **rules engine now drives the live scoreboard (games/serve/match)** | re-point scouting at the 3D shot timeline once L4 runs on real video |
| Flywheel | 🟢 **built**: `/training/flywheel` aggregates all human labels (moat); **`/training/eval-ball` held-out eval harness** (no silent regression — validated: median 4.66px / mean 92px on real labels, the good-median-bad-mean signal); `/training/eval-history`; FlywheelPanel UI in Training tab | scheduled auto-retrain trigger (cron) |
| Integration | 🟢 **`perception/timeline3d.py` runs L1→L6 on a real rally → canonical RallyTimeline** (POST /analysis/timeline3d). Confidence-first: on real Game4 it correctly self-reports LOW confidence (87px consistency) because rough calib + noisy ball → bad 3D. Works; quality gated by ball-detection + calibration | improves automatically as flywheel improves the ball model |

**Takeaway (updated 2026-06-27):** the foundations are now built and validated.
Layers 1,2,4,6 validated against ground truth (synthetic geometry / unit-tested
rules); Layers 3,5 built (pose real, shot scaffold); Layer 7 connects the rules
engine to the live scoreboard + opens the flywheel. **The remaining work is
INTEGRATION + DATA, not missing foundations:** run the L1→L4 3D pipeline on real
footage end-to-end (gated by ball-detection quality → flywheel), train the shot
classifier from labels, and add the scheduled-retrain/eval harness.

---

## 8. Build order (bottom-up; each unlocks the next)

1. **Court 3D + auto-detect (Layer 1)** — the coordinate system everything needs;
   removes per-match manual calibration and the spectator/extrapolation problems.
2. **Ball 3D trajectory + physics (Layer 2)** — unlocks tin/out/2nd-bounce with
   real confidence. Needs (1).
3. **Events on 3D (Layer 4)** + **pose (Layer 3+)** — clean contacts/bounces.
4. **Shot classifier (Layer 5)** — needs (2)+(3) and a label set (flywheel).
5. **Rules engine / Squash Brain (Layer 6)** — encode WSF rules over events; emits
   outcomes + score with confidence.
6. **Strategy + LLM (Layer 7)** — re-point the existing scouting at the now-solid
   timeline.
7. **Close the flywheel** — wire tagging → labels → scheduled retrain + eval.

Layers 3 (players/identity) and 7 (scouting) already exist as working slices and
get *re-pointed* at the firmer foundation rather than rebuilt.

---

## 9. Non-negotiable invariants

- Perception never lives in the LLM; the LLM never officiates.
- Every derived fact carries a confidence; the UI never shows fake certainty.
- One Rally Timeline is the only thing analytics/rules/LLM read.
- Human-confirmed labels outrank model-generated ones (drift guard).
- Each model is versioned and evaluated; no silent regressions.
- The same pipeline serves phone and pro tiers; only confidence differs.
