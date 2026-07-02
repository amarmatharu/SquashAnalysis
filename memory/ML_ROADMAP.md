# SquashSense AI — ML Roadmap to a Squash-Specialized Model

**Status:** living document. Created 2026-06-21.
**Author context:** written after building the perception spine (player tracking +
court homography + movement metrics) and confirming the original frame-sampling +
generic-LLM approach cannot deliver fine-grained squash analysis.

---

## 0. The core principle

> A squash analyzer is a **perception system** with a **reasoning layer** on top.
> The LLM is the reasoning layer, never the perception layer.

```
video + audio
  └─► PERCEPTION (squash-specialized, the "trained on squash" part)
        player tracking · court geometry · pose · BALL TRACKING · shot events
  └─► STRUCTURED RALLY TIMELINE  (the single source of truth)
  └─► ANALYTICS  (statistics over the timeline → strengths/weaknesses)
  └─► REASONING (generic LLM over numbers → scouting report, "how to beat X")
```

**Why "just train a squash model" is not a step you can take yet:** a trained model
is the *output* of a labeled dataset + training pipeline. We currently have **zero
labeled squash data** (`training_corrections` is empty). There is no off-the-shelf
"squash shot model" to download. The entire first half of this roadmap exists to
*manufacture the dataset* that a squash-specialized model requires. There is no
shortcut around the data.

**The irony to accept up front:** the fastest path to *our* trained squash model
runs *through* a generic model used temporarily as a **labeling tool** (first-pass
auto-labels), plus commentary audio and heuristic rules — not as the shipped product.

---

## 1. What already exists (M0 — done)

- `backend/perception/court.py` — homography mapping image px → real court metres on
  the standard 6.4×9.75 m singles court; T, service boxes, depth zones, distance-to-T.
- `backend/perception/tracking.py` — YOLO11 + ByteTrack player detection/tracking,
  foot-point extraction, court-polygon filtering (drops refs/crowd).
- `backend/perception/pipeline.py` — measured per-player movement: distance, avg/peak
  speed, court coverage, **T-dominance**, depth split. **No LLM used.**
- Court-calibration UI (`PlayerSelectPage.jsx`) + `/set-court` endpoint (normalized
  corners → native px at analysis time).
- `np.random` fabricated movement data **removed**; real metrics wired into `server.py`.

**Known gaps in M0:** ByteTrack id fragmentation under occlusion (needs re-id); manual
4-corner calibration (no auto court detection); no ball, no shot events.

---

## 2. The structured rally timeline (the data schema)

Everything depends on this. It is the contract between perception and analytics, the
format we label, train on, and reason over. Target schema (per match):

```jsonc
{
  "match_id": "uuid",
  "court_calibration": { "front_left":[fx,fy], ... },   // normalized
  "fps": 30, "duration_s": 363.9,
  "players": { "player1": {...identity}, "player2": {...} },
  "ball_track": [ { "t": 12.30, "x_m": 3.1, "y_m": 8.2, "conf": 0.8 } ],  // court metres
  "rallies": [
    {
      "rally_id": 1, "start_t": 10.0, "end_t": 28.4, "winner": "player1",
      "shots": [
        {
          "shot_id": 1, "t_contact": 11.2, "striker": "player2",
          "shot_type": "drive",            // label (drive/drop/boast/volley/lob/kill/cross/serve)
          "swing": "backhand",
          "striker_pos_m": [0.8, 8.9],
          "ball_in_traj": [...], "ball_out_traj": [...],   // segments around contact
          "ball_speed_ms": 18.4, "target_zone": "back-left",
          "quality": 0.72,                 // tightness to wall / length
          "outcome": "in_play",            // in_play | winner | error | let/stroke
          "label_source": "human|asr|heuristic|vlm|model",  // provenance for training
          "label_conf": 0.9
        }
      ]
    }
  ]
}
```

`label_source`/`label_conf` are first-class: weak-supervision needs provenance so the
training label-model can weight sources and resolve conflicts.

---

## 3. Perception models (the squash-specialized core)

| Component | Recommended | Notes / trade-offs |
|---|---|---|
| Player detection+track | **YOLO11 + ByteTrack** (done) | Off-the-shelf is strong for people. |
| Player re-id (occlusion) | **OSNet / torchreid** appearance embeddings | Fixes the id-fragmentation seen behind glass. Pin players by jersey colour + embedding. |
| Court calibration | manual 4-corner (done) → **line-segment/seg model** auto-detect | Auto-detection lets us scale to many videos without hand-marking. |
| Pose | MediaPipe (current) → **RTMPose / ViTPose** | Better keypoints = better swing + contact detection; needed for forehand/backhand and racket-up frames. |
| **Ball tracking** | **TrackNetV2/V3** (heatmap, built for racket sports) | THE hard, decisive module. Squash ball is small, black, fast, motion-blurred, with glass reflections. Pretrained tennis/badminton weights transfer partially; **will need squash fine-tuning** on a few thousand annotated ball positions. |
| Shot-event detection | geometry on ball+pose | Contact = local min of ball↔racket-hand distance + ball direction reversal. Defines shot boundaries; no learning needed initially. |
| Shot-type classifier | **temporal model** (TCN / 1D-CNN / small Transformer) over features | Input = ball in/out trajectory + striker pose + court position window. This is the model we *train on squash*. |

**Order of attack:** ball tracking → shot-event detection → shot-type classifier.
Without ball tracking, shot classification has no real signal and we are back to guessing.

---

## 4. Labeling strategy (how we manufacture the dataset)

We will **not** hand-label from scratch. Combine cheap weak labels, then human-correct
the uncertain ones (active learning). Sources, by cost/quality:

1. **Heuristic / programmatic labels (squash rules over geometry)** — cheapest, highly
   squash-specific. Examples once ball+court exist:
   - ball hits front wall low and dies in front third → **drop**
   - long, parallel to side wall, lands back third → **drive**
   - strikes side wall first, then front → **boast**
   - struck above shoulder before bounce → **volley**
   - high, slow, to back corner → **lob**
   These become Snorkel-style labeling functions with confidences.
2. **Commentary ASR (weak supervision)** — **Whisper large-v3** transcribes commentary;
   keyword spotting ("drop", "boast", "nick", "tin", "stroke") aligned to the nearest
   shot event timestamps. Free labels from broadcast footage at scale.
3. **Generic VLM first-pass** — GPT-4o / Gemini on **short clips** (not single frames)
   for shots heuristics can't resolve. Temporary labeling tool only.
4. **Human correction** — existing correction UI promotes weak labels to ground truth;
   active learning routes the **lowest-confidence / most-disagreed** shots to humans first.

A **label model** (Snorkel `LabelModel` or simple weighted vote using `label_conf`)
fuses sources into a single training label + confidence per shot.

---

## 5. Training pipeline

1. **Features** per shot event: ball in/out trajectory (resampled, court metres),
   striker pose keypoints (normalized), striker court position, ball speed, wall-contact
   sequence. All derived from the structured timeline — reproducible, inspectable.
2. **Model**: start small — Temporal CNN or gradient-boosted trees on engineered
   trajectory features (robust with little data) → graduate to a 1D-CNN/Transformer as
   data grows. Squash has ~7–9 shot classes; this is a tractable classification problem
   *once features are good*.
3. **Eval harness**: held-out matches (never mix rallies across splits), per-class
   precision/recall, confusion matrix, and a "human-agreement" metric on a gold set.
   Track accuracy vs. dataset size to know when more labels stop helping.
4. **Versioning**: dataset snapshots + model versions; every shot carries
   `label_source` so we can ablate (heuristic-only vs +ASR vs +human).

**Data targets (rough):** first usable classifier ≈ **2–4k labeled shots** (≈ 20–40
fully processed matches). Ball-tracking fine-tune ≈ **3–5k annotated ball frames**.

---

## 6. Analytics layer (strengths & weaknesses)

Pure statistics over the structured timeline — no ML, fully explainable:
- conditional win-rates: rally length buckets, front vs back, FH vs BH side
- shot quality distributions (length/tightness), error rates under pressure
- T-control (already have distance-to-T), recovery time after each shot
- pattern mining: frequent shot n-grams that precede won/lost points
Output is a per-player profile object that the reasoning layer consumes.

---

## 7. Reasoning layer ("how to beat player X")

Here a **generic LLM is correct and strong**, because it reasons over *numbers*, not
pixels. Input = two players' profile objects + head-to-head structured stats. Output =
scouting report + game plan ("attack the backhand deep; their error rate triples in
rallies >12 shots; they over-commit to the front off your boast"). Grounded in the
analytics object, with every claim traceable to a stat → low hallucination risk.

---

## 8. Self-learning / continuous improvement

The flywheel that makes it "learn over time":
1. New match → perception → weak labels → model predicts.
2. Low-confidence/disagreed shots → human correction UI.
3. Corrections appended to dataset → periodic retrain → model improves.
4. Commentary on every new broadcast adds free weak labels.
This is **active learning**, not a model that magically watches TV. It is real,
incremental, and measurable on the eval harness.

---

## 9. Compute & cost

- **Inference**: player+ball tracking are GPU-friendly; CPU works for short clips but is
  slow (the M0 demo ran on CPU). A single cloud GPU (T4/A10) makes full-match processing
  practical.
- **Training**: shot classifier is light (single GPU, hours). TrackNet fine-tune is the
  heavier job (single GPU, days, plus the ball-annotation effort).
- **Whisper ASR**: large-v3 on GPU is cheap relative to value.
- **Generic VLM labeling**: metered API cost, used sparingly and only to bootstrap.

---

## 10. Phased milestones

- **M0 — Perception spine** ✅ player tracking + court homography + movement metrics.
- **M1 — Ball tracking** TrackNet integration + ball-strike event detection.
- **M2 — Structured timeline** schema (§2) persisted per match; replaces ad-hoc shot list.
- **M3 — Labeling pipeline** heuristic LFs + Whisper ASR + correction-UI wiring + label model.
- **M4 — First shot classifier** trained on weak+human labels; eval harness; replaces the
  generic first-pass labeler.
- **M5 — Analytics engine** conditional strength/weakness profiles.
- **M6 — Reasoning layer** LLM scouting report + "how to beat X" over structured stats.
- **M7 — Self-learning loop** active learning + scheduled retrain; accuracy tracked over time.

Each milestone is independently demoable. M0 already produces real, LLM-free squash
metrics today (once a court is marked).

---

## 11. Risks & honest unknowns

- **Squash ball tracking is genuinely hard** — small, black, motion-blurred, glass
  reflections, occlusion by bodies. This is the highest-risk module; budget for squash
  fine-tuning, don't assume pretrained transfers cleanly.
- **Broadcast variety** — camera angles/quality differ; auto court detection and re-id
  must generalize.
- **Label noise** — weak supervision is noisy; the eval gold set must be human-verified.
- **Data cost** — manufacturing the dataset is the real project cost, not the modelling.
- **`gpt-5.2`** referenced in current code may not be a real model on a given account; the
  reasoning layer should target a verified vision/text model.
- **Cold start** — until M4, shot analysis quality is bounded by weak labels; set
  expectations accordingly and lean on movement metrics (M0) for early value.

---

### TL;DR
We are not choosing "generic model vs squash model." We are building a squash-specialized
**perception + data pipeline** whose by-product is the labeled dataset that a trained
squash shot-classifier requires — then putting a generic LLM on top *only* to reason over
the resulting numbers. The next concrete build is **ball tracking (M1)**.
