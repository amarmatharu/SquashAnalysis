#!/usr/bin/env python
"""
Train the TrackNet squash-ball model from labelled data.

Pulls every confirmed ball point (from the annotation tool — both reviewed
candidate tracks and manual marks) out of MongoDB, locates each match's video in
uploads/, and trains TrackNet, saving weights to ``perception/weights/tracknet.pt``
— the exact path ``get_ball_detector()`` auto-loads, so the trained model goes
live everywhere with no further changes.

Usage:
    cd backend
    .venv/bin/python train_ball_model.py                 # train with defaults
    .venv/bin/python train_ball_model.py --epochs 40     # more epochs
    .venv/bin/python train_ball_model.py --dry-run       # just report data counts

Prerequisite: enough labelled ball points (default >= 200). Label them in the app
(History -> match -> Label Ball Data -> Review candidates / Manual mark).
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
UPLOAD_DIR = ROOT / "uploads"
WEIGHTS_PATH = ROOT / "perception" / "weights" / "tracknet.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--out", type=str, default=str(WEIGHTS_PATH))
    ap.add_argument("--dry-run", action="store_true", help="report data, don't train")
    args = ap.parse_args()

    # ----- pull labelled ball points from Mongo -----
    mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.environ.get("DB_NAME", "squashsense")
    db = MongoClient(mongo_url)[db_name]

    import random as _random
    manual, selftrain = [], []
    match_files = {}
    for doc in db.ball_labels.find({"label": "ball"}):
        mid = doc["match_id"]
        if mid not in match_files:
            m = db.matches.find_one({"id": mid}, {"video_filename": 1})
            match_files[mid] = m["video_filename"] if m else None
        if not match_files[mid]:
            continue
        bucket = selftrain if doc.get("source") == "selftrain" else manual
        for p in doc.get("points", []):
            bucket.append({"match_id": mid, "frame_index": p["frame_index"],
                           "x": p["x"], "y": p["y"]})

    # Drift guard: cap self-trained pseudo-labels at the manual-label count (≤50%).
    if manual and len(selftrain) > len(manual):
        _random.shuffle(selftrain)
        print(f"Capping self-trained {len(selftrain)} -> {len(manual)} (drift guard)")
        selftrain = selftrain[: len(manual)]
    samples = manual + selftrain

    n_matches = len({s["match_id"] for s in samples})
    print(f"Labelled ball points: {len(samples)} ({len(manual)} manual + {len(selftrain)} self-train) across {n_matches} match(es)")

    if args.dry_run:
        print("Dry run — not training.")
        return

    if len(samples) < args.min_samples:
        print(f"\n⚠ Need >= {args.min_samples} points to train, have {len(samples)}.")
        print("  Label more in the app (History -> match -> Label Ball Data).")
        print("  A model trained on too few points is worse than the classical fallback.")
        sys.exit(1)

    def video_resolver(match_id: str) -> str:
        return str(UPLOAD_DIR / match_files[match_id])

    from perception.tracknet import train_tracknet, pick_device
    print(f"Device: {pick_device()}")
    os.makedirs(Path(args.out).parent, exist_ok=True)

    report = train_tracknet(
        samples=samples,
        video_resolver=video_resolver,
        out_weights=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_samples=args.min_samples,
    )
    print("\nResult:", report)
    if report.get("trained"):
        print(f"\n✅ Weights saved to {args.out}")
        print("   The ball detector will now use TrackNet automatically — rebuild a")
        print("   timeline (Timeline tab) to see real shot events.")


if __name__ == "__main__":
    main()
