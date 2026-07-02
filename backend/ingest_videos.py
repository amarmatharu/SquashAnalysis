#!/usr/bin/env python
"""
Bulk-ingest a folder of past-game videos into the training library.

Registers every video in a folder as a training source (a "library" match) so it
flows through the ball-labelling and training tools. Videos are hardlinked into
uploads/ when possible (instant, no extra disk), else copied.

Usage:
    cd backend
    .venv/bin/python ingest_videos.py /path/to/your/squash/videos
    .venv/bin/python ingest_videos.py /path/to/videos --copy   # force copy
    .venv/bin/python ingest_videos.py /path/to/videos --recursive

After ingesting, open the app -> Training Library to label and train across them.
"""

import argparse
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing video files")
    ap.add_argument("--recursive", action="store_true", help="scan subfolders too")
    ap.add_argument("--copy", action="store_true", help="copy instead of hardlink")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    it = folder.rglob("*") if args.recursive else folder.iterdir()
    files = sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        print("No video files found.")
        sys.exit(1)

    db = MongoClient(os.environ.get("MONGO_URL", "mongodb://localhost:27017"))[
        os.environ.get("DB_NAME", "squashsense")
    ]

    # Generic container folder names that shouldn't appear in titles.
    GENERIC = {"videos", "video", "training", "footage", folder.name.lower()}

    def make_title(f: Path) -> str:
        parent = f.parent.name
        if parent and parent.lower() not in GENERIC:
            return f"{parent} — {f.stem}"   # e.g. "Amira Ahluwalia — RynaVsAmira_Game1"
        return f.stem

    print(f"Found {len(files)} video(s). Ingesting…")
    n = 0
    for f in files:
        ext = f.suffix.lower() or ".mp4"
        unique = f"{uuid.uuid4()}{ext}"
        dest = UPLOAD_DIR / unique
        try:
            if args.copy:
                shutil.copy2(f, dest)
            else:
                try:
                    os.link(f, dest)
                except Exception:
                    shutil.copy2(f, dest)
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
            continue

        db.matches.insert_one({
            "id": str(uuid.uuid4()),
            "title": make_title(f),
            "opponent": f.parent.name if f.parent.name.lower() not in GENERIC else None,
            "video_filename": unique,
            "upload_time": datetime.now(timezone.utc).isoformat(),
            "status": "library",
            "source": "library",
            "progress": 0,
            "total_shots": 0, "total_rallies": 0,
            "shots": [], "rallies": [], "shot_distribution": {},
            "player1_stats": {}, "player2_stats": {}, "movement_data": [],
            "player_metrics": {}, "swing_analysis": [], "key_insights": [],
        })
        n += 1
        print(f"  ✓ {f.name}")

    print(f"\nIngested {n}/{len(files)} videos into the training library.")
    print("Open the app -> Training Library to label balls and train.")


if __name__ == "__main__":
    main()
