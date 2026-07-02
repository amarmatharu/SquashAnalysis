"""
Structured rally timeline orchestration (M2).

Ties the perception modules together over a time window into the single source of
truth the analytics/reasoning layers consume:

    frames
      -> YOLO players (per frame)  ──► player court tracks (homography)
      -> ball detector + tracker   ──► ball court trajectory
      -> shot-event detection      ──► shots (contacts)
      -> rally grouping            ──► structured timeline (events.build_timeline)

Runs on a bounded window (a rally / segment) to stay tractable on CPU; full-match
processing is the same call over successive windows. The ball detector is chosen
by ``get_ball_detector`` — TrackNet if trained weights exist, else classical — so
this whole pipeline upgrades automatically when the model is trained.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .ball import BallTracker, get_ball_detector
from .court import COURT_WIDTH, CourtCalibration, CourtModel
from .events import BallSample, PlayerSample, build_timeline, detect_shot_events

PERSON_CLASS_ID = 0


def _assign_players_by_side(
    foot_points_court: List[Tuple[float, float]]
) -> Dict[int, str]:
    """Within a short window, label the two players left/right by court x.

    player1 = left half (smaller x), player2 = right half. Sufficient for striker
    attribution over a rally; full cross-window identity is a later re-id concern.
    """
    if not foot_points_court:
        return {}
    xs = sorted(range(len(foot_points_court)), key=lambda i: foot_points_court[i][0])
    mapping = {}
    for rank, i in enumerate(xs):
        mapping[i] = "player1" if rank < len(xs) / 2 else "player2"
    return mapping


def analyze_rally_window(
    video_path: str,
    calibration: CourtCalibration,
    start_s: float = 0.0,
    duration_s: float = 8.0,
    model_name: str = "yolo11n.pt",
    device: Optional[str] = None,
    setup: str = "phone",
) -> Dict:
    """Build a structured rally timeline for a window. Requires court calibration
    (shots/positions are reported in court metres)."""
    import cv2
    from ultralytics import YOLO

    court = CourtModel(calibration)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = int(start_s * fps)
    n = int(duration_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(n):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        return {"error": "not enough frames", "frame_count": len(frames)}

    model = YOLO(model_name)

    # Per-frame: player boxes + each player's court foot position (left/right).
    player_boxes_per_frame: List[List] = []
    players: Dict[str, List[PlayerSample]] = {"player1": [], "player2": []}
    for i, frame in enumerate(frames):
        t = (start_f + i) / fps
        boxes = []
        r = model.predict(frame, classes=[PERSON_CLASS_ID], conf=0.35, imgsz=640,
                          device=device, verbose=False)
        if r and r[0].boxes is not None and r[0].boxes.xyxy is not None:
            for b in r[0].boxes.xyxy.cpu().numpy():
                boxes.append(tuple(float(v) for v in b))
        # keep two largest (the players)
        boxes.sort(key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]), reverse=True)
        boxes = boxes[:2]
        player_boxes_per_frame.append(boxes)

        feet_court = []
        for (x1, y1, x2, y2) in boxes:
            foot = ((x1 + x2) / 2.0, y2)
            cxy = court.to_court(foot)
            feet_court.append(cxy)
        side = _assign_players_by_side(feet_court)
        for idx, cxy in enumerate(feet_court):
            label = side.get(idx, "player1")
            players[label].append(PlayerSample(t=t, court_x=cxy[0], court_y=cxy[1]))

    # Ball: detector (TrackNet if available) -> best trajectory -> court coords.
    detector = get_ball_detector(setup)
    per_frame = detector.detect_window(frames, start_f, fps, player_boxes_per_frame)
    ball_img = BallTracker().best_trajectory(per_frame)
    ball: List[BallSample] = []
    for p in ball_img:
        cx, cy = court.to_court((p.x, p.y))
        ball.append(BallSample(frame_index=p.frame_index, t=p.timestamp,
                               court_x=cx, court_y=cy))

    events = detect_shot_events(ball, players)
    timeline = build_timeline(events, ball)
    timeline.update({
        "start_s": start_s,
        "duration_s": duration_s,
        "fps": fps,
        "detector": type(detector).__name__,
        "ball_points": len(ball),
        "frame_count": len(frames),
    })
    return timeline
