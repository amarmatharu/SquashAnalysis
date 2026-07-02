"""
Ball-annotation bootstrap engine.

Classical ball tracking can *propose* ball-like tracks but cannot confirm them
(reflections, racket motion, and gallery movement all look ball-like). This module
turns those proposals into a human-labelling task: run the detector+tracker over a
time window, keep the top candidate tracks ranked by plausibility, and attach a
small image crop per point so a reviewer can confirm/reject at a glance.

Confirmed tracks become ground-truth ball positions — the labelled dataset a future
TrackNet trains on. This is the data-first first step of M1: no dataset, no model.

Output is intentionally compact (few tracks, small crops) so it can be stored and
shipped to a review UI without huge payloads.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ball import MotionBallDetector, BallTracker, BallPoint
from .court import CourtCalibration, CourtModel

PERSON_CLASS_ID = 0
# Cap processing width so high-res (1080p/4K) clips don't exhaust memory when a
# whole window of frames is held at once. Detectors downscale to ~512px anyway.
PROC_MAX_WIDTH = 1280


@dataclass
class CandidatePoint:
    frame_index: int
    timestamp: float
    x: float                       # image pixels
    y: float
    crop_b64: Optional[str] = None  # small JPEG crop centred on the point
    court_x_m: Optional[float] = None
    court_y_m: Optional[float] = None


@dataclass
class CandidateTrack:
    track_id: int
    quality: float
    points: List[CandidatePoint] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "quality": round(self.quality, 3),
            "num_points": len(self.points),
            "points": [
                {
                    "frame_index": p.frame_index,
                    "timestamp": round(p.timestamp, 3),
                    "x": round(p.x, 1),
                    "y": round(p.y, 1),
                    "crop_b64": p.crop_b64,
                    "court_x_m": p.court_x_m,
                    "court_y_m": p.court_y_m,
                }
                for p in self.points
            ],
        }


def _crop_b64(frame, x: float, y: float, half: int = 30, out: int = 120) -> Optional[str]:
    """JPEG crop centred on (x, y), upscaled for visibility.

    ``half`` controls how much context around the candidate is captured (a 2*half
    px square); ``out`` is the upscaled output size. Larger values make the (tiny)
    ball easier to see in the review UI.
    """
    import cv2

    h, w = frame.shape[:2]
    x0, y0 = int(max(0, x - half)), int(max(0, y - half))
    x1, y1 = int(min(w, x + half)), int(min(h, y + half))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame[y0:y1, x0:x1]
    patch = cv2.resize(patch, (out, out), interpolation=cv2.INTER_LANCZOS4)
    ok, buf = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")


def extract_frames_for_marking(
    video_path: str,
    start_s: float = 0.0,
    count: int = 24,
    step: int = 2,
    max_width: int = 640,
) -> Dict:
    """Return a strip of frames (base64 JPEG) for manual ball marking.

    ``step`` skips source frames between samples (e.g. step=2 → every other
    frame). Frames are downscaled to ``max_width`` for transport; the reviewer
    clicks the ball and the UI sends back *normalized* coordinates, so the scale
    does not matter. Native dimensions are returned for reference.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    start_f = int(start_s * fps)

    frames_out: List[Dict] = []
    for k in range(count):
        fi = start_f + k * max(1, step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, f = cap.read()
        if not ret:
            break
        h, w = f.shape[:2]
        if w > max_width:
            f = cv2.resize(f, (max_width, int(h * max_width / w)))
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        frames_out.append({
            "frame_index": fi,
            "timestamp": round(fi / fps, 3),
            "b64": base64.b64encode(buf).decode("utf-8"),
        })
    cap.release()
    return {
        "fps": fps,
        "native_width": native_w,
        "native_height": native_h,
        "step": step,
        "frames": frames_out,
    }


def propagate_ball(
    video_path: str,
    start_frame_index: int,
    x: float,
    y: float,
    n_frames: int = 30,
    win: int = 21,
) -> Dict:
    """Track a single clicked ball point across the next frames via optical flow.

    You click the ball once; Lucas-Kanade optical flow follows that point frame to
    frame, so one click yields a whole short track instead of one label. Tracking
    stops if the point is lost (status 0) or leaves the frame. Each tracked point
    gets a review crop. The human verifies/trims the result before saving — so a
    wrong propagation never silently poisons the dataset.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return {"points": [], "error": "could not read start frame"}

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
    lk_params = dict(
        winSize=(win, win), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    points = [{
        "frame_index": start_frame_index,
        "timestamp": round(start_frame_index / fps, 3),
        "x": round(float(x), 1), "y": round(float(y), 1),
        "crop_b64": _crop_b64(frame, x, y), "seed": True,
    }]

    h, w = frame.shape[:2]
    fi = start_frame_index
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        if p1 is None or st is None or st[0][0] == 0:
            break
        nx, ny = float(p1[0][0][0]), float(p1[0][0][1])
        if not (0 <= nx < w and 0 <= ny < h):
            break
        points.append({
            "frame_index": fi,
            "timestamp": round(fi / fps, 3),
            "x": round(nx, 1), "y": round(ny, 1),
            "crop_b64": _crop_b64(frame, nx, ny), "seed": False,
        })
        prev_gray, p0 = gray, p1

    cap.release()
    return {"fps": fps, "native_width": w, "native_height": h,
            "tracked": len(points), "points": points}


def extract_candidate_tracks(
    video_path: str,
    start_s: float = 0.0,
    duration_s: float = 8.0,
    max_tracks: int = 12,
    model_name: str = "yolo11n.pt",
    calibration: Optional[CourtCalibration] = None,
    device: Optional[str] = None,
) -> Dict:
    """Run detection+tracking over a window and return ranked candidate tracks.

    Each track carries per-point image crops for human review. A court
    calibration, if given, adds real court-metre coordinates to each point.
    """
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = int(start_s * fps)
    n_frames = int(duration_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    # Process at a capped width so big (1080p/4K) clips don't exhaust memory —
    # the detectors resize internally anyway. ``scale`` maps proc px → native px.
    frames: List[np.ndarray] = []
    scale = 1.0
    for _ in range(n_frames):
        ret, f = cap.read()
        if not ret:
            break
        nh, nw = f.shape[:2]
        if nw > PROC_MAX_WIDTH:
            f = cv2.resize(f, (PROC_MAX_WIDTH, int(round(nh * PROC_MAX_WIDTH / nw))))
            scale = nw / float(f.shape[1])
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        return {"fps": fps, "start_s": start_s, "tracks": [], "frame_count": len(frames)}

    model = YOLO(model_name)
    detector = MotionBallDetector()

    # Player boxes per frame (to mask out limb/racket motion).
    player_boxes_per_frame: List[List] = []
    for i in range(len(frames)):
        boxes = []
        r = model.predict(
            frames[i], classes=[PERSON_CLASS_ID], conf=0.35, imgsz=640,
            device=device, verbose=False,
        )
        if r and r[0].boxes is not None and r[0].boxes.xyxy is not None:
            for b in r[0].boxes.xyxy.cpu().numpy():
                boxes.append(tuple(float(v) for v in b))
        player_boxes_per_frame.append(boxes)

    per_frame = detector.detect_window(
        frames, start_f, fps, player_boxes_per_frame=player_boxes_per_frame
    )

    tracker = BallTracker()
    raw_tracks = tracker.build_tracks(per_frame)
    scored = [(tracker._track_quality(t), t) for t in raw_tracks]
    scored = [(q, t) for q, t in scored if q > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:max_tracks]

    court = CourtModel(calibration) if calibration else None

    out_tracks: List[CandidateTrack] = []
    for tid, (q, t) in enumerate(scored):
        ct = CandidateTrack(track_id=tid, quality=float(q))
        for p in t.points:
            local_idx = p.frame_index - start_f
            frame = frames[local_idx] if 0 <= local_idx < len(frames) else None
            crop = _crop_b64(frame, p.x, p.y) if frame is not None else None  # proc coords
            nx, ny = p.x * scale, p.y * scale                                  # native coords
            cx_m = cy_m = None
            if court is not None:
                cm = court.to_court((nx, ny))
                cx_m, cy_m = round(cm[0], 2), round(cm[1], 2)
            ct.points.append(
                CandidatePoint(
                    frame_index=p.frame_index,
                    timestamp=p.timestamp,
                    x=round(nx, 1), y=round(ny, 1), crop_b64=crop,
                    court_x_m=cx_m, court_y_m=cy_m,
                )
            )
        out_tracks.append(ct)

    return {
        "fps": fps,
        "start_s": start_s,
        "duration_s": duration_s,
        "frame_count": len(frames),
        "num_tracks": len(out_tracks),
        "tracks": [t.to_dict() for t in out_tracks],
    }
