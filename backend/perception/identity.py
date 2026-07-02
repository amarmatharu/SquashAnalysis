"""
Player identity — extract recognisable crops of each player and a colour
signature used to lock identity (so P1/P2 never swap when players cross).

Product flow:
  1. extract_player_crops()  → finds a frame where both players are clearly
     separated, returns a crop image + colour signature for each.
  2. The user names each crop in the UI.
  3. The colour signatures are stored on the match and fed back into player
     detection so every detection is assigned to the right named player by
     appearance, not just position.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def color_signature(frame_bgr: np.ndarray, box: Tuple[float, float, float, float]) -> List[float]:
    """HSV histogram of the player's TORSO region (upper-middle of the box).

    The torso carries the shirt colour — the most reliable identity cue. We skip
    the head and legs to avoid skin/floor contamination. Returned as a flat,
    L1-normalised list so it is JSON-serialisable and comparable via intersection.
    """
    import cv2

    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    w = x2 - x1
    if h < 8 or w < 8:
        return []
    # Torso: vertical 20–55% of the box, horizontal middle 60%.
    ty1 = y1 + int(0.20 * h)
    ty2 = y1 + int(0.55 * h)
    tx1 = x1 + int(0.20 * w)
    tx2 = x1 + int(0.80 * w)
    H, W = frame_bgr.shape[:2]
    ty1, ty2 = max(0, ty1), min(H, ty2)
    tx1, tx2 = max(0, tx1), min(W, tx2)
    patch = frame_bgr[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return []
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 2D hue-saturation histogram (robust to brightness changes)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    return hist.tolist()


def signature_similarity(a: List[float], b: List[float]) -> float:
    """Histogram intersection similarity in [0, 1]; higher = more alike."""
    if not a or not b or len(a) != len(b):
        return 0.0
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return float(np.minimum(aa, bb).sum())


def _crop_b64(frame_bgr: np.ndarray, box: Tuple[float, float, float, float],
              pad: float = 0.12) -> str:
    """Return a base64 JPEG crop of the player box (with padding)."""
    import cv2

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = frame_bgr.shape[:2]
    cx1 = max(0, int(x1 - pad * w))
    cy1 = max(0, int(y1 - pad * h))
    cx2 = min(W, int(x2 + pad * w))
    cy2 = min(H, int(y2 + pad * h))
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return ""
    # Upscale small crops so the UI shows a clear image
    ch, cw = crop.shape[:2]
    if cw < 180:
        scale = 180 / cw
        crop = cv2.resize(crop, (180, int(ch * scale)))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return ""
    import base64
    return base64.b64encode(buf).decode("utf-8")


def _on_court_filter(calibration, native_w: int, native_h: int, proc_w: int):
    """Return a predicate(feet_x, feet_y) -> bool for 'is on the court', using a
    processed-space court model. Returns None if no usable calibration."""
    if calibration is None:
        return None
    try:
        from .court import CourtModel, CourtCalibration, COURT_WIDTH, COURT_LENGTH
        scale = proc_w / native_w if native_w else 1.0

        def _sc(pt):
            return (pt[0] * scale, pt[1] * scale) if pt is not None else None

        proc_calib = CourtCalibration(
            front_left=_sc(calibration.front_left), front_right=_sc(calibration.front_right),
            back_right=_sc(calibration.back_right), back_left=_sc(calibration.back_left),
        )
        model = CourtModel(proc_calib)

        def _pred(fx, fy):
            cx, cy = model.to_court((fx, fy))
            # small margin — players can stand just on the lines
            return -0.3 <= cx <= COURT_WIDTH + 0.3 and -0.3 <= cy <= COURT_LENGTH + 0.3

        return _pred
    except Exception:
        return None


def extract_player_crops(
    video_path: str,
    calibration=None,
    start_s: float = 0.0,
    scan_s: float = 60.0,
    samples: int = 40,
) -> Dict:
    """Find a frame where both players are clearly separated and return a crop +
    colour signature for each.

    Spectators/bystanders are excluded by keeping only people whose feet map
    INSIDE the court (requires calibration). Without calibration we fall back to
    the two largest central detections, which is less reliable.
    """
    import cv2
    from .players import get_player_detector

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(start_s * fps)
    end_f = min(total, int((start_s + scan_s) * fps))
    if end_f <= start_f:
        cap.release()
        return {"ok": False, "error": "Video too short"}

    proc_w = min(native_w, 1280)
    on_court = _on_court_filter(calibration, native_w, native_h, proc_w)

    sample_frames = np.linspace(start_f, end_f - 1, num=min(samples, end_f - start_f)).astype(int)
    detector = get_player_detector()

    best = None  # (score, frame, [box_a, box_b])
    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if w > 1280:
            frame = cv2.resize(frame, (1280, int(h * 1280 / w)))

        pf = detector.detect_frames([frame], int(fi), fps)[0]
        # Keep only on-court people (filters spectators behind/beside the court)
        people = pf.players
        if on_court is not None:
            people = [p for p in people if on_court(p.feet_x, p.feet_y)]
        if len(people) < 2:
            continue
        # Two most confident on-court boxes
        boxes = sorted(people, key=lambda p: -p.conf)[:2]
        a, b = boxes[0], boxes[1]
        sep = abs(a.cx - b.cx)
        min_w = max(20, min(a.width, b.width))
        if sep < min_w * 1.1:
            continue  # too close / overlapping → bad crops
        score = (a.height + b.height) + 200 * (a.conf + b.conf)
        if best is None or score > best[0]:
            best = (score, frame.copy(), [a, b])

    cap.release()
    if best is None:
        msg = ("Could not find a clean two-player frame on the court. "
               + ("Check the court calibration." if on_court else
                  "Calibrate the court first so spectators can be excluded."))
        return {"ok": False, "error": msg}

    _, frame, boxes = best
    # Order players left→right for a stable, predictable presentation
    boxes = sorted(boxes, key=lambda p: p.cx)
    players = []
    for i, pb in enumerate(boxes, start=1):
        box = (pb.x1, pb.y1, pb.x2, pb.y2)
        players.append({
            "slot": i,                       # 1 = left-most, 2 = right-most
            "crop_b64": _crop_b64(frame, box),
            "color_sig": color_signature(frame, box),
            "box": [round(v, 1) for v in box],
        })
    return {"ok": True, "players": players}
