"""
Layer 3 (extension) — player pose / skeleton.

Adds 17-keypoint COCO pose per player (YOLOv8-pose). The skeleton gives us:
  • the racket-arm WRIST → sharper racket-contact detection (the ball contact is
    at the wrist, not the body centre) and forehand/backhand from which side the
    wrist is on relative to the body.
  • biomechanics: stance width, lunge depth, reach, balance — the raw material for
    swing/movement quality later.

This is a thin wrapper that returns keypoints in the processed-frame pixel space,
to be matched to the colour-identified players from players.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
POSE_WEIGHTS = os.path.join(WEIGHTS_DIR, "yolov8n-pose.pt")

# COCO-17 keypoint indices
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}


@dataclass
class PlayerPose:
    frame_index: int
    keypoints: np.ndarray            # (17, 3): x, y, conf  (pixels)
    box: Tuple[float, float, float, float]
    conf: float

    def kp(self, name: str) -> Optional[Tuple[float, float, float]]:
        i = KP[name]
        x, y, c = self.keypoints[i]
        return (float(x), float(y), float(c)) if c > 0.2 else None

    def racket_wrist(self) -> Optional[Tuple[float, float]]:
        """Best-guess racket hand = the higher-confidence wrist, breaking ties by
        the wrist that is farther from the body centre (mid-swing reach)."""
        lw, rw = self.kp("l_wrist"), self.kp("r_wrist")
        cands = [w for w in (lw, rw) if w]
        if not cands:
            return None
        # centre of hips as body reference
        lh, rh = self.kp("l_hip"), self.kp("r_hip")
        hips = [h for h in (lh, rh) if h]
        if hips:
            cx = np.mean([h[0] for h in hips])
            best = max(cands, key=lambda w: abs(w[0] - cx) + w[2])
        else:
            best = max(cands, key=lambda w: w[2])
        return (best[0], best[1])

    def swing_side(self) -> Optional[str]:
        """forehand/backhand cue: which side of the body the racket wrist is on."""
        w = self.racket_wrist()
        ls, rs = self.kp("l_shoulder"), self.kp("r_shoulder")
        sh = [s for s in (ls, rs) if s]
        if not w or not sh:
            return None
        cx = np.mean([s[0] for s in sh])
        return "right_side" if w[0] >= cx else "left_side"

    def stance_width_px(self) -> Optional[float]:
        la, ra = self.kp("l_ankle"), self.kp("r_ankle")
        if la and ra:
            return float(abs(la[0] - ra[0]))
        return None


class PoseDetector:
    def __init__(self, conf: float = 0.35):
        from ultralytics import YOLO
        self._model = YOLO(POSE_WEIGHTS)
        self._conf = conf

    def detect(self, frame_bgr, frame_index: int = 0) -> List[PlayerPose]:
        """Return pose for each detected person in a frame."""
        res = self._model(frame_bgr, conf=self._conf, verbose=False, device="")[0]
        out: List[PlayerPose] = []
        if res.keypoints is None or res.boxes is None:
            return out
        kpts = res.keypoints.data.cpu().numpy()      # (n, 17, 3)
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for i in range(len(kpts)):
            out.append(PlayerPose(frame_index=frame_index, keypoints=kpts[i],
                                  box=tuple(boxes[i].tolist()), conf=float(confs[i])))
        return out


_DETECTOR: Optional[PoseDetector] = None


def get_pose_detector() -> PoseDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = PoseDetector()
    return _DETECTOR
