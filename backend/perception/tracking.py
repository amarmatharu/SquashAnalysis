"""
Player detection and tracking.

Uses an Ultralytics YOLO person detector with a built-in multi-object tracker
(ByteTrack) to follow the (usually two) players across frames. Each detection's
*foot point* — the bottom-centre of the bounding box — is the contact point with
the floor, which is what the court homography needs to produce real-world
position. Tracks are kept stable by track id so we can attribute movement to a
consistent player over time.

The two on-court players are picked per frame as the two largest person boxes
inside the court polygon, which is robust to referees / crowd in broadcast video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# COCO class id for "person" in the standard YOLO models.
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    torso_hist: Optional[np.ndarray] = None   # colour signature for player re-id

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Bottom-centre of the box: the player's contact with the floor."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def torso_histogram(frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]):
    """HS colour histogram of a player's torso — the re-identification signature.

    Following Baclig et al. (2020), players are identified by their shirt colour,
    which is robust to the ByteTrack id switches that occur under occlusion. We
    sample the upper-middle of the bounding box (torso, avoiding head and legs)
    and build a normalised Hue-Saturation histogram.
    """
    import cv2

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None
    tx1, tx2 = int(x1 + 0.20 * w), int(x1 + 0.80 * w)
    ty1, ty2 = int(y1 + 0.20 * h), int(y1 + 0.55 * h)
    H, W = frame_bgr.shape[:2]
    tx1, tx2 = max(0, tx1), min(W, tx2)
    ty1, ty2 = max(0, ty1), min(H, ty2)
    if tx2 <= tx1 or ty2 <= ty1:
        return None
    crop = frame_bgr[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def _hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram correlation in [-1, 1]; higher = more likely the same player."""
    import cv2
    if a is None or b is None:
        return -1.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


@dataclass
class FrameDetections:
    frame_index: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)


def _point_in_polygon(pt: Tuple[float, float], poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test. poly is Nx2 image points."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


class PlayerTracker:
    """Thin wrapper over an Ultralytics YOLO tracker scoped to court players.

    Parameters
    ----------
    model_name:
        Ultralytics weights to load. ``yolo11n.pt`` (nano) is the fast default;
        swap to ``yolo11s/m`` for more accuracy. Weights auto-download on first
        use and are cached.
    court_polygon:
        Optional Nx2 array of the court's image corners. When provided, only
        detections whose foot point sits inside the polygon are kept, which
        filters out referees, ball boys and crowd.
    conf:
        Minimum detection confidence.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        court_polygon: Optional[np.ndarray] = None,
        conf: float = 0.35,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.court_polygon = court_polygon
        self.conf = conf
        self.device = device

    def _keep(self, det: Detection) -> bool:
        if self.court_polygon is None:
            return True
        return _point_in_polygon(det.foot_point, self.court_polygon)

    def track_video(
        self,
        video_path: str,
        sample_every: int = 3,
        max_players: int = 2,
        imgsz: int = 640,
        max_frames: Optional[int] = None,
    ) -> List[FrameDetections]:
        """Run tracking over the video.

        Returns one FrameDetections per *sampled* frame. ``sample_every`` keeps
        cost bounded (e.g. every 3rd frame ~= 10 fps on 30 fps footage), which is
        plenty dense for movement analysis while staying tractable on CPU.
        ``max_frames`` optionally caps how many *source* frames are read (e.g. to
        analyse only the opening of a long video).
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        results: List[FrameDetections] = []
        frame_index = -1

        # persist=True keeps ByteTrack ids stable across the streamed frames.
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
            if frame_index % sample_every != 0:
                continue

            timestamp = frame_index / fps
            yolo_out = self.model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=self.conf,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
                tracker="bytetrack.yaml",
            )

            dets: List[Detection] = []
            if yolo_out and yolo_out[0].boxes is not None:
                boxes = yolo_out[0].boxes
                ids = boxes.id
                xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                id_arr = ids.cpu().numpy() if ids is not None else None
                for i in range(len(xyxy)):
                    tid = int(id_arr[i]) if id_arr is not None else -1
                    det = Detection(
                        track_id=tid,
                        bbox=tuple(float(v) for v in xyxy[i]),
                        confidence=float(confs[i]) if len(confs) else 0.0,
                    )
                    if self._keep(det):
                        dets.append(det)

            # Keep the most prominent on-court figures (the players).
            dets.sort(key=lambda d: d.area, reverse=True)
            dets = dets[:max_players]

            # Colour signature for appearance-based player identification.
            for d in dets:
                d.torso_hist = torso_histogram(frame, d.bbox)

            results.append(
                FrameDetections(
                    frame_index=frame_index, timestamp=timestamp, detections=dets
                )
            )

        cap.release()
        return results


def build_appearance_references(
    frames: List[FrameDetections],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build a torso-colour reference histogram for player1 and player2.

    Bootstraps from the two ByteTrack ids that appear most (the two players), then
    averages each one's torso histograms into a stable colour signature. These
    references let us re-identify players by shirt colour even when ByteTrack
    fragments an id under occlusion (Baclig et al. 2020).
    """
    label_map = assign_stable_player_labels(frames)
    # collect histograms per logical label
    acc: Dict[str, List[np.ndarray]] = {"player1": [], "player2": []}
    for fd in frames:
        for d in fd.detections:
            if d.torso_hist is None:
                continue
            label = label_map.get(d.track_id)
            if label in acc:
                acc[label].append(d.torso_hist)

    def _avg(hists):
        if not hists:
            return None
        m = np.mean(np.stack(hists), axis=0)
        n = np.linalg.norm(m)
        return (m / n).astype(np.float32) if n > 0 else m.astype(np.float32)

    return _avg(acc["player1"]), _avg(acc["player2"])


def assign_stable_player_labels(
    frames: List[FrameDetections],
) -> Dict[int, str]:
    """Map raw ByteTrack ids to stable 'player1'/'player2' labels.

    ByteTrack ids can fragment when a player is occluded. We collapse them to two
    logical players by total time-on-court: the two track ids that appear in the
    most sampled frames become player1/player2. Remaining ids are attached to
    whichever of the two they most overlap with in time is left for a future
    re-id pass; for now they map to the nearest of the two by frame co-occurrence.
    """
    from collections import Counter

    counts: Counter = Counter()
    for fd in frames:
        for d in fd.detections:
            counts[d.track_id] += 1

    top = [tid for tid, _ in counts.most_common(2)]
    label_map: Dict[int, str] = {}
    for rank, tid in enumerate(top):
        label_map[tid] = f"player{rank + 1}"

    # Any other ids: assign to the player they are seen alongside least (i.e. the
    # missing one in frames where only one main id is present) as a cheap re-id.
    for tid in counts:
        if tid in label_map:
            continue
        with_p1 = with_p2 = 0
        for fd in frames:
            present = {d.track_id for d in fd.detections}
            if tid in present:
                if top and top[0] in present:
                    with_p1 += 1
                if len(top) > 1 and top[1] in present:
                    with_p2 += 1
        # If it rarely co-occurs with player1, it is probably player1 fragmented.
        label_map[tid] = "player1" if with_p1 <= with_p2 else "player2"

    return label_map
