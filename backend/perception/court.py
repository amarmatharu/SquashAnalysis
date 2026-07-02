"""
Court geometry and homography for a standard singles squash court.

A squash *singles* court floor is 6.4 m wide and 9.75 m long. We define a
real-world floor coordinate system in metres:

    origin (0, 0) = front-left floor corner (left as seen from behind the court,
                    i.e. from the back wall looking towards the front wall)
    x axis        = across the court, 0 .. 6.4   (left wall -> right wall)
    y axis        = down the court,   0 .. 9.75  (front wall -> back wall)

Key reference geometry (World Squash dimensions):

    front wall          y = 0.0
    back wall           y = 9.75
    short line          y = 5.49      (4.26 m from the back wall)
    half-court line     x = 3.2, for y in [5.49, 9.75]
    the "T"             (3.2, 5.49)   intersection of short + half-court lines
    service boxes       1.6 m x 1.6 m squares in each back quarter, their
                        front edge on the short line, outer edge on the side wall

A homography maps points on the (planar) court floor in the image to these
metre coordinates. Calibration needs the four floor corners in image pixels;
those can be supplied manually (user clicks) or by an auto-detector later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----- Standard singles court constants (metres) -----
COURT_WIDTH = 6.4
COURT_LENGTH = 9.75
SHORT_LINE_FROM_BACK = 4.26
SHORT_LINE_Y = COURT_LENGTH - SHORT_LINE_FROM_BACK   # 5.49 from front wall
HALF_COURT_X = COURT_WIDTH / 2.0                      # 3.2
SERVICE_BOX = 1.6
T_POINT = (HALF_COURT_X, SHORT_LINE_Y)               # (3.2, 5.49)

# Named landmarks on the floor, in metres. Useful for drawing/validation and
# for "distance to the T" style metrics.
LANDMARKS: Dict[str, Tuple[float, float]] = {
    "front_left": (0.0, 0.0),
    "front_right": (COURT_WIDTH, 0.0),
    "back_left": (0.0, COURT_LENGTH),
    "back_right": (COURT_WIDTH, COURT_LENGTH),
    "t": T_POINT,
    "short_left": (0.0, SHORT_LINE_Y),
    "short_right": (COURT_WIDTH, SHORT_LINE_Y),
    "left_service_box": (SERVICE_BOX / 2.0, SHORT_LINE_Y + SERVICE_BOX / 2.0),
    "right_service_box": (COURT_WIDTH - SERVICE_BOX / 2.0,
                          SHORT_LINE_Y + SERVICE_BOX / 2.0),
}

# Court depth thirds (front / mid / back) measured from the front wall.
FRONT_THIRD_Y = COURT_LENGTH / 3.0          # 3.25
BACK_THIRD_Y = 2.0 * COURT_LENGTH / 3.0     # 6.5


@dataclass
class CourtCalibration:
    """The four floor corners of the court in image pixel coordinates, plus the
    optional tin line and front-wall out line.

    All coordinates are in image pixels. For storage and the calibration UI,
    use the normalized (0..1) form via ``from_normalized`` / ``to_normalized``.
    """

    front_left: Tuple[float, float]
    front_right: Tuple[float, float]
    back_right: Tuple[float, float]
    back_left: Tuple[float, float]
    # Tin line — the bottom strip on the front wall (~48 cm high).
    # Two pixel points that define the line (left and right ends visible in frame).
    tin_left: Optional[Tuple[float, float]] = None
    tin_right: Optional[Tuple[float, float]] = None

    def image_points(self) -> np.ndarray:
        return np.array(
            [self.front_left, self.front_right, self.back_right, self.back_left],
            dtype=np.float32,
        )

    @staticmethod
    def world_points() -> np.ndarray:
        return np.array(
            [
                LANDMARKS["front_left"],
                LANDMARKS["front_right"],
                LANDMARKS["back_right"],
                LANDMARKS["back_left"],
            ],
            dtype=np.float32,
        )

    def tin_y_at_x(self, px_x: float) -> Optional[float]:
        """Pixel y-coordinate of the tin line at a given x by linear interpolation."""
        if self.tin_left is None or self.tin_right is None:
            return None
        x0, y0 = self.tin_left
        x1, y1 = self.tin_right
        if abs(x1 - x0) < 1:
            return float(y0)
        t = (px_x - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def is_tin_hit(self, px_x: float, px_y: float, margin: float = 8.0) -> bool:
        """True if a ball at (px_x, px_y) is at or below the tin line."""
        tin_y = self.tin_y_at_x(px_x)
        return tin_y is not None and px_y >= tin_y - margin

    def to_dict(self) -> Dict[str, List[float]]:
        d: Dict[str, Any] = {
            "front_left": list(self.front_left),
            "front_right": list(self.front_right),
            "back_right": list(self.back_right),
            "back_left": list(self.back_left),
        }
        if self.tin_left is not None:
            d["tin_left"] = list(self.tin_left)
        if self.tin_right is not None:
            d["tin_right"] = list(self.tin_right)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "CourtCalibration":
        return cls(
            front_left=tuple(d["front_left"]),
            front_right=tuple(d["front_right"]),
            back_right=tuple(d["back_right"]),
            back_left=tuple(d["back_left"]),
            tin_left=tuple(d["tin_left"]) if d.get("tin_left") else None,
            tin_right=tuple(d["tin_right"]) if d.get("tin_right") else None,
        )

    @classmethod
    def from_normalized(
        cls, d: Dict[str, List[float]], width: int, height: int
    ) -> "CourtCalibration":
        """Build pixel-space corners from normalized (0..1) image fractions."""
        def px(corner: List[float]) -> Tuple[float, float]:
            return (corner[0] * width, corner[1] * height)

        return cls(
            front_left=px(d["front_left"]),
            front_right=px(d["front_right"]),
            back_right=px(d["back_right"]),
            back_left=px(d["back_left"]),
            tin_left=px(d["tin_left"]) if d.get("tin_left") else None,
            tin_right=px(d["tin_right"]) if d.get("tin_right") else None,
        )

    def to_normalized(self, width: int, height: int) -> Dict[str, List[float]]:
        """Convert pixel coords to normalized (0..1) fractions."""
        def norm(pt: Tuple[float, float]) -> List[float]:
            return [pt[0] / width, pt[1] / height]
        d: Dict[str, Any] = {
            "front_left": norm(self.front_left),
            "front_right": norm(self.front_right),
            "back_right": norm(self.back_right),
            "back_left": norm(self.back_left),
        }
        if self.tin_left:
            d["tin_left"] = norm(self.tin_left)
        if self.tin_right:
            d["tin_right"] = norm(self.tin_right)
        return d


@dataclass
class CourtModel:
    """Maps image pixels <-> real court metres via a planar homography.

    Build one of these from a CourtCalibration, then call ``to_court`` on any
    image point (typically a player's foot position) to get metres.
    """

    calibration: CourtCalibration
    _H: np.ndarray = field(init=False, repr=False)        # image -> world
    _H_inv: np.ndarray = field(init=False, repr=False)    # world -> image

    def __post_init__(self):
        # Imported lazily so this module's constants/geometry remain importable
        # even before OpenCV is installed.
        import cv2

        img = self.calibration.image_points()
        world = self.calibration.world_points()
        H, _ = cv2.findHomography(img, world, method=0)
        if H is None:
            raise ValueError("Could not compute court homography from corners")
        self._H = H
        self._H_inv = np.linalg.inv(H)

    # ----- coordinate transforms -----
    @staticmethod
    def _apply(H: np.ndarray, pt: Tuple[float, float]) -> Tuple[float, float]:
        v = H @ np.array([pt[0], pt[1], 1.0])
        if abs(v[2]) < 1e-9:
            return (float("nan"), float("nan"))
        return float(v[0] / v[2]), float(v[1] / v[2])

    def to_court(self, image_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Image pixel -> court metres (x across, y front->back)."""
        return self._apply(self._H, image_xy)

    def to_image(self, court_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Court metres -> image pixel."""
        return self._apply(self._H_inv, court_xy)

    # ----- court-aware helpers used by analytics -----
    @staticmethod
    def in_bounds(court_xy: Tuple[float, float], margin: float = 0.5) -> bool:
        x, y = court_xy
        return (-margin <= x <= COURT_WIDTH + margin
                and -margin <= y <= COURT_LENGTH + margin)

    @staticmethod
    def depth_zone(court_xy: Tuple[float, float]) -> str:
        """front / mid / back third of the court by depth."""
        y = court_xy[1]
        if y < FRONT_THIRD_Y:
            return "front"
        if y < BACK_THIRD_Y:
            return "mid"
        return "back"

    @staticmethod
    def distance_to_t(court_xy: Tuple[float, float]) -> float:
        """Metres from the T. Low values = good central court control."""
        dx = court_xy[0] - T_POINT[0]
        dy = court_xy[1] - T_POINT[1]
        return float(np.hypot(dx, dy))
