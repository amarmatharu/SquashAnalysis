"""
Layer 1 — Court 3D.

Upgrades the 2D floor homography (court.py) to a full 3D camera calibration.

A standard singles court has fixed real dimensions, so a handful of identifiable
points at known 3D positions (the four floor corners at z=0 and the two tin-line
ends at z=TIN_HEIGHT on the front wall) are enough to solve the camera pose
(extrinsics) and refine the focal length (intrinsics) via PnP. From that we get a
single projection that maps ANY 3D court point to the image and back-projects any
image pixel known to lie on a court plane (floor or a wall) to 3D.

This is the coordinate system every higher layer needs:
  • project the tin / out-lines / wall planes into the image (visualisation, and
    the "is the ball above/below the tin" question once we have the ball in 3D)
  • back-project player feet (floor plane) → court metres (principled 3D version
    of the old homography)
  • back-project a ball known to be at the front wall → its height on that wall

Coordinate frame (metres, z up):
  x: 0..COURT_WIDTH    across court (left→right, looking from behind)
  y: 0..COURT_LENGTH   front wall (y=0) → back wall (y=COURT_LENGTH)
  z: 0 at floor, up the walls
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .court import COURT_WIDTH, COURT_LENGTH

# ── Heights (metres), WSF singles ────────────────────────────────────────────
TIN_HEIGHT = 0.48           # top of the tin (board) on the front wall
SERVICE_LINE_HEIGHT = 1.78  # front-wall service line
FRONT_OUT_HEIGHT = 4.57     # front-wall out-line
BACK_OUT_HEIGHT = 2.13      # back-wall out-line
# Side-wall out-line runs diagonally from FRONT_OUT_HEIGHT (front) to
# BACK_OUT_HEIGHT (back).

# Court planes as (normal, offset) with plane = {X : n·X = d}
PLANE_FLOOR = (np.array([0.0, 0.0, 1.0]), 0.0)        # z = 0
PLANE_FRONT_WALL = (np.array([0.0, 1.0, 0.0]), 0.0)   # y = 0
PLANE_BACK_WALL = (np.array([0.0, 1.0, 0.0]), COURT_LENGTH)  # y = L
PLANE_LEFT_WALL = (np.array([1.0, 0.0, 0.0]), 0.0)    # x = 0
PLANE_RIGHT_WALL = (np.array([1.0, 0.0, 0.0]), COURT_WIDTH)  # x = W


@dataclass
class Court3D:
    """Full 3D camera calibration for a court view."""
    K: np.ndarray                 # 3x3 intrinsics
    rvec: np.ndarray              # Rodrigues rotation (world→camera)
    tvec: np.ndarray              # translation (world→camera)
    image_w: int
    image_h: int
    reproj_err_px: float = 0.0    # mean reprojection error on the calibration pts

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_calibration(cls, calib, image_w: int, image_h: int) -> Optional["Court3D"]:
        """Solve camera pose + focal length from a CourtCalibration (pixel coords).

        Requires the four floor corners; uses the tin endpoints too when present
        (they break the planar degeneracy and let us estimate focal length).
        """
        import cv2

        obj_pts: List[Tuple[float, float, float]] = [
            (0.0, 0.0, 0.0),                  # front_left  floor
            (COURT_WIDTH, 0.0, 0.0),          # front_right floor
            (COURT_WIDTH, COURT_LENGTH, 0.0), # back_right  floor
            (0.0, COURT_LENGTH, 0.0),         # back_left   floor
        ]
        img_pts: List[Tuple[float, float]] = [
            tuple(calib.front_left), tuple(calib.front_right),
            tuple(calib.back_right), tuple(calib.back_left),
        ]
        has_tin = calib.tin_left is not None and calib.tin_right is not None
        if has_tin:
            obj_pts += [(0.0, 0.0, TIN_HEIGHT), (COURT_WIDTH, 0.0, TIN_HEIGHT)]
            img_pts += [tuple(calib.tin_left), tuple(calib.tin_right)]

        obj = np.array(obj_pts, dtype=np.float64)
        img = np.array(img_pts, dtype=np.float64)
        cx, cy = image_w / 2.0, image_h / 2.0

        def _solve_for_f(f: float):
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            flag = cv2.SOLVEPNP_ITERATIVE
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
            if not ok:
                return None
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
            return K, rvec, tvec, err

        # Search focal length (phone wide-ish lens → ~0.6..2.0 × image width).
        best = None
        if has_tin:
            for f in np.linspace(0.5 * image_w, 2.2 * image_w, 60):
                r = _solve_for_f(float(f))
                if r and (best is None or r[3] < best[3]):
                    best = r
        else:
            # Coplanar only → focal length unobservable; assume a sane default.
            best = _solve_for_f(1.1 * image_w)

        if best is None:
            return None
        K, rvec, tvec, err = best
        return cls(K=K, rvec=rvec, tvec=tvec, image_w=image_w, image_h=image_h,
                   reproj_err_px=round(err, 2))

    # ── derived matrices ───────────────────────────────────────────────────────
    @property
    def R(self) -> np.ndarray:
        import cv2
        R, _ = cv2.Rodrigues(self.rvec)
        return R

    @property
    def cam_center(self) -> np.ndarray:
        """Camera centre in world coords: C = -R^T t."""
        return (-self.R.T @ self.tvec).reshape(3)

    # ── projection / back-projection ───────────────────────────────────────────
    def project(self, world_xyz) -> Tuple[float, float]:
        """3D court point (metres) → image pixel."""
        X = np.asarray(world_xyz, dtype=np.float64).reshape(3)
        cam = self.R @ X + self.tvec.reshape(3)
        if cam[2] <= 1e-6:
            return (float("nan"), float("nan"))
        u = self.K[0, 0] * cam[0] / cam[2] + self.K[0, 2]
        v = self.K[1, 1] * cam[1] / cam[2] + self.K[1, 2]
        return float(u), float(v)

    def project_line(self, a, b, n: int = 20) -> List[Tuple[float, float]]:
        a = np.asarray(a, float); b = np.asarray(b, float)
        return [self.project(a + (b - a) * t) for t in np.linspace(0, 1, n)]

    def backproject_to_plane(self, u: float, v: float, plane) -> Optional[np.ndarray]:
        """Image pixel → 3D point on the given plane (normal, d). The pixel is
        assumed to lie on that plane (e.g. feet on the floor, ball on a wall)."""
        n, d = plane
        n = np.asarray(n, float)
        # Ray in world: C + s * dir, dir = R^T K^-1 [u,v,1]
        Kinv = np.linalg.inv(self.K)
        ray_cam = Kinv @ np.array([u, v, 1.0])
        dir_world = self.R.T @ ray_cam
        C = self.cam_center
        denom = float(n @ dir_world)
        if abs(denom) < 1e-9:
            return None
        s = (d - float(n @ C)) / denom
        if s <= 0:
            return None
        return C + s * dir_world

    def ground_to_court(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """Image pixel on the floor → (x, y) court metres."""
        P = self.backproject_to_plane(u, v, PLANE_FLOOR)
        if P is None:
            return None
        return float(P[0]), float(P[1])

    def frontwall_height_at(self, u: float, v: float) -> Optional[float]:
        """For a pixel on the front-wall plane (y=0), return its height z."""
        P = self.backproject_to_plane(u, v, PLANE_FRONT_WALL)
        if P is None:
            return None
        return float(P[2])

    def calibration_quality(self) -> str:
        """Coarse quality label from reprojection error (px)."""
        e = self.reproj_err_px
        if e <= 6:
            return "excellent"
        if e <= 15:
            return "good"
        if e <= 40:
            return "rough"
        return "bad"

    # ── named court geometry as 3D segments (for drawing / reasoning) ───────────
    def court_lines_3d(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        W, L = COURT_WIDTH, COURT_LENGTH
        return {
            "tin": (np.array([0, 0, TIN_HEIGHT]), np.array([W, 0, TIN_HEIGHT])),
            "front_out": (np.array([0, 0, FRONT_OUT_HEIGHT]),
                          np.array([W, 0, FRONT_OUT_HEIGHT])),
            "service_line": (np.array([0, 0, SERVICE_LINE_HEIGHT]),
                             np.array([W, 0, SERVICE_LINE_HEIGHT])),
            "left_out": (np.array([0, 0, FRONT_OUT_HEIGHT]),
                         np.array([0, L, BACK_OUT_HEIGHT])),
            "right_out": (np.array([W, 0, FRONT_OUT_HEIGHT]),
                          np.array([W, L, BACK_OUT_HEIGHT])),
            "back_out": (np.array([0, L, BACK_OUT_HEIGHT]),
                         np.array([W, L, BACK_OUT_HEIGHT])),
            "floor_front": (np.array([0, 0, 0]), np.array([W, 0, 0])),
            "floor_back": (np.array([0, L, 0]), np.array([W, L, 0])),
            "floor_left": (np.array([0, 0, 0]), np.array([0, L, 0])),
            "floor_right": (np.array([W, 0, 0]), np.array([W, L, 0])),
            "short_line": (np.array([0, COURT_LENGTH - 4.26, 0]),
                           np.array([W, COURT_LENGTH - 4.26, 0])),
        }


# ── overlay rendering (calibration self-check) ────────────────────────────────

_LINE_STYLE = {
    "tin": ((0, 0, 255), 3),            # red, thick
    "front_out": ((0, 200, 255), 2),    # amber
    "left_out": ((0, 200, 255), 2),
    "right_out": ((0, 200, 255), 2),
    "back_out": ((0, 200, 255), 2),
    "service_line": ((255, 200, 0), 1),
    "short_line": ((200, 200, 200), 1),
    "floor_front": ((0, 255, 0), 2),    # green floor outline
    "floor_back": ((0, 255, 0), 2),
    "floor_left": ((0, 255, 0), 2),
    "floor_right": ((0, 255, 0), 2),
}


def draw_court_overlay(frame_bgr, court3d: "Court3D"):
    """Draw the projected 3D court lines on a frame (for visual calibration check).
    If the lines land on the real court lines, the calibration is correct."""
    import cv2
    vis = frame_bgr.copy()
    H, W = vis.shape[:2]

    def _clip(pts):
        return [(int(round(u)), int(round(v))) for (u, v) in pts
                if np.isfinite(u) and np.isfinite(v) and -5000 < u < 5000 and -5000 < v < 5000]

    for name, (a, b) in court3d.court_lines_3d().items():
        color, thick = _LINE_STYLE.get(name, ((255, 255, 255), 1))
        pts = _clip(court3d.project_line(a, b, n=24))
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i - 1], pts[i], color, thick, cv2.LINE_AA)

    # Mark the T
    tu, tv = court3d.project((COURT_WIDTH / 2, COURT_LENGTH - 4.26, 0))
    if np.isfinite(tu):
        cv2.circle(vis, (int(tu), int(tv)), 5, (255, 255, 0), -1)

    # Quality banner
    q = court3d.calibration_quality()
    txt = f"calibration: {q}  (reproj {court3d.reproj_err_px}px)"
    color = (0, 220, 0) if q in ("excellent", "good") else (0, 165, 255) if q == "rough" else (0, 0, 255)
    cv2.rectangle(vis, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(vis, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis
