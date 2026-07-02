"""
Layer 2 — Ball 3D trajectory + physics.

A single 2D ball detection is only a *ray* in 3D — depth is unknown. But between
contacts the ball is in free flight, so it follows a ballistic arc:

    P(t) = P0 + V0·t + ½·g·t²          g = (0, 0, -9.81) m/s²

A sequence of 2D detections of that arc therefore constrains a unique 3D parabola:
we recover (P0, V0) by minimising the reprojection error of the 3D arc against the
observed 2D track, using the Layer-1 camera calibration (Court3D).

From the recovered 3D arc we can answer the questions monocular 2D never could:
  • where does the ball cross the FRONT WALL plane (y=0), and at what HEIGHT?
    → above the tin (good) or below (down); below the out-line or above (out)
  • where does it cross the FLOOR plane (z=0)?  → bounce locations, double-bounce
  • does it pass beyond an out-line?            → out

This module fits ONE free-flight segment. Segmenting a rally into segments at
contacts (racket / wall / floor) is done upstream from velocity discontinuities.

Honesty: depth from a monocular parabola is well-posed only when the arc shows
curvature in the image and the ball track is clean. Quality is bounded by ball
detection. Confidence is reported per fit (reprojection error) so the rest of the
system can weight it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .court import COURT_WIDTH, COURT_LENGTH
from .court3d import (Court3D, TIN_HEIGHT, FRONT_OUT_HEIGHT, BACK_OUT_HEIGHT,
                      SERVICE_LINE_HEIGHT)

G = np.array([0.0, 0.0, -9.81])


@dataclass
class Ball3DArc:
    P0: np.ndarray          # position at t=0 (first observation), metres
    V0: np.ndarray          # velocity at t=0, m/s
    t0: float               # time (s) of first observation (absolute)
    t_end: float            # time (s) of last observation
    reproj_err_px: float
    n_obs: int

    def position(self, t_abs: float) -> np.ndarray:
        dt = t_abs - self.t0
        return self.P0 + self.V0 * dt + 0.5 * G * dt * dt

    def velocity(self, t_abs: float) -> np.ndarray:
        return self.V0 + G * (t_abs - self.t0)

    # ── physics queries ────────────────────────────────────────────────────────
    def _crossing_time(self, axis: int, value: float) -> List[float]:
        """Times (absolute) where coordinate `axis` equals `value` (solve quadratic)."""
        # axis position: P0[a] + V0[a]*dt + 0.5*g[a]*dt^2 = value
        a = 0.5 * G[axis]
        b = self.V0[axis]
        c = self.P0[axis] - value
        out = []
        if abs(a) < 1e-9:
            if abs(b) > 1e-9:
                out.append(self.t0 + (-c / b))
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                r = np.sqrt(disc)
                out += [self.t0 + (-b + r) / (2 * a), self.t0 + (-b - r) / (2 * a)]
        return sorted(out)

    def front_wall_hit(self) -> Optional[dict]:
        """Where/if the arc reaches the front wall (y=0) within this segment's time.
        Returns height z + within-width flag + tin/out classification."""
        for t in self._crossing_time(1, 0.0):
            if self.t0 - 0.05 <= t <= self.t_end + 0.2:
                P = self.position(t)
                in_width = -0.1 <= P[0] <= COURT_WIDTH + 0.1
                z = float(P[2])
                cls = ("down_tin" if z < TIN_HEIGHT else
                       "out_top" if z > FRONT_OUT_HEIGHT else "good")
                return {"t": round(t, 3), "height_m": round(z, 2),
                        "x_m": round(float(P[0]), 2), "in_width": in_width,
                        "classification": cls}
        return None

    def floor_bounces(self) -> List[dict]:
        """Floor (z=0) crossings within the segment → bounce points (x,y)."""
        res = []
        for t in self._crossing_time(2, 0.0):
            if self.t0 - 0.02 <= t <= self.t_end + 0.05:
                P = self.position(t)
                inb = (-0.1 <= P[0] <= COURT_WIDTH + 0.1 and -0.1 <= P[1] <= COURT_LENGTH + 0.1)
                res.append({"t": round(t, 3), "x_m": round(float(P[0]), 2),
                            "y_m": round(float(P[1]), 2), "in_bounds": inb})
        return res


def fit_ballistic_segment(
    observations: List[Tuple[float, float, float]],  # (t_abs_s, u_px, v_px)
    court3d: Court3D,
    init_plane: str = "floor",
) -> Optional[Ball3DArc]:
    """Fit a 3D ballistic arc to 2D ball observations via reprojection minimisation.

    ``init_plane`` seeds the initial depth by back-projecting the observations onto
    a court plane ('floor' is a reasonable default mid-flight guess).
    """
    from scipy.optimize import least_squares

    if len(observations) < 4:
        return None
    obs = sorted(observations, key=lambda o: o[0])
    t0 = obs[0][0]
    ts = np.array([o[0] - t0 for o in obs])
    uv = np.array([(o[1], o[2]) for o in obs], dtype=float)

    # ── initial guess ──────────────────────────────────────────────────────────
    # Back-project first/last obs onto the seed plane to get rough 3D endpoints,
    # then derive P0 (at t=0) and a straight-line V0; gravity is added by the fit.
    from .court3d import PLANE_FLOOR
    plane = PLANE_FLOOR
    p_first = court3d.backproject_to_plane(uv[0, 0], uv[0, 1], plane)
    p_last = court3d.backproject_to_plane(uv[-1, 0], uv[-1, 1], plane)
    if p_first is None or p_last is None:
        # fall back to a point in front of the camera
        p_first = np.array([COURT_WIDTH / 2, COURT_LENGTH / 2, 1.0])
        p_last = np.array([COURT_WIDTH / 2, COURT_LENGTH / 4, 1.0])
    span = max(1e-2, ts[-1])
    V0_init = (p_last - p_first) / span
    # lift the seed off the floor a bit so it's a mid-air arc, not a ground slide
    P0_init = p_first + np.array([0, 0, 0.8])
    x0 = np.concatenate([P0_init, V0_init])

    def residuals(x):
        P0 = x[:3]; V0 = x[3:]
        res = []
        for i, dt in enumerate(ts):
            P = P0 + V0 * dt + 0.5 * G * dt * dt
            u, v = court3d.project(P)
            res += [u - uv[i, 0], v - uv[i, 1]]
        return res

    try:
        sol = least_squares(residuals, x0, method="lm", max_nfev=400)
    except Exception:
        return None

    r = np.array(residuals(sol.x)).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(r, axis=1)))
    return Ball3DArc(P0=sol.x[:3], V0=sol.x[3:], t0=t0, t_end=obs[-1][0],
                     reproj_err_px=round(err, 2), n_obs=len(obs))


# Plane lookup by name
def _plane(name: str):
    from .court3d import (PLANE_FLOOR, PLANE_FRONT_WALL, PLANE_BACK_WALL,
                          PLANE_LEFT_WALL, PLANE_RIGHT_WALL)
    return {
        "floor": PLANE_FLOOR, "front_wall": PLANE_FRONT_WALL,
        "back_wall": PLANE_BACK_WALL, "left_wall": PLANE_LEFT_WALL,
        "right_wall": PLANE_RIGHT_WALL,
    }.get(name)


def fit_anchored_segment(
    observations: List[Tuple[float, float, float]],  # (t_abs_s, u, v)
    court3d: Court3D,
    start_surface: str,     # surface the segment STARTS on (a contact)
    end_surface: str,       # surface the segment ENDS on (a contact)
) -> Optional[Tuple["Ball3DArc", dict]]:
    """Reconstruct a 3D arc whose endpoints lie on known court surfaces.

    Because a contact on a surface fixes the ball's depth (back-project the pixel
    onto that plane), anchoring both ends makes the arc fully determined by
    gravity — no ill-posed depth search. We then report the reprojection error of
    the INTERMEDIATE observations as a confidence/consistency check (a wrong
    surface assumption yields a large error).

    This is the robust monocular reconstruction: it works whenever the contacts
    are detectable and their surfaces identifiable.
    """
    if len(observations) < 3:
        return None
    obs = sorted(observations, key=lambda o: o[0])
    t0, te = obs[0][0], obs[-1][0]
    T = te - t0
    if T <= 1e-3:
        return None
    pA = _plane(start_surface); pB = _plane(end_surface)
    if pA is None or pB is None:
        return None

    P_start = court3d.backproject_to_plane(obs[0][1], obs[0][2], pA)
    P_end = court3d.backproject_to_plane(obs[-1][1], obs[-1][2], pB)
    if P_start is None or P_end is None:
        return None

    # Ballistic arc through P_start@t0 and P_end@te:  V0 = (ΔP − ½gT²)/T
    V0 = (P_end - P_start - 0.5 * G * T * T) / T
    arc = Ball3DArc(P0=np.asarray(P_start, float), V0=V0, t0=t0, t_end=te,
                    reproj_err_px=0.0, n_obs=len(obs))

    # Consistency: reproject the intermediate observations
    errs = []
    for (ta, u, v) in obs[1:-1]:
        pu, pv = court3d.project(arc.position(ta))
        errs.append(float(np.hypot(pu - u, pv - v)))
    arc.reproj_err_px = round(float(np.mean(errs)) if errs else 0.0, 2)
    info = {
        "start_surface": start_surface, "end_surface": end_surface,
        "P_start": np.round(P_start, 2).tolist(), "P_end": np.round(P_end, 2).tolist(),
        "consistency_px": arc.reproj_err_px,
    }
    return arc, info
