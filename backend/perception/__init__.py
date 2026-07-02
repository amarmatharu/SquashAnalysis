"""
Perception spine for SquashSense AI.

This package turns raw video into *structured, real-world* data:

    video frames
        -> player detection + tracking      (tracking.py)
        -> court homography (pixels->meters) (court.py)
        -> structured rally/movement timeline (pipeline.py)

Everything downstream (strength/weakness analysis, "how to beat player X"
strategy) consumes the structured output of this package rather than guessing
from isolated still frames.
"""

from .court import CourtModel, CourtCalibration  # noqa: F401
