"""Dataclasses and DTO schemas."""

from backend.app.schemas.detection import BBoxXYXY, Detection
from backend.app.schemas.frame import FrameMetadata
from backend.app.schemas.track import FrameTrackState, Track

__all__ = [
    "BBoxXYXY",
    "Detection",
    "FrameMetadata",
    "FrameTrackState",
    "Track"
]
