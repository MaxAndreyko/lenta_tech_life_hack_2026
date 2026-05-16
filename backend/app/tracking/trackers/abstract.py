from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.schemas.detection import Detection
from backend.app.schemas.track import Track


class AbstractTracker(ABC):
    """Abstract base class for detector-agnostic MOT trackers."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize tracker runtime state."""

    @abstractmethod
    def update(self, frame_id: int, detections: list[Detection]) -> list[Track]:
        """Update tracker state with detections for a single frame."""

    @abstractmethod
    def get_active_tracks(self) -> list[Track]:
        """Return currently active confirmed tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state to an empty session."""

    @abstractmethod
    def get_tracker_stats(self) -> dict[str, Any]:
        """Return tracker runtime statistics."""
