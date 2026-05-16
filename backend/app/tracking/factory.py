from __future__ import annotations

from backend.app.tracking.config import TrackerConfig
from backend.app.tracking.trackers.abstract import AbstractTracker
from backend.app.tracking.trackers.bytetrack import ByteTrackerAdapter


class TrackerFactory:
    """Factory for creating tracker adapters by tracker type."""

    _TRACKERS = {
        "bytetrack": ByteTrackerAdapter,
    }

    @classmethod
    def create(cls, tracker_type: str = "bytetrack", config: TrackerConfig | None = None) -> AbstractTracker:
        """Create a tracker adapter by name."""

        normalized_type = tracker_type.lower()
        if normalized_type not in cls._TRACKERS:
            available = ", ".join(sorted(cls._TRACKERS))
            raise ValueError(f"Unknown tracker_type={tracker_type!r}. Available trackers: {available}")
        return cls._TRACKERS[normalized_type](config=config)

    @classmethod
    def register(cls, tracker_type: str, tracker_cls: type[AbstractTracker]) -> None:
        """Register a new tracker adapter class."""

        cls._TRACKERS[tracker_type.lower()] = tracker_cls

    @classmethod
    def available_trackers(cls) -> list[str]:
        """Return names of registered tracker adapters."""

        return sorted(cls._TRACKERS)
