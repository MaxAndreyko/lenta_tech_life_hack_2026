from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration shared by tracker adapters."""

    high_thresh: float = 0.6
    low_thresh: float = 0.1
    match_thresh: float = 0.3
    track_buffer: int = 30
    min_box_area: float = 10.0
    fps: float = 30.0
    history_size: int = 64
    min_hits: int = 1

    def __post_init__(self) -> None:
        """Validate tracker configuration thresholds."""

        if not 0.0 <= self.low_thresh <= self.high_thresh <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= low_thresh <= high_thresh <= 1")
        if not 0.0 <= self.match_thresh <= 1.0:
            raise ValueError("match_thresh must be between 0 and 1")
        if self.track_buffer < 0:
            raise ValueError("track_buffer must be non-negative")
        if self.min_box_area < 0:
            raise ValueError("min_box_area must be non-negative")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")
        if self.min_hits <= 0:
            raise ValueError("min_hits must be positive")
