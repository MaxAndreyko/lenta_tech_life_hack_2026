from __future__ import annotations

import logging
from copy import deepcopy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

from backend.app.tracking.config import TrackerConfig
from backend.app.tracking.factory import TrackerFactory
from backend.app.schemas.detection import Detection
from backend.app.schemas.track import Track
from backend.app.tracking.trackers.abstract import AbstractTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingServiceConfig:
    """Configuration for assigning stable track IDs to frame detections."""

    tracker_type: str = "bytetrack"
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    frame_start_id: int = 1
    normalize_frame_ids: bool = True
    show_progress: bool = True

    def __post_init__(self) -> None:
        """Validate tracking service configuration values."""

        if self.frame_start_id < 0:
            raise ValueError("frame_start_id must be non-negative")
        if not self.tracker_type:
            raise ValueError("tracker_type must not be empty")

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> TrackingServiceConfig:
        """Load tracking service configuration from a YAML file."""

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Tracking config does not exist: {config_path}")

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("tracking YAML config must contain a mapping")

        tracker_data = data.get("tracker", {})
        if tracker_data is None:
            tracker_data = {}
        if not isinstance(tracker_data, dict):
            raise ValueError("tracking YAML field `tracker` must contain a mapping")

        return cls(
            tracker_type=str(data.get("tracker_type", "bytetrack")),
            tracker=TrackerConfig(**tracker_data),
            frame_start_id=int(data.get("frame_start_id", 1)),
            normalize_frame_ids=bool(data.get("normalize_frame_ids", True)),
            show_progress=bool(data.get("show_progress", True)),
        )


class TrackingService:
    """Assign stable track IDs to detector-agnostic frame detections."""

    def __init__(self, config: TrackingServiceConfig | None = None, tracker: AbstractTracker | None = None) -> None:
        """Initialize tracking service with configurable tracker adapter."""

        self.config = config or TrackingServiceConfig()
        self.tracker = tracker or TrackerFactory.create(self.config.tracker_type, self.config.tracker)

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> TrackingService:
        """Create a tracking service from a YAML configuration file."""

        return cls(config=TrackingServiceConfig.from_yaml(config_path))

    def track(self, detections_by_frame: Sequence[Sequence[Detection]]) -> list[list[Track]]:
        """Assign track IDs to a finite list of frame detections."""

        return list(self.iter_tracks(detections_by_frame, total_frames=len(detections_by_frame)))

    def iter_tracks(
        self,
        detections_by_frame: Iterable[Sequence[Detection]],
        total_frames: int | None = None,
    ) -> Iterable[list[Track]]:
        """Yield per-frame tracked objects."""

        self.tracker.initialize()
        iterator = enumerate(detections_by_frame, start=self.config.frame_start_id)
        if self.config.show_progress:
            iterator = tqdm(iterator, total=total_frames, desc="Tracking detections", unit="frame")

        for frame_id, frame_detections in iterator:
            normalized_detections = self._normalize_frame_detections(frame_id, frame_detections)
            tracks = self.tracker.update(frame_id, normalized_detections)
            yield [deepcopy(track) for track in tracks]

        logger.info("Tracking finished: stats=%s", self.tracker.get_tracker_stats())

    def get_tracker_stats(self) -> dict[str, Any]:
        """Return runtime statistics from the underlying tracker."""

        return self.tracker.get_tracker_stats()

    def reset(self) -> None:
        """Reset underlying tracker state."""

        self.tracker.reset()

    def _normalize_frame_detections(self, frame_id: int, detections: Sequence[Detection]) -> list[Detection]:
        """Return detections with frame IDs aligned to the current frame."""

        if not self.config.normalize_frame_ids:
            return list(detections)

        return [
            detection if detection.frame_id == frame_id else replace(detection, frame_id=frame_id)
            for detection in detections
        ]
