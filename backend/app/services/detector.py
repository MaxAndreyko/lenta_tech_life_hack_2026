from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from backend.app.schemas.detection import Detection
from backend.app.processor import PriceTagProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for price tag bounding-box detector."""

    detector_model: str
    frame_interval: int = 1
    confidence_threshold: float = 0.4

    def __post_init__(self) -> None:
        """Validate bounding-box detector configuration values."""

        if not self.detector_model:
            raise ValueError("detector_model must not be empty")
        if self.frame_interval <= 0:
            raise ValueError("frame_interval must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")


class DetectorService:
    """Predict price tag bounding boxes and expose them grouped by frame."""

    def __init__(
        self,
        config: DetectorConfig,
        processor: PriceTagProcessor,
    ) -> None:
        """Initialize prediction service with PriceTagProcessor-compatible backend."""

        self.config = config
        self.processor = processor

    def predict_video(self, video_path: Path | str) -> list[list[Detection]]:
        """Predict bounding boxes grouped by frame."""

        video_path = Path(video_path)
        detections_by_frame = list(self.processor.process_video(
            video_path=str(video_path),
            frame_interval=self.config.frame_interval,
            confidence_threshold=self.config.confidence_threshold,
        ))
        logger.info(
            "Predicted bounding boxes: video=%s frames=%d detections=%d",
            video_path,
            len(detections_by_frame),
            sum(len(detections) for detections in detections_by_frame),
        )
        return detections_by_frame

    def iter_frame_detections(self, video_path: Path | str) -> Iterator[list[Detection]]:
        """Yield predicted frame detections for tracking services."""

        video_path = Path(video_path)
        yield from self.processor.process_video(
            video_path=str(video_path),
            frame_interval=self.config.frame_interval,
            confidence_threshold=self.config.confidence_threshold,
        )
