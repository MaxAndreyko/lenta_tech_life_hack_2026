from __future__ import annotations

import logging
from collections.abc import Iterator

from ultralytics import YOLO

from backend.app.schemas.detection import Detection
from backend.app.services.frame_extractor import FrameExtractionConfig, FrameExtractorService

logger = logging.getLogger(__name__)


class PriceTagProcessor:
    def __init__(self, detector_model: str, batch_size: int = 8):
        self.model = YOLO(detector_model)
        self.batch_size = batch_size

    def process_video(
        self,
        video_path: str,
        frame_interval: int = 1,
        confidence_threshold: float = 0.4,
    ) -> Iterator[list[Detection]]:
        """Yield per-frame detection batches predicted from a video."""

        config = FrameExtractionConfig(every_n_frames=frame_interval)
        extractor = FrameExtractorService(config)

        batch_metas: list[tuple[int, float]] = []
        batch_frames = []

        for frame_id, timestamp_ms, frame in extractor.iter_frames(video_path):
            batch_metas.append((frame_id, timestamp_ms))
            batch_frames.append(frame)

            if len(batch_frames) == self.batch_size:
                yield from self._predict_batch(batch_frames, batch_metas, confidence_threshold)
                batch_frames.clear()
                batch_metas.clear()

        if batch_frames:
            yield from self._predict_batch(batch_frames, batch_metas, confidence_threshold)

    def _predict_batch(
        self,
        frames: list,
        metas: list[tuple[int, float]],
        confidence_threshold: float,
    ) -> list[list[Detection]]:
        """Predict detections for a frame batch."""

        results = self.model.predict(
            source=frames,
            conf=confidence_threshold,
            imgsz=640,
            batch=len(frames),
            verbose=False,
        )

        detections_by_frame: list[list[Detection]] = []
        for (frame_id, _), result in zip(metas, results):
            frame_detections: list[Detection] = []
            for box_idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                frame_detections.append(
                    Detection(
                        detection_id=f"{frame_id}_{box_idx}",
                        frame_id=frame_id,
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=round(float(box.conf[0]), 4),
                        class_name=result.names[int(box.cls[0])],
                    )
                )
            detections_by_frame.append(frame_detections)

        return detections_by_frame
