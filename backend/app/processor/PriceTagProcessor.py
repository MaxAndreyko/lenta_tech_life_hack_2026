from __future__ import annotations

import logging

from ultralytics import YOLO

from backend.app.schemas.detection import Detection
from backend.app.services.frame_extractor import FrameExtractionConfig, FrameExtractorService

logger = logging.getLogger(__name__)


class PriceTagProcessor:
    def __init__(self, detector_model: str):
        self.model = YOLO(detector_model)

    def process_video(
        self,
        video_path: str,
        frame_interval: int = 1,
        confidence_threshold: float = 0.4,
    ) -> list[Detection]:
        config = FrameExtractionConfig(every_n_frames=frame_interval)
        extractor = FrameExtractorService(config)

        frame_metas: list[tuple[int, float]] = []
        frames = []

        for frame_id, timestamp_ms, frame in extractor.iter_frames(video_path):
            frame_metas.append((frame_id, timestamp_ms))
            frames.append(frame)

        detections: list[Detection] = []

        if frames:
            results = self.model.predict(
                source=frames,
                conf=confidence_threshold,
                imgsz=640,
                verbose=False,
            )

            for (frame_id, _), result in zip(frame_metas, results):
                for box_idx, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    detections.append(Detection(
                        detection_id=f"{frame_id}_{box_idx}",
                        frame_id=frame_id,
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=round(float(box.conf[0]), 4),
                        class_name=result.names[int(box.cls[0])],
                    ))

        return detections
