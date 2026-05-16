from __future__ import annotations

import logging

from ultralytics import YOLO

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
    ) -> list[dict]:
        config = FrameExtractionConfig(every_n_frames=frame_interval)
        extractor = FrameExtractorService(config)

        frame_metas: list[tuple[int, float]] = []
        frames = []

        for frame_id, timestamp_ms, frame in extractor.iter_frames(video_path):
            frame_metas.append((frame_id, timestamp_ms))
            frames.append(frame)

        records: list[dict] = []

        if frames:
            results = self.model.predict(
                source=frames,
                conf=confidence_threshold,
                imgsz=640,
                verbose=False,
            )

            for (frame_id, timestamp_ms), result in zip(frame_metas, results):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    records.append({
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": round(float(box.conf[0]), 4),
                    })

        return records
