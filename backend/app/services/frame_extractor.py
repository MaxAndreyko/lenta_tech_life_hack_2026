from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from backend.app.core.settings import settings
from backend.app.schemas.frame import FrameMetadata
from backend.app.utils.image_quality import calculate_blur_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameExtractionConfig:
    """Configuration for extracting frames from a single video."""

    every_n_frames: int | None = 1
    target_fps: float | None = None
    resize_width: int | None = None
    resize_height: int | None = None
    jpeg_quality: int = 95
    output_root: Path = settings.storage.frames_dir
    overwrite: bool = True

    def __post_init__(self) -> None:
        """Validate frame extraction configuration values."""

        if self.every_n_frames is not None and self.every_n_frames <= 0:
            raise ValueError("every_n_frames must be positive when provided")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("target_fps must be positive when provided")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if self.resize_width is not None and self.resize_width <= 0:
            raise ValueError("resize_width must be positive when provided")
        if self.resize_height is not None and self.resize_height <= 0:
            raise ValueError("resize_height must be positive when provided")
        if self.every_n_frames is None and self.target_fps is None:
            raise ValueError("either every_n_frames or target_fps must be provided")


class FrameExtractorService:
    """Extract frames and frame metadata from local video files."""

    def __init__(self, config: FrameExtractionConfig | None = None) -> None:
        """Initialize the extractor with optional custom configuration."""

        self.config = config or FrameExtractionConfig()

    def extract_to_disk(self, video_path: Path | str) -> list[FrameMetadata]:
        """Extract selected frames to data/frames/{video_name}/."""

        video_path = Path(video_path)
        output_dir = self._prepare_output_dir(video_path)
        metadata: list[FrameMetadata] = []

        logger.info("Starting frame extraction: video=%s output=%s", video_path, output_dir)
        for frame_id, timestamp_ms, frame in self.iter_frames(video_path):
            resized_frame = self._resize_frame(frame)
            frame_path = output_dir / f"frame_{frame_id:06d}.jpg"
            self._write_jpeg(frame_path, resized_frame)

            height, width = resized_frame.shape[:2]
            item = FrameMetadata(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                blur_score=calculate_blur_score(resized_frame),
                frame_path=frame_path,
                width=width,
                height=height,
            )
            metadata.append(item)

        self._write_metadata_csv(output_dir, metadata)
        logger.info("Finished frame extraction: video=%s frames=%d", video_path, len(metadata))
        return metadata

    def iter_frames(self, video_path: Path | str) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield selected raw frames as (frame_id, timestamp_ms, frame)."""

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = self._resolve_frame_stride(source_fps)

        try:
            with tqdm(total=total_frames, desc=f"Extracting {video_path.name}", unit="frame") as progress:
                frame_id = 0
                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    if frame_id % stride == 0:
                        timestamp_ms = self._resolve_timestamp_ms(capture, frame_id, source_fps)
                        yield frame_id, timestamp_ms, frame

                    frame_id += 1
                    progress.update(1)
        finally:
            capture.release()

    def _prepare_output_dir(self, video_path: Path) -> Path:
        """Create and optionally clean the output directory for a video."""

        output_dir = self.config.output_root / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.overwrite:
            for frame_file in output_dir.glob("frame_*.jpg"):
                frame_file.unlink()
            metadata_file = output_dir / "metadata.csv"
            if metadata_file.exists():
                metadata_file.unlink()

        return output_dir

    def _resolve_frame_stride(self, source_fps: float) -> int:
        """Resolve the integer frame stride from FPS or explicit sampling config."""

        if self.config.target_fps is not None:
            if source_fps <= 0:
                logger.warning(
                    "Video FPS is unavailable; falling back to every_n_frames=%s",
                    self.config.every_n_frames,
                )
                return self.config.every_n_frames or 1
            return max(round(source_fps / self.config.target_fps), 1)

        return self.config.every_n_frames or 1

    @staticmethod
    def _resolve_timestamp_ms(capture: cv2.VideoCapture, frame_id: int, source_fps: float) -> float:
        """Resolve a frame timestamp in milliseconds."""

        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if timestamp_ms > 0:
            return timestamp_ms
        if source_fps > 0:
            return frame_id / source_fps * 1000.0
        return 0.0

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame according to configured dimensions."""

        width = self.config.resize_width
        height = self.config.resize_height

        if width is None and height is None:
            return frame

        original_height, original_width = frame.shape[:2]
        if width is None:
            scale = height / original_height
            width = round(original_width * scale)
        elif height is None:
            scale = width / original_width
            height = round(original_height * scale)

        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _write_jpeg(self, frame_path: Path, frame: np.ndarray) -> None:
        """Write a frame as JPEG to a Unicode-safe filesystem path."""

        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        success, encoded_frame = cv2.imencode(".jpg", frame, params)
        if not success:
            raise IOError(f"Failed to encode frame as JPEG: {frame_path}")

        frame_path.write_bytes(encoded_frame.tobytes())

    @staticmethod
    def _write_metadata_csv(output_dir: Path, metadata: list[FrameMetadata]) -> None:
        """Write extracted frame metadata to metadata.csv."""

        metadata_path = output_dir / "metadata.csv"
        dataframe = pd.DataFrame([item.to_dict() for item in metadata])
        dataframe.to_csv(metadata_path, index=False)
