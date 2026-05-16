from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv

from backend.app.tracking.config import TrackerConfig
from backend.app.schemas.detection import Detection
from backend.app.schemas.track import FrameTrackState, Track
from backend.app.tracking.trackers.abstract import AbstractTracker

logger = logging.getLogger(__name__)


@dataclass
class TrackerStats:
    """Runtime counters emitted by tracker adapters."""

    frames_processed: int = 0
    tracks_created: int = 0
    matches: int = 0
    lost_tracks: int = 0
    removed_tracks: int = 0
    id_switches: int = 0
    total_update_time_sec: float = 0.0
    last_active_tracks: int = 0
    last_unmatched_detections: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats counters to a serializable dictionary."""

        average_fps = 0.0
        if self.total_update_time_sec > 0:
            average_fps = self.frames_processed / self.total_update_time_sec
        return {
            "frames_processed": self.frames_processed,
            "tracks_created": self.tracks_created,
            "matches": self.matches,
            "lost_tracks": self.lost_tracks,
            "removed_tracks": self.removed_tracks,
            "id_switches": self.id_switches,
            "average_fps": average_fps,
            "last_active_tracks": self.last_active_tracks,
            "last_unmatched_detections": self.last_unmatched_detections,
        }


class ByteTrackerAdapter(AbstractTracker):
    """Adapter around Roboflow trackers ByteTrack implementation."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        """Initialize Roboflow ByteTrack with project-level state mapping."""

        self.config = config or TrackerConfig()
        self._tracker_cls = self._import_roboflow_bytetrack()
        self._tracker = self._create_roboflow_tracker()
        self._tracks_by_id: dict[int, Track] = {}
        self._seen_track_ids: set[int] = set()
        self._class_name_to_id: dict[str, int] = {}
        self._class_id_to_name: dict[int, str] = {}
        self._last_active_track_ids: set[int] = set()
        self.stats = TrackerStats()

    def initialize(self) -> None:
        """Initialize tracker runtime state."""

        self.reset()

    def update(self, frame_id: int, detections: list[Detection]) -> list[Track]:
        """Update Roboflow ByteTrack with detections for one frame."""

        start_time = time.perf_counter()
        frame_detections = self._filter_detections(frame_id, detections)
        sv_detections = self._detections_to_supervision(frame_detections)
        tracked_detections = self._update_roboflow_tracker(sv_detections)
        tracks = self._supervision_detections_to_tracks(frame_id, tracked_detections, frame_detections)

        active_track_ids = {track.track_id for track in tracks}
        for track_id, track in self._tracks_by_id.items():
            if track_id not in active_track_ids and track_id in self._last_active_track_ids:
                track.missed_frames += 1

        self._last_active_track_ids = active_track_ids
        self.stats.frames_processed += 1
        self.stats.matches += len(tracks)
        self.stats.tracks_created = len(self._seen_track_ids)
        self.stats.lost_tracks = max(len(self._tracks_by_id) - len(active_track_ids), 0)
        self.stats.last_active_tracks = len(tracks)
        self.stats.last_unmatched_detections = max(len(frame_detections) - len(tracks), 0)
        self.stats.total_update_time_sec += time.perf_counter() - start_time

        logger.debug(
            "Frame %s Roboflow ByteTrack: detections=%d active=%d lost=%d",
            frame_id,
            len(frame_detections),
            self.stats.last_active_tracks,
            self.stats.lost_tracks,
        )
        return tracks

    def get_active_tracks(self) -> list[Track]:
        """Return currently active confirmed tracks."""

        return [track for track_id, track in self._tracks_by_id.items() if track_id in self._last_active_track_ids]

    def reset(self) -> None:
        """Reset Roboflow ByteTrack state and compatibility buffers."""

        self._tracker = self._create_roboflow_tracker()
        self._tracks_by_id.clear()
        self._seen_track_ids.clear()
        self._class_name_to_id.clear()
        self._class_id_to_name.clear()
        self._last_active_track_ids.clear()
        self.stats = TrackerStats()

    def get_tracker_stats(self) -> dict[str, Any]:
        """Return Roboflow ByteTrack adapter runtime statistics."""

        return self.stats.to_dict()

    def _create_roboflow_tracker(self) -> object:
        """Create a Roboflow ByteTrackTracker instance from TrackerConfig."""

        candidates = {
            "lost_track_buffer": self.config.track_buffer,
            "track_activation_threshold": self.config.low_thresh,
            "minimum_consecutive_frames": self.config.min_hits,
            "minimum_iou_threshold": self.config.match_thresh,
            "high_conf_det_threshold": self.config.high_thresh,
            "frame_rate": int(round(self.config.fps)),
        }
        signature = inspect.signature(self._tracker_cls)
        kwargs = {key: value for key, value in candidates.items() if key in signature.parameters}
        return self._tracker_cls(**kwargs)

    def _update_roboflow_tracker(self, detections: sv.Detections) -> sv.Detections:
        """Update Roboflow tracker while supporting minor API naming differences."""

        if hasattr(self._tracker, "update"):
            return self._tracker.update(detections)
        if hasattr(self._tracker, "update_with_detections"):
            return self._tracker.update_with_detections(detections)
        raise AttributeError("Roboflow ByteTrackTracker has no update method")

    def _filter_detections(self, frame_id: int, detections: list[Detection]) -> list[Detection]:
        """Filter detections by frame ID, confidence, and box area."""

        return [
            detection
            for detection in detections
            if detection.frame_id == frame_id
            and detection.confidence >= self.config.low_thresh
            and detection.area() >= self.config.min_box_area
        ]

    def _detections_to_supervision(self, detections: list[Detection]) -> sv.Detections:
        """Convert project detections to supervision Detections."""

        if not detections:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
                data={"source_index": np.empty((0,), dtype=int)},
            )

        xyxy = np.array([detection.bbox_xyxy for detection in detections], dtype=np.float32)
        confidence = np.array([detection.confidence for detection in detections], dtype=np.float32)
        class_id = np.array([self._class_id_for_name(detection.class_name) for detection in detections], dtype=int)
        source_index = np.arange(len(detections), dtype=int)
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data={"source_index": source_index},
        )

    def _supervision_detections_to_tracks(
        self,
        frame_id: int,
        detections: sv.Detections,
        source_detections: list[Detection],
    ) -> list[Track]:
        """Convert tracked supervision Detections to project Track objects."""

        if detections.tracker_id is None:
            return []

        tracks: list[Track] = []
        for index, track_id_value in enumerate(detections.tracker_id):
            if track_id_value is None or int(track_id_value) < 0:
                continue

            track_id = int(track_id_value)
            source_detection = self._source_detection_for_tracked_index(index, detections, source_detections)
            if source_detection is None:
                logger.debug("Skipping track_id=%s because it has no current source detection", track_id)
                continue

            bbox_xyxy = source_detection.bbox_xyxy
            confidence = source_detection.confidence
            class_id = int(detections.class_id[index]) if detections.class_id is not None else -1
            class_name = source_detection.class_name or self._class_id_to_name.get(class_id, "unknown")

            if track_id not in self._tracks_by_id:
                self._tracks_by_id[track_id] = Track(
                    track_id=track_id,
                    bbox_xyxy=bbox_xyxy,
                    confidence=confidence,
                    age=1,
                    hits=1,
                    missed_frames=0,
                    is_confirmed=True,
                    history=[
                        FrameTrackState(frame_id=frame_id, bbox_xyxy=bbox_xyxy, confidence=confidence),
                    ],
                    metadata={
                        "source_tracker": "roboflow/trackers",
                        "class_name": class_name,
                        "source_detection_id": source_detection.detection_id,
                    },
                )
                self._seen_track_ids.add(track_id)
            else:
                track = self._tracks_by_id[track_id]
                track.age += 1
                track.hits += 1
                track.missed_frames = 0
                track.is_confirmed = True
                track.metadata["source_tracker"] = "roboflow/trackers"
                track.metadata["class_name"] = class_name
                track.metadata["source_detection_id"] = source_detection.detection_id
                track.set_state(
                    frame_id=frame_id,
                    bbox_xyxy=bbox_xyxy,
                    confidence=confidence,
                    history_size=self.config.history_size,
                )

            tracks.append(self._tracks_by_id[track_id])

        return tracks

    @staticmethod
    def _source_detection_for_tracked_index(
        tracked_index: int,
        detections: sv.Detections,
        source_detections: list[Detection],
    ) -> Detection | None:
        """Return the source detection that produced a tracked detection."""

        if not source_detections:
            return None

        source_indexes = detections.data.get("source_index") if detections.data else None
        if source_indexes is not None and tracked_index < len(source_indexes):
            source_index = int(source_indexes[tracked_index])
            if 0 <= source_index < len(source_detections):
                return source_detections[source_index]

    def _class_id_for_name(self, class_name: str) -> int:
        """Return a stable integer class ID for a class name."""

        if class_name not in self._class_name_to_id:
            class_id = len(self._class_name_to_id)
            self._class_name_to_id[class_name] = class_id
            self._class_id_to_name[class_id] = class_name
        return self._class_name_to_id[class_name]

    @staticmethod
    def _import_roboflow_bytetrack() -> type:
        """Import Roboflow ByteTrackTracker from the trackers package."""

        try:
            from trackers import ByteTrackTracker
        except ImportError as exc:
            raise ImportError(
                "Roboflow trackers is required. Install it with "
                "`python -m pip install -r backend/requirements.txt`."
            ) from exc

        return ByteTrackTracker
