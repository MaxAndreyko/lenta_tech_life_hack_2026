from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from backend.app.schemas.detection import BBoxXYXY


@dataclass(frozen=True)
class FrameTrackState:
    """Single-frame state stored inside a track history buffer."""

    frame_id: int
    bbox_xyxy: BBoxXYXY
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert frame track state to a serializable dictionary."""

        return asdict(self)


@dataclass
class Track:
    """Tracked object state with stable identity and history."""

    track_id: int
    bbox_xyxy: BBoxXYXY
    confidence: float
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    is_confirmed: bool = False
    history: list[FrameTrackState] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_state(self, frame_id: int, bbox_xyxy: BBoxXYXY, confidence: float, history_size: int) -> None:
        """Set a frame state and keep the history buffer bounded."""
        self.bbox_xyxy = bbox_xyxy
        self.confidence = confidence
        self.history.append(FrameTrackState(frame_id=frame_id, bbox_xyxy=bbox_xyxy, confidence=confidence))
        if len(self.history) > history_size:
            self.history = self.history[-history_size:]

    def to_dict(self) -> dict[str, Any]:
        """Convert track state to a serializable dictionary."""

        data = asdict(self)
        data["history"] = [state.to_dict() for state in self.history]
        return data
