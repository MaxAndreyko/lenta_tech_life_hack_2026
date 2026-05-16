from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """Detection object."""

    detection_id: str
    frame_id: int
    bbox_xyxy: BBoxXYXY
    confidence: float
    class_name: str

    def area(self) -> float:
        """Return detection bounding-box area."""

        x1, y1, x2, y2 = self.bbox_xyxy
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert detection to a serializable dictionary."""

        data = asdict(self)
        return data
