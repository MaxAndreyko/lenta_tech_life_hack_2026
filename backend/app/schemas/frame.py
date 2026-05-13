from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameMetadata:
    """Metadata emitted for every extracted video frame."""

    frame_id: int
    timestamp_ms: float
    blur_score: float
    frame_path: Path
    width: int
    height: int

    @property
    def sharpness_score(self) -> float:
        """Alias for downstream modules that reason about sharpness."""

        return self.blur_score

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a CSV-friendly dictionary."""

        data = asdict(self)
        data["frame_path"] = str(self.frame_path)
        data["sharpness_score"] = self.sharpness_score
        return data
