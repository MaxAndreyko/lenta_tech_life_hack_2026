from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from backend.app.schemas.detection import BBoxXYXY
from backend.app.schemas.ocr import BBoxQuad


@dataclass(frozen=True)
class QRDetection:
    """A single QR detected on an image, optionally decoded."""

    bbox_xyxy: BBoxXYXY
    confidence: float
    payload: str = ""
    decoder: str = ""
    quad_xy: BBoxQuad | None = None

    @property
    def is_decoded(self) -> bool:
        return bool(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
