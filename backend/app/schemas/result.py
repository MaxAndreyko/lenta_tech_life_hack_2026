from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from backend.app.schemas.detection import Detection
from backend.app.schemas.price_tag import PriceTagRecord


@dataclass
class PriceTagResult:
    """Combined detection + OCR result for a single price tag."""

    detection: Detection
    record: PriceTagRecord
    color: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self.detection),
            **asdict(self.record),
            "color": self.color,
        }
