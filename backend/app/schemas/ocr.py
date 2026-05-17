from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BBoxQuad = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


@dataclass(frozen=True)
class OcrLine:
    """Single line of text recognised by OCR."""

    text: str
    confidence: float
    bbox_quad: BBoxQuad

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
