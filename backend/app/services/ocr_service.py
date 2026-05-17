from __future__ import annotations

import logging
import re

import numpy as np
from paddleocr import PaddleOCR

from backend.app.schemas.ocr import OcrLine
from backend.app.schemas.price_tag import PriceTagRecord

logger = logging.getLogger(__name__)

# Matches prices like 3 789.49 / 1899,99 / 99.90
_PRICE_RE = re.compile(r'\b(\d[\d\s]{0,4}\d[.,]\d{2})\b')
# Discount: -38% / 38%
_DISCOUNT_RE = re.compile(r'-?\d{1,3}\s*%')
# EAN-8 / EAN-13 barcode digits
_BARCODE_RE = re.compile(r'\b(\d{8}|\d{13})\b')
# SKU / article: long numeric id that is NOT a price or barcode
_SKU_RE = re.compile(r'\b(\d{9,12})\b')
# Date-time on tag: 17.02.2026 1:14
_DATETIME_RE = re.compile(r'\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}')
# Wholesale: "от 6 шт." style
_WHOLESALE_QTY_RE = re.compile(r'от\s+(\d+)\s+шт', re.IGNORECASE)
# Special single-char symbols (Ш = шампанское, etc.)
_SPECIAL_SYMBOL_RE = re.compile(r'^[А-ЯA-Z]$')


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(" ", "").replace(",", "."))
    except ValueError:
        return None


class OcrService:
    """Extract text lines from a price tag crop using PaddleOCR."""

    def __init__(
        self,
        lang: str = "ru",
        use_angle_cls: bool = True,
        min_confidence: float = 0.4,
    ) -> None:
        self.min_confidence = min_confidence
        self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)

    def recognize(self, img: np.ndarray) -> list[OcrLine]:
        """Return all text lines above min_confidence for the given image."""

        result = self._ocr.ocr(img, cls=True)
        if not result or not result[0]:
            return []

        lines: list[OcrLine] = []
        for item in result[0]:
            quad, (text, confidence) = item
            if confidence < self.min_confidence:
                continue
            lines.append(OcrLine(
                text=text,
                confidence=round(float(confidence), 4),
                bbox_quad=tuple(tuple(pt) for pt in quad),
            ))

        return lines

    def parse(self, lines: list[OcrLine]) -> PriceTagRecord:
        """Parse a list of OCR lines into a structured PriceTagRecord."""

        record = PriceTagRecord()
        prices: list[float] = []
        name_parts: list[str] = []

        for line in lines:
            text = line.text.strip()

            # --- datetime ---
            dt_match = _DATETIME_RE.search(text)
            if dt_match and not record.print_datetime:
                record.print_datetime = dt_match.group(0)
                continue

            # --- barcode (EAN-8/13) ---
            bc_match = _BARCODE_RE.search(text)
            if bc_match and not record.barcode:
                record.barcode = bc_match.group(1)
                continue

            # --- SKU ---
            sku_match = _SKU_RE.search(text)
            if sku_match and not record.id_sku:
                record.id_sku = sku_match.group(1)
                continue

            # --- discount ---
            disc_match = _DISCOUNT_RE.search(text)
            if disc_match and not record.discount_amount:
                record.discount_amount = disc_match.group(0).strip()
                continue

            # --- wholesale ---
            ws_match = _WHOLESALE_QTY_RE.search(text)
            if ws_match:
                qty = ws_match.group(1)
                price_hit = _PRICE_RE.search(text)
                price_val = _to_float(price_hit.group(1)) if price_hit else None
                if not record.wholesale_level_1_count:
                    record.wholesale_level_1_count = qty
                    record.wholesale_level_1_price = price_val
                else:
                    record.wholesale_level_2_count = qty
                    record.wholesale_level_2_price = price_val
                continue

            # --- prices ---
            price_hits = _PRICE_RE.findall(text)
            if price_hits:
                for hit in price_hits:
                    val = _to_float(hit)
                    if val is not None:
                        prices.append(val)
                continue

            # --- special single-char symbol ---
            if _SPECIAL_SYMBOL_RE.match(text) and not record.special_symbols:
                record.special_symbols = text
                continue

            # --- product name / additional info ---
            if len(text) > 3 and not re.fullmatch(r'[\d\s.,%-]+', text):
                name_parts.append(text)

        # assign prices: largest → default, then descending
        if prices:
            prices_sorted = sorted(set(prices), reverse=True)
            record.price_default = prices_sorted[0]
            if len(prices_sorted) > 1:
                record.price_card = prices_sorted[1]
            if len(prices_sorted) > 2:
                record.price_discount = prices_sorted[2]

        if name_parts:
            record.product_name = " ".join(name_parts[:4])

        return record
