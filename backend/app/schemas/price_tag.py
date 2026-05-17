from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PriceTagRecord:
    """Structured data extracted from a single price tag."""

    # --- identification ---
    barcode: str = ""
    id_sku: str = ""
    qr_code_barcode: str = ""

    # --- product ---
    product_name: str = ""
    additional_info: str = ""
    color: str = ""
    special_symbols: str = ""

    # --- pricing ---
    price_default: float | None = None
    price_card: float | None = None
    price_discount: float | None = None
    discount_amount: str = ""

    # --- QR prices ---
    price1_qr: float | None = None
    price2_qr: float | None = None
    price3_qr: float | None = None
    price4_qr: float | None = None
    action_price_qr: float | None = None
    action_code_qr: str = ""

    # --- wholesale ---
    wholesale_level_1_count: str = ""
    wholesale_level_1_price: float | None = None
    wholesale_level_2_count: str = ""
    wholesale_level_2_price: float | None = None

    # --- meta ---
    print_datetime: str = ""
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
