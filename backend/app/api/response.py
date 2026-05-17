from pydantic import BaseModel, Field
from typing import List, Optional


class PriceTagResponse(BaseModel):
    filename: str = Field(default="")
    product_name: str = Field(default="")
    price_default: float = Field(default=0.0)
    price_card: float = Field(default=0.0)
    price_discount: float = Field(default=0.0)
    barcode: str = Field(default="")
    discount_amount: float = Field(default=0.0)
    id_sku: str = Field(default="")
    print_datetime: str = Field(default="")
    code: str = Field(default="")
    additional_info: str = Field(default="")
    color: str = Field(default="")
    special_symbols: str = Field(default="")
    frame_timestamp: float = Field(default=0.0)
    x_min: int = Field(default=0)
    y_min: int = Field(default=0)
    x_max: int = Field(default=0)
    y_max: int = Field(default=0)
    qr_code_barcode: str = Field(default="")
    price1_qr: float = Field(default=0.0)
    price2_qr: float = Field(default=0.0)
    price3_qr: float = Field(default=0.0)
    price4_qr: float = Field(default=0.0)
    wholesale_level_1_count: int = Field(default=0)
    wholesale_level_1_price: float = Field(default=0.0)
    wholesale_level_2_count: int = Field(default=0)
    wholesale_level_2_price: float = Field(default=0.0)
    action_price_qr: float = Field(default=0.0)
    action_code_qr: str = Field(default="")


class ProcessResponse(BaseModel):
    video_id: str
    total_tags: int = 0
    tags: List[PriceTagResponse] = Field(default_factory=list)
    csv_download_url: str = ""


class ProcessRequest(BaseModel):
    frame_interval: int = Field(default=5, ge=1, le=30)
    confidence_threshold: float = Field(default=0.3, ge=0.1, le=0.9)


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: bool = False
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None