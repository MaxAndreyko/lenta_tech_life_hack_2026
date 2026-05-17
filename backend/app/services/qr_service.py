"""QR detection + decoding service.

Detection is done with qrdet (https://github.com/Eric-Canas/qrdet),
which uses a YOLOv8 model fine-tuned on QR codes and is robust to
small/rotated/blurred codes.

Decoding runs a cascade of available decoders against a grid of
preprocessing variants of each detection (quad warp + bbox upscale,
times {raw, gray, CLAHE, Otsu, adaptive threshold, sharpen, ...}).
This recovers small price-tag QRs that no single decoder reads.

The cascade is, in order:
    WeChat QR (CNN + super-resolution)   — requires opencv-contrib-python
    pyzbar (zbar)                        — requires the pyzbar package
    zxing-cpp                            — requires the zxing-cpp package
    OpenCV QRCodeDetector                — always available with opencv

Any decoder that is not installed is silently skipped; qrdet is the
only hard dependency.

Typical usage:
    >>> service = QRService()
    >>> for det in service.detect_and_decode(image):
    ...     if det.is_decoded:
    ...         print(det.decoder, det.payload)
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from backend.app.core.settings import settings
from backend.app.schemas.qr import QRDetection

logger = logging.getLogger(__name__)

# Optional decoder backends — soft-imported.
try:
    from qrdet import QRDetector as _QRDetector
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "qrdet is required for QRService. Install with `pip install qrdet`."
    ) from exc

try:
    from pyzbar.pyzbar import decode as _zbar_decode
except Exception:  # pragma: no cover
    _zbar_decode = None

try:
    import zxingcpp as _zxingcpp
except Exception:  # pragma: no cover
    _zxingcpp = None


_WECHAT_MODEL_URLS = {
    "detect.prototxt":
        "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt",
    "detect.caffemodel":
        "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel",
    "sr.prototxt":
        "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt",
    "sr.caffemodel":
        "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel",
}


def _ensure_wechat_models(cache_dir: Path) -> Path | None:
    """Download the 4 WeChat QR model files into `cache_dir` if missing.
    Returns the directory on success, or None if opencv-contrib is unavailable
    or the download fails."""

    if not hasattr(cv2, "wechat_qrcode"):
        logger.info(
            "cv2.wechat_qrcode unavailable; install opencv-contrib-python "
            "to enable the strongest decoder."
        )
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, url in _WECHAT_MODEL_URLS.items():
        dst = cache_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        try:
            logger.info("Downloading WeChat QR model %s ...", name)
            urllib.request.urlretrieve(url, dst)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to download %s: %s", name, exc)
            return None
    return cache_dir


class QRService:
    """Detect and decode QR codes on price-tag crops."""

    def __init__(
        self,
        model_size: str = "s",
        conf_threshold: float = 0.5,
        nms_iou: float = 0.3,
        wechat_cache_dir: Path | None = None,
    ) -> None:
        self._detector = _QRDetector(
            model_size=model_size,
            conf_th=conf_threshold,
            nms_iou=nms_iou,
        )
        self._cv2_qr = cv2.QRCodeDetector()
        self._wechat = self._init_wechat(wechat_cache_dir)
        backends = ["cv2"]
        if self._wechat is not None:
            backends.insert(0, "wechat")
        if _zbar_decode is not None:
            backends.append("zbar")
        if _zxingcpp is not None:
            backends.append("zxing")
        logger.info("QRService ready; decoders=%s", backends)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def detect(self, image: np.ndarray) -> list[QRDetection]:
        """Detect QR codes in the BGR image (no decoding)."""

        raw = self._detector.detect(image=image, is_bgr=True)
        out: list[QRDetection] = []
        for d in raw:
            bbox = tuple(float(v) for v in d["bbox_xyxy"])
            quad = d.get("quad_xy")
            quad_t: tuple | None = None
            if quad is not None:
                quad_t = tuple(tuple(float(c) for c in pt) for pt in quad)
            out.append(QRDetection(
                bbox_xyxy=bbox,
                confidence=float(d["confidence"]),
                quad_xy=quad_t,
            ))
        return out

    def decode(
        self,
        image: np.ndarray,
        detections: Iterable[QRDetection],
        raw_qrdet: list[dict] | None = None,
    ) -> list[QRDetection]:
        """Attempt to decode each detection. Returns new QRDetection objects
        with `payload` and `decoder` filled in where possible."""

        results: list[QRDetection] = []
        raw_list = list(raw_qrdet) if raw_qrdet is not None else None
        for i, det in enumerate(detections):
            raw = raw_list[i] if raw_list and i < len(raw_list) else None
            decoder, payload = self._decode_one(image, det, raw)
            results.append(QRDetection(
                bbox_xyxy=det.bbox_xyxy,
                confidence=det.confidence,
                quad_xy=det.quad_xy,
                payload=payload,
                decoder=decoder,
            ))
        return results

    def detect_and_decode(self, image: np.ndarray) -> list[QRDetection]:
        """Convenience: detect, then decode every detection."""

        raw = self._detector.detect(image=image, is_bgr=True)
        detections = [QRDetection(
            bbox_xyxy=tuple(float(v) for v in d["bbox_xyxy"]),
            confidence=float(d["confidence"]),
            quad_xy=(tuple(tuple(float(c) for c in pt) for pt in d["quad_xy"])
                     if d.get("quad_xy") is not None else None),
        ) for d in raw]
        return self.decode(image, detections, raw_qrdet=raw)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _init_wechat(self, cache_dir: Path | None):
        if cache_dir is None:
            cache_dir = settings.storage.data_dir / "qr_models"
        ready = _ensure_wechat_models(cache_dir)
        if ready is None:
            return None
        try:
            return cv2.wechat_qrcode.WeChatQRCode(
                str(cache_dir / "detect.prototxt"),
                str(cache_dir / "detect.caffemodel"),
                str(cache_dir / "sr.prototxt"),
                str(cache_dir / "sr.caffemodel"),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("WeChat QR init failed: %s", exc)
            return None

    def _try_decoders(self, img: np.ndarray) -> tuple[str, str] | None:
        """Run every available decoder on `img`. Returns (decoder, payload)
        on first success, else None."""

        if img is None or img.size == 0:
            return None
        if self._wechat is not None:
            try:
                texts, _ = self._wechat.detectAndDecode(img)
                for t in texts:
                    if t:
                        return ("wechat", t)
            except cv2.error:
                pass
        if _zbar_decode is not None:
            try:
                for r in _zbar_decode(img):
                    if r.data:
                        return ("zbar", r.data.decode("utf-8", errors="replace"))
            except Exception:
                pass
        if _zxingcpp is not None:
            try:
                r = _zxingcpp.read_barcodes(img)
                if r:
                    return ("zxing", r[0].text)
            except Exception:
                pass
        try:
            data, _, _ = self._cv2_qr.detectAndDecode(img)
            if data:
                return ("cv2", data)
        except cv2.error:
            pass
        return None

    @staticmethod
    def _preprocess_variants(bgr_or_gray: np.ndarray):
        """Yield (tag, image) preprocessing variants. Stays compact —
        each variant is cheap and the cascade short-circuits on success."""

        if bgr_or_gray.ndim == 3:
            yield "raw", bgr_or_gray
            gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_or_gray
        yield "gray", gray

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        yield "clahe", clahe

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield "otsu", otsu
        yield "otsu_inv", 255 - otsu

        _, otsu_c = cv2.threshold(
            clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        yield "otsu_clahe", otsu_c

        for bs in (15, 31, 51):
            for c in (2, 5, 10):
                yield f"adap_{bs}_{c}", cv2.adaptiveThreshold(
                    clahe, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    bs, c,
                )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        yield "otsu_close", cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        yield "otsu_open", cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

        blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
        sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
        yield "sharp", sharp

    @staticmethod
    def _warp_quad(image: np.ndarray, quad_xy, side: int) -> np.ndarray:
        """Perspective-warp the QR quad to a `side` x `side` square."""

        quad = np.asarray(quad_xy, dtype=np.float32)
        s = quad.sum(axis=1)
        diff = np.diff(quad, axis=1).ravel()
        src = np.zeros((4, 2), dtype=np.float32)
        src[0] = quad[np.argmin(s)]
        src[2] = quad[np.argmax(s)]
        src[1] = quad[np.argmin(diff)]
        src[3] = quad[np.argmax(diff)]
        dst = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, m, (side, side), flags=cv2.INTER_CUBIC)

    def _decode_one(
        self,
        image: np.ndarray,
        det: QRDetection,
        raw: dict | None,
    ) -> tuple[str, str]:
        """Try every (warp/upscale × variant) combination through every decoder."""

        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
        pad = max(10, int(0.2 * max(x2 - x1, y2 - y1)))
        bbox_crop = image[
            max(0, y1 - pad):min(h, y2 + pad),
            max(0, x1 - pad):min(w, x2 + pad),
        ]

        quad = None
        if raw is not None:
            quad = raw.get("padded_quad_xy")
            if quad is None:
                quad = raw.get("quad_xy")
        if quad is None and det.quad_xy is not None:
            quad = np.asarray(det.quad_xy, dtype=np.float32)

        sources: list[np.ndarray] = []
        if quad is not None:
            for side in (384, 512, 768):
                sources.append(self._warp_quad(image, quad, side))
        for fx in (4, 6):
            if bbox_crop.size:
                sources.append(cv2.resize(
                    bbox_crop, None,
                    fx=fx, fy=fx,
                    interpolation=cv2.INTER_CUBIC,
                ))

        for src in sources:
            for _, variant in self._preprocess_variants(src):
                hit = self._try_decoders(variant)
                if hit is not None:
                    return hit
        return ("", "")
