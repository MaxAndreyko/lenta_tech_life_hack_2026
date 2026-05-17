from __future__ import annotations

import cv2
import numpy as np


class CropService:
    """Perspective-correct a price tag crop produced by YOLO detection."""

    def __init__(
        self,
        max_width: int = 900,
        block: int = 15,
        c_start: int = 15,
        c_end: int = 1,
        c_step: int = 1,
        gamma_start: int = 10,
        gamma_end: int = 50,
        gamma_step: int = 3,
        morph_k: int = 1,
        morph_it: int = 1,
        min_area_ratio: float = 0.002,
        eps: float = 0.40,
        area_inv_ratio: float = 0.40,
        edge_ratio_error: float = 0.15,
        valid_ratios: list[float] | None = None,
        ratio_tolerance: float = 0.12,
    ) -> None:
        self.max_width = max_width
        self.block = self._make_odd(block)
        self.c_start = c_start
        self.c_end = c_end
        self.c_step = c_step
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.gamma_step = gamma_step
        self.morph_k = max(1, morph_k)
        self.morph_it = morph_it
        self.min_area_ratio = min_area_ratio
        self.eps = eps
        self.area_inv_ratio = area_inv_ratio
        self.edge_ratio_error = edge_ratio_error
        self.valid_ratios = valid_ratios if valid_ratios is not None else [0.7, 1.0, 1.4, 2.8]
        self.ratio_tolerance = ratio_tolerance

    def process(self, img: np.ndarray) -> np.ndarray:
        """Return perspective-corrected price tag. Falls back to the original crop on failure."""

        h0, w0 = img.shape[:2]
        if w0 > self.max_width:
            scale = self.max_width / float(w0)
            small = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = img.copy()

        h, w = small.shape[:2]
        min_contour_area = max(50, int(w * h * self.min_area_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_k, self.morph_k))

        quad_small = self._find_quad(small, min_contour_area, kernel)
        if quad_small is None:
            return img

        quad_orig = quad_small * (1.0 / scale)
        warped = self._four_point_warp(img, quad_orig)
        return warped if warped is not None else img

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _find_quad(
        self,
        small: np.ndarray,
        min_contour_area: int,
        kernel: np.ndarray,
    ) -> np.ndarray | None:
        for c in range(self.c_start, self.c_end - 1, -self.c_step):
            for g100 in range(self.gamma_start, self.gamma_end + 1, self.gamma_step):
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                gray_g = self._adjust_gamma(gray, g100 / 100.0)
                blur = cv2.GaussianBlur(gray_g, (5, 5), 0)

                th = cv2.adaptiveThreshold(
                    blur, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    self.block, c,
                )

                if self._should_invert(th, gray_g):
                    th = cv2.bitwise_not(th)

                morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=self.morph_it)
                morph = cv2.dilate(morph, kernel, iterations=1)

                contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
                if not significant:
                    continue

                all_pts = np.vstack([c_.reshape(-1, 2) for c_ in significant])
                hull = cv2.convexHull(all_pts.astype(np.int32))
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, self.eps * peri / 100.0, True)

                if len(approx) == 4:
                    quad = approx.reshape(4, 2).astype("float32")
                else:
                    rect = cv2.minAreaRect(hull)
                    quad = cv2.boxPoints(rect).astype("float32")

                if not self._quad_edges_ok(quad):
                    continue

                return quad

        return None

    def _quad_edges_ok(self, quad: np.ndarray) -> bool:
        tl, tr, br, bl = quad
        widthA = float(np.linalg.norm(br - bl))
        widthB = float(np.linalg.norm(tr - tl))
        heightA = float(np.linalg.norm(tr - br))
        heightB = float(np.linalg.norm(tl - bl))

        if min(widthA, widthB) == 0 or min(heightA, heightB) == 0:
            return False
        if max(widthA, widthB) / min(widthA, widthB) > 1 + self.edge_ratio_error:
            return False
        if max(heightA, heightB) / min(heightA, heightB) > 1 + self.edge_ratio_error:
            return False

        aspect = max(widthA, widthB) / max(heightA, heightB)
        if not any(abs(aspect - r) <= self.ratio_tolerance for r in self.valid_ratios):
            return False

        return True

    @staticmethod
    def _adjust_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)], dtype="uint8")
        return cv2.LUT(gray, table)

    def _should_invert(self, mask: np.ndarray, gray: np.ndarray) -> bool:
        mask_u = mask == 255
        if not np.any(mask_u):
            return True
        mean_white = float(np.mean(gray[mask_u]))
        mean_black = float(np.mean(gray[~mask_u])) if np.any(~mask_u) else 0.0
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        max_area = max((cv2.contourArea(c) for c in cnts), default=0)
        area_ratio = max_area / float(h * w) if h * w > 0 else 0.0
        return (mean_white < mean_black) or (area_ratio > self.area_inv_ratio)

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @classmethod
    def _four_point_warp(cls, img: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
        rect = cls._order_points(pts)
        tl, tr, br, bl = rect
        max_w = max(int(np.hypot(br[0] - bl[0], br[1] - bl[1])),
                    int(np.hypot(tr[0] - tl[0], tr[1] - tl[1])))
        max_h = max(int(np.hypot(tr[0] - br[0], tr[1] - br[1])),
                    int(np.hypot(tl[0] - bl[0], tl[1] - bl[1])))
        if max_w <= 0 or max_h <= 0:
            return None
        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (max_w, max_h))

    @staticmethod
    def _make_odd(x: int, at_least: int = 3) -> int:
        x = int(max(at_least, x))
        return x if x % 2 == 1 else x + 1
