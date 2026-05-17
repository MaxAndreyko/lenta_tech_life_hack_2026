from __future__ import annotations

import cv2
import numpy as np

_MIN_PROC_SIZE = 400


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
        eps: float = 0.04,
        area_inv_ratio: float = 0.40,
        edge_ratio_error: float = 0.15,
        valid_ratios: list[float] | None = None,
        ratio_tolerance: float = 0.12,
        tight_crop_margin: int = 4,
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
        self.tight_crop_margin = tight_crop_margin

    def process(self, img: np.ndarray) -> np.ndarray:
        """Return perspective-corrected and tightly-cropped price tag."""

        h0, w0 = img.shape[:2]

        proc_scale = max(_MIN_PROC_SIZE / max(h0, w0), 1.0)
        proc = cv2.resize(img, (int(w0 * proc_scale), int(h0 * proc_scale)),
                          interpolation=cv2.INTER_CUBIC)

        if proc.shape[1] > self.max_width:
            s = self.max_width / proc.shape[1]
            proc = cv2.resize(proc, (self.max_width, int(proc.shape[0] * s)),
                              interpolation=cv2.INTER_AREA)
            proc_scale *= s

        # 1. Color-segmentation quad (red / white / yellow regions)
        quad_proc = self._find_quad_by_color(proc)

        # 2. Canny-edge quad (works even when the tag fills the crop)
        if quad_proc is None:
            quad_proc = self._find_quad_canny(proc)

        # 3. Adaptive-threshold loop (legacy fallback)
        if quad_proc is None:
            h, w = proc.shape[:2]
            min_contour_area = max(50, int(w * h * self.min_area_ratio))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_k, self.morph_k))
            quad_proc = self._find_quad(proc, min_contour_area, kernel)

        if quad_proc is not None:
            quad_orig = quad_proc / proc_scale
            warped = self._four_point_warp(img, quad_orig)
            if warped is not None:
                return self._tight_crop(warped)

        return self._tight_crop(img)

    def detect_color(self, img: np.ndarray, border_width: int = 10) -> str:
        """Detect the dominant frame/background color of a price tag crop."""

        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=bool)
        mask[:border_width, :] = True
        mask[-border_width:, :] = True
        mask[:, :border_width] = True
        mask[:, -border_width:] = True

        border_pixels = img[mask]

        hsv = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        hsv = hsv.reshape(-1, 3)

        saturated = hsv[(hsv[:, 1] >= 50) & (hsv[:, 2] >= 50)]

        if len(saturated) == 0:
            return "white"

        hues = saturated[:, 0]
        bins = np.bincount((hues // 5).astype(np.int32), minlength=36)
        dominant_hue = int(np.argmax(bins)) * 5 * 2

        return self._hue_to_color(dominant_hue)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _tight_crop(self, img: np.ndarray) -> np.ndarray:
        """Remove background bleed by cropping to the content bounding box."""

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[(s >= 40)] = 255                   # saturated pixels  → colored tag areas
        mask[(v >= 200) & (s < 40)] = 255       # bright + neutral  → white tag area

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img

        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        m = self.tight_crop_margin
        x = max(0, x - m)
        y = max(0, y - m)
        w = min(img.shape[1] - x, w + 2 * m)
        h = min(img.shape[0] - y, h + 2 * m)

        cropped = img[y:y + h, x:x + w]
        return cropped if cropped.size > 0 else img

    def _find_quad_by_color(self, img: np.ndarray) -> np.ndarray | None:
        """Find the price tag quad using HSV color segmentation (red + white + yellow)."""

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        red1  = cv2.inRange(hsv, (0,   80,  80),  (15,  255, 255))
        red2  = cv2.inRange(hsv, (160, 80,  80),  (180, 255, 255))
        red   = cv2.bitwise_or(red1, red2)
        white  = cv2.inRange(hsv, (0,   0,   185), (180, 50,  255))
        yellow = cv2.inRange(hsv, (15,  80,  150), (35,  255, 255))

        mask = cv2.bitwise_or(cv2.bitwise_or(red, white), yellow)

        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  close_k, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        largest = max(cnts, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / float(w * h)
        if area_ratio < 0.10:
            return None

        # ── Try approxPolyDP on convex hull (finds real corners, not just bbox) ──
        hull = cv2.convexHull(largest)
        peri = cv2.arcLength(hull, True)
        for eps_f in [0.02, 0.04, 0.06, 0.08]:
            approx = cv2.approxPolyDP(hull, eps_f * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
                quad_fill = cv2.contourArea(quad.astype(np.int32)) / float(w * h)
                # Skip quads that essentially cover the whole image (useless warp)
                if quad_fill < 0.92:
                    return quad

        # ── Fallback: minAreaRect – only keep if the tag is meaningfully tilted ──
        rect = cv2.minAreaRect(largest)
        angle = rect[2]
        # Normalise to [0, 45]: the tilt away from the nearest axis
        tilt = abs(angle % 90)
        tilt = min(tilt, 90 - tilt)

        if tilt < 3 and area_ratio > 0.75:
            # Nearly axis-aligned and fills the crop → warp would do nothing useful
            return None

        return cv2.boxPoints(rect).astype("float32")

    def _find_quad_canny(self, img: np.ndarray) -> np.ndarray | None:
        """Find price tag quad via Canny edge detection.

        Works even when the tag fills most of the YOLO crop, because it looks
        for sharp transitions (borders, text bands) rather than colour blobs.
        """

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try two blur levels – fine and coarse
        for ksize in [3, 5]:
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

            # Auto thresholds from the median intensity (Otsu-like)
            med = float(np.median(blurred))
            lo  = max(0,   int(0.66 * med))
            hi  = min(255, int(1.33 * med))
            edges = cv2.Canny(blurred, lo, hi)

            # Dilate to bridge short gaps in tag border
            k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, k, iterations=2)

            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
                if cv2.contourArea(cnt) < w * h * 0.10:
                    break

                hull = cv2.convexHull(cnt)
                peri = cv2.arcLength(hull, True)

                # Try to get an exact 4-corner approximation
                for eps_f in [0.02, 0.03, 0.04, 0.06]:
                    approx = cv2.approxPolyDP(hull, eps_f * peri, True)
                    if len(approx) == 4:
                        quad = approx.reshape(4, 2).astype("float32")
                        if self._quad_edges_ok(self._order_points(quad)):
                            return quad

                # Fallback: accept minAreaRect only when noticeably tilted
                rect  = cv2.minAreaRect(cnt)
                angle = rect[2]
                tilt  = abs(angle % 90)
                tilt  = min(tilt, 90 - tilt)
                if tilt > 3:
                    quad = cv2.boxPoints(rect).astype("float32")
                    if self._quad_edges_ok(self._order_points(quad)):
                        return quad

        return None

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
                approx = cv2.approxPolyDP(hull, self.eps * peri, True)

                if len(approx) == 4:
                    quad = approx.reshape(4, 2).astype("float32")
                else:
                    rect = cv2.minAreaRect(hull)
                    quad = cv2.boxPoints(rect).astype("float32")

                if not self._quad_edges_ok(self._order_points(quad)):
                    continue

                return quad

        return None

    def _quad_edges_ok(self, quad: np.ndarray) -> bool:
        """Validate a [tl, tr, br, bl] ordered quad for side-ratio and aspect ratio."""
        tl, tr, br, bl = quad
        widthA  = float(np.linalg.norm(br - bl))
        widthB  = float(np.linalg.norm(tr - tl))
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
    def _adjust_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)], dtype="uint8")
        return cv2.LUT(gray, table)

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
    def _hue_to_color(hue: int) -> str:
        if hue <= 15 or hue >= 345:
            return "red"
        elif hue <= 45:
            return "orange"
        elif hue <= 90:
            return "yellow"
        elif hue <= 150:
            return "green"
        elif hue <= 210:
            return "cyan"
        elif hue <= 270:
            return "blue"
        elif hue <= 330:
            return "purple"
        else:
            return "pink"

    @staticmethod
    def _make_odd(x: int, at_least: int = 3) -> int:
        x = int(max(at_least, x))
        return x if x % 2 == 1 else x + 1
