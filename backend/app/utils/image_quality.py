from __future__ import annotations

import cv2
import numpy as np


def calculate_blur_score(frame: np.ndarray) -> float:
    """Estimate frame sharpness via variance of the Laplacian."""

    if frame.ndim == 3:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = frame

    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
