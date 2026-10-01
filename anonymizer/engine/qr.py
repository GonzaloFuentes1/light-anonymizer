"""QR codes (OpenCV). The cédula's QR encodes the RUN and other data: always redacted."""

from __future__ import annotations

import cv2
import numpy as np

from anonymizer.engine.common import Zone


def detect(bgr: np.ndarray) -> list[Zone]:
    """QR codes of a BGR image, with their box enlarged by 15 % around the center."""
    try:
        ok, texts, points, _ = cv2.QRCodeDetector().detectAndDecodeMulti(bgr)
    except cv2.error:
        return []
    if not ok or points is None:
        return []
    output = []
    for pts, txt in zip(points, texts, strict=False):
        center = pts.mean(axis=0)
        output.append(Zone("qr", (pts - center) * 1.15 + center, txt or "", "qr", 1.0))
    return output
