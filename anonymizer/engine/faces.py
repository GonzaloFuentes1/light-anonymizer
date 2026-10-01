"""Faces with YuNet (OpenCV) in 4 orientations and several scales, merged into one box per face.

The box of each detection is enlarged (20 % to the sides, 30 % above for the hair and 20 %
below) and the repeated detections of a face are merged into their union, so the redaction is
never smaller than any of them.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from anonymizer.engine.common import Zone
from anonymizer.engine.ocr import rotate_points
from anonymizer.engine.text import face_doubt
from anonymizer.paths import resource

# ``models/`` of the repository, or of the bundle in the packaged app (see anonymizer.paths).
MODELS_DIR = Path(os.environ.get("ANONYMIZER_MODELS_DIR", resource("models")))
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
# The detector object keeps its input size between calls: one call at a time.
FACE_LOCK = threading.Lock()


def available() -> bool:
    return YUNET_MODEL.is_file()


def load_detector(model: Path):
    """A YuNet detector for the ONNX file ``model``.

    The model is read by Python and handed to OpenCV as bytes: on Windows OpenCV cannot open a
    path with non-ASCII characters (a user folder "Muñoz", "OneDrive - Gobierno Regional de Ñuble").
    """
    weights = np.frombuffer(Path(model).read_bytes(), np.uint8)
    return cv2.FaceDetectorYN.create("onnx", weights, np.empty(0, np.uint8), (320, 320), 0.5, 0.3, 5000)


@lru_cache(maxsize=1)
def detector():
    """The detector of the app (YUNET_MODEL), loaded once."""
    return load_detector(YUNET_MODEL)


def detect(bgr: np.ndarray, threshold: float = 0.5, check: Callable[[], None] | None = None) -> list[Zone]:
    """Raw detections (not merged) in a BGR image."""
    h, w = bgr.shape[:2]
    output: list[Zone] = []
    for k in range(4):
        if check is not None:
            check()
        rot = np.ascontiguousarray(np.rot90(bgr, k))
        rh, rw = rot.shape[:2]
        for side in sorted({640, 1280, max(rh, rw)}):
            if side > max(rh, rw) * 1.01 and side != 640:
                continue
            f = side / max(rh, rw)
            img = cv2.resize(rot, (max(1, round(rw * f)), max(1, round(rh * f))))
            with FACE_LOCK:
                det = detector()
                det.setScoreThreshold(threshold)
                det.setInputSize((img.shape[1], img.shape[0]))
                _, detections = det.detect(img)
            for c in detections if detections is not None else []:
                x, y, cw, ch, score = c[0] / f, c[1] / f, c[2] / f, c[3] / f, float(c[14])
                mx, my = 0.2 * cw, 0.2 * ch
                pts = np.array(
                    [
                        [x - mx, y - my * 1.5],
                        [x + cw + mx, y - my * 1.5],
                        [x + cw + mx, y + ch + my],
                        [x - mx, y + ch + my],
                    ]
                )
                output.append(Zone("face", rotate_points(pts, k, w, h), "", "faces", score))
    return output


def merge(zones, min_iou: float = 0.4, min_inside: float = 0.7) -> list[Zone]:
    """Merges the repeated detections of a face (4 rotations x several scales) into their union.

    ``zones`` are tuples whose items 1 and 4 are the polygon (Nx2) and the score. Two boxes are
    the same face when their IoU is ``>= min_iou`` or when ``min_inside`` of the smaller one lies
    inside the other. The result is the union of the group (never smaller than any detection, so
    recall is kept) with the best score; small or unclear faces are marked doubtful.
    """
    groups: list[list[float]] = []  # x0, y0, x1, y1, score
    for z in zones:
        pol, score = np.asarray(z[1], np.float64), float(z[4])
        (x0, y0), (x1, y1) = pol.min(axis=0), pol.max(axis=0)
        groups.append([float(x0), float(y0), float(x1), float(y1), score])

    def same(a: list[float], b: list[float]) -> bool:
        iw = min(a[2], b[2]) - max(a[0], b[0])
        ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return False
        inter = iw * ih
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter) >= min_iou or inter / max(min(area_a, area_b), 1e-9) >= min_inside

    changed = True
    while changed:  # a union can reach boxes it did not touch before: repeat until stable
        changed = False
        output: list[list[float]] = []
        for g in groups:
            for o in output:
                if same(o, g):
                    o[:] = [min(o[0], g[0]), min(o[1], g[1]), max(o[2], g[2]), max(o[3], g[3]), max(o[4], g[4])]
                    changed = True
                    break
            else:
                output.append(g)
        groups = output
    return [
        Zone(
            "face",
            np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
            "",
            "faces",
            score,
            face_doubt(min(x1 - x0, y1 - y0), score),
        )
        for x0, y0, x1, y1, score in groups
    ]
