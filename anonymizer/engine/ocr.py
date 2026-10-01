"""Text in images with RapidOCR (PP-OCR models on onnxruntime), at 0°, 90° and 270°.

The line classifier of RapidOCR already covers 180°. One OCR engine is shared by the whole
process and its calls are serialized (``OCR_LOCK``): two analyses at once never run two OCRs
at the same time, which keeps memory bounded.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from anonymizer.engine.common import waiting_for

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

OCR_LOCK = threading.Lock()
ROTATIONS = (0, 1, 3)  # np.rot90 turns: 0°, 90° and 270°
OCR_THREADS = 4
# The PP-OCR models that ship inside the ``rapidocr`` wheel. They are passed by path: without a
# path, RapidOCR downloads any model that is missing or does not match its checksum, and the app
# must never use the network.
MODEL_FILES = {
    "Det": "PP-OCRv6_det_small.onnx",
    "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "Rec": "PP-OCRv6_rec_small.onnx",
}


class OcrLine(NamedTuple):
    polygon: np.ndarray  # 4x2, pixels of the original image
    text: str
    score: float
    turn: int  # np.rot90 turn of the pass that read it (0 = upright)


def model_paths() -> dict[str, Path] | None:
    """Path of each OCR model inside the installed ``rapidocr`` package (None when it is not installed)."""
    spec = importlib.util.find_spec("rapidocr")
    if spec is None or not spec.origin:
        return None
    folder = Path(spec.origin).parent / "models"
    return {stage: folder / name for stage, name in MODEL_FILES.items()}


def available() -> bool:
    paths = model_paths()
    return (
        paths is not None
        and importlib.util.find_spec("onnxruntime") is not None
        and all(path.is_file() for path in paths.values())
    )


@lru_cache(maxsize=1)
def engine():
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from rapidocr import RapidOCR

    params: dict[str, object] = {
        "Global.log_level": "critical",
        "EngineConfig.onnxruntime.intra_op_num_threads": OCR_THREADS,
    }
    for stage, path in (model_paths() or {}).items():
        params[f"{stage}.model_path"] = str(path)
    return RapidOCR(params=params)


def rotate_points(pts: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Maps points of ``np.rot90(img, k)`` back to the original image of width ``w`` and height ``h``."""
    x, y = pts[:, 0], pts[:, 1]
    if k == 0:
        return pts
    if k == 1:
        return np.stack([w - y, x], axis=1)
    if k == 2:
        return np.stack([w - x, h - y], axis=1)
    return np.stack([y, h - x], axis=1)


def read_lines(bgr: np.ndarray, min_side: int = 0, check: Callable[[], None] | None = None) -> list[OcrLine]:
    """Every text line of a BGR image, read in the three orientations.

    ``min_side``: small crops are padded with white up to this side before reading (the boxes
    are clipped back to the image). The text detector scales the image so that its shorter side
    reaches 736 px: without padding, a thin crop of 800x120 px was enlarged six times and took
    tens of seconds. ``check`` is called before each pass (it raises to cancel).
    """
    h, w = bgr.shape[:2]
    img = bgr
    if h < min_side or w < min_side:
        img = np.full((max(h, min_side), max(w, min_side), 3), 255, np.uint8)
        img[:h, :w] = bgr
    ph, pw = img.shape[:2]
    lines: list[OcrLine] = []
    reader = engine()
    for k in ROTATIONS:
        if check is not None:
            check()
        rot = np.ascontiguousarray(np.rot90(img, k))
        with waiting_for(OCR_LOCK):
            r = reader(rot)
        if r.boxes is None or r.txts is None:
            continue
        for box, txt, score in zip(r.boxes, r.txts, r.scores, strict=False):
            pol = rotate_points(np.asarray(box, np.float64), k, pw, ph)
            pol = np.stack([pol[:, 0].clip(0, w), pol[:, 1].clip(0, h)], axis=1)
            lines.append(OcrLine(pol, txt, float(score), k))
    return lines
