"""Text in images with RapidOCR (PP-OCR models on onnxruntime), at 0°, 90° and 270°.

The line classifier of RapidOCR already covers 180°. An image whose text only reads mirrored (a
photo taken with a front camera) is read again mirrored, but only when the normal pass found no
legible text (D6). One OCR engine is shared by the whole process and its calls are serialized
(``OCR_LOCK``): two analyses at once never run two OCRs at the same time, which keeps memory
bounded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from anonymizer.engine.common import waiting_for
from anonymizer.paths import package_dir

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

OCR_LOCK = threading.Lock()
# What OCR read in images seen before in this session (a letterhead or a logo repeated on every page
# and in every file): key, the decoded pixels' hash and the reading's parameters. Memory only, never
# written anywhere, at most ``CACHE_IMAGES`` images (the least recently used goes first).
CACHE_IMAGES = 256
_cache: OrderedDict[tuple, list[OcrLine]] = OrderedDict()
_cache_lock = threading.Lock()
ROTATIONS = (0, 1, 3)  # np.rot90 turns: 0°, 90° and 270°
OCR_THREADS = 4
# A line is legible text with this score and at least this many letters or digits (D6, ``legible``).
LEGIBLE_SCORE = 0.9
LEGIBLE_LETTERS = 4
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
    package = package_dir("rapidocr")
    if package is None:
        return None
    return {stage: package / "models" / name for stage, name in MODEL_FILES.items()}


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


def _letters(text: str) -> int:
    return sum(c.isalnum() for c in text)


def legible(lines: list[OcrLine]) -> bool:
    """Some line reads as text: at least ``LEGIBLE_LETTERS`` letters or digits read with a score of
    ``LEGIBLE_SCORE`` or more. On the test set, every image with text had lines at 0.999 or more,
    and the two mirrored ones none above 0.83."""
    return any(ln.score >= LEGIBLE_SCORE and _letters(ln.text) >= LEGIBLE_LETTERS for ln in lines)


def needs_mirror(lines: list[OcrLine]) -> bool:
    """D6: the detector found what looks like text (lines of a few letters or more) but none of it is
    legible: it may be mirrored (a photo taken with a front camera). A photo without text, whose
    lines are a few stray characters at most, is not read again."""
    return not legible(lines) and any(_letters(ln.text) >= LEGIBLE_LETTERS for ln in lines)


def read_lines(
    bgr: np.ndarray, min_side: int = 0, check: Callable[[], None] | None = None, mirror: bool = True
) -> list[OcrLine]:
    """Every text line of a BGR image, read in the three orientations.

    ``min_side``: small crops are padded with white up to this side before reading (the boxes
    are clipped back to the image). The text detector scales the image so that its shorter side
    reaches 736 px: without padding, a thin crop of 800x120 px was enlarged six times and took
    tens of seconds. ``check`` is called before each pass (it raises to cancel).

    An image read before in this session with the same parameters (the same decoded pixels: a
    logo or a letterhead repeated across pages and files) is not read again (``CACHE_IMAGES``).

    ``mirror`` (D6): when none of the lines is legible but some look like text, the mirrored
    image is read too, in the same three orientations, and its lines are mapped back to the
    image. That doubles the OCR time, but only of images without legible text.
    """
    key = (
        bgr.shape,
        str(bgr.dtype),
        hashlib.blake2b(np.ascontiguousarray(bgr).tobytes(), digest_size=16).digest(),
        min_side,
        mirror,
        ROTATIONS,
    )
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
    if cached is not None:
        if check is not None:
            check()
        return [ln._replace(polygon=ln.polygon.copy()) for ln in cached]
    lines = _read(bgr, min_side, check, mirrored=False)
    if mirror and needs_mirror(lines):
        lines += _read(bgr, min_side, check, mirrored=True)
    with _cache_lock:
        _cache[key] = [ln._replace(polygon=ln.polygon.copy()) for ln in lines]
        while len(_cache) > CACHE_IMAGES:
            _cache.popitem(last=False)
    return lines


def clear_cache() -> None:
    """Forgets every reading kept by ``read_lines``."""
    with _cache_lock:
        _cache.clear()


def _read(bgr: np.ndarray, min_side: int, check: Callable[[], None] | None, mirrored: bool) -> list[OcrLine]:
    h, w = bgr.shape[:2]
    img = np.ascontiguousarray(bgr[:, ::-1]) if mirrored else bgr
    if h < min_side or w < min_side:
        padded = np.full((max(h, min_side), max(w, min_side), 3), 255, np.uint8)
        padded[:h, :w] = img
        img = padded
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
            if mirrored:  # back from the mirrored image to the image
                pol = np.stack([w - pol[:, 0], pol[:, 1]], axis=1)
            lines.append(OcrLine(pol, txt, float(score), k))
    return lines
