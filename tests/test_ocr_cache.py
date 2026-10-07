"""The session's in-memory OCR cache: an image read before (a logo repeated across pages and files)
is not read again; nothing is written anywhere. Invented images only."""

from __future__ import annotations

import numpy as np

from anonymizer.engine import ocr


def _counting(monkeypatch):
    calls = []

    def fake_read(bgr, min_side, check, mirrored):
        calls.append((bgr.shape, min_side, mirrored))
        return [ocr.OcrLine(np.array([[0, 0], [10, 0], [10, 5], [0, 5]], np.float64), "Texto 1234", 0.99, 0)]

    monkeypatch.setattr(ocr, "_read", fake_read)
    return calls


def test_the_same_pixels_are_read_once(monkeypatch):
    calls = _counting(monkeypatch)
    logo = np.full((40, 120, 3), 255, np.uint8)
    logo[10:30, 10:110] = 0
    first = ocr.read_lines(logo, min_side=736)
    again = ocr.read_lines(logo.copy(), min_side=736)  # another array, the same pixels (another page)
    assert len(calls) == 1
    assert [(ln.text, ln.score, ln.turn) for ln in again] == [(ln.text, ln.score, ln.turn) for ln in first]
    again[0].polygon[0, 0] = 99  # callers get their own copy
    assert ocr.read_lines(logo, min_side=736)[0].polygon[0, 0] == 0


def test_other_pixels_or_parameters_are_read(monkeypatch):
    calls = _counting(monkeypatch)
    logo = np.full((40, 120, 3), 255, np.uint8)
    ocr.read_lines(logo)
    ocr.read_lines(logo, min_side=736)  # another reading parameter
    other = logo.copy()
    other[0, 0] = 0  # one pixel differs
    ocr.read_lines(other)
    assert len(calls) == 3


def test_the_cache_is_bounded_and_still_checks_for_cancel(monkeypatch):
    calls = _counting(monkeypatch)
    monkeypatch.setattr(ocr, "CACHE_IMAGES", 2)
    images = [np.full((8, 8, 3), v, np.uint8) for v in (10, 20, 30)]
    for img in images:
        ocr.read_lines(img)
    ocr.read_lines(images[0])  # the oldest went out
    assert len(calls) == 4
    checked = []
    ocr.read_lines(images[0], check=lambda: checked.append(1))
    assert checked == [1] and len(calls) == 4
