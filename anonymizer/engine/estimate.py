"""Time estimates before processing.

``profile`` reads cheap facts about a file (pages, scanned pages, the images placed on text pages
and their size, the frames and megapixels of an image) without rendering any page. ``CostModel``
turns them into seconds per analysis stage with a few rates (seconds per unit of work) and, after
every analysis, moves those rates towards the measured times (exponential moving average). Only
these numbers are saved, in a small JSON file: never file names, paths or content.

The default rates were fitted to the stage times measured on the 95 readable files of the test
set, on the development machine (i7-13620H, 8 GB, Windows 11, one analysis at a time, every group
on); the first analyses on another computer adjust them.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path

from anonymizer.engine.common import OCR_DPI, SCANNED_MAX_CHARS, sniff
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import DETECTION_GROUPS, DetectionOptions

log = logging.getLogger(__name__)

# Stage of the analysis -> detection group that pays for it. "render" (rendering a page or
# decoding an image) is shared by OCR, faces and QR, and runs when any of them is on.
STAGE_GROUP = {"text": "patterns", "ocr": "ocr", "faces": "faces", "qr": "qr"}
STAGES = ("text", "render", "ocr", "faces", "qr")

# Seconds per unit of work, per stage. Units (see ``units``): every PDF page (pdf_page), PDF pages
# that are rendered (raster_page: scanned, or with images), scanned pages (scanned_page), distinct
# images placed on text pages (region) and their megapixels at OCR resolution (region_mp), frames
# of an image file (image_frame) and their megapixels (image_mp). OCR of an image is per frame,
# not per megapixel: the text detector scales every image to the same size, and a 12 MP photo
# usually has less text than a screenshot (measured: photos about 2 s, screenshots about 10 s).
DEFAULT_RATES: dict[str, dict[str, float]] = {
    "text": {"pdf_page": 0.075},
    "render": {"raster_page": 0.14, "image_mp": 0.02},
    "ocr": {"scanned_page": 10.0, "region": 1.45, "region_mp": 1.9, "image_frame": 3.5},
    "faces": {"scanned_page": 1.2, "region": 0.15, "image_mp": 0.35},
    "qr": {"raster_page": 0.1, "image_mp": 0.02},
}
SMOOTHING = 0.3  # weight of the newest measurement in the moving average
_MIN_RATE, _MAX_RATE = 1e-5, 1e4
_MAX_RATIO = 5.0  # one odd measurement (the computer was asleep) moves a rate at most this much
_FILE_VERSION = 1


# ---------------------------------------------------------------------------
# Profile: cheap facts about a file
# ---------------------------------------------------------------------------


def empty_profile(kind: str | None = None) -> dict:
    return {
        "kind": kind,
        "pages": 0,
        "scanned_pages": 0,
        "raster_pages": 0,
        "regions": 0,
        "regions_mp": 0.0,
        "frames": 0,
        "megapixels": 0.0,
    }


def profile(path: str, kind: str | None = None) -> dict:
    """Facts about ``path`` for the estimate. A file that cannot be read gets an empty profile
    (the analysis reports the problem)."""
    try:
        kind = kind or sniff(path)
        if kind == "pdf":
            return _pdf_profile(path)
        if kind == "image":
            return _image_profile(path)
    except Exception:  # noqa: BLE001 - an estimate never fails: the analysis explains the problem
        log.debug("could not profile a file", exc_info=True)
        return empty_profile(kind if kind in ("pdf", "image") else None)
    return empty_profile()


def _pdf_profile(path: str) -> dict:
    """Pages, scanned pages (same rule as ``pdf.TextPage.scanned``) and the images of the text pages.

    No page is rendered: the text comes from PyMuPDF and the images from ``get_image_info``. An
    image placed with the same size on several pages counts once (the engine reads it once).
    ``PDF_LOCK`` is taken one page at a time, so a long document does not hold up the analyses.
    """
    import pymupdf

    facts = empty_profile("pdf")
    zoom = OCR_DPI / 72
    seen: set[tuple[int, int, int]] = set()
    flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_MEDIABOX_CLIP
    with PDF_LOCK:
        doc = pymupdf.open(path, filetype="pdf")
    try:
        with PDF_LOCK:
            if doc.needs_pass:
                return facts
            doc.xref_set_key(doc.pdf_catalog(), "OCProperties", "null")  # hidden layers are read too
            facts["pages"] = count = doc.page_count
        for n in range(count):
            with PDF_LOCK:
                page = doc[n]
                text = page.get_textpage(clip=pymupdf.INFINITE_RECT(), flags=flags).extractText()
                scanned = len(text.strip()) < SCANNED_MAX_CHARS
                info = [] if scanned else page.get_image_info(xrefs=True)
                to_pix = page.rotation_matrix * pymupdf.Matrix(zoom, zoom)
                page_rect = pymupdf.Rect(page.rect)
            if scanned:
                facts["scanned_pages"] += 1
                facts["raster_pages"] += 1
                continue
            if not info:
                continue
            facts["raster_pages"] += 1
            for item in info:
                r = (pymupdf.Rect(item["bbox"]) & page_rect) * to_pix
                w, h = round(abs(r.width)) + 16, round(abs(r.height)) + 16  # the engine adds a margin
                if r.is_empty or w < 24 or h < 24:
                    continue
                key = (int(item.get("xref") or 0), w, h)
                if key in seen:
                    continue
                seen.add(key)
                facts["regions"] += 1
                facts["regions_mp"] += w * h / 1e6
    finally:
        with PDF_LOCK:
            doc.close()
    facts["regions_mp"] = round(facts["regions_mp"], 3)
    return facts


def _image_profile(path: str) -> dict:
    """Frames and megapixels, read from the header (nothing is decoded)."""
    from PIL import Image

    facts = empty_profile("image")
    with Image.open(path) as img:
        count = 1 if img.format == "MPO" else getattr(img, "n_frames", 1)
        for n in range(count):
            if n:
                img.seek(n)
            facts["frames"] += 1
            facts["megapixels"] += img.width * img.height / 1e6
    facts["pages"] = facts["frames"]
    facts["megapixels"] = round(facts["megapixels"], 3)
    return facts


def units(facts: dict) -> dict[str, dict[str, float]]:
    """Units of work of each stage for a profile."""
    return {
        "text": {"pdf_page": facts["pages"] if facts.get("kind") == "pdf" else 0},
        "render": {"raster_page": facts["raster_pages"], "image_mp": facts["megapixels"]},
        "ocr": {
            "scanned_page": facts["scanned_pages"],
            "region": facts["regions"],
            "region_mp": facts["regions_mp"],
            "image_frame": facts["frames"],
        },
        "faces": {"scanned_page": facts["scanned_pages"], "region": facts["regions"], "image_mp": facts["megapixels"]},
        "qr": {"raster_page": facts["raster_pages"], "image_mp": facts["megapixels"]},
    }


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


class CostModel:
    """Seconds per unit of work for each stage, adjusted with the measured times.

    ``path``: where the rates are kept between sessions (None: only in memory, as in the tests).
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.rates = {stage: dict(rates) for stage, rates in DEFAULT_RATES.items()}
        self.samples = 0  # analyses measured so far (here and in earlier sessions)
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path | None) -> CostModel:
        """The rates saved in ``path``. A missing or broken file (not JSON, wrong shape, values out
        of range) gives the default rates: the app always starts. Each valid rate is used."""
        model = cls(path)
        if model.path is None or not model.path.is_file():
            return model
        try:
            data = json.loads(model.path.read_text(encoding="utf-8"))
            saved = data["rates"]
            rates = {stage: dict(values) for stage, values in model.rates.items()}
            for stage, values in DEFAULT_RATES.items():
                for unit in values:
                    value = saved.get(stage, {}).get(unit)
                    # bool is an int in Python, and "Infinity" or "NaN" are valid in Python's JSON.
                    if (
                        isinstance(value, int | float)
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and _MIN_RATE <= value <= _MAX_RATE
                    ):
                        rates[stage][unit] = float(value)
            samples = data.get("samples", 0)
            if isinstance(samples, bool) or not isinstance(samples, int | float) or not math.isfinite(samples):
                samples = 0
        except Exception:  # noqa: BLE001 - a broken estimates file never stops the app
            log.warning("time estimates file unreadable: starting from the default rates", exc_info=True)
            return model
        model.rates, model.samples = rates, max(0, int(samples))
        return model

    @property
    def calibrated(self) -> bool:
        """At least one analysis on this computer has adjusted the rates."""
        return self.samples > 0

    def stages(self, facts: dict) -> dict[str, float]:
        """Estimated seconds of each stage for a profile (every group on)."""
        work = units(facts)
        with self._lock:
            return {
                stage: sum(self.rates[stage][unit] * count for unit, count in work[stage].items()) for stage in STAGES
            }

    def update(self, facts: dict, timings: dict[str, float]) -> bool:
        """Moves the rates towards the times measured for one analysis. Returns True when it changed them.

        Only stages that ran (present in ``timings``) are used. Each rate moves in proportion to
        its share of the estimate of its stage, so the unit that explains most of the work learns most.
        """
        work = units(facts)
        changed = False
        with self._lock:
            for stage in STAGES:
                measured = timings.get(stage)
                if measured is None or not math.isfinite(measured) or measured < 0:
                    continue
                estimated = sum(self.rates[stage][unit] * count for unit, count in work[stage].items())
                if estimated <= 0:
                    continue
                ratio = min(_MAX_RATIO, max(1 / _MAX_RATIO, measured / estimated))
                for unit, count in work[stage].items():
                    share = self.rates[stage][unit] * count / estimated
                    if share <= 0:
                        continue
                    rate = self.rates[stage][unit] * (1 + SMOOTHING * share * (ratio - 1))
                    self.rates[stage][unit] = min(_MAX_RATE, max(_MIN_RATE, rate))
                    changed = True
            if changed:
                self.samples += 1
        if changed:
            self.save()
        return changed

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {
                "version": _FILE_VERSION,
                "samples": self.samples,
                "rates": {
                    stage: {unit: round(v, 6) for unit, v in rates.items()} for stage, rates in self.rates.items()
                },
            }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_suffix(".tmp")
            partial.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(partial, self.path)
        except OSError:
            log.warning("could not save the time estimates", exc_info=True)


def by_group(stages: dict[str, float]) -> dict[str, float]:
    """Seconds per detection group (the groups without a stage of their own cost nothing extra)."""
    groups = {g.key: 0.0 for g in DETECTION_GROUPS}
    for stage, group in STAGE_GROUP.items():
        groups[group] += stages.get(stage, 0.0)
    return groups


def total(stages: dict[str, float], options: DetectionOptions) -> float:
    """Seconds with the groups of ``options``: theirs, plus rendering when the pixels are searched."""
    groups = by_group(stages)
    seconds = sum(value for key, value in groups.items() if getattr(options, key))
    return seconds + (stages.get("render", 0.0) if options.raster else 0.0)
