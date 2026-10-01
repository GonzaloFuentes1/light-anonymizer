"""Engine interface used by the API.

The real implementation is :mod:`anonymizer.engine.real`: the prototype's detection
(``test_bench/baselines/prototype.py`` now runs it for the test bench), split into *detect* and
*apply* so the user can review findings in between. :mod:`anonymizer.engine.fake` follows the
same contract with simple, fast detection, for UI development and as a fallback when the
models are not installed.

Contract
--------
``analyze(file, name_list, *, progress=None, cancel=None) -> AnalyzedFile``
    Detect personal data in ``file.path`` without modifying it, with the detection groups of
    ``file.options`` (``model.DetectionOptions``; defaults when empty). Fills ``kind``, ``pages``,
    ``findings`` (view space), ``status`` ("ready" or "error"), ``error``/``error_message``,
    ``options`` (the groups it used) and ``timings`` (seconds per stage and in total).
    ``progress(fraction, step_text_es)`` is called as work advances; ``cancel`` is a
    ``threading.Event`` that stops the work early (status "cancelled"). URLs that are not
    personal are ``optional`` findings that start "suggested" (D12).

``profile(file) -> dict``
    Cheap facts about the file for the time estimate (``estimate.profile``): fast, nothing is
    rendered.

``render_page(file, page, zoom) -> bytes``
    PNG of one page in view space (orientation as the user sees it), scaled by ``zoom``
    (1.0 = 1 pixel per point for PDFs, 1 pixel per pixel for images).

``export(file, dest_dir) -> ExportResult``
    Apply every active finding (not "removed" nor "suggested") as real redaction, remove metadata,
    write a new file into ``dest_dir`` (never overwriting the original), then run the leak
    check on the output. A file with leaks is not exported.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from anonymizer.engine.model import AnalyzedFile, ExportResult

log = logging.getLogger(__name__)


class Engine(Protocol):
    def analyze(self, file: AnalyzedFile, name_list: list[str], *, progress=None, cancel=None) -> AnalyzedFile: ...

    def profile(self, file: AnalyzedFile) -> dict: ...

    def render_page(self, file: AnalyzedFile, page: int, zoom: float = 1.0) -> bytes: ...

    def export(self, file: AnalyzedFile, dest_dir: str) -> ExportResult: ...


def get_engine() -> Engine:
    """Engine selected by ``ANONYMIZER_ENGINE`` (``fake`` for UI development, default ``real``).

    When the real engine cannot run (its models or libraries are missing), the development engine
    is used and a warning is logged; ``/api/state`` reports it as ``engine: "fake"``.
    """
    from anonymizer.engine import fake

    if os.environ.get("ANONYMIZER_ENGINE", "real") == "fake":
        return fake.FakeEngine()
    try:
        from anonymizer.engine import real
    except ImportError as exc:
        log.warning("real engine unavailable (%s): using the development engine", exc)
        return fake.FakeEngine()
    missing = real.missing_requirements()
    if missing:
        log.warning("real engine unavailable, missing %s: using the development engine", ", ".join(missing))
        return fake.FakeEngine()
    return real.RealEngine()
