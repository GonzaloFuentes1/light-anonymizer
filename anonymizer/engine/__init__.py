"""Engine interface used by the API.

The real implementation will be built from the prototype (``test_bench/baselines/prototype.py``),
split into *detect* and *apply* so the user can review findings in between. Until then the API
can run against :mod:`anonymizer.engine.fake`, which follows the same contract.

Contract
--------
``analyze(file, name_list, *, progress=None, cancel=None) -> AnalyzedFile``
    Detect personal data in ``file.path`` without modifying it. Fills ``kind``, ``pages``,
    ``findings`` (view space), ``status`` ("ready" or "error"), ``error``/``error_message``.
    ``progress(fraction, step_text_es)`` is called as work advances; ``cancel`` is a
    ``threading.Event`` that stops the work early (status "cancelled").

``render_page(file, page, zoom) -> bytes``
    PNG of one page in view space (orientation as the user sees it), scaled by ``zoom``
    (1.0 = 1 pixel per point for PDFs, 1 pixel per pixel for images).

``export(file, dest_dir) -> ExportResult``
    Apply every active finding (status != "removed") as real redaction, remove metadata,
    write a new file into ``dest_dir`` (never overwriting the original), then run the leak
    check on the output. A file with leaks is not exported.
"""

from __future__ import annotations

import os
from typing import Protocol

from anonymizer.engine.model import AnalyzedFile, ExportResult


class Engine(Protocol):
    def analyze(self, file: AnalyzedFile, name_list: list[str], *, progress=None, cancel=None) -> AnalyzedFile: ...

    def render_page(self, file: AnalyzedFile, page: int, zoom: float = 1.0) -> bytes: ...

    def export(self, file: AnalyzedFile, dest_dir: str) -> ExportResult: ...


def get_engine() -> Engine:
    """Engine selected by ``ANONYMIZER_ENGINE`` (``fake`` for UI development, default ``real``)."""
    if os.environ.get("ANONYMIZER_ENGINE", "real") == "fake":
        from anonymizer.engine import fake

        return fake.FakeEngine()
    from anonymizer.engine import real  # noqa: F401  (provided in the engine step)

    return real.RealEngine()
