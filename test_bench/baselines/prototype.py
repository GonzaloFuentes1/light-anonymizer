"""The real engine (``anonymizer.engine.real``) as a baseline of the test bench.

The prototype's detection now lives in the engine, split into ``analyze`` (detect) and
``export`` (apply, clean and check for leaks). Here every proposed finding stays active, as if
the reviewer accepted all of them, and the findings are converted back to the bench's polygon
convention for the redaction report: PDF in unrotated page space (the engine works in view
space, after ``/Rotate``), images in pixels after EXIF orientation (the same as the engine).

It runs with every detection group on and the URLs that are not personal applied
(``DetectionOptions.everything()``), so its numbers stay comparable with the runs made before
the groups could be chosen and before D12 left those URLs unapplied by default.

A file whose output does not pass the engine's own leak check is not exported (error
``leak``; the messages go to ``details["leaks"]``). The engine's time per stage goes to
``details["engine_timings"]``.

The helpers ``_process_pdf`` / ``_process_image`` keep the prototype's interface for
``test_bench.process_folder`` and the tests.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pymupdf

from anonymizer.engine.common import bbox_of, rect_polygon
from anonymizer.engine.faces import merge as _merge_faces  # noqa: F401  (prototype interface)
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import AnalyzedFile, DetectionOptions
from anonymizer.engine.raster import detect_in_image  # noqa: F401  (prototype interface)
from anonymizer.engine.real import RealEngine
from anonymizer.engine.text import find_spans  # noqa: F401  (prototype interface)
from test_bench.schema import FileEntry, FileResult, Manifest, Redaction

# Every group on and the other URLs applied: comparable with earlier runs of the bench.
ALL_ON = DetectionOptions.everything()
_ENGINES: list[RealEngine] = []


def _engine() -> RealEngine:
    if not _ENGINES:
        _ENGINES.append(RealEngine())
    return _ENGINES[0]


class LeakError(RuntimeError):
    """The engine's leak check blocked the export."""

    def __init__(self, messages: list[str]):
        super().__init__("; ".join(messages))
        self.messages = messages


def _redactions(file: AnalyzedFile) -> list[Redaction]:
    """Findings of ``file`` in the bench's convention."""
    to_page: list[pymupdf.Matrix] = []
    if file.kind == "pdf":
        with PDF_LOCK:
            with pymupdf.open(file.path, filetype="pdf") as doc:
                to_page = [pymupdf.Matrix(page.derotation_matrix) for page in doc]
    output = []
    for f in file.findings:
        if file.kind == "pdf":
            x0, y0, x1, y1 = bbox_of(f.polygon)
            r = (pymupdf.Rect(x0, y0, x1, y1) * to_page[f.page]).normalize()
            polygon = rect_polygon(r.x0, r.y0, r.x1, r.y1)
        else:
            polygon = [[float(x), float(y)] for x, y in f.polygon]
        output.append(
            Redaction(
                page=f.page,
                polygon=polygon,
                type=f.type,
                detector=f.detector,
                text=f.text,
                score=f.score,
                status="redact" if f.active else "dismissed",
            )
        )
    return output


def run(source: Path, dest: Path, name_list: tuple[str, ...], options: DetectionOptions = ALL_ON) -> AnalyzedFile:
    """Analyzes ``source`` and exports it with every proposed finding active to ``dest`` (replacing a
    previous run). Suggested findings (D12, when ``options.urls_other`` is off) are not applied."""
    engine = _engine()
    file = AnalyzedFile(id=uuid.uuid4().hex[:12], name=dest.name, path=str(source), options=options.to_dict())
    engine.analyze(file, list(name_list))
    if file.status != "ready":
        return file
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)  # the bench's own output folder: a previous run is replaced
    result = engine.export(file, str(dest.parent))
    if not result.exported:
        raise LeakError([leak.message for leak in result.leaks])
    if Path(result.output_path) != dest:
        Path(result.output_path).replace(dest)
    return file


def _process(source: Path, dest: Path, name_list, options: DetectionOptions) -> tuple[list[Redaction], int]:
    file = run(source, dest, tuple(name_list), options)
    if file.status != "ready":
        raise RuntimeError(f"{file.error}: {file.error_message}")
    return _redactions(file), len(file.pages)


def _process_pdf(
    source: Path, dest: Path, name_list, options: DetectionOptions = ALL_ON
) -> tuple[list[Redaction], int]:
    return _process(source, dest, name_list, options)


def _process_image(
    source: Path, dest: Path, name_list, options: DetectionOptions = ALL_ON
) -> tuple[list[Redaction], int]:
    return _process(source, dest, name_list, options)


def process(file_entry: FileEntry, manifest: Manifest, folder: Path, details: dict[str, Any]) -> FileResult:
    source = Path(manifest.root) / file_entry.path
    dest = folder / file_entry.path
    try:
        file = run(source, dest, tuple(manifest.name_list))
    except LeakError as err:
        details.setdefault("leaks", {})[file_entry.id] = err.messages
        return FileResult(input=file_entry.path, output=None, error="leak")
    if file.status != "ready":
        return FileResult(input=file_entry.path, output=None, error=file.error)
    details.setdefault("engine_timings", {})[file_entry.id] = dict(file.timings)
    return FileResult(
        input=file_entry.path,
        output=file_entry.path,
        redactions=_redactions(file),
        pages_processed=len(file.pages),
    )
