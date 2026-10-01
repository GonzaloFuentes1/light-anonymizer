"""``notebook`` baseline: the logic of the CoP 33 notebook (original sketch, already removed from the repository) as is.

It is a reference: it documents what the notebook covers today and what it does not. That is why
its behaviour is kept even though it is weak (it does not clean metadata, saves with a plain
``doc.save``, looks up each finding with ``search_for`` and does not process scans or images).

Minimal adaptations for the test bench:
- ``NAMES`` is the manifest's ``name_list`` (in the notebook it is a fixed list).
- Detection runs ``detect() + detect_names()`` in a single pass (notebook cell 22).
- The output file keeps the relative path of the input (without the ``CENSURADO_`` prefix).
- The functions return their results instead of printing them.
- PDF without text -> error ``no_text`` (the notebook skips it); image -> ``unsupported``;
  any exception -> ``exception:<Type>``.

The identifiers are translated, but the regular expressions of ``PATTERNS`` are the notebook's,
byte for byte; its keys are the English type codes of the schema.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf

from test_bench.schema import FileEntry, FileResult, Manifest, Redaction

# Notebook cell 8, unchanged (only the keys use the schema's type codes).
PATTERNS = {
    # 12.345.678-9 or 12345678-9, with or without dots, with or without spaces
    "rut": r"\b\d{1,2}\.?\d{3}\.?\d{3}\s*-\s*[\dkK]\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    # mobile +56 9 XXXX XXXX and landline 2 XXXX XXXX
    "phone": r"(?:\+?56\s?)?(?:9\s?\d{4}\s?\d{4}|\b2\s?\d{4}\s?\d{4}\b)",
    "url": r"https?://\S+|\bwww\.\S+",
}

# Notebook cell 22; in the bench it is replaced by the manifest's list.
NAMES = [
    "Ana Maria Rojas Pena",
]

Finding = tuple[int, str, str]  # (page from 1, type, text found)


def has_text(path: Path | str, minimum: int = 50) -> bool:
    """Returns True if the PDF has extractable text (cell 10)."""
    doc = pymupdf.open(path)
    chars = sum(len(p.get_text()) for p in doc)
    doc.close()
    return chars >= minimum


def detect(path: Path | str) -> list[Finding]:
    """Returns a list of (page, type, text found) (cell 12)."""
    doc = pymupdf.open(path)
    findings = []
    for n_page, page in enumerate(doc, start=1):
        text = page.get_text()
        for type_, pattern in PATTERNS.items():
            for m in re.finditer(pattern, text):
                findings.append((n_page, type_, m.group(0)))
    doc.close()
    return findings


def detect_names(path: Path | str, names: list[str] | None = None) -> list[Finding]:
    """Looks up each name of the list with ``search_for`` (cell 22)."""
    names = NAMES if names is None else names
    doc = pymupdf.open(path)
    h = []
    for n, page in enumerate(doc, start=1):
        for name in names:
            if page.search_for(name):
                h.append((n, "name", name))
    doc.close()
    return h


def redact(source: Path | str, output: Path | str, findings: list[Finding]) -> list[Redaction]:
    """Marks each finding, applies the redactions and saves (cell 16). Returns the marked zones."""
    doc = pymupdf.open(source)
    redactions: list[Redaction] = []
    seen: set[tuple[int, str, tuple[float, ...]]] = set()

    # 1. mark each finding
    for n_page, type_, value in findings:
        page = doc[n_page - 1]
        for rect in page.search_for(value):
            page.add_redact_annot(rect, fill=(0, 0, 0))
            key = (n_page, type_, tuple(round(v, 3) for v in rect))
            if key not in seen:
                seen.add(key)
                redactions.append(
                    Redaction(
                        page=n_page - 1,
                        polygon=[[rect.x0, rect.y0], [rect.x1, rect.y0], [rect.x1, rect.y1], [rect.x0, rect.y1]],
                        type=type_,
                        detector="name_list" if type_ == "name" else "regex",
                        text=value,
                    )
                )

    # 2. apply: here the content disappears from the file
    for page in doc:
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)

    doc.save(output)
    doc.close()
    return redactions


def verify(output: Path | str, findings: list[Finding]) -> list[tuple[str, str]]:
    """Detected data still present in the text of the redacted file (cell 18)."""
    doc = pymupdf.open(output)
    text = "".join(p.get_text() for p in doc)
    doc.close()
    return [(t, v) for _, t, v in findings if v in text]


def final_sweep(output: Path | str) -> dict[str, list[str]]:
    """Runs the patterns again over the already redacted file (cell 18)."""
    doc = pymupdf.open(output)
    text = "".join(p.get_text() for p in doc)
    doc.close()
    leftovers = {}
    for type_, pattern in PATTERNS.items():
        found = re.findall(pattern, text)
        if found:
            leftovers[type_] = found
    return leftovers


# ---------------------------------------------------------------------------


def process(file_entry: FileEntry, manifest: Manifest, folder: Path, details: dict[str, Any]) -> FileResult:
    source = Path(manifest.root) / file_entry.path
    dest = folder / file_entry.path
    if source.suffix.lower() != ".pdf":
        return FileResult(input=file_entry.path, output=None, error="unsupported")
    try:
        if not has_text(source):
            return FileResult(input=file_entry.path, output=None, error="no_text")
        findings = detect(source) + detect_names(source, manifest.name_list)
        dest.parent.mkdir(parents=True, exist_ok=True)
        redactions = redact(source, dest, findings)
        with pymupdf.open(dest) as doc:
            n_pages = doc.page_count
    except Exception as err:  # noqa: BLE001 - the report records the exception type
        dest.unlink(missing_ok=True)
        return FileResult(input=file_entry.path, output=None, error=f"exception:{type(err).__name__}")

    # What the notebook itself would report (cell 18); informational only.
    try:
        verification: dict[str, Any] = {
            "detected": len(findings),
            "leaks_per_notebook": len(verify(dest, findings)),
            "final_sweep": final_sweep(dest),
        }
    except Exception as err:  # noqa: BLE001
        verification = {"detected": len(findings), "verification_error": type(err).__name__}
    details.setdefault("notebook_verification", {})[file_entry.path] = verification
    return FileResult(input=file_entry.path, output=file_entry.path, redactions=redactions, pages_processed=n_pages)
