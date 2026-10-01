"""Schema of the manifest (ground truth) and of the redaction report under evaluation.

Coordinate conventions (the same for the manifest and for the engine's report):

- Images: pixels, origin at the top-left corner, in the geometry that results from applying
  the EXIF orientation (``PIL.ImageOps.exif_transpose``). That is the geometry the output
  file must have, because the output carries no EXIF.
- PDF: points (1/72 inch), origin at the top left, in PyMuPDF's *unrotated* page space
  (the same as ``page.search_for`` and ``page.add_redact_annot``). If the page has
  ``/Rotate``, the evaluator transforms with ``page.rotation_matrix`` only to compare pixels.
- A polygon is a list of ``[x, y]`` points in order (clockwise or counterclockwise). Rotated
  quadrilaterals are stored with their 4 real corners, not with their bounding box.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1

Point = list[float]
Polygon = list[Point]

# Types of personal data the manifest records.
TYPES = (
    "rut",
    "email",
    "phone",
    "url",
    "name",
    "address",
    "face",
    "signature",
    "qr",
    "text",  # text without personal data: used for the "redact all text" mode and to measure over-redaction
)

# Types whose acceptance criterion is zero leaks at the "base" level.
ZERO_LEAK_TYPES = ("rut", "email", "phone")

# Difficulty levels.
#   base:         realistic, legible cases. Phase 1 acceptance criterion.
#   stress:       hard cases (tiny text, mirrored, heavy noise). Reported, never blocking.
#   out_of_scope: documented limits (names not in the list, signatures, handwriting).
#                 Reported to keep the limit visible; detecting them is not expected.
LEVELS = ("base", "stress", "out_of_scope")

# Where the data lives in the file.
#   text:   text layer of a PDF (extractable).
#   raster: pixels (standalone image, scanned page or image embedded in a PDF).
#   vector: glyphs converted to paths in a PDF (visible, not extractable).
#   hidden: present in the file but not visible (optional layer turned off, outside the page,
#           white on white, under an image). Evaluated through text and byte extraction.
LAYERS = ("text", "raster", "vector", "hidden")

Unit = Literal["pt", "px"]


@dataclass
class Element:
    """A piece of data that must end up redacted (or, if ``type == "text"``, a neutral reference text)."""

    type: str
    page: int
    polygon: Polygon | None
    value: str | None = None
    level: str = "base"
    layer: str = "raster"
    tags: dict[str, Any] = field(default_factory=dict)
    # For faces: essential region (eyes, nose, mouth). It must be fully covered.
    core: Polygon | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"unknown type: {self.type}")
        if self.level not in LEVELS:
            raise ValueError(f"unknown level: {self.level}")
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer: {self.layer}")
        if self.polygon is not None and len(self.polygon) < 3:
            raise ValueError("a polygon needs at least 3 points")


@dataclass
class MetadataEntry:
    """A piece of personal data hidden in metadata or in non-visible structures of the file.

    ``location`` uses paths such as ``exif.gps``, ``exif.artist``, ``exif.thumbnail``,
    ``xmp.dc:creator``, ``png.text.Author``, ``tiff.artist``, ``pdf.info.author``, ``pdf.xmp``,
    ``pdf.annotation``, ``pdf.attachment``, ``pdf.ocg``, ``pdf.form``, ``pdf.bookmark``,
    ``pdf.javascript``, ``pdf.previous_revision``.
    ``value`` is a unique string the evaluator looks for in the output (canary). For
    non-textual data (for example GPS coordinates or a thumbnail) ``value`` is ``None`` and
    the presence of the structure is evaluated.
    """

    location: str
    value: str | None
    tags: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class Page:
    index: int
    width: float
    height: float
    unit: Unit
    rotation: int = 0  # /Rotate of the PDF page (0, 90, 180, 270); 0 for images


@dataclass
class FileEntry:
    id: str
    path: str  # relative to the dataset root, with "/"
    format: str  # pdf, jpg, png, webp, tiff, heic
    category: str
    description: str
    pages: list[Page]
    elements: list[Element] = field(default_factory=list)
    sensitive_metadata: list[MetadataEntry] = field(default_factory=list)
    # "process", or "error:<code>" (password, corrupt, empty, format) for files that must be rejected
    expected: str = "process"
    tags: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""


@dataclass
class Manifest:
    root: str
    seed: int
    files: list[FileEntry] = field(default_factory=list)
    name_list: list[str] = field(default_factory=list)  # the list handed to the engine
    face_sources: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    version: int = SCHEMA_VERSION

    def add(self, file_entry: FileEntry) -> FileEntry:
        if any(f.id == file_entry.id for f in self.files):
            raise ValueError(f"duplicate file id: {file_entry.id}")
        for i, e in enumerate(file_entry.elements):
            e.id = f"{file_entry.id}#e{i:03d}"
        for i, m in enumerate(file_entry.sensitive_metadata):
            m.id = f"{file_entry.id}#m{i:02d}"
        path = Path(self.root) / file_entry.path
        if path.exists():
            file_entry.sha256 = file_sha256(path)
        self.files.append(file_entry)
        return file_entry

    def validate(self) -> list[str]:
        """Checks consistency: existing files, valid pages, unique canaries.

        The generator CLI prints these problems to the user, so their text stays in Spanish.
        """
        problems: list[str] = []
        seen: dict[str, str] = {}
        for f in self.files:
            if not (Path(self.root) / f.path).exists():
                problems.append(f"{f.id}: no existe {f.path}")
            n_pages = len(f.pages)
            for e in f.elements:
                if not 0 <= e.page < n_pages:
                    problems.append(f"{e.id}: página {e.page} fuera de rango")
                if e.type in ZERO_LEAK_TYPES and e.value:
                    key = normalize(e.type, e.value)
                    previous = seen.get(key)
                    if previous and previous.split("#")[0] != f.id:
                        problems.append(f"{e.id}: valor repetido con {previous} ({e.value})")
                    seen.setdefault(key, e.id)
        return problems

    def save(self, path: Path | None = None) -> Path:
        path = path or Path(self.root) / "manifest.json"
        data = asdict(self)
        data["root"] = "."
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        files = []
        for f in data.pop("files"):
            f["pages"] = [Page(**p) for p in f["pages"]]
            f["elements"] = [Element(**e) for e in f["elements"]]
            f["sensitive_metadata"] = [MetadataEntry(**m) for m in f["sensitive_metadata"]]
            files.append(FileEntry(**f))
        data["root"] = str(Path(path).parent)
        return cls(files=files, **data)


# ---------------------------------------------------------------------------
# Contract of the redaction report that the engine (or a baseline) hands to the evaluator.
# ---------------------------------------------------------------------------


@dataclass
class Redaction:
    """A zone the engine redacted (or marked for redaction) in the output file."""

    page: int
    polygon: Polygon
    type: str
    detector: str  # regex, name_list, yunet, rapidocr, qr, manual, full_page...
    text: str | None = None
    score: float | None = None
    status: str = "redact"  # redact | dismissed (the reviewer removed it)


@dataclass
class FileResult:
    input: str  # relative path of the input file (the same as in the manifest)
    output: str | None  # relative path of the output file inside the output folder
    redactions: list[Redaction] = field(default_factory=list)
    error: str | None = None  # error code if the file was rejected (password, corrupt...)
    time_s: float | None = None
    pages_processed: int | None = None
    all_text_mode: bool = False


@dataclass
class RedactionReport:
    system: str  # name of the evaluated system: engine, identity, oracle, notebook...
    results: list[FileResult] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> RedactionReport:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = []
        for r in data.pop("results"):
            r["redactions"] = [Redaction(**c) for c in r["redactions"]]
            results.append(FileResult(**r))
        return cls(results=results, **data)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_SEPARATORS = str.maketrans("", "", " \t\n\r.-‐‑–—_()+/")


def normalize(type: str, value: str) -> str:
    """Canonical form used to look for a value in extracted text (the same function searches the output)."""
    import unicodedata

    if type in ("rut", "phone"):
        clean = value.upper().translate(_SEPARATORS)
        for prefix in ("RUT:", "RUN:", "RUT", "RUN", "R.U.T."):
            clean = clean.removeprefix(prefix)
        return clean
    if type in ("email", "url"):
        return "".join(value.split()).casefold()
    no_accents = unicodedata.normalize("NFKD", value)
    no_accents = "".join(c for c in no_accents if not unicodedata.combining(c))
    return " ".join(no_accents.casefold().split())
