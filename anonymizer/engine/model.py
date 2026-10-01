"""Data model shared by the engine, the API and the UI.

Coordinates of a finding are stored in *view space*: the page as the user sees it (PDF pages
after /Rotate, images after EXIF orientation), in points for PDFs (1/72 inch) and pixels for
images. The engine converts to its internal page space when it applies redactions. The UI only
needs to scale view space to its own zoom level.

User-facing strings (``step``, ``error_message``, ``doubt_reason``) are Spanish: the app is
used by Chilean public officials.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FindingType = Literal["rut", "email", "phone", "url", "name", "address", "face", "signature", "qr", "text", "manual"]
FindingStatus = Literal["proposed", "removed", "added"]
FileStatus = Literal["queued", "processing", "ready", "confirmed", "error", "exported", "cancelled"]

# Spanish labels for the UI and the audit report.
TYPE_LABELS: dict[str, str] = {
    "rut": "RUT",
    "email": "Correo",
    "phone": "Teléfono",
    "url": "Enlace",
    "name": "Nombre",
    "address": "Dirección",
    "face": "Rostro",
    "signature": "Firma",
    "qr": "Código QR",
    "text": "Texto en imagen",
    "manual": "Agregada",
}


@dataclass
class HistoryEntry:
    at: str  # ISO 8601 timestamp
    action: Literal["proposed", "removed", "restored", "added"]
    reason: str | None = None  # Spanish, chosen by the reviewer
    note: str | None = None  # free text from the reviewer


@dataclass
class Finding:
    id: str
    file_id: str
    page: int  # 0-based
    type: str  # one of FindingType
    polygon: list[list[float]]  # view space, at least 3 points
    text: str | None = None
    detector: str = "regex"  # regex | name_list | context | ocr | faces | qr | reviewer
    score: float | None = None
    doubtful: bool = False
    doubt_reason: str | None = None  # Spanish, e.g. "El dígito verificador no coincide"
    status: str = "proposed"  # one of FindingStatus
    history: list[HistoryEntry] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.status != "removed"


@dataclass
class PageInfo:
    index: int
    width: float  # view space
    height: float
    unit: Literal["pt", "px"]


@dataclass
class Leak:
    """Something still recoverable from an exported file (see the leak check)."""

    page: int | None
    type: str
    message: str  # Spanish, shown to the user
    finding_id: str | None = None


@dataclass
class AnalyzedFile:
    id: str
    name: str  # original file name, shown to the user
    path: str  # working copy on disk (never the user's original)
    kind: Literal["pdf", "image"] | None = None
    pages: list[PageInfo] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    all_text: bool = False  # "censurar todo el texto de esta imagen"
    status: str = "queued"  # one of FileStatus
    progress: float = 0.0  # 0..1
    step: str = "En espera"  # Spanish, what the engine is doing right now
    error: str | None = None  # code: password | corrupt | empty | format | unsupported | internal
    error_message: str | None = None  # Spanish, plain language
    leaks: list[Leak] = field(default_factory=list)
    output_path: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportResult:
    file_id: str
    output_path: str | None
    leaks: list[Leak]
    redactions_applied: int
    removed_by_reviewer: int
    exported: bool  # False when a leak blocked the export
    message: str  # Spanish


# Plain-language error messages (Spanish) by error code.
ERROR_MESSAGES: dict[str, str] = {
    "password": "Este archivo está protegido con contraseña. Ingresa la contraseña o pide una versión sin contraseña.",
    "corrupt": "Este archivo está dañado y no se puede abrir. Pide una copia nueva a quien lo envió.",
    "empty": "Este archivo está vacío.",
    "format": "Este tipo de archivo no se puede procesar. Usa PDF, JPG, PNG, WEBP o TIFF.",
    "unsupported": "Este archivo no se puede procesar todavía.",
    "internal": "Ocurrió un problema al procesar este archivo. El detalle quedó en el registro técnico.",
}
