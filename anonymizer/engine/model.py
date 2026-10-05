"""Data model shared by the engine, the API and the UI.

Coordinates of a finding are stored in *view space*: the page as the user sees it (PDF pages
after /Rotate, images after EXIF orientation), in points for PDFs (1/72 inch) and pixels for
images. The engine converts to its internal page space when it applies redactions. The UI only
needs to scale view space to its own zoom level.

User-facing strings (``step``, ``error_message``, ``doubt_reason`` and the texts of the
detection groups) are Spanish: the app is used by Chilean public officials.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FindingType = Literal["rut", "email", "phone", "url", "name", "address", "face", "signature", "qr", "text", "manual"]
# "suggested": detected and shown to the reviewer, but not applied unless the reviewer applies it
# (D12: URLs that are not personal). "removed": the reviewer kept it visible.
FindingStatus = Literal["proposed", "suggested", "removed", "added"]
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
    action: Literal["proposed", "suggested", "removed", "restored", "added", "applied", "skipped"]
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
    # D12: every datum it covers is a URL that is not personal. It starts "suggested" (unless the
    # option to redact the other URLs is on) and the reviewer applies it or skips it.
    optional: bool = False
    history: list[HistoryEntry] = field(default_factory=list)

    @property
    def active(self) -> bool:
        """Applied on export: neither kept visible by the reviewer nor a suggestion left unapplied."""
        return self.status not in ("removed", "suggested")


@dataclass
class PageInfo:
    index: int
    width: float  # view space
    height: float
    unit: Literal["pt", "px"]
    scanned: bool = False  # a PDF page without a text layer: everything on it is pixels


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
    # Detection groups it was analyzed with (``DetectionOptions.to_dict``), set when it is processed.
    options: dict[str, bool] = field(default_factory=dict)
    # Seconds: "analyze" (total), its stages ("text", "render", "ocr", "faces", "qr", only those that
    # ran; see ``common.StageClock``) and "export".
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


# ---------------------------------------------------------------------------
# Detection groups: what the user chooses to search for
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionGroup:
    key: str
    label: str  # Spanish, the name of the switch
    description: str  # Spanish, one line under the name
    warning: str  # Spanish, shown under the switch while it is off
    short: str  # Spanish, lowercase, used inside sentences ("no se buscaron: rostros y códigos QR")
    default: bool
    locked: bool = False  # cannot be turned off
    locked_reason: str = ""  # Spanish, shown instead of the switch
    # False for a switch that does not decide what is searched but whether it starts applied.
    detection: bool = True


DETECTION_GROUPS: tuple[DetectionGroup, ...] = (
    DetectionGroup(
        "patterns",
        "RUT, correos y teléfonos",
        "Números de RUT, direcciones de correo y teléfonos escritos en el documento.",
        "",
        "RUT, correos y teléfonos",
        default=True,
        # The leak check runs these patterns again over every output, and decision D2 requires
        # zero leaks of them at the base level: they can never be turned off.
        locked=True,
        locked_reason="Siempre activo: es la base de la verificación de fugas.",
    ),
    DetectionGroup(
        "urls_personal",
        "Enlaces personales",
        "Redes sociales, reuniones, archivos compartidos y enlaces que llevan un RUT, un correo o un nombre "
        "de la lista.",
        "Los enlaces a redes sociales, reuniones y archivos compartidos quedarán visibles.",
        "enlaces personales",
        default=True,
    ),
    DetectionGroup(
        "urls_other",
        "Censurar también los otros enlaces",
        "Sitios institucionales y documentos públicos. Siempre aparecen en la revisión: si esto está "
        "apagado, quedan sin censurar y tú eliges cuáles censurar.",
        "",
        "otros enlaces",
        default=False,
        detection=False,
    ),
    DetectionGroup(
        "names_list",
        "Nombres y direcciones de la lista",
        "Las personas y direcciones de tu lista, aunque estén sin tildes, en mayúsculas o en otro orden.",
        "No se buscarán los nombres ni las direcciones de tu lista.",
        "nombres y direcciones de la lista",
        default=True,
    ),
    DetectionGroup(
        "names_context",
        "Nombres por contexto",
        "Nombres en firmas, tablas y campos como «Nombre:». Se marcan como dudosos para que los revises primero.",
        "Los nombres que no están en tu lista (en firmas, tablas o «Nombre:») quedarán visibles.",
        "nombres por contexto",
        default=True,
    ),
    DetectionGroup(
        "ocr",
        "Texto en imágenes y escaneos (OCR)",
        "Lee el texto de las páginas escaneadas, las fotos y las imágenes dentro de los PDF. Es lo que más tarda.",
        "No se leerá el texto de los escaneos ni de las fotos: los RUT, correos, teléfonos, nombres y "
        "enlaces que estén dentro de una imagen quedarán visibles.",
        "texto en imágenes y escaneos",
        default=True,
    ),
    DetectionGroup(
        "faces",
        "Rostros",
        "Caras de personas en fotos, escaneos e imágenes.",
        "Los rostros quedarán visibles.",
        "rostros",
        default=True,
    ),
    DetectionGroup(
        "qr",
        "Códigos QR",
        "Por ejemplo, el de la cédula de identidad, que guarda el RUN.",
        "Los códigos QR quedarán visibles, y pueden guardar datos personales.",
        "códigos QR",
        default=True,
    ),
)
GROUPS_BY_KEY: dict[str, DetectionGroup] = {g.key: g for g in DETECTION_GROUPS}


@dataclass(frozen=True)
class DetectionOptions:
    """Which detection groups run (one flag per group of ``DETECTION_GROUPS``).

    A group that is off skips its work (with OCR off no page or image is read), it does not just
    hide its results. ``urls_other`` does not change what is searched: it decides whether the
    URLs that are not personal start applied (D12).
    """

    patterns: bool = True
    urls_personal: bool = True
    urls_other: bool = False
    names_list: bool = True
    names_context: bool = True
    ocr: bool = True
    faces: bool = True
    qr: bool = True

    @classmethod
    def everything(cls) -> DetectionOptions:
        """Every group on and the other URLs applied (the test bench: comparable with earlier runs)."""
        return cls(**{g.key: True for g in DETECTION_GROUPS})

    @classmethod
    def from_dict(cls, values: dict[str, bool] | None) -> DetectionOptions:
        """Options from a dict; missing keys take their default, unknown keys are ignored and
        locked groups are always on."""
        values = values or {}
        return cls(**{g.key: True if g.locked else bool(values.get(g.key, g.default)) for g in DETECTION_GROUPS})

    def to_dict(self) -> dict[str, bool]:
        return {g.key: bool(getattr(self, g.key)) for g in DETECTION_GROUPS}

    def replace(self, **changes: bool) -> DetectionOptions:
        return DetectionOptions.from_dict({**self.to_dict(), **changes})

    @property
    def raster(self) -> bool:
        """Some pixels must be read: OCR, faces or QR codes are on."""
        return self.ocr or self.faces or self.qr


# Plain-language error messages (Spanish) by error code.
ERROR_MESSAGES: dict[str, str] = {
    "password": "Este archivo está protegido con contraseña. Ingresa la contraseña o pide una versión sin contraseña.",
    "corrupt": "Este archivo está dañado y no se puede abrir. Pide una copia nueva a quien lo envió.",
    "empty": "Este archivo está vacío.",
    # D5: HEIC/HEIF photos are not supported in version 1.
    "heic": "Las fotos HEIC (por ejemplo de iPhone) todavía no se pueden abrir. Conviértelas a JPG y vuelve a agregarlas.",
    "format": "Este tipo de archivo no se puede procesar. Usa PDF, JPG, PNG, WEBP o TIFF.",
    "unsupported": "Este archivo no se puede procesar todavía.",
    "internal": "Ocurrió un problema al procesar este archivo. El detalle quedó en el registro técnico.",
}
