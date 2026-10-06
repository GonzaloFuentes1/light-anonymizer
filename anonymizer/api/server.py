"""Local HTTP API of the app (FastAPI), bound to 127.0.0.1 and protected by a session token.

Every route lives under ``/api`` and needs the header ``X-Session-Token``; ``GET /`` serves the
UI with that token injected into ``<meta name="session-token">``. Everything is kept in memory
for one process (a :class:`Session`); uploaded files are copied into a per-session temporary
folder and the user's originals are never touched. Errors reach the user as plain Spanish
messages; technical details go only to the rotating log file.

Routes (JSON unless noted)::

    GET    /api/state
    POST   /api/files                      multipart "files" (many)
    POST   /api/files/from-paths           {paths: [str]}   extension: native dialog of the desktop shell
    DELETE /api/files/{id}
    POST   /api/files/{id}/options         {all_text: bool}
    GET    /api/names                      PUT /api/names {entries: [str]}
    GET    /api/exceptions                 PUT /api/exceptions {entries: [str]}   D10: RUTs, phones, 600/800
    GET    /api/options                    PUT /api/options {groups: {key: bool}}   detection groups
    GET    /api/estimate?ids=a,b           seconds per group (default: the files not processed yet)
    POST   /api/process                    {file_ids?: [str]}
    POST   /api/cancel                     {file_id?: str}
    GET    /api/files/{id}                 full AnalyzedFile
    GET    /api/files/{id}/pages/{n}.png?zoom=1.5   image/png
    GET    /api/files/{id}/pages/{n}.png?zoom=1.5&redacted=true  image/png (the page as it will be exported)
    POST   /api/files/{id}/findings        {page, polygon, note?}
    PATCH  /api/files/{id}/findings/{fid}  {action: remove|restore|apply|skip, reason?, note?}
    POST   /api/files/{id}/findings/apply-optional   {reason?: url|exception}   apply the suggested findings
    POST   /api/files/{id}/confirm
    GET    /api/default-export-dir
    POST   /api/export                     {dest_dir, file_ids?, audit_pdf, audit_json}
    GET    /api/about                      name, version, license, source and components

The ``redacted=true`` page (the review's after) is rendered with the export's own per-page
redaction, so it is the page exactly as it will be exported. When that page will be exported as
an image (decided 2026-10-06: something drawn might have stayed under a zone), the response has
the headers ``X-Page-As-Image: 1`` and ``X-Page-As-Image-Reason`` (the Spanish reason,
percent-encoded as UTF-8, ``urllib.parse.quote``); otherwise neither header is sent.

Extensions beyond the base contract: ``POST /api/files/from-paths``; the extra summary fields
``size`` (bytes) and ``leaks`` (count); ``skipped`` in the from-paths answer; and, when the app
is started with a launch key, ``GET /?k=<key>`` sets a session cookie that ``GET /`` requires
before it hands out the token (so other local programs cannot read it).

Detection groups (``model.DETECTION_GROUPS``) live only in the session: every start of the app
goes back to the defaults, so a group turned off once is never off by surprise later. Each file
records the groups it was processed with (``options``) and the time of each stage (``timings``);
those times adjust the estimate (``engine.estimate.CostModel``), whose rates alone are saved.

The name list and the exceptions list (D10) also live only in the session, and apply to the files
processed after they are saved. A finding covered by the exceptions list is not discarded: it
becomes optional and starts unapplied (``engine.exceptions``), like the other URLs of D12.
"""

from __future__ import annotations

import hmac
import logging
import logging.handlers
import math
import mimetypes
import os
import re
import secrets
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path, PurePath, PureWindowsPath
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool
from starlette.exceptions import HTTPException as StarletteHTTPException

from anonymizer import about
from anonymizer.engine import audit, estimate, exceptions
from anonymizer.engine.common import HEAD_BYTES, HEIC_SUFFIXES, is_heic
from anonymizer.engine.model import (
    DETECTION_GROUPS,
    ERROR_MESSAGES,
    GROUPS_BY_KEY,
    AnalyzedFile,
    DetectionOptions,
    ExportResult,
    Finding,
    HistoryEntry,
)
from anonymizer.paths import resource

log = logging.getLogger("anonymizer")

UI_DIR = resource("anonymizer", "ui")  # see anonymizer.paths (development and packaged app)
SESSION_PREFIX = "anonimizador_session_"
SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 1024**3  # 1 GiB per file
MAX_FILES_FROM_PATHS = 1000
# FastAPI's built-in OpenTelemetry, all off: no spans, metrics or log records (validation errors
# with what was sent), and no exporter set up from OTEL_* environment variables.
NO_TELEMETRY = {"tracing": False, "metrics": False, "logs": False, "operation_spans": False, "auto_configure": False}
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})
LAUNCH_COOKIE = "anonimizador_launch"
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    # Nothing may be loaded from or sent to the network: only this local server.
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; "
        "font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'none'"
    ),
}

# Windows may map .js to text/plain in the registry; with nosniff that would break the UI.
for _type, _ext in (
    ("text/javascript", ".js"),
    ("text/javascript", ".mjs"),
    ("text/css", ".css"),
    ("image/svg+xml", ".svg"),
    ("font/woff2", ".woff2"),
    ("font/woff", ".woff"),
    ("font/ttf", ".ttf"),
    ("application/json", ".json"),
):
    mimetypes.add_type(_type, _ext)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def app_data_dir() -> Path:
    """%LOCALAPPDATA%/Anonimizador on Windows, ~/.anonimizador elsewhere."""
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "Anonimizador" if local else Path.home() / ".anonimizador"


def setup_logging(level: int = logging.INFO) -> Path | None:
    """Rotating technical log (never shown to the user). Returns the log file path.

    Falls back to the temporary folder when %LOCALAPPDATA% cannot be written, and to no log at all
    (None) when neither can: the app must still open.
    """
    handler: logging.Handler = logging.NullHandler()
    path = None
    for folder in (app_data_dir() / "logs", Path(tempfile.gettempdir()) / "anonimizador_logs"):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                folder / "anonimizador.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
            )
        except OSError:
            continue
        path = folder / "anonimizador.log"
        break
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
    for name in ("anonymizer", "uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
            logger.addHandler(handler)
        logger.propagate = False
    return path


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(code)
        self.status, self.code, self.message = status, code, message


def not_found() -> ApiError:
    return ApiError(404, "not_found", "No se encontró ese archivo. Puede que ya se haya quitado de la lista.")


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status, headers=SECURITY_HEADERS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sniff_kind(path: Path) -> str | None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
    except OSError:
        return None
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith((b"\xff\xd8\xff", b"\x89PNG", b"II*\x00", b"MM\x00*")) or (
        head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    ):
        return "image"
    return None


def is_heic_file(path: Path) -> bool:
    """A HEIC/HEIF photo, by its extension or by its first bytes (D5: not supported)."""
    if path.suffix.lower() in HEIC_SUFFIXES:
        return True
    try:
        with open(path, "rb") as fh:
            return is_heic(fh.read(HEAD_BYTES))
    except OSError:
        return False


def heic_folder_message(count: int) -> str:
    """Spanish: the HEIC photos of a chosen folder were not added (D5)."""
    if count == 1:
        return (
            "Esta carpeta tiene 1 foto HEIC (por ejemplo de iPhone), que todavía no se puede abrir. "
            "Conviértela a JPG y vuelve a agregarla."
        )
    return (
        f"Esta carpeta tiene {count} fotos HEIC (por ejemplo de iPhone), que todavía no se pueden abrir. "
        "Conviértelas a JPG y vuelve a agregarlas."
    )


def display_name(raw: str | None) -> str:
    """Base name of an uploaded file (browsers may send folder paths), safe to show and to export."""
    name = PureWindowsPath(raw or "").name
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name).strip(" .")
    return name[:200] or "archivo"


def safe_suffix(name: str) -> str:
    suffix = PurePath(name).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,6}", suffix) else ""


def documents_dir() -> Path:
    """The user's Documents folder (follows OneDrive redirection on Windows)."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            # FOLDERID_Documents {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
            guid = GUID(
                0xFDD39AD0, 0x238F, 0x46AF, (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7)
            )
            out = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) == 0:
                path = Path(out.value)
                ctypes.windll.ole32.CoTaskMemFree(out)
                return path
        except Exception:  # noqa: BLE001 - fall back to the home folder
            log.debug("SHGetKnownFolderPath failed", exc_info=True)
    docs = Path.home() / "Documents"
    return docs if docs.is_dir() else Path.home()


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.windll.kernel32.GetLastError() == 5  # access denied: it exists
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_sessions() -> int:
    """Delete working folders left by sessions whose process is gone (e.g. after a crash)."""
    removed = 0
    for folder in Path(tempfile.gettempdir()).glob(SESSION_PREFIX + "*"):
        try:
            pid = int((folder / "pid").read_text().strip())
        except (OSError, ValueError):
            pid = None
        if pid is not None and pid_alive(pid):
            continue
        shutil.rmtree(folder, ignore_errors=True)
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """All the state of one run of the app: files, name list, detection groups, background analyses.

    ``estimates_path``: where the rates of the time estimate are kept between sessions (None:
    only in memory).
    """

    def __init__(self, engine, max_workers: int = 2, estimates_path: Path | None = None):
        self.engine = engine
        self.dir = Path(tempfile.mkdtemp(prefix=SESSION_PREFIX))
        (self.dir / "pid").write_text(str(os.getpid()))
        self.files: dict[str, AnalyzedFile] = {}
        self.sizes: dict[str, int] = {}
        self.sources: dict[str, str] = {}
        self.names: list[str] = []
        self.exceptions: list[str] = []  # D10: values not censored by default (engine.exceptions)
        self.options = DetectionOptions()  # session only: every start of the app uses the defaults
        self.profiles: dict[str, dict] = {}  # file id -> cheap facts for the estimate
        self.costs = estimate.CostModel.load(estimates_path)
        self.lock = threading.RLock()
        self.cancels: dict[str, threading.Event] = {}
        self.pending: set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis")
        self.closed = False

    @property
    def engine_name(self) -> str:
        return getattr(self.engine, "name", type(self.engine).__name__.lower().removesuffix("engine"))

    # -- files -----------------------------------------------------------

    def get(self, file_id: str) -> AnalyzedFile:
        file = self.files.get(file_id)
        if file is None:
            raise not_found()
        return file

    def new_file(self, name: str) -> AnalyzedFile:
        file_id = uuid.uuid4().hex[:12]
        path = self.dir / f"{file_id}{safe_suffix(name)}"
        return AnalyzedFile(id=file_id, name=display_name(name), path=str(path), status="queued", step="Sin procesar")

    def register(self, file: AnalyzedFile, size: int, source: str | None = None) -> None:
        file.kind = sniff_kind(Path(file.path))
        with self.lock:
            self.files[file.id] = file
            self.sizes[file.id] = size
            if source:
                self.sources[file.id] = source
        log.info("file %s registered (%d bytes, kind %s)", file.id, size, file.kind)

    def remove(self, file_id: str) -> None:
        with self.lock:
            file = self.get(file_id)
            event = self.cancels.get(file_id)
            if event:
                event.set()
            del self.files[file_id]
            self.sizes.pop(file_id, None)
            self.sources.pop(file_id, None)
            self.profiles.pop(file_id, None)
        try:
            Path(file.path).unlink(missing_ok=True)
        except OSError:  # still open by a running analysis: removed with the session folder
            log.warning("could not delete working copy of %s yet", file_id)

    def summary(self, file: AnalyzedFile) -> dict:
        findings = list(file.findings)
        active = [f for f in findings if f.active]
        pages = len(file.pages)
        return {
            "id": file.id,
            "name": file.name,
            "kind": file.kind,
            "status": file.status,
            "progress": round(file.progress, 3),
            "step": file.step,
            "error": file.error,
            "error_message": file.error_message,
            "all_text": file.all_text,
            "counts": {
                "total": len(active),
                "doubtful": sum(1 for f in active if f.doubtful),
                "removed": sum(1 for f in findings if f.status == "removed"),
                "added": sum(1 for f in active if f.status == "added"),
                "suggested": sum(1 for f in findings if f.status == "suggested"),
                "suggested_exceptions": sum(
                    1 for f in findings if f.status == "suggested" and f.optional_reason == exceptions.REASON
                ),
            },
            "pages": pages,
            "size": self.sizes.get(file.id, 0),
            "leaks": len(file.leaks),
            "options": dict(file.options) or None,
            "timings": dict(file.timings),
        }

    # -- time estimate ---------------------------------------------------

    def profile(self, file: AnalyzedFile) -> dict:
        """Cheap facts about a file (cached: the working copy never changes)."""
        with self.lock:
            cached = self.profiles.get(file.id)
        if cached is not None:
            return cached
        try:
            profile = getattr(self.engine, "profile", None)
            facts = profile(file) if profile else estimate.profile(file.path, file.kind)
        except Exception:  # noqa: BLE001 - an estimate must never break the app
            log.warning("profile of %s failed", file.id, exc_info=True)
            facts = estimate.empty_profile(file.kind)
        with self.lock:
            if file.id in self.files:
                self.profiles[file.id] = facts
        return facts

    def estimate(self, files: list[AnalyzedFile]) -> dict:
        """Estimated seconds per detection group, per file and in total with the current options."""
        options = self.options
        stages = dict.fromkeys(estimate.STAGES, 0.0)
        rows = []
        for file in files:
            own = self.costs.stages(self.profile(file))
            for name, value in own.items():
                stages[name] += value
            rows.append(
                {
                    "id": file.id,
                    "groups": _rounded(estimate.by_group(own)),
                    "render": round(own["render"], 2),
                    # Same rounding as the groups, so a total is never shown below one of its parts.
                    "total": round(estimate.total(own, options), 2),
                }
            )
        return {
            "files": rows,
            "groups": _rounded(estimate.by_group(stages)),
            # Rendering pages and decoding images: shared by OCR, faces and QR, counted when one is on.
            "render": round(stages["render"], 2),
            "total": round(estimate.total(stages, options), 2),
            "calibrated": self.costs.calibrated,
        }

    # -- analysis --------------------------------------------------------

    def process(self, file_ids: list[str] | None) -> list[str]:
        started = []
        with self.lock:
            if file_ids is None:
                targets = [f for f in self.files.values() if f.status in ("queued", "cancelled")]
            else:
                targets = [self.get(i) for i in file_ids]
            for file in targets:
                if file.id in self.pending:
                    continue
                file.status, file.progress, file.step = "queued", 0.0, "En espera"
                file.error = file.error_message = None
                file.findings, file.pages, file.leaks, file.output_path = [], [], [], None
                file.options, file.timings = self.options.to_dict(), {}
                event = threading.Event()
                self.cancels[file.id] = event
                self.pending.add(file.id)
                self.executor.submit(self._run, file, event)
                started.append(file.id)
        return started

    def _run(self, file: AnalyzedFile, event: threading.Event) -> None:
        try:
            if event.is_set() or self.closed:
                file.status, file.step = "cancelled", "Cancelado"
                return
            file.status, file.step = "processing", "Comenzando"
            names = list(self.names)
            file.exceptions = list(self.exceptions)  # applied by the engine before the file is ready

            def progress(fraction: float, step: str) -> None:
                file.progress, file.step = fraction, step

            log.info("analysis of %s started", file.id)
            result = self.engine.analyze(file, names, progress=progress, cancel=event)
            if result is not file:
                with self.lock:
                    if self.files.get(file.id) is file:
                        self.files[file.id] = result
            log.info("analysis of %s finished: %s %s", file.id, result.status, result.error or "")
            if result.status == "ready":
                self.learn(result)
        except Exception:
            log.exception("analysis of %s failed", file.id)
            file.status, file.error, file.error_message = "error", "internal", ERROR_MESSAGES["internal"]
            file.step = "No se pudo procesar"
        finally:
            with self.lock:
                self.pending.discard(file.id)

    def learn(self, file: AnalyzedFile) -> None:
        """Adjusts the time estimate with the stages measured in this analysis (only numbers are kept)."""
        try:
            self.costs.update(self.profile(file), file.timings)
        except Exception:  # noqa: BLE001 - the estimate is a convenience: never fail an analysis for it
            log.warning("could not update the time estimate", exc_info=True)

    def cancel(self, file_id: str | None) -> None:
        with self.lock:
            ids = [file_id] if file_id else list(self.pending)
            for i in ids:
                file = self.get(i)
                event = self.cancels.get(i)
                if event and i in self.pending:
                    event.set()
                    if file.status == "queued":
                        file.status, file.step = "cancelled", "Cancelado"

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for event in self.cancels.values():
            event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(self.dir, ignore_errors=True)
        log.info("session closed, working folder deleted")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class OptionsBody(BaseModel):
    all_text: bool


class DetectionBody(BaseModel):
    groups: dict[str, StrictBool] = Field(default_factory=dict, max_length=50)


class NamesBody(BaseModel):
    entries: list[str] = Field(default_factory=list, max_length=20000)


class ExceptionsBody(BaseModel):
    entries: list[str] = Field(default_factory=list, max_length=2000)


class ApplyOptionalBody(BaseModel):
    reason: Literal["url", "exception"] | None = None  # None: every suggested finding


class ProcessBody(BaseModel):
    file_ids: list[str] | None = None


class CancelBody(BaseModel):
    file_id: str | None = None


class FindingBody(BaseModel):
    page: int
    polygon: list[list[float]]
    note: str | None = Field(default=None, max_length=2000)


class FindingPatch(BaseModel):
    action: str
    reason: str | None = Field(default=None, max_length=300)
    note: str | None = Field(default=None, max_length=2000)


class ExportBody(BaseModel):
    dest_dir: str
    file_ids: list[str] | None = None
    audit_pdf: bool = True
    audit_json: bool = True


class PathsBody(BaseModel):
    paths: list[str] = Field(max_length=MAX_FILES_FROM_PATHS)


def _rounded(values: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 2) for key, value in values.items()}


def detection_groups(options: DetectionOptions) -> dict:
    """The detection groups as the UI shows them (Spanish texts), with their current state."""
    return {
        "groups": [
            {
                "key": g.key,
                "label": g.label,
                "description": g.description,
                "warning": g.warning,
                "short": g.short,
                "default": g.default,
                "locked": g.locked,
                "locked_reason": g.locked_reason,
                "detection": g.detection,
                "enabled": getattr(options, g.key),
            }
            for g in DETECTION_GROUPS
        ]
    }


def clean_entries(entries: list[str]) -> list[str]:
    out, seen = [], set()
    for entry in entries:
        value = re.sub(r"\s+", " ", str(entry)).strip()[:200]
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            out.append(value)
    return out


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _host_of(header: str) -> str:
    header = header.strip().lower()
    if header.startswith("["):
        return header.split("]")[0] + "]"
    return header.rsplit(":", 1)[0] if ":" in header else header


def create_app(
    engine=None,
    token: str | None = None,
    *,
    launch_key: str | None = None,
    ui_dir: Path | None = None,
    allowed_hosts: frozenset[str] = ALLOWED_HOSTS,
    estimates_path: Path | None = None,
) -> FastAPI:
    """Build the API. ``engine`` defaults to :func:`anonymizer.engine.get_engine`.

    With ``launch_key``, ``GET /`` only reveals the token to a client that first opened
    ``/?k=<launch_key>`` (the desktop window or the URL printed by ``--browser``).
    ``estimates_path``: file for the rates of the time estimate (None: kept only in memory).
    """
    if engine is None:
        from anonymizer.engine import get_engine

        engine = get_engine()
    token = token or secrets.token_urlsafe(32)
    ui = Path(ui_dir) if ui_dir else UI_DIR
    session = Session(engine, estimates_path=estimates_path)
    cookie_secret = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        session.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan, telemetry=NO_TELEMETRY)
    app.state.session = session
    app.state.token = token

    # -- middleware and errors -------------------------------------------

    async def refuse(request: Request, status: int, code: str, message: str) -> JSONResponse:
        # A small body is read before answering: closing with it unread makes Windows reset the
        # connection, and the client then sees a network error instead of the refusal.
        size = request.headers.get("content-length", "")
        if request.method not in ("GET", "HEAD") and size.isdigit() and int(size) <= 65536:
            await request.body()
        return error_response(status, code, message)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if _host_of(request.headers.get("host", "")) not in allowed_hosts:
            return await refuse(request, 400, "host", "Solicitud no permitida.")
        origin = request.headers.get("origin")
        if origin and origin != "null" and _host_of(re.sub(r"^[a-z]+://", "", origin)) not in allowed_hosts:
            return await refuse(request, 403, "origin", "Solicitud no permitida.")
        if request.url.path.startswith("/api"):
            sent = request.headers.get("x-session-token", "")
            if not hmac.compare_digest(sent.encode(), token.encode()):
                return await refuse(
                    request, 401, "unauthorized", "La sesión no es válida. Cierra y vuelve a abrir la aplicación."
                )
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ApiError)
    async def api_error(_request: Request, exc: ApiError):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid(_request: Request, exc: RequestValidationError):
        # Only the kind of error and the field: never the value sent (a note, a list of names).
        log.info("invalid request: %s", [(error.get("type"), error.get("loc")) for error in exc.errors()[:3]])
        return error_response(422, "invalid", "Los datos enviados no son válidos.")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return error_response(404, "not_found", "No se encontró lo que se pidió.")
        if exc.status_code == 405:
            return error_response(405, "method", "Esa acción no está disponible.")
        return error_response(exc.status_code, "http", "No se pudo completar la solicitud.")

    @app.exception_handler(Exception)
    async def unexpected(_request: Request, exc: Exception):
        log.error("unexpected error", exc_info=exc)
        return error_response(
            500, "internal", "Ocurrió un problema inesperado. El detalle quedó en el registro técnico."
        )

    # -- UI -------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index(request: Request, k: str | None = None):
        if launch_key:
            if k is not None:
                if not hmac.compare_digest(k.encode(), launch_key.encode()):
                    return HTMLResponse(_plain_page("Este enlace no es válido."), status_code=403)
                redirect = RedirectResponse("/", status_code=303)
                redirect.set_cookie(LAUNCH_COOKIE, cookie_secret, httponly=True, samesite="strict", path="/")
                return redirect
            cookie = request.cookies.get(LAUNCH_COOKIE, "")
            if not hmac.compare_digest(cookie.encode(), cookie_secret.encode()):
                return HTMLResponse(_plain_page("Abre el Anonimizador desde su acceso directo."), status_code=403)
        page = ui / "index.html"
        if not page.is_file():
            return HTMLResponse(_plain_page("La interfaz todavía no está instalada.", token))
        return HTMLResponse(inject_token(page.read_text(encoding="utf-8"), token))

    # -- state and files ------------------------------------------------

    @app.get("/api/state")
    def state():
        with session.lock:
            files = [session.summary(f) for f in session.files.values()]
        return {
            "files": files,
            "names_count": len(session.names),
            "exceptions_count": len(session.exceptions),
            "engine": session.engine_name,
        }

    @app.post("/api/files")
    def upload(files: Annotated[list[UploadFile], File()]):
        added = []
        for upload_file in files:
            file = session.new_file(upload_file.filename or "archivo")
            size = 0
            try:
                with open(file.path, "wb") as out:
                    while chunk := upload_file.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise ApiError(413, "too_large", f"«{file.name}» es demasiado grande: el máximo es 1 GB.")
                        out.write(chunk)
            except ApiError:
                Path(file.path).unlink(missing_ok=True)
                raise
            session.register(file, size)
            added.append({"id": file.id, "name": file.name, "status": file.status})
        return {"files": added}

    @app.post("/api/files/from-paths")
    def upload_from_paths(body: PathsBody):
        added, skipped = [], []
        candidates: list[Path] = []
        for raw in body.paths:
            path = Path(raw)
            if path.is_dir():
                found, heic = [], 0
                for p in sorted(path.rglob("*")):
                    if not p.is_file() or p.name.startswith((".", "~$")):
                        continue
                    if p.suffix.lower() in SUPPORTED_SUFFIXES:
                        found.append(p)
                    elif p.suffix.lower() in HEIC_SUFFIXES:
                        heic += 1
                if heic:  # D5: say why the iPhone photos were left out, instead of ignoring them
                    skipped.append({"name": path.name, "message": heic_folder_message(heic)})
                elif not found:
                    skipped.append({"name": path.name, "message": "Esta carpeta no tiene PDF ni imágenes."})
                candidates.extend(found)
            elif path.is_file():
                candidates.append(path)
            else:
                skipped.append({"name": display_name(raw), "message": "No se encontró este archivo."})
        for path in candidates[:MAX_FILES_FROM_PATHS]:
            if is_heic_file(path):  # by its extension or, with another extension, by its content
                skipped.append({"name": display_name(path.name), "message": ERROR_MESSAGES["heic"]})
                continue
            file = session.new_file(path.name)
            try:
                size = path.stat().st_size
                if size > MAX_UPLOAD_BYTES:
                    skipped.append(
                        {"name": file.name, "message": "Este archivo es demasiado grande: el máximo es 1 GB."}
                    )
                    continue
                shutil.copyfile(path, file.path)
            except OSError:
                log.warning("could not copy a selected file", exc_info=True)
                skipped.append(
                    {"name": file.name, "message": "No se pudo leer este archivo. Revisa que no esté abierto."}
                )
                continue
            session.register(file, size, source=str(path))
            added.append({"id": file.id, "name": file.name, "status": file.status})
        if len(candidates) > MAX_FILES_FROM_PATHS:
            skipped.append({"name": "", "message": f"Se agregaron solo los primeros {MAX_FILES_FROM_PATHS} archivos."})
        return {"files": added, "skipped": skipped}

    @app.delete("/api/files/{file_id}")
    def delete_file(file_id: str):
        session.remove(file_id)
        return {"ok": True}

    @app.post("/api/files/{file_id}/options")
    def options(file_id: str, body: OptionsBody):
        file = session.get(file_id)
        file.all_text = body.all_text
        return session.summary(file)

    @app.get("/api/names")
    def get_names():
        return {"entries": list(session.names)}

    @app.put("/api/names")
    def put_names(body: NamesBody):
        session.names = clean_entries(body.entries)
        return {"entries": list(session.names)}

    @app.get("/api/exceptions")
    def get_exceptions():
        return {"entries": list(session.exceptions)}

    @app.put("/api/exceptions")
    def put_exceptions(body: ExceptionsBody):
        kept, invalid = exceptions.clean(body.entries)
        if invalid:  # the list is not changed: the dialog stays open with the line to fix
            raise ApiError(
                422,
                "invalid_exception",
                f"«{invalid[0]}» no es un RUT, un teléfono ni un número 600 u 800. Escribe uno por línea.",
            )
        session.exceptions = kept
        return {"entries": list(session.exceptions)}

    @app.get("/api/options")
    def get_detection_options():
        with session.lock:
            return detection_groups(session.options)

    @app.put("/api/options")
    def put_detection_options(body: DetectionBody):
        for key, value in body.groups.items():
            group = GROUPS_BY_KEY.get(key)
            if group is None:
                raise ApiError(422, "unknown_group", "Esa detección no existe.")
            if group.locked and not value:
                raise ApiError(400, "locked", f"«{group.label}» no se puede apagar. {group.locked_reason}")
        with session.lock:
            session.options = session.options.replace(**body.groups)
            log.info("detection groups: %s", session.options.to_dict())
            return detection_groups(session.options)

    @app.get("/api/estimate")
    def get_estimate(ids: str | None = None):
        with session.lock:
            if ids is None:
                targets = [f for f in session.files.values() if f.status in ("queued", "cancelled")]
            else:  # unknown ids are skipped: the UI may still list a file that was just removed
                targets = [session.files[i] for i in dict.fromkeys(ids.split(",")) if i in session.files]
        return session.estimate(targets)

    @app.get("/api/about")
    def get_about():
        return about.info()

    @app.post("/api/process")
    def process(body: ProcessBody | None = None):
        return {"started": session.process(body.file_ids if body else None)}

    @app.post("/api/cancel")
    def cancel(body: CancelBody | None = None):
        session.cancel(body.file_id if body else None)
        return {"ok": True}

    @app.get("/api/files/{file_id}")
    def get_file(file_id: str):
        file = session.get(file_id)
        with session.lock:
            return file.to_dict()

    @app.get("/api/files/{file_id}/pages/{page}.png")
    def page_png(file_id: str, page: int, zoom: float = 1.0, redacted: bool = False):
        if redacted:
            # One critical section: a re-process that starts in between empties the findings under
            # this lock, so the after can never be rendered with nothing applied.
            with session.lock:
                file = session.get(file_id)
                check_render(file, page, zoom)
                if file.status not in ("ready", "confirmed", "exported"):
                    raise ApiError(409, "not_ready", "Este archivo todavía no está listo para revisar.")
                snapshot = [
                    replace(f, text=None, history=[], polygon=[list(p) for p in f.polygon])
                    for f in file.findings
                    if f.page == page
                ]
            info: dict = {}
            render = lambda: session.engine.render_result(  # noqa: E731
                file, page, min(max(zoom, 0.05), 8.0), snapshot, info=info
            )
        else:
            info = {}
            file = session.get(file_id)
            check_render(file, page, zoom)
            render = lambda: session.engine.render_page(file, page, min(max(zoom, 0.05), 8.0))  # noqa: E731
        try:
            png = render()
        except Exception:
            log.exception("render of %s page %d failed (redacted=%s)", file_id, page, redacted)
            raise ApiError(409, "render", "No se pudo mostrar esta página.") from None
        headers = {}
        if info.get("as_image"):
            headers = {"X-Page-As-Image": "1", "X-Page-As-Image-Reason": quote(info.get("reason") or "", safe="")}
        return Response(png, media_type="image/png", headers=headers)

    def check_render(file: AnalyzedFile, page: int, zoom: float) -> None:
        if not math.isfinite(zoom) or zoom <= 0:
            raise ApiError(422, "invalid", "El zoom no es válido.")
        if file.kind is None or file.status == "error":
            raise ApiError(409, "not_viewable", "Este archivo no se puede mostrar.")
        if page < 0 or (file.pages and page >= len(file.pages)):
            raise ApiError(404, "not_found", "Esa página no existe.")

    # -- review ---------------------------------------------------------

    def editable(file: AnalyzedFile) -> None:
        # An exported file can still be corrected: it goes back to "ready" and must be confirmed
        # and exported again (the new export never overwrites the previous one).
        if file.status not in ("ready", "confirmed", "exported"):
            raise ApiError(409, "not_ready", "Este archivo todavía no está listo para revisar.")

    def reopen(file: AnalyzedFile) -> None:
        """A change to a confirmed or exported file needs a new confirmation."""
        if file.status in ("confirmed", "exported"):
            file.status, file.step = "ready", "Listo para revisar"

    @app.post("/api/files/{file_id}/findings")
    def add_finding(file_id: str, body: FindingBody):
        file = session.get(file_id)
        with session.lock:
            editable(file)
            if not 0 <= body.page < len(file.pages):
                raise ApiError(422, "invalid", "Esa página no existe.")
            info = file.pages[body.page]
            points = []
            for point in body.polygon:
                if len(point) != 2 or not all(math.isfinite(v) for v in point):
                    raise ApiError(422, "invalid", "La zona dibujada no es válida.")
                x = min(max(point[0], 0.0), info.width)
                y = min(max(point[1], 0.0), info.height)
                points.append([round(x, 2), round(y, 2)])
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            if len(points) < 3 or max(xs) - min(xs) < 1 or max(ys) - min(ys) < 1:
                raise ApiError(422, "invalid", "La zona es demasiado pequeña. Dibújala de nuevo.")
            note = (body.note or "").strip() or None
            finding = Finding(
                id="m" + uuid.uuid4().hex[:11],
                file_id=file.id,
                page=body.page,
                type="manual",
                polygon=points,
                detector="reviewer",
                status="added",
                history=[HistoryEntry(at=now_iso(), action="added", note=note)],
            )
            file.findings.append(finding)
            reopen(file)
            return asdict(finding)

    @app.patch("/api/files/{file_id}/findings/{finding_id}")
    def patch_finding(file_id: str, finding_id: str, body: FindingPatch):
        file = session.get(file_id)
        with session.lock:
            editable(file)
            finding = next((f for f in file.findings if f.id == finding_id), None)
            if finding is None:
                raise ApiError(404, "not_found", "No se encontró esa censura.")
            reason = (body.reason or "").strip() or None
            note = (body.note or "").strip() or None
            # D12 (other URLs) and D10 (exceptions list): applied or skipped, never "removed".
            what = "este dato" if finding.optional_reason == exceptions.REASON else "este enlace"
            if body.action == "remove":
                if finding.optional:
                    raise ApiError(409, "optional", f"Para dejar visible {what}, usa «No censurar».")
                if not finding.active:
                    raise ApiError(409, "already_removed", "Esta censura ya estaba quitada.")
                finding.status = "removed"
                finding.history.append(HistoryEntry(at=now_iso(), action="removed", reason=reason, note=note))
            elif body.action == "restore":
                if finding.status != "removed":
                    raise ApiError(409, "not_removed", "Esta censura no estaba quitada.")
                added = finding.detector == "reviewer" or (finding.history and finding.history[0].action == "added")
                finding.status = "added" if added else "proposed"
                finding.history.append(HistoryEntry(at=now_iso(), action="restored", reason=reason, note=note))
            elif body.action == "apply":
                if not finding.optional:
                    raise ApiError(409, "not_optional", "Esta censura no es opcional.")
                if finding.status != "suggested":
                    raise ApiError(409, "not_suggested", f"{what.capitalize()} ya está censurado.")
                finding.status = "proposed"
                finding.history.append(HistoryEntry(at=now_iso(), action="applied", reason=reason, note=note))
            elif body.action == "skip":
                if not finding.optional:
                    raise ApiError(409, "not_optional", "Esta censura no es opcional: se quita con «Quitar».")
                if finding.status != "proposed":
                    raise ApiError(409, "not_applied", f"{what.capitalize()} ya estaba sin censurar.")
                finding.status = "suggested"
                finding.history.append(HistoryEntry(at=now_iso(), action="skipped", reason=reason, note=note))
            else:
                raise ApiError(422, "invalid", "Esa acción no existe.")
            reopen(file)
            return asdict(finding)

    @app.post("/api/files/{file_id}/findings/apply-optional")
    def apply_optional(file_id: str, body: ApplyOptionalBody | None = None):
        """Applies the suggested findings of the file: every one, or those of one ``reason``
        ("url", D12: "Censurar todos los otros enlaces"; "exception", D10: "Censurar todas las
        excepciones")."""
        file = session.get(file_id)
        only = body.reason if body else None
        with session.lock:
            editable(file)
            at = now_iso()
            applied = [
                f
                for f in file.findings
                if f.status == "suggested" and (only is None or (f.optional_reason or "url") == only)
            ]
            for f in applied:
                f.status = "proposed"
                f.history.append(HistoryEntry(at=at, action="applied"))
            if applied:
                reopen(file)
            return {"applied": [asdict(f) for f in applied], "file": session.summary(file)}

    @app.post("/api/files/{file_id}/confirm")
    def confirm(file_id: str):
        file = session.get(file_id)
        with session.lock:
            if file.status != "ready":
                raise ApiError(409, "not_ready", "Solo se puede confirmar un archivo que está listo para revisar.")
            file.status, file.step = "confirmed", "Revisión confirmada"
            return session.summary(file)

    # -- export ---------------------------------------------------------

    @app.get("/api/default-export-dir")
    def default_export_dir():
        return {"path": str(documents_dir() / "Anonimizados")}

    @app.post("/api/export")
    def export(body: ExportBody):
        with session.lock:
            if body.file_ids is None:
                targets = [f for f in session.files.values() if f.status == "confirmed"]
            else:
                targets = [session.get(i) for i in body.file_ids]
                targets = [f for f in targets if f.status == "confirmed"]
        if not targets:
            raise ApiError(409, "nothing_to_export", "No hay archivos confirmados para exportar.")
        dest = Path(body.dest_dir.strip()).expanduser() if body.dest_dir.strip() else None
        if dest is None or not dest.is_absolute():
            raise ApiError(422, "invalid_dest", "Elige una carpeta de destino.")
        try:
            dest = dest.resolve()
            if dest == session.dir or session.dir in dest.parents:
                raise ApiError(422, "invalid_dest", "Elige otra carpeta de destino.")
            dest.mkdir(parents=True, exist_ok=True)
            # One attempt: on Windows, tempfile retries for ever when the folder denies writing
            # (a read-only share, "Controlled folder access" guarding Documents).
            probe = dest / f".anonimizador_{secrets.token_hex(8)}"
            os.close(os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            probe.unlink()
        except ApiError:
            raise
        except OSError:
            log.warning("export folder not writable", exc_info=True)
            raise ApiError(422, "dest_not_writable", "No se puede guardar en esa carpeta. Elige otra.") from None
        # The engine never overwrites an existing file (it adds " (2)"), so originals are safe
        # even when the destination is the folder they came from.
        results: list[ExportResult] = []
        for file in targets:
            try:
                result = session.engine.export(file, str(dest))
            except Exception:
                log.exception("export of %s failed", file.id)
                result = ExportResult(
                    file_id=file.id,
                    output_path=None,
                    leaks=[],
                    redactions_applied=0,
                    removed_by_reviewer=sum(1 for f in file.findings if f.status == "removed"),
                    exported=False,
                    message="No se pudo exportar este archivo. El detalle quedó en el registro técnico.",
                )
            else:
                note = audit.not_searched(file) if result.exported else None
                if note:  # "no leaks" must not read as "nothing left": say what was not searched
                    result = replace(result, message=f"{result.message} {note}")
                with session.lock:
                    # A review change made during the export reopened the file ("ready"): the change
                    # is not in this output, so the file must be confirmed and exported again.
                    if file.status != "confirmed":
                        pass
                    elif result.exported:
                        file.status, file.step = "exported", "Exportado"
                    elif result.leaks:
                        file.status, file.step = "ready", "Tiene datos que siguen legibles: vuelve a revisar"
            log.info("export of %s: exported=%s leaks=%d", file.id, result.exported, len(result.leaks))
            results.append(result)
        audit_paths = {"json_path": None, "pdf_path": None}
        if body.audit_json or body.audit_pdf:
            try:
                json_path, pdf_path = audit.write_audit(
                    targets, results, str(dest), write_json=body.audit_json, write_pdf=body.audit_pdf
                )
                audit_paths = {"json_path": json_path, "pdf_path": pdf_path}
            except Exception:
                log.exception("audit report failed")
                audit_paths["message"] = (
                    "No se pudo crear el informe de auditoría. El detalle quedó en el registro técnico."
                )
        return {"results": [asdict(r) for r in results], "audit": audit_paths}

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    def api_not_found(rest: str):
        raise ApiError(404, "not_found", "No se encontró lo que se pidió.")

    app.mount("/", StaticFiles(directory=ui, check_dir=False), name="ui")
    return app


TOKEN_META = re.compile(r"<meta\s+name=[\"']session-token[\"'][^>]*>", re.IGNORECASE)


def inject_token(page: str, token: str) -> str:
    tag = f'<meta name="session-token" content="{token}">'
    if TOKEN_META.search(page):
        return TOKEN_META.sub(lambda _m: tag, page, count=1)
    head = re.search(r"<head[^>]*>", page, re.IGNORECASE)
    if head:
        return page[: head.end()] + tag + page[head.end() :]
    return tag + page


def _plain_page(message: str, token: str | None = None) -> str:
    meta = f'<meta name="session-token" content="{token}">' if token else ""
    return (
        f'<!doctype html><html lang="es-CL"><head><meta charset="utf-8">{meta}<title>Anonimizador</title></head>'
        f'<body style="font-family:Segoe UI,system-ui,sans-serif;padding:40px;color:#17212D">'
        f"<h1>Anonimizador</h1><p>{message}</p></body></html>"
    )
