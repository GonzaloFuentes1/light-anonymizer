"""Development engine: follows the engine contract with simple, fast detection.

It only reads the text layer of PDFs (RUT, e-mail and Chilean phone patterns, URLs, plus the
name list) and does no OCR and no face detection: images are opened and shown, but they get no
findings. Export applies real redaction and removes metadata, so the whole flow of the app can
be tried end to end with invented documents. It honours the detection groups that apply to it
(the name list, personal and other URLs, D12) and records them like the real engine.

Set ``ANONYMIZER_FAKE_DELAY`` (seconds per page) to slow analysis down and see progress bars.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import time
import unicodedata
import uuid
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageOps, ImageSequence, UnidentifiedImageError

from anonymizer.engine import estimate
from anonymizer.engine.common import (  # noqa: F401  (re-exported for older imports)
    IMAGE_FORMATS,
    IMAGE_SAVE_OPTIONS,
    SCANNED_MAX_CHARS,
    Cancelled,
    FileError,
    StageClock,
    bbox_of,
    now_iso,
    publish,
    sniff,
    stage,
    waiting_for,
)
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import (
    ERROR_MESSAGES,
    TYPE_LABELS,
    AnalyzedFile,
    DetectionOptions,
    ExportResult,
    Finding,
    HistoryEntry,
    Leak,
    PageInfo,
)
from anonymizer.engine.patterns import PERSONAL_URL, URL

RUT = re.compile(r"(?<![\d.])\d{1,2}\.?\d{3}\.?\d{3}\s*-\s*[\dkK](?![\w])")
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")
# Digits with spaces, parentheses, dashes or a leading +: validated as a Chilean number below.
PHONE_CANDIDATE = re.compile(r"(?<![\w.])\+?\(?\d[\d \t()-]{6,18}\d(?![\w])")
DOUBTFUL_RUT = "El dígito verificador no coincide: revisa el original"


def rut_check_digit(body: str) -> str:
    total, factor = 0, 2
    for digit in reversed(body):
        total += int(digit) * factor
        factor = 2 if factor == 7 else factor + 1
    rest = 11 - total % 11
    return {11: "0", 10: "K"}.get(rest, str(rest))


def rut_is_valid(text: str) -> bool:
    clean = re.sub(r"[^\dkK]", "", text).upper()
    return len(clean) >= 2 and rut_check_digit(clean[:-1]) == clean[-1]


def is_chilean_phone(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if digits.startswith("0056"):
        digits = digits[4:]
    elif digits.startswith("56") and len(digits) == 11:
        digits = digits[2:]
    return len(digits) == 9 and digits[0] in "23456789"


def fold(text: str) -> str:
    """Lowercase without accents, one character per character (keeps offsets)."""
    out = []
    for c in text:
        base = unicodedata.normalize("NFD", c)[0]
        out.append(base.lower() if len(base.lower()) == 1 else c)
    return "".join(out)


def is_personal_url(url: str, name_list: list[str]) -> bool:
    """Simple D12 rule: a known personal site, or a RUT, e-mail, phone or listed name inside."""
    if PERSONAL_URL.search(url) or RUT.search(url) or "@" in url or is_chilean_phone(url):
        return True
    folded = fold(url.replace("-", " ").replace("_", " "))
    return any(fold(entry).strip() and fold(entry).strip() in folded for entry in name_list)


def find_spans(text: str, name_list: list[str], personal_urls: bool = True) -> list[tuple[str, int, int, str]]:
    """(type, start, end, detector) of every piece of data found in ``text``.

    Every URL is a span (the ones that are not personal become optional findings), except the
    personal ones when ``personal_urls`` is off.
    """
    spans: list[tuple[str, int, int, str]] = []
    for m in RUT.finditer(text):
        spans.append(("rut", m.start(), m.end(), "regex"))
    for m in EMAIL.finditer(text):
        spans.append(("email", m.start(), m.end(), "regex"))
    taken = [(a, b) for _, a, b, _ in spans]
    for m in PHONE_CANDIDATE.finditer(text):
        a, b = m.start(), m.end()
        if any(a < tb and ta < b for ta, tb in taken) or not is_chilean_phone(m.group(0)):
            continue
        spans.append(("phone", a, b, "regex"))
    for m in URL.finditer(text):
        if personal_urls or not is_personal_url(m.group(0), name_list):
            spans.append(("url", m.start(), m.end(), "regex"))
    folded = fold(text)
    for entry in name_list:
        words = fold(entry).split()
        if not words:
            continue
        pattern = re.compile(r"(?<!\w)" + r"[\s,.-]+".join(re.escape(w) for w in words) + r"(?!\w)")
        for m in pattern.finditer(folded):
            spans.append(("name", m.start(), m.end(), "name_list"))
    return sorted(spans, key=lambda s: (s[1], s[2]))


def polygon_of(rect: pymupdf.Rect) -> list[list[float]]:
    r = rect.normalize()
    return [[round(x, 2), round(y, 2)] for x, y in ((r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1))]


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


class FakeEngine:
    name = "fake"

    def __init__(self, delay: float | None = None):
        self.delay = float(os.environ.get("ANONYMIZER_FAKE_DELAY", "0")) if delay is None else delay

    def profile(self, file: AnalyzedFile) -> dict:
        return estimate.profile(file.path, file.kind)

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def analyze(self, file: AnalyzedFile, name_list: list[str], *, progress=None, cancel=None) -> AnalyzedFile:
        started = time.perf_counter()

        def report(fraction: float, step: str) -> None:
            if cancel is not None and cancel.is_set():
                raise Cancelled
            file.progress = max(0.0, min(1.0, fraction))
            file.step = step
            if progress is not None:
                progress(file.progress, step)

        options = DetectionOptions.from_dict(file.options)
        file.options = options.to_dict()
        file.timings = {}
        names = list(name_list) if options.names_list else []
        file.status = "processing"
        file.error = file.error_message = None
        clock = StageClock()
        try:
            report(0.02, "Revisando el archivo")
            kind = sniff(file.path)
            if kind in ("empty", "format"):
                raise FileError(kind)
            file.kind = kind
            with clock.running():
                if kind == "pdf":
                    pages, findings = self._analyze_pdf(file, names, options, report)
                else:
                    pages, findings = self._analyze_image(file, report)
            report(0.98, "Preparando la revisión")
            file.pages = pages
            file.findings = findings
            file.status = "ready"
            file.progress = 1.0
            file.step = "Listo para revisar"
        except Cancelled:
            file.status = "cancelled"
            file.step = "Cancelado"
        except FileError as exc:
            file.status = "error"
            file.error = exc.code
            file.error_message = ERROR_MESSAGES[exc.code]
            file.step = "No se pudo abrir"
        file.timings = {**clock.rounded(), "analyze": round(time.perf_counter() - started, 3)}
        return file

    def _analyze_pdf(self, file: AnalyzedFile, name_list: list[str], options: DetectionOptions, report):
        with PDF_LOCK:
            try:
                doc = pymupdf.open(file.path, filetype="pdf")
            except Exception as exc:
                raise FileError("corrupt", repr(exc)) from exc
            if doc.needs_pass:
                doc.close()
                raise FileError("password")
            count = doc.page_count
        if count == 0:
            doc.close()
            raise FileError("empty")
        pages: list[PageInfo] = []
        findings: list[Finding] = []
        try:
            for n in range(count):
                report(0.05 + 0.9 * n / count, f"Leyendo el documento: página {n + 1} de {count}")
                if self.delay:
                    time.sleep(self.delay)
                with stage("text"), waiting_for(PDF_LOCK):
                    try:
                        page = doc[n]
                        view = page.rect
                        text, boxes = self._chars(page)
                        to_view = page.rotation_matrix
                        scanned = len(text.strip()) < SCANNED_MAX_CHARS
                        pages.append(
                            PageInfo(
                                index=n, width=round(view.width, 2), height=round(view.height, 2), unit="pt",
                                scanned=scanned,
                            )
                        )  # fmt: skip
                    except Exception as exc:
                        raise FileError("corrupt", repr(exc)) from exc
                report(0.05 + 0.9 * (n + 0.5) / count, f"Buscando datos personales: página {n + 1} de {count}")
                with stage("text"):
                    spans = find_spans(text, name_list, personal_urls=options.urls_personal)
                    for type_, a, b, detector in spans:
                        found = text[a:b]
                        doubtful = type_ == "rut" and not rut_is_valid(found)
                        # D12: only a URL that is not personal and covers no other datum is optional.
                        optional = (
                            type_ == "url"
                            and not is_personal_url(found, name_list)
                            and not any(t != "url" and sa < b and a < sb for t, sa, sb, _ in spans)
                        )
                        status = "suggested" if optional and not options.urls_other else "proposed"
                        for rect in self._span_rects(boxes, a, b):
                            view_rect = (rect * to_view).normalize() + (-1, -1, 1, 1)
                            findings.append(
                                Finding(
                                    id=uuid.uuid4().hex[:12],
                                    file_id=file.id,
                                    page=n,
                                    type=type_,
                                    polygon=polygon_of(view_rect),
                                    text=found,
                                    detector=detector,
                                    score=0.6 if doubtful else 1.0,
                                    doubtful=doubtful,
                                    doubt_reason=DOUBTFUL_RUT if doubtful else None,
                                    status=status,
                                    optional=optional,
                                    history=[HistoryEntry(at=now_iso(), action=status)],
                                )
                            )
        finally:
            with PDF_LOCK:
                doc.close()
        return pages, findings

    @staticmethod
    def _chars(page: pymupdf.Page) -> tuple[str, list[pymupdf.Rect | None]]:
        """Text of the page (one line per text line) and the box of each character (page space)."""
        text: list[str] = []
        boxes: list[pymupdf.Rect | None] = []
        for block in page.get_text("rawdict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for ch in span.get("chars", []):
                        text.append(ch["c"])
                        boxes.append(pymupdf.Rect(ch["bbox"]))
                text.append("\n")
                boxes.append(None)
        return "".join(text), boxes

    @staticmethod
    def _span_rects(boxes: list[pymupdf.Rect | None], a: int, b: int) -> list[pymupdf.Rect]:
        """One rectangle per text line covered by characters ``a..b``."""
        rects: list[pymupdf.Rect] = []
        current: pymupdf.Rect | None = None
        for box in boxes[a:b]:
            if box is None:
                if current is not None:
                    rects.append(current)
                current = None
            elif not box.is_empty:
                current = pymupdf.Rect(box) if current is None else current | box
        if current is not None:
            rects.append(current)
        return rects

    def _analyze_image(self, file: AnalyzedFile, report):
        report(0.2, "Leyendo la imagen")
        try:
            with Image.open(file.path) as img:
                img.load()
                pages = []
                for n, frame in enumerate(ImageSequence.Iterator(img)):
                    width, height = ImageOps.exif_transpose(frame.copy()).size
                    pages.append(PageInfo(index=n, width=float(width), height=float(height), unit="px"))
        except UnidentifiedImageError as exc:
            raise FileError("format", repr(exc)) from exc
        except (OSError, SyntaxError, ValueError) as exc:
            raise FileError("corrupt", repr(exc)) from exc
        if self.delay:
            time.sleep(self.delay)
        # No OCR or face detection in the development engine.
        report(0.7, "Buscando datos personales")
        return pages, []

    # ------------------------------------------------------------------
    # render_page
    # ------------------------------------------------------------------

    def render_page(self, file: AnalyzedFile, page: int, zoom: float = 1.0) -> bytes:
        zoom = max(0.05, min(8.0, float(zoom)))
        kind = file.kind or sniff(file.path)
        if kind == "pdf":
            with PDF_LOCK:
                with pymupdf.open(file.path, filetype="pdf") as doc:
                    pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                    return pix.tobytes("png")
        with Image.open(file.path) as img:
            img.seek(page)
            frame = ImageOps.exif_transpose(img.copy())
            if frame.mode not in ("RGB", "RGBA", "L"):
                frame = frame.convert("RGB")
            if zoom != 1.0:
                size = (max(1, round(frame.width * zoom)), max(1, round(frame.height * zoom)))
                frame = frame.resize(size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            frame.save(buf, "PNG")
            return buf.getvalue()

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def export(self, file: AnalyzedFile, dest_dir: str) -> ExportResult:
        started = time.perf_counter()
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        active = [f for f in file.findings if f.active]
        # Left visible on purpose: removed by the reviewer, or a suggestion (D12) not applied.
        kept = [f for f in file.findings if not f.active]
        removed = sum(1 for f in file.findings if f.status == "removed")
        kind = file.kind or sniff(file.path)
        name = Path(file.name).name or "archivo"
        with tempfile.TemporaryDirectory(prefix="anonimizador_export_") as tmp:
            if kind == "pdf":
                if Path(name).suffix.lower() != ".pdf":
                    name = Path(name).stem + ".pdf"
                staged = Path(tmp) / "output.pdf"
                self._export_pdf(file, active, staged)
                leaks = self._leaks_pdf(staged, active, kept)
            else:
                staged, fmt = self._export_image(file, active, Path(tmp))
                if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
                    name = Path(name).stem + IMAGE_FORMATS[fmt]
                leaks = self._leaks_image(staged)
            file.leaks = leaks
            output: Path | None = None
            if not leaks:
                output = publish(staged, dest, name)
        file.output_path = str(output) if output else None
        file.timings["export"] = round(time.perf_counter() - started, 3)
        if leaks:
            n = len(leaks)
            message = (
                f"No se exportó: la verificación encontró {n} {'dato' if n == 1 else 'datos'} "
                "que siguen legibles. Vuelve a Revisar."
            )
        else:
            message = "Archivo exportado. La verificación automática no encontró datos censurados legibles."
        return ExportResult(
            file_id=file.id,
            output_path=file.output_path,
            leaks=leaks,
            redactions_applied=len(active) if not leaks else 0,
            removed_by_reviewer=removed,
            exported=not leaks,
            message=message,
        )

    def _export_pdf(self, file: AnalyzedFile, active: list[Finding], staged: Path) -> None:
        with PDF_LOCK:
            doc = pymupdf.open(file.path, filetype="pdf")
            try:
                by_page: dict[int, list[Finding]] = {}
                for f in active:
                    by_page.setdefault(f.page, []).append(f)
                for n in range(doc.page_count):
                    page = doc[n]
                    to_page = pymupdf.Matrix(page.derotation_matrix)
                    # Zones are placed on the unrotated page (see ``pdf.redact``): MuPDF misplaces them
                    # on rotated pages whose CropBox or MediaBox does not start at (0, 0).
                    rotation = page.rotation
                    if rotation:
                        page.set_rotation(0)
                    for f in by_page.get(n, []):
                        x0, y0, x1, y1 = bbox_of(f.polygon)
                        rect = (pymupdf.Rect(x0, y0, x1, y1) * to_page).normalize()
                        page.add_redact_annot(rect, fill=(0, 0, 0))
                    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
                    if rotation:
                        page.set_rotation(rotation)
                    for annot in list(page.annots() or []):
                        page.delete_annot(annot)
                    for widget in list(page.widgets() or []):
                        page.delete_widget(widget)
                for embedded in list(doc.embfile_names()):
                    doc.embfile_del(embedded)
                doc.set_toc([])
                doc.set_metadata({})
                doc.del_xml_metadata()
                doc.save(staged, garbage=4, deflate=True, clean=True)
            finally:
                doc.close()

    def _export_image(self, file: AnalyzedFile, active: list[Finding], folder: Path) -> tuple[Path, str]:
        with Image.open(file.path) as img:
            img.load()
            fmt = img.format if img.format in IMAGE_FORMATS else "PNG"
            frames = []
            for n, frame in enumerate(ImageSequence.Iterator(img)):
                out = ImageOps.exif_transpose(frame.copy()).convert("RGB")
                draw = ImageDraw.Draw(out)
                for f in active:
                    if f.page == n:
                        draw.polygon([(x, y) for x, y in f.polygon], fill=(0, 0, 0))
                # A new image from raw pixels: no EXIF, XMP, ICC or text chunks are carried over.
                frames.append(Image.frombytes("RGB", out.size, out.tobytes()))
        staged = folder / ("output" + IMAGE_FORMATS[fmt])
        options = IMAGE_SAVE_OPTIONS[fmt]
        if len(frames) > 1 and fmt in ("TIFF", "WEBP", "PNG"):
            frames[0].save(staged, fmt, save_all=True, append_images=frames[1:], **options)
        else:
            frames[0].save(staged, fmt, **options)
        return staged, fmt

    @staticmethod
    def _leaks_pdf(path: Path, active: list[Finding], kept_visible: list[Finding]) -> list[Leak]:
        leaks: list[Leak] = []
        with PDF_LOCK:
            with pymupdf.open(path) as doc:
                texts = [squash(page.get_text("text")) for page in doc]
                metadata = {k: v for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")}
                if metadata or doc.get_xml_metadata():
                    leaks.append(Leak(page=None, type="metadata", message="El archivo todavía tiene metadatos."))
        kept = {(f.page, squash(f.text or "")) for f in kept_visible}
        seen: set[tuple[int, str]] = set()
        for f in active:
            needle = squash(f.text or "")
            key = (f.page, needle)
            if not needle or key in kept or key in seen or f.page >= len(texts):
                continue
            seen.add(key)
            if needle in texts[f.page]:
                label = TYPE_LABELS.get(f.type, f.type)
                leaks.append(
                    Leak(
                        page=f.page,
                        type=f.type,
                        message=f"{label} sigue legible en la página {f.page + 1}.",
                        finding_id=f.id,
                    )
                )
        return leaks

    @staticmethod
    def _leaks_image(path: Path) -> list[Leak]:
        with Image.open(path) as img:
            if len(img.getexif()) or any(k in img.info for k in ("exif", "xmp", "XML:com.adobe.xmp", "comment")):
                return [Leak(page=None, type="metadata", message="La imagen todavía tiene metadatos.")]
        return []
