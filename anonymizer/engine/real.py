"""The real engine: detection from the prototype, split into analyze (detect) and export (apply).

``analyze`` reads the working copy and proposes findings in view space; nothing is modified.
``export`` applies only the active findings (the reviewer's decisions), cleans the file, runs
the leak check over a temporary output and copies it to the destination only when it is clean.

Detectors: patterns (``regex``), the user's list (``name_list``), context rules and the
given-name dictionary (``context``), OCR at 0/90/270° (``ocr``), YuNet faces (``faces``), rules
for handwritten and drawn signatures (``signatures``) and QR codes (``qr``). See the modules of this package for each one. Which groups of detectors run is
decided per file (``AnalyzedFile.options``, see ``model.DetectionOptions``); a group that is off
skips its work. The time of each stage is measured (``AnalyzedFile.timings``).
"""

from __future__ import annotations

import importlib.util
import io
import logging
import tempfile
import time
import traceback
import uuid
from contextlib import nullcontext
from pathlib import Path

from anonymizer.engine import context, estimate, faces, ocr
from anonymizer.engine.common import (
    IMAGE_FORMATS,
    IMAGE_SUFFIXES,
    Cancelled,
    FileError,
    StageClock,
    disable_power_throttling,
    now_iso,
    publish,
    rect_polygon,
    sniff,
    stage,
    waiting_for,
)
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import (
    ERROR_MESSAGES,
    AnalyzedFile,
    DetectionOptions,
    ExportResult,
    Finding,
    HistoryEntry,
    PageInfo,
)
from anonymizer.engine.patterns import norm, normalize_1to1
from anonymizer.engine.text import needle

log = logging.getLogger(__name__)

_STAGE_TEXT = {
    "ocr": "Leyendo texto en imágenes (OCR)",
    "faces": "Buscando rostros",
    "signatures": "Buscando firmas",
    "qr": "Buscando códigos QR",
}
# Where each stage starts inside the share of progress of one page or image.
_STAGE_OFFSET = {"ocr": 0.0, "faces": 0.6, "signatures": 0.85, "qr": 0.9}
# Detectors whose text comes from the text layer of a PDF (checked again in the output).
_TEXT_LAYER_DETECTORS = ("regex", "name_list", "context")
# Types whose value, found again inside a URL, makes that URL personal (D12). A RUT, e-mail or
# phone is recognized inside the URL itself (``text.has_data``).
_PERSONAL_IN_URL = ("name", "address", "signature")


def settle_optional(findings: list[Finding]) -> int:
    """D12 over the whole file: an optional URL that contains a name or address found anywhere in
    the file is not optional (it starts applied). ``pdf.data_in_urls`` does this inside the text
    layer; this also covers what OCR read (a URL on a scanned page with a name signed on another
    page, or on another line of the same photo). Returns how many URLs it changed."""
    values = set()
    for f in findings:
        if f.optional or f.type not in _PERSONAL_IN_URL or not f.text:
            continue
        text = norm(f.text)
        label = context.LABEL_VALUE.match(text)  # an OCR zone is the whole line: "Nombre: Ana Soto"
        values.add(text[label.start(1) :] if label else text)
    needles = [p for p in map(needle, values) if p is not None]
    changed = 0
    for f in findings:
        if not (needles and f.optional and f.text):
            continue
        normalized = normalize_1to1(f.text).replace("_", " ")  # ".../ana_soto": "_" is a word character
        if any(p.search(normalized) for p in needles):
            f.optional = False
            if f.status == "suggested":
                f.status = "proposed"
                f.history = [HistoryEntry(at=f.history[0].at if f.history else now_iso(), action="proposed")]
            changed += 1
    return changed


def missing_requirements() -> list[str]:
    """What the real engine needs and is not installed (empty when it can run)."""
    missing = []
    for module in ("cv2", "numpy", "pymupdf", "PIL"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if not ocr.available():
        missing.append("rapidocr/onnxruntime")
    if not missing:
        from anonymizer.engine import strokes

        if not strokes.self_test():  # private PyMuPDF bindings: a PyMuPDF update could break them
            missing.append("pymupdf (content filter that removes strokes)")
    if not faces.available():
        missing.append(str(faces.YUNET_MODEL))
    else:
        try:
            faces.detector()  # loads it now: an unreadable or damaged model shows up here, not on the first photo
        except Exception:  # noqa: BLE001 - OpenCV raises cv2.error
            log.warning("face model cannot be loaded", exc_info=True)
            missing.append(f"{faces.YUNET_MODEL} (cannot be loaded)")
    return missing


class RealEngine:
    name = "real"

    def __init__(self, ocr_dpi: int | None = None):
        self.ocr_dpi = ocr_dpi
        # finding id -> "text" | "raster": where its text was read (internal, used by the leak check).
        self._sources: dict[str, str] = {}
        # OCR and faces run in background threads: without this, Windows slows them down 5-6x
        # whenever the app has no foreground window.
        disable_power_throttling()

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def profile(self, file: AnalyzedFile) -> dict:
        """Cheap facts about the file for the time estimate (``estimate.profile``)."""
        return estimate.profile(file.path, file.kind)

    def analyze(self, file: AnalyzedFile, name_list: list[str], *, progress=None, cancel=None) -> AnalyzedFile:
        """Findings of ``file`` with the detection groups of ``file.options`` (defaults when empty)."""
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
        # With the list off its entries are not searched at all (context names may still be found).
        names = tuple(dict.fromkeys(n.strip() for n in name_list if n and n.strip())) if options.names_list else ()
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
                    pages, findings = self._analyze_image(file, names, options, report)
            report(0.98, "Preparando la revisión")
            settle_optional(findings)
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
            log.info("file %s rejected: %s", file.id, exc.code)
        except Exception:  # noqa: BLE001 - reported to the user as an internal problem, detail in the log
            log.error("analysis of %s failed:\n%s", file.id, traceback.format_exc())
            file.status = "error"
            file.error = "internal"
            file.error_message = ERROR_MESSAGES["internal"]
            file.step = "No se pudo procesar"
        file.timings = {**clock.rounded(), "analyze": round(time.perf_counter() - started, 3)}
        return file

    def _finding(self, file: AnalyzedFile, options: DetectionOptions, page: int, zone, polygon, source: str):
        # D12: a URL that is not personal is shown but starts unapplied, unless the user asked to
        # redact the other URLs too.
        status = "suggested" if zone.optional and not options.urls_other else "proposed"
        finding = Finding(
            id=uuid.uuid4().hex[:12],
            file_id=file.id,
            page=page,
            type=zone.type,
            polygon=polygon,
            text=zone.text or None,
            detector=zone.detector,
            score=None if zone.score is None else round(float(zone.score), 4),
            doubtful=zone.doubt is not None,
            doubt_reason=zone.doubt,
            status=status,
            optional=bool(zone.optional),
            history=[HistoryEntry(at=now_iso(), action=status)],
        )
        self._sources[finding.id] = source
        return finding

    def _analyze_pdf(self, file: AnalyzedFile, names: tuple[str, ...], options: DetectionOptions, report):
        import pymupdf

        from anonymizer.engine import pdf

        with waiting_for(PDF_LOCK):
            doc = pdf.open_pdf(file.path)
        try:
            with stage("text"):
                with waiting_for(PDF_LOCK):
                    pdf.reveal_layers(doc)
                    count = doc.page_count
                pages: list[PageInfo] = []
                to_view: list[pymupdf.Matrix] = []
                text_pages = []
                for n in range(count):
                    report(0.05 + 0.15 * n / count, f"Leyendo el texto de la página {n + 1} de {count}")
                    with waiting_for(PDF_LOCK):
                        try:
                            page = doc[n]
                            view = page.rect
                            tp = pdf.read_text_page(page, names, options)
                            pages.append(
                                PageInfo(
                                    index=n,
                                    width=round(view.width, 3),
                                    height=round(view.height, 3),
                                    unit="pt",
                                    scanned=tp.scanned,
                                )
                            )
                            to_view.append(pymupdf.Matrix(page.rotation_matrix))
                            text_pages.append(tp)
                        except (RuntimeError, ValueError) as exc:
                            raise FileError("corrupt", repr(exc)) from exc
                report(0.2, "Buscando los mismos datos en todo el documento")
                pdf.propagate(text_pages)
                pdf.data_in_urls(text_pages)
                zones = [z for tp in text_pages for z in pdf.text_zones(tp, names)]
            if options.signatures:  # signatures drawn as vector paths on the pages with text
                with stage("signatures"):
                    for tp in text_pages:
                        if tp.scanned:
                            continue
                        report(0.2, f"Buscando firmas dibujadas: página {tp.index + 1} de {count}")
                        with waiting_for(PDF_LOCK):
                            zones += pdf.vector_signatures(doc[tp.index], tp, columns=not options.names_context)
            cache: dict = {}
            share = 0.75 / count
            for n, tp in enumerate(text_pages):
                if not options.raster:
                    break
                base = 0.2 + share * n
                report(base, f"Revisando las imágenes de la página {n + 1} de {count}")

                def step(name: str, base=base, n=n) -> None:
                    report(base + share * _STAGE_OFFSET[name], f"{_STAGE_TEXT[name]}: página {n + 1} de {count}")

                try:
                    zones += pdf.raster_zones(
                        doc, tp, names, options, cache, step, **({"dpi": self.ocr_dpi} if self.ocr_dpi else {})
                    )
                except (RuntimeError, ValueError) as exc:
                    raise FileError("corrupt", repr(exc)) from exc
        finally:
            with PDF_LOCK:
                doc.close()
        findings = []
        for z in zones:
            view = (z.rect * to_view[z.page]).normalize()
            polygon = rect_polygon(view.x0, view.y0, view.x1, view.y1)
            findings.append(self._finding(file, options, z.page, z, polygon, z.source))
        return pages, findings

    def _analyze_image(self, file: AnalyzedFile, names: tuple[str, ...], options: DetectionOptions, report):
        import numpy as np

        from anonymizer.engine import image, raster

        report(0.05, "Leyendo la imagen")
        count = image.frame_count(file.path)
        pages: list[PageInfo] = []
        findings: list[Finding] = []
        share = 0.9 / count
        frames = image.frames(file.path)
        try:
            for n in range(count):
                base = 0.07 + share * n
                where = f": imagen {n + 1} de {count}" if count > 1 else ""
                report(base, "Leyendo la imagen" + where)
                # Decoding counts as "render" only when the pixels are searched (OCR, faces or QR).
                with stage("render") if options.raster else nullcontext():
                    frame = next(frames, None)
                    rgb = np.array(frame) if frame is not None and options.raster else None
                if frame is None:  # fewer frames than the header said
                    break
                pages.append(PageInfo(index=n, width=float(frame.width), height=float(frame.height), unit="px"))
                del frame
                if rgb is None:
                    continue

                def step(name: str, base=base, where=where) -> None:
                    report(base + share * _STAGE_OFFSET[name], _STAGE_TEXT[name] + where)

                zones = raster.detect_in_image(rgb, names, all_text=file.all_text, step=step, options=options)
                for z in zones:
                    polygon = [[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(z.polygon)]
                    findings.append(self._finding(file, options, n, z, polygon, "raster"))
        finally:
            frames.close()
        return pages, findings

    # ------------------------------------------------------------------
    # render_page
    # ------------------------------------------------------------------

    def render_page(self, file: AnalyzedFile, page: int, zoom: float = 1.0) -> bytes:
        """PNG of one page in view space. Hidden PDF layers are shown: they are exported visible (and redacted)."""
        zoom = max(0.05, min(8.0, float(zoom)))
        kind = file.kind or sniff(file.path)
        if kind == "pdf":
            import pymupdf

            from anonymizer.engine import pdf

            with PDF_LOCK:
                with pymupdf.open(file.path, filetype="pdf") as doc:
                    pdf.reveal_layers(doc)
                    pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                    return pix.tobytes("png")
        from PIL import Image, ImageOps

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

    def _from_text_layer(self, finding: Finding) -> bool:
        source = self._sources.get(finding.id)
        if source is None:  # not proposed by this engine instance: check it when it could be text
            return finding.detector in _TEXT_LAYER_DETECTORS
        return source == "text"

    def export(self, file: AnalyzedFile, dest_dir: str) -> ExportResult:
        from anonymizer.engine import verify

        started = time.perf_counter()
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        active = [f for f in file.findings if f.active]
        # Left visible on purpose: removed by the reviewer, or a suggestion (D12) not applied.
        kept = [f for f in file.findings if not f.active]
        removed = sum(1 for f in file.findings if f.status == "removed")
        kind = file.kind or sniff(file.path)
        name = Path(file.name).name or "archivo"
        whole: list[dict] = []  # PDF strokes removed whole (ExportResult.strokes_removed_whole)
        with tempfile.TemporaryDirectory(prefix="anonimizador_export_") as tmp:
            if kind == "pdf":
                if Path(name).suffix.lower() != ".pdf":
                    name = Path(name).stem + ".pdf"
                staged = Path(tmp) / "output.pdf"
                whole = self._export_pdf(file, active, staged)
                leaks = verify.pdf_leaks(staged, active, kept, self._from_text_layer)
            else:
                from anonymizer.engine import image

                polygons: dict[int, list] = {}
                for f in active:
                    polygons.setdefault(f.page, []).append(f.polygon)
                staged, fmt = image.redact(file.path, polygons, Path(tmp))
                if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                    name = Path(name).stem + IMAGE_FORMATS[fmt]
                leaks = verify.image_leaks(staged)
            leaks += verify.uncovered(staged, kind, active)
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
            log.warning("export of %s blocked: %d leaks", file.id, n)
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
            strokes_removed_whole=whole if not leaks else [],
        )

    @staticmethod
    def _export_pdf(file: AnalyzedFile, active: list[Finding], staged: Path) -> list[dict]:
        """Writes the redacted PDF; returns the strokes removed whole (``ExportResult``), in view space."""
        import pymupdf

        from anonymizer.engine import pdf, strokes

        with PDF_LOCK:
            with pymupdf.open(file.path, filetype="pdf") as doc:
                to_page = [pymupdf.Matrix(page.derotation_matrix) for page in doc]
                to_view = [pymupdf.Matrix(page.rotation_matrix) for page in doc]
        # The leak check (strokes.leaks) uses these same rectangles.
        zones = strokes.zones_by_page(active, to_page)
        rects = {n: [r for r, _ in items] for n, items in zones.items()}
        drawn = {n: [r for r, f in items if f.type in strokes.DRAWN_TYPES] for n, items in zones.items()}
        whole: list[tuple[int, pymupdf.Rect]] = []
        pdf.redact(file.path, str(staged), rects, drawn, whole)
        out = []
        for n, box in whole:
            r = (box * to_view[n]).normalize()
            out.append({"page": n, "polygon": [[round(x, 2), round(y, 2)] for x, y in rect_polygon(*r)]})
        return out
