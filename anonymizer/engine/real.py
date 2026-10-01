"""The real engine: detection from the prototype, split into analyze (detect) and export (apply).

``analyze`` reads the working copy and proposes findings in view space; nothing is modified.
``export`` applies only the active findings (the reviewer's decisions), cleans the file, runs
the leak check over a temporary output and copies it to the destination only when it is clean.

Detectors: patterns (``regex``), the user's list (``name_list``), context rules and the
given-name dictionary (``context``), OCR at 0/90/270° (``ocr``), YuNet faces (``faces``) and QR
codes (``qr``). See the modules of this package for each one.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import tempfile
import time
import traceback
import uuid
from pathlib import Path

from anonymizer.engine import faces, ocr
from anonymizer.engine.common import (
    IMAGE_FORMATS,
    IMAGE_SUFFIXES,
    Cancelled,
    FileError,
    bbox_of,
    disable_power_throttling,
    now_iso,
    publish,
    rect_polygon,
    sniff,
)
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import (
    ERROR_MESSAGES,
    AnalyzedFile,
    ExportResult,
    Finding,
    HistoryEntry,
    PageInfo,
)

log = logging.getLogger(__name__)

_STAGE_TEXT = {
    "ocr": "Leyendo texto en imágenes (OCR)",
    "faces": "Buscando rostros",
    "qr": "Buscando códigos QR",
}
# Where each stage starts inside the share of progress of one page or image.
_STAGE_OFFSET = {"ocr": 0.0, "faces": 0.6, "qr": 0.9}
# Detectors whose text comes from the text layer of a PDF (checked again in the output).
_TEXT_LAYER_DETECTORS = ("regex", "name_list", "context")


def missing_requirements() -> list[str]:
    """What the real engine needs and is not installed (empty when it can run)."""
    missing = []
    for module in ("cv2", "numpy", "pymupdf", "PIL"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if not ocr.available():
        missing.append("rapidocr/onnxruntime")
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

    def __init__(self, all_urls: bool = True, ocr_dpi: int | None = None):
        # D12 (pending): every URL is redacted; with ``all_urls=False`` only personal URLs are.
        self.all_urls = all_urls
        self.ocr_dpi = ocr_dpi
        # finding id -> "text" | "raster": where its text was read (internal, used by the leak check).
        self._sources: dict[str, str] = {}
        # OCR and faces run in background threads: without this, Windows slows them down 5-6x
        # whenever the app has no foreground window.
        disable_power_throttling()

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

        names = tuple(dict.fromkeys(n.strip() for n in name_list if n and n.strip()))
        file.status = "processing"
        file.error = file.error_message = None
        try:
            report(0.02, "Revisando el archivo")
            kind = sniff(file.path)
            if kind in ("empty", "format"):
                raise FileError(kind)
            file.kind = kind
            if kind == "pdf":
                pages, findings = self._analyze_pdf(file, names, report)
            else:
                pages, findings = self._analyze_image(file, names, report)
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
            log.info("file %s rejected: %s", file.id, exc.code)
        except Exception:  # noqa: BLE001 - reported to the user as an internal problem, detail in the log
            log.error("analysis of %s failed:\n%s", file.id, traceback.format_exc())
            file.status = "error"
            file.error = "internal"
            file.error_message = ERROR_MESSAGES["internal"]
            file.step = "No se pudo procesar"
        file.timings["analyze"] = round(time.perf_counter() - started, 3)
        return file

    def _finding(self, file: AnalyzedFile, page: int, type_: str, polygon, text, detector, score, doubt, source):
        finding = Finding(
            id=uuid.uuid4().hex[:12],
            file_id=file.id,
            page=page,
            type=type_,
            polygon=polygon,
            text=text or None,
            detector=detector,
            score=None if score is None else round(float(score), 4),
            doubtful=doubt is not None,
            doubt_reason=doubt,
            history=[HistoryEntry(at=now_iso(), action="proposed")],
        )
        self._sources[finding.id] = source
        return finding

    def _analyze_pdf(self, file: AnalyzedFile, names: tuple[str, ...], report):
        import pymupdf

        from anonymizer.engine import pdf

        with PDF_LOCK:
            doc = pdf.open_pdf(file.path)
        try:
            with PDF_LOCK:
                pdf.reveal_layers(doc)
                count = doc.page_count
            pages: list[PageInfo] = []
            to_view: list[pymupdf.Matrix] = []
            text_pages = []
            for n in range(count):
                report(0.05 + 0.15 * n / count, f"Leyendo el texto de la página {n + 1} de {count}")
                with PDF_LOCK:
                    try:
                        page = doc[n]
                        view = page.rect
                        pages.append(
                            PageInfo(index=n, width=round(view.width, 3), height=round(view.height, 3), unit="pt")
                        )
                        to_view.append(pymupdf.Matrix(page.rotation_matrix))
                        text_pages.append(pdf.read_text_page(page, names, self.all_urls))
                    except (RuntimeError, ValueError) as exc:
                        raise FileError("corrupt", repr(exc)) from exc
            report(0.2, "Buscando los mismos datos en todo el documento")
            pdf.propagate(text_pages)
            zones = [z for tp in text_pages for z in pdf.text_zones(tp, names)]
            cache: dict = {}
            share = 0.75 / count
            for n, tp in enumerate(text_pages):
                base = 0.2 + share * n
                report(base, f"Revisando las imágenes de la página {n + 1} de {count}")

                def step(stage: str, base=base, n=n) -> None:
                    report(base + share * _STAGE_OFFSET[stage], f"{_STAGE_TEXT[stage]}: página {n + 1} de {count}")

                try:
                    zones += pdf.raster_zones(
                        doc, tp, names, self.all_urls, cache, step, **({"dpi": self.ocr_dpi} if self.ocr_dpi else {})
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
            findings.append(
                self._finding(file, z.page, z.type, polygon, z.text, z.detector, z.score, z.doubt, z.source)
            )
        return pages, findings

    def _analyze_image(self, file: AnalyzedFile, names: tuple[str, ...], report):
        import numpy as np

        from anonymizer.engine import image, raster

        report(0.05, "Leyendo la imagen")
        count = image.frame_count(file.path)
        pages: list[PageInfo] = []
        findings: list[Finding] = []
        share = 0.9 / count
        for n, frame in enumerate(image.frames(file.path)):
            base = 0.07 + share * n
            where = f": imagen {n + 1} de {count}" if count > 1 else ""
            report(base, "Leyendo la imagen" + where)
            pages.append(PageInfo(index=n, width=float(frame.width), height=float(frame.height), unit="px"))
            rgb = np.array(frame)
            del frame

            def step(stage: str, base=base, where=where) -> None:
                report(base + share * _STAGE_OFFSET[stage], _STAGE_TEXT[stage] + where)

            zones = raster.detect_in_image(rgb, names, all_urls=self.all_urls, all_text=file.all_text, step=step)
            for z in zones:
                polygon = [[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(z.polygon)]
                findings.append(self._finding(file, n, z.type, polygon, z.text, z.detector, z.score, z.doubt, "raster"))
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
        removed = [f for f in file.findings if not f.active]
        kind = file.kind or sniff(file.path)
        name = Path(file.name).name or "archivo"
        with tempfile.TemporaryDirectory(prefix="anonimizador_export_") as tmp:
            if kind == "pdf":
                if Path(name).suffix.lower() != ".pdf":
                    name = Path(name).stem + ".pdf"
                staged = Path(tmp) / "output.pdf"
                self._export_pdf(file, active, staged)
                leaks = verify.pdf_leaks(staged, active, removed, self._from_text_layer)
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
            removed_by_reviewer=len(removed),
            exported=not leaks,
            message=message,
        )

    @staticmethod
    def _export_pdf(file: AnalyzedFile, active: list[Finding], staged: Path) -> None:
        import pymupdf

        from anonymizer.engine import pdf

        with PDF_LOCK:
            with pymupdf.open(file.path, filetype="pdf") as doc:
                to_page = [pymupdf.Matrix(page.derotation_matrix) for page in doc]
        rects: dict[int, list[pymupdf.Rect]] = {}
        for f in active:
            if 0 <= f.page < len(to_page):
                x0, y0, x1, y1 = bbox_of(f.polygon)
                rects.setdefault(f.page, []).append((pymupdf.Rect(x0, y0, x1, y1) * to_page[f.page]).normalize())
        pdf.redact(file.path, str(staged), rects)
