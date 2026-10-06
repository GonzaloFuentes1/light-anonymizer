"""Audit report of an export: JSON for systems and a readable PDF (Spanish) for people.

The report never contains the censored data itself: only its type, position, detector and the
reviewer's decisions. The text of a censure removed by the reviewer, of a URL left visible (D12)
and of a value of the exceptions list left visible (D10), is included, because that text stays
visible in the published document anyway. Each file also records which detection groups were
searched and how long the analysis took.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pymupdf

from anonymizer import __version__
from anonymizer.engine.exceptions import REASON as EXCEPTION
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import (
    DETECTION_GROUPS,
    TYPE_LABELS,
    AnalyzedFile,
    DetectionOptions,
    ExportResult,
    Finding,
)

TITLE = "Informe de auditoría de anonimización"
CLOSING = (
    "Este informe deja constancia de la revisión realizada. La herramienta no certifica que un documento "
    "esté libre de datos personales: la decisión de publicar es de quien revisa."
)
PRIVACY_NOTE = "Este informe no incluye los datos censurados: solo su tipo, su ubicación y las decisiones de revisión."
MONTHS = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
DETECTOR_LABELS = {
    "regex": "patrones",
    "name_list": "lista de nombres",
    "context": "contexto",
    "ocr": "texto en imagen (OCR)",
    "faces": "rostros",
    "signatures": "firmas",
    "qr": "códigos QR",
    "reviewer": "agregada por quien revisó",
}
# Stages of the analysis (``AnalyzedFile.timings``), in the order they are shown.
STAGE_LABELS = {
    "text": "texto",
    "render": "preparar imágenes",
    "ocr": "texto en imágenes",
    "faces": "rostros",
    "signatures": "firmas",
    "qr": "códigos QR",
}
UNREAD_IMAGES = (
    "No se leyó el texto de las imágenes ni de las páginas escaneadas de este archivo (OCR apagado): "
    "los datos que estén dentro de ellas no se buscaron."
)


def _unique(folder: Path, name: str) -> Path:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = folder / name, 2
    while candidate.exists():
        candidate = folder / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def _spanish_date(moment: datetime) -> str:
    return f"{moment.day} de {MONTHS[moment.month - 1]} de {moment.year}, {moment:%H:%M}"


def _last(finding: Finding, action: str):
    for entry in reversed(finding.history):
        if entry.action == action:
            return entry
    return None


def _is_added(finding: Finding) -> bool:
    return finding.detector == "reviewer" or finding.type == "manual"


def _options(file: AnalyzedFile) -> DetectionOptions:
    return DetectionOptions.from_dict(file.options)


def _unread_images(file: AnalyzedFile) -> bool:
    """OCR was off and the file has pixels with possible text (an image or a scanned page)."""
    return not _options(file).ocr and (file.kind == "image" or any(p.scanned for p in file.pages))


def not_searched(file: AnalyzedFile) -> str | None:
    """Spanish warning for a file analyzed with detection groups off (None when every one was on).

    The leak check only looks for what was censored, so "no leaks" says nothing about what was
    not searched: the export result repeats it next to that verdict.
    """
    options = _options(file)
    off = [g.short for g in DETECTION_GROUPS if g.detection and not getattr(options, g.key)]
    return f"En este archivo no se buscaron: {_join(off)}. Revisa esas partes a mano." if off else None


def seconds_text(value: float) -> str:
    """Spanish duration: "0,4 s", "12 s", "1 min 20 s"."""
    if value < 0.1:
        return "menos de 0,1 s"
    if value < 10:
        return f"{value:.1f}".replace(".", ",") + " s"
    total = round(value)
    if total < 60:
        return f"{total} s"
    minutes, rest = divmod(total, 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def _timing_text(file: AnalyzedFile) -> str | None:
    if "analyze" not in file.timings:
        return None
    parts = [
        f"{label} {seconds_text(file.timings[stage])}" for stage, label in STAGE_LABELS.items() if stage in file.timings
    ]
    detail = f" ({' · '.join(parts)})" if parts else ""
    return f"Tiempo de análisis: {seconds_text(file.timings['analyze'])}{detail}."


def _file_record(file: AnalyzedFile, result: ExportResult | None) -> dict:
    unit = file.pages[0].unit if file.pages else ("pt" if file.kind == "pdf" else "px")
    findings = []
    changes = []
    for f in file.findings:
        record = {
            "id": f.id,
            "type": f.type,
            "type_label": TYPE_LABELS.get(f.type, f.type),
            "page": f.page,
            "page_number": f.page + 1,
            "polygon": f.polygon,
            "unit": unit,
            "detector": f.detector,
            "score": f.score,
            "doubtful": f.doubtful,
            "doubt_reason": f.doubt_reason,
            "status": f.status,
            "optional": f.optional,
            "optional_reason": f.optional_reason,
            "applied": f.active,
            "history": [asdict(h) for h in f.history],
        }
        if not f.active and f.text:
            record["visible_text"] = f.text
        findings.append(record)
        for h in f.history:
            if h.action in ("removed", "restored", "added", "applied", "skipped"):
                changes.append(
                    {
                        "finding_id": f.id,
                        "type": f.type,
                        "page_number": f.page + 1,
                        "action": h.action,
                        "at": h.at,
                        "reason": h.reason,
                        "note": h.note,
                    }
                )
    active = [f for f in file.findings if f.active]
    leaks = result.leaks if result else file.leaks
    options = _options(file)
    return {
        "id": file.id,
        "name": file.name,
        "kind": file.kind,
        "pages": len(file.pages),
        "status": file.status,
        "exported": bool(result and result.exported),
        "output_file": Path(result.output_path).name if result and result.output_path else None,
        "message": result.message if result else None,
        "detections": [{"key": g.key, "label": g.label, "enabled": getattr(options, g.key)} for g in DETECTION_GROUPS],
        "detections_off": [g.key for g in DETECTION_GROUPS if g.detection and not getattr(options, g.key)],
        "images_unread": _unread_images(file),
        "timings": dict(file.timings),
        "other_urls": _optional_record(_other_urls(file)),
        # D10: the list in effect when the file was processed, and what it left unapplied.
        "exceptions": {"entries": list(file.exceptions), **_optional_record(_exceptions(file))},
        # D8: zones that also took whole letters drawn as paths under them (small rectangles, view space).
        "letters_covered": [
            {**g, "page_number": g["page"] + 1} for g in (result.grown if result and result.exported else [])
        ],
        # Decided 2026-10-06: pages exported as one image, because something drawn might have stayed
        # under a zone, and why.
        "rasterized_pages": [
            {"page_number": r["page"] + 1, "reason": r["reason"]}
            for r in (result.rasterized_pages if result and result.exported else [])
        ],
        "redactions_applied": len(active),
        # Drawn strokes removed whole although part of them lay outside the zones: the page shows that.
        "strokes_removed_whole": [
            {"page_number": s["page"] + 1, "polygon": s["polygon"], "unit": unit}
            for s in (result.strokes_removed_whole if result else [])
        ],
        "removed_by_reviewer": sum(1 for f in file.findings if f.status == "removed"),
        "added_by_reviewer": sum(1 for f in active if _is_added(f)),
        "doubtful": sum(1 for f in file.findings if f.doubtful),
        "applied_by_type": dict(Counter(f.type for f in active)),
        "findings": findings,
        "reviewer_changes": changes,
        "leak_check": {
            "ran": result is not None,
            "passed": result is not None and not leaks,
            "leaks": [asdict(leak) for leak in leaks],
        },
    }


def _other_urls(file: AnalyzedFile) -> list[Finding]:
    """Optional findings of D12: URLs that are not personal."""
    return [f for f in file.findings if f.optional and f.optional_reason != EXCEPTION]


def _exceptions(file: AnalyzedFile) -> list[Finding]:
    """Optional findings of D10: values of the user's exceptions list."""
    return [f for f in file.findings if f.optional and f.optional_reason == EXCEPTION]


def _optional_record(optional: list[Finding]) -> dict:
    return {
        "applied": sum(1 for f in optional if f.active),
        "left_visible": sum(1 for f in optional if f.status == "suggested"),
        "items": [
            {
                "finding_id": f.id,
                "page_number": f.page + 1,
                "applied": f.active,
                # Only what stays visible: the text of an applied one is censored data.
                **({"visible_text": f.text} if not f.active and f.text else {}),
            }
            for f in optional
        ],
    }


def build_report(files: list[AnalyzedFile], results: list[ExportResult], generated: datetime | None = None) -> dict:
    generated = generated or datetime.now().astimezone()
    by_id = {r.file_id: r for r in results}
    return {
        "report": TITLE,
        "generated_at": generated.isoformat(timespec="seconds"),
        "tool": {"name": "Anonimizador", "version": __version__},
        "privacy_note": PRIVACY_NOTE,
        "notice": CLOSING,
        "files": [_file_record(f, by_id.get(f.id)) for f in files],
    }


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

CSS = """
* { font-family: sans-serif; }
body { font-size: 10pt; line-height: 1.4; color: #17212D; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; color: #2A4C9C; }
h2 { font-size: 13pt; margin: 16pt 0 4pt 0; border-bottom: 1px solid #D3DAE3; padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 8pt 0 2pt 0; }
p { margin: 2pt 0; }
.muted { color: #526073; }
.ok { color: #0A6A44; font-weight: bold; }
.bad { color: #AF2216; font-weight: bold; }
table { border-collapse: collapse; width: 100%; margin: 2pt 0; }
th { text-align: left; font-size: 9pt; color: #526073; border-bottom: 1px solid #D3DAE3; padding: 2pt 4pt; }
td { font-size: 9.5pt; border-bottom: 1px solid #E9EDF3; padding: 2pt 4pt; vertical-align: top; }
.closing { margin-top: 18pt; border-top: 1px solid #D3DAE3; padding-top: 6pt; font-weight: bold; }
"""


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _join(items: list[str]) -> str:
    """Spanish list: "a", "a y b", "a, b y c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _file_html(file: AnalyzedFile, result: ExportResult | None) -> str:
    parts = [f"<h2>{_e(file.name)}</h2>"]
    active = [f for f in file.findings if f.active]
    removed = [f for f in file.findings if f.status == "removed"]
    added = [f for f in active if _is_added(f)]
    optional = _other_urls(file)
    listed = _exceptions(file)
    if result is None:
        parts.append("<p class='muted'>No se intentó exportar este archivo.</p>")
    elif result.exported:
        out = Path(result.output_path).name if result.output_path else ""
        parts.append(f"<p>Exportado como <b>{_e(out)}</b>.</p>")
    else:
        parts.append(f"<p class='bad'>No se exportó. {_e(result.message)}</p>")
    pages = len(file.pages)
    parts.append(
        f"<p class='muted'>{_count(pages, 'página', 'páginas')} · "
        f"{_count(len(active), 'censura aplicada', 'censuras aplicadas')}</p>"
    )

    parts.append("<h3>Qué se buscó</h3>")
    options = _options(file)
    rows = [
        [g.label, ("Sí" if getattr(options, g.key) else "No") if not g.detection else
         ("Se buscó" if getattr(options, g.key) else "No se buscó")]
        for g in DETECTION_GROUPS
    ]  # fmt: skip
    parts.append(_table(["Detección", "Estado"], rows))
    off = [g.short for g in DETECTION_GROUPS if g.detection and not getattr(options, g.key)]
    if off:
        parts.append(f"<p class='bad'>No se buscaron: {_e(_join(off))}.</p>")
    if _unread_images(file):
        parts.append(f"<p class='bad'>{_e(UNREAD_IMAGES)}</p>")
    timing = _timing_text(file)
    if timing:
        parts.append(f"<p class='muted'>{_e(timing)}</p>")

    parts.append("<h3>Censuras aplicadas por tipo</h3>")
    counts = Counter(f.type for f in active)
    if counts:
        rows = [[TYPE_LABELS.get(t, t), n] for t, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        parts.append(_table(["Tipo de dato", "Cantidad"], rows))
    else:
        parts.append("<p>Ninguna.</p>")
    doubtful = [f for f in file.findings if f.doubtful]
    if doubtful:
        parts.append(
            f"<p class='muted'>{_count(len(doubtful), 'hallazgo dudoso se mostró', 'hallazgos dudosos se mostraron')}"
            " primero para revisión.</p>"
        )

    parts.append("<h3>Censuras quitadas por quien revisó</h3>")
    if removed:
        rows = []
        for f in removed:
            entry = _last(f, "removed")
            rows.append(
                [
                    TYPE_LABELS.get(f.type, f.type),
                    f.page + 1,
                    f.text or "—",
                    (entry.reason if entry else None) or "Sin motivo indicado",
                    (entry.note if entry else None) or "—",
                ]
            )
        parts.append(_table(["Tipo", "Página", "Queda visible", "Motivo", "Comentario"], rows))
    else:
        parts.append("<p>Ninguna.</p>")

    if optional:
        parts.append("<h3>Otros enlaces (no personales)</h3>")
        applied = [f for f in optional if f.active]
        visible = [f for f in optional if f.status == "suggested"]
        parts.append(
            f"<p>{_count(len(applied), 'enlace censurado', 'enlaces censurados')} · "
            f"{_count(len(visible), 'enlace queda visible', 'enlaces quedan visibles')}.</p>"
        )
        if visible:
            parts.append(_table(["Página", "Queda visible"], [[f.page + 1, f.text or "—"] for f in visible]))

    if result is not None and result.exported and result.rasterized_pages:
        parts.append("<h3>Páginas exportadas como imagen</h3>")
        parts.append(
            "<p>En estas páginas algo dibujado podía seguir en el archivo bajo una zona censurada. Para quitarlo con "
            "certeza, cada una se exportó como una sola imagen de la página ya censurada (300 dpi): ya no tiene texto "
            "seleccionable ni dibujos.</p>"
        )
        rows = [[r["page"] + 1, r["reason"]] for r in result.rasterized_pages]
        parts.append(_table(["Página", "Motivo"], rows))

    if result is not None and result.strokes_removed_whole:
        parts.append("<h3>Trazos dibujados quitados enteros</h3>")
        parts.append(
            "<p>Un trazo dibujado que estaba casi todo bajo una zona de firma o una zona agregada se quitó entero, "
            "también la parte que sobresalía de la zona:</p>"
        )
        rows = [[s["page"] + 1, "Se quitó entero"] for s in result.strokes_removed_whole]
        parts.append(_table(["Página", "Trazo"], rows))

    if listed or file.exceptions:
        parts.append("<h3>Lista de excepciones (RUT de instituciones, números 600 y 800)</h3>")
        if file.exceptions:
            parts.append(f"<p class='muted'>Lista usada al procesar: {_e(', '.join(file.exceptions))}.</p>")
        applied = [f for f in listed if f.active]
        visible = [f for f in listed if f.status == "suggested"]
        parts.append(
            f"<p>{_count(len(applied), 'dato censurado', 'datos censurados')} · "
            f"{_count(len(visible), 'dato queda visible', 'datos quedan visibles')}.</p>"
        )
        if visible:
            rows = [[TYPE_LABELS.get(f.type, f.type), f.page + 1, f.text or "—"] for f in visible]
            parts.append(_table(["Tipo", "Página", "Queda visible"], rows))

    grown = result.grown if result is not None and result.exported else []
    if grown:  # D8: the rectangles applied, besides the zones, for the letters under them
        by_id = {f.id: f for f in file.findings}
        parts.append("<h3>Letras cubiertas por las zonas</h3>")
        parts.append(
            "<p>Estas zonas tenían debajo letras dibujadas como trazos. Cada una se tapó entera con un "
            "rectángulo propio, para quitarla del archivo.</p>"
        )
        rows = [
            [g["page"] + 1, TYPE_LABELS.get(by_id[g["finding_id"]].type, "") if g["finding_id"] in by_id else "—",
             len(g["rects"])]
            for g in grown
        ]  # fmt: skip
        parts.append(_table(["Página", "Zona", "Letras cubiertas"], rows))

    parts.append("<h3>Zonas agregadas por quien revisó</h3>")
    if added:
        rows = [[f.page + 1, (_last(f, "added").note if _last(f, "added") else None) or "—"] for f in added]
        parts.append(_table(["Página", "Comentario"], rows))
    else:
        parts.append("<p>Ninguna.</p>")

    parts.append("<h3>Verificación de fugas</h3>")
    if result is None:
        parts.append("<p class='muted'>No se realizó.</p>")
    elif not result.leaks:
        parts.append("<p class='ok'>La verificación automática no encontró datos censurados legibles.</p>")
    else:
        rows = [[leak.page + 1 if leak.page is not None else "—", leak.message] for leak in result.leaks]
        parts.append(f"<p class='bad'>{_count(len(result.leaks), 'fuga sin resolver', 'fugas sin resolver')}.</p>")
        parts.append(_table(["Página", "Detalle"], rows))
    return "".join(parts)


def _write_pdf(path: Path, files: list[AnalyzedFile], results: list[ExportResult], generated: datetime) -> None:
    by_id = {r.file_id: r for r in results}
    exported = sum(1 for r in results if r.exported)
    body = (
        f"<h1>{_e(TITLE)}</h1>"
        f"<p class='muted'>Generado el {_e(_spanish_date(generated))} · Anonimizador {_e(__version__)}</p>"
        f"<p>{_count(len(files), 'archivo revisado', 'archivos revisados')} · "
        f"{_count(exported, 'exportado', 'exportados')}.</p>"
        f"<p class='muted'>{_e(PRIVACY_NOTE)}</p>"
        + "".join(_file_html(f, by_id.get(f.id)) for f in files)
        + f"<p class='closing'>{_e(CLOSING)}</p>"
    )
    story = pymupdf.Story(html=f"<body>{body}</body>", user_css=CSS)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (48, 48, -48, -56)
    writer = pymupdf.DocumentWriter(str(path))
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    # Page numbers and clean metadata.
    with pymupdf.open(path) as doc:
        for n, page in enumerate(doc):
            page.insert_text(
                (48, mediabox.height - 28),
                f"{TITLE} · página {n + 1} de {doc.page_count}",
                fontsize=8,
                color=(0.32, 0.38, 0.45),
            )
        doc.set_metadata({"title": TITLE, "producer": "Anonimizador", "creator": "Anonimizador"})
        data = doc.tobytes(garbage=3, deflate=True)
    path.write_bytes(data)


def write_audit(
    files: list[AnalyzedFile],
    results: list[ExportResult],
    dest_dir: str,
    *,
    write_json: bool = True,
    write_pdf: bool = True,
) -> tuple[str | None, str | None]:
    """Write the audit report into ``dest_dir`` (never overwriting). Returns (json_path, pdf_path)."""
    folder = Path(dest_dir)
    folder.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().astimezone()
    stamp = generated.strftime("%Y-%m-%d_%H%M%S")
    json_path = pdf_path = None
    if write_json:
        json_path = _unique(folder, f"informe_auditoria_{stamp}.json")
        report = build_report(files, results, generated)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if write_pdf:
        pdf_path = _unique(folder, f"informe_auditoria_{stamp}.pdf")
        with PDF_LOCK:
            _write_pdf(pdf_path, files, results, generated)
    return (str(json_path) if json_path else None, str(pdf_path) if pdf_path else None)
