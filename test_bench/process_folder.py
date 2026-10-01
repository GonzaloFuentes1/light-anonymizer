"""Anonymizes every file of a folder (without a manifest) with the prototype and builds sheets.

Usage:
    uv run python -m test_bench.process_folder <folder> --output results/gore [--processes 6]

For each file it writes ``<output>/anonymized/<name>``, one sheet per page (original on the
left, anonymized on the right) in ``<output>/sheets/`` and a summary in ``<output>/resumen.md``.
Since there is no name list, it uses the words of the file name that look like surnames
(for example "ROJAS TAPIA INFORME.pdf" -> "Rojas Tapia").
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from test_bench.baselines import prototype
from test_bench.canvas import font
from test_bench.visualize import visible_pages

_GENERIC_WORDS = {"informe", "prueba", "ficticio", "de", "del", "la", "el", "y", "final", "mensual", "pdf"}

# Spanish labels for the redaction types shown in resumen.md (the users read Spanish).
_TYPE_LABELS = {
    "email": "correo",
    "phone": "telefono",
    "name": "nombre",
    "address": "direccion",
    "face": "rostro",
    "signature": "firma",
    "text": "texto",
}


def list_from_name(path: Path) -> tuple[str, ...]:
    words = [w for w in re.split(r"[\s_\-.]+", path.stem) if w and w.lower() not in _GENERIC_WORDS]
    words = [w for w in words if w.isalpha() and len(w) >= 3]
    return (" ".join(w.capitalize() for w in words),) if words else ()


def _sheet(original: Image.Image, anonymized: Image.Image | None, title: str) -> Image.Image:
    height = 1100
    a = original.resize((max(1, int(original.width * height / original.height)), height))
    b = anonymized.resize(a.size) if anonymized is not None else Image.new("RGB", a.size, (230, 230, 230))
    canvas = Image.new("RGB", (a.width * 2 + 48, height + 70), "white")
    d = ImageDraw.Draw(canvas)
    d.text((16, 14), f"Original — {title}", font=font("sans_bold", 20), fill=(20, 20, 20))
    d.text((a.width + 32, 14), "Anonimizado (prototipo)", font=font("sans_bold", 20), fill=(20, 20, 20))
    canvas.paste(a, (16, 56))
    canvas.paste(b, (a.width + 32, 56))
    return canvas


def process_one(path: Path, output: Path) -> dict:
    name_list = list_from_name(path)
    dest = output / "anonymized" / path.name
    start = time.perf_counter()
    row: dict = {"file": path.name, "name_list": name_list}
    try:
        if path.suffix.lower() == ".pdf":
            redactions, n = prototype._process_pdf(path, dest, name_list, all_urls=False)
        else:
            redactions, n = prototype._process_image(path, dest, name_list, all_urls=False)
    except Exception as err:  # noqa: BLE001 - reported, and the others carry on
        row["error"] = f"{type(err).__name__}: {err}"
        return row
    row["seconds"] = round(time.perf_counter() - start, 1)
    row["pages"] = n
    row["zones"] = dict(Counter(r.type for r in redactions))
    # Quick check of the output: patterns over the text layer and metadata.
    if dest.suffix.lower() == ".pdf":
        with pymupdf.open(dest) as doc:
            text = "\n".join(p.get_text() for p in doc)
            row["leftovers_in_text"] = len(prototype.find_spans(text, name_list, all_urls=False))
            row["metadata"] = [k for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")]
    # Sheets for every page
    format = "pdf" if path.suffix.lower() == ".pdf" else path.suffix.lower().lstrip(".")
    before, after = visible_pages(path, format), visible_pages(dest, format)
    sheets_dir = output / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(before):
        sheet = _sheet(img, after[i] if i < len(after) else None, f"{path.stem}, página {i + 1}")
        sheet.save(sheets_dir / f"{path.stem}_p{i + 1:02d}.jpg", quality=85)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Anonimiza con el prototipo todos los archivos de una carpeta (sin manifiesto) y arma láminas."
    )
    ap.add_argument("folder", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=6)
    args = ap.parse_args(argv)
    files = sorted(
        p
        for p in args.folder.iterdir()
        if p.suffix.lower() in (".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
    )
    (args.output / "anonymized").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(min(args.processes, len(files))) as pool:
        rows = list(pool.map(process_one, files, [args.output] * len(files)))
    lines = ["# Documentos procesados con el prototipo", "",
             "| Archivo | Páginas | Zonas censuradas | Datos que siguen en el texto | Metadatos que quedan | Tiempo |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['file']} | — | error: {r['error']} | — | — | — |")
            continue
        labelled = sorted((_TYPE_LABELS.get(k, k), v) for k, v in r["zones"].items())
        zones = ", ".join(f"{k} {v}" for k, v in labelled) or "ninguna"
        lines.append(
            f"| {r['file']} | {r['pages']} | {zones} | {r.get('leftovers_in_text', '—')} | "
            f"{', '.join(r.get('metadata', [])) or 'ninguno'} | {r['seconds']} s |"
        )
    lines += ["", "Lista de nombres usada (tomada del nombre de cada archivo): "
              + "; ".join(f"{r['file']}: {', '.join(r['name_list']) or '(ninguno)'}" for r in rows)]  # fmt: skip
    (args.output / "resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
