"""Renders the user manual (docs/user-manual/manual-de-usuario.md) as a printable A4 PDF.

Usage (from the repository root, on Windows):
    uv run --with markdown python scripts/build_manual_pdf.py

The Markdown becomes HTML with a simple stylesheet and the app's own font (Atkinson Hyperlegible,
anonymizer/ui/fonts); headless Microsoft Edge prints it to PDF; PyMuPDF then adds the page
numbers and the document's title. Writes docs/user-manual/manual-de-usuario.pdf next to the
Markdown (the screenshots in docs/user-manual/img are embedded).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "user-manual" / "manual-de-usuario.md"
FONTS = ROOT / "anonymizer" / "ui" / "fonts"
TITLE = "Manual de usuario del Anonimizador"
EDGE_PATHS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

CSS = """
@font-face { font-family: "Atkinson"; src: url("{fonts}/AtkinsonHyperlegible-Regular.ttf"); }
@font-face { font-family: "Atkinson"; font-weight: bold; src: url("{fonts}/AtkinsonHyperlegible-Bold.ttf"); }
@font-face { font-family: "Atkinson"; font-style: italic; src: url("{fonts}/AtkinsonHyperlegible-Italic.ttf"); }
@font-face { font-family: "Plex Mono"; src: url("{fonts}/IBMPlexMono-Regular.ttf"); }
@page { size: A4; margin: 18mm 18mm 20mm 18mm; }
html { font-family: "Atkinson", sans-serif; font-size: 10.5pt; line-height: 1.45; color: #17212D; }
body { margin: 0; }
h1 { font-size: 22pt; color: #2A4C9C; margin: 0 0 4pt 0; }
h1 + p { color: #526073; margin-top: 0; }
h2 { font-size: 15pt; color: #2A4C9C; margin: 20pt 0 6pt 0; padding-bottom: 3pt;
     border-bottom: 1px solid #D3DAE3; break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 4pt 0; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 18pt; }
li { margin: 2pt 0; }
a { color: #2A4C9C; text-decoration: none; }
code { font-family: "Plex Mono", monospace; font-size: 9pt; background: #EEF1F5; padding: 0 2pt; border-radius: 2pt; }
img { display: block; max-width: 100%; margin: 8pt auto 10pt auto; border: 1px solid #C9D1DC; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; break-inside: auto; }
tr { break-inside: avoid; }
th { text-align: left; font-size: 9.5pt; color: #526073; border-bottom: 1.5px solid #C9D1DC; padding: 3pt 5pt; }
td { font-size: 9.5pt; border-bottom: 1px solid #E3E8EF; padding: 3pt 5pt; vertical-align: top; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #FBEFE2; border-left: 4px solid #C77A1E; }
blockquote p { margin: 0; }
.toc-page { break-after: page; }
"""


def slug(text: str, separator: str = "-") -> str:
    """GitHub's heading anchors: lowercase, punctuation dropped (accents kept), spaces as hyphens."""
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", separator)


def find_edge() -> Path | None:
    for path in EDGE_PATHS:
        if Path(path).is_file():
            return Path(path)
    return None


def to_html(markdown_text: str) -> str:
    import markdown

    body = markdown.markdown(
        markdown_text,
        extensions=["tables", "toc", "sane_lists"],
        extension_configs={"toc": {"slugify": slug}},
    )
    # The first page holds the title and the contents; each numbered section then follows.
    body = body.replace('<h2 id="1-', '</div><h2 id="1-', 1)
    base = MANUAL.parent.as_uri() + "/"
    css = CSS.replace("{fonts}", FONTS.as_uri())
    return (
        '<!doctype html><html lang="es-CL"><head><meta charset="utf-8">'
        f'<base href="{base}"><title>{TITLE}</title><style>{css}</style></head>'
        f'<body><div class="toc-page">{body}</body></html>'
    )


def print_pdf(edge: Path, html: Path, target: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="manual_edge_") as profile:
        command = [
            str(edge),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-extensions",
            "--no-pdf-header-footer",
            "--allow-file-access-from-files",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={target}",
            html.as_uri(),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=180)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("Edge did not write the PDF")


def finish(raw: Path, target: Path) -> int:
    """Page numbers in the footer, the title in the metadata, and a compact file."""
    with pymupdf.open(raw) as doc:
        for n, page in enumerate(doc):
            if n == 0:
                continue
            text = f"{TITLE} · página {n + 1} de {doc.page_count}"
            page.insert_text((51, page.rect.height - 28), text, fontsize=8, color=(0.32, 0.38, 0.45))
        doc.set_metadata(
            {"title": TITLE, "author": "Anonimizador", "creator": "Anonimizador", "producer": "Anonimizador"}
        )
        target.write_bytes(doc.tobytes(garbage=3, deflate=True))
        return doc.page_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=MANUAL.with_suffix(".pdf"))
    args = parser.parse_args(argv)
    edge = find_edge()
    if edge is None:
        print("Microsoft Edge was not found: this script prints the manual with it.", file=sys.stderr)
        return 1
    html_text = to_html(MANUAL.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="manual_pdf_") as tmp:
        html = Path(tmp) / "manual.html"
        html.write_text(html_text, encoding="utf-8")
        raw = Path(tmp) / "manual.pdf"
        print_pdf(edge, html, raw)
        pages = finish(raw, args.output)
    print(f"{os.path.relpath(args.output, ROOT)}: {pages} pages, {args.output.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
