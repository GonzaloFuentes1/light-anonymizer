"""Files the anonymizer must reject with an understandable error code.

None of them carries personal data or elements: what is evaluated is the error code
(``expected = "error:<code>"``). Since they cannot be opened (or must not be processed), the
page list is empty: the number of pages is unknown to the evaluator and there are no
elements to index. ``Manifest.validate`` accepts it because it only checks the pages of the
elements.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

import pymupdf

from test_bench.canvas import Canvas, font_path
from test_bench.context import Context, save_image
from test_bench.fake_data import ADMINISTRATIVE_PHRASES
from test_bench.schema import FileEntry

SEED_NAME = "errores"  # seed string: kept in Spanish so the generated files do not change
CATEGORY = "errors"
PREFIX = "err"

# Fictitious passwords (not personal data); they are left in the tags so the file can be opened.
USER_PASSWORD = "clave-ficticia-2026"
OWNER_PASSWORD = "propietario-ficticio-2026"

TRUNCATED_FRACTION = 0.40
ZIP_DATE = (2026, 3, 12, 9, 15, 0)


def _neutral_pdf(rng: Any, pages: int, title: str) -> pymupdf.Document:
    """Valid PDF with neutral administrative text (no personal data)."""
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_font(fontname="dvs", fontfile=str(font_path("serif")))
        page.insert_text((64, 80), f"{title} (página {n + 1} de {pages})", fontname="dvs", fontsize=13)
        y = 120.0
        for i in rng.permutation(len(ADMINISTRATIVE_PHRASES)):
            page.insert_text((64, y), ADMINISTRATIVE_PHRASES[int(i)], fontname="dvs", fontsize=10)
            y += 17
        page.draw_rect(pymupdf.Rect(64, y + 10, 548, y + 120), color=(0.3, 0.3, 0.3), width=0.6)
    doc.subset_fonts()
    return doc


def _file_entry(name: str, path: str, format: str, code: str, description: str, **tags: Any) -> FileEntry:
    return FileEntry(
        id=f"{PREFIX}_{name}",
        path=path,
        format=format,
        category=CATEGORY,
        description=description,
        pages=[],
        expected=f"error:{code}",
        tags=dict(tags),
    )


def _deterministic_zip(entries: list[tuple[str, bytes]]) -> bytes:
    """Zip with fixed dates, so the bytes do not depend on the generation time."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    return buf.getvalue()


def generate(ctx: Context) -> list[FileEntry]:
    rng = ctx.rng(SEED_NAME)
    files: list[FileEntry] = []

    # 5. PDF protected with a user password (AES-256).
    path = "errors/protegido_contrasena.pdf"
    doc = _neutral_pdf(rng, 1, "Oficio reservado")
    doc.save(
        ctx.path(path),
        garbage=3,
        deflate=True,
        no_new_id=True,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw=USER_PASSWORD,
        owner_pw=OWNER_PASSWORD,
    )
    doc.close()
    files.append(
        _file_entry(
            "protegido_contrasena",
            path,
            "pdf",
            "password",
            "PDF cifrado con AES-256 y contraseña de usuario: no se puede abrir sin ella.",
            encryption="AES-256",
            user_password=USER_PASSWORD,
        )
    )

    # 6. Valid PDF truncated at 40 %.
    path = "errors/corrupto.pdf"
    doc = _neutral_pdf(rng, 3, "Informe técnico")
    full = doc.tobytes(garbage=3, deflate=True, use_objstms=1, no_new_id=True)
    doc.close()
    cut = int(len(full) * TRUNCATED_FRACTION)
    ctx.path(path).write_bytes(full[:cut])
    files.append(
        _file_entry(
            "corrupto",
            path,
            "pdf",
            "corrupt",
            "PDF válido de 3 páginas (con flujos de objetos) truncado al 40 % de sus bytes: sin tabla xref, "
            'trailer ni árbol de páginas; PyMuPDF lo "repara" y queda con 0 páginas.',
            original_bytes=len(full),
            truncated_bytes=cut,
        )
    )

    # 7. Empty file.
    path = "errors/vacio.pdf"
    ctx.path(path).write_bytes(b"")
    files.append(_file_entry("vacio", path, "pdf", "empty", "Archivo .pdf de 0 bytes."))

    # 8. Plain text with a .pdf extension.
    path = "errors/no_es_pdf.pdf"
    text = (
        "Este archivo no es un PDF: es texto plano con extensión .pdf.\n" + "\n".join(ADMINISTRATIVE_PHRASES[:4]) + "\n"
    )
    ctx.path(path).write_bytes(text.encode("utf-8"))
    files.append(
        _file_entry("no_es_pdf", path, "pdf", "format", "Texto plano UTF-8 con extensión .pdf (sin cabecera %PDF).")
    )

    # 9. Truncated JPEG.
    path = "errors/imagen_corrupta.jpg"
    canvas = Canvas.new(900, 600, background=(250, 250, 246))
    y = 80
    for i in rng.permutation(len(ADMINISTRATIVE_PHRASES))[:8]:
        canvas.write_text(40, y, ADMINISTRATIVE_PHRASES[int(i)][:60], size=22, register=False)
        y += 50
    dest = ctx.path(path)
    save_image(canvas.img, dest, "jpg", quality=88)
    full = dest.read_bytes()
    cut = int(len(full) * TRUNCATED_FRACTION)
    dest.write_bytes(full[:cut])
    files.append(
        _file_entry(
            "imagen_corrupta",
            path,
            "jpg",
            "corrupt",
            "JPEG válido truncado al 40 %: la cabecera se lee, pero la decodificación falla (sin marcador EOI).",
            original_bytes=len(full),
            truncated_bytes=cut,
        )
    )

    # 10. Plain text with a .png extension.
    path = "errors/no_es_imagen.png"
    ctx.path(path).write_bytes(b"Nota: este archivo no es una imagen PNG, es texto plano.\n" * 3)
    files.append(_file_entry("no_es_imagen", path, "png", "format", "Texto plano con extensión .png (sin firma PNG)."))

    # 11. Zip with garbage content and a .docx extension (unsupported format).
    path = "errors/documento.docx"
    content_types = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        b'<Override PartName="/word/document.xml" '
        b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        b"</Types>"
    )
    garbage = rng.integers(0, 256, size=4096, dtype="uint8").tobytes()
    ctx.path(path).write_bytes(
        _deterministic_zip([("[Content_Types].xml", content_types), ("word/document.xml", garbage)])
    )
    files.append(
        _file_entry(
            "documento_docx",
            path,
            "docx",
            "format",
            "Zip con apariencia de .docx y contenido basura: formato no soportado por el anonimizador.",
        )
    )
    return files
