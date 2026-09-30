"""Archivos que el anonimizador debe rechazar con un código de error comprensible.

Ninguno lleva datos personales ni elementos: lo que se evalúa es el código de error
(``esperado = "error:<codigo>"``). Como no se pueden abrir (o no deben procesarse), la lista
de páginas va vacía: el número de páginas es desconocido para el evaluador y no hay
elementos que indexar. ``Manifiesto.validar`` lo acepta porque solo revisa las páginas de
los elementos.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

import pymupdf

from banco_pruebas.contexto import Contexto, guardar_imagen
from banco_pruebas.esquema import Archivo
from banco_pruebas.ficticios import FRASES_ADMINISTRATIVAS
from banco_pruebas.lienzo import Lienzo, ruta_fuente

MODULO = "errores"
CATEGORIA = "errores"
PREFIJO = "err"

# Contraseñas ficticias (no son datos personales); se dejan en las etiquetas para poder abrir el archivo.
CONTRASENA_USUARIO = "clave-ficticia-2026"
CONTRASENA_PROPIETARIO = "propietario-ficticio-2026"

FRACCION_TRUNCADO = 0.40
FECHA_ZIP = (2026, 3, 12, 9, 15, 0)


def _pdf_neutro(rng: Any, paginas: int, titulo: str) -> pymupdf.Document:
    """PDF válido con texto administrativo neutro (sin datos personales)."""
    doc = pymupdf.open()
    for n in range(paginas):
        pag = doc.new_page(width=612, height=792)
        pag.insert_font(fontname="dvs", fontfile=str(ruta_fuente("serif")))
        pag.insert_text((64, 80), f"{titulo} (página {n + 1} de {paginas})", fontname="dvs", fontsize=13)
        y = 120.0
        for i in rng.permutation(len(FRASES_ADMINISTRATIVAS)):
            pag.insert_text((64, y), FRASES_ADMINISTRATIVAS[int(i)], fontname="dvs", fontsize=10)
            y += 17
        pag.draw_rect(pymupdf.Rect(64, y + 10, 548, y + 120), color=(0.3, 0.3, 0.3), width=0.6)
    doc.subset_fonts()
    return doc


def _archivo(nombre: str, ruta: str, formato: str, codigo: str, descripcion: str, **etiquetas: Any) -> Archivo:
    return Archivo(
        id=f"{PREFIJO}_{nombre}",
        ruta=ruta,
        formato=formato,
        categoria=CATEGORIA,
        descripcion=descripcion,
        paginas=[],
        esperado=f"error:{codigo}",
        etiquetas=dict(etiquetas),
    )


def _zip_determinista(entradas: list[tuple[str, bytes]]) -> bytes:
    """Zip con fechas fijas, para que los bytes no dependan del momento de generación."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for nombre, datos in entradas:
            info = zipfile.ZipInfo(nombre, date_time=FECHA_ZIP)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, datos)
    return buf.getvalue()


def generar(ctx: Contexto) -> list[Archivo]:
    rng = ctx.rng(MODULO)
    archivos: list[Archivo] = []

    # 5. PDF protegido con contraseña de usuario (AES-256).
    ruta = "errores/protegido_contrasena.pdf"
    doc = _pdf_neutro(rng, 1, "Oficio reservado")
    doc.save(
        ctx.ruta(ruta),
        garbage=3,
        deflate=True,
        no_new_id=True,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw=CONTRASENA_USUARIO,
        owner_pw=CONTRASENA_PROPIETARIO,
    )
    doc.close()
    archivos.append(
        _archivo(
            "protegido_contrasena",
            ruta,
            "pdf",
            "contrasena",
            "PDF cifrado con AES-256 y contraseña de usuario: no se puede abrir sin ella.",
            cifrado="AES-256",
            contrasena_usuario=CONTRASENA_USUARIO,
        )
    )

    # 6. PDF válido truncado al 40 %.
    ruta = "errores/corrupto.pdf"
    doc = _pdf_neutro(rng, 3, "Informe técnico")
    completo = doc.tobytes(garbage=3, deflate=True, use_objstms=1, no_new_id=True)
    doc.close()
    corte = int(len(completo) * FRACCION_TRUNCADO)
    ctx.ruta(ruta).write_bytes(completo[:corte])
    archivos.append(
        _archivo(
            "corrupto",
            ruta,
            "pdf",
            "corrupto",
            "PDF válido de 3 páginas (con flujos de objetos) truncado al 40 % de sus bytes: sin tabla xref, "
            'trailer ni árbol de páginas; PyMuPDF lo "repara" y queda con 0 páginas.',
            bytes_originales=len(completo),
            bytes_truncado=corte,
        )
    )

    # 7. Archivo vacío.
    ruta = "errores/vacio.pdf"
    ctx.ruta(ruta).write_bytes(b"")
    archivos.append(_archivo("vacio", ruta, "pdf", "vacio", "Archivo .pdf de 0 bytes."))

    # 8. Texto plano con extensión .pdf.
    ruta = "errores/no_es_pdf.pdf"
    texto = (
        "Este archivo no es un PDF: es texto plano con extensión .pdf.\n" + "\n".join(FRASES_ADMINISTRATIVAS[:4]) + "\n"
    )
    ctx.ruta(ruta).write_bytes(texto.encode("utf-8"))
    archivos.append(
        _archivo("no_es_pdf", ruta, "pdf", "formato", "Texto plano UTF-8 con extensión .pdf (sin cabecera %PDF).")
    )

    # 9. JPEG truncado.
    ruta = "errores/imagen_corrupta.jpg"
    lz = Lienzo.nuevo(900, 600, fondo=(250, 250, 246))
    y = 80
    for i in rng.permutation(len(FRASES_ADMINISTRATIVAS))[:8]:
        lz.escribir(40, y, FRASES_ADMINISTRATIVAS[int(i)][:60], tam=22, registrar=False)
        y += 50
    destino = ctx.ruta(ruta)
    guardar_imagen(lz.img, destino, "jpg", quality=88)
    completo = destino.read_bytes()
    corte = int(len(completo) * FRACCION_TRUNCADO)
    destino.write_bytes(completo[:corte])
    archivos.append(
        _archivo(
            "imagen_corrupta",
            ruta,
            "jpg",
            "corrupto",
            "JPEG válido truncado al 40 %: la cabecera se lee, pero la decodificación falla (sin marcador EOI).",
            bytes_originales=len(completo),
            bytes_truncado=corte,
        )
    )

    # 10. Texto plano con extensión .png.
    ruta = "errores/no_es_imagen.png"
    ctx.ruta(ruta).write_bytes(b"Nota: este archivo no es una imagen PNG, es texto plano.\n" * 3)
    archivos.append(_archivo("no_es_imagen", ruta, "png", "formato", "Texto plano con extensión .png (sin firma PNG)."))

    # 11. Zip con contenido basura y extensión .docx (formato no soportado).
    ruta = "errores/documento.docx"
    tipos = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        b'<Override PartName="/word/document.xml" '
        b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        b"</Types>"
    )
    basura = rng.integers(0, 256, size=4096, dtype="uint8").tobytes()
    ctx.ruta(ruta).write_bytes(_zip_determinista([("[Content_Types].xml", tipos), ("word/document.xml", basura)]))
    archivos.append(
        _archivo(
            "documento_docx",
            ruta,
            "docx",
            "formato",
            "Zip con apariencia de .docx y contenido basura: formato no soportado por el anonimizador.",
        )
    )
    return archivos
