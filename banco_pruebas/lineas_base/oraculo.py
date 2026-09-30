"""Línea base ``oraculo``: censura exactamente la verdad de terreno y elimina los metadatos.

Debe dar recall 100 % y cero fugas. Si no, la verdad de terreno (o sus coordenadas) está mal, o
el evaluador tiene un error.

- PDF: cada página se dibuja en su orientación visible a 200 ppp, se rellenan de negro los
  polígonos de todos los elementos con ``tipo != "texto"`` en capas visibles (``texto``,
  ``raster``, ``vector``; la capa ``oculto`` desaparece sola al rasterizar) y se arma un PDF nuevo
  de páginas imagen, del mismo tamaño visible y sin metadatos.
- Imágenes: se aplica la orientación EXIF, se rellenan los polígonos (en todos los cuadros de un
  TIFF multipágina) y se guarda en el mismo formato sin ningún metadato.
- Archivos con ``esperado == "error:<codigo>"``: se reporta ``error=<codigo>`` sin salida.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw, ImageOps, ImageSequence

from banco_pruebas.esquema import Archivo, Censura, Elemento, Manifiesto, ResultadoArchivo

DPI = 200
# Margen con que se agranda cada relleno, para que el borde suavizado al volver a dibujar la
# página (el evaluador la dibuja a 144 ppp) o la compresión JPEG no dejen píxeles fuera del color.
MARGEN_PDF_PT = 1.0
MARGEN_IMAGEN_PX = 2.0
MARGEN_JPEG_PX = 4.0
CAPAS_VISIBLES = ("texto", "raster", "vector")


class ErrorOraculo(Exception):
    """Archivo que el oráculo no puede procesar; ``codigo`` va al campo ``error`` del informe."""

    def __init__(self, codigo: str) -> None:
        super().__init__(codigo)
        self.codigo = codigo


def objetivos(archivo: Archivo) -> list[Elemento]:
    """Elementos que se rellenan: todo dato personal con polígono en una capa visible."""
    return [e for e in archivo.elementos if e.tipo != "texto" and e.poligono is not None and e.capa in CAPAS_VISIBLES]


def rellenar(dibujo: ImageDraw.ImageDraw, puntos: list[tuple[float, float]], color: Any, margen: float) -> None:
    """Rellena el polígono y lo agranda ``margen`` píxeles hacia afuera (trazo redondeado en el borde)."""
    dibujo.polygon(puntos, fill=color)
    if margen <= 0:
        return
    dibujo.line([*puntos, puntos[0]], fill=color, width=max(1, round(2 * margen)), joint="curve")
    for x, y in puntos:
        dibujo.ellipse([x - margen, y - margen, x + margen, y + margen], fill=color)


def _negro(modo: str) -> Any:
    return {"L": 0, "LA": (0, 255), "RGB": (0, 0, 0), "RGBA": (0, 0, 0, 255)}[modo]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _pdf(archivo: Archivo, entrada: Path, destino: Path) -> int:
    doc = pymupdf.open(entrada)
    try:
        if doc.needs_pass:
            raise ErrorOraculo("contrasena")
        nuevo = pymupdf.open()
        por_pagina: dict[int, list[Elemento]] = {}
        for e in objetivos(archivo):
            por_pagina.setdefault(e.pagina, []).append(e)
        for i, pagina in enumerate(doc):
            pix = pagina.get_pixmap(dpi=DPI, alpha=False, colorspace=pymupdf.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            visible = pagina.rect  # tamaño visible (ya rotado), origen en (0, 0)
            sx, sy = pix.width / visible.width, pix.height / visible.height
            # De espacio sin rotar a píxeles visibles: la misma transformación que usa el evaluador.
            matriz = pagina.rotation_matrix * pymupdf.Matrix(sx, sy)
            dibujo = ImageDraw.Draw(img)
            for e in por_pagina.get(i, []):
                assert e.poligono is not None
                puntos = [tuple(pymupdf.Point(x, y) * matriz) for x, y in e.poligono]
                rellenar(dibujo, puntos, (0, 0, 0), MARGEN_PDF_PT * sx)
            limpio = pymupdf.Pixmap(pymupdf.csRGB, img.width, img.height, img.tobytes(), False)
            hoja = nuevo.new_page(width=visible.width, height=visible.height)
            hoja.insert_image(hoja.rect, pixmap=limpio)
        n_paginas = doc.page_count
    finally:
        doc.close()
    nuevo.set_metadata({})
    nuevo.del_xml_metadata()
    destino.parent.mkdir(parents=True, exist_ok=True)
    nuevo.save(destino, garbage=4, deflate=True)
    nuevo.close()
    return n_paginas


# ---------------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------------

_MODOS_DIRECTOS = ("L", "LA", "RGB", "RGBA")


def _normalizar_modo(img: Image.Image) -> Image.Image:
    if img.mode in _MODOS_DIRECTOS:
        return img
    if img.mode == "P":
        return img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("1", "I;16", "I;16B", "I;16L", "I", "F"):
        return img.convert("L")
    if img.mode == "PA":
        return img.convert("RGBA")
    return img.convert("RGB")


def _sin_metadatos(img: Image.Image) -> Image.Image:
    """Copia de solo píxeles: sin ``info`` (EXIF, ICC, XMP, textos PNG, etiquetas TIFF)."""
    return Image.frombytes(img.mode, img.size, img.tobytes())


def _imagen(archivo: Archivo, entrada: Path, destino: Path) -> int:
    with Image.open(entrada) as original:
        formato = (original.format or archivo.formato).upper()
        cuadros = [ImageOps.exif_transpose(c.copy()) for c in ImageSequence.Iterator(original)]
    if formato == "MPO":
        formato, cuadros = "JPEG", cuadros[:1]
    margen = MARGEN_JPEG_PX if formato == "JPEG" else MARGEN_IMAGEN_PX
    limpios: list[Image.Image] = []
    for i, cuadro in enumerate(cuadros):
        img = _normalizar_modo(cuadro)
        if formato == "JPEG" and img.mode in ("RGBA", "LA"):
            img = img.convert(img.mode[:-1])
        img = _sin_metadatos(img)
        dibujo = ImageDraw.Draw(img)
        for e in objetivos(archivo):
            if e.pagina == i:
                assert e.poligono is not None
                rellenar(dibujo, [(x, y) for x, y in e.poligono], _negro(img.mode), margen)
        limpios.append(img)

    destino.parent.mkdir(parents=True, exist_ok=True)
    primero, resto = limpios[0], limpios[1:]
    if formato == "JPEG":
        primero.save(destino, "JPEG", quality=95, subsampling=0)
    elif formato == "PNG":
        primero.save(destino, "PNG")
    elif formato == "WEBP":
        opciones: dict[str, Any] = {"lossless": True}
        if resto:
            opciones.update(save_all=True, append_images=resto)
        primero.save(destino, "WEBP", **opciones)
    elif formato == "TIFF":
        primero.save(destino, "TIFF", compression="tiff_deflate", save_all=True, append_images=resto)
    else:
        primero.save(destino, formato)
    return len(limpios)


# ---------------------------------------------------------------------------


def procesar(archivo: Archivo, manifiesto: Manifiesto, carpeta: Path, detalles: dict[str, Any]) -> ResultadoArchivo:
    if archivo.esperado.startswith("error:"):
        return ResultadoArchivo(entrada=archivo.ruta, salida=None, error=archivo.esperado.removeprefix("error:"))
    entrada = Path(manifiesto.raiz) / archivo.ruta
    destino = carpeta / archivo.ruta
    try:
        if archivo.formato == "pdf":
            n_paginas = _pdf(archivo, entrada, destino)
        else:
            n_paginas = _imagen(archivo, entrada, destino)
    except ErrorOraculo as err:
        destino.unlink(missing_ok=True)
        return ResultadoArchivo(entrada=archivo.ruta, salida=None, error=err.codigo)
    except Exception as err:  # noqa: BLE001 - el informe registra el tipo de excepción
        destino.unlink(missing_ok=True)
        return ResultadoArchivo(entrada=archivo.ruta, salida=None, error=f"excepcion:{type(err).__name__}")
    censuras = [
        Censura(pagina=e.pagina, poligono=e.poligono, tipo=e.tipo, detector="oraculo", texto=e.valor)
        for e in objetivos(archivo)
        if e.poligono is not None
    ]
    return ResultadoArchivo(entrada=archivo.ruta, salida=archivo.ruta, censuras=censuras, paginas_procesadas=n_paginas)
