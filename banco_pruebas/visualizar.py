"""Superpone la verdad de terreno sobre cada página, para revisar a ojo que los polígonos calzan.

Uso:
    uv run python -m banco_pruebas.visualizar datos_prueba/generado/manifiesto.json --salida superposiciones/ [--id ID ...]

Colores: rojo = dato personal, azul = texto neutro, verde = cara completa, amarillo = núcleo del
rostro, magenta = capa oculta (se dibuja con línea punteada aunque no se vea en la página).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageOps, ImageSequence

from banco_pruebas.esquema import Archivo, Manifiesto

ESCALA_PDF = 2.0  # 144 ppp


def paginas_visibles(ruta: Path, formato: str) -> list[Image.Image]:
    """Cada página tal como se ve (imágenes con EXIF aplicado; PDF en su orientación visible)."""
    if formato == "pdf":
        doc = pymupdf.open(ruta)
        salida = []
        for p in doc:
            pix = p.get_pixmap(matrix=pymupdf.Matrix(ESCALA_PDF, ESCALA_PDF), alpha=False)
            salida.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        doc.close()
        return salida
    img = Image.open(ruta)
    if getattr(img, "n_frames", 1) > 1:
        return [ImageOps.exif_transpose(f.copy()).convert("RGB") for f in ImageSequence.Iterator(img)]
    return [ImageOps.exif_transpose(img).convert("RGB")]


def a_visual(archivo: Archivo, indice: int, poligono: list[list[float]]) -> list[tuple[float, float]]:
    """Lleva un polígono de coordenadas de página (sin rotar) a píxeles de la página visible."""
    pag = archivo.paginas[indice]
    if archivo.formato != "pdf":
        return [(x, y) for x, y in poligono]
    w, h, r = pag.ancho, pag.alto, pag.rotacion % 360
    puntos = []
    for x, y in poligono:
        if r == 90:
            x, y = h - y, x
        elif r == 180:
            x, y = w - x, h - y
        elif r == 270:
            x, y = y, w - x
        puntos.append((x * ESCALA_PDF, y * ESCALA_PDF))
    return puntos


def _punteado(d: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color, paso: float = 8) -> None:
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True):
        largo = max(1.0, float(np.hypot(x1 - x0, y1 - y0)))
        for t in np.arange(0, largo, paso * 2):
            a, b = t / largo, min(1.0, (t + paso) / largo)
            d.line(
                [(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                fill=color,
                width=2,
            )


def superponer(archivo: Archivo, raiz: Path) -> list[Image.Image]:
    paginas = paginas_visibles(raiz / archivo.ruta, archivo.formato)
    for i, img in enumerate(paginas):
        capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(capa)
        for e in archivo.elementos:
            if e.pagina != i or e.poligono is None:
                continue
            pts = a_visual(archivo, i, e.poligono)
            if e.capa == "oculto":
                _punteado(d, pts, (220, 0, 220, 255))
                continue
            if e.tipo == "rostro":
                d.polygon(pts, outline=(0, 170, 0, 255), width=3)
                if e.nucleo:
                    d.polygon(a_visual(archivo, i, e.nucleo), outline=(240, 200, 0, 255), width=2)
            elif e.tipo == "texto":
                d.polygon(pts, outline=(40, 90, 255, 200), width=1)
            else:
                d.polygon(pts, fill=(255, 0, 0, 50), outline=(255, 0, 0, 255), width=2)
        paginas[i] = Image.alpha_composite(img.convert("RGBA"), capa).convert("RGB")
    return paginas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dibuja la verdad de terreno sobre los archivos del conjunto.")
    ap.add_argument("manifiesto", type=Path)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--id", nargs="*", help="solo estos archivos")
    ap.add_argument("--max-lado", type=int, default=1600)
    args = ap.parse_args(argv)
    man = Manifiesto.cargar(args.manifiesto)
    args.salida.mkdir(parents=True, exist_ok=True)
    for a in man.archivos:
        if args.id and a.id not in args.id:
            continue
        if a.esperado != "procesar":
            continue
        for i, img in enumerate(superponer(a, Path(man.raiz))):
            img.thumbnail((args.max_lado, args.max_lado))
            img.save(args.salida / f"{a.id}_p{i}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
