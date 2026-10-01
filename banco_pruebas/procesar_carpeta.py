"""Anonimiza con el prototipo todos los archivos de una carpeta (sin manifiesto) y arma láminas.

Uso:
    uv run python -m banco_pruebas.procesar_carpeta <carpeta> --salida resultados/gore [--procesos 6]

Para cada archivo escribe ``<salida>/anonimizados/<nombre>``, una lámina por página
(original a la izquierda, anonimizado a la derecha) en ``<salida>/laminas/`` y un resumen en
``<salida>/resumen.md``. Como no hay lista de nombres, usa las palabras del nombre del archivo
que parecen apellidos (por ejemplo "ROJAS TAPIA INFORME.pdf" -> "Rojas Tapia").
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

from banco_pruebas.lienzo import fuente
from banco_pruebas.lineas_base import prototipo
from banco_pruebas.visualizar import paginas_visibles

_PALABRAS_GENERICAS = {"informe", "prueba", "ficticio", "de", "del", "la", "el", "y", "final", "mensual", "pdf"}


def lista_desde_nombre(ruta: Path) -> tuple[str, ...]:
    palabras = [p for p in re.split(r"[\s_\-.]+", ruta.stem) if p and p.lower() not in _PALABRAS_GENERICAS]
    palabras = [p for p in palabras if p.isalpha() and len(p) >= 3]
    return (" ".join(p.capitalize() for p in palabras),) if palabras else ()


def _lamina(original: Image.Image, anonimizado: Image.Image | None, titulo: str) -> Image.Image:
    alto = 1100
    a = original.resize((max(1, int(original.width * alto / original.height)), alto))
    b = anonimizado.resize(a.size) if anonimizado is not None else Image.new("RGB", a.size, (230, 230, 230))
    hoja = Image.new("RGB", (a.width * 2 + 48, alto + 70), "white")
    d = ImageDraw.Draw(hoja)
    d.text((16, 14), f"Original — {titulo}", font=fuente("sans_negrita", 20), fill=(20, 20, 20))
    d.text((a.width + 32, 14), "Anonimizado (prototipo)", font=fuente("sans_negrita", 20), fill=(20, 20, 20))
    hoja.paste(a, (16, 56))
    hoja.paste(b, (a.width + 32, 56))
    return hoja


def procesar_uno(ruta: Path, salida: Path) -> dict:
    lista = lista_desde_nombre(ruta)
    destino = salida / "anonimizados" / ruta.name
    inicio = time.perf_counter()
    fila: dict = {"archivo": ruta.name, "lista": lista}
    try:
        if ruta.suffix.lower() == ".pdf":
            censuras, n = prototipo._procesar_pdf(ruta, destino, lista, todas_url=False)
        else:
            censuras, n = prototipo._procesar_imagen(ruta, destino, lista, todas_url=False)
    except Exception as err:  # noqa: BLE001 - se informa y se sigue con los demás
        fila["error"] = f"{type(err).__name__}: {err}"
        return fila
    fila["segundos"] = round(time.perf_counter() - inicio, 1)
    fila["paginas"] = n
    fila["zonas"] = dict(Counter(c.tipo for c in censuras))
    # Verificación rápida de la salida: patrones sobre la capa de texto y metadatos.
    if destino.suffix.lower() == ".pdf":
        with pymupdf.open(destino) as doc:
            texto = "\n".join(p.get_text() for p in doc)
            fila["restos_en_texto"] = len(prototipo.buscar(texto, lista, todas_url=False))
            fila["metadatos"] = [k for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")]
    # Láminas de cada página
    formato = "pdf" if ruta.suffix.lower() == ".pdf" else ruta.suffix.lower().lstrip(".")
    antes, despues = paginas_visibles(ruta, formato), paginas_visibles(destino, formato)
    carpeta_laminas = salida / "laminas"
    carpeta_laminas.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(antes):
        lam = _lamina(img, despues[i] if i < len(despues) else None, f"{ruta.stem}, página {i + 1}")
        lam.save(carpeta_laminas / f"{ruta.stem}_p{i + 1:02d}.jpg", quality=85)
    return fila


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("carpeta", type=Path)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--procesos", type=int, default=6)
    args = ap.parse_args(argv)
    archivos = sorted(
        p
        for p in args.carpeta.iterdir()
        if p.suffix.lower() in (".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
    )
    (args.salida / "anonimizados").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(min(args.procesos, len(archivos))) as pool:
        filas = list(pool.map(procesar_uno, archivos, [args.salida] * len(archivos)))
    lineas = ["# Documentos procesados con el prototipo", "",
              "| Archivo | Páginas | Zonas censuradas | Datos que siguen en el texto | Metadatos que quedan | Tiempo |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for f in filas:
        if "error" in f:
            lineas.append(f"| {f['archivo']} | — | error: {f['error']} | — | — | — |")
            continue
        zonas = ", ".join(f"{k} {v}" for k, v in sorted(f["zonas"].items())) or "ninguna"
        lineas.append(
            f"| {f['archivo']} | {f['paginas']} | {zonas} | {f.get('restos_en_texto', '—')} | "
            f"{', '.join(f.get('metadatos', [])) or 'ninguno'} | {f['segundos']} s |"
        )
    lineas += ["", "Lista de nombres usada (tomada del nombre de cada archivo): "
               + "; ".join(f"{f['archivo']}: {', '.join(f['lista']) or '(ninguno)'}" for f in filas)]  # fmt: skip
    (args.salida / "resumen.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
