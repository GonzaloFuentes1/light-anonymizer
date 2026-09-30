"""Experimento con documentos públicos reales: qué logra hoy la lógica del cuaderno.

Para cada PDF de ``datos_prueba/reales`` aplica el cuaderno tal cual (detectar + censurar) y
revisa el archivo censurado:

- datos que el cuaderno detectó y que siguen recuperables dentro del archivo (en flujos que
  ninguna página usa, como el contenido original antes de censurar);
- datos que siguen visibles en el texto de la página (no detectados o no censurados);
- si queda autor u otros metadatos;
- páginas sin capa de texto, que el cuaderno no puede procesar.

Por privacidad, el informe solo muestra conteos, nunca los valores encontrados.

Uso:
    uv run python -m banco_pruebas.experimento_reales [--carpeta datos_prueba/reales] [--salida resultados/reales]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pymupdf

from banco_pruebas.evaluacion.pdf import cadenas_pdf
from banco_pruebas.lineas_base import cuaderno


def _texto_de_flujo(doc: pymupdf.Document, xref: int) -> str:
    try:
        return "\n".join(cadenas_pdf(doc.xref_stream(xref)))
    except Exception:  # noqa: BLE001 - flujos de imagen o dañados
        return ""


def analizar(ruta: Path, carpeta_salida: Path) -> dict:
    fila: dict = {"archivo": ruta.as_posix(), "paginas": 0}
    doc = pymupdf.open(ruta)
    fila["paginas"] = doc.page_count
    fila["paginas_sin_texto"] = sum(1 for p in doc if len(p.get_text().strip()) < 20)
    fila["metadatos_entrada"] = sorted(
        k for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")
    )
    doc.close()

    if not cuaderno.tiene_texto(ruta):
        fila["resultado"] = "el cuaderno lo rechaza: no tiene texto (escaneo)"
        return fila

    hallazgos = cuaderno.detectar(ruta)
    salida = carpeta_salida / ruta.name
    cuaderno.censurar(ruta, salida, hallazgos)
    fila["hallazgos"] = len(hallazgos)
    fila["por_tipo"] = {t: sum(1 for h in hallazgos if h[1] == t) for t in cuaderno.PATRONES}

    doc = pymupdf.open(salida)
    visible = "\n".join(p.get_text() for p in doc)
    usados = {x for p in doc for x in p.get_contents()}
    huerfanos = "\n".join(
        _texto_de_flujo(doc, x) for x in range(1, doc.xref_length()) if doc.xref_is_stream(x) and x not in usados
    )
    valores = {h[2] for h in hallazgos}
    fila["detectados_aun_visibles"] = sum(1 for v in valores if v in visible)
    fila["detectados_recuperables_en_el_archivo"] = sum(1 for v in valores if v in huerfanos)
    restos = {t: len(re.findall(p, visible)) for t, p in cuaderno.PATRONES.items()}
    fila["patrones_en_texto_visible_tras_censura"] = {t: n for t, n in restos.items() if n}
    fila["metadatos_salida"] = sorted(
        k for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")
    )
    doc.close()
    return fila


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--carpeta", type=Path, default=Path("datos_prueba/reales"))
    ap.add_argument("--salida", type=Path, default=Path("resultados/reales"))
    args = ap.parse_args(argv)
    args.salida.mkdir(parents=True, exist_ok=True)
    filas = []
    vistos: set[bytes] = set()
    for ruta in sorted(args.carpeta.rglob("*.pdf")):
        contenido = ruta.read_bytes()
        if contenido in vistos:
            continue
        vistos.add(contenido)
        try:
            filas.append(analizar(ruta, args.salida))
        except Exception as ex:  # noqa: BLE001 - se reporta y se sigue
            filas.append({"archivo": ruta.as_posix(), "resultado": f"error: {type(ex).__name__}"})
    (args.salida / "experimento.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")

    procesados = [f for f in filas if "hallazgos" in f]
    print(
        f"{len(filas)} PDF distintos; {len(procesados)} procesados por el cuaderno; {len(filas) - len(procesados)} rechazados"
    )
    print(f"{'archivo':<62} {'pág':>3} {'hall':>4} {'recup':>5} {'visib':>5}  metadatos tras censura")
    for f in filas:
        nombre = Path(f["archivo"]).name[:60]
        if "hallazgos" not in f:
            print(f"{nombre:<62} {f.get('paginas', 0):>3}  {f.get('resultado', '')}")
            continue
        print(
            f"{nombre:<62} {f['paginas']:>3} {f['hallazgos']:>4} {f['detectados_recuperables_en_el_archivo']:>5} "
            f"{f['detectados_aun_visibles']:>5}  {', '.join(f['metadatos_salida']) or '-'}"
        )
    total = sum(f["hallazgos"] for f in procesados)
    recuperables = sum(f["detectados_recuperables_en_el_archivo"] for f in procesados)
    con_meta = sum(1 for f in procesados if f["metadatos_salida"])
    print(f"\nDatos detectados y 'censurados': {total}; recuperables dentro del PDF censurado: {recuperables}")
    print(f"PDF censurados que conservan metadatos: {con_meta} de {len(procesados)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
