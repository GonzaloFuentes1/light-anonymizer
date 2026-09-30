"""Genera el conjunto de prueba ficticio y su manifiesto (verdad de terreno).

Uso:
    uv run python -m banco_pruebas.generar [--salida datos_prueba/generado] [--semilla 33] [--solo pdf_texto imagenes]

Todos los datos personales son inventados. Los rostros vienen de fuentes con licencia libre
documentada (ver ``banco_pruebas/rostros.py`` y LICENCIAS.md).
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Manifiesto
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.rostros import ProveedorRostros

# Orden fijo: el conjunto es determinista para una semilla dada.
MODULOS = [
    "pdf_texto",
    "pdf_metadatos",
    "pdf_escaneado",
    "imagenes",
    "cedula",
    "pantallazos",
    "rostros_escenas",
    "errores",
]


def _versiones() -> dict[str, str]:
    salida = {"python": sys.version.split()[0]}
    for paquete in ("pymupdf", "pillow", "numpy", "opencv-python-headless", "piexif", "matplotlib"):
        try:
            salida[paquete] = version(paquete)
        except Exception:  # noqa: BLE001 - informativo
            salida[paquete] = "?"
    return salida


def _limpiar(raiz: Path) -> None:
    """Borra una generación anterior, solo si la carpeta parece ser una (tiene manifiesto o está vacía)."""
    if not raiz.exists():
        return
    if (raiz / "manifiesto.json").exists() or not any(raiz.iterdir()):
        shutil.rmtree(raiz)
    else:
        raise SystemExit(f"{raiz} existe y no parece un conjunto generado; no se borra. Use otra --salida.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera el conjunto de prueba ficticio.")
    ap.add_argument("--salida", type=Path, default=Path("datos_prueba/generado"))
    ap.add_argument("--cache", type=Path, default=Path("datos_prueba/cache"))
    ap.add_argument("--semilla", type=int, default=33)
    ap.add_argument("--solo", nargs="*", choices=MODULOS, help="generar solo estos módulos")
    ap.add_argument("--sin-descargas", action="store_true", help="no descargar rostros (usa caché o dibujados)")
    args = ap.parse_args(argv)

    raiz: Path = args.salida
    _limpiar(raiz)
    raiz.mkdir(parents=True)
    ctx = Contexto(
        raiz=raiz,
        semilla=args.semilla,
        fict=Ficticios(args.semilla),
        rostros=ProveedorRostros(args.cache / "rostros", args.semilla, permitir_descarga=not args.sin_descargas),
    )
    man = Manifiesto(raiz=str(raiz), semilla=args.semilla)
    for nombre in args.solo or MODULOS:
        inicio = time.perf_counter()
        modulo = importlib.import_module(f"banco_pruebas.generadores.{nombre}")
        archivos = modulo.generar(ctx)
        for a in archivos:
            man.agregar(a)
        print(f"  {nombre:<16} {len(archivos):>3} archivos  {time.perf_counter() - inicio:5.1f} s")

    man.lista_nombres = ctx.fict.lista_nombres()
    (raiz / "lista_nombres.txt").write_text("\n".join(man.lista_nombres) + "\n", encoding="utf-8")
    man.fuentes_rostros = ctx.rostros.fuentes_usadas()
    man.versiones = _versiones()
    problemas = man.validar()
    ruta = man.guardar()

    tipos = Counter((e.tipo, e.nivel) for a in man.archivos for e in a.elementos)
    print(f"\nManifiesto: {ruta}  ({len(man.archivos)} archivos)")
    for (tipo, nivel), n in sorted(tipos.items()):
        print(f"  {tipo:<10} {nivel:<17} {n:>4}")
    n_meta = sum(len(a.metadatos_sensibles) for a in man.archivos)
    print(f"  metadatos sensibles: {n_meta}")
    if ctx.rostros.solo_sinteticos:
        print(
            "\nATENCION: no hay rostros reales disponibles; se usaron rostros dibujados (no sirven para medir recall)."
        )
    if problemas:
        print("\nPROBLEMAS EN EL MANIFIESTO:")
        for p in problemas:
            print("  -", p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
