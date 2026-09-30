"""Corre una línea base sobre el conjunto de prueba y escribe su salida y su informe de censura.

Uso:
    uv run python -m banco_pruebas.linea_base <identidad|oraculo|cuaderno> --manifiesto <manifiesto.json> --salida <dir>

Escribe los archivos de salida en ``<dir>/archivos/<misma ruta relativa>`` y el informe en
``<dir>/informe.json`` (formato ``esquema.InformeCensura``). En el informe, ``salida`` es la ruta
relativa a ``<dir>/archivos`` (igual a la de entrada); ``detalles.carpeta_archivos`` lo indica.
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

from banco_pruebas.esquema import InformeCensura, Manifiesto, ResultadoArchivo
from banco_pruebas.lineas_base import SISTEMAS

CARPETA_ARCHIVOS = "archivos"


def _preparar(salida: Path) -> Path:
    """Crea ``<salida>/archivos``; borra una corrida anterior solo si parece serlo (tiene informe)."""
    carpeta = salida / CARPETA_ARCHIVOS
    if carpeta.exists():
        if (salida / "informe.json").exists() or not any(carpeta.iterdir()):
            shutil.rmtree(carpeta)
        else:
            raise SystemExit(f"{carpeta} existe y no parece una corrida anterior; no se borra. Use otra --salida.")
    carpeta.mkdir(parents=True)
    return carpeta


def _uno(sistema: str, archivo, manifiesto: Manifiesto, carpeta: Path) -> tuple[ResultadoArchivo, dict]:
    modulo = importlib.import_module(f"banco_pruebas.lineas_base.{sistema}")
    detalles: dict = {}
    inicio = time.perf_counter()
    try:
        resultado: ResultadoArchivo = modulo.procesar(archivo, manifiesto, carpeta, detalles)
    except Exception as err:  # noqa: BLE001 - un error de la línea base no detiene la corrida
        resultado = ResultadoArchivo(entrada=archivo.ruta, salida=None, error=f"excepcion:{type(err).__name__}")
    resultado.tiempo_s = round(time.perf_counter() - inicio, 4)
    return resultado, detalles


def ejecutar(
    sistema: str, manifiesto: Manifiesto, salida: Path, ids: set[str] | None = None, procesos: int = 1
) -> InformeCensura:
    """Procesa todos los archivos del manifiesto con la línea base ``sistema`` y guarda el informe."""
    if sistema not in SISTEMAS:
        raise ValueError(f"línea base desconocida: {sistema}")
    importlib.import_module(f"banco_pruebas.lineas_base.{sistema}")  # falla temprano si no existe
    salida.mkdir(parents=True, exist_ok=True)
    carpeta = _preparar(salida)
    versiones = {"python": sys.version.split()[0]}
    for paquete in ("pymupdf", "pillow"):
        try:
            versiones[paquete] = version(paquete)
        except Exception:  # noqa: BLE001 - informativo
            versiones[paquete] = "?"
    informe = InformeCensura(
        sistema=sistema,
        detalles={"carpeta_archivos": CARPETA_ARCHIVOS, "versiones": versiones},
    )
    archivos = [a for a in manifiesto.archivos if not ids or a.id in ids]
    if procesos > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(procesos) as pool:
            pares = list(
                pool.map(
                    _uno, [sistema] * len(archivos), archivos, [manifiesto] * len(archivos), [carpeta] * len(archivos)
                )
            )
    else:
        pares = [_uno(sistema, a, manifiesto, carpeta) for a in archivos]
    for resultado, detalles in pares:
        informe.resultados.append(resultado)
        for clave, valor in detalles.items():
            if isinstance(valor, dict):
                informe.detalles.setdefault(clave, {}).update(valor)
            else:
                informe.detalles[clave] = valor
    informe.guardar(salida / "informe.json")
    return informe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Corre una línea base (identidad, oraculo, cuaderno, prototipo) sobre el conjunto."
    )
    ap.add_argument("sistema", choices=SISTEMAS)
    ap.add_argument("--manifiesto", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--id", nargs="*", help="procesar solo estos archivos")
    ap.add_argument("--procesos", type=int, default=1, help="archivos en paralelo")
    args = ap.parse_args(argv)
    man = Manifiesto.cargar(args.manifiesto)
    inicio = time.perf_counter()
    informe = ejecutar(args.sistema, man, args.salida, set(args.id or []) or None, args.procesos)
    errores = Counter(r.error for r in informe.resultados if r.error)
    n_censuras = sum(len(r.censuras) for r in informe.resultados)
    print(
        f"{args.sistema}: {len(informe.resultados)} archivos, {n_censuras} censuras, "
        f"{time.perf_counter() - inicio:.1f} s -> {args.salida / 'informe.json'}"
    )
    for codigo, n in sorted(errores.items()):
        print(f"  error {codigo:<28} {n:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
