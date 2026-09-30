"""Evalúa la salida de un sistema de censura contra la verdad de terreno (métrica de docs/metricas.md).

Uso:
    uv run python -m banco_pruebas.evaluar --manifiesto datos_prueba/generado/manifiesto.json \\
        --informe salida/informe.json --salida-archivos salida/archivos --reporte salida/evaluacion

Escribe ``evaluacion.json`` y ``evaluacion.md`` en la carpeta de ``--reporte``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from banco_pruebas.esquema import InformeCensura, Manifiesto
from banco_pruebas.evaluacion import evaluar
from banco_pruebas.evaluacion.reporte import escribir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evalúa un informe de censura contra el manifiesto del conjunto de prueba."
    )
    ap.add_argument("--manifiesto", type=Path, required=True)
    ap.add_argument("--informe", type=Path, required=True, help="InformeCensura en JSON")
    ap.add_argument("--salida-archivos", type=Path, required=True, help="carpeta con los archivos censurados")
    ap.add_argument("--reporte", type=Path, required=True, help="carpeta donde escribir evaluacion.json y .md")
    ap.add_argument("--procesos", type=int, default=None, help="procesos en paralelo (por defecto, automático)")
    ap.add_argument("--estricto", action="store_true", help="salir con código 1 si el veredicto es NO APROBADO")
    args = ap.parse_args(argv)

    inicio = time.perf_counter()
    man = Manifiesto.cargar(args.manifiesto)
    informe = InformeCensura.cargar(args.informe)
    resultado = evaluar(man, informe, args.salida_archivos, procesos=args.procesos)
    resultado["duracion_evaluacion_s"] = round(time.perf_counter() - inicio, 2)
    ruta_json, ruta_md = escribir(resultado, args.reporte)

    v, s = resultado["veredicto"], resultado["resumen"]
    print(f"Veredicto: {'APROBADO' if v['aprobado'] else 'NO APROBADO'}")
    for m in v["motivos"]:
        print(f"  - {m}")
    recall = "—" if s["recall"] is None else f"{100 * s['recall']:.1f} %"
    print(
        f"Recall {recall}; fugas {s['fugas']} (críticas {s['fugas_criticas']}); "
        f"metadatos con fuga {s['fugas_metadatos']} de {s['metadatos']}"
    )
    print(f"Evaluación en {resultado['duracion_evaluacion_s']} s -> {ruta_json}, {ruta_md}")
    return 1 if args.estricto and not v["aprobado"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
