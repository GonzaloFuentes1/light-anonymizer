"""Evaluates the output of a redaction system against the ground truth (metric of docs/metrics.md).

Usage:
    uv run python -m test_bench.evaluate --manifest test_data/generated/manifest.json \\
        --redaction-report output/report.json --outputs-dir output/files --report-dir output/evaluation

Writes ``evaluation.json`` and ``evaluation.md`` in the ``--report-dir`` folder.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from test_bench.evaluation import evaluate
from test_bench.evaluation.report import write_report
from test_bench.schema import Manifest, RedactionReport


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evalúa un informe de censura contra el manifiesto del conjunto de prueba."
    )
    ap.add_argument("--manifest", type=Path, required=True, help="manifiesto (manifest.json) del conjunto de prueba")
    ap.add_argument("--redaction-report", type=Path, required=True, help="informe de censura en JSON")
    ap.add_argument("--outputs-dir", type=Path, required=True, help="carpeta con los archivos censurados")
    ap.add_argument(
        "--report-dir", type=Path, required=True, help="carpeta donde escribir evaluation.json y evaluation.md"
    )
    ap.add_argument("--processes", type=int, default=None, help="procesos en paralelo (por defecto, automático)")
    ap.add_argument("--strict", action="store_true", help="salir con código 1 si el veredicto es NO APROBADO")
    args = ap.parse_args(argv)

    start = time.perf_counter()
    man = Manifest.load(args.manifest)
    report = RedactionReport.load(args.redaction_report)
    result = evaluate(man, report, args.outputs_dir, processes=args.processes)
    result["evaluation_duration_s"] = round(time.perf_counter() - start, 2)
    json_path, md_path = write_report(result, args.report_dir)

    v, s = result["verdict"], result["summary"]
    print(f"Veredicto: {'APROBADO' if v['passed'] else 'NO APROBADO'}")
    for m in v["reasons"]:
        print(f"  - {m}")
    recall = "—" if s["recall"] is None else f"{100 * s['recall']:.1f} %"
    print(
        f"Recall {recall}; fugas {s['leaks']} (críticas {s['critical_leaks']}); "
        f"metadatos con fuga {s['metadata_leaks']} de {s['metadata']}"
    )
    print(f"Evaluación en {result['evaluation_duration_s']} s -> {json_path}, {md_path}")
    return 1 if args.strict and not v["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
