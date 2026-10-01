"""Runs a baseline over the test set and writes its output and its redaction report.

Usage:
    uv run python -m test_bench.baseline <identity|oracle|notebook|prototype> --manifest <manifest.json> --output <dir>

Writes the output files to ``<dir>/files/<same relative path>`` and the report to
``<dir>/report.json`` (``schema.RedactionReport`` format). In the report, ``output`` is the path
relative to ``<dir>/files`` (the same as the input path); ``details.files_dir`` says so.
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

from test_bench.baselines import SYSTEMS
from test_bench.evaluation.report import spanish_label
from test_bench.schema import FileResult, Manifest, RedactionReport

FILES_DIR = "files"


def _prepare(output: Path) -> Path:
    """Creates ``<output>/files``; deletes a previous run only if it looks like one (it has a report)."""
    folder = output / FILES_DIR
    if folder.exists():
        if (output / "report.json").exists() or not any(folder.iterdir()):
            shutil.rmtree(folder)
        else:
            raise SystemExit(f"{folder} existe y no parece una corrida anterior; no se borra. Use otra --output.")
    folder.mkdir(parents=True)
    return folder


def _one(system: str, file_entry, manifest: Manifest, folder: Path) -> tuple[FileResult, dict]:
    module = importlib.import_module(f"test_bench.baselines.{system}")
    details: dict = {}
    start = time.perf_counter()
    try:
        result: FileResult = module.process(file_entry, manifest, folder, details)
    except Exception as err:  # noqa: BLE001 - a baseline error does not stop the run
        result = FileResult(input=file_entry.path, output=None, error=f"exception:{type(err).__name__}")
    result.time_s = round(time.perf_counter() - start, 4)
    return result, details


def run(
    system: str, manifest: Manifest, output: Path, ids: set[str] | None = None, processes: int = 1
) -> RedactionReport:
    """Processes every file of the manifest with the ``system`` baseline and saves the report."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown baseline: {system}")
    importlib.import_module(f"test_bench.baselines.{system}")  # fail early if it does not exist
    output.mkdir(parents=True, exist_ok=True)
    folder = _prepare(output)
    versions = {"python": sys.version.split()[0]}
    for package in ("pymupdf", "pillow"):
        try:
            versions[package] = version(package)
        except Exception:  # noqa: BLE001 - informational
            versions[package] = "?"
    report = RedactionReport(
        system=system,
        details={"files_dir": FILES_DIR, "versions": versions},
    )
    entries = [f for f in manifest.files if not ids or f.id in ids]
    if processes > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(processes) as pool:
            pairs = list(
                pool.map(_one, [system] * len(entries), entries, [manifest] * len(entries), [folder] * len(entries))
            )
    else:
        pairs = [_one(system, f, manifest, folder) for f in entries]
    for result, details in pairs:
        report.results.append(result)
        for key, value in details.items():
            if isinstance(value, dict):
                report.details.setdefault(key, {}).update(value)
            else:
                report.details[key] = value
    report.save(output / "report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Corre una línea base (identity, oracle, notebook, prototype) sobre el conjunto."
    )
    ap.add_argument("system", choices=SYSTEMS)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--id", nargs="*", help="procesar solo estos archivos")
    ap.add_argument("--processes", type=int, default=1, help="archivos en paralelo")
    args = ap.parse_args(argv)
    man = Manifest.load(args.manifest)
    start = time.perf_counter()
    report = run(args.system, man, args.output, set(args.id or []) or None, args.processes)
    errors = Counter(r.error for r in report.results if r.error)
    n_redactions = sum(len(r.redactions) for r in report.results)
    print(
        f"{args.system}: {len(report.results)} archivos, {n_redactions} censuras, "
        f"{time.perf_counter() - start:.1f} s -> {args.output / 'report.json'}"
    )
    # error codes shown with their Spanish label (only the code before ':'), in the original order
    labelled: Counter[str] = Counter()
    for error, n in errors.items():
        code, sep, rest = error.partition(":")
        labelled[spanish_label(code) + sep + rest] += n
    for code, n in sorted(labelled.items()):
        print(f"  error {code:<28} {n:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
