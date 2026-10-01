"""Generates the fictitious test dataset and its manifest (ground truth).

Usage:
    uv run python -m test_bench.generate [--output test_data/generated] [--seed 33] [--only pdf_text images]

All personal data is made up. The faces come from sources with a documented free license
(see ``test_bench/faces.py`` and LICENSES.md).
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

from test_bench.context import Context
from test_bench.faces import FaceProvider
from test_bench.fake_data import FakeData
from test_bench.schema import Manifest

# Fixed order: the dataset is deterministic for a given seed.
MODULES = [
    "pdf_text",
    "pdf_metadata",
    "pdf_scanned",
    "images",
    "id_card",
    "screenshots",
    "face_scenes",
    "errors",
]

# Spanish labels for the console summary (the users read Spanish).
_TYPE_LABELS = {
    "email": "correo",
    "phone": "telefono",
    "name": "nombre",
    "address": "direccion",
    "face": "rostro",
    "signature": "firma",
    "text": "texto",
}
_LEVEL_LABELS = {"stress": "estres", "out_of_scope": "fuera_de_alcance"}


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for package in ("pymupdf", "pillow", "numpy", "opencv-python-headless", "piexif", "matplotlib"):
        try:
            out[package] = version(package)
        except Exception:  # noqa: BLE001 - informational
            out[package] = "?"
    return out


def _clean(root: Path) -> None:
    """Deletes a previous generation, only if the folder looks like one (has a manifest or is empty)."""
    if not root.exists():
        return
    if (root / "manifest.json").exists() or not any(root.iterdir()):
        shutil.rmtree(root)
    else:
        raise SystemExit(f"{root} existe y no parece un conjunto generado; no se borra. Use otra --output.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera el conjunto de prueba ficticio.")
    ap.add_argument("--output", type=Path, default=Path("test_data/generated"))
    ap.add_argument("--cache", type=Path, default=Path("test_data/cache"))
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--only", nargs="*", choices=MODULES, help="generar solo estos módulos")
    ap.add_argument("--no-downloads", action="store_true", help="no descargar rostros (usa caché o dibujados)")
    args = ap.parse_args(argv)

    root: Path = args.output
    _clean(root)
    root.mkdir(parents=True)
    ctx = Context(
        root=root,
        seed=args.seed,
        fake=FakeData(args.seed),
        faces=FaceProvider(args.cache / "faces", args.seed, allow_download=not args.no_downloads),
    )
    man = Manifest(root=str(root), seed=args.seed)
    for name in args.only or MODULES:
        start = time.perf_counter()
        module = importlib.import_module(f"test_bench.generators.{name}")
        files = module.generate(ctx)
        for f in files:
            man.add(f)
        print(f"  {name:<16} {len(files):>3} archivos  {time.perf_counter() - start:5.1f} s")

    man.name_list = ctx.fake.name_list()
    (root / "name_list.txt").write_text("\n".join(man.name_list) + "\n", encoding="utf-8")
    man.face_sources = ctx.faces.used_sources()
    man.versions = _versions()
    problems = man.validate()
    path = man.save()

    # Counted and sorted by the Spanish labels, so the summary keeps its original order.
    types = Counter(
        (_TYPE_LABELS.get(e.type, e.type), _LEVEL_LABELS.get(e.level, e.level)) for f in man.files for e in f.elements
    )
    print(f"\nManifiesto: {path}  ({len(man.files)} archivos)")
    for (type_label, level_label), n in sorted(types.items()):
        print(f"  {type_label:<10} {level_label:<17} {n:>4}")
    n_meta = sum(len(f.sensitive_metadata) for f in man.files)
    print(f"  metadatos sensibles: {n_meta}")
    if ctx.faces.synthetic_only:
        print(
            "\nATENCION: no hay rostros reales disponibles; se usaron rostros dibujados (no sirven para medir recall)."
        )
    if problems:
        print("\nPROBLEMAS EN EL MANIFIESTO:")
        for p in problems:
            print("  -", p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
