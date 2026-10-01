"""Builds the Windows executable: dist/LightAnonymizer/LightAnonymizer.exe and a zip of that folder.

Usage (from the repository root, on Windows, with every change committed):
    uv run --group build python scripts/build_exe.py
    uv run --group build python scripts/build_exe.py --allow-dirty    # test build of uncommitted changes

It checks the models first (YuNet in ``models/`` with its SHA-256, from download_models.py; the
PP-OCR models inside the installed ``rapidocr`` package, whose wheel is pinned by ``uv.lock``),
runs PyInstaller with ``packaging/light_anonymizer.spec`` (one folder, windowed), writes the
third-party license texts (``THIRD_PARTY_LICENSES/``, see collect_licenses.py) and ``LEEME.txt``
(how to open it, the license and where its exact source code is: the commit it was built from)
next to the executable, checks that nothing development-only or document-like ended up in the
bundle, and writes ``dist/LightAnonymizer-<version>-windows.zip``.

The AGPL obliges whoever hands the executable to someone else to offer its exact source code, so
a working copy with uncommitted changes is refused unless ``--allow-dirty`` is given; such a build
says so in ``LEEME.txt`` and must not be distributed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import re
import subprocess
import sys
import time
import tomllib
import zipfile
from datetime import date
from pathlib import Path

from collect_licenses import collect as collect_licenses
from download_models import DESTINATION as MODELS_DIR
from download_models import MODELS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # anonymizer.engine.ocr, for the OCR model paths

NAME = "LightAnonymizer"
SPEC = ROOT / "packaging" / "light_anonymizer.spec"
DIST = ROOT / "dist"
WORK = ROOT / "build"
REPOSITORY = "https://github.com/GonzaloFuentes1/light-anonymizer"
# Modules that must never be bundled (development tools, test bench, unused heavy libraries).
FORBIDDEN_MODULES = {"test_bench", "tests", "scripts", "matplotlib", "pypdfium2", "pytest", "_pytest", "tkinter"}
# Files and folders that must never be bundled: FFmpeg (LGPL, unused) and any test data or results.
FORBIDDEN_NAMES = ("opencv_videoio_ffmpeg", "test_data", "results", "test_bench")
# No document, image or data file belongs in the application: one would be a test or a user file.
FORBIDDEN_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".bmp", ".gif"}
FORBIDDEN_SUFFIXES |= {".doc", ".docx", ".xls", ".xlsx", ".odt", ".csv", ".log"}
# What the app's own data folders may contain.
UI_SUFFIXES = {".html", ".js", ".css", ".ttf", ".txt"}
DOTTED_RUT = re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[\dkK]\b")


def check_models() -> list[str]:
    problems = []
    for name, (_url, sha) in MODELS.items():
        path = MODELS_DIR / name
        if not path.is_file():
            problems.append(f"{path} is missing: run  uv run python scripts/download_models.py")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            problems.append(f"{path} has the wrong SHA-256: run  uv run python scripts/download_models.py")
    from anonymizer.engine.ocr import model_paths

    paths = model_paths()
    if paths is None:
        problems.append("the rapidocr package is not installed: run  uv sync")
    else:
        problems += [f"OCR model missing: {path}" for path in paths.values() if not path.is_file()]
    return problems


def git_state() -> tuple[str | None, bool]:
    """(commit of HEAD, whether the working copy has changes); (None, True) without git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return commit, bool(status.strip())


def check_bundle(app_dir: Path) -> list[str]:
    """Development-only modules, forbidden or unexpected files and RUTs that ended up in the bundle."""
    problems = []
    tocs = list(WORK.glob("light_anonymizer/PYZ-*.toc"))  # (archive path, [(module, source, type), ...])
    if not tocs:
        problems.append("PyInstaller's module list (PYZ-00.toc) was not found: cannot check the modules")
    own_sources = []
    for toc in tocs:
        _archive, entries = ast.literal_eval(toc.read_text(encoding="utf-8"))
        for module, source, _kind in entries:
            if module.split(".")[0] in FORBIDDEN_MODULES:
                problems.append(f"module bundled: {module}")
            if module.split(".")[0] == "anonymizer" and source:
                own_sources.append(Path(source))
    internal = app_dir / "_internal"
    expected_models = set(MODELS)
    for path in app_dir.rglob("*"):
        relative = path.relative_to(app_dir)
        if any(part.lower().startswith(FORBIDDEN_NAMES) for part in relative.parts):
            problems.append(f"file bundled: {relative}")
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"document or data file bundled: {relative}")
        if path.parent == internal / "models" and path.name not in expected_models:
            problems.append(f"unexpected file in models/: {relative}")
        if internal / "anonymizer" in path.parents and path.suffix.lower() not in UI_SUFFIXES:
            problems.append(f"unexpected file in the interface folder: {relative}")
    # Real-looking RUTs in the app's own code and data (not in third-party license texts).
    texts = own_sources + [p for p in (internal / "anonymizer").rglob("*") if p.suffix.lower() in {".html", ".js"}]
    for path in texts:
        if path.is_file() and DOTTED_RUT.search(path.read_text(encoding="utf-8", errors="replace")):
            problems.append(f"a dotted RUT appears in {path}")
    return problems


def write_readme(app_dir: Path, version: str, commit: str | None, dirty: bool) -> None:
    """LEEME.txt next to the executable, for whoever receives it (Spanish)."""
    if commit is None:
        origin = "desde una copia sin control de versiones: NO LA DISTRIBUYAS."
        source = REPOSITORY
    elif dirty:
        origin = f"desde el commit {commit} MÁS CAMBIOS SIN CONFIRMAR: es una compilación de prueba, NO LA DISTRIBUYAS."
        source = f"{REPOSITORY}/tree/{commit}"
    else:
        origin = f"desde el commit {commit}."
        source = f"{REPOSITORY}/tree/{commit}"
    text = f"""Anonimizador (Light Anonymizer) {version}
{"=" * (32 + len(version))}

Anonimiza documentos PDF e imágenes en tu computador: detecta datos personales (RUT, nombres,
correos, teléfonos, direcciones, rostros y otros), te deja revisarlos y exporta copias censuradas.
Funciona sin conexión: los documentos no salen del computador.

Cómo abrirlo
------------
1. Descomprime el .zip completo en una carpeta con una ruta corta y fuera de OneDrive, por
   ejemplo C:\\Anonimizador. No abras el programa desde dentro del .zip.
2. Abre LightAnonymizer.exe. La carpeta _internal es parte del programa: déjala junto al .exe.
3. Necesita Microsoft Edge WebView2 Runtime, que ya viene en Windows 10 y 11 actualizados.

Windows puede avisar "Windows protegió tu PC" porque el programa todavía no tiene firma digital.
Si lo recibiste de una fuente confiable, elige "Más información" y luego "Ejecutar de todos
modos". Si tu equipo tiene Control inteligente de aplicaciones o reglas de la institución, puede
que Windows no lo deje abrir: en ese caso pide ayuda a soporte informático.

Mientras trabajas, la aplicación guarda una copia de los documentos en la carpeta temporal de
Windows y la borra al cerrarse. Si se cierra de forma inesperada, esa copia se borra la próxima
vez que la abras.

Licencia
--------
© 2026 Gonzalo Fuentes. Este programa es software libre: puedes redistribuirlo y modificarlo
según los términos de la Licencia Pública General Affero de GNU (GNU AGPL), versión 3 o
posterior. Se distribuye SIN NINGUNA GARANTÍA, ni siquiera la garantía implícita de
COMERCIALIZACIÓN o de IDONEIDAD PARA UN PROPÓSITO PARTICULAR. El texto completo de la licencia
(en inglés) está en LICENSE.txt.

Incluye componentes de terceros con sus propias licencias: la lista y sus textos están en
THIRD_PARTY_LICENSES\\INDEX.txt.

Código fuente
-------------
Versión {version}, compilada el {date.today().isoformat()} {origin}
Código fuente de esta versión:
  {source}
Para compilarla en Windows: instala uv (https://docs.astral.sh/uv/) y, en la carpeta del código,
  uv sync --group build
  uv run python scripts/download_models.py
  uv run --group build python scripts/build_exe.py
(detalles en README.md, sección "Building the Windows executable").
Dónde está el código fuente de los componentes de terceros con licencias copyleft (PyMuPDF y
MuPDF, GEOS, certifi, tqdm): THIRD_PARTY_LICENSES\\INDEX.txt.
"""
    (app_dir / "LEEME.txt").write_text(text, encoding="utf-8-sig", newline="\r\n")  # BOM: Notepad shows the accents


def folder_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def write_zip(app_dir: Path, target: Path) -> None:
    target.unlink(missing_ok=True)
    partial = target.with_name(target.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(app_dir.name) / path.relative_to(app_dir))
    partial.replace(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Windows executable and its zip.")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="build from uncommitted changes (test builds only: do not hand out)"
    )
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        print("This script builds the Windows executable: run it on Windows.", file=sys.stderr)
        return 1
    if importlib.util.find_spec("PyInstaller") is None:
        print("PyInstaller is not installed: run  uv run --group build python scripts/build_exe.py", file=sys.stderr)
        return 1
    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        print(
            "The working copy has uncommitted changes (or is not a git repository): commit them, so that the\n"
            "executable can point to its exact source code, or pass --allow-dirty for a test build.",
            file=sys.stderr,
        )
        return 1
    problems = check_models()
    if problems:
        print("Cannot build:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

    started = time.perf_counter()
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--log-level", "WARN"]
    command += ["--distpath", str(DIST), "--workpath", str(WORK), str(SPEC)]
    if subprocess.run(command, cwd=ROOT).returncode != 0:
        print("PyInstaller failed (see its messages above).", file=sys.stderr)
        return 1
    app_dir = DIST / NAME
    exe = app_dir / f"{NAME}.exe"
    if not exe.is_file():
        print(f"PyInstaller finished but {exe} does not exist.", file=sys.stderr)
        return 1
    problems = collect_licenses(app_dir, WORK / "light_anonymizer" / "PYZ-00.toc")
    if problems:
        print("License texts missing:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    write_readme(app_dir, version, commit, dirty)
    problems = check_bundle(app_dir)
    if problems:
        print("The bundle contains forbidden content:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    archive = DIST / f"{NAME}-{version}-windows.zip"
    write_zip(app_dir, archive)
    mb = 1024 * 1024
    print(
        f"\nBuilt in {time.perf_counter() - started:.0f} s from {commit or 'no commit'}{' + changes' if dirty else ''}"
    )
    print(f"  executable: {exe}")
    print(f"  folder:     {folder_size(app_dir) / mb:.0f} MB")
    print(f"  zip:        {archive} ({archive.stat().st_size / mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
