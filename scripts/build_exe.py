"""Builds the Windows executable: dist/LightAnonymizer/LightAnonymizer.exe and a zip of that folder.

Usage (from the repository root, on Windows, with every change committed):
    uv run --group build python scripts/build_exe.py
    uv run --group build python scripts/build_exe.py --installer      # also the installer (Inno Setup 6)
    uv run --group build python scripts/build_exe.py --allow-dirty    # test build of uncommitted changes

It checks the models first (YuNet in ``models/`` with its SHA-256, from download_models.py; the
PP-OCR models inside the installed ``rapidocr`` package, whose wheel is pinned by ``uv.lock``),
runs PyInstaller with ``packaging/light_anonymizer.spec`` (one folder, windowed), writes the
third-party license texts (``THIRD_PARTY_LICENSES/``, see collect_licenses.py) and ``LEEME.txt``
(how to open it, the license and where its exact source code is: the commit it was built from)
next to the executable, copies the user manual there (``docs/user-manual/manual-de-usuario.pdf``,
see build_manual_pdf.py), checks that nothing development-only or document-like ended up in the
bundle, and writes ``dist/LightAnonymizer-<version>-windows.zip`` and the build record the
installer needs (``dist/LightAnonymizer-build.json``). With ``--installer`` it then compiles
``dist/LightAnonymizer-<version>-setup.exe`` (see build_installer.py).

The AGPL obliges whoever hands the executable to someone else to offer its exact source code, so
a working copy with uncommitted changes is refused unless ``--allow-dirty`` is given; such a build
says so in ``LEEME.txt`` (and its installer in its own texts) and must not be distributed.
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

from build_installer import BUILD_INFO, find_iscc, write_build_info
from build_installer import build as build_installer
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
# The user manual (Spanish), next to LEEME.txt: the only PDF the bundle may hold, byte for byte this one.
MANUAL = ROOT / "docs" / "user-manual" / "manual-de-usuario.pdf"
# Modules that must never be bundled (development tools, test bench, unused heavy libraries).
FORBIDDEN_MODULES = {"test_bench", "tests", "scripts", "matplotlib", "pypdfium2", "pytest", "_pytest", "tkinter"}
# Files and folders that must never be bundled: FFmpeg (LGPL, unused) and any test data or results.
FORBIDDEN_NAMES = ("opencv_videoio_ffmpeg", "test_data", "results", "test_bench")
# No document, image or data file belongs in the application: one would be a test or a user file.
FORBIDDEN_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".bmp", ".gif"}
FORBIDDEN_SUFFIXES |= {".doc", ".docx", ".xls", ".xlsx", ".odt", ".csv", ".log"}
# What the app's own data folders may contain (the interface and the given-name dictionary).
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


def changed_during_build(commit: str | None, dirty: bool) -> bool:
    """Whether HEAD moved or the working copy changed since a clean build started."""
    return not dirty and git_state() != (commit, False)


def remove_previous_outputs(version: str) -> None:
    """This version's zip, installers (release or test) and build record: they describe an older build."""
    for path in [DIST / f"{NAME}-{version}-windows.zip", BUILD_INFO, *DIST.glob(f"{NAME}-{version}-setup*.exe")]:
        path.unlink(missing_ok=True)


def check_bundle(app_dir: Path, manual: bytes | None = None) -> list[str]:
    """Development-only modules, forbidden or unexpected files and RUTs that ended up in the bundle.

    ``manual``: the user manual's bytes; a file with them, named like it, next to the executable is
    the one document the bundle may hold.
    """
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
        is_manual = manual is not None and relative == Path(MANUAL.name) and path.read_bytes() == manual
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not is_manual:
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

Cómo instalarlo y abrirlo
-------------------------
Se entrega de dos formas; las dos traen el mismo programa.

Con el instalador ({NAME}-{version}-setup.exe):
1. Ábrelo y sigue los pasos. No pide permisos de administrador: se instala solo para tu usuario,
   en %LOCALAPPDATA%\\Programs\\{NAME}.
2. Abre el Anonimizador desde el menú Inicio.
Una versión nueva se instala encima de la anterior. Para desinstalarlo: Configuración >
Aplicaciones > Aplicaciones instaladas (en Windows 10, Aplicaciones y características) >
Anonimizador.

Con el .zip ({NAME}-{version}-windows.zip):
1. Descomprime el .zip completo en una carpeta con una ruta corta y fuera de OneDrive, por
   ejemplo C:\\Anonimizador. No abras el programa desde dentro del .zip.
2. Abre {NAME}.exe. La carpeta _internal es parte del programa: déjala junto al .exe.

El manual de usuario está en {MANUAL.name}, en la carpeta del programa (con el
instalador, también en el menú Inicio: "Manual del Anonimizador").

Necesita Microsoft Edge WebView2 Runtime, que ya viene en Windows 10 y 11 actualizados.

Windows puede avisar "Windows protegió su PC" (editor desconocido) al abrir el instalador o el
programa, porque todavía no tienen firma digital. Si lo recibiste de una fuente confiable, elige
"Más información" y luego "Ejecutar de todas formas". Si tu equipo tiene Control inteligente de
aplicaciones o reglas de la institución, puede que Windows no lo deje abrir: en ese caso pide
ayuda a soporte informático.

Mientras trabajas, la aplicación guarda una copia de los documentos en la carpeta temporal de
Windows y la borra al cerrarse. Si se cierra de forma inesperada, esa copia se borra la próxima
vez que la abras. En %LOCALAPPDATA%\\Anonimizador guarda su registro técnico y sus estimaciones
de tiempo (nunca documentos): desinstalar el programa no los borra, salvo que lo pidas.

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
(con --installer también genera el instalador, para lo que se necesita Inno Setup 6; detalles en
README.md, sección "Building the Windows executable").
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
    parser.add_argument("--installer", action="store_true", help="also compile the installer (needs Inno Setup 6)")
    parser.add_argument("--iscc", help="with --installer: path of ISCC.exe (default: looked up)")
    args = parser.parse_args(argv)
    if args.iscc and not args.installer:
        print("--iscc has no effect without --installer: no installer will be compiled.", file=sys.stderr)
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
    iscc = find_iscc(args.iscc) if args.installer else None
    if args.installer and iscc is None:
        print("Inno Setup 6 (ISCC.exe) was not found: see scripts/build_installer.py.", file=sys.stderr)
        return 1
    problems = check_models()
    if not MANUAL.is_file():
        problems.append(f"{MANUAL} is missing: run  uv run --with markdown python scripts/build_manual_pdf.py")
    if problems:
        print("Cannot build:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

    manual = MANUAL.read_bytes()  # read before PyInstaller, like everything else from the working copy
    started = time.perf_counter()
    remove_previous_outputs(version)
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
    # Nothing is read from the working copy after this point: a change made while PyInstaller ran
    # may or may not be in the bundle, so such a build cannot vouch for its commit.
    if changed_during_build(commit, dirty):
        print("The working copy or HEAD changed during the build: it is marked as a test build.", file=sys.stderr)
        dirty = True
    write_readme(app_dir, version, commit, dirty)
    (app_dir / MANUAL.name).write_bytes(manual)
    problems = check_bundle(app_dir, manual)
    if problems:
        print("The bundle contains forbidden content:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    archive = DIST / f"{NAME}-{version}-windows.zip"
    write_zip(app_dir, archive)
    write_build_info(BUILD_INFO, version, commit, dirty, app_dir)
    mb = 1024 * 1024
    print(
        f"\nBuilt in {time.perf_counter() - started:.0f} s from {commit or 'no commit'}{' + changes' if dirty else ''}"
    )
    print(f"  executable: {exe}")
    print(f"  folder:     {folder_size(app_dir) / mb:.0f} MB")
    print(f"  zip:        {archive} ({archive.stat().st_size / mb:.0f} MB)")
    if iscc is not None:
        return build_installer(iscc, app_dir, BUILD_INFO, allow_dirty=args.allow_dirty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
