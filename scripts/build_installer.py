"""Builds the Windows installer, dist/LightAnonymizer-<version>-setup.exe, with Inno Setup 6.

Usage (on Windows, after build_exe.py):
    uv run python scripts/build_installer.py
    uv run python scripts/build_installer.py --iscc "C:\\path\\to\\ISCC.exe"
    uv run --group build python scripts/build_exe.py --installer      # executable, zip and installer

Inno Setup 6 (https://jrsoftware.org/isinfo.php) must be installed, for example with
``winget install --id JRSoftware.InnoSetup -e``; ISCC.exe is looked up in ``--iscc``, the ``ISCC``
environment variable, the PATH and Inno Setup's usual folders.

It compiles ``packaging/installer.iss`` from the one-folder build in ``dist/LightAnonymizer`` and the
record build_exe.py writes next to it (``dist/LightAnonymizer-build.json``: version, commit, whether
the working copy had changes, SHA-256 of the executable), after checking that the folder is that
build and carries LEEME.txt and the license texts. The installer gets the build's version.

The AGPL rules of build_exe.py apply: a build of uncommitted changes gives an installer marked
"compilación de prueba: no distribuir" (welcome page, window title, file properties, installed
apps list). The installer's own sources (``INSTALLER_SOURCES``: the script, the LICENSE it shows,
this file) are source code too: if they differ from those of the build's commit, committed or not,
this refuses, unless ``--allow-dirty`` is given (a test installer, marked the same way). Other
changes, such as documentation committed after the build, do not matter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "LightAnonymizer"
DIST = ROOT / "dist"
APP_DIR = DIST / NAME
BUILD_INFO = DIST / f"{NAME}-build.json"
ISS = ROOT / "packaging" / "installer.iss"
# What must travel with the program (the AGPL's notices and the source-code offer).
REQUIRED = (f"{NAME}.exe", "LEEME.txt", "LICENSE.txt", "LICENSES.md", "THIRD_PARTY_LICENSES/INDEX.txt")
# What goes into the installer besides the build folder: they must be those of the build's commit.
INSTALLER_SOURCES = ("packaging/installer.iss", "LICENSE", "scripts/build_installer.py")


def numeric_version(version: str) -> str:
    """The four numbers Windows file properties need: "0.1.0" -> "0.1.0.0", "2.0rc1" -> "2.0.0.0"."""
    match = re.match(r"\d+(?:\.\d+)*", version)
    numbers = [int(n) for n in match.group().split(".")][:4] if match else []
    return ".".join(str(n) for n in numbers + [0] * (4 - len(numbers)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_build_info(path: Path, version: str, commit: str | None, dirty: bool, exe: Path) -> None:
    """Called by build_exe.py: what the installer needs to know about the build it packs."""
    info = {"version": version, "commit": commit, "dirty": dirty, "exe_sha256": sha256(exe)}
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def read_build_info(path: Path, app_dir: Path) -> tuple[dict | None, list[str]]:
    """(the build record, problems): the folder must be the build the record describes."""
    if not path.is_file():
        return None, [f"{path} is missing: build the executable first (scripts/build_exe.py)"]
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        missing = [key for key in ("version", "commit", "dirty", "exe_sha256") if key not in info]
    except (ValueError, TypeError) as exc:
        info, missing = None, [str(exc)]
    if info is None or missing:
        return None, [f"{path} is not a valid build record ({', '.join(missing)}): rebuild with scripts/build_exe.py"]
    problems = [f"{app_dir / name} is missing" for name in REQUIRED if not (app_dir / name).is_file()]
    exe = app_dir / f"{NAME}.exe"
    if exe.is_file() and sha256(exe) != info["exe_sha256"]:
        problems.append(f"{exe} is not the one {path.name} describes: rebuild with scripts/build_exe.py")
    return info, problems


def sources_changed(commit: str, root: Path = ROOT) -> bool:
    """Whether the installer's own sources differ from those of ``commit`` (committed or not)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", commit, "--", *INSTALLER_SOURCES], cwd=root, capture_output=True
        )
    except OSError:
        return True
    return result.returncode != 0  # 1: they differ; 128: no git, or an unknown commit


def is_test_build(info: dict, changed: bool, allow_dirty: bool) -> tuple[bool, str | None]:
    """(whether the installer is a test build, the reason to refuse to compile it, if any).

    ``changed``: whether the installer's sources differ from those of the build's commit.
    """
    if info["dirty"] or info["commit"] is None:
        return True, None  # the build itself is a test build: its installer just says so too
    if not changed:
        return False, None
    if allow_dirty:
        return True, None
    return True, (
        f"The installer's sources ({', '.join(INSTALLER_SOURCES)}) differ from those of commit "
        f"{info['commit']}, which the executable was built from: the installer could not be rebuilt "
        "from that commit.\nCommit the changes and rebuild the executable, or pass --allow-dirty for a "
        "test installer."
    )


def find_iscc(explicit: str | None = None) -> Path | None:
    """ISCC.exe (Inno Setup 6's command-line compiler), or None."""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    candidates = [os.environ.get("ISCC"), shutil.which("iscc")]
    if os.environ.get("LOCALAPPDATA"):  # per-user installation (winget's default)
        candidates.append(str(Path(os.environ["LOCALAPPDATA"], "Programs", "Inno Setup 6", "ISCC.exe")))
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        if os.environ.get(variable):
            candidates.append(str(Path(os.environ[variable], "Inno Setup 6", "ISCC.exe")))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def installer_name(version: str) -> str:
    return f"{NAME}-{version}-setup"


def iscc_command(
    iscc: Path, version: str, app_dir: Path, output_dir: Path, test_build: bool, extra: dict[str, str] | None = None
) -> list[str]:
    """ISCC's command line: the values installer.iss expects as /D defines, plus the output."""
    defines = {"AppVersion": version, "AppNumericVersion": numeric_version(version), "SourceDir": str(app_dir)}
    if test_build:
        defines["TestBuild"] = "1"
    defines.update(extra or {})
    command = [str(iscc), "/Q"]
    command += [f"/D{key}={value}" for key, value in defines.items()]
    command += [f"/O{output_dir}", f"/F{installer_name(version)}", str(ISS)]
    return command


def build(iscc: Path, app_dir: Path = APP_DIR, info_path: Path = BUILD_INFO, allow_dirty: bool = False) -> int:
    info, problems = read_build_info(info_path, app_dir)
    if problems:
        print("Cannot build the installer:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    changed = info["commit"] is None or sources_changed(info["commit"])
    test_build, refusal = is_test_build(info, changed, allow_dirty)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    version = info["version"]
    target = app_dir.parent / f"{installer_name(version)}.exe"
    started = time.perf_counter()
    if subprocess.run(iscc_command(iscc, version, app_dir, app_dir.parent, test_build), cwd=ROOT).returncode != 0:
        print("Inno Setup could not compile the installer (see its messages above).", file=sys.stderr)
        return 1
    if not target.is_file():
        print(f"Inno Setup finished but {target} does not exist.", file=sys.stderr)
        return 1
    print(f"\nInstaller compiled in {time.perf_counter() - started:.0f} s")
    print(f"  installer:  {target} ({target.stat().st_size / (1024 * 1024):.0f} MB)")
    if test_build:
        print("  TEST BUILD: marked 'compilación de prueba: no distribuir'. Do not hand it out.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the Windows installer from dist/LightAnonymizer.")
    parser.add_argument("--iscc", help="path of ISCC.exe (default: ISCC, the PATH or Inno Setup 6's usual folders)")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="compile from a changed working copy (test installer: do not hand out)",
    )
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        print("This script builds the Windows installer: run it on Windows.", file=sys.stderr)
        return 1
    iscc = find_iscc(args.iscc)
    if iscc is None:
        print(
            "Inno Setup 6 was not found (ISCC.exe): install it with\n"
            "  winget install --id JRSoftware.InnoSetup -e\n"
            "or pass its path with --iscc.",
            file=sys.stderr,
        )
        return 1
    return build(iscc, allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    sys.exit(main())
