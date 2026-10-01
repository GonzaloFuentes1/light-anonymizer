"""Writes THIRD_PARTY_LICENSES/ next to the executable: the license texts of everything bundled.

Usage (normally run by build_exe.py after PyInstaller):
    uv run --group build python scripts/collect_licenses.py dist/LightAnonymizer

A Python distribution is bundled when one of its modules is in PyInstaller's module list
(``build/light_anonymizer/PYZ-00.toc``) or one of its packages, extension modules, ``*.libs`` or
``*.dist-info`` folders is under ``_internal/``. For each one this copies every license, copying,
notice and authors file it installed (in its dist-info or in its package folder, such as
onnxruntime's ``ThirdPartyNotices.txt`` or OpenCV's ``LICENSE-3RD-PARTY.txt``) into
``THIRD_PARTY_LICENSES/<name>-<version>/``, plus ``packaging/licenses/packages/<name>/`` for the
texts its wheel does not ship. ``packaging/licenses/components/`` (models, the Python runtime's
libraries) and the runtime's own ``LICENSE.txt`` go in as well. ``INDEX.txt`` lists every
component with its version, license and where its source code is (the sdist recorded in
``uv.lock``), and says where to get the source of the copyleft parts.

It fails (exit code 1) when a bundled distribution has no license text at all.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata as md
import re
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "packaging" / "licenses"
PYZ_TOC = ROOT / "build" / "light_anonymizer" / "PYZ-00.toc"
FOLDER = "THIRD_PARTY_LICENSES"
PYTHON_SOURCE = "https://www.python.org/downloads/source/"
OWN_DISTRIBUTION = "light-anonymizer"
LICENSE_FILE = re.compile(
    r"^(licen[cs]e|copying|notice|authors|thirdpartynotices|third[-_]party|privacy)[^/]*$", re.IGNORECASE
)
# Copyleft parts: the license obliges us to say where their source code is.
COPYLEFT_SOURCES = {
    "pymupdf": "AGPL-3.0: PyMuPDF and MuPDF (MuPDF 1.28.2: https://github.com/ArtifexSoftware/mupdf/tree/1.28.2)",
    "shapely": "LGPL-2.1 for the bundled GEOS {geos} DLLs (Shapely.libs/): https://download.osgeo.org/geos/",
    "certifi": "MPL-2.0: cacert.pem ships as is; the package source is the sdist below",
    "tqdm": "MPL-2.0 (and MIT): the bundle has only its bytecode; the source is the sdist below",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def bundled_top_levels(app_dir: Path, toc: Path) -> set[str]:
    """Top-level import names whose code or files are in the bundle."""
    names: set[str] = set()
    if toc.is_file():
        _archive, entries = ast.literal_eval(toc.read_text(encoding="utf-8"))
        names |= {module.split(".")[0] for module, _source, _kind in entries}
    for entry in (app_dir / "_internal").iterdir():
        name = entry.name
        if name.endswith(".dist-info"):
            continue
        name = re.sub(r"\.libs$", "", name)  # numpy.libs, Shapely.libs
        name = name.split(".")[0]  # _cffi_backend.cp312-win_amd64.pyd
        names.add(name)
    return names


def bundled_distributions(app_dir: Path, toc: Path) -> dict[str, md.Distribution]:
    """Normalized name -> distribution, for every third-party distribution in the bundle."""
    owners = {top.lower(): dists for top, dists in md.packages_distributions().items()}
    found: dict[str, md.Distribution] = {}
    names = set()
    for top in bundled_top_levels(app_dir, toc):
        names.update(owners.get(top.lower(), []))
    for info in (app_dir / "_internal").glob("*.dist-info"):
        names.add(info.name.rsplit("-", 1)[0])
    for name in names:
        if normalize(name) == OWN_DISTRIBUTION:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        found[normalize(dist.metadata["Name"])] = dist
    return dict(sorted(found.items()))


def license_name(dist: md.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    text = (meta.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 80:
        return text
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    return "; ".join(classifiers) or "see the license files"


def lock_sources() -> dict[str, str]:
    """Normalized name -> URL of the source (sdist) of the locked version."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    sources = {}
    for package in lock.get("package", []):
        sdist = package.get("sdist") or {}
        if sdist.get("url"):
            sources.setdefault(f"{normalize(package['name'])}=={package['version']}", sdist["url"])
    return sources


def source_url(dist: md.Distribution, sources: dict[str, str]) -> str:
    """The sdist of the locked version, else the project's source or home page."""
    key = f"{normalize(dist.metadata['Name'])}=={dist.version}"
    if key in sources:
        return sources[key]
    urls = dict(entry.split(", ", 1) for entry in dist.metadata.get_all("Project-URL") or [] if ", " in entry)
    for label in ("Source", "Sources", "Source Code", "Repository", "Code", "Homepage", "Changelog"):
        if label in urls:
            return urls[label]
    return dist.metadata.get("Home-page") or ""


def license_files(dist: md.Distribution) -> list[tuple[Path, str]]:
    """(installed file, relative name to keep) for each license-like file of ``dist``."""
    picked = []
    for file in dist.files or []:
        parts = Path(str(file)).parts
        if file.suffix in {".py", ".pyc", ".pyi"} or "tests" in parts or not LICENSE_FILE.match(parts[-1]):
            continue
        path = Path(dist.locate_file(file))
        if path.is_file():
            # dist-info/licenses/numpy/x/LICENSE.txt -> numpy/x/LICENSE.txt; cv2/LICENSE.txt stays
            relative = Path(*parts[1:]) if parts[0].endswith(".dist-info") else Path(*parts)
            if relative.parts and relative.parts[0] == "licenses":
                relative = Path(*relative.parts[1:])
            picked.append((path, relative.as_posix()))
    return picked


def copy_tree(source: Path, target: Path) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_file():
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            count += 1
    return count


def geos_version() -> str:
    try:
        import shapely

        return shapely.geos_version_string
    except Exception:  # noqa: BLE001 - only used in a note
        return "(unknown version)"


def collect(app_dir: Path, toc: Path = PYZ_TOC) -> list[str]:
    """Writes ``app_dir/THIRD_PARTY_LICENSES``; returns the problems found (empty when complete)."""
    out = app_dir / FOLDER
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    problems: list[str] = []
    if not toc.is_file():
        problems.append(f"{toc} not found: cannot tell which Python modules are bundled")
    sources = lock_sources()
    rows: list[tuple[str, str, str, str]] = []
    copyleft: list[str] = []

    for key, dist in bundled_distributions(app_dir, toc).items():
        name, version = dist.metadata["Name"], dist.version
        folder = out / f"{key}-{version}"
        count = 0
        seen: set[str] = set()  # the same text installed twice (OpenCV: cv2/ and dist-info/)
        for path, relative in license_files(dist):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            destination = folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            count += 1
        if (VENDORED / "packages" / key).is_dir():
            count += copy_tree(VENDORED / "packages" / key, folder)
        if key == "pymupdf":  # its wheel only says "Dual Licensed - GNU AFFERO GPL 3.0 or Artifex..."
            shutil.copyfile(ROOT / "LICENSE", folder / "AGPL-3.0.txt")
            count += 1
        if not count:
            problems.append(f"{name} {version}: no license text (add it to packaging/licenses/packages/{key}/)")
        source = source_url(dist, sources)
        rows.append((name, version, license_name(dist), source))
        if key in COPYLEFT_SOURCES:
            copyleft.append(f"- {name} {version}: {COPYLEFT_SOURCES[key].format(geos=geos_version())}\n  {source}")

    runtime = out / f"python-{sys.version.split()[0]}"
    runtime_license = Path(sys.base_prefix) / "LICENSE.txt"
    if runtime_license.is_file():
        runtime.mkdir()
        shutil.copyfile(runtime_license, runtime / "LICENSE.txt")
    else:
        problems.append(f"{runtime_license} not found: the Python runtime's license is missing")
    copy_tree(VENDORED / "components" / "python", runtime)
    copy_tree(VENDORED / "components" / "models", out / "models")
    fonts = out / "fonts"
    fonts.mkdir()
    for text in sorted((ROOT / "anonymizer" / "ui" / "fonts").glob("OFL-*.txt")):
        shutil.copyfile(text, fonts / text.name)
    rows += [
        ("Python runtime", runtime.name.removeprefix("python-"), "PSF-2.0 and others (see its folder)", PYTHON_SOURCE),
        ("YuNet model (face_detection_yunet_2023mar)", "2023mar", "MIT", "https://github.com/opencv/opencv_zoo"),
        ("PP-OCR models (inside rapidocr)", "PP-OCRv6", "Apache-2.0", "https://github.com/PaddlePaddle/PaddleOCR"),
        ("Atkinson Hyperlegible, IBM Plex Mono fonts", "", "OFL-1.1", "fonts/"),
    ]
    (out / "INDEX.txt").write_text(index_text(rows, copyleft), encoding="utf-8", newline="\r\n")
    return problems


def index_text(rows: list[tuple[str, str, str, str]], copyleft: list[str]) -> str:
    width = max(len(name) for name, *_ in rows)
    lines = [
        "Licencias de los componentes de terceros / Third-party licenses",
        "",
        "Light Anonymizer itself is free software under the GNU AGPL-3.0-or-later (LICENSE.txt, next",
        "to the executable). It bundles the components below; the folder of each one holds the",
        "license texts it ships with. The source column is the source package of the bundled version.",
        "",
        f"{'Component':<{width}}  {'Version':<12}  {'License':<40}  Source",
        f"{'-' * width}  {'-' * 12}  {'-' * 40}  {'-' * 6}",
    ]
    lines += [f"{name:<{width}}  {version:<12}  {lic:<40}  {source}" for name, version, lic, source in rows]
    lines += [
        "",
        "Source code of the copyleft components",
        "--------------------------------------",
        *copyleft,
        "",
        "Other notes",
        "-----------",
        "- The Microsoft Visual C++ runtime (vcruntime140*.dll, msvcp140*.dll, ucrtbase.dll and the",
        "  api-ms-win-*.dll forwarders) is redistributed under Microsoft's distributable-code terms,",
        "  quoted in python-*/LICENSE.txt.",
        "- OpenCV (cv2) links Intel IPP ICV; its terms are in opencv-python-headless-*/LICENSE-3RD-PARTY.txt.",
        "- pywebview bundles Microsoft's WebView2 SDK (pywebview-*/webview2-sdk/).",
        "- MuPDF, inside PyMuPDF, compiles in the libraries in pymupdf-*/mupdf-thirdparty/.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    app_dir = Path(args[0]) if args else ROOT / "dist" / "LightAnonymizer"
    if not (app_dir / "_internal").is_dir():
        print(f"{app_dir} is not a built application folder (no _internal/).", file=sys.stderr)
        return 1
    problems = collect(app_dir)
    if problems:
        print("License texts missing:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"Wrote {app_dir / FOLDER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
