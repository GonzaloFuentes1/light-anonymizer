"""Where the bundled resources live, both in development and in the packaged executable.

In development the resources are read from the repository: ``anonymizer/ui``, ``models/`` and
the ``rapidocr`` package installed in the virtual environment. The Windows executable is built
by PyInstaller in one-folder mode (``packaging/light_anonymizer.spec``): ``sys.frozen`` is set
and every resource is unpacked under ``sys._MEIPASS`` (the ``_internal`` folder next to the
``.exe``) with the same relative layout, so the same relative path works in both cases::

    <root>/anonymizer/ui/...         interface (HTML, JS, CSS, fonts and their OFL texts)
    <root>/models/*.onnx             YuNet face model
    <root>/rapidocr/models/*.onnx    PP-OCR models (in development: inside site-packages)
    <root>/LICENSE, LICENSES.md

Next to the executable, not under the root, the build also leaves LEEME.txt (Spanish: how to open
it, license, source code), LICENSE.txt, LICENSES.md and THIRD_PARTY_LICENSES/ (scripts/build_exe.py).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True inside the executable built by PyInstaller."""
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """The repository root in development; the bundle folder (``sys._MEIPASS``) in the executable."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def resource(*parts: str) -> Path:
    """A bundled file or folder, given by its path relative to :func:`resource_root`."""
    return resource_root().joinpath(*parts)


def package_dir(name: str) -> Path | None:
    """Folder of an installed package (None when it is not installed).

    In the executable the package's data files are collected into ``<bundle>/<name>``, while its
    code may live only in the bundle's compressed archive (so ``spec.origin`` is not a real file).
    """
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if is_frozen():
        return resource(name)
    return Path(spec.origin).parent if spec.origin else None
