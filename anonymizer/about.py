"""What the "Acerca de" dialog shows: name, version, copyright, license, source and components.

These are the "appropriate legal notices" the AGPL asks an interactive program to display. The
LICENSE text is read from the installed files: next to the package, in the repository root (a
development checkout), in the PyInstaller bundle folder or in the package metadata. When none is
found the dialog still shows everything else and points to the source.

User-facing values are Spanish (license name, what each component is used for).
"""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from pathlib import Path

from anonymizer import __version__

log = logging.getLogger(__name__)

APP_NAME = "Light Anonymizer"
DISTRIBUTION = "light-anonymizer"
COPYRIGHT = "© 2026 Gonzalo Fuentes"
LICENSE_NAME = "GNU AGPL v3 o posterior"
LICENSE_ID = "AGPL-3.0-or-later"
SOURCE_URL = "https://github.com/GonzaloFuentes1/light-anonymizer"
# The components that ship with the application and their licenses (see LICENSES.md).
COMPONENTS: tuple[dict[str, str], ...] = (
    {"name": "PyMuPDF (MuPDF)", "license": "AGPL-3.0", "use": "Leer, censurar y limpiar los PDF"},
    {"name": "OpenCV", "license": "Apache-2.0", "use": "Procesar imágenes, rostros y códigos QR"},
    {"name": "RapidOCR", "license": "Apache-2.0", "use": "Leer el texto de las imágenes"},
    {"name": "Modelos PP-OCR (PaddleOCR)", "license": "Apache-2.0", "use": "Modelos de lectura de texto"},
    {"name": "ONNX Runtime", "license": "MIT", "use": "Ejecutar los modelos en este computador"},
    {"name": "YuNet (opencv_zoo)", "license": "MIT", "use": "Modelo de detección de rostros"},
    {"name": "NumPy", "license": "BSD-3-Clause", "use": "Cálculo con imágenes"},
    {"name": "Pillow", "license": "MIT-CMU", "use": "Leer y escribir imágenes"},
    {"name": "FastAPI y Starlette", "license": "MIT y BSD-3-Clause", "use": "Servidor local de la aplicación"},
    {"name": "Uvicorn", "license": "BSD-3-Clause", "use": "Servidor local de la aplicación"},
    {"name": "Pydantic", "license": "MIT", "use": "Validación de datos"},
    {"name": "pywebview", "license": "BSD-3-Clause", "use": "Ventana de la aplicación"},
    {"name": "Atkinson Hyperlegible", "license": "OFL-1.1", "use": "Tipografía de la interfaz"},
    {"name": "IBM Plex Mono", "license": "OFL-1.1", "use": "Tipografía de los datos"},
)


def version() -> str:
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return __version__


def license_candidates() -> list[Path]:
    """Where the LICENSE file may be, in order."""
    package = Path(__file__).resolve().parent
    places = [package / "LICENSE", package.parent / "LICENSE"]
    bundle = getattr(sys, "_MEIPASS", None)  # PyInstaller
    if bundle:
        places.insert(0, Path(bundle) / "LICENSE")
    return places


def license_text() -> str | None:
    """The full text of the project's license, or None when this installation does not have it."""
    for path in license_candidates():
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            log.debug("could not read %s", path, exc_info=True)
    try:
        dist = importlib.metadata.distribution(DISTRIBUTION)
        for name in ("licenses/LICENSE", "LICENSE"):
            text = dist.read_text(name)
            if text:
                return text
    except importlib.metadata.PackageNotFoundError:
        pass
    log.warning("LICENSE not found in this installation")
    return None


def info() -> dict:
    return {
        "name": APP_NAME,
        "version": version(),
        "copyright": COPYRIGHT,
        "license": LICENSE_NAME,
        "license_id": LICENSE_ID,
        "source_url": SOURCE_URL,
        "license_text": license_text(),
        "components": [dict(c) for c in COMPONENTS],
    }
