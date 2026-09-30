"""Genera el conjunto de prueba ficticio en datos_prueba/generado/.

Uso (desde la raíz del repositorio):
    uv run python scripts/generar_datos_prueba.py [--semilla 33] [--solo pdf_texto imagenes] [--sin-descargas]

Es un atajo a ``python -m banco_pruebas.generar``; ver ese módulo para las opciones.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banco_pruebas.generar import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
