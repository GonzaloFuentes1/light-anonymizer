"""Línea base ``identidad``: copia cada archivo sin tocarlo.

Todo elemento y todo metadato sensible debe aparecer como fuga. Si el evaluador no marca alguno,
es ciego para ese caso. Nunca reporta errores, ni siquiera para los archivos que deberían rechazarse.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from banco_pruebas.esquema import Archivo, Manifiesto, ResultadoArchivo


def procesar(archivo: Archivo, manifiesto: Manifiesto, carpeta: Path, detalles: dict[str, Any]) -> ResultadoArchivo:
    entrada = Path(manifiesto.raiz) / archivo.ruta
    destino = carpeta / archivo.ruta
    if not entrada.exists():
        # No hay nada que copiar; igual no se reporta error (la identidad no rechaza nada).
        detalles.setdefault("sin_entrada", []).append(archivo.ruta)
        return ResultadoArchivo(entrada=archivo.ruta, salida=None, paginas_procesadas=0)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(entrada, destino)
    return ResultadoArchivo(entrada=archivo.ruta, salida=archivo.ruta, paginas_procesadas=len(archivo.paginas))
