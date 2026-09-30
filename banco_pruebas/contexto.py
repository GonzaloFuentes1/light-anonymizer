"""Contexto que comparten todos los generadores del conjunto de prueba."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from banco_pruebas.esquema import Archivo, Elemento, Metadato, Pagina
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.lienzo import Lienzo
from banco_pruebas.rostros import ProveedorRostros


@dataclass
class Contexto:
    raiz: Path
    semilla: int
    fict: Ficticios
    rostros: ProveedorRostros
    opciones: dict[str, Any] = field(default_factory=dict)

    def ficticios(self, modulo: str) -> Ficticios:
        """Fábrica de datos del módulo: determinista por nombre, sin repetir valores de otros módulos."""
        return self.fict.derivar(modulo)

    def rng(self, modulo: str) -> np.random.Generator:
        return np.random.default_rng([self.semilla, zlib.crc32(modulo.encode())])

    def ruta(self, relativa: str) -> Path:
        destino = self.raiz / relativa
        destino.parent.mkdir(parents=True, exist_ok=True)
        return destino


def archivo_imagen(
    *,
    id: str,
    ruta: str,
    formato: str,
    categoria: str,
    descripcion: str,
    lienzo: Lienzo | None = None,
    tam: tuple[int, int] | None = None,
    elementos: list[Elemento] | None = None,
    metadatos: list[Metadato] | None = None,
    etiquetas: dict[str, Any] | None = None,
) -> Archivo:
    """Arma el registro del manifiesto para una imagen de una página.

    El tamaño y los elementos deben estar en la geometría *mostrada* (tras aplicar EXIF).
    """
    if lienzo is not None:
        tam = tam or (lienzo.ancho, lienzo.alto)
        elementos = elementos if elementos is not None else lienzo.elementos
    assert tam is not None and elementos is not None
    for e in elementos:
        e.pagina = 0
    return Archivo(
        id=id,
        ruta=ruta,
        formato=formato,
        categoria=categoria,
        descripcion=descripcion,
        paginas=[Pagina(indice=0, ancho=tam[0], alto=tam[1], unidad="px")],
        elementos=list(elementos),
        metadatos_sensibles=list(metadatos or []),
        etiquetas=dict(etiquetas or {}),
    )


def guardar_imagen(img: Image.Image, destino: Path, formato: str, **opciones: Any) -> None:
    """Guarda sin arrastrar metadatos por accidente (los metadatos se agregan explícitamente)."""
    limpia = Image.frombytes(img.mode, img.size, img.tobytes())
    if img.mode == "P":
        limpia.putpalette(img.getpalette())
    formato_pil = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "tiff": "TIFF"}[formato]
    if formato_pil == "JPEG":
        opciones.setdefault("quality", 90)
        limpia = limpia.convert("RGB")
    limpia.save(destino, formato_pil, **opciones)
