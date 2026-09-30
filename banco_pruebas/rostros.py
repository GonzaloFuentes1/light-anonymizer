"""Rostros para el conjunto de prueba, con su caja verdadera y su núcleo (ojos, nariz y boca).

Fuentes con licencia libre documentada (ver ``FUENTES`` y LICENCIAS.md). Las imágenes se
descargan una vez a ``datos_prueba/cache/rostros`` y se verifican por SHA-256. Si no hay
conexión ni caché, se usan rostros dibujados (``sintetico_dibujado``): sirven para probar el
flujo de generación, pero no para medir recall, y el manifiesto lo deja registrado.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from banco_pruebas.esquema import Elemento, Poligono
from banco_pruebas.lienzo import rect

POSES = ("frente", "tres_cuartos", "perfil")


@dataclass
class Rostro:
    id: str
    img: Image.Image  # RGB
    caja: Poligono  # cara completa: frente a mentón, oreja a oreja (coordenadas de ``img``)
    nucleo: Poligono  # ojos, nariz y boca: debe quedar cubierto siempre
    pose: str
    fuente: str  # clave en FUENTES
    etiquetas: dict[str, Any] = field(default_factory=dict)


@dataclass
class Escena:
    """Foto real con varias personas y sus cajas de rostro anotadas."""

    id: str
    img: Image.Image
    rostros: list[tuple[Poligono, Poligono | None]]  # (caja, núcleo o None)
    fuente: str
    etiquetas: dict[str, Any] = field(default_factory=dict)


# Registro de fuentes: id -> datos de licencia y atribución. Se completa con la investigación
# de fuentes (Face Research Lab London Set, Open Images, etc.).
FUENTES: dict[str, dict[str, Any]] = {
    "sintetico_dibujado": {
        "descripcion": "Rostros esquemáticos dibujados por el propio generador (sin personas reales).",
        "licencia": "Generado por el proyecto (mismo licenciamiento que el repositorio).",
        "atribucion": "",
        "valido_para_recall": False,
    },
}


class ProveedorRostros:
    def __init__(self, dir_cache: Path, semilla: int, permitir_descarga: bool = True) -> None:
        self.dir_cache = dir_cache
        self.rng = np.random.default_rng([semilla, 7])
        self.permitir_descarga = permitir_descarga
        self._rostros: dict[str, list[Rostro]] = {p: [] for p in POSES}
        self._escenas: list[Escena] = []
        self._indices: dict[str, int] = {p: 0 for p in POSES}
        self._usadas: set[str] = set()
        self.solo_sinteticos = True
        self._cargar()

    # -- carga -----------------------------------------------------------------

    def _cargar(self) -> None:
        reales = self._cargar_reales()
        if reales:
            self.solo_sinteticos = False
            return
        for i in range(12):
            for pose in POSES:
                self._rostros[pose].append(_rostro_dibujado(f"dib{i:02d}_{pose}", pose, self.rng))

    def _cargar_reales(self) -> bool:
        """Descarga/verifica y carga las fuentes reales. Devuelve False si no hay ninguna disponible."""
        return False

    # -- uso -------------------------------------------------------------------

    def tomar(self, pose: str = "frente", n: int = 1) -> list[Rostro]:
        """Entrega ``n`` rostros de la pose pedida, rotando de forma determinista por el catálogo."""
        lista = self._rostros[pose] or self._rostros["frente"]
        salida = []
        for _ in range(n):
            r = lista[self._indices[pose] % len(lista)]
            self._indices[pose] += 1
            self._usadas.add(r.fuente)
            salida.append(copy.copy(r))
        return salida

    def escenas(self) -> list[Escena]:
        for e in self._escenas:
            self._usadas.add(e.fuente)
        return list(self._escenas)

    @staticmethod
    def elementos(r: Rostro, nivel: str = "base", etiquetas: dict[str, Any] | None = None) -> list[Elemento]:
        """Elemento de tipo rostro en coordenadas de ``r.img`` (listo para ``Lienzo.pegar``)."""
        return [
            Elemento(
                tipo="rostro",
                pagina=0,
                poligono=copy.deepcopy(r.caja),
                nucleo=copy.deepcopy(r.nucleo),
                valor=None,
                nivel=nivel,
                capa="raster",
                etiquetas={"pose": r.pose, "fuente_rostro": r.fuente, "rostro_id": r.id, **(etiquetas or {})},
            )
        ]

    def fuentes_usadas(self) -> list[dict[str, Any]]:
        return [{"id": f, **FUENTES[f]} for f in sorted(self._usadas)]


def _rostro_dibujado(id: str, pose: str, rng: np.random.Generator) -> Rostro:
    """Cara esquemática (óvalo, ojos, nariz, boca, pelo). Solo para probar el flujo."""
    w, h = 300, 380
    fondo = tuple(int(c) for c in rng.integers(150, 230, 3))
    img = Image.new("RGB", (w, h), fondo)
    d = ImageDraw.Draw(img)
    piel = tuple(int(c) for c in rng.choice([[241, 194, 167], [198, 134, 99], [141, 85, 54], [224, 172, 105]]))
    pelo = tuple(int(c) for c in rng.choice([[30, 20, 15], [90, 60, 30], [160, 120, 60], [120, 120, 120]]))
    desplaz = {"frente": 0, "tres_cuartos": 28, "perfil": 55}[pose]
    x0, y0, x1, y1 = 60, 60, 240, 320
    d.ellipse((x0 - 10, y0 - 25, x1 + 10, y0 + 120), fill=pelo)
    d.ellipse((x0, y0, x1, y1), fill=piel)
    cx = (x0 + x1) / 2 + desplaz
    for dx in (-40, 40):
        if pose == "perfil" and dx < 0:
            continue
        d.ellipse((cx + dx - 14, 160, cx + dx + 14, 176), fill=(255, 255, 255))
        d.ellipse((cx + dx - 6, 162, cx + dx + 6, 174), fill=(40, 30, 20))
    d.polygon([(cx, 180), (cx - 12, 230), (cx + 10, 232)], fill=tuple(max(0, c - 30) for c in piel))
    d.arc((cx - 35, 240, cx + 35, 280), 10, 170, fill=(150, 50, 50), width=5)
    return Rostro(
        id=id,
        img=img,
        caja=rect(x0, y0 - 20, x1, y1),
        nucleo=rect(cx - 60, 150, cx + 60, 285),
        pose=pose,
        fuente="sintetico_dibujado",
    )
