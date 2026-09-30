"""Rostros para el conjunto de prueba, con su caja verdadera y su núcleo (ojos, nariz y boca).

Fuentes con licencia libre documentada (ver ``FUENTES`` y LICENCIAS.md). Las imágenes se
descargan una vez a ``datos_prueba/cache/rostros`` y se verifican por SHA-256. Si no hay
conexión ni caché, se usan rostros dibujados (``sintetico_dibujado``): sirven para probar el
flujo de generación, pero no para medir recall, y el manifiesto lo deja registrado.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
        """Descarga/verifica y carga las fuentes reales. Devuelve False si no hay ninguna disponible.

        Los rostros sueltos (``tomar``) vienen solo del Face Research Lab London Set, cuyas
        personas firmaron consentimiento. Las fotos de personas reales identificables (Open
        Images, retratos oficiales de dominio público) solo se usan como escenas tal cual,
        nunca pegadas en documentos ficticios.
        """
        if not _CATALOGO.exists():
            return False
        for e in json.loads(_CATALOGO.read_text(encoding="utf-8")):
            if e.get("opcional"):
                continue
            ruta = self._asegurar(e)
            if ruta is None:
                continue
            FUENTES[e["id"]] = {
                "descripcion": e.get("note") or e.get("subject") or e["kind"],
                "licencia": e["license"],
                "atribucion": e["attribution"],
                "url": _url(e),
                "sha256": e["sha256"],
                "valido_para_recall": True,
            }
            img = Image.open(ruta).convert("RGB")
            tipo = e["kind"]
            if tipo.startswith("frl_"):
                if tipo == "frl_front":
                    caja, nucleo = _gt_desde_tem(ruta.with_suffix(".tem"))
                else:
                    caja, nucleo = rect(*e["faces"][0]), rect(*e["core"][0])
                pose = "frente" if tipo == "frl_front" else ("perfil" if "profile" in tipo else "tres_cuartos")
                r = Rostro(e["id"], img, caja, nucleo, pose, e["id"], {"sujeto": e.get("subject"), "gt": e.get("gt")})
                self._rostros[pose].append(_recortar_retrato(r))
            else:
                self._escenas.append(
                    Escena(
                        id=e["id"],
                        img=img,
                        rostros=[(rect(*b), None) for b in e["faces"]],
                        fuente=e["id"],
                        etiquetas={"tipo_fuente": tipo, "marcas": e.get("flags"), "gt": e.get("gt")},
                    )
                )
        return bool(self._rostros["frente"])

    def _asegurar(self, e: dict[str, Any]) -> Path | None:
        """Ruta local verificada de la imagen (y su .tem); la descarga si falta y está permitido."""
        ext = Path(e.get("zip_entry") or urlparse(_url(e)).path).suffix or ".jpg"
        destino = self.dir_cache / f"{e['id']}{ext}"
        tem = destino.with_suffix(".tem") if e.get("tem_entry") else None
        if _verificado(destino, e["sha256"]) and (tem is None or _verificado(tem, e["tem_sha256"])):
            return destino
        if not self.permitir_descarga:
            return None
        self.dir_cache.mkdir(parents=True, exist_ok=True)
        try:
            if e.get("zip_entry"):
                zip_local = self.dir_cache / "zips" / f"{Path(urlparse(e['url']).path).name}.zip"
                if not zip_local.exists() or hashlib.md5(zip_local.read_bytes()).hexdigest() != e.get("zip_md5"):
                    _descargar(e["url"], zip_local)
                with zipfile.ZipFile(zip_local) as z:
                    destino.write_bytes(z.read(e["zip_entry"]))
                    if tem is not None:
                        tem.write_bytes(z.read(e["tem_entry"]))
            else:
                _descargar(_url(e), destino)
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            print(f"  aviso: no se pudo obtener {e['id']}: {error}")
            return None
        if not _verificado(destino, e["sha256"]):
            print(f"  aviso: {e['id']} no coincide con su SHA-256; se descarta")
            destino.unlink(missing_ok=True)
            return None
        return destino

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


_CATALOGO = Path(__file__).with_name("fuentes_rostros.json")
_AGENTE = "anonimizador-banco-pruebas/0.1 (CENIA; descarga de datos de prueba)"

# Grupos de puntos de las plantillas .tem de 189 puntos (Psychomorph/WebMorph).
_CONTORNO = [*range(109, 115), *range(125, 134)]
_CEJAS = range(71, 87)
_MENTON = range(125, 134)
_NUCLEO = [*range(18, 50), *range(50, 71), *range(87, 109)]  # ojos y párpados, nariz, boca


def _url(e: dict[str, Any]) -> str:
    return str(e["url"]).split()[0]


def _verificado(ruta: Path, sha256: str) -> bool:
    return ruta.exists() and hashlib.sha256(ruta.read_bytes()).hexdigest() == sha256


def _descargar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".parcial")
    solicitud = urllib.request.Request(url, headers={"User-Agent": _AGENTE})
    with urllib.request.urlopen(solicitud, timeout=120) as respuesta, open(temporal, "wb") as f:
        shutil.copyfileobj(respuesta, f)
    temporal.replace(destino)


def _gt_desde_tem(ruta: Path) -> tuple[Poligono, Poligono]:
    """Caja de la cara (cejas a mentón, contorno a contorno) y núcleo (ojos, nariz, boca)."""
    lineas = ruta.read_text(encoding="utf-8").split("\n")
    n = int(lineas[0])
    p = np.array([[float(v) for v in linea.split()[:2]] for linea in lineas[1 : n + 1]])
    x0, x1 = p[_CONTORNO, 0].min(), p[_CONTORNO, 0].max()
    y0, y1 = p[list(_CEJAS), 1].min(), p[list(_MENTON), 1].max()
    nx0, ny0 = p[_NUCLEO].min(axis=0)
    nx1, ny1 = p[_NUCLEO].max(axis=0)
    margen = 0.04 * (x1 - x0)
    return rect(x0, y0, x1, y1), rect(nx0 - margen, ny0 - margen, nx1 + margen, ny1 + margen)


def _recortar_retrato(r: Rostro, ancho_final: int = 480) -> Rostro:
    """Recorte de cabeza y hombros (tipo foto carné) con la verdad de terreno trasladada."""
    xs, ys = [q[0] for q in r.caja], [q[1] for q in r.caja]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    w, h = bx1 - bx0, by1 - by0
    cx0 = int(max(0, bx0 - 0.45 * w))
    cy0 = int(max(0, by0 - 0.7 * h))
    cx1 = int(min(r.img.width, bx1 + 0.45 * w))
    cy1 = int(min(r.img.height, by1 + 0.45 * h))
    img = r.img.crop((cx0, cy0, cx1, cy1))
    escala = min(1.0, ancho_final / img.width)
    if escala < 1.0:
        img = img.resize((ancho_final, max(1, round(img.height * escala))), Image.Resampling.LANCZOS)
    sx, sy = img.width / (cx1 - cx0), img.height / (cy1 - cy0)

    def mover(pol: Poligono) -> Poligono:
        return [[round((x - cx0) * sx, 3), round((y - cy0) * sy, 3)] for x, y in pol]

    return Rostro(r.id, img, mover(r.caja), mover(r.nucleo), r.pose, r.fuente, dict(r.etiquetas))


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
