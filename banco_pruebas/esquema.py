"""Esquema del manifiesto (verdad de terreno) y del informe de censura que se evalúa.

Convenciones de coordenadas (las mismas para el manifiesto y para el informe del motor):

- Imágenes: píxeles, origen en la esquina superior izquierda, en la geometría que resulta
  de aplicar la orientación EXIF (``PIL.ImageOps.exif_transpose``). Es la geometría en que
  el archivo de salida debe quedar, porque la salida no lleva EXIF.
- PDF: puntos (1/72 pulgada), origen arriba a la izquierda, en el espacio de página
  *sin rotar* de PyMuPDF (el mismo de ``page.search_for`` y ``page.add_redact_annot``).
  Si la página tiene ``/Rotate``, el evaluador transforma con ``page.rotation_matrix``
  solo para comparar píxeles.
- Un polígono es una lista de puntos ``[x, y]`` en orden (horario o antihorario). Los
  cuadriláteros rotados se guardan con sus 4 esquinas reales, no con su caja envolvente.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

VERSION_ESQUEMA = 1

Punto = list[float]
Poligono = list[Punto]

# Tipos de dato personal que el manifiesto registra.
TIPOS = (
    "rut",
    "correo",
    "telefono",
    "url",
    "nombre",
    "direccion",
    "rostro",
    "firma",
    "qr",
    "texto",  # texto sin dato personal: se usa para el modo "censurar todo el texto" y para medir sobrecensura
)

# Tipos cuyo criterio de aceptación es cero fugas en el nivel "base".
TIPOS_CERO_FUGAS = ("rut", "correo", "telefono")

# Niveles de dificultad.
#   base:             casos realistas y legibles. Criterio de aceptación de la fase 1.
#   estres:           casos difíciles (texto diminuto, espejado, mucho ruido). Se reporta, no bloquea.
#   fuera_de_alcance: límites documentados (nombres fuera de la lista, firmas, manuscrito).
#                     Se reporta para dejar el límite a la vista; no se espera detectarlos.
NIVELES = ("base", "estres", "fuera_de_alcance")

# Dónde vive el dato en el archivo.
#   texto:  capa de texto de un PDF (extraíble).
#   raster: píxeles (imagen suelta, página escaneada o imagen incrustada en un PDF).
#   vector: glifos convertidos a trazos en un PDF (visibles, no extraíbles).
#   oculto: presente en el archivo pero no visible (capa opcional apagada, fuera de la página,
#           blanco sobre blanco, bajo una imagen). Se evalúa por extracción de texto y bytes.
CAPAS = ("texto", "raster", "vector", "oculto")

Unidad = Literal["pt", "px"]


@dataclass
class Elemento:
    """Un dato que debe quedar censurado (o, si ``tipo == "texto"``, un texto neutro de referencia)."""

    tipo: str
    pagina: int
    poligono: Poligono | None
    valor: str | None = None
    nivel: str = "base"
    capa: str = "raster"
    etiquetas: dict[str, Any] = field(default_factory=dict)
    # Para rostros: región esencial (ojos, nariz, boca). Debe quedar cubierta por completo.
    nucleo: Poligono | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS:
            raise ValueError(f"tipo desconocido: {self.tipo}")
        if self.nivel not in NIVELES:
            raise ValueError(f"nivel desconocido: {self.nivel}")
        if self.capa not in CAPAS:
            raise ValueError(f"capa desconocida: {self.capa}")
        if self.poligono is not None and len(self.poligono) < 3:
            raise ValueError("un polígono necesita al menos 3 puntos")


@dataclass
class Metadato:
    """Un dato personal escondido en metadatos o estructuras no visibles del archivo.

    ``donde`` usa rutas como ``exif.gps``, ``exif.artist``, ``exif.miniatura``, ``xmp.dc:creator``,
    ``png.texto.Author``, ``tiff.artist``, ``pdf.info.author``, ``pdf.xmp``, ``pdf.anotacion``,
    ``pdf.adjunto``, ``pdf.ocg``, ``pdf.formulario``, ``pdf.marcador``, ``pdf.javascript``,
    ``pdf.revision_anterior``.
    ``valor`` es una cadena única que el evaluador busca en la salida (canario). Para datos
    no textuales (por ejemplo coordenadas GPS o una miniatura) ``valor`` es ``None`` y se evalúa
    la presencia de la estructura.
    """

    donde: str
    valor: str | None
    etiquetas: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class Pagina:
    indice: int
    ancho: float
    alto: float
    unidad: Unidad
    rotacion: int = 0  # /Rotate de la página PDF (0, 90, 180, 270); 0 para imágenes


@dataclass
class Archivo:
    id: str
    ruta: str  # relativa a la raíz del conjunto, con "/"
    formato: str  # pdf, jpg, png, webp, tiff, heic
    categoria: str
    descripcion: str
    paginas: list[Pagina]
    elementos: list[Elemento] = field(default_factory=list)
    metadatos_sensibles: list[Metadato] = field(default_factory=list)
    # "procesar" o "error:<codigo>" (contrasena, corrupto, vacio, formato) para archivos que deben rechazarse
    esperado: str = "procesar"
    etiquetas: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""


@dataclass
class Manifiesto:
    raiz: str
    semilla: int
    archivos: list[Archivo] = field(default_factory=list)
    lista_nombres: list[str] = field(default_factory=list)  # la lista que se entrega al motor
    fuentes_rostros: list[dict[str, Any]] = field(default_factory=list)
    versiones: dict[str, str] = field(default_factory=dict)
    version: int = VERSION_ESQUEMA

    def agregar(self, archivo: Archivo) -> Archivo:
        if any(a.id == archivo.id for a in self.archivos):
            raise ValueError(f"id de archivo repetido: {archivo.id}")
        for i, e in enumerate(archivo.elementos):
            e.id = f"{archivo.id}#e{i:03d}"
        for i, m in enumerate(archivo.metadatos_sensibles):
            m.id = f"{archivo.id}#m{i:02d}"
        ruta = Path(self.raiz) / archivo.ruta
        if ruta.exists():
            archivo.sha256 = sha256_archivo(ruta)
        self.archivos.append(archivo)
        return archivo

    def validar(self) -> list[str]:
        """Revisa consistencia: archivos existentes, páginas válidas, canarios únicos."""
        problemas: list[str] = []
        vistos: dict[str, str] = {}
        for a in self.archivos:
            if not (Path(self.raiz) / a.ruta).exists():
                problemas.append(f"{a.id}: no existe {a.ruta}")
            n_paginas = len(a.paginas)
            for e in a.elementos:
                if not 0 <= e.pagina < n_paginas:
                    problemas.append(f"{e.id}: página {e.pagina} fuera de rango")
                if e.tipo in TIPOS_CERO_FUGAS and e.valor:
                    clave = normalizar(e.tipo, e.valor)
                    previo = vistos.get(clave)
                    if previo and previo.split("#")[0] != a.id:
                        problemas.append(f"{e.id}: valor repetido con {previo} ({e.valor})")
                    vistos.setdefault(clave, e.id)
        return problemas

    def guardar(self, ruta: Path | None = None) -> Path:
        ruta = ruta or Path(self.raiz) / "manifiesto.json"
        datos = asdict(self)
        datos["raiz"] = "."
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
        return ruta

    @classmethod
    def cargar(cls, ruta: Path) -> Manifiesto:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        archivos = []
        for a in datos.pop("archivos"):
            a["paginas"] = [Pagina(**p) for p in a["paginas"]]
            a["elementos"] = [Elemento(**e) for e in a["elementos"]]
            a["metadatos_sensibles"] = [Metadato(**m) for m in a["metadatos_sensibles"]]
            archivos.append(Archivo(**a))
        datos["raiz"] = str(Path(ruta).parent)
        return cls(archivos=archivos, **datos)


# ---------------------------------------------------------------------------
# Contrato del informe de censura que el motor (o una línea base) entrega al evaluador.
# ---------------------------------------------------------------------------


@dataclass
class Censura:
    """Una zona que el motor censuró (o marcó para censurar) en el archivo de salida."""

    pagina: int
    poligono: Poligono
    tipo: str
    detector: str  # regex, lista_nombres, yunet, rapidocr, qr, manual, pagina_completa...
    texto: str | None = None
    score: float | None = None
    estado: str = "censurar"  # censurar | descartado (el revisor la quitó)


@dataclass
class ResultadoArchivo:
    entrada: str  # ruta relativa del archivo de entrada (la misma del manifiesto)
    salida: str | None  # ruta relativa del archivo de salida dentro de la carpeta de salida
    censuras: list[Censura] = field(default_factory=list)
    error: str | None = None  # código de error si el archivo se rechazó (contrasena, corrupto...)
    tiempo_s: float | None = None
    paginas_procesadas: int | None = None
    modo_todo_el_texto: bool = False


@dataclass
class InformeCensura:
    sistema: str  # nombre del sistema evaluado: motor, identidad, oraculo, cuaderno...
    resultados: list[ResultadoArchivo] = field(default_factory=list)
    detalles: dict[str, Any] = field(default_factory=dict)

    def guardar(self, ruta: Path) -> Path:
        ruta.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        return ruta

    @classmethod
    def cargar(cls, ruta: Path) -> InformeCensura:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        resultados = []
        for r in datos.pop("resultados"):
            r["censuras"] = [Censura(**c) for c in r["censuras"]]
            resultados.append(ResultadoArchivo(**r))
        return cls(resultados=resultados, **datos)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


_SEPARADORES = str.maketrans("", "", " \t\n\r.-‐‑–—_()+/")


def normalizar(tipo: str, valor: str) -> str:
    """Forma canónica para buscar un valor en texto extraído (misma función para buscar en la salida)."""
    import unicodedata

    if tipo in ("rut", "telefono"):
        limpio = valor.upper().translate(_SEPARADORES)
        for prefijo in ("RUT:", "RUN:", "RUT", "RUN", "R.U.T."):
            limpio = limpio.removeprefix(prefijo)
        return limpio
    if tipo in ("correo", "url"):
        return "".join(valor.split()).casefold()
    sin_tildes = unicodedata.normalize("NFKD", valor)
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return " ".join(sin_tildes.casefold().split())
