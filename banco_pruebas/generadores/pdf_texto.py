"""PDF con capa de texto real, construidos con PyMuPDF.

Cada dato se escribe con ``page.insert_text`` y su polígono se calcula de dos formas: con la
geometría de la inserción (avance de los glifos según la fuente, ascendente y descendente) y
con ``page.search_for(..., quads=True)`` sobre la página terminada. Si ambas coinciden (±2 pt
en cada esquina) se guarda el cuadrilátero de ``search_for``, que es la convención del
manifiesto; si no, se guarda el calculado y queda anotado en ``etiquetas["gt"]``.

Coordenadas: espacio de página *sin rotar* de PyMuPDF (origen en la esquina superior
izquierda del CropBox). En las páginas con ``/Rotate`` se escribe en coordenadas visibles y se
convierte con ``page.derotation_matrix``.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Archivo, Elemento, Pagina, Poligono
from banco_pruebas.ficticios import (
    COMUNAS,
    FORMATOS_CORREO,
    FORMATOS_POR_CLASE,
    FORMATOS_RUT,
    FORMATOS_TELEFONO,
    FRASES_ADMINISTRATIVAS,
    Ficticios,
    Persona,
    Telefono,
    sin_tildes,
)
from banco_pruebas.lienzo import Lienzo, mapear_a_rect, rect, ruta_fuente
from banco_pruebas.rostros import ProveedorRostros

MODULO = "pdf_texto"
CATEGORIA = "pdf_texto"
A4 = (595.0, 842.0)

# Fuentes TrueType incrustadas (DejaVu: tildes, ñ y guion largo).
_TTF = {
    "dejavu": "sans",
    "dejavu_n": "sans_negrita",
    "dejavu_serif": "serif",
    "dejavu_serif_n": "serif_negrita",
    "dejavu_mono": "mono",
    "dejavu_mono_n": "mono_negrita",
}
# Las Base 14 de PyMuPDF solo codifican Latin-1: si el texto trae otro carácter (p. ej. "–"),
# la línea se escribe con la TrueType equivalente.
_RESPALDO_TTF = {
    "helv": "dejavu",
    "hebo": "dejavu_n",
    "tiro": "dejavu_serif",
    "tibo": "dejavu_serif_n",
    "cour": "dejavu_mono",
    "cobo": "dejavu_mono_n",
}
_NEGRITA = {
    "helv": "hebo",
    "tiro": "tibo",
    "cour": "cobo",
    "dejavu": "dejavu_n",
    "dejavu_serif": "dejavu_serif_n",
    "dejavu_mono": "dejavu_mono_n",
}
# Banderas de extracción para calcular la verdad de terreno: sin recortar a la página.
_BANDERAS = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES
_TOLERANCIA_GT = 2.0  # pt: diferencia máxima por esquina entre geometría y search_for

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]  # fmt: skip


@lru_cache(maxsize=32)
def _fuente(nombre: str) -> pymupdf.Font:
    if nombre in _TTF:
        return pymupdf.Font(fontfile=str(ruta_fuente(_TTF[nombre])))
    return pymupdf.Font(nombre)


@lru_cache(maxsize=8192)
def _avance(nombre: str, ch: str) -> float:
    return _fuente(nombre).text_length(ch, 1.0)


def largo_texto(nombre: str, texto: str, tam: float) -> float:
    """Ancho del texto en pt (PyMuPDF no aplica interletraje: es la suma de los avances)."""
    return tam * sum(_avance(nombre, ch) for ch in texto)


def _es_latin1(texto: str) -> bool:
    try:
        texto.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def _fuente_para(nombre: str, texto: str) -> str:
    if nombre not in _TTF and not _es_latin1(texto):
        return _RESPALDO_TTF[nombre]
    return nombre


def textpage_completa(page: pymupdf.Page) -> pymupdf.TextPage:
    """TextPage sin recorte: incluye el texto fuera del CropBox (coordenadas relativas al CropBox)."""
    return page.get_textpage(clip=pymupdf.INFINITE_RECT(), flags=_BANDERAS)


def _quad_a_poligono(q: pymupdf.Quad) -> Poligono:
    return [[round(p.x, 3), round(p.y, 3)] for p in (q.ul, q.ur, q.lr, q.ll)]


def _distancia(a: Poligono, b: Poligono) -> float:
    return max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Partes de una línea
# ---------------------------------------------------------------------------


@dataclass
class Parte:
    """Segmento de una línea: texto neutro (``tipo="texto"``) o un dato a registrar."""

    texto: str
    tipo: str = "texto"
    nivel: str = "base"
    etiquetas: dict[str, Any] = field(default_factory=dict)
    valor: str | None = None
    unida: bool = False  # texto neutro que no admite salto de línea (p. ej. el " <" de "Nombre <correo>")

    @property
    def atomica(self) -> bool:
        """Los datos, los señuelos y las partes unidas no se cortan al envolver un párrafo."""
        return self.tipo != "texto" or bool(self.etiquetas) or self.unida


def T(texto: str) -> Parte:  # noqa: N802 - atajo de uso frecuente
    return Parte(texto)


def _partes(partes: str | Parte | Sequence[Parte]) -> list[Parte]:
    if isinstance(partes, str):
        return [Parte(partes)]
    if isinstance(partes, Parte):
        return [partes]
    return list(partes)


def rut_p(p: Persona, formato: str, nivel: str | None = None, **etiquetas: Any) -> Parte:
    if formato == "k_minuscula":
        texto, nivel_f = p.rut("puntos").replace("K", "k"), "base"
    else:
        texto, nivel_f = p.rut(formato), FORMATOS_RUT[formato]
    et = {"formato": formato, "dv_valido": p.rut_dv_valido, **etiquetas}
    if p.rut_cuerpo < 10_000_000:
        et["siete_digitos"] = True
    return Parte(texto, "rut", nivel or nivel_f, et)


def correo_p(p: Persona, formato: str, nivel: str | None = None) -> Parte:
    return Parte(p.correo(formato), "correo", nivel or FORMATOS_CORREO[formato], {"formato": formato})


def tel_p(t: Telefono, formato: str, nivel: str | None = None) -> Parte:
    return Parte(
        t.formatear(formato), "telefono", nivel or FORMATOS_TELEFONO[formato], {"formato": formato, "clase": t.clase}
    )


def nombre_texto(p: Persona, variante: str) -> str:
    return {
        "exacta": p.nombre_completo,
        "sin_tildes": sin_tildes(p.nombre_completo),
        "mayusculas": p.nombre_completo.upper(),
        "apellidos_nombres": f"{p.apellido_p} {p.apellido_m}, {p.nombres}",
        "parcial": f"{p.nombres.split()[0]} {p.apellido_p}",
        "solo_nombre": p.nombres.split()[0],  # saludo de un correo: "Estimada Carolina:"
    }[variante]


def nombre_p(p: Persona, variante: str = "exacta") -> Parte:
    texto = nombre_texto(p, variante)
    if variante == "sin_tildes" and texto == p.nombre_completo:
        variante = "exacta"
    if not p.en_lista:
        nivel = "fuera_de_alcance"
    else:
        nivel = "estres" if variante in ("parcial", "solo_nombre") else "base"
    return Parte(texto, "nombre", nivel, {"en_lista": p.en_lista, "variante": variante})


def direccion_p(p: Persona) -> Parte:
    return Parte(p.direccion, "direccion", "base" if p.en_lista else "fuera_de_alcance", {"en_lista": p.en_lista})


def url_p(texto: str, nivel: str = "base") -> Parte:
    return Parte(texto, "url", nivel)


def monto_p(texto: str) -> Parte:
    return Parte(texto, "texto", etiquetas={"senuelo": "monto"})


def fecha_p(texto: str) -> Parte:
    return Parte(texto, "texto", etiquetas={"senuelo": "fecha"})


def clase_de(formato: str) -> str:
    return next(c for c, fs in FORMATOS_POR_CLASE.items() if formato in fs)


def telefono_para(f: Ficticios, p: Persona, formato: str) -> Telefono:
    """El teléfono de la persona que corresponde a la clase del formato (móvil, Santiago o regional)."""
    clase = clase_de(formato)
    if clase == "movil":
        return p.telefono
    if p.telefono_fijo.clase != clase:
        p.telefono_fijo = f.telefono(clase)
    return p.telefono_fijo


# ---------------------------------------------------------------------------
# Hoja: una página con registro de elementos
# ---------------------------------------------------------------------------


class Hoja:
    """Página PDF que registra la geometría de todo lo que se escribe en ella.

    Con ``visible=True`` las coordenadas que recibe son las de la página tal como se ve
    (tras ``/Rotate``); se convierten al espacio sin rotar antes de escribir.
    """

    def __init__(self, page: pymupdf.Page, indice: int, visible: bool = False) -> None:
        self._doc = page.parent
        self._page = page
        self.indice = indice
        self.visible = visible
        self.elementos: list[Elemento] = []
        self._por_refinar: list[tuple[Elemento, str]] = []
        self._fuentes: set[str] = set()

    @property
    def page(self) -> pymupdf.Page:
        # PyMuPDF invalida los objetos Page al agregar páginas al documento: se recargan.
        if self._page.parent is None:
            self._page = self._doc[self.indice]
        return self._page

    # -- utilidades ----------------------------------------------------------

    def _preparar_fuente(self, nombre: str, texto: str) -> str:
        nombre = _fuente_para(nombre, texto)
        if nombre in _TTF and nombre not in self._fuentes:
            self.page.insert_font(fontname=nombre, fontfile=str(ruta_fuente(_TTF[nombre])))
            self._fuentes.add(nombre)
        return nombre

    def a_pagina(self, x: float, y: float) -> pymupdf.Point:
        p = pymupdf.Point(x, y)
        return p * self.page.derotation_matrix if self.visible else p

    @staticmethod
    def ancho(texto: str, fuente: str = "helv", tam: float = 10) -> float:
        return largo_texto(_fuente_para(fuente, texto), texto, tam)

    def linea_recta(self, x0: float, y0: float, x1: float, y1: float, grosor: float = 0.5, color=(0, 0, 0)) -> None:
        self.page.draw_line(self.a_pagina(x0, y0), self.a_pagina(x1, y1), color=color, width=grosor)

    def rectangulo(self, x0: float, y0: float, x1: float, y1: float, relleno=None, borde=None, grosor=0.5) -> None:
        a, b = self.a_pagina(x0, y0), self.a_pagina(x1, y1)
        self.page.draw_rect(pymupdf.Rect(a, b).normalize(), color=borde, fill=relleno, width=grosor)

    def agregar(self, elementos: Sequence[Elemento]) -> None:
        for e in elementos:
            e.pagina = self.indice
            self.elementos.append(e)

    # -- texto ---------------------------------------------------------------

    def escribir(
        self,
        x: float,
        y: float,
        partes: str | Parte | Sequence[Parte],
        *,
        fuente: str = "helv",
        tam: float = 10,
        color: tuple[float, float, float] = (0, 0, 0),
        angulo: float = 0,
        capa: str = "texto",
        render_mode: int = 0,
        registrar: bool = True,
        dibujar: bool = True,
        etiquetas: dict[str, Any] | None = None,
    ) -> float:
        """Escribe una línea con su línea base en (x, y) y registra cada parte. Devuelve la x final.

        ``angulo`` es el giro visible (antihorario, en grados). Los múltiplos de 90 usan
        ``rotate``; el resto, ``morph`` con pivote en el punto de inserción. Con ``dibujar=False``
        solo registra (para un dato que ya está escrito dentro de otro, como un RUT en una URL).
        """
        lista = _partes(partes)
        texto = "".join(p.texto for p in lista)
        nombre = self._preparar_fuente(fuente, texto)
        f = _fuente(nombre)
        punto = self.a_pagina(x, y)
        giro = (angulo + (self.page.rotation if self.visible else 0)) % 360
        opciones: dict[str, Any] = {"fontsize": tam, "fontname": nombre, "color": color, "render_mode": render_mode}
        if not dibujar:
            pass
        elif giro % 90 == 0:
            self.page.insert_text(punto, texto, rotate=int(giro), **opciones)
        else:
            self.page.insert_text(punto, texto, morph=(punto, pymupdf.Matrix(giro)), **opciones)
        largo = largo_texto(nombre, texto, tam)
        if registrar:
            v0, v1 = -f.ascender * tam, -f.descender * tam
            inicio = 0
            for p in lista:
                limpio = p.texto.strip()
                previo = texto[:inicio] + p.texto[: len(p.texto) - len(p.texto.lstrip())]
                inicio += len(p.texto)
                if not limpio or (p.tipo == "texto" and not any(c.isalnum() for c in limpio)):
                    continue
                u0 = largo_texto(nombre, previo, tam)
                u1 = u0 + largo_texto(nombre, limpio, tam)
                e = Elemento(
                    tipo=p.tipo,
                    pagina=self.indice,
                    poligono=_quad_geometrico(punto, giro, u0, u1, v0, v1),
                    valor=p.valor if p.valor is not None else limpio,
                    nivel=p.nivel,
                    capa=capa,
                    etiquetas={"fuente": nombre, "tam_pt": tam, "angulo": angulo, **(etiquetas or {}), **p.etiquetas},
                )
                self.elementos.append(e)
                if capa in ("texto", "oculto", "vector"):
                    self._por_refinar.append((e, limpio))
        return x + largo

    def escribir_espaciado(
        self,
        x: float,
        y: float,
        parte: Parte,
        espaciado: float,
        *,
        fuente: str = "helv",
        tam: float = 10,
    ) -> float:
        """Escribe cada carácter por separado, con ``espaciado`` pt extra entre ellos (solo sin giro)."""
        nombre = self._preparar_fuente(fuente, parte.texto)
        f = _fuente(nombre)
        cx = x
        for ch in parte.texto:
            self.page.insert_text(self.a_pagina(cx, y), ch, fontsize=tam, fontname=nombre)
            cx += largo_texto(nombre, ch, tam) + espaciado
        fin = cx - espaciado
        e = Elemento(
            tipo=parte.tipo,
            pagina=self.indice,
            poligono=rect(round(x, 3), round(y - f.ascender * tam, 3), round(fin, 3), round(y - f.descender * tam, 3)),
            valor=parte.valor or parte.texto,
            nivel=parte.nivel,
            capa="texto",
            etiquetas={
                "fuente": nombre,
                "tam_pt": tam,
                "angulo": 0,
                "espaciado_pt": espaciado,
                "gt": "geometria",
                **parte.etiquetas,
            },  # fmt: skip
        )
        self.elementos.append(e)
        return fin

    def refinar(self) -> None:
        """Reemplaza el cuadrilátero calculado por el de ``search_for`` cuando coinciden."""
        if not self._por_refinar:
            return
        tp = textpage_completa(self.page)
        cache: dict[str, list[Poligono]] = {}
        for e, buscado in self._por_refinar:
            if buscado not in cache:
                cache[buscado] = [_quad_a_poligono(q) for q in self.page.search_for(buscado, quads=True, textpage=tp)]
            candidatos = cache[buscado]
            mejor = min(candidatos, key=lambda q: _distancia(q, e.poligono), default=None)
            if mejor is not None and _distancia(mejor, e.poligono) <= _TOLERANCIA_GT:
                e.poligono = mejor
                e.etiquetas["gt"] = "search_for"
            else:
                e.poligono = [[round(px, 3), round(py, 3)] for px, py in e.poligono]
                e.etiquetas["gt"] = "geometria"
        self._por_refinar.clear()


def _quad_geometrico(p: pymupdf.Point, giro: float, u0: float, u1: float, v0: float, v1: float) -> Poligono:
    """Cuadrilátero de un tramo de texto: ``u`` a lo largo de la línea, ``v`` hacia abajo del glifo."""
    t = math.radians(giro)
    d = (math.cos(t), -math.sin(t))  # dirección de lectura (y hacia abajo)
    n = (math.sin(t), math.cos(t))  # "abajo" del glifo
    return [[p.x + u * d[0] + v * n[0], p.y + u * d[1] + v * n[1]] for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]


def envolver(partes: Sequence[Parte], ancho_max: float, fuente: str, tam: float) -> list[list[Parte]]:
    """Reparte las partes en líneas de ``ancho_max`` sin cortar nunca un dato ni un señuelo."""
    palabras: list[list[tuple[str, int]]] = [[]]
    for i, p in enumerate(partes):
        if p.atomica:
            palabras[-1].append((p.texto, i))
            continue
        for trozo in re.findall(r"\s+|\S+", p.texto):
            palabras[-1].append((trozo, i))
            if trozo.isspace():
                palabras.append([])
    lineas: list[list[tuple[str, int]]] = [[]]
    for palabra in palabras:
        if not palabra:
            continue
        candidata = lineas[-1] + palabra
        texto = "".join(t for t, _ in candidata).rstrip()
        if lineas[-1] and Hoja.ancho(texto, fuente, tam) > ancho_max:
            if palabra[0][0].isspace():
                palabra = palabra[1:]
            lineas.append(list(palabra))
        else:
            lineas[-1] = candidata
    salida = []
    for linea in lineas:
        while linea and linea[0][0].isspace():
            linea = linea[1:]
        grupos: list[Parte] = []
        ultimo = -1
        for trozo, i in linea:
            if i == ultimo and not partes[i].atomica:
                grupos[-1].texto += trozo
            elif partes[i].atomica:
                grupos.append(partes[i])
            else:
                grupos.append(Parte(trozo))
            ultimo = i
        if grupos:
            salida.append(grupos)
    return salida


# ---------------------------------------------------------------------------
# Documento y cursor de flujo
# ---------------------------------------------------------------------------


class Documento:
    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.hojas: list[Hoja] = []

    def nueva_hoja(self, ancho: float = A4[0], alto: float = A4[1], rotacion: int = 0, visible: bool = False) -> Hoja:
        page = self.doc.new_page(width=ancho, height=alto)
        if rotacion:
            page.set_rotation(rotacion)
        h = Hoja(page, len(self.hojas), visible)
        self.hojas.append(h)
        return h

    def elementos(self) -> list[Elemento]:
        return [e for h in self.hojas for e in h.elementos]

    def refinar(self) -> None:
        for h in self.hojas:
            h.refinar()

    def guardar(self, ctx: Contexto, ruta: str, subconjunto: bool = True) -> None:
        self.refinar()
        if subconjunto:
            self.doc.subset_fonts()
        self.doc.save(ctx.ruta(ruta), garbage=3, deflate=True, no_new_id=True)

    def archivo(self, id: str, ruta: str, descripcion: str, etiquetas: dict[str, Any] | None = None) -> Archivo:
        return Archivo(
            id=id,
            ruta=ruta,
            formato="pdf",
            categoria=CATEGORIA,
            descripcion=descripcion,
            paginas=paginas_de(self.doc),
            elementos=self.elementos(),
            etiquetas=dict(etiquetas or {}),
        )


def paginas_de(doc: pymupdf.Document) -> list[Pagina]:
    salida = []
    for i, page in enumerate(doc):
        caja = page.cropbox
        salida.append(Pagina(indice=i, ancho=round(caja.width, 3), alto=round(caja.height, 3), unidad="pt",
                             rotacion=page.rotation))  # fmt: skip
    return salida


class Cursor:
    """Escritura en flujo, de arriba hacia abajo, con salto de página automático."""

    def __init__(
        self,
        documento: Documento,
        *,
        fuente: str = "helv",
        tam: float = 10,
        x: float = 60,
        ancho: float = 475,
        arriba: float = 80,
        abajo: float = 780,
        al_crear: Callable[[Hoja], None] | None = None,
    ) -> None:
        self.documento = documento
        self.fuente, self.tam = fuente, tam
        self.x, self.ancho = x, ancho
        self.arriba, self.abajo = arriba, abajo
        self.al_crear = al_crear
        self.hoja = self._nueva()
        self.y = arriba

    def _nueva(self) -> Hoja:
        h = self.documento.nueva_hoja()
        if self.al_crear:
            self.al_crear(h)
        return h

    def salto(self) -> None:
        self.hoja = self._nueva()
        self.y = self.arriba

    def asegurar(self, alto: float) -> None:
        if self.y + alto > self.abajo:
            self.salto()

    def espacio(self, alto: float) -> None:
        self.y += alto

    def linea(
        self,
        partes: str | Parte | Sequence[Parte],
        *,
        fuente: str | None = None,
        tam: float | None = None,
        sangria: float = 0,
        interlineado: float = 1.45,
        **opciones: Any,
    ) -> None:
        tam = tam or self.tam
        self.asegurar(tam * interlineado)
        fuente = fuente or self.fuente
        texto = "".join(p.texto for p in _partes(partes)).rstrip()
        if sangria + Hoja.ancho(texto, fuente, tam) > self.ancho + 1:
            raise ValueError(f"la línea no cabe en el ancho del cursor: {texto!r}")
        self.hoja.escribir(self.x + sangria, self.y + tam, partes, fuente=fuente, tam=tam, **opciones)
        self.y += tam * interlineado

    def parrafo(
        self,
        partes: str | Parte | Sequence[Parte],
        *,
        fuente: str | None = None,
        tam: float | None = None,
        sangria: float = 0,
        interlineado: float = 1.45,
        despues: float = 4,
    ) -> None:
        fuente, tam = fuente or self.fuente, tam or self.tam
        for grupo in envolver(_partes(partes), self.ancho - sangria, fuente, tam):
            self.linea(grupo, fuente=fuente, tam=tam, sangria=sangria, interlineado=interlineado)
        self.y += despues

    def campo(self, etiqueta: str, partes: str | Parte | Sequence[Parte], tab: float = 112) -> None:
        tam = self.tam
        self.asegurar(tam * 1.5)
        base = self.y + tam
        self.hoja.escribir(self.x, base, etiqueta, fuente=_NEGRITA[self.fuente], tam=tam)
        self.hoja.escribir(self.x + tab, base, partes, fuente=self.fuente, tam=tam)
        self.y += tam * 1.5

    def seccion(self, titulo: str) -> None:
        self.asegurar(self.tam * 4)
        self.y += self.tam * 0.6
        self.linea(titulo, fuente=_NEGRITA[self.fuente], tam=self.tam + 1, interlineado=1.6)

    def tabla(self, columnas: list[str], filas: list[list[Parte]], *, fuente: str, tam: float = 8) -> None:
        alto_fila = tam * 2.0
        self.asegurar(alto_fila * (len(filas) + 1) + 4)
        self.y = dibujar_tabla(self.hoja, self.x, self.y, self.ancho, columnas, filas, fuente=fuente, tam=tam) + 6


def dibujar_tabla(
    hoja: Hoja,
    x: float,
    y: float,
    ancho: float,
    columnas: list[str],
    filas: list[list[Parte]],
    *,
    fuente: str,
    tam: float = 8,
    tam_min: float = 6.5,
) -> float:
    """Tabla con líneas dibujadas y celdas de texto. Reduce la letra si no cabe. Devuelve la y final."""
    negrita = _NEGRITA[fuente]
    relleno = 4.0
    while True:
        anchos = []
        for j, titulo in enumerate(columnas):
            m = Hoja.ancho(titulo, negrita, tam)
            for fila in filas:
                m = max(m, Hoja.ancho(fila[j].texto, fuente, tam))
            anchos.append(m + 2 * relleno)
        if sum(anchos) <= ancho or tam <= tam_min:
            break
        tam -= 0.25
    if sum(anchos) > ancho:
        raise ValueError("la tabla no cabe en el ancho disponible")
    extra = (ancho - sum(anchos)) / len(anchos)
    anchos = [a + extra for a in anchos]
    alto_fila = tam * 2.0
    xs = [x]
    for a in anchos:
        xs.append(xs[-1] + a)
    n = len(filas) + 1
    hoja.rectangulo(x, y, x + ancho, y + alto_fila, relleno=(0.88, 0.9, 0.93))
    for i in range(n + 1):
        hoja.linea_recta(x, y + i * alto_fila, x + ancho, y + i * alto_fila)
    for cx in xs:
        hoja.linea_recta(cx, y, cx, y + n * alto_fila)
    base = y + alto_fila / 2 + tam * 0.35
    for j, titulo in enumerate(columnas):
        hoja.escribir(xs[j] + relleno, base, titulo, fuente=negrita, tam=tam)
    for i, fila in enumerate(filas, start=1):
        for j, celda in enumerate(fila):
            hoja.escribir(xs[j] + relleno, base + i * alto_fila, celda, fuente=fuente, tam=tam)
    return y + n * alto_fila


def pie_de_pagina(documento: Documento, izquierda: str, fuente: str = "helv", y: float = 812) -> None:
    n = len(documento.hojas)
    for h in documento.hojas:
        h.linea_recta(60, y - 12, 535, y - 12, grosor=0.4, color=(0.5, 0.5, 0.5))
        h.escribir(60, y, izquierda, fuente=fuente, tam=7.5, color=(0.3, 0.3, 0.3))
        derecha = f"Página {h.indice + 1} de {n}"
        h.escribir(535 - Hoja.ancho(derecha, fuente, 7.5), y, derecha, fuente=fuente, tam=7.5, color=(0.3, 0.3, 0.3))


def encabezado_gore(fuente: str, unidad: str, folio: str) -> Callable[[Hoja], None]:
    def dibujar(h: Hoja) -> None:
        h.escribir(60, 44, "GOBIERNO REGIONAL FICTICIO", fuente=_NEGRITA[fuente], tam=11, color=(0.1, 0.2, 0.45))
        h.escribir(60, 56, unidad, fuente=fuente, tam=8, color=(0.3, 0.3, 0.3))
        derecha = f"Folio N° {folio}"
        h.escribir(535 - Hoja.ancho(derecha, fuente, 8), 44, derecha, fuente=fuente, tam=8)
        h.linea_recta(60, 63, 535, 63, grosor=0.8, color=(0.1, 0.2, 0.45))

    return dibujar


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _pesos(texto: str) -> int:
    return int(re.sub(r"\D", "", texto))


def _formato_monto(valor: int) -> str:
    return f"$ {valor:,}".replace(",", ".")


def _clave_fecha(fecha: str) -> tuple[int, int, int]:
    """Orden cronológico de una fecha "16 de octubre de 2026"."""
    dia, _, mes, _, anio = fecha.split()
    return int(anio), MESES.index(mes.lower()), int(dia)


# ---------------------------------------------------------------------------
# 1. Réplica del documento de prueba del cuaderno
# ---------------------------------------------------------------------------

TEXTO_CUADERNO = """INFORME DE HONORARIOS - AGOSTO 2026

Nombre: Ana Maria Rojas Pena
RUT: 15.782.334-9
Correo: ana.rojas@ejemplo.cl
Telefono: +56 9 8123 4567
Direccion: Pasaje Los Alerces 442, depto 31

Producto 1: Informe de avance del programa
Monto bruto: $ 1.450.000

Contraparte: Jefatura de la unidad
RUT contraparte: 9.876.543-3
Sitio: https://www.ejemplo.cl/rendiciones
"""


def _cuaderno(ctx: Contexto, f: Ficticios) -> Archivo:
    f.reservar(
        ruts=(15782334, 9876543),
        telefonos=("981234567",),
        correos=(("ana", "rojas"),),
        nombres=("Ana Maria Rojas Pena",),
    )
    ana = Persona(
        nombres="Ana Maria",
        apellido_p="Rojas",
        apellido_m="Pena",
        sexo="F",
        rut_cuerpo=15782334,
        rut_dv="9",
        rut_dv_valido=False,
        telefono=Telefono("movil", "981234567", "9"),
        telefono_fijo=f.telefono("regional"),
        dominio="ejemplo.cl",
        direccion="Pasaje Los Alerces 442, depto 31",
        comuna="Concepción",
        nacimiento="14-03-1987",
        en_lista=True,
        extra={"origen": "cuaderno CoP33, celda 6"},
    )
    f.personas.append(ana)
    contraparte_rut = Parte(
        "9.876.543-3", "rut", "base", {"formato": "puntos", "dv_valido": True, "siete_digitos": True}
    )
    lineas: list[list[Parte]] = [
        [T("INFORME DE HONORARIOS - AGOSTO 2026")],
        [T("Nombre:"), nombre_p(ana, "exacta")],
        [T("RUT:"), rut_p(ana, "puntos")],
        [T("Correo:"), correo_p(ana, "punto")],
        [T("Telefono:"), tel_p(ana.telefono, "movil_internacional")],
        [T("Direccion:"), direccion_p(ana)],
        [T("Producto 1: Informe de avance del programa")],
        [T("Monto bruto:"), monto_p("$ 1.450.000")],
        [T("Contraparte: Jefatura de la unidad")],
        [T("RUT contraparte:"), contraparte_rut],
        [T("Sitio:"), url_p("https://www.ejemplo.cl/rendiciones")],
    ]
    doc = pymupdf.open()
    pagina = doc.new_page()
    pagina.insert_text((60, 70), TEXTO_CUADERNO, fontsize=11, fontname="helv")
    tp = textpage_completa(pagina)
    elementos: list[Elemento] = []
    for partes in lineas:
        completo = " ".join(p.texto for p in partes)
        (q_linea,) = pagina.search_for(completo, quads=True, textpage=tp)
        caja = q_linea.rect + (-1, -1, 1, 1)
        for p in partes:
            hits = [q for q in pagina.search_for(p.texto, quads=True, textpage=tp) if caja.contains(q.rect)]
            assert len(hits) == 1, (p.texto, hits)
            elementos.append(
                Elemento(
                    tipo=p.tipo,
                    pagina=0,
                    poligono=_quad_a_poligono(hits[0]),
                    valor=p.texto,
                    nivel=p.nivel,
                    capa="texto",
                    etiquetas={"fuente": "helv", "tam_pt": 11, "angulo": 0, "gt": "search_for", **p.etiquetas},
                )
            )
    ruta = "pdf_texto/cuaderno_informe_de_prueba.pdf"
    doc.save(ctx.ruta(ruta), no_new_id=True)
    archivo = Archivo(
        id="pdft_cuaderno",
        ruta=ruta,
        formato="pdf",
        categoria=CATEGORIA,
        descripcion="Réplica exacta del documento de prueba del cuaderno CoP 33 (celda 6): insert_text helv 11 pt.",
        paginas=paginas_de(doc),
        elementos=elementos,
        etiquetas={"replica_cuaderno": True},
    )
    doc.close()
    return archivo


# ---------------------------------------------------------------------------
# 2. Informes de honorarios
# ---------------------------------------------------------------------------

# Casillas por informe: 0 = prestador, 1 = contraparte, 2..6 = filas de la tabla.
# (formato RUT, formato correo, formato teléfono, opciones)
#   opciones: dv_malo, k (DV K), siete (cuerpo de 7 dígitos), fuera (no está en la lista)
_CASILLAS: list[list[tuple[str, str, str, set[str]]]] = [
    [
        ("puntos", "punto", "movil_internacional", set()),
        ("sin_puntos", "inicial", "regional_parentesis", set()),
        ("espacios_guion", "con_anio", "movil_compacto", {"dv_malo"}),
        ("guion_largo", "guion_bajo", "santiago_nacional", set()),
        ("puntos", "mayusculas", "regional_compacto", set()),
        ("k_minuscula", "mas", "movil_nacional", {"k"}),
        ("sin_puntos", "subdominio", "santiago_internacional", set()),
    ],
    [
        ("espacios_guion", "punto", "movil_bloque", set()),
        ("guion_largo", "inicial", "santiago_parentesis", set()),
        ("puntos", "con_anio", "movil_parentesis", set()),
        ("sin_guion", "guion_bajo", "regional_internacional", {"fuera"}),
        ("sin_puntos", "mayusculas", "movil_guiones", {"siete"}),
        ("puntos", "mas", "regional_guion", {"dv_malo"}),
        ("guion_largo", "subdominio", "antiguo_movil_09", set()),
    ],
    [
        ("puntos", "subdominio", "movil_internacional", {"k"}),
        ("guion_largo", "punto", "regional_guion", set()),
        ("espacios_guion", "inicial", "santiago_nacional", set()),
        ("comas", "con_anio", "movil_compacto", {"dv_malo"}),
        ("sin_puntos", "guion_bajo", "regional_parentesis", set()),
        ("puntos", "mayusculas", "movil_nacional", set()),
        ("espacios_internos", "arroba_texto", "santiago_parentesis", set()),
    ],
    [
        ("sin_puntos", "mas", "movil_bloque", {"fuera"}),
        ("puntos", "punto", "santiago_internacional", {"siete"}),
        ("guion_largo", "inicial", "movil_parentesis", set()),
        ("espacios_guion", "con_anio", "regional_internacional", {"dv_malo"}),
        ("puntos", "guion_bajo", "movil_guiones", set()),
        ("sin_puntos", "subdominio", "regional_compacto", set()),
        ("puntos", "espacios", "antiguo_regional_0", set()),
    ],
]
# (fuente del cuerpo, fuente de la tabla, líneas de registro de actividades, mes)
_INFORMES = [
    ("helv", "helv", 0, 7),
    ("tiro", "cour", 22, 8),
    ("dejavu", "dejavu", 62, 9),
    ("helv", "tiro", 18, 10),
]
_VARIANTES_TABLA = ["exacta", "sin_tildes", "mayusculas", "apellidos_nombres", "parcial"]
_VARIANTES_CONTRAPARTE = ["exacta", "sin_tildes", "exacta", "sin_tildes"]
_ACTIVIDADES = [
    "Taller de formulación de proyectos con organizaciones comunitarias",
    "Reunión de coordinación con el equipo municipal de fomento productivo",
    "Visita a terreno para levantamiento de información de emprendimientos",
    "Mesa técnica con servicios públicos regionales",
    "Revisión de rendiciones de cuentas de proyectos adjudicados",
    "Capacitación en uso de la plataforma de postulación",
    "Entrevistas a beneficiarios del programa",
]


def _persona_casilla(f: Ficticios, opciones: set[str]) -> Persona:
    p = f.persona(
        dv_valido="dv_malo" not in opciones, en_lista="fuera" not in opciones, siete_digitos="siete" in opciones
    )
    if "k" in opciones:
        p.rut_cuerpo, p.rut_dv = f.rut_con_k()
        p.rut_dv_valido = True
    return p


def _honorarios(ctx: Contexto, f: Ficticios, rng: np.random.Generator, n: int) -> Archivo:
    fuente, fuente_tabla, n_actividades, mes_i = _INFORMES[n - 1]
    mes = MESES[mes_i - 1]
    casillas = _CASILLAS[n - 1]
    personas = [_persona_casilla(f, op) for *_, op in casillas]
    folio = f.folio()
    doc = Documento()
    cur = Cursor(doc, fuente=fuente, tam=10, al_crear=encabezado_gore(fuente, "División de Fomento e Industria", folio))
    negrita = _NEGRITA[fuente]

    def datos(i: int) -> tuple[Parte, Parte, Parte]:
        rut_f, correo_f, tel_f, _ = casillas[i]
        p = personas[i]
        return rut_p(p, rut_f), correo_p(p, correo_f), tel_p(telefono_para(f, p, tel_f), tel_f)

    cur.linea(f"INFORME DE HONORARIOS - {mes.upper()} 2026", fuente=negrita, tam=14, interlineado=1.8)
    cur.linea([T("Fecha de emisión: "), fecha_p(f.fecha())], tam=9)
    cur.linea([T("Período informado: "), fecha_p(f"1 al 30 de {mes} de 2026")], tam=9)

    p0 = personas[0]
    rut0, correo0, tel0 = datos(0)
    cur.seccion("1. Antecedentes del prestador de servicios")
    cur.campo("Nombre:", nombre_p(p0, "exacta"))
    cur.campo("RUT:", rut0)
    cur.campo("Correo:", correo0)
    cur.campo("Teléfono:", tel0)
    cur.campo("Dirección:", [direccion_p(p0), T(f", {p0.comuna}")])
    cur.campo("Calidad jurídica:", "Honorario a suma alzada")

    cur.seccion("2. Productos del período")
    k = 3 + n % 3
    for i in rng.choice(len(FRASES_ADMINISTRATIVAS), size=k, replace=False):
        cur.parrafo([T("- " + FRASES_ADMINISTRATIVAS[int(i)])], sangria=8, despues=1)
    url1 = f"https://transparencia.goreficticio.cl/honorarios/2026/{mes}/{folio.replace('/', '-')}"
    cur.parrafo([T("Los medios de verificación están disponibles en "), url_p(url1), T(".")], sangria=8)

    cur.seccion("3. Montos")
    bruto = _pesos(f.monto())
    retencion = round(bruto * 0.1525)
    cur.campo("Monto bruto:", monto_p(_formato_monto(bruto)))
    cur.campo("Retención 15,25 %:", monto_p(_formato_monto(retencion)))
    cur.campo("Monto líquido:", monto_p(_formato_monto(bruto - retencion)))
    cur.campo("Fecha de pago:", fecha_p(f.fecha()))

    p1 = personas[1]
    rut1, correo1, tel1 = datos(1)
    anexo = str(int(rng.integers(2000, 7999)))
    cur.seccion("4. Contraparte técnica")
    cur.campo("Nombre:", nombre_p(p1, _VARIANTES_CONTRAPARTE[n - 1]))
    cur.campo("Cargo:", "Profesional de la División de Fomento e Industria")
    cur.campo("RUT:", rut1)
    cur.campo("Correo:", correo1)
    cur.campo("Teléfono:", [tel1, T(f", anexo {anexo}")])

    seccion = 5
    if n_actividades:
        cur.seccion(f"{seccion}. Registro de actividades")
        seccion += 1
        for _ in range(n_actividades):
            dia = int(rng.integers(1, 29))
            actividad = _ACTIVIDADES[int(rng.integers(len(_ACTIVIDADES)))]
            comuna = COMUNAS[int(rng.integers(len(COMUNAS)))]
            horas = int(rng.integers(2, 9))
            cur.parrafo(
                [fecha_p(f"{dia:02d}-{mes_i:02d}-2026"), T(f": {actividad}, comuna de {comuna} ({horas} horas).")],
                tam=9,
                sangria=8,
                interlineado=1.4,
                despues=0,
            )

    cur.seccion(f"{seccion}. Equipo de apoyo y participantes")
    filas = []
    for j, i in enumerate(range(2, 7)):
        rut, correo, tel = datos(i)
        variante = _VARIANTES_TABLA[(j + n - 1) % len(_VARIANTES_TABLA)]
        filas.append([nombre_p(personas[i], variante), rut, correo, tel])
    cur.tabla(["Nombre", "RUT", "Correo", "Teléfono"], filas, fuente=fuente_tabla, tam=8)
    url2 = f"https://www.goreficticio.cl/fomento/equipos/{mes}-2026?folio={folio.split('/')[0]}"
    cur.parrafo([T("Nómina vigente publicada en "), url_p(url2), T(".")], tam=9)

    # Firmas
    cur.asegurar(90)
    y = cur.y + 45
    h = cur.hoja
    for x0, persona, variante, rol in (
        (70, p0, "mayusculas", "Prestador(a) de servicios"),
        (320, p1, "apellidos_nombres", "V°B° Contraparte técnica"),
    ):
        h.linea_recta(x0, y, x0 + 200, y, grosor=0.6)
        h.escribir(x0, y + 12, nombre_p(persona, variante), fuente=fuente, tam=9)
        h.escribir(x0, y + 24, rol, fuente=fuente, tam=8, color=(0.3, 0.3, 0.3))
    cur.y = y + 30

    pie_de_pagina(doc, f"Gobierno Regional Ficticio · Informe de honorarios {mes} 2026", fuente=fuente)
    ruta = f"pdf_texto/informe_honorarios_{n:02d}.pdf"
    doc.guardar(ctx, ruta)
    return doc.archivo(
        f"pdft_honorarios_{n:02d}",
        ruta,
        f"Informe de honorarios de {len(doc.hojas)} página(s) con bloque de datos, contraparte, montos y tabla de "
        f"5 personas (fuente {fuente}, tabla {fuente_tabla}); variantes de formato de RUT, teléfono, correo y nombre.",
        {"fuente": fuente, "fuente_tabla": fuente_tabla},
    )


# ---------------------------------------------------------------------------
# 3. Resolución exenta
# ---------------------------------------------------------------------------


def _resolucion(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    fuente = "tiro"
    folio = f.folio()
    doc = Documento()
    cur = Cursor(
        doc,
        fuente=fuente,
        tam=11,
        x=70,
        ancho=455,
        al_crear=encabezado_gore(fuente, "División de Administración y Finanzas", folio),
    )
    negrita = _NEGRITA[fuente]
    prestador = f.persona()
    contraparte = f.persona()
    error = f.persona(dv_valido=False)
    jefa = f.persona()
    distribucion = [f.persona() for _ in range(3)]
    bruto = _formato_monto(_pesos(f.monto()))
    trato = "doña" if prestador.sexo == "F" else "don"
    domiciliado = "domiciliada" if prestador.sexo == "F" else "domiciliado"

    cur.espacio(6)
    for titulo in (f"RESOLUCIÓN EXENTA N° {folio}", "APRUEBA CONTRATO DE PRESTACIÓN DE SERVICIOS A HONORARIOS"):
        cur.linea(titulo, fuente=negrita, tam=11, sangria=(cur.ancho - Hoja.ancho(titulo, negrita, 11)) / 2)
    cur.linea([T("Concepción, "), fecha_p(f.fecha())], sangria=300)
    cur.espacio(8)

    cur.linea("VISTOS:", fuente=negrita)
    cur.parrafo(
        [
            T(
                "Lo dispuesto en la Ley N° 19.175, Orgánica Constitucional sobre Gobierno y Administración Regional; "
                "en la Ley N° 18.575, Orgánica Constitucional de Bases Generales de la Administración del Estado; "
                "en la Resolución N° 7 de 2019 de la Contraloría General de la República; y el certificado de "
                "disponibilidad presupuestaria N° 245 de fecha "
            ),
            fecha_p(f.fecha()),
            T("."),
        ]
    )
    cur.linea("CONSIDERANDO:", fuente=negrita)
    rut_prest = rut_p(prestador, "puntos")
    cur.parrafo(
        [
            T(f"1. Que {trato} "),
            nombre_p(prestador, "exacta"),
            T(", cédula nacional de identidad N° "),
            rut_prest,
            T(f", {domiciliado} en "),
            direccion_p(prestador),
            T(
                f", comuna de {prestador.comuna}, presentó su oferta para prestar servicios de apoyo profesional "
                "a la División de Fomento e Industria."
            ),  # fmt: skip
        ]
    )
    tel_prest = telefono_para(f, prestador, "regional_internacional")
    cur.parrafo(
        [
            T(
                "2. Que la persona individualizada acredita el título profesional requerido e informó como medios "
                "de contacto el correo electrónico "
            ),  # fmt: skip
            correo_p(prestador, "punto"),
            T(", el teléfono móvil "),
            tel_p(prestador.telefono, "movil_internacional"),
            T(" y el teléfono fijo "),
            tel_p(tel_prest, "regional_internacional"),
            T("."),
        ]
    )
    cur.parrafo(
        [
            T("3. Que la contraparte técnica, "),
            nombre_p(contraparte, "exacta"),
            T(", RUT "),
            rut_p(contraparte, "sin_puntos"),
            T(", validó los antecedentes según consta en el memorándum N° 58 de "),
            fecha_p(f.fecha()),
            T(", enviado desde la casilla "),
            correo_p(contraparte, "inicial"),
            T("."),
        ]
    )
    cur.parrafo(
        [
            T("4. Que en el oficio anterior se consignó por error el RUT "),
            rut_p(error, "puntos"),
            T(" a nombre de "),
            nombre_p(error, "sin_tildes"),
            T(", dato que se corrige mediante el presente acto administrativo."),
        ]
    )
    cur.parrafo(
        [
            T("5. Que existe disponibilidad presupuestaria para financiar un monto bruto mensual de "),
            monto_p(bruto),
            T(", con cargo al subtítulo 21, ítem 03, del presupuesto vigente."),
        ]
    )
    cur.linea("RESUELVO:", fuente=negrita)
    desde, hasta = sorted((f.fecha(), f.fecha()), key=_clave_fecha)  # el contrato no termina antes de empezar
    cur.parrafo(
        [
            T("1° APRUÉBASE el contrato de prestación de servicios a honorarios suscrito con "),
            nombre_p(prestador, "mayusculas"),
            T(", RUT "),
            rut_p(prestador, "espacios_guion"),
            T(", por un monto bruto mensual de "),
            monto_p(bruto),
            T(", desde el "),
            fecha_p(desde),
            T(" hasta el "),
            fecha_p(hasta),
            T("."),
        ]
    )
    cur.parrafo("2° IMPÚTESE el gasto al subtítulo 21, ítem 03, asignación 001, del presupuesto del Gobierno Regional.")
    cur.parrafo(
        [
            T("3° NOTIFÍQUESE la presente resolución a la persona interesada al correo "),
            correo_p(prestador, "guion_bajo"),
            T(" o, en su defecto, por carta certificada al domicilio señalado en el considerando 1."),
        ]
    )
    url = f"https://transparencia.goreficticio.cl/resoluciones/2026/exenta-{folio.split('/')[0]}"
    cur.parrafo([T("4° PUBLÍQUESE en el sitio de transparencia activa "), url_p(url), T(".")])
    cur.espacio(6)
    cur.linea("ANÓTESE, COMUNÍQUESE Y ARCHÍVESE.", fuente=negrita, sangria=110)
    cur.espacio(34)
    cur.asegurar(40)
    firma = nombre_p(jefa, "mayusculas")
    cur.linea(firma, fuente=negrita, sangria=(cur.ancho - Hoja.ancho(firma.texto, negrita, 11)) / 2)
    cargo = (
        "Jefa División de Administración y Finanzas"
        if jefa.sexo == "F"
        else "Jefe División de Administración y Finanzas"
    )
    cur.linea(cargo, sangria=(cur.ancho - Hoja.ancho(cargo, fuente, 11)) / 2)
    cur.espacio(14)
    cur.asegurar(90)
    cur.linea("Distribución:", fuente=negrita, tam=9)
    formatos = ["punto", "con_anio", "subdominio"]
    for p, formato in zip(distribucion, formatos, strict=True):
        cur.linea([T("- "), nombre_p(p, "exacta"), T(", "), correo_p(p, formato)], tam=9, interlineado=1.35)
    cur.linea(
        [T("- "), nombre_p(prestador, "sin_tildes"), T(", "), correo_p(prestador, "mayusculas")],
        tam=9,
        interlineado=1.35,
    )
    for resto in ("- Oficina de Partes", "- Archivo"):
        cur.linea(resto, tam=9, interlineado=1.35)
    iniciales = "".join(w[0] for w in jefa.nombre_completo.split()[:3]).upper()
    cur.linea(f"{iniciales}/mfr", tam=7)
    if len(doc.hojas) < 2:  # el documento debe ocupar dos páginas
        cur.salto()
        cur.linea("Documento firmado electrónicamente conforme a la Ley N° 19.799.", tam=9)
    pie_de_pagina(doc, "Gobierno Regional Ficticio · Resolución exenta", fuente=fuente)
    ruta = "pdf_texto/resolucion_exenta.pdf"
    doc.guardar(ctx, ruta)
    return doc.archivo(
        "pdft_resolucion_exenta",
        ruta,
        "Resolución exenta de 2 páginas (VISTOS, CONSIDERANDO, RESUELVO) con datos personales dentro de las "
        "oraciones y lista de distribución con correos.",
        {"fuente": fuente},
    )


# ---------------------------------------------------------------------------
# 4. Correo impreso
# ---------------------------------------------------------------------------


def _correo_impreso(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    fuente = "dejavu"
    doc = Documento()
    cur = Cursor(doc, fuente=fuente, tam=9.5, x=50, ancho=495, arriba=40, abajo=790)
    negrita = _NEGRITA[fuente]
    a, b, c = f.persona(), f.persona(), f.persona()
    externo = f.persona(en_lista=False)
    a.telefono_fijo = f.telefono("regional")
    b.telefono_fijo = f.telefono("regional")
    asunto = "Informe de honorarios y rendición del mes"
    buzon = correo_p(b, "punto")

    h = cur.hoja
    h.escribir(50, 30, "Correo institucional", fuente=negrita, tam=10, color=(0.2, 0.2, 0.2))
    x = 545 - Hoja.ancho(f"{b.nombre_completo} <{buzon.texto}>", fuente, 8.5)
    h.escribir(x, 30, [nombre_p(b, "exacta"), T(" <"), buzon, T(">")], fuente=fuente, tam=8.5)
    h.linea_recta(50, 38, 545, 38, grosor=1.0)
    cur.y = 46
    cur.linea(asunto, fuente=negrita, tam=13, interlineado=1.6)
    cur.linea("2 mensajes", tam=8.5, color=(0.4, 0.4, 0.4))
    cur.hoja.linea_recta(50, cur.y + 2, 545, cur.y + 2, grosor=0.5, color=(0.6, 0.6, 0.6))
    cur.espacio(8)

    def mensaje(
        emisor: Persona,
        f_emisor: str,
        para: list[tuple[Persona, str]],
        cc: list[tuple[Persona, str]],
        asunto_m: str,
        fecha: str,
        cuerpo: list[list[Parte]],
        firma: list[list[Parte]],
    ) -> None:
        cur.asegurar(60)
        de = [nombre_p(emisor, "exacta"), T(" <"), correo_p(emisor, f_emisor), T(">")]
        cur.hoja.escribir(cur.x, cur.y + 10, "De:", fuente=negrita, tam=9.5)
        x_fin = cur.hoja.escribir(cur.x + 45, cur.y + 10, de, fuente=fuente, tam=9.5)
        x_fecha = 545 - Hoja.ancho(fecha, fuente, 8.5)
        if x_fin + 12 < x_fecha:
            cur.hoja.escribir(x_fecha, cur.y + 10, fecha_p(fecha), fuente=fuente, tam=8.5)
        cur.y += 14
        for etiqueta, lista in (("Para:", para), ("CC:", cc)):
            partes: list[Parte] = []
            for i, (p, formato) in enumerate(lista):
                if i:
                    partes.append(T(", "))
                # nombre y correo de un destinatario van juntos: solo se corta después de la coma
                partes += [nombre_p(p, "exacta"), Parte(" <", unida=True), correo_p(p, formato), Parte(">", unida=True)]
            cur.hoja.escribir(cur.x, cur.y + 10, etiqueta, fuente=negrita, tam=9.5)
            for linea in envolver(partes, cur.ancho - 45, fuente, 9.5):
                cur.hoja.escribir(cur.x + 45, cur.y + 10, linea, fuente=fuente, tam=9.5)
                cur.y += 14
        cur.hoja.escribir(cur.x, cur.y + 10, "Asunto:", fuente=negrita, tam=9.5)
        cur.hoja.escribir(cur.x + 45, cur.y + 10, asunto_m, fuente=fuente, tam=9.5)
        cur.y += 14
        cur.hoja.escribir(cur.x, cur.y + 10, "Fecha:", fuente=negrita, tam=9.5)
        cur.hoja.escribir(cur.x + 45, cur.y + 10, fecha_p(fecha), fuente=fuente, tam=9.5)
        cur.y += 22
        for parrafo in cuerpo:
            cur.parrafo(parrafo, despues=6)
        cur.espacio(4)
        for linea in firma:
            cur.linea(linea, tam=9, interlineado=1.35)
        cur.espacio(10)
        cur.hoja.linea_recta(50, cur.y, 545, cur.y, grosor=0.5, color=(0.6, 0.6, 0.6))
        cur.espacio(10)

    anexo_a, anexo_b = (str(int(v)) for v in rng.integers(2000, 7999, 2))
    mensaje(
        a,
        "inicial",
        [(b, "punto")],
        [(c, "con_anio"), (externo, "mas")],
        asunto,
        "lunes, 3 de agosto de 2026, 10:42",
        [
            [T("Estimada " if b.sexo == "F" else "Estimado "), nombre_p(b, "solo_nombre"), T(":")],
            [
                T(
                    "Junto con saludar, adjunto el informe de honorarios del mes y la planilla de rendición. "
                    "Si necesita aclarar algún punto, me puede llamar al "
                ),  # fmt: skip
                tel_p(a.telefono, "movil_nacional"),
                T(" durante la mañana o escribirme a "),
                correo_p(a, "guion_bajo"),
                T("."),
            ],
            [
                T("También copio a "),
                nombre_p(externo, "parcial"),
                T(", de la consultora externa, que revisó los respaldos del producto 2."),
            ],
            [T("Saludos cordiales.")],
        ],
        [
            [T("--")],
            [nombre_p(a, "exacta")],
            [T("Profesional de apoyo, División de Fomento e Industria")],
            [T("Gobierno Regional Ficticio")],
            [T("Móvil: "), tel_p(a.telefono, "movil_internacional")],
            [T("Fono: "), tel_p(a.telefono_fijo, "regional_parentesis"), T(f" · Anexo {anexo_a}")],
        ],
    )
    mensaje(
        b,
        "punto",
        [(a, "inicial")],
        [(c, "con_anio")],
        f"RE: {asunto}",
        "martes, 4 de agosto de 2026, 16:05",
        [
            [T("Hola "), nombre_p(a, "solo_nombre"), T(":")],
            [
                T(
                    "Recibido, muchas gracias. Revisé el informe y está conforme; solo falta la firma de la "
                    "contraparte. Te llamo mañana al "
                ),  # fmt: skip
                tel_p(a.telefono_fijo, "regional_guion"),
                T(" para coordinar la entrega en la oficina de "),
                T(a.comuna),
                T("."),
            ],
            [T("Un abrazo.")],
        ],
        [
            [nombre_p(b, "apellidos_nombres")],
            [T("Contraparte técnica · Unidad de Control de Gestión")],
            [T("Celular: "), tel_p(b.telefono, "movil_guiones")],
            [T("Teléfono: "), tel_p(b.telefono_fijo, "regional_internacional"), T(f" · anexo {anexo_b}")],
            [url_p(f"https://www.goreficticio.cl/directorio/{sin_tildes(b.apellido_p).lower()}-{anexo_b}")],
        ],
    )
    pie = f"https://correo.goreficticio.cl/mail/u/0/?ik=3fa9c{anexo_a}&view=pt&search=all"
    for hoja in doc.hojas:
        hoja.escribir(50, 818, url_p(pie), fuente=fuente, tam=7, color=(0.4, 0.4, 0.4))
        n = f"{hoja.indice + 1}/{len(doc.hojas)}"
        hoja.escribir(545 - Hoja.ancho(n, fuente, 7), 818, n, fuente=fuente, tam=7, color=(0.4, 0.4, 0.4))
    ruta = "pdf_texto/correo_impreso.pdf"
    doc.guardar(ctx, ruta)
    return doc.archivo(
        "pdft_correo_impreso",
        ruta,
        "Hilo de correo impreso (2 mensajes) con De/Para/CC como 'Nombre <correo>', cuerpo con teléfonos y "
        "firmas con móvil, fijo regional y anexo. Fuente DejaVu incrustada.",
        {"fuente": fuente},
    )


# ---------------------------------------------------------------------------
# 5. Página rotada (/Rotate 90)
# ---------------------------------------------------------------------------


def _pagina_rotada(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    doc = Documento()
    cur = Cursor(
        doc, fuente="helv", tam=10, al_crear=encabezado_gore("helv", "División de Desarrollo Social", f.folio())
    )
    responsable = f.persona()
    cur.linea("ANEXO N° 2 - NÓMINA DE BENEFICIARIOS DEL FONDO REGIONAL", fuente="hebo", tam=12, interlineado=1.8)
    cur.parrafo(
        [
            T(
                "La nómina de beneficiarios se presenta en la página siguiente, en orientación horizontal. "
                "Consultas a la encargada del programa, "
            ),  # fmt: skip
            nombre_p(responsable, "exacta"),
            T(", RUT "),
            rut_p(responsable, "puntos"),
            T(", correo "),
            correo_p(responsable, "punto"),
            T(", teléfono "),
            tel_p(telefono_para(f, responsable, "santiago_parentesis"), "santiago_parentesis"),
            T("."),
        ]
    )
    cur.parrafo([T("Monto total transferido: "), monto_p(f.monto()), T(".")])

    # Página 2: rotada 90°; se escribe en coordenadas visibles (842 x 595).
    h = doc.nueva_hoja(rotacion=90, visible=True)
    h.escribir(50, 50, "NÓMINA DE BENEFICIARIOS - FONDO REGIONAL DE INICIATIVAS LOCALES 2026", fuente="hebo", tam=12)
    h.escribir(50, 66, [T("Resolución de adjudicación de "), fecha_p(f.fecha())], fuente="helv", tam=9)
    formatos_rut = ["puntos", "sin_puntos", "espacios_guion", "guion_largo", "puntos", "sin_puntos", "puntos"]
    formatos_tel = ["movil_internacional", "regional_parentesis", "movil_bloque", "santiago_nacional",
                    "movil_compacto", "regional_guion", "movil_parentesis"]  # fmt: skip
    formatos_correo = ["punto", "inicial", "con_anio", "guion_bajo", "mas", "subdominio", "punto"]
    filas = []
    for i in range(7):
        p = f.persona(dv_valido=i != 2)
        filas.append(
            [
                nombre_p(p, "exacta" if i % 3 else "mayusculas"),
                rut_p(p, formatos_rut[i]),
                correo_p(p, formatos_correo[i]),
                tel_p(telefono_para(f, p, formatos_tel[i]), formatos_tel[i]),
                direccion_p(p),
                T(p.comuna),
                monto_p(f.monto()),
            ]
        )
    y = dibujar_tabla(h, 50, 82, 742, ["Nombre", "RUT", "Correo", "Teléfono", "Dirección", "Comuna", "Monto"], filas,
                      fuente="helv", tam=8)  # fmt: skip
    h.escribir(50, y + 18, "Fuente: Unidad de Control de Gestión, Gobierno Regional Ficticio.", fuente="helv", tam=8)
    pie_de_pagina_rotado = f"Página 2 de 2 · Folio de nómina {f.folio()}"
    h.escribir(792 - Hoja.ancho(pie_de_pagina_rotado, "helv", 7.5), 570, pie_de_pagina_rotado, fuente="helv", tam=7.5)
    doc.hojas[0].escribir(60, 812, "Gobierno Regional Ficticio · Página 1 de 2", fuente="helv", tam=7.5)
    ruta = "pdf_texto/pagina_rotada.pdf"
    doc.guardar(ctx, ruta)
    return doc.archivo(
        "pdft_pagina_rotada",
        ruta,
        "Página 1 vertical normal; página 2 con /Rotate 90 que contiene una tabla horizontal con datos personales "
        "(el texto se escribió girado en el espacio sin rotar para verse horizontal).",
        {"rotaciones": [0, 90]},
    )


# ---------------------------------------------------------------------------
# 6. Texto girado (rotate 90/180/270 y timbre a 30° con morph)
# ---------------------------------------------------------------------------


def _texto_girado(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    doc = Documento()
    cur = Cursor(doc, fuente="helv", tam=10, al_crear=encabezado_gore("helv", "Oficina de Partes", f.folio()))
    firmante, destinatario, receptor, titular = f.persona(), f.persona(), f.persona(), f.persona()
    cur.linea("CERTIFICADO DE RECEPCIÓN DE DOCUMENTOS", fuente="hebo", tam=12, interlineado=1.8)
    cur.parrafo(
        [
            T("Se certifica que "),
            nombre_p(titular, "exacta"),
            T(", RUT "),
            rut_p(titular, "puntos"),
            T(", ingresó la documentación de respaldo del proyecto el día "),
            fecha_p(f.fecha()),
            T(". Para consultas escribir a "),
            correo_p(titular, "punto"),
            T("."),
        ]
    )
    for frase in rng.choice(len(FRASES_ADMINISTRATIVAS), size=4, replace=False):
        cur.parrafo(FRASES_ADMINISTRATIVAS[int(frase)])
    h = doc.hojas[0]

    # Margen izquierdo, 90°: se lee de abajo hacia arriba.
    h.escribir(
        34,
        760,
        [
            T("Firmado electrónicamente por "),
            nombre_p(firmante, "exacta"),
            T(", RUT "),
            rut_p(firmante, "sin_puntos"),
            T(", el "),
            fecha_p(f.fecha()),
        ],
        fuente="helv",
        tam=8,
        angulo=90,
    )
    # Margen derecho, 270°: se lee de arriba hacia abajo.
    url = f"https://verificador.goreficticio.cl/doc/{int(rng.integers(10**7, 10**8))}"
    h.escribir(
        562,
        90,
        [T("Verifique este documento en "), url_p(url), T(" · contacto "), correo_p(firmante, "inicial")],
        fuente="helv",
        tam=8,
        angulo=270,
    )
    # Pie invertido, 180°.
    h.escribir(
        480,
        770,
        [
            T("Copia para "),
            nombre_p(destinatario, "exacta"),
            T(" · fono "),
            tel_p(destinatario.telefono, "movil_internacional"),
        ],  # fmt: skip
        fuente="tiro",
        tam=9,
        angulo=180,
    )
    # Timbre a 30° (morph), centrado en (400, 600).
    angulo = 30.0
    cx, cy, w, alto = 400.0, 600.0, 190.0, 84.0
    t = math.radians(angulo)
    c, s = math.cos(t), math.sin(t)

    def girar(lx: float, ly: float) -> tuple[float, float]:
        return cx + lx * c + ly * s, cy - lx * s + ly * c

    tinta = (0.1, 0.2, 0.65)
    for margen in (0.0, 4.0):
        esquinas = [
            girar(sx * (w / 2 - margen), sy * (alto / 2 - margen)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        h.page.draw_polyline([*esquinas, esquinas[0]], color=tinta, width=1.2 if margen == 0 else 0.6)
    lineas = [
        ([T("OFICINA DE PARTES")], "hebo", 10, -22),
        ([T("RECIBIDO "), fecha_p(f.fecha())], "helv", 8.5, -6),
        ([nombre_p(receptor, "mayusculas")], "helv", 8.5, 10),
        ([T("RUT "), rut_p(receptor, "puntos")], "helv", 8.5, 26),
    ]
    for partes, fuente, tam, ly in lineas:
        largo = Hoja.ancho("".join(p.texto for p in partes), fuente, tam)
        px, py = girar(-largo / 2, ly)
        h.escribir(px, py, partes, fuente=fuente, tam=tam, angulo=angulo, color=tinta, etiquetas={"timbre": True})
    pie_de_pagina(doc, "Gobierno Regional Ficticio · Oficina de Partes")
    ruta = "pdf_texto/texto_girado.pdf"
    doc.guardar(ctx, ruta)
    return doc.archivo(
        "pdft_texto_girado",
        ruta,
        "Texto en la capa de texto girado 90°, 180° y 270° (márgenes y pie) y un timbre a 30° con morph.",
        {"angulos": [0, 90, 180, 270, 30]},
    )


# ---------------------------------------------------------------------------
# 7. Casos de estrés de la capa de texto
# ---------------------------------------------------------------------------


def _estres_capa_texto(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    doc = Documento()
    h = doc.nueva_hoja()
    # CropBox más chico que la MediaBox: desde aquí todas las coordenadas son relativas al CropBox.
    mediabox = pymupdf.Rect(0, 0, *A4)
    recorte = pymupdf.Rect(30, 30, A4[0] - 30, A4[1] - 30)
    h.page.set_cropbox(recorte)
    x = 30
    y = 40.0

    def titulo(texto: str) -> None:
        nonlocal y
        y += 22
        h.escribir(x, y, texto, fuente="hebo", tam=10)
        y += 16

    h.escribir(x, y, "PRUEBAS DE LA CAPA DE TEXTO - CASOS DIFÍCILES", fuente="hebo", tam=13)
    y += 6

    titulo("a) Texto diminuto (4 y 5 pt)")
    p = f.persona()
    h.escribir(
        x,
        y,
        [
            T("Nota al pie: consultas a "),
            correo_p(p, "punto", "estres"),
            T(" o al "),
            tel_p(p.telefono, "movil_internacional", "estres"),
            T("."),
        ],  # fmt: skip
        tam=5,
        etiquetas={"diminuto": True},
    )
    y += 8
    h.escribir(
        x, y, [T("Titular del registro: RUT "), rut_p(p, "puntos", "estres")], tam=4, etiquetas={"diminuto": True}
    )

    titulo("b) Blanco sobre blanco, modo de dibujo invisible y texto bajo una imagen")
    q = f.persona()
    h.escribir(x, y, "Campo reservado:", tam=10)
    h.escribir(x + 95, y, [T("RUT "), rut_p(q, "puntos")], tam=10, color=(1, 1, 1), capa="oculto",
               etiquetas={"oculto": "blanco_sobre_blanco"})  # fmt: skip
    y += 16
    h.escribir(x, y, "Texto invisible:", tam=10)
    h.escribir(x + 95, y, [T("Correo "), correo_p(q, "con_anio"), T(" RUT "), rut_p(q, "sin_puntos")], tam=10,
               render_mode=3, capa="oculto", etiquetas={"oculto": "render_mode_3"})  # fmt: skip
    y += 30
    r = f.persona()
    h.escribir(x, y + 12, "Nota adhesiva pegada sobre el texto:", tam=10)
    bajo = {"oculto": "bajo_imagen"}
    h.escribir(x + 200, y + 10, [T("Correo: "), correo_p(r, "punto"), T("  RUT: "), rut_p(r, "puntos")], tam=8,
               capa="oculto", etiquetas=bajo)  # fmt: skip
    h.escribir(
        x + 200, y + 22, [T("Fono: "), tel_p(r.telefono, "movil_nacional")], tam=8, capa="oculto", etiquetas=bajo
    )
    # Imagen opaca y lisa encima (la marca "REVISADO" queda lejos del texto tapado).
    nota = Image.new("RGB", (580, 80), (255, 238, 150))
    ImageDraw.Draw(nota).text((500, 62), "REVISADO", fill=(120, 90, 20))
    h.page.insert_image(pymupdf.Rect(x + 190, y - 6, x + 480, y + 34), stream=_png(nota), keep_proportion=False)
    y += 36

    titulo("c) Texto fuera del CropBox (presente en el archivo, fuera de la página visible)")
    s = f.persona()
    fuera = {
        "oculto": "fuera_del_cropbox",
        "sistema_coordenadas": "espacio de página PyMuPDF: origen en la esquina superior izquierda del CropBox; "
        "coordenadas negativas o mayores que el ancho/alto quedan fuera de page.rect",
        "mediabox": list(mediabox),
        "cropbox": list(recorte),
        "extraccion": "page.get_textpage(clip=pymupdf.INFINITE_RECT())",
    }
    h.escribir(-24, -12, [T("Borrador: "), replace(nombre_p(s, "exacta"), nivel="estres"), T(" RUT "),
               rut_p(s, "puntos", "estres")],
               tam=9, capa="oculto", etiquetas=fuera)  # fmt: skip
    # Franja derecha entre el CropBox y la MediaBox, girada 270° para que quepa.
    h.escribir(recorte.width + 12, 120, [T("Correo "), correo_p(s, "punto", "estres")], tam=9, angulo=270, capa="oculto",
               etiquetas=fuera)  # fmt: skip
    h.escribir(x, y, "Hay un nombre, un RUT y un correo escritos fuera del recorte visible de esta página.", tam=9)

    titulo("d) RUT cortado entre dos líneas")
    u = f.persona()
    completo = u.rut("puntos")
    corte = completo.index(".", 3) + 1
    fr0, fr1 = completo[:corte], completo[corte:]
    base_et = {"valor_completo": completo, "formato": "puntos", "dv_valido": u.rut_dv_valido}
    x_fin = h.escribir(x, y, [T("El trámite fue presentado por "), nombre_p(u, "exacta"), T(", con RUT ")], tam=10)
    h.escribir(x_fin, y, Parte(fr0, "rut", "estres", {"fragmento": 0, **base_et}), tam=10)
    y += 14
    h.escribir(x, y, [Parte(fr1, "rut", "estres", {"fragmento": 1, **base_et}),
               T(", según consta en el expediente del programa.")], tam=10)  # fmt: skip

    titulo("e) Texto con letras espaciadas (cada carácter por separado)")
    v = f.persona()
    x_fin = h.escribir(x, y, "RUT del beneficiario: ", tam=10)
    h.escribir_espaciado(x_fin, y, rut_p(v, "puntos", "estres"), 2.6, tam=10)
    y += 16
    x_fin = h.escribir(x, y, "Correo: ", tam=10)
    h.escribir_espaciado(x_fin, y, correo_p(v, "punto", "estres"), 1.8, tam=10)

    titulo("f) RUT con guion largo (autocorrección del procesador de texto)")
    w = f.persona()
    h.escribir(
        x,
        y,
        [T("Postulante: "), nombre_p(w, "exacta"), T(", RUT "), rut_p(w, "guion_largo"), T(".")],
        fuente="helv",
        tam=10,
    )
    # Formatos antiguos o disfrazados que no aparecen en otros archivos (sin sorteos nuevos).
    titulo("g) Correo con (at) y celular antiguo de 8 dígitos")
    h.escribir(
        x,
        y,
        [T("Contacto: "), correo_p(w, "at"), T(", celular antiguo "), tel_p(w.telefono, "antiguo_movil_8"), T(".")],
        tam=10,
    )

    titulo("h) URL que contiene un RUT")
    cuerpo, dv = f.rut()
    rut_url = f"{cuerpo}-{dv}"
    url = f"https://tramites.ejemplo.cl/certificado?rut={rut_url}&tipo=honorarios"
    x_fin = h.escribir(x, y, "Descargue su certificado en ", tam=10)
    h.escribir(x_fin, y, url_p(url), tam=10)
    prefijo = url[: url.index(rut_url)]
    x_rut = x_fin + Hoja.ancho(prefijo, "helv", 10)
    h.escribir(x_rut, y, Parte(rut_url, "rut", "base", {"formato": "sin_puntos", "dv_valido": True, "dentro_de_url": True}),
               tam=10, dibujar=False)  # fmt: skip

    ruta = "pdf_texto/estres_capa_texto.pdf"
    doc.guardar(ctx, ruta)
    return doc.archivo(
        "pdft_estres_capa_texto",
        ruta,
        "Casos difíciles de la capa de texto: letra de 4-5 pt, blanco sobre blanco, render mode 3, texto bajo "
        "una imagen, texto fuera del CropBox, RUT cortado entre líneas, letras espaciadas, guion largo, correo "
        "con (at), celular antiguo de 8 dígitos y RUT dentro de una URL.",
        {"cropbox": list(recorte), "mediabox": list(mediabox)},
    )


# ---------------------------------------------------------------------------
# 8. Texto vectorizado (glifos como trazos)
# ---------------------------------------------------------------------------


def _texto_vectorizado(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    origen = Documento()
    cur = Cursor(
        origen, fuente="helv", tam=10, al_crear=encabezado_gore("helv", "División de Fomento e Industria", f.folio())
    )
    p, q = f.persona(), f.persona()
    cur.linea("INFORME DE HONORARIOS (VERSIÓN IMPRESA COMO IMAGEN VECTORIAL)", fuente="hebo", tam=12, interlineado=1.8)
    cur.campo("Nombre:", nombre_p(p, "exacta"))
    cur.campo("RUT:", rut_p(p, "puntos"))
    cur.campo("Correo:", correo_p(p, "punto"))
    cur.campo("Teléfono:", tel_p(p.telefono, "movil_internacional"))
    cur.campo("Dirección:", [direccion_p(p), T(f", {p.comuna}")])
    cur.campo("Monto bruto:", monto_p(f.monto()))
    cur.seccion("Contraparte técnica")
    cur.parrafo(
        [
            T("La contraparte técnica, "),
            nombre_p(q, "exacta"),
            T(", RUT "),
            rut_p(q, "guion_largo"),
            T(", correo "),
            correo_p(q, "inicial"),
            T(", teléfono "),
            tel_p(telefono_para(f, q, "regional_parentesis"), "regional_parentesis"),
            T(", validó los productos el "),
            fecha_p(f.fecha()),
            T("."),
        ],
        fuente="dejavu",
    )
    for frase in rng.choice(len(FRASES_ADMINISTRATIVAS), size=3, replace=False):
        cur.parrafo(FRASES_ADMINISTRATIVAS[int(frase)])
    origen.refinar()
    pagina = origen.doc[0]
    svg = pagina.get_svg_image(text_as_path=True)
    doc_svg = pymupdf.open("svg", svg.encode("utf-8"))
    final = pymupdf.open("pdf", doc_svg.convert_to_pdf())
    doc_svg.close()
    destino = final[0]
    sx, sy = destino.rect.width / pagina.rect.width, destino.rect.height / pagina.rect.height
    if final[0].get_text().strip():
        raise RuntimeError("el PDF vectorizado todavía tiene texto extraíble")
    elementos = []
    for e in origen.elementos():
        e.capa = "vector"
        e.poligono = [[round(px * sx, 3), round(py * sy, 3)] for px, py in e.poligono]
        e.etiquetas["origen_gt"] = "search_for en la página original, antes de vectorizar"
        elementos.append(e)
    ruta = "pdf_texto/texto_vectorizado.pdf"
    final.save(ctx.ruta(ruta), garbage=3, deflate=True, no_new_id=True)
    archivo = Archivo(
        id="pdft_texto_vectorizado",
        ruta=ruta,
        formato="pdf",
        categoria=CATEGORIA,
        descripcion="Página de texto convertida a SVG con text_as_path=True y de vuelta a PDF: los glifos son trazos "
        "(visibles, no extraíbles).",
        paginas=paginas_de(final),
        elementos=elementos,
        etiquetas={"escala": [sx, sy]},
    )
    final.close()
    origen.doc.close()
    return archivo


# ---------------------------------------------------------------------------
# 9. Texto con imágenes incrustadas
# ---------------------------------------------------------------------------


def _lz_linea(lz: Lienzo, x: float, y: float, partes: Sequence[Parte], tam: int, fuente: str = "sans",
              color: tuple[int, int, int] = (30, 30, 30)) -> float:  # fmt: skip
    for p in partes:
        visible = any(c.isalnum() for c in p.texto)
        lz.escribir(x, y, p.texto, tipo=p.tipo, valor=p.valor or p.texto.strip(), nombre_fuente=fuente, tam=tam, color=color,
                    nivel=p.nivel, etiquetas=dict(p.etiquetas), registrar=visible)  # fmt: skip
        x += lz.ancho_texto(p.texto, fuente, tam)
    return x


def _imagen_en(h: Hoja, caja: tuple[float, float, float, float], img: Image.Image, elementos: Sequence[Elemento],
               etiquetas: dict[str, Any], xref: int = 0, stream: bytes | None = None) -> int:  # fmt: skip
    """Inserta la imagen estirada exactamente a ``caja`` y registra sus elementos en coordenadas de página."""
    r = pymupdf.Rect(*caja)
    if xref:
        nuevo = h.page.insert_image(r, xref=xref, keep_proportion=False)
    else:
        nuevo = h.page.insert_image(r, stream=stream or _png(img), keep_proportion=False)
    mapeados = mapear_a_rect(elementos, img.width, img.height, caja)
    for e in mapeados:
        e.etiquetas.update(etiquetas)
        e.etiquetas["escala_pt_por_px"] = round((caja[2] - caja[0]) / img.width, 4)
    h.agregar(mapeados)
    return nuevo


def _mixto_imagenes(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    doc = Documento()
    cur = Cursor(
        doc,
        fuente="helv",
        tam=10,
        ancho=340,
        al_crear=encabezado_gore("helv", "Programa de Prácticas Profesionales", f.folio()),
    )
    postulante, remitente, credencial = f.persona(), f.persona(), f.persona()
    cur.linea("FICHA DE POSTULACIÓN - PROGRAMA DE PRÁCTICAS", fuente="hebo", tam=12, interlineado=1.8)
    cur.campo("Nombre:", nombre_p(postulante, "exacta"), tab=80)
    cur.campo("RUT:", rut_p(postulante, "puntos"), tab=80)
    cur.campo("Correo:", correo_p(postulante, "punto"), tab=80)
    cur.campo("Teléfono:", tel_p(postulante.telefono, "movil_internacional"), tab=80)
    cur.campo("Dirección:", direccion_p(postulante), tab=80)
    cur.campo("Comuna:", postulante.comuna, tab=80)
    cur.ancho = 475
    cur.espacio(10)
    for frase in rng.choice(len(FRASES_ADMINISTRATIVAS), size=3, replace=False):
        cur.parrafo(FRASES_ADMINISTRATIVAS[int(frase)])
    h1 = doc.hojas[0]

    # (a) Foto de la postulación.
    (foto,) = ctx.rostros.tomar("frente", 1)
    ancho_foto = 100.0
    caja = (435.0, 80.0, 435.0 + ancho_foto, 80.0 + ancho_foto * foto.img.height / foto.img.width)
    _imagen_en(h1, caja, foto.img, ProveedorRostros.elementos(foto), {"imagen": "foto_postulacion"})
    h1.escribir(435, caja[3] + 10, "Fotografía", tam=7.5)

    # (b) Pantallazo de un mensaje, insertado a escala.
    cur.espacio(8)
    cur.linea("Respaldo: captura del mensaje recibido", fuente="hebo", tam=9)
    lz = Lienzo.nuevo(640, 230, fondo=(246, 247, 249))
    d = ImageDraw.Draw(lz.img)
    d.rectangle((0, 0, 640, 34), fill=(46, 84, 150))
    _lz_linea(lz, 14, 23, [T("Mensajería institucional")], 15, "sans_negrita", (255, 255, 255))
    _lz_linea(lz, 20, 64, [T("De: "), nombre_p(remitente, "exacta"), T(" <"), correo_p(remitente, "punto"), T(">")], 14)
    _lz_linea(lz, 20, 88, [T("Asunto: Documentos de la postulación")], 14)
    d.line((20, 100, 620, 100), fill=(200, 200, 205), width=1)
    _lz_linea(lz, 20, 128, [T("Hola, adjunto los documentos. Cualquier duda me llamas al")], 14)
    _lz_linea(lz, 20, 150, [tel_p(remitente.telefono, "movil_nacional"), T(" o al fijo "),
              tel_p(telefono_para(f, remitente, "regional_parentesis"), "regional_parentesis"), T(".")], 14)  # fmt: skip
    _lz_linea(lz, 20, 185, [T("Saludos, "), nombre_p(remitente, "parcial")], 14)
    y0 = cur.y + 4
    caja = (60.0, y0, 60.0 + 400.0, y0 + 400.0 * lz.alto / lz.ancho)
    _imagen_en(h1, caja, lz.img, lz.elementos, {"imagen": "pantallazo"})
    cur.y = caja[3] + 10
    cur.parrafo("La captura se incorpora como antecedente de la postulación, a solicitud de la contraparte.")

    # Página 2
    cur.salto()
    h2 = cur.hoja
    cur.linea("CREDENCIAL Y CÓDIGOS DE VERIFICACIÓN", fuente="hebo", tam=12, interlineado=1.8)
    cur.linea([T("Titular de la credencial: "), nombre_p(credencial, "exacta")])

    # (e) La misma foto dos veces (mismo xref), la segunda más chica.
    (foto2,) = ctx.rostros.tomar("frente", 1)
    alto_rel = foto2.img.height / foto2.img.width
    caja_grande = (60.0, 130.0, 200.0, 130.0 + 140.0 * alto_rel)
    xref = _imagen_en(h2, caja_grande, foto2.img, ProveedorRostros.elementos(foto2),
                      {"imagen": "credencial", "xref_compartido": True, "ocurrencia": 1})  # fmt: skip
    caja_chica = (215.0, 130.0, 255.0, 130.0 + 40.0 * alto_rel)
    _imagen_en(h2, caja_chica, foto2.img, ProveedorRostros.elementos(foto2),
               {"imagen": "credencial", "xref_compartido": True, "ocurrencia": 2}, xref=xref)  # fmt: skip

    # (c) RUT en una imagen girada 90°.
    rut_c = rut_p(credencial, "puntos")
    lz_rut = Lienzo.nuevo(340, 56)
    _lz_linea(lz_rut, 10, 40, [T("RUT "), rut_c], 30, "sans_negrita")
    girado = lz_rut.rotar(90)
    caja_rut = (290.0, 130.0, 290.0 + 22.0, 130.0 + 22.0 * girado.alto / girado.ancho)
    _imagen_en(h2, caja_rut, girado.img, girado.elementos, {"imagen": "rut_girado"})

    # (d) Código QR con un RUT dentro de una URL.
    cuerpo, dv = f.rut()
    contenido = f"https://portal.ejemplo.cl/verificar?run={cuerpo}-{dv}"
    img_qr, caja_modulos = _qr(contenido)
    caja_qr = (350.0, 130.0, 460.0, 240.0)
    elemento_qr = Elemento(tipo="qr", pagina=0, poligono=rect(*caja_modulos), valor=contenido, nivel="estres", capa="raster",
                           etiquetas={"contenido": "url_con_rut", "rut_codificado": f"{cuerpo}-{dv}"})  # fmt: skip
    _imagen_en(h2, caja_qr, img_qr, [elemento_qr], {"imagen": "qr"})
    h2.escribir(350, 252, "Escanee para verificar la credencial", tam=7.5)

    # (f) PNG con transparencia (SMask) sobre un fondo de color.
    fono = f.telefono("movil")
    lz_fono = Lienzo.nuevo(420, 56)
    _lz_linea(lz_fono, 10, 38, [T("Tel. "), tel_p(fono, "movil_internacional")], 28, "sans_negrita")
    alfa = Image.eval(lz_fono.img.convert("L"), lambda v: 255 - v)
    rgba = Image.new("RGBA", lz_fono.img.size, (20, 60, 120, 0))
    rgba.putalpha(alfa)
    caja_fono = (60.0, 330.0, 60.0 + 280.0, 330.0 + 280.0 * 56 / 420)
    h2.rectangulo(50, 320, 360, caja_fono[3] + 10, relleno=(0.86, 0.93, 1.0))
    _imagen_en(h2, caja_fono, lz_fono.img, lz_fono.elementos, {"imagen": "png_transparente"}, stream=_png(rgba))
    h2.escribir(60, caja_fono[3] + 26, "Mesa de ayuda del programa (imagen PNG con transparencia).", tam=8)
    pie_de_pagina(doc, "Gobierno Regional Ficticio · Programa de Prácticas Profesionales")
    ruta = "pdf_texto/mixto_imagenes.pdf"
    doc.guardar(ctx, ruta)
    return doc.archivo(
        "pdft_mixto_imagenes",
        ruta,
        "Texto con datos en la capa de texto más imágenes incrustadas: foto, pantallazo escalado, RUT en imagen "
        "girada 90°, código QR con RUT, la misma foto dos veces (mismo xref) y PNG con transparencia.",
        {"rostros_sinteticos": ctx.rostros.solo_sinteticos},
    )


def _qr(contenido: str, modulo: int = 6, borde: int = 4) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Imagen del QR (con zona de silencio) y la caja de los módulos en píxeles."""
    matriz = cv2.QRCodeEncoder.create().encode(contenido)
    if matriz.ndim == 3:
        matriz = matriz[..., 0]
    n = matriz.shape[0]
    grande = np.kron(matriz, np.ones((modulo, modulo), np.uint8))
    lado = (n + 2 * borde) * modulo
    lienzo = np.full((lado, lado), 255, np.uint8)
    o = borde * modulo
    lienzo[o : o + n * modulo, o : o + n * modulo] = grande
    return Image.fromarray(lienzo).convert("RGB"), (o, o, o + n * modulo, o + n * modulo)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def generar(ctx: Contexto) -> list[Archivo]:
    f = ctx.ficticios(MODULO)
    rng = ctx.rng(MODULO)
    archivos = [_cuaderno(ctx, f)]
    for n in range(1, 5):
        archivos.append(_honorarios(ctx, f, rng, n))
    archivos.append(_resolucion(ctx, f, rng))
    archivos.append(_correo_impreso(ctx, f, rng))
    archivos.append(_pagina_rotada(ctx, f, rng))
    archivos.append(_texto_girado(ctx, f, rng))
    archivos.append(_estres_capa_texto(ctx, f, rng))
    archivos.append(_texto_vectorizado(ctx, f, rng))
    archivos.append(_mixto_imagenes(ctx, f, rng))
    return archivos
