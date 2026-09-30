"""PDF escaneados: páginas que son imágenes, sin capa de texto (o con una capa OCR invisible).

Cada página se dibuja con ``Lienzo`` a la resolución del escáner (A4 = 8,27 x 11,69 pulgadas),
se degrada como lo haría un escáner (papel, inclinación, desenfoque, ruido, JPEG, gris,
binarizado de fax) y se inserta como imagen de página completa en una página PDF de
595 x 842 pt. La verdad de terreno se lleva de píxeles a puntos con ``mapear_a_rect`` sobre la
misma caja en que se insertó la imagen (equivale a ``px_a_pt`` con la resolución efectiva).

Archivos:

1. ``informe_300dpi.pdf``: informe de honorarios de 2 páginas, 300 ppp, gris, inclinado 1,2°.
2. ``informe_200dpi_torcido.pdf``: 200 ppp, -2,5°, ruido 10, desenfoque 1,0, JPEG 55.
3. ``ficha_con_foto.pdf``: ficha de postulación con foto pegada y campos de formulario.
4. ``escaneo_invertido.pdf``: página cabeza abajo y planilla apaisada escaneada de lado.
5. ``mixto_texto_y_escaneo.pdf``: página con capa de texto real y página escaneada.
6. ``sandwich_ocr.pdf``: página escaneada con capa OCR invisible (modo de render 3) encima.
7. ``fax_150dpi.pdf``: fax binarizado a 1 bit con tramado, 150 ppp (todo nivel estrés).
8. ``timbre_y_firma.pdf``: timbre azul semitransparente, firma y nota manuscrita con teléfono.
"""

from __future__ import annotations

import copy
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Archivo, Elemento, Pagina, Poligono
from banco_pruebas.ficticios import (
    FORMATOS_CORREO,
    FORMATOS_POR_CLASE,
    FORMATOS_RUT,
    FORMATOS_TELEFONO,
    FRASES_ADMINISTRATIVAS,
    Ficticios,
    Persona,
    formatear_rut,
)
from banco_pruebas.lienzo import (
    Lienzo,
    caja_envolvente,
    desenfocar,
    mapear_a_rect,
    matriz_rotacion,
    rect,
    ruido,
    ruta_fuente,
    sal_pimienta,
    textura_papel,
    transformar_elementos,
    transformar_puntos,
)

MODULO = "pdf_escaneado"
CATEGORIA = "pdf_escaneado"
PREFIJO = "pdfe_"

A4_PULGADAS = (8.27, 11.69)
A4_PT = (595.0, 842.0)
TINTA = (25, 25, 25)
TINTA_GRIS = (95, 95, 95)
AZUL_TIMBRE = (38, 72, 190)
AZUL_LAPIZ = (20, 30, 90)
FONDO_ESCANER = (236, 236, 232)


# ---------------------------------------------------------------------------
# Segmentos de texto: un trozo de línea con su tipo, nivel y etiquetas
# ---------------------------------------------------------------------------


@dataclass
class Seg:
    texto: str
    tipo: str = "texto"
    nivel: str = "base"
    etiquetas: dict[str, Any] = field(default_factory=dict)


def _nivel_lista(p: Persona) -> str:
    return "base" if p.en_lista else "fuera_de_alcance"


def s_nombre(p: Persona, variante: str = "completo") -> Seg:
    texto = {
        "completo": p.nombre_completo,
        "mayusculas": p.nombre_completo.upper(),
        "apellidos_primero": f"{p.apellido_p} {p.apellido_m}, {p.nombres}",
    }[variante]
    return Seg(texto, "nombre", _nivel_lista(p), {"en_lista": p.en_lista, "variante": variante})


def s_direccion(p: Persona) -> Seg:
    return Seg(p.direccion, "direccion", _nivel_lista(p), {"en_lista": p.en_lista, "variante": "completa"})


def s_rut(p: Persona, formato: str, nivel: str | None = None) -> Seg:
    return Seg(
        p.rut(formato), "rut", nivel or FORMATOS_RUT[formato], {"formato": formato, "dv_valido": p.rut_dv_valido}
    )


def s_rut_suelto(cuerpo: int, dv: str, formato: str, dv_valido: bool = True) -> Seg:
    return Seg(
        formatear_rut(cuerpo, dv, formato), "rut", FORMATOS_RUT[formato], {"formato": formato, "dv_valido": dv_valido}
    )


def s_movil(p: Persona, formato: str) -> Seg:
    return Seg(
        p.telefono.formatear(formato), "telefono", FORMATOS_TELEFONO[formato], {"formato": formato, "clase": "movil"}
    )


def s_fijo(p: Persona, rng: np.random.Generator, solo_base: bool = True) -> Seg:
    clase = p.telefono_fijo.clase
    opciones = [f for f in FORMATOS_POR_CLASE[clase] if not solo_base or FORMATOS_TELEFONO[f] == "base"]
    formato = str(rng.choice(opciones))
    return Seg(
        p.telefono_fijo.formatear(formato), "telefono", FORMATOS_TELEFONO[formato], {"formato": formato, "clase": clase}
    )


def s_correo(p: Persona, formato: str) -> Seg:
    return Seg(p.correo(formato), "correo", FORMATOS_CORREO[formato], {"formato": formato})


def s_monto(f: Ficticios) -> Seg:
    return Seg(f.monto(), "texto", "base", {"senuelo": "monto"})


def _pesos(n: int) -> str:
    return f"$ {n:,}".replace(",", ".")


def s_fecha(texto: str) -> Seg:
    return Seg(texto, "texto", "base", {"senuelo": "fecha"})


# ---------------------------------------------------------------------------
# Hoja: lienzo de una página a la resolución del escáner, posiciones en milímetros
# ---------------------------------------------------------------------------


@dataclass
class LineaEscrita:
    """Una línea dibujada (para reconstruir la capa OCR invisible del PDF sándwich)."""

    x: float  # px, origen de la línea base
    y: float
    texto: str
    tam: int  # px
    fuente: str


class Hoja:
    """Página A4 (vertical u horizontal) a ``dpi`` puntos por pulgada, con posiciones en mm."""

    def __init__(self, dpi: int, horizontal: bool = False) -> None:
        self.dpi = dpi
        w, h = (int(round(A4_PULGADAS[0] * dpi)), int(round(A4_PULGADAS[1] * dpi)))
        if horizontal:
            w, h = h, w
        self.lz = Lienzo.nuevo(w, h)
        self.lineas: list[LineaEscrita] = []
        self.ancho_mm = w / dpi * 25.4
        self.alto_mm = h / dpi * 25.4

    def px(self, mm: float) -> float:
        return mm * self.dpi / 25.4

    def tam(self, pt: float) -> int:
        return max(6, int(round(pt * self.dpi / 72)))

    def ancho_mm_texto(self, texto: str, pt: float = 11, fuente: str = "sans") -> float:
        return self.lz.ancho_texto(texto, fuente, self.tam(pt)) * 25.4 / self.dpi

    def linea(
        self,
        x_mm: float,
        y_mm: float,
        segs: list[Seg],
        *,
        pt: float = 11,
        fuente: str = "sans",
        color: tuple[int, int, int] = TINTA,
    ) -> float:
        """Escribe los segmentos seguidos en una línea base; devuelve la x final en mm."""
        tam = self.tam(pt)
        x, y = self.px(x_mm), self.px(y_mm)
        x_ini = x
        for s in segs:
            if s.texto.strip():
                self.lz.escribir(
                    x,
                    y,
                    s.texto,
                    tipo=s.tipo,
                    valor=s.texto.strip(),
                    nombre_fuente=fuente,
                    tam=tam,
                    color=color,
                    nivel=s.nivel,
                    etiquetas={**s.etiquetas, "tam_pt": pt, "dpi": self.dpi},
                )
            x += self.lz.ancho_texto(s.texto, fuente, tam)
        self.lineas.append(LineaEscrita(x_ini, y, "".join(s.texto for s in segs), tam, fuente))
        return x * 25.4 / self.dpi

    def texto(self, x_mm: float, y_mm: float, texto: str, **kw: Any) -> float:
        return self.linea(x_mm, y_mm, [Seg(texto)], **kw)

    def centrado(self, y_mm: float, texto: str, *, pt: float = 11, fuente: str = "sans", **kw: Any) -> float:
        w = self.ancho_mm_texto(texto, pt, fuente)
        return self.linea((self.ancho_mm - w) / 2, y_mm, [Seg(texto)], pt=pt, fuente=fuente, **kw)

    def raya(self, x0: float, y0: float, x1: float, y1: float, grosor_mm: float = 0.25, color=TINTA) -> None:
        d = ImageDraw.Draw(self.lz.img)
        d.line(
            [(self.px(x0), self.px(y0)), (self.px(x1), self.px(y1))], fill=color, width=max(1, int(self.px(grosor_mm)))
        )

    def caja(self, x0: float, y0: float, x1: float, y1: float, grosor_mm: float = 0.25, color=TINTA) -> None:
        d = ImageDraw.Draw(self.lz.img)
        d.rectangle(
            [self.px(x0), self.px(y0), self.px(x1), self.px(y1)], outline=color, width=max(1, int(self.px(grosor_mm)))
        )

    def tabla(
        self,
        x_mm: float,
        y_mm: float,
        anchos_mm: list[float],
        filas: list[list[Seg]],
        *,
        alto_mm: float = 7.0,
        pt: float = 10,
    ) -> float:
        """Tabla con bordes; la primera fila es encabezado en negrita. Devuelve la y final en mm."""
        x_final = x_mm + sum(anchos_mm)
        for i, fila in enumerate(filas):
            y0 = y_mm + i * alto_mm
            self.raya(x_mm, y0, x_final, y0)
            x = x_mm
            for ancho, celda in zip(anchos_mm, fila, strict=True):
                fuente = "sans_negrita" if i == 0 else "sans"
                self.linea(x + 1.5, y0 + alto_mm * 0.7, [celda], pt=pt, fuente=fuente)
                x += ancho
        y_fin = y_mm + len(filas) * alto_mm
        self.raya(x_mm, y_fin, x_final, y_fin)
        x = x_mm
        for ancho in [0.0, *anchos_mm]:
            x += ancho
            self.raya(x, y_mm, x, y_fin)
        return y_fin


# ---------------------------------------------------------------------------
# Escaneo: degradaciones y paso a PDF
# ---------------------------------------------------------------------------


@dataclass
class Degradacion:
    inclinacion: float = 0.0  # grados, antihorario
    giro: int = 0  # 0, 90, 180, 270: hoja puesta de lado o cabeza abajo en el escáner
    ruido: float = 4.0
    desenfoque: float = 0.6
    calidad: int = 80  # JPEG
    gris: bool = False
    sal_pimienta: float = 0.0
    binarizar: bool = False  # fax: 1 bit con tramado Floyd-Steinberg (se guarda sin pérdida)
    papel: bool = True

    def nombres(self) -> list[str]:
        salida = []
        if self.papel:
            salida.append("papel")
        if self.giro:
            salida.append(f"giro_{self.giro}")
        if self.inclinacion:
            salida.append("inclinacion")
        if self.desenfoque:
            salida.append("desenfoque")
        if self.ruido:
            salida.append("ruido")
        if self.sal_pimienta:
            salida.append("sal_pimienta")
        if self.binarizar:
            salida.append("binarizado")
        else:
            salida.append("jpeg")
        if self.gris:
            salida.append("gris")
        return salida


@dataclass
class Escaneada:
    datos: bytes
    ancho: int
    alto: int
    elementos: list[Elemento]  # en píxeles de la imagen final
    matriz: np.ndarray  # píxeles de la hoja -> píxeles de la imagen final


def _matriz_giro(w: int, h: int, grados: int) -> np.ndarray:
    """Misma matriz que usa ``Lienzo.rotar`` para múltiplos de 90."""
    return {
        0: np.eye(3),
        90: np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64),
        180: np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float64),
        270: np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
    }[grados % 360]


def escanear(hoja: Hoja, rng: np.random.Generator, d: Degradacion) -> Escaneada:
    lz = hoja.lz
    if d.papel:
        # textura a media resolución y reescalada: el grano del papel no necesita 300 ppp
        # y textura_papel a tamaño completo cuesta varios segundos por página
        papel_chico = textura_papel((lz.ancho + 1) // 2, (lz.alto + 1) // 2, rng)
        papel = np.asarray(papel_chico.resize((lz.ancho, lz.alto), Image.Resampling.BILINEAR), np.float32)
        arr = np.asarray(lz.img, np.float32) * (papel / 255.0)
        lz = Lienzo(Image.fromarray(arr.clip(0, 255).astype(np.uint8)), lz.elementos)
    m = np.eye(3)
    if d.giro % 360:
        m = _matriz_giro(lz.ancho, lz.alto, d.giro) @ m
        lz = lz.rotar(d.giro)
    if d.inclinacion:
        m = matriz_rotacion(lz.ancho, lz.alto, d.inclinacion, expandir=False)[0] @ m
        lz = lz.inclinar_escaneo(d.inclinacion, fondo=FONDO_ESCANER)
    img = lz.img
    if d.gris or d.binarizar:
        img = img.convert("L")
    if d.desenfoque:
        img = desenfocar(img, d.desenfoque)
    if d.ruido:
        img = ruido(img, rng, d.ruido)
    if d.binarizar:
        # niveles de fax: el fondo satura a blanco y el tramado queda en bordes y grises
        arr = (np.asarray(img, np.float32) - 60.0) * (255.0 / (222.0 - 60.0))
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if d.sal_pimienta:
        img = sal_pimienta(img, rng, d.sal_pimienta)
    buf = io.BytesIO()
    if d.binarizar:
        img.convert("1").save(buf, "PNG", optimize=True)
    else:
        img.save(buf, "JPEG", quality=d.calidad, optimize=True)
    elementos = copy.deepcopy(lz.elementos)
    for e in elementos:
        e.capa = "raster"
        e.etiquetas["degradacion"] = d.nombres()
    return Escaneada(buf.getvalue(), img.width, img.height, elementos, m)


def pagina_escaneada(doc: pymupdf.Document, esc: Escaneada) -> tuple[pymupdf.Page, list[Elemento]]:
    """Agrega una página A4 con la imagen a página completa y devuelve los elementos en puntos."""
    indice = doc.page_count
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    page.insert_image(page.rect, stream=esc.datos, keep_proportion=False)
    elementos = mapear_a_rect(esc.elementos, esc.ancho, esc.alto, (0.0, 0.0, *A4_PT))
    for e in elementos:
        e.pagina = indice
    return page, elementos


def _paginas(doc: pymupdf.Document) -> list[Pagina]:
    return [
        Pagina(indice=i, ancho=p.rect.width, alto=p.rect.height, unidad="pt", rotacion=p.rotation)
        for i, p in enumerate(doc)
    ]


def _guardar(doc: pymupdf.Document, ctx: Contexto, ruta: str) -> None:
    doc.set_metadata({"creator": "Escaner de oficina ficticio", "producer": "banco_pruebas"})
    # no_new_id: sin /ID aleatorio en el trailer, para que el archivo sea idéntico en cada corrida
    doc.save(ctx.ruta(ruta), garbage=3, deflate=True, no_new_id=True)
    doc.close()


def _archivo(
    id: str, ruta: str, descripcion: str, doc: pymupdf.Document, elementos: list[Elemento], **etiquetas: Any
) -> Archivo:
    return Archivo(
        id=PREFIJO + id,
        ruta=ruta,
        formato="pdf",
        categoria=CATEGORIA,
        descripcion=descripcion,
        paginas=_paginas(doc),
        elementos=elementos,
        etiquetas=etiquetas,
    )


# ---------------------------------------------------------------------------
# Texto en PDF (capa de texto real o capa OCR invisible), con GT desde search_for
# ---------------------------------------------------------------------------


def _quad_a_poligono(q: pymupdf.Quad) -> Poligono:
    return [[round(p.x, 3), round(p.y, 3)] for p in (q.ul, q.ur, q.lr, q.ll)]


def _centro(poligono: Poligono) -> tuple[float, float]:
    x0, y0, x1, y1 = caja_envolvente(poligono)
    return (x0 + x1) / 2, (y0 + y1) / 2


def _buscar_cerca(page: pymupdf.Page, valor: str, cerca: tuple[float, float]) -> Poligono:
    """Quad de ``page.search_for(valor)`` más cercano a ``cerca`` (los textos neutros se repiten)."""
    hits = page.search_for(valor, quads=True)
    if not hits:
        raise RuntimeError(f"no se encontró {valor!r} en la página {page.number}")
    mejor = min(hits, key=lambda q: math.dist(_centro(_quad_a_poligono(q)), cerca))
    poligono = _quad_a_poligono(mejor)
    if math.dist(_centro(poligono), cerca) > 12:
        raise RuntimeError(f"{valor!r} quedó lejos de donde se escribió ({poligono} vs {cerca})")
    return poligono


class PaginaTexto:
    """Escribe líneas con ``insert_text`` y registra cada segmento con el quad de ``search_for``."""

    ALIAS = {"sans": "dvs", "sans_negrita": "dvsb", "serif": "dvse", "mono": "dvm"}

    def __init__(self, page: pymupdf.Page, capa: str = "texto", render_mode: int = 0) -> None:
        self.page = page
        self.capa = capa
        self.render_mode = render_mode
        self._fuentes: dict[str, pymupdf.Font] = {}
        self._pendientes: list[tuple[Seg, tuple[float, float], dict[str, Any]]] = []

    def _fuente(self, nombre: str) -> tuple[str, pymupdf.Font]:
        alias = self.ALIAS[nombre]
        if nombre not in self._fuentes:
            self.page.insert_font(fontname=alias, fontfile=str(ruta_fuente(nombre)))
            self._fuentes[nombre] = pymupdf.Font(fontfile=str(ruta_fuente(nombre)))
        return alias, self._fuentes[nombre]

    def linea(
        self,
        x: float,
        y: float,
        segs: list[Seg],
        *,
        pt: float = 11,
        fuente: str = "sans",
        angulo: float = 0.0,
        registrar: bool = True,
        etiquetas: dict[str, Any] | None = None,
    ) -> None:
        alias, f = self._fuente(fuente)
        texto = "".join(s.texto for s in segs)
        punto = pymupdf.Point(x, y)
        morph = (punto, pymupdf.Matrix(angulo)) if angulo else None
        self.page.insert_text(
            punto, texto, fontsize=pt, fontname=alias, color=(0.1, 0.1, 0.1), render_mode=self.render_mode, morph=morph
        )
        if not registrar:
            return
        t = math.radians(angulo)
        dx, dy = math.cos(t), -math.sin(t)  # dirección de la línea base (y hacia abajo)
        avance = 0.0
        for s in segs:
            ancho = f.text_length(s.texto, fontsize=pt)
            if s.texto.strip():
                medio = avance + f.text_length(s.texto.rstrip(), fontsize=pt) / 2
                altura = (f.ascender + f.descender) / 2 * pt  # centro vertical de la caja de search_for
                cx = x + dx * medio + dy * altura
                cy = y + dy * medio - dx * altura
                self._pendientes.append((s, (cx, cy), {"tam_pt": pt, "fuente": fuente, **(etiquetas or {})}))
            avance += ancho

    def resolver(self) -> list[Elemento]:
        salida = []
        for s, cerca, extra in self._pendientes:
            valor = s.texto.strip()
            salida.append(
                Elemento(
                    tipo=s.tipo,
                    pagina=self.page.number,
                    poligono=_buscar_cerca(self.page, valor, cerca),
                    valor=valor,
                    nivel=s.nivel,
                    capa=self.capa,
                    etiquetas={**s.etiquetas, **extra},
                )
            )
        self._pendientes.clear()
        return salida


# ---------------------------------------------------------------------------
# Contenidos
# ---------------------------------------------------------------------------


def _encabezado(h: Hoja, f: Ficticios, titulo: str, subtitulo: str | None = None) -> None:
    h.texto(20, 18, "GOBIERNO REGIONAL DE EJEMPLO", pt=11, fuente="sans_negrita")
    h.texto(20, 23, "División de Fomento e Industria", pt=9, color=TINTA_GRIS)
    h.linea(145, 18, [Seg("Folio N° "), Seg(f.folio())], pt=10)
    h.raya(20, 27, h.ancho_mm - 20, 27, 0.35)
    h.centrado(38, titulo, pt=14, fuente="sans_negrita")
    if subtitulo:
        h.centrado(45, subtitulo, pt=10.5)


def _informe_pagina1(
    h: Hoja, f: Ficticios, rng: np.random.Generator, p: Persona, formatos: dict[str, str], pt: float = 11
) -> None:
    _encabezado(h, f, "INFORME MENSUAL DE ACTIVIDADES", "Contrato de prestación de servicios a honorarios")
    y = 58.0
    h.texto(20, y, "1. ANTECEDENTES DEL PRESTADOR", pt=pt, fuente="sans_negrita")
    salto = pt * 0.6
    y += salto + 2
    h.linea(24, y, [Seg("Nombre: "), s_nombre(p, formatos.get("nombre", "completo"))], pt=pt)
    y += salto
    h.linea(24, y, [Seg("RUT: "), s_rut(p, formatos["rut"])], pt=pt)
    y += salto
    h.linea(24, y, [Seg("Domicilio: "), s_direccion(p), Seg(f", {p.comuna}")], pt=pt)
    y += salto
    x = h.linea(24, y, [Seg("Teléfono: "), s_movil(p, formatos["movil"])], pt=pt)
    h.linea(max(x + 8, 110), y, [Seg("Correo: "), s_correo(p, formatos["correo"])], pt=pt)
    y += salto
    h.linea(24, y, [Seg("Teléfono fijo: "), s_fijo(p, rng)], pt=pt)
    y += salto + 6
    h.texto(20, y, "2. PRODUCTOS COMPROMETIDOS", pt=pt, fuente="sans_negrita")
    y += salto + 2
    for frase in rng.choice(FRASES_ADMINISTRATIVAS[:8], 4, replace=False):
        h.texto(24, y, str(frase), pt=pt - 1)
        y += salto
    y += 6
    h.texto(20, y, "3. MONTOS DEL PERÍODO", pt=pt, fuente="sans_negrita")
    y += salto + 2
    bruto = s_monto(f)
    n_bruto = int(bruto.texto.removeprefix("$ ").replace(".", ""))
    retencion = int(round(n_bruto * 0.1525))
    for etiqueta, monto in (
        ("Monto bruto: ", bruto),
        ("Retención de impuesto (15,25%): ", Seg(_pesos(retencion), etiquetas={"senuelo": "monto"})),
        ("Monto líquido a pagar: ", Seg(_pesos(n_bruto - retencion), etiquetas={"senuelo": "monto"})),
    ):
        h.linea(24, y, [Seg(etiqueta), monto], pt=pt)
        y += salto
    h.linea(24, y, [Seg("Fecha de pago estimada: "), s_fecha(f.fecha())], pt=pt)
    y += salto + 6
    for frase in FRASES_ADMINISTRATIVAS[8:11]:
        h.texto(20, y, frase, pt=pt - 1.5)
        y += salto - 0.5
    # pie diminuto: correo en 5 pt (nivel estrés por tamaño)
    h.linea(
        20,
        h.alto_mm - 14,
        [
            Seg("Consultas sobre este informe: "),
            Seg(p.correo("mas"), "correo", "estres", {"formato": "mas", "diminuto": True}),
        ],
        pt=5,
        color=TINTA_GRIS,
    )


def _informe_pagina2(h: Hoja, f: Ficticios, rng: np.random.Generator, q: Persona, pt: float = 11) -> None:
    salto = pt * 0.6
    h.texto(20, 22, "4. DETALLE DE ACTIVIDADES REALIZADAS", pt=pt, fuente="sans_negrita")
    actividades = [
        "Reunión de coordinación",
        "Taller con organizaciones",
        "Visita a terreno",
        "Elaboración de informe",
        "Mesa técnica regional",
    ]
    comunas = rng.choice(["Talca", "Chillán", "Temuco", "Valdivia", "Rancagua", "Copiapó"], 5, replace=False)
    filas = [[Seg("N°"), Seg("Actividad"), Seg("Fecha"), Seg("Comuna"), Seg("Monto")]]
    for i, (act, com) in enumerate(zip(actividades, comunas, strict=True), 1):
        filas.append([Seg(str(i)), Seg(act), s_fecha(f.fecha()), Seg(str(com)), s_monto(f)])
    y = h.tabla(20, 27, [10, 56, 52, 24, 28], filas, alto_mm=7.5, pt=pt - 1.5)
    y += 12
    h.texto(20, y, "5. CONTRAPARTE TÉCNICA", pt=pt, fuente="sans_negrita")
    y += salto + 2
    h.linea(24, y, [Seg("Nombre: "), s_nombre(q, "mayusculas")], pt=pt)
    y += salto
    h.linea(24, y, [Seg("RUT: "), s_rut(q, "sin_puntos")], pt=pt)
    y += salto
    h.linea(24, y, [Seg("Correo institucional: "), s_correo(q, "inicial")], pt=pt)
    y += salto
    h.linea(24, y, [Seg("Anexo telefónico: "), s_fijo(q, rng)], pt=pt)
    y += salto + 6
    h.texto(20, y, "6. VALIDACIÓN", pt=pt, fuente="sans_negrita")
    y += salto + 2
    h.texto(24, y, FRASES_ADMINISTRATIVAS[3], pt=pt - 1)
    y += salto
    h.texto(24, y, FRASES_ADMINISTRATIVAS[4], pt=pt - 1)
    y += 30
    h.raya(30, y, 90, y)
    h.raya(120, y, 180, y)
    h.texto(38, y + 5, "Firma del prestador", pt=9)
    h.texto(126, y + 5, "Firma contraparte técnica", pt=9)
    h.centrado(h.alto_mm - 12, "Página 2 de 2", pt=8, color=TINTA_GRIS)


def _gen_informe_300(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    ruta = "pdf_escaneado/informe_300dpi.pdf"
    p = f.persona()
    q = f.persona(en_lista=False)
    d = Degradacion(inclinacion=1.2, ruido=5, desenfoque=0.6, calidad=78, gris=True)
    doc = pymupdf.open()
    elementos: list[Elemento] = []
    h1 = Hoja(300)
    _informe_pagina1(h1, f, rng, p, {"rut": "puntos", "movil": "movil_internacional", "correo": "punto"})
    h2 = Hoja(300)
    _informe_pagina2(h2, f, rng, q)
    for h in (h1, h2):
        elementos += pagina_escaneada(doc, escanear(h, rng, d))[1]
    a = _archivo(
        "informe_300dpi",
        ruta,
        "Informe de honorarios escaneado a 300 ppp en gris, inclinado 1,2°; contraparte fuera de la lista.",
        doc,
        elementos,
        dpi=300,
        degradacion=d.__dict__,
    )
    _guardar(doc, ctx, ruta)
    return a


def _gen_informe_200_torcido(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    ruta = "pdf_escaneado/informe_200dpi_torcido.pdf"
    p = f.persona(dv_valido=False)
    d = Degradacion(inclinacion=-2.5, ruido=10, desenfoque=1.0, calidad=55, sal_pimienta=0.0005)
    doc = pymupdf.open()
    h = Hoja(200)
    _informe_pagina1(
        h,
        f,
        rng,
        p,
        {"rut": "espacios_guion", "movil": "movil_parentesis", "correo": "guion_bajo", "nombre": "apellidos_primero"},
        pt=10.5,
    )
    _, elementos = pagina_escaneada(doc, escanear(h, rng, d))
    a = _archivo(
        "informe_200dpi_torcido",
        ruta,
        "Informe de honorarios a 200 ppp, -2,5°, ruido 10, desenfoque 1,0 y JPEG calidad 55; RUT con DV inválido.",
        doc,
        elementos,
        dpi=200,
        degradacion=d.__dict__,
    )
    _guardar(doc, ctx, ruta)
    return a


def _campo(h: Hoja, x: float, y: float, ancho: float, etiqueta: str, valor: Seg, pt: float = 11) -> None:
    """Campo de formulario: etiqueta chica arriba y caja con el valor escrito a máquina."""
    h.texto(x, y, etiqueta, pt=7.5, color=TINTA_GRIS)
    h.caja(x, y + 1.2, x + ancho, y + 8.2, 0.2, (90, 90, 90))
    h.linea(x + 1.8, y + 6.2, [valor], pt=pt, fuente="mono")


def _gen_ficha(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    ruta = "pdf_escaneado/ficha_con_foto.pdf"
    p = f.persona()
    emergencia = f.persona(en_lista=False)
    h = Hoja(300)
    _encabezado(h, f, "FICHA DE POSTULACIÓN", "Programa Regional de Becas de Especialización 2026")
    # foto tamaño carné (35 x 45 mm) pegada en la esquina
    (rostro,) = ctx.rostros.tomar("frente", 1)
    fx, fy, fw = 152.0, 52.0, 35.0
    h.lz.pegar(
        rostro.img,
        h.px(fx),
        h.px(fy),
        ancho=int(round(h.px(fw))),
        elementos=ctx.rostros.elementos(rostro, "base", {"origen": "escaneo", "foto_carne": True}),
    )
    alto_foto = rostro.img.height * fw / rostro.img.width
    h.caja(fx - 1, fy - 1, fx + fw + 1, fy + alto_foto + 1, 0.3)
    h.texto(20, 56, "1. ANTECEDENTES PERSONALES", pt=11, fuente="sans_negrita")
    _campo(h, 20, 62, 125, "Nombre completo", s_nombre(p, "mayusculas"))
    _campo(h, 20, 74, 60, "RUT", s_rut(p, "puntos"))
    _campo(h, 85, 74, 60, "Fecha de nacimiento", s_fecha(p.nacimiento))
    _campo(h, 20, 86, 125, "Dirección", s_direccion(p))
    _campo(h, 20, 98, 60, "Comuna", Seg(p.comuna))
    _campo(h, 85, 98, 60, "Sexo", Seg("Femenino" if p.sexo == "F" else "Masculino"))
    _campo(h, 20, 110, 80, "Teléfono móvil", s_movil(p, "movil_nacional"))
    _campo(h, 105, 110, 82, "Teléfono fijo", s_fijo(p, rng))
    _campo(h, 20, 122, 167, "Correo electrónico", s_correo(p, "con_anio"))
    h.texto(20, 142, "2. ANTECEDENTES ACADÉMICOS", pt=11, fuente="sans_negrita")
    _campo(h, 20, 148, 167, "Institución", Seg("Universidad de Ejemplo del Sur"))
    _campo(h, 20, 160, 110, "Título profesional", Seg("Ingeniería en Recursos Naturales"))
    _campo(h, 135, 160, 52, "Año de titulación", Seg(str(int(rng.integers(2010, 2024)))))
    h.texto(20, 180, "3. CONTACTO EN CASO DE EMERGENCIA", pt=11, fuente="sans_negrita")
    _campo(h, 20, 186, 110, "Nombre", s_nombre(emergencia, "completo"))
    _campo(h, 135, 186, 52, "Parentesco", Seg("Hermana" if emergencia.sexo == "F" else "Hermano"))
    _campo(h, 20, 198, 80, "Teléfono", s_movil(emergencia, "movil_bloque"))
    y = 218.0
    for marcada, texto in (
        (True, "Declaro que la información entregada es fidedigna."),
        (True, "Autorizo el uso de mis datos para fines del proceso de selección."),
        (False, "Solicito ser notificado por carta certificada."),
    ):
        h.caja(20, y - 3.4, 23.6, y + 0.2, 0.25)
        if marcada:
            h.raya(20.6, y - 1.6, 21.8, y - 0.4, 0.4)
            h.raya(21.8, y - 0.4, 23.2, y - 3.0, 0.4)
        h.texto(26, y, texto, pt=10)
        y += 7
    y += 18
    h.raya(25, y, 90, y)
    h.texto(35, y + 5, "Firma del postulante", pt=9)
    h.linea(120, y, [Seg("Fecha: "), s_fecha(f.fecha())], pt=10)
    d = Degradacion(inclinacion=0.7, ruido=4, desenfoque=0.5, calidad=80)
    doc = pymupdf.open()
    _, elementos = pagina_escaneada(doc, escanear(h, rng, d))
    a = _archivo(
        "ficha_con_foto",
        ruta,
        "Ficha de postulación escaneada a 300 ppp con foto carné pegada y campos de formulario.",
        doc,
        elementos,
        dpi=300,
        degradacion=d.__dict__,
    )
    _guardar(doc, ctx, ruta)
    return a


def _gen_invertido(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    ruta = "pdf_escaneado/escaneo_invertido.pdf"
    doc = pymupdf.open()
    elementos: list[Elemento] = []
    # Página 1: certificado escaneado cabeza abajo
    p = f.persona()
    h = Hoja(200)
    _encabezado(h, f, "CERTIFICADO DE PARTICIPACIÓN")
    y = 62.0
    h.linea(20, y, [Seg("Se certifica que "), s_nombre(p), Seg(",")], pt=11.5)
    y += 7
    h.linea(20, y, [Seg("cédula de identidad N° "), s_rut(p, "puntos"), Seg(", con domicilio en")], pt=11.5)
    y += 7
    h.linea(20, y, [s_direccion(p), Seg(f", comuna de {p.comuna},")], pt=11.5)
    y += 7
    h.texto(20, y, "participó en el programa de capacitación en gestión territorial.", pt=11.5)
    y += 12
    h.linea(20, y, [Seg("Contacto: "), s_movil(p, "movil_guiones"), Seg(" / "), s_correo(p, "subdominio")], pt=11.5)
    y += 12
    h.linea(20, y, [Seg("Se extiende el presente certificado con fecha "), s_fecha(f.fecha()), Seg(".")], pt=11.5)
    y += 40
    h.raya(70, y, 140, y)
    h.centrado(y + 5, "Coordinación del Programa", pt=9)
    d1 = Degradacion(giro=180, inclinacion=0.6, ruido=5, desenfoque=0.6, calidad=75, gris=True)
    elementos += pagina_escaneada(doc, escanear(h, rng, d1))[1]
    # Página 2: planilla apaisada escaneada de lado (contenido girado 90° en una página vertical)
    h = Hoja(200, horizontal=True)
    h.texto(15, 16, "GOBIERNO REGIONAL DE EJEMPLO", pt=10, fuente="sans_negrita")
    h.centrado(26, "PLANILLA DE ASISTENCIA – TALLER DE FORMULACIÓN DE PROYECTOS", pt=12.5, fuente="sans_negrita")
    h.linea(15, 34, [Seg("Fecha: "), s_fecha(f.fecha()), Seg("    Lugar: Salón auditorio")], pt=10)
    filas: list[list[Seg]] = [[Seg("N°"), Seg("Nombre"), Seg("RUT"), Seg("Teléfono"), Seg("Correo"), Seg("Firma")]]
    formatos_rut = ["puntos", "sin_puntos", "puntos", "guion_largo", "sin_puntos", "puntos"]
    formatos_tel = ["movil_nacional", "movil_internacional", "movil_bloque", "movil_compacto", "movil_nacional"]
    firmantes = []
    for i in range(6):
        per = f.persona(en_lista=i not in (2, 4), siete_digitos=i == 3)
        firmantes.append(per)
        if i == 5:
            cuerpo, dv = f.rut_con_k()
            rut = s_rut_suelto(cuerpo, dv, "puntos")
            tel = s_fijo(per, rng)
        else:
            rut = s_rut(per, formatos_rut[i])
            tel = s_movil(per, formatos_tel[i])
        filas.append([Seg(str(i + 1)), s_nombre(per), rut, tel, s_correo(per, "punto"), Seg("")])
    anchos = [9, 72, 36, 40, 74, 36]
    y0 = 42.0
    h.tabla(15, y0, anchos, filas, alto_mm=11, pt=9.5)
    x_firma = 15 + sum(anchos[:-1]) + 3
    for i, per in enumerate(firmantes):
        yb = y0 + (i + 1) * 11 + 8
        h.lz.escribir_irregular(
            h.px(x_firma),
            h.px(yb),
            f"{per.nombres[0]}. {per.apellido_p}",
            rng,
            tipo="firma",
            nombre_fuente="stix_cursiva",
            tam=h.tam(12),
            nivel="fuera_de_alcance",
            etiquetas={"manuscrito": True},
        )
    d2 = Degradacion(giro=90, inclinacion=-0.8, ruido=5, desenfoque=0.6, calidad=75, gris=True)
    elementos += pagina_escaneada(doc, escanear(h, rng, d2))[1]
    a = _archivo(
        "escaneo_invertido",
        ruta,
        "Página 1 escaneada cabeza abajo (180°); página 2 planilla apaisada escaneada de lado (90°), sin /Rotate.",
        doc,
        elementos,
        dpi=200,
        degradacion=[d1.__dict__, d2.__dict__],
    )
    _guardar(doc, ctx, ruta)
    return a


def _gen_mixto(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    ruta = "pdf_escaneado/mixto_texto_y_escaneo.pdf"
    doc = pymupdf.open()
    # Página 1: oficio con capa de texto real
    p = f.persona()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    pt_ = PaginaTexto(page)
    pt_.linea(57, 60, [Seg("GOBIERNO REGIONAL DE EJEMPLO")], pt=11, fuente="sans_negrita")
    pt_.linea(57, 74, [Seg("División de Administración y Finanzas")], pt=9)
    pt_.linea(340, 100, [Seg("ORD. N° "), Seg(f.folio())], pt=10.5, fuente="sans_negrita")
    pt_.linea(340, 115, [Seg("ANT.: Solicitud de acceso a la información")], pt=9.5)
    pt_.linea(340, 128, [Seg("MAT.: Remite antecedentes solicitados")], pt=9.5)
    pt_.linea(340, 150, [Seg("Concepción, "), s_fecha(f.fecha())], pt=9.5)
    pt_.linea(57, 185, [Seg("A: "), s_nombre(p, "mayusculas")], pt=11)
    pt_.linea(57, 200, [Seg("DE: JEFATURA DIVISIÓN DE ADMINISTRACIÓN Y FINANZAS")], pt=11)
    y = 235.0
    for frase in (FRASES_ADMINISTRATIVAS[7], FRASES_ADMINISTRATIVAS[2], FRASES_ADMINISTRATIVAS[5]):
        pt_.linea(57, y, [Seg(frase)], pt=10.5)
        y += 17
    y += 12
    pt_.linea(57, y, [Seg("Datos del solicitante registrados en la solicitud:")], pt=10.5)
    y += 20
    pt_.linea(75, y, [Seg("RUT: "), s_rut(p, "guion_largo")], pt=10.5)
    y += 16
    pt_.linea(75, y, [Seg("Correo de contacto: "), s_correo(p, "mayusculas")], pt=10.5)
    y += 16
    pt_.linea(75, y, [Seg("Teléfono: "), s_fijo(p, rng)], pt=10.5)
    y += 16
    pt_.linea(75, y, [Seg("Dirección: "), s_direccion(p), Seg(f", {p.comuna}")], pt=10.5)
    y += 34
    pt_.linea(57, y, [Seg("Se adjunta como anexo el comprobante de recepción escaneado.")], pt=10.5)
    y += 40
    pt_.linea(57, y, [Seg("Saluda atentamente,")], pt=10.5)
    elementos = pt_.resolver()
    # Página 2: anexo escaneado
    q = f.persona()
    h = Hoja(200)
    _encabezado(h, f, "ANEXO N° 1 – COMPROBANTE DE RECEPCIÓN")
    y = 60.0
    h.linea(22, y, [Seg("Recibí conforme de "), s_nombre(q), Seg(",")], pt=11)
    y += 7
    h.linea(22, y, [Seg("RUT "), s_rut(q, "sin_puntos"), Seg(", la documentación indicada.")], pt=11)
    y += 10
    h.linea(22, y, [Seg("Teléfono de contacto: "), s_movil(q, "movil_guiones")], pt=11)
    y += 7
    h.linea(22, y, [Seg("Correo: "), s_correo(q, "subdominio")], pt=11)
    y += 7
    h.linea(22, y, [Seg("Fecha de recepción: "), s_fecha(f.fecha())], pt=11)
    y += 35
    h.raya(30, y, 95, y)
    h.texto(40, y + 5, "Recibido por", pt=9)
    d = Degradacion(inclinacion=0.9, ruido=5, desenfoque=0.6, calidad=72, gris=True)
    elementos += pagina_escaneada(doc, escanear(h, rng, d))[1]
    a = _archivo(
        "mixto_texto_y_escaneo",
        ruta,
        "Página 1 con capa de texto real (oficio); página 2 anexo escaneado sin texto.",
        doc,
        elementos,
        dpi=200,
        degradacion=d.__dict__,
    )
    _guardar(doc, ctx, ruta)
    return a


def _gen_sandwich(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    ruta = "pdf_escaneado/sandwich_ocr.pdf"
    entrega = f.persona()
    recibe = f.persona(dv_valido=False)
    h = Hoja(300)
    _encabezado(h, f, "ACTA DE ENTREGA DE EQUIPAMIENTO", "Programa de Apoyo a Emprendedores Rurales")
    y = 60.0
    h.texto(20, y, "ENTREGA", pt=11, fuente="sans_negrita")
    y += 7
    h.linea(24, y, [Seg("Nombre: "), s_nombre(entrega)], pt=11)
    y += 6.5
    h.linea(24, y, [Seg("RUT: "), s_rut(entrega, "puntos"), Seg("    Correo: "), s_correo(entrega, "punto")], pt=11)
    y += 6.5
    h.linea(24, y, [Seg("Teléfono: "), s_fijo(entrega, rng)], pt=11)
    y += 11
    h.texto(20, y, "RECIBE", pt=11, fuente="sans_negrita")
    y += 7
    h.linea(24, y, [Seg("Nombre: "), s_nombre(recibe)], pt=11)
    y += 6.5
    h.linea(24, y, [Seg("RUT: "), s_rut(recibe, "sin_puntos")], pt=11)
    y += 6.5
    h.linea(24, y, [Seg("Domicilio: "), s_direccion(recibe), Seg(f", {recibe.comuna}")], pt=11)
    y += 6.5
    h.linea(24, y, [Seg("Teléfono: "), s_movil(recibe, "movil_nacional")], pt=11)
    y += 6.5
    h.linea(24, y, [Seg("Correo: "), s_correo(recibe, "con_anio")], pt=11)
    y += 11
    h.texto(20, y, "DETALLE", pt=11, fuente="sans_negrita")
    y += 7
    h.linea(24, y, [Seg("Valor total del equipamiento: "), s_monto(f)], pt=11)
    y += 6.5
    h.linea(24, y, [Seg("Fecha de entrega: "), s_fecha(f.fecha())], pt=11)
    y += 6.5
    for frase in (FRASES_ADMINISTRATIVAS[11], FRASES_ADMINISTRATIVAS[2], FRASES_ADMINISTRATIVAS[9]):
        h.texto(24, y, frase, pt=10)
        y += 6
    y += 30
    h.raya(25, y, 90, y)
    h.raya(120, y, 185, y)
    h.texto(40, y + 5, "Firma entrega", pt=9)
    h.texto(138, y + 5, "Firma recibe", pt=9)
    d = Degradacion(inclinacion=0.8, ruido=5, desenfoque=0.6, calidad=72, gris=True)
    esc = escanear(h, rng, d)
    doc = pymupdf.open()
    page, raster = pagina_escaneada(doc, esc)
    # Capa OCR invisible: cada línea en la posición (transformada) de la línea de la imagen
    sx, sy = A4_PT[0] / esc.ancho, A4_PT[1] / esc.alto
    capa = PaginaTexto(page, capa="oculto", render_mode=3)
    for ln in h.lineas:
        ((x, y_),) = transformar_puntos([[ln.x, ln.y]], esc.matriz)
        capa.linea(
            x * sx,
            y_ * sy,
            [Seg(ln.texto)],
            pt=ln.tam * sy,
            fuente=ln.fuente if ln.fuente in PaginaTexto.ALIAS else "sans",
            angulo=d.inclinacion,
            registrar=False,
        )
    ocultos = []
    for e in raster:
        e.etiquetas["sandwich"] = True
        e2 = copy.deepcopy(e)
        e2.capa = "oculto"
        e2.poligono = _buscar_cerca(page, e.valor or "", _centro(e.poligono))
        e2.etiquetas = {k: v for k, v in e.etiquetas.items() if k not in ("degradacion",)}
        e2.etiquetas["render_mode"] = 3
        ocultos.append(e2)
    a = _archivo(
        "sandwich_ocr",
        ruta,
        "Acta escaneada a 300 ppp con capa OCR invisible (render mode 3) sobre el texto de la imagen; "
        "cada dato se registra dos veces (raster y oculto).",
        doc,
        raster + ocultos,
        dpi=300,
        degradacion=d.__dict__,
    )
    _guardar(doc, ctx, ruta)
    return a


def _gen_fax(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    ruta = "pdf_escaneado/fax_150dpi.pdf"
    p = f.persona()
    oficina = f.persona(en_lista=False)
    h = Hoja(150)
    h.linea(
        8,
        7,
        [
            Seg("DE: OFICINA DE PARTES   FAX: "),
            s_fijo(oficina, rng),
            Seg("   "),
            s_fecha(f.fecha()),
            Seg("   PÁG. 01/01"),
        ],
        pt=8,
        fuente="mono",
    )
    h.raya(8, 9, h.ancho_mm - 8, 9, 0.3)
    h.centrado(30, "TRANSMISIÓN POR FAX", pt=15, fuente="sans_negrita")
    h.centrado(38, "Solicitud de antecedentes para pago de subsidio", pt=11)
    y = 55.0
    h.linea(22, y, [Seg("PARA: "), s_nombre(p, "mayusculas")], pt=11, fuente="mono")
    y += 7
    h.linea(22, y, [Seg("RUT: "), s_rut(p, "comas")], pt=11, fuente="mono")
    y += 7
    h.linea(22, y, [Seg("FONO CONTACTO: "), s_movil(p, "antiguo_movil_09")], pt=11, fuente="mono")
    y += 7
    h.linea(22, y, [Seg("CORREO: "), s_correo(p, "at")], pt=11, fuente="mono")
    y += 7
    h.linea(22, y, [Seg("DIRECCIÓN: "), s_direccion(p)], pt=11, fuente="mono")
    y += 14
    for frase in (FRASES_ADMINISTRATIVAS[4], FRASES_ADMINISTRATIVAS[2], FRASES_ADMINISTRATIVAS[10]):
        h.texto(22, y, frase, pt=10.5, fuente="serif")
        y += 7
    y += 7
    h.linea(22, y, [Seg("Monto del subsidio: "), s_monto(f)], pt=10.5, fuente="serif")
    y += 7
    h.linea(
        22, y, [Seg("Remitente: "), s_nombre(oficina), Seg(" – anexo "), s_fijo(oficina, rng)], pt=10.5, fuente="serif"
    )
    for e in h.lz.elementos:
        if e.nivel == "base":
            e.nivel = "estres"
        e.etiquetas["fax"] = True
    d = Degradacion(inclinacion=0.6, ruido=12, desenfoque=0.7, binarizar=True, sal_pimienta=0.001, papel=False)
    doc = pymupdf.open()
    _, elementos = pagina_escaneada(doc, escanear(h, rng, d))
    a = _archivo(
        "fax_150dpi",
        ruta,
        "Fax a 150 ppp binarizado a 1 bit con tramado; todos los datos en nivel estrés.",
        doc,
        elementos,
        dpi=150,
        degradacion=d.__dict__,
    )
    _guardar(doc, ctx, ruta)
    return a


def _mezclar_timbre(h: Hoja, timbre: Lienzo, x_mm: float, y_mm: float, opacidad: float = 0.78) -> None:
    """Multiplica el timbre (tinta azul sobre blanco) sobre la hoja, con opacidad parcial."""
    px, py = int(round(h.px(x_mm))), int(round(h.px(y_mm)))
    base = np.asarray(h.lz.img, np.float32).copy()
    t = np.asarray(timbre.img, np.float32) / 255.0
    region = base[py : py + timbre.alto, px : px + timbre.ancho]
    region *= 1 - opacidad + opacidad * t[: region.shape[0], : region.shape[1]]
    h.lz.img = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    m = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]], dtype=np.float64)
    for e in transformar_elementos(timbre.elementos, m):
        e.etiquetas["timbre"] = True
        h.lz.elementos.append(e)


def _timbre_redondo(h: Hoja, s: Persona, grados: float) -> Lienzo:
    lado = int(round(h.px(46)))
    t = Lienzo.nuevo(lado, lado)
    d = ImageDraw.Draw(t.img)
    g = max(2, int(h.px(0.7)))
    d.ellipse([g, g, lado - g, lado - g], outline=AZUL_TIMBRE, width=g)
    d.ellipse([h.px(3), h.px(3), lado - h.px(3), lado - h.px(3)], outline=AZUL_TIMBRE, width=max(1, g // 2))
    c = lado / 2
    rad = math.radians(grados)
    perp = (math.sin(rad), math.cos(rad))  # hacia "abajo" del texto girado
    lineas = [
        (Seg("RECEPCIÓN CONFORME"), 9.5, "sans_negrita", -1.25),
        (s_nombre(s), 7.5, "sans_negrita", 0.0),
        (s_rut(s, "puntos", nivel="estres"), 8.5, "sans_negrita", 1.2),
    ]
    radio_interior = c - h.px(3.5)
    for seg, pt, fuente, k in lineas:
        off = h.px(6.2) * k
        tam = h.tam(pt)

        def cuerda(tam_px: int, off: float = off) -> float:
            # ancho disponible dentro del anillo interior a la distancia de la línea más alejada del centro
            d = abs(off) + tam_px * 0.55
            return 2 * math.sqrt(max(0.0, radio_interior**2 - d**2)) * 0.94

        # cada línea debe caber dentro del anillo interior del timbre
        while t.ancho_texto(seg.texto, fuente, tam) > cuerda(tam) and tam > 8:
            tam -= 1
        # tinta azul semitransparente girada 20°: los datos del timbre quedan en estrés
        nivel = "estres" if seg.nivel == "base" and seg.tipo != "texto" else seg.nivel
        t.escribir_rotado(
            c + perp[0] * off,
            c + perp[1] * off,
            seg.texto,
            grados,
            tipo=seg.tipo,
            nombre_fuente=fuente,
            tam=tam,
            color=AZUL_TIMBRE,
            nivel=nivel,
            etiquetas={**seg.etiquetas, "dpi": h.dpi},
        )
    return t


def _timbre_rectangular(h: Hoja, f: Ficticios, grados: float) -> Lienzo:
    w, alto = int(round(h.px(58))), int(round(h.px(30)))
    t = Lienzo.nuevo(w, alto)
    d = ImageDraw.Draw(t.img)
    c = (w / 2, alto / 2)
    rad = math.radians(grados)
    bw, bh = h.px(50) / 2, h.px(19) / 2

    def rot(x: float, y: float) -> tuple[float, float]:
        return (c[0] + x * math.cos(rad) + y * math.sin(rad), c[1] - x * math.sin(rad) + y * math.cos(rad))

    esquinas = [rot(-bw, -bh), rot(bw, -bh), rot(bw, bh), rot(-bw, bh)]
    d.polygon(esquinas, outline=AZUL_TIMBRE, width=max(2, int(h.px(0.6))))
    perp = (math.sin(rad), math.cos(rad))
    for seg, pt, k in ((Seg("OFICINA DE PARTES"), 8, -1.1), (Seg("RECIBIDO"), 12, 0.15), (s_fecha(f.fecha()), 8, 1.25)):
        off = h.px(5.5) * k
        t.escribir_rotado(
            c[0] + perp[0] * off,
            c[1] + perp[1] * off,
            seg.texto,
            grados,
            tipo="texto",
            nombre_fuente="sans_negrita",
            tam=h.tam(pt),
            color=AZUL_TIMBRE,
            etiquetas={**seg.etiquetas, "dpi": h.dpi},
        )
    return t


def _gen_timbre_firma(ctx: Contexto, f: Ficticios, rng: np.random.Generator) -> Archivo:
    ruta = "pdf_escaneado/timbre_y_firma.pdf"
    p = f.persona()
    encargado = f.persona()
    ministro = f.persona()
    nota = f.persona(en_lista=False)
    h = Hoja(300)
    _encabezado(h, f, "ACTA DE RECEPCIÓN CONFORME", "Servicios profesionales a honorarios")
    y = 62.0
    h.texto(20, y, "En virtud del contrato vigente, se deja constancia de la recepción conforme de los", pt=11)
    y += 6.5
    h.texto(20, y, "productos entregados por el prestador que se individualiza:", pt=11)
    y += 11
    h.linea(24, y, [Seg("Prestador: "), s_nombre(p)], pt=11)
    y += 6.5
    h.linea(24, y, [Seg("RUT: "), s_rut(p, "puntos")], pt=11)
    y += 6.5
    h.linea(
        24, y, [Seg("Correo: "), s_correo(p, "punto"), Seg("    Teléfono: "), s_movil(p, "movil_internacional")], pt=11
    )
    y += 6.5
    h.linea(24, y, [Seg("Monto a pagar: "), s_monto(f), Seg("    Período: "), s_fecha(f.fecha())], pt=11)
    y += 11
    for frase in (FRASES_ADMINISTRATIVAS[3], FRASES_ADMINISTRATIVAS[4], FRASES_ADMINISTRATIVAS[10]):
        h.texto(20, y, frase, pt=10.5)
        y += 6.3
    # bloque de firma
    y_linea = 200.0
    firma = h.lz.escribir_irregular(
        h.px(32),
        h.px(y_linea - 3),
        f"{encargado.nombres.split()[0]} {encargado.apellido_p}",
        rng,
        tipo="firma",
        nombre_fuente="stix_cursiva",
        tam=h.tam(24),
        color=AZUL_LAPIZ,
        nivel="fuera_de_alcance",
        etiquetas={"manuscrito": True, "dpi": h.dpi},
    )
    assert firma is not None
    # rasgo final de la firma: una curva que cruza el nombre
    x0f, y0f, x1f, y1f = caja_envolvente(firma.poligono)
    ts = np.linspace(0, 1, 40)
    curva = [
        (x0f - h.px(2) + (x1f - x0f + h.px(8)) * t, y1f - (y1f - y0f) * 0.35 + math.sin(t * math.pi * 2.3) * h.px(2.2))
        for t in ts
    ]
    ImageDraw.Draw(h.lz.img).line(curva, fill=AZUL_LAPIZ, width=max(2, int(h.px(0.45))), joint="curve")
    cx = [c[0] for c in curva]
    cy = [c[1] for c in curva]
    g = h.px(0.3)
    firma.poligono = rect(min(x0f, min(cx) - g), min(y0f, min(cy) - g), max(x1f, max(cx) + g), max(y1f, max(cy) + g))
    firma.etiquetas["rasgo"] = True
    h.raya(25, y_linea, 95, y_linea)
    h.linea(25, y_linea + 5, [s_nombre(encargado)], pt=10)
    h.linea(25, y_linea + 10, [Seg("RUT "), s_rut(encargado, "sin_puntos")], pt=10)
    h.texto(25, y_linea + 15, "Encargado(a) Unidad de Control de Gestión", pt=9)
    # timbres azules semitransparentes (sobre la línea de firma y arriba a la derecha)
    _mezclar_timbre(h, _timbre_redondo(h, ministro, 20.0), 80, y_linea - 30)
    _mezclar_timbre(h, _timbre_rectangular(h, f, -7.0), 147, 30, opacidad=0.7)
    # nota manuscrita en el margen inferior con un teléfono
    yn = 250.0
    e_nota = h.lz.escribir_irregular(
        h.px(118), h.px(yn), "Llamar a", rng, nombre_fuente="stix_cursiva", tam=h.tam(15), color=AZUL_LAPIZ
    )
    assert e_nota is not None
    e_nota.etiquetas["manuscrito"] = True
    x_fin = caja_envolvente(e_nota.poligono)[2] + h.px(2.5)
    formato = "movil_nacional"
    tel = h.lz.escribir_irregular(
        x_fin,
        h.px(yn),
        nota.telefono.formatear(formato),
        rng,
        tipo="telefono",
        nombre_fuente="stix_cursiva",
        tam=h.tam(15),
        color=AZUL_LAPIZ,
        nivel="estres",
        etiquetas={"formato": formato, "clase": "movil", "manuscrito": True, "dpi": h.dpi},
    )
    assert tel is not None
    e_obs = h.lz.escribir_irregular(
        h.px(118),
        h.px(yn + 8),
        "por boleta pendiente",
        rng,
        nombre_fuente="stix_cursiva",
        tam=h.tam(13),
        color=AZUL_LAPIZ,
    )
    assert e_obs is not None
    e_obs.etiquetas["manuscrito"] = True
    d = Degradacion(inclinacion=0.5, ruido=4, desenfoque=0.5, calidad=80)
    doc = pymupdf.open()
    _, elementos = pagina_escaneada(doc, escanear(h, rng, d))
    a = _archivo(
        "timbre_y_firma",
        ruta,
        "Acta escaneada en color con timbre azul semitransparente girado (nombre y RUT), firma manuscrita "
        "y nota manuscrita con un teléfono.",
        doc,
        elementos,
        dpi=300,
        degradacion=d.__dict__,
    )
    _guardar(doc, ctx, ruta)
    return a


def generar(ctx: Contexto) -> list[Archivo]:
    f = ctx.ficticios(MODULO)
    rng = ctx.rng(MODULO)
    return [
        _gen_informe_300(ctx, f, rng),
        _gen_informe_200_torcido(ctx, f, rng),
        _gen_ficha(ctx, f, rng),
        _gen_invertido(ctx, f, rng),
        _gen_mixto(ctx, f, rng),
        _gen_sandwich(ctx, f, rng),
        _gen_fax(ctx, f, rng),
        _gen_timbre_firma(ctx, f, rng),
    ]
