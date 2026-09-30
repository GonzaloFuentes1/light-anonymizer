"""Imágenes sueltas: giradas, con metadatos EXIF/XMP/IPTC/PNG y TIFF (multipágina y 1 bit).

Categorías:

- ``imagen_rotada``: dos diseños (nota impresa fotografiada y tarjeta de presentación) a 0, 90,
  180, 270, 15 y 45 grados, más variantes de estrés (espejado, ruido fuerte, texto diminuto y
  bajo contraste). En los ángulos no rectos el papel se compone sobre una mesa.
- ``exif``: fotos con orientación EXIF (3, 6, 8 y una incorrecta), GPS, autor, miniatura EXIF
  sin censurar, XMP, IPTC, WEBP con EXIF y PNG con bloques de texto y ``eXIf``.
- ``tiff``: TIFF de tres páginas con compresión mixta y etiquetas, y escaneo de 1 bit en Grupo 4.

La verdad de terreno está siempre en la geometría *mostrada* (después de aplicar la orientación
EXIF). HEIC queda pendiente de la decisión de licencia del códec.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageDraw, ImageOps, TiffImagePlugin
from PIL.PngImagePlugin import PngInfo

from banco_pruebas.contexto import Contexto, archivo_imagen, guardar_imagen
from banco_pruebas.esquema import Archivo, Elemento, Metadato, Pagina
from banco_pruebas.ficticios import (
    FORMATOS_CORREO,
    FORMATOS_POR_CLASE,
    FORMATOS_RUT,
    FORMATOS_TELEFONO,
    FRASES_ADMINISTRATIVAS,
    Ficticios,
    Persona,
    sin_tildes,
)
from banco_pruebas.lienzo import (
    Lienzo,
    comprimir_jpeg,
    desenfocar,
    iluminacion,
    ruido,
    sal_pimienta,
    textura_mesa,
    textura_papel,
)
from banco_pruebas.rostros import ProveedorRostros

MODULO = "imagenes"
CAT_ROTADA = "imagen_rotada"
CAT_EXIF = "exif"
CAT_TIFF = "tiff"

ANGULOS = (0, 90, 180, 270, 15, 45)
FORMATOS_SALIDA = ("jpg", "png", "webp")

NEGRITA = {"sans": "sans_negrita", "serif": "serif_negrita", "mono": "mono_negrita", "stix": "stix"}
PAPEL = (248, 246, 240)

# Combinaciones de fuentes de la tarjeta: nombre, cargo, datos, etiquetas.
FAMILIAS_TARJETA = [
    {"nombre": "serif_negrita", "cargo": "serif_cursiva", "datos": "sans", "etiqueta": "sans_negrita"},
    {"nombre": "sans_negrita", "cargo": "sans_oblicua", "datos": "mono", "etiqueta": "sans"},
    {"nombre": "stix", "cargo": "stix_cursiva", "datos": "stix", "etiqueta": "stix_cursiva"},
    {"nombre": "mono_negrita", "cargo": "serif_cursiva", "datos": "serif", "etiqueta": "serif_negrita"},
]
CARGOS = [
    "Profesional de Apoyo, División de Fomento",
    "Encargado(a) de Transparencia",
    "Analista de Presupuesto",
    "Jefatura de Planificación Regional",
    "Asesor(a) Jurídico(a)",
    "Coordinación de Programas Sociales",
]
INSTITUCIONES = [
    "Gobierno Regional Ficticio",
    "Municipalidad de Ejemplo",
    "Servicio Regional Ficticio",
    "Delegación Presidencial Ficticia",
]
ENCABEZADOS_NOTA = ["FICHA DE CONTACTO", "REGISTRO DE ATENCIÓN", "DATOS DEL SOLICITANTE", "HOJA DE VISITA"]

# Coordenadas públicas aproximadas de ciudades de Chile; se les suma un desplazamiento al azar.
CIUDADES_GPS = [
    ("Concepción", -36.827, -73.050),
    ("Temuco", -38.736, -72.590),
    ("Valdivia", -39.814, -73.246),
    ("Talca", -35.427, -71.655),
    ("La Serena", -29.904, -71.249),
    ("Antofagasta", -23.650, -70.400),
    ("Puerto Montt", -41.469, -72.942),
    ("Chillán", -36.606, -72.103),
]

# Orientación EXIF -> transposición que aplica ``ImageOps.exif_transpose`` al mostrar.
TRANSPOSICION_EXIF = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_270, 8: Image.Transpose.ROTATE_90}
# Transposición que hay que aplicar a la imagen mostrada D para obtener los píxeles guardados.
INVERSA_EXIF = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270}


class _Ciclo:
    """Recorre una lista en orden, de forma circular (cubre todos los formatos de manera pareja)."""

    def __init__(self, items: list[str]) -> None:
        self.items = items
        self.i = 0

    def siguiente(self) -> str:
        v = self.items[self.i % len(self.items)]
        self.i += 1
        return v


def _base(formatos: dict[str, str], candidatos: list[str] | None = None) -> list[str]:
    return [f for f in (candidatos or list(formatos)) if formatos[f] == "base"]


def _estres_primero(formatos: dict[str, str], candidatos: list[str] | None = None) -> list[str]:
    """Todos los formatos, con los de nivel ``estres`` al comienzo.

    Las imágenes de estrés son pocas: si el ciclo partiera por los formatos base, los formatos
    raros (RUT con comas, teléfonos antiguos, correo con ``[arroba]``) nunca llegarían a usarse.
    """
    todos = candidatos or list(formatos)
    return [f for f in todos if formatos[f] == "estres"] + [f for f in todos if formatos[f] != "estres"]


@dataclass
class _Estilo:
    tam: int
    fuente: str
    color: tuple[int, int, int] = (28, 28, 34)
    estres: bool = False


class _Generador:
    def __init__(self, ctx: Contexto) -> None:
        self.ctx = ctx
        self.f: Ficticios = ctx.ficticios(MODULO)
        self.rng = ctx.rng(MODULO)
        self._n_personas = 0
        self.c_rut = _Ciclo(_base(FORMATOS_RUT))
        self.c_rut_estres = _Ciclo(_estres_primero(FORMATOS_RUT))
        self.c_correo = _Ciclo(_base(FORMATOS_CORREO))
        self.c_correo_estres = _Ciclo(_estres_primero(FORMATOS_CORREO))
        self.c_tel = {c: _Ciclo(_base(FORMATOS_TELEFONO, fs)) for c, fs in FORMATOS_POR_CLASE.items()}
        self.c_tel_estres = {c: _Ciclo(_estres_primero(FORMATOS_TELEFONO, fs)) for c, fs in FORMATOS_POR_CLASE.items()}
        self.c_familia = 0
        self.archivos: list[Archivo] = []

    # -- datos -----------------------------------------------------------------

    def persona(self, en_lista: bool | None = None) -> Persona:
        """Persona nueva; algunas quedan fuera de la lista, con DV inválido, de 7 dígitos o con K."""
        self._n_personas += 1
        n = self._n_personas
        if en_lista is None:
            en_lista = n % 5 != 3
        p = self.f.persona(dv_valido=n % 6 != 4, en_lista=en_lista, siete_digitos=n % 9 == 7)
        if n % 11 == 5:
            p.rut_cuerpo, p.rut_dv = self.f.rut_con_k()
            p.rut_dv_valido = True
        return p

    def rut(self, p: Persona, estres: bool) -> tuple[str, str, dict[str, Any]]:
        fmt = (self.c_rut_estres if estres else self.c_rut).siguiente()
        nivel = "estres" if estres else FORMATOS_RUT[fmt]
        return p.rut(fmt), nivel, {"formato": fmt, "dv_valido": p.rut_dv_valido}

    def correo(self, p: Persona, estres: bool) -> tuple[str, str, dict[str, Any]]:
        fmt = (self.c_correo_estres if estres else self.c_correo).siguiente()
        nivel = "estres" if estres else FORMATOS_CORREO[fmt]
        return p.correo(fmt), nivel, {"formato": fmt}

    def telefono(self, p: Persona, estres: bool, fijo: bool = False) -> tuple[str, str, dict[str, Any]]:
        tel = p.telefono_fijo if fijo else p.telefono
        fmt = (self.c_tel_estres if estres else self.c_tel)[tel.clase].siguiente()
        nivel = "estres" if estres else FORMATOS_TELEFONO[fmt]
        return tel.formatear(fmt), nivel, {"formato": fmt, "clase": tel.clase}

    @staticmethod
    def nivel_lista(p: Persona, estres: bool) -> str:
        if not p.en_lista:
            return "fuera_de_alcance"
        return "estres" if estres else "base"

    # -- escritura -------------------------------------------------------------

    @staticmethod
    def campo(
        lz: Lienzo,
        x: float,
        y: float,
        etiqueta: str,
        dato: str,
        tipo: str,
        *,
        nivel: str,
        etiquetas: dict[str, Any] | None,
        tam: int,
        fuente_etiqueta: str,
        fuente_dato: str,
        color: tuple[int, int, int],
        nivel_texto: str = "base",
    ) -> float:
        """Escribe ``etiqueta`` (texto neutro) seguida de ``dato`` en la misma línea base."""
        if etiqueta:
            lz.escribir(
                x, y, etiqueta, valor=etiqueta.strip(), nombre_fuente=fuente_etiqueta, tam=tam, color=color,
                nivel=nivel_texto,
            )  # fmt: skip
            x += lz.ancho_texto(etiqueta, fuente_etiqueta, tam)
        lz.escribir(
            x, y, dato, tipo=tipo, nombre_fuente=fuente_dato, tam=tam, color=color, nivel=nivel, etiquetas=etiquetas
        )
        return x + lz.ancho_texto(dato, fuente_dato, tam)

    # -- diseños -----------------------------------------------------------------

    def nota(self, estilo: _Estilo, color_papel: tuple[int, int, int] = PAPEL) -> Lienzo:
        """Diseño (a): hoja impresa con nombre, RUT, correo y teléfono (más una fecha señuelo)."""
        p = self.persona()
        e = estilo.estres
        rut, n_rut, et_rut = self.rut(p, e)
        correo, n_correo, et_correo = self.correo(p, e)
        tel, n_tel, et_tel = self.telefono(p, e)
        nivel_texto = "estres" if e else "base"
        filas = [
            ("Nombre: ", p.nombre_completo, "nombre", self.nivel_lista(p, e),
             {"en_lista": p.en_lista, "variante": "completo"}),
            ("RUT: ", rut, "rut", n_rut, et_rut),
            ("Correo: ", correo, "correo", n_correo, et_correo),
            ("Teléfono: ", tel, "telefono", n_tel, et_tel),
            ("Fecha: ", self.f.fecha(), "texto", nivel_texto, {"senuelo": "fecha"}),
        ]  # fmt: skip
        tam, fnt = estilo.tam, estilo.fuente
        encabezado = ENCABEZADOS_NOTA[int(self.rng.integers(len(ENCABEZADOS_NOTA)))]
        tam_enc = int(round(tam * 1.1))
        medidor = Lienzo.nuevo(8, 8)
        anchos = [medidor.ancho_texto(a + b, fnt, tam) for a, b, *_ in filas]
        anchos.append(medidor.ancho_texto(encabezado, NEGRITA[fnt], tam_enc))
        margen = int(tam * 1.8)
        interlinea = int(tam * 1.7)
        ancho = int(max(anchos)) + 2 * margen
        alto = 2 * margen + tam_enc + int(interlinea * 0.6) + interlinea * len(filas)
        lz = Lienzo(textura_papel(ancho, alto, self.rng, color_papel))
        y = margen + tam_enc
        lz.escribir(
            margen, y, encabezado, nombre_fuente=NEGRITA[fnt], tam=tam_enc, color=estilo.color, nivel=nivel_texto
        )
        y += int(interlinea * 1.4)
        for etq, dato, tipo, nivel, etiquetas in filas:
            self.campo(
                lz, margen, y, etq, dato, tipo, nivel=nivel, etiquetas=etiquetas, tam=tam, fuente_etiqueta=fnt,
                fuente_dato=fnt, color=estilo.color, nivel_texto=nivel_texto,
            )  # fmt: skip
            y += interlinea
        return lz

    def tarjeta(self, estres: bool = False, bajo_contraste: bool = False) -> Lienzo:
        """Diseño (b): tarjeta de presentación con nombre, cargo, correo, móvil, fijo, web y dirección."""
        p = self.persona()
        fam = FAMILIAS_TARJETA[self.c_familia % len(FAMILIAS_TARJETA)]
        self.c_familia += 1
        correo, n_correo, et_correo = self.correo(p, estres)
        movil, n_movil, et_movil = self.telefono(p, estres)
        fijo, n_fijo, et_fijo = self.telefono(p, estres, fijo=True)
        usuario = sin_tildes(p.correo("punto").split("@")[0]).replace(".", "-")
        web = f"www.{p.dominio}/{usuario}"
        nivel_texto = "estres" if estres else "base"
        if bajo_contraste:
            fondo, tinta, banda, tinta_banda = (196, 196, 190), (150, 150, 146), (172, 172, 170), (206, 206, 202)
        else:
            tonos = [(22, 60, 110), (120, 30, 40), (20, 90, 70), (70, 60, 110)]
            banda = tonos[int(self.rng.integers(len(tonos)))]
            fondo, tinta, tinta_banda = (251, 250, 247), (30, 30, 36), (255, 255, 255)
        filas = [
            ("Correo: ", correo, "correo", n_correo, et_correo),
            ("Móvil: ", movil, "telefono", n_movil, et_movil),
            ("Fono: ", fijo, "telefono", n_fijo, et_fijo),
            ("Web: ", web, "url", "estres" if estres else "base", {"formato": "web_personal"}),
            ("Dirección: ", p.direccion, "direccion", self.nivel_lista(p, estres), {"en_lista": p.en_lista}),
        ]
        tam_d, tam_n, tam_c = 27, 46, 26
        medidor = Lienzo.nuevo(8, 8)
        necesario = max(
            medidor.ancho_texto(a, fam["etiqueta"], tam_d)
            + medidor.ancho_texto(b + (f", {p.comuna}" if t == "direccion" else ""), fam["datos"], tam_d)
            for a, b, t, *_ in filas
        )
        necesario = max(necesario, medidor.ancho_texto(p.nombre_completo, fam["nombre"], tam_n))
        ancho, alto = max(1050, int(necesario) + 140), 600
        lz = Lienzo.nuevo(ancho, alto, fondo)
        d = ImageDraw.Draw(lz.img)
        d.rectangle((0, 0, ancho, 96), fill=banda)
        d.rectangle((70, 262, 70 + int(ancho * 0.35), 265), fill=banda)
        inst = INSTITUCIONES[int(self.rng.integers(len(INSTITUCIONES)))]
        lz.escribir(70, 60, inst, nombre_fuente="sans_negrita", tam=28, color=tinta_banda, nivel=nivel_texto)
        lz.escribir(
            70, 190, p.nombre_completo, tipo="nombre", nombre_fuente=fam["nombre"], tam=tam_n, color=tinta,
            nivel=self.nivel_lista(p, estres), etiquetas={"en_lista": p.en_lista, "variante": "completo"},
        )  # fmt: skip
        cargo = CARGOS[int(self.rng.integers(len(CARGOS)))]
        lz.escribir(70, 240, cargo, nombre_fuente=fam["cargo"], tam=tam_c, color=tinta, nivel=nivel_texto)
        y = 322
        for etq, dato, tipo, nivel, etiquetas in filas:
            x = self.campo(
                lz, 70, y, etq, dato, tipo, nivel=nivel, etiquetas=etiquetas, tam=tam_d,
                fuente_etiqueta=fam["etiqueta"], fuente_dato=fam["datos"], color=tinta, nivel_texto=nivel_texto,
            )  # fmt: skip
            if tipo == "direccion":
                lz.escribir(
                    x, y, f", {p.comuna}", nombre_fuente=fam["datos"], tam=tam_d, color=tinta, nivel=nivel_texto,
                )  # fmt: skip
            y += 44
        return lz

    def foto_rostro(self, pose: str = "frente", leyenda: bool = False) -> Lienzo:
        """Foto de una persona sobre un fondo liso con degradado (opcionalmente con su nombre al pie)."""
        (r,) = self.ctx.rostros.tomar(pose, 1)
        ancho = int(self.rng.uniform(420, 540))
        alto_r = int(round(r.img.height * ancho / r.img.width))
        mx, my = int(self.rng.uniform(60, 140)), int(self.rng.uniform(50, 110))
        extra = 90 if leyenda else 0
        p = self.persona() if leyenda else None
        etq_leyenda = "Credencial: "
        w = ancho + 2 * mx
        if p is not None:
            medidor = Lienzo.nuevo(8, 8)
            largo = medidor.ancho_texto(etq_leyenda, "sans", 30) + medidor.ancho_texto(
                p.nombre_completo, "sans_negrita", 30
            )
            w = max(w, int(largo) + 2 * mx)
        h = alto_r + 2 * my + extra
        c0 = self.rng.uniform(150, 220, 3)
        c1 = c0 * self.rng.uniform(0.7, 0.95)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        fondo = (c0 * (1 - t) + c1 * t) * np.ones((h, w, 3), np.float32)
        lz = Lienzo(Image.fromarray(fondo.clip(0, 255).astype(np.uint8)))
        lz.pegar(r.img, (w - ancho) // 2, my, ancho=ancho, elementos=ProveedorRostros.elementos(r, "base"))
        if p is not None:
            self.campo(
                lz, mx, my + alto_r + 60, etq_leyenda, p.nombre_completo, "nombre", nivel=self.nivel_lista(p, False),
                etiquetas={"en_lista": p.en_lista, "variante": "completo"}, tam=30, fuente_etiqueta="sans",
                fuente_dato="sans_negrita", color=(20, 20, 20),
            )  # fmt: skip
        return lz

    def captura(self) -> Lienzo:
        """Pantallazo de un sistema de gestión con la ficha de una persona."""
        p = self.persona()
        rut, n_rut, et_rut = self.rut(p, False)
        correo, n_correo, et_correo = self.correo(p, False)
        movil, n_movil, et_movil = self.telefono(p, False)
        fijo, n_fijo, et_fijo = self.telefono(p, False, fijo=True)
        w, h = 1280, 780
        lz = Lienzo.nuevo(w, h, (236, 239, 243))
        d = ImageDraw.Draw(lz.img)
        d.rectangle((0, 0, w, 46), fill=(44, 62, 80))
        lz.escribir(24, 30, "Sistema de Gestión Documental - Ficha de persona", tam=19, color=(250, 250, 250))
        d.rectangle((40, 80, w - 40, h - 40), fill=(255, 255, 255), outline=(200, 205, 212))
        filas = [
            ("Nombre", p.nombre_completo, "nombre", self.nivel_lista(p, False),
             {"en_lista": p.en_lista, "variante": "completo"}),
            ("RUT", rut, "rut", n_rut, et_rut),
            ("Correo electrónico", correo, "correo", n_correo, et_correo),
            ("Teléfono móvil", movil, "telefono", n_movil, et_movil),
            ("Teléfono fijo", fijo, "telefono", n_fijo, et_fijo),
            ("Domicilio", p.direccion, "direccion", self.nivel_lista(p, False), {"en_lista": p.en_lista}),
            ("Fecha de ingreso", self.f.fecha(), "texto", "base", {"senuelo": "fecha"}),
            ("Monto asignado", self.f.monto(), "texto", "base", {"senuelo": "monto"}),
        ]  # fmt: skip
        y = 150
        for etq, dato, tipo, nivel, etiquetas in filas:
            lz.escribir(80, y, etq, tam=20, color=(90, 96, 110))
            d.rectangle((330, y - 30, w - 90, y + 14), fill=(250, 251, 253), outline=(180, 186, 196))
            lz.escribir(344, y, dato, tipo=tipo, tam=21, color=(20, 24, 30), nivel=nivel, etiquetas=etiquetas)
            y += 72
        return lz

    # -- composición ----------------------------------------------------------------

    def girar(self, lz: Lienzo, grados: float, fondo: tuple[int, int, int], margen: int = 40) -> Lienzo:
        """Gira el lienzo; en ángulos no rectos lo compone sobre una mesa (la geometría sigue exacta)."""
        g = grados % 360
        if g in (0, 90, 180, 270):
            return lz.rotar(g)
        rot = lz.rotar(g, fondo=fondo)
        mascara = Lienzo(Image.new("RGB", lz.img.size, (255, 255, 255))).rotar(g, fondo=(0, 0, 0)).img.convert("L")
        final = Lienzo(textura_mesa(rot.ancho + 2 * margen, rot.alto + 2 * margen, self.rng))
        final.pegar(rot.img, margen, margen, elementos=rot.elementos, mascara=mascara)
        return final

    def foto_documento(self, estilo: _Estilo | None = None) -> Lienzo:
        """Nota fotografiada sobre la mesa con un giro leve e iluminación de celular."""
        estilo = estilo or _Estilo(tam=int(self.rng.integers(28, 37)), fuente="sans")
        lz = self.nota(estilo)
        grados = float(self.rng.uniform(2, 5)) * (1 if self.rng.random() < 0.5 else -1)
        foto = self.girar(lz, grados, PAPEL, margen=60)
        foto.img = iluminacion(foto.img, self.rng, 0.25)
        return foto

    def degradar_leve(self, img: Image.Image, formato: str) -> Image.Image:
        img = ruido(img, self.rng, 2.5)
        return img if formato == "jpg" else comprimir_jpeg(img, 85)

    def guardar(self, img: Image.Image, relativa: str, formato: str, **opciones: Any) -> None:
        if formato == "jpg":
            opciones.setdefault("quality", 85)
        elif formato == "webp":
            opciones.setdefault("quality", 90)
        guardar_imagen(img, self.ctx.ruta(relativa), formato, **opciones)

    # -- categoría imagen_rotada ---------------------------------------------------------

    def rotadas(self) -> None:
        fuentes_nota = ["sans", "serif", "mono", "stix"]
        i = 0
        for desfase, diseno in enumerate(("nota", "tarjeta")):
            for angulo in ANGULOS:
                # El desfase evita que un ángulo quede siempre con el mismo formato en ambos diseños.
                formato = FORMATOS_SALIDA[(i + desfase) % len(FORMATOS_SALIDA)]
                if diseno == "nota":
                    estilo = _Estilo(tam=int(self.rng.integers(28, 41)), fuente=fuentes_nota[i % len(fuentes_nota)])
                    lz = self.girar(self.nota(estilo), angulo, PAPEL)
                    lz.img = iluminacion(lz.img, self.rng, 0.22)
                    degr = ["iluminacion", "ruido_leve", "jpeg_85"]
                else:
                    lz = self.girar(self.tarjeta(), angulo, (251, 250, 247))
                    degr = ["ruido_leve", "jpeg_85"]
                img = self.degradar_leve(lz.img, formato)
                self._agregar_rotada(lz, img, diseno, str(angulo), angulo, formato, degr, estres=False)
                i += 1
        self._rotadas_estres()

    def _rotadas_estres(self) -> None:
        # Espejado horizontal a 0 grados (nota) y a 90 grados (tarjeta).
        lz = self.nota(_Estilo(tam=34, fuente="serif", estres=True)).espejar()
        lz.img = iluminacion(lz.img, self.rng, 0.2)
        self._agregar_rotada(
            lz, self.degradar_leve(lz.img, "jpg"), "nota", "espejo_0", 0, "jpg", ["espejo", "iluminacion"], True
        )
        lz = self.tarjeta(estres=True).rotar(90).espejar()
        self._agregar_rotada(lz, self.degradar_leve(lz.img, "png"), "tarjeta", "espejo_90", 90, "png", ["espejo"], True)
        # 45 grados con ruido fuerte y desenfoque.
        lz = self.girar(self.nota(_Estilo(tam=32, fuente="sans", estres=True)), 45, PAPEL)
        lz.img = iluminacion(lz.img, self.rng, 0.3)
        img = comprimir_jpeg(desenfocar(ruido(lz.img, self.rng, 24), 1.6), 60)
        self._agregar_rotada(
            lz, img, "nota", "45_ruido", 45, "jpg", ["iluminacion", "ruido_fuerte", "desenfoque", "jpeg_60"], True
        )
        # Texto diminuto (~12 px) a 15 grados.
        lz = self.girar(self.nota(_Estilo(tam=12, fuente="sans", estres=True)), 15, PAPEL, margen=24)
        self._agregar_rotada(
            lz, self.degradar_leve(lz.img, "png"), "nota", "diminuta_15", 15, "png", ["texto_diminuto"], True
        )
        # Bajo contraste.
        lz = self.tarjeta(estres=True, bajo_contraste=True)
        self._agregar_rotada(
            lz, self.degradar_leve(lz.img, "jpg"), "tarjeta", "bajo_contraste_0", 0, "jpg", ["bajo_contraste"], True
        )

    def _agregar_rotada(
        self,
        lz: Lienzo,
        img: Image.Image,
        diseno: str,
        sufijo: str,
        angulo: int,
        formato: str,
        degradacion: list[str],
        estres: bool,
    ) -> None:
        ruta = f"imagenes_rotadas/{diseno}_{sufijo}.{formato}"
        self.guardar(img, ruta, formato)
        espejo = "espejo" in degradacion
        for e in lz.elementos:
            e.etiquetas["degradacion"] = list(degradacion)
        descripcion = (
            f"{'Nota fotografiada' if diseno == 'nota' else 'Tarjeta de presentación'} girada {angulo} grados"
            + (" y espejada" if espejo else "")
            + (f" ({', '.join(degradacion)})" if estres else "")
        )
        self.archivos.append(
            archivo_imagen(
                id=f"img_rot_{diseno}_{sufijo}",
                ruta=ruta,
                formato=formato,
                categoria=CAT_ROTADA,
                descripcion=descripcion,
                lienzo=lz,
                etiquetas={
                    "diseno": diseno,
                    "angulo": angulo,
                    "espejo": espejo,
                    "degradacion": degradacion,
                    "estres": estres,
                },
            )  # fmt: skip
        )

    # -- categoría exif ------------------------------------------------------------------

    def gps(self) -> tuple[dict[int, Any], dict[str, Any]]:
        ciudad, lat0, lon0 = CIUDADES_GPS[int(self.rng.integers(len(CIUDADES_GPS)))]
        lat = round(lat0 + float(self.rng.uniform(-0.05, 0.05)), 6)
        lon = round(lon0 + float(self.rng.uniform(-0.05, 0.05)), 6)
        alt = round(float(self.rng.uniform(5, 400)), 1)
        ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: b"S",
            piexif.GPSIFD.GPSLatitude: _gms(lat),
            piexif.GPSIFD.GPSLongitudeRef: b"W",
            piexif.GPSIFD.GPSLongitude: _gms(lon),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (int(round(alt * 10)), 10),
            piexif.GPSIFD.GPSDateStamp: b"2026:05:12",
        }
        return ifd, {"lat": lat, "lon": lon, "alt": alt, "zona": ciudad, "ficticio": True}

    def meta_gps(self, gps_et: dict[str, Any], donde: str = "exif.gps") -> Metadato:
        return Metadato(donde=donde, valor=None, etiquetas=dict(gps_et))

    def exif(self) -> None:
        self._orientacion_6()
        for orientacion in (3, 8):
            self._orientacion_simple(orientacion)
        self._orientacion_incorrecta()
        self._webp_gps()
        self._png_metadatos()
        self._rostro_gps()
        self._iptc()

    def _guardar_orientada(self, d: Lienzo, orientacion: int, ruta: str, exif: bytes, **opciones: Any) -> None:
        """Guarda los píxeles tal que, al aplicar la orientación EXIF, se vea exactamente ``d``."""
        guardada = d.img.transpose(INVERSA_EXIF[orientacion])
        self.guardar(guardada, ruta, "jpg", exif=exif, quality=92, **opciones)
        with Image.open(self.ctx.ruta(ruta)) as leida:
            vista = ImageOps.exif_transpose(leida).convert("RGB")
        if vista.size != d.img.size:
            raise RuntimeError(f"{ruta}: la orientación EXIF no reproduce la geometría mostrada")
        dif = np.abs(np.asarray(vista, np.int16) - np.asarray(d.img, np.int16)).mean()
        if dif > 4:
            raise RuntimeError(f"{ruta}: la imagen mostrada difiere de la esperada (dif media {dif:.1f})")

    def _orientacion_6(self) -> None:
        d = self.foto_documento()
        ruta = "exif/foto_orientacion_6.jpg"
        autor, autor_xp, creador, dueno = self.persona(), self.persona(), self.persona(), self.persona()
        p_desc, p_com = self.persona(), self.persona()
        rut_desc = p_desc.rut("puntos")
        correo = p_com.correo("punto")
        gps_ifd, gps_et = self.gps()
        guardada = d.img.transpose(INVERSA_EXIF[6])
        miniatura = guardada.copy()
        miniatura.thumbnail((160, 160))
        buf = io.BytesIO()
        miniatura.save(buf, "JPEG", quality=75)
        datos = {
            "0th": {
                piexif.ImageIFD.Orientation: 6,
                piexif.ImageIFD.Make: b"Ficticia",
                piexif.ImageIFD.Model: b"FotoFono FX-200",
                piexif.ImageIFD.Software: b"Camara 4.2",
                piexif.ImageIFD.DateTime: b"2026:05:12 10:31:07",
                piexif.ImageIFD.Artist: autor.nombre_completo.encode("utf-8"),
                piexif.ImageIFD.ImageDescription: f"Foto RUT {rut_desc}".encode(),
                piexif.ImageIFD.XPAuthor: tuple(autor_xp.nombre_completo.encode("utf-16-le") + b"\x00\x00"),
                piexif.ImageIFD.Copyright: f"(c) 2026 {dueno.nombre_completo}".encode(),
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: b"2026:05:12 10:31:07",
                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(correo, encoding="ascii"),
            },
            "GPS": gps_ifd,
            # IFD1 de una miniatura JPEG: Compression=6 (piexif agrega el desplazamiento y el largo).
            "1st": {
                piexif.ImageIFD.Compression: 6,
                piexif.ImageIFD.XResolution: (72, 1),
                piexif.ImageIFD.YResolution: (72, 1),
                piexif.ImageIFD.ResolutionUnit: 2,
            },
            "thumbnail": buf.getvalue(),
        }
        xmp = _xmp(creador.nombre_completo)
        self._guardar_orientada(d, 6, ruta, piexif.dump(datos), xmp=xmp)
        metadatos = [
            self.meta_gps(gps_et),
            Metadato("exif.artist", autor.nombre_completo, {"tipo": "nombre", "tag": 315}),
            Metadato("exif.image_description", rut_desc, {"tipo": "rut", "texto": f"Foto RUT {rut_desc}", "tag": 270}),
            Metadato("exif.user_comment", correo, {"tipo": "correo", "tag": 37510, "codificacion": "ascii"}),
            Metadato("exif.xp_author", autor_xp.nombre_completo, {"tipo": "nombre", "tag": 40093,
                                                                  "codificacion": "utf-16-le"}),
            Metadato("exif.copyright", dueno.nombre_completo, {"tipo": "nombre", "tag": 33432}),
            Metadato("exif.miniatura", None, {"ancho": miniatura.width, "alto": miniatura.height,
                                              "sin_censurar": True}),
            Metadato("xmp.dc:creator", creador.nombre_completo, {"tipo": "nombre", "contenedor": "jpeg.app1"}),
        ]  # fmt: skip
        _marcar_degradacion(d.elementos, ["iluminacion", "jpeg_92"])
        self.archivos.append(
            archivo_imagen(
                id="img_exif_orientacion_6",
                ruta=ruta,
                formato="jpg",
                categoria=CAT_EXIF,
                descripcion="Foto de documento guardada girada con EXIF Orientation=6; EXIF completo, GPS, "
                "miniatura sin censurar y XMP",
                lienzo=d,
                metadatos=metadatos,
                etiquetas={"orientacion_exif": 6, "tam_guardado": list(guardada.size)},
            )
        )

    def _orientacion_simple(self, orientacion: int) -> None:
        d = self.foto_documento()
        ruta = f"exif/foto_orientacion_{orientacion}.jpg"
        gps_ifd, gps_et = self.gps()
        datos = {
            "0th": {
                piexif.ImageIFD.Orientation: orientacion,
                piexif.ImageIFD.Make: b"Ficticia",
                piexif.ImageIFD.Model: b"FotoFono FX-100",
            },
            "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:06:03 16:02:44"},
            "GPS": gps_ifd,
        }
        self._guardar_orientada(d, orientacion, ruta, piexif.dump(datos))
        guardada = d.img.transpose(INVERSA_EXIF[orientacion])
        _marcar_degradacion(d.elementos, ["iluminacion", "jpeg_92"])
        self.archivos.append(
            archivo_imagen(
                id=f"img_exif_orientacion_{orientacion}",
                ruta=ruta,
                formato="jpg",
                categoria=CAT_EXIF,
                descripcion=f"Foto de documento con EXIF Orientation={orientacion} y GPS",
                lienzo=d,
                metadatos=[self.meta_gps(gps_et)],
                etiquetas={"orientacion_exif": orientacion, "tam_guardado": list(guardada.size)},
            )
        )

    def _orientacion_incorrecta(self) -> None:
        """Píxeles derechos pero Orientation=6: al mostrarse (y en la salida) el texto queda de lado."""
        u = self.foto_documento()
        ruta = "exif/foto_orientacion_incorrecta.jpg"
        datos = {
            "0th": {piexif.ImageIFD.Orientation: 6, piexif.ImageIFD.Make: b"Ficticia"},
            "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:04:22 09:15:00"},
        }
        self.guardar(u.img, ruta, "jpg", exif=piexif.dump(datos), quality=92)
        d = u.rotar(270)  # misma transposición que aplica exif_transpose para Orientation=6
        for e in d.elementos:
            e.etiquetas["exif_incorrecto"] = True
        _marcar_degradacion(d.elementos, ["iluminacion", "jpeg_92"])
        self.archivos.append(
            archivo_imagen(
                id="img_exif_orientacion_incorrecta",
                ruta=ruta,
                formato="jpg",
                categoria=CAT_EXIF,
                descripcion="Foto derecha con EXIF Orientation=6 erróneo: al aplicar EXIF el texto queda de lado",
                lienzo=d,
                etiquetas={"orientacion_exif": 6, "exif_incorrecto": True, "tam_guardado": list(u.img.size)},
            )
        )

    def _webp_gps(self) -> None:
        lz = self.girar(self.tarjeta(), float(self.rng.uniform(-6, 6)), (251, 250, 247), margen=50)
        lz.img = iluminacion(lz.img, self.rng, 0.2)
        ruta = "exif/foto_gps.webp"
        autor, creador = self.persona(), self.persona()
        gps_ifd, gps_et = self.gps()
        datos = {
            "0th": {piexif.ImageIFD.Artist: autor.nombre_completo.encode("utf-8"), piexif.ImageIFD.Make: b"Ficticia"},
            "GPS": gps_ifd,
        }
        self.guardar(
            ruido(lz.img, self.rng, 2.5), ruta, "webp", exif=piexif.dump(datos), xmp=_xmp(creador.nombre_completo)
        )
        metadatos = [
            self.meta_gps(gps_et),
            Metadato("exif.artist", autor.nombre_completo, {"tipo": "nombre", "tag": 315, "contenedor": "webp.exif"}),
            Metadato("xmp.dc:creator", creador.nombre_completo, {"tipo": "nombre", "contenedor": "webp.xmp"}),
        ]
        _marcar_degradacion(lz.elementos, ["iluminacion", "ruido_leve", "webp_90"])
        self.archivos.append(
            archivo_imagen(
                id="img_exif_webp_gps",
                ruta=ruta,
                formato="webp",
                categoria=CAT_EXIF,
                descripcion="Foto de tarjeta en WEBP con EXIF (GPS, Artist) y XMP",
                lienzo=lz,
                metadatos=metadatos,
            )
        )

    def _png_metadatos(self) -> None:
        lz = self.captura()
        ruta = "exif/captura_metadatos.png"
        autor, p_com, p_desc, creador = self.persona(), self.persona(), self.persona(), self.persona()
        correo = p_com.correo("punto")
        rut = p_desc.rut("sin_puntos")
        gps_ifd, gps_et = self.gps()
        info = PngInfo()
        info.add_text("Author", autor.nombre_completo)
        info.add_text("Comment", correo)
        info.add_text("Description", f"Captura ficha RUT {rut}")
        info.add_itxt("XML:com.adobe.xmp", _xmp(creador.nombre_completo).decode("utf-8"))
        self.guardar(lz.img, ruta, "png", pnginfo=info, exif=piexif.dump({"GPS": gps_ifd}))
        metadatos = [
            Metadato("png.texto.Author", autor.nombre_completo, {"tipo": "nombre", "chunk": "tEXt"}),
            Metadato("png.texto.Comment", correo, {"tipo": "correo", "chunk": "tEXt"}),
            Metadato("png.texto.Description", rut, {"tipo": "rut", "chunk": "tEXt",
                                                    "texto": f"Captura ficha RUT {rut}"}),
            Metadato("xmp.dc:creator", creador.nombre_completo, {"tipo": "nombre", "contenedor": "png.iTXt"}),
            self.meta_gps(gps_et, "png.exif.gps"),
        ]  # fmt: skip
        _marcar_degradacion(lz.elementos, [])  # pantallazo limpio, sin pérdida
        self.archivos.append(
            archivo_imagen(
                id="img_exif_png_metadatos",
                ruta=ruta,
                formato="png",
                categoria=CAT_EXIF,
                descripcion="Pantallazo PNG con tEXt (Author, Comment, Description), iTXt XMP y eXIf con GPS",
                lienzo=lz,
                metadatos=metadatos,
            )
        )

    def _rostro_gps(self) -> None:
        lz = self.foto_rostro("frente")
        ruta = "exif/foto_rostro_gps.jpg"
        autor = self.persona()
        gps_ifd, gps_et = self.gps()
        datos = {
            "0th": {
                piexif.ImageIFD.Make: b"Ficticia",
                piexif.ImageIFD.Model: b"FotoFono FX-300",
                piexif.ImageIFD.Artist: autor.nombre_completo.encode("utf-8"),
            },
            "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:03:18 12:40:10"},
            "GPS": gps_ifd,
        }
        self.guardar(ruido(lz.img, self.rng, 2.0), ruta, "jpg", exif=piexif.dump(datos), quality=90)
        _marcar_degradacion(lz.elementos, ["ruido_leve", "jpeg_90"])
        self.archivos.append(
            archivo_imagen(
                id="img_exif_rostro_gps",
                ruta=ruta,
                formato="jpg",
                categoria=CAT_EXIF,
                descripcion="Foto de una persona con GPS y Artist en EXIF",
                lienzo=lz,
                metadatos=[
                    self.meta_gps(gps_et),
                    Metadato("exif.artist", autor.nombre_completo, {"tipo": "nombre", "tag": 315}),
                ],
            )
        )

    def _iptc(self) -> None:
        d = self.foto_documento()
        ruta = "exif/foto_iptc.jpg"
        autor, p_leyenda = self.persona(), self.persona()
        rut = p_leyenda.rut("puntos")
        leyenda = f"Documento de {p_leyenda.nombre_completo}, RUT {rut}"
        buf = io.BytesIO()
        Image.frombytes("RGB", d.img.size, d.img.tobytes()).save(buf, "JPEG", quality=88)
        segmento = _app13_iptc(
            [
                (1, 90, b"\x1b%G"),  # CodedCharacterSet = UTF-8
                (2, 0, b"\x00\x04"),  # RecordVersion
                (2, 80, autor.nombre_completo.encode("utf-8")),  # By-line
                (2, 120, leyenda.encode("utf-8")),  # Caption/Abstract
            ]
        )
        self.ctx.ruta(ruta).write_bytes(_insertar_segmento_jpeg(buf.getvalue(), segmento))
        _marcar_degradacion(d.elementos, ["iluminacion", "jpeg_88"])
        self.archivos.append(
            archivo_imagen(
                id="img_exif_iptc",
                ruta=ruta,
                formato="jpg",
                categoria=CAT_EXIF,
                descripcion="Foto de documento con bloque IPTC (APP13) con By-line y Caption",
                lienzo=d,
                metadatos=[
                    Metadato("iptc.byline", autor.nombre_completo, {"tipo": "nombre", "dataset": "2:80"}),
                    Metadato("iptc.caption", rut, {"tipo": "rut", "dataset": "2:120", "texto": leyenda}),
                    Metadato("iptc.caption", p_leyenda.nombre_completo, {"tipo": "nombre", "dataset": "2:120"}),
                ],
            )
        )

    # -- categoría tiff --------------------------------------------------------------------

    def tiff(self) -> None:
        self._tiff_multipagina()
        self._tiff_escaneo_bn()

    def _tiff_multipagina(self) -> None:
        paginas = [
            self.nota(_Estilo(tam=32, fuente="serif")),
            self.tarjeta().rotar(90),
            self.foto_rostro("tres_cuartos", leyenda=True),
        ]
        compresiones = ["tiff_lzw", "tiff_adobe_deflate", "jpeg"]
        ruta = "tiff/multipagina.tif"
        autor, p_desc, p_doc, p_pag = self.persona(), self.persona(), self.persona(), self.persona()
        rut = p_desc.rut("puntos")
        correo = p_doc.correo("guion_bajo")
        ifd = TiffImagePlugin.ImageFileDirectory_v2()
        ifd[315] = autor.nombre_completo.encode("utf-8")  # Artist (bytes para conservar UTF-8)
        ifd[270] = f"Expediente RUT {rut}".encode()  # ImageDescription
        ifd[305] = "Escaner Ficticio 3.1"  # Software
        ifd[269] = f"Respaldo {correo}".encode()  # DocumentName
        ifd[285] = f"Ficha de {p_pag.nombre_completo}".encode()  # PageName
        destino = self.ctx.ruta(ruta)
        # Cada página se codifica primero a un archivo real: si libtiff escribe a memoria (como pasa al
        # guardar directo en AppendingTiffWriter) deja bytes de relleno sin inicializar y el archivo
        # deja de ser determinista.
        temporal = destino.with_name(destino.name + ".pagina.tmp")
        with TiffImagePlugin.AppendingTiffWriter(destino, new=True) as tf:
            for i, (lz, comp) in enumerate(zip(paginas, compresiones, strict=True)):
                opciones: dict[str, Any] = {"compression": comp, "dpi": (200, 200)}
                if i == 0:
                    opciones["tiffinfo"] = ifd
                if comp == "jpeg":
                    opciones["quality"] = 88
                guardar_imagen(lz.img, temporal, "tiff", **opciones)
                tf.write(temporal.read_bytes())
                tf.newFrame()
        temporal.unlink()
        elementos: list[Elemento] = []
        for i, lz in enumerate(paginas):
            _marcar_degradacion(lz.elementos, [] if compresiones[i].startswith("tiff_") else ["jpeg_88"])
            for e in lz.elementos:
                e.pagina = i
                elementos.append(e)
        self.archivos.append(
            Archivo(
                id="img_tiff_multipagina",
                ruta=ruta,
                formato="tiff",
                categoria=CAT_TIFF,
                descripcion="TIFF de 3 páginas de distinto tamaño (nota, tarjeta girada 90, foto con rostro), "
                "compresión LZW/Deflate/JPEG y etiquetas con datos en la página 0",
                paginas=[Pagina(indice=i, ancho=lz.ancho, alto=lz.alto, unidad="px") for i, lz in enumerate(paginas)],
                elementos=elementos,
                metadatos_sensibles=[
                    Metadato("tiff.artist", autor.nombre_completo, {"tipo": "nombre", "tag": 315, "pagina": 0}),
                    Metadato("tiff.image_description", rut, {"tipo": "rut", "tag": 270, "pagina": 0}),
                    Metadato("tiff.document_name", correo, {"tipo": "correo", "tag": 269, "pagina": 0}),
                    Metadato("tiff.page_name", p_pag.nombre_completo, {"tipo": "nombre", "tag": 285, "pagina": 0}),
                ],
                etiquetas={"compresiones": compresiones},
            )
        )

    def _tiff_escaneo_bn(self) -> None:
        """Página carta a 200 ppp, binarizada a 1 bit y comprimida en CCITT Grupo 4."""
        w, h = 1700, 2200
        lz = Lienzo.nuevo(w, h)
        margen, tam, inter = 150, 30, 50
        tinta = (15, 15, 15)
        y = 230
        lz.escribir(margen, y, "GOBIERNO REGIONAL FICTICIO", nombre_fuente="serif_negrita", tam=42, color=tinta)
        y += 70
        lz.escribir(margen, y, f"ACTA DE ENTREGA N° {self.f.folio()}", nombre_fuente="serif", tam=34, color=tinta)
        y += 90
        frases = list(FRASES_ADMINISTRATIVAS)
        orden = self.rng.permutation(len(frases))
        for linea in _ajustar(lz, " ".join(frases[i] for i in orden[:3]), "serif", tam, w - 2 * margen):
            lz.escribir(margen, y, linea, nombre_fuente="serif", tam=tam, color=tinta)
            y += inter
        y += 30
        for rol in ("Entrega", "Recibe"):
            p = self.persona()
            rut, n_rut, et_rut = self.rut(p, False)
            correo, n_correo, et_correo = self.correo(p, False)
            tel, n_tel, et_tel = self.telefono(p, False, fijo=rol == "Recibe")
            lz.escribir(margen, y, f"{rol}:", nombre_fuente="serif_negrita", tam=tam, color=tinta)
            y += inter
            filas = [
                ("Nombre: ", p.nombre_completo, "nombre", self.nivel_lista(p, False),
                 {"en_lista": p.en_lista, "variante": "completo"}),
                ("RUT: ", rut, "rut", n_rut, et_rut),
                ("Domicilio: ", p.direccion, "direccion", self.nivel_lista(p, False), {"en_lista": p.en_lista}),
                ("Teléfono: ", tel, "telefono", n_tel, et_tel),
                ("Correo: ", correo, "correo", n_correo, et_correo),
            ]  # fmt: skip
            for etq, dato, tipo, nivel, etiquetas in filas:
                self.campo(
                    lz, margen + 40, y, etq, dato, tipo, nivel=nivel, etiquetas=etiquetas, tam=tam,
                    fuente_etiqueta="serif", fuente_dato="serif", color=tinta,
                )  # fmt: skip
                y += inter
            y += 30
        for etq, dato, senuelo in (("Fecha: ", self.f.fecha(), "fecha"), ("Monto: ", self.f.monto(), "monto")):
            self.campo(
                lz, margen, y, etq, dato, "texto", nivel="base", etiquetas={"senuelo": senuelo}, tam=tam,
                fuente_etiqueta="serif", fuente_dato="serif", color=tinta,
            )  # fmt: skip
            y += inter
        y += 20
        for linea in _ajustar(lz, " ".join(frases[i] for i in orden[3:5]), "serif", tam, w - 2 * margen):
            lz.escribir(margen, y, linea, nombre_fuente="serif", tam=tam, color=tinta)
            y += inter
        lz = lz.inclinar_escaneo(float(self.rng.uniform(-0.8, 0.8)), fondo=(255, 255, 255))
        gris = sal_pimienta(lz.img, self.rng, 0.0004).convert("L")
        bn = gris.point(lambda v: 255 if v > 140 else 0).convert("1", dither=Image.Dither.NONE)
        ruta = "tiff/escaneo_bn.tif"
        guardar_imagen(bn, self.ctx.ruta(ruta), "tiff", compression="group4", dpi=(200, 200))
        _marcar_degradacion(lz.elementos, ["inclinacion", "sal_pimienta", "binarizado_1bit"])
        self.archivos.append(
            archivo_imagen(
                id="img_tiff_escaneo_bn",
                ruta=ruta,
                formato="tiff",
                categoria=CAT_TIFF,
                descripcion="Escaneo de 1 bit a 200 ppp con compresión CCITT Grupo 4 (acta con datos de dos personas)",
                lienzo=lz,
                etiquetas={"dpi": 200, "modo": "1", "compresion": "group4"},
            )
        )


# ---------------------------------------------------------------------------
# Utilidades de metadatos
# ---------------------------------------------------------------------------


def _marcar_degradacion(elementos: list[Elemento], degradacion: list[str]) -> None:
    """Anota la degradación en cada elemento (para el desglose de recall por degradación)."""
    for e in elementos:
        e.etiquetas.setdefault("degradacion", list(degradacion))


def _gms(valor: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Grados decimales -> (grados, minutos, segundos) como racionales EXIF (sin signo)."""
    v = abs(valor)
    g = int(v)
    m = int((v - g) * 60)
    s = (v - g - m / 60) * 3600
    return ((g, 1), (m, 1), (int(round(s * 1000)), 1000))


def _xmp(creador: str) -> bytes:
    paquete = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f"   <dc:creator><rdf:Seq><rdf:li>{escape(creador)}</rdf:li></rdf:Seq></dc:creator>\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )
    return paquete.encode("utf-8")


def _app13_iptc(campos: list[tuple[int, int, bytes]]) -> bytes:
    """Segmento JPEG APP13 (Photoshop 3.0, recurso 0x0404) con los datasets IPTC dados."""
    datos = b"".join(b"\x1c" + bytes([r, d]) + struct.pack(">H", len(v)) + v for r, d, v in campos)
    recurso = b"8BIM" + struct.pack(">H", 0x0404) + b"\x00\x00" + struct.pack(">I", len(datos)) + datos
    if len(datos) % 2:
        recurso += b"\x00"
    carga = b"Photoshop 3.0\x00" + recurso
    return b"\xff\xed" + struct.pack(">H", len(carga) + 2) + carga


def _insertar_segmento_jpeg(jpeg: bytes, segmento: bytes) -> bytes:
    """Inserta ``segmento`` después de los segmentos APPn iniciales del JPEG."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("no es un JPEG")
    i = 2
    while jpeg[i] == 0xFF and 0xE0 <= jpeg[i + 1] <= 0xEF:
        i += 2 + struct.unpack(">H", jpeg[i + 2 : i + 4])[0]
    return jpeg[:i] + segmento + jpeg[i:]


def _ajustar(lz: Lienzo, texto: str, nombre_fuente: str, tam: int, ancho: float) -> list[str]:
    """Corta ``texto`` en líneas que caben en ``ancho`` píxeles."""
    lineas: list[str] = []
    actual = ""
    for palabra in texto.split():
        prueba = f"{actual} {palabra}".strip()
        if actual and lz.ancho_texto(prueba, nombre_fuente, tam) > ancho:
            lineas.append(actual)
            actual = palabra
        else:
            actual = prueba
    if actual:
        lineas.append(actual)
    return lineas


def generar(ctx: Contexto) -> list[Archivo]:
    g = _Generador(ctx)
    g.rotadas()
    g.exif()
    g.tiff()
    return g.archivos
