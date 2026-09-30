"""Cédulas de identidad FICTICIAS: anverso, reverso, fotos con perspectiva, reflejo y copia en PDF.

La disposición de los campos se parece a la de la cédula chilena para que el OCR se enfrente a
algo realista, pero la tarjeta no lleva emblemas, escudos, logos institucionales ni elementos de
seguridad reales, y tiene una franja visible "DOCUMENTO FICTICIO - SOLO PRUEBAS".

Tamaño ID-1 (85,6 x 54 mm) a 12 px/mm: 1027 x 648 px.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from banco_pruebas.contexto import Contexto, archivo_imagen, guardar_imagen
from banco_pruebas.esquema import Archivo, Elemento, Pagina
from banco_pruebas.ficticios import FORMATOS_RUT, Ficticios, Persona, sin_tildes
from banco_pruebas.lienzo import (
    Lienzo,
    caja_envolvente,
    desenfocar,
    iluminacion,
    mapear_a_rect,
    rect,
    ruta_fuente,
    textura_mesa,
    transformar_elementos,
)
from banco_pruebas.rostros import ProveedorRostros

MODULO = "cedula"
CATEGORIA = "cedula"

PX_POR_MM = 12
ANCHO = round(85.6 * PX_POR_MM)  # 1027
ALTO = round(54 * PX_POR_MM)  # 648

TINTA = (22, 30, 45)
TINTA_ETIQUETA = (70, 85, 105)
ROJO_FRANJA = (185, 28, 38)
TEXTO_FRANJA = "DOCUMENTO FICTICIO - SOLO PRUEBAS"
MESES = ("ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC")
PROFESIONES = (
    "INGENIERÍA COMERCIAL",
    "PROFESORA",
    "TÉCNICO EN ENFERMERÍA",
    "ARQUITECTO",
    "TRABAJADORA SOCIAL",
    "CONTADOR AUDITOR",
    "ADMINISTRATIVA",
    "SIN PROFESIÓN",
)


# ---------------------------------------------------------------------------
# Datos de la tarjeta
# ---------------------------------------------------------------------------


@dataclass
class DatosCedula:
    persona: Persona
    numero: str  # 9 dígitos sin puntos
    emision: tuple[int, int, int]  # (día, mes, año)
    vencimiento: tuple[int, int, int]
    profesion: str
    lugar_nacimiento: str

    @property
    def numero_formateado(self) -> str:
        n = self.numero
        return f"{n[:3]}.{n[3:6]}.{n[6:]}"

    @property
    def nacimiento(self) -> tuple[int, int, int]:
        d, m, a = (int(x) for x in self.persona.nacimiento.split("-"))
        return d, m, a

    @property
    def url_qr(self) -> str:
        p = self.persona
        return f"https://portal.ejemplo.cl/docstatus?RUN={p.rut_cuerpo}-{p.rut_dv}&type=CEDULA&serial={self.numero}"


def _fecha_tarjeta(f: tuple[int, int, int]) -> str:
    return f"{f[0]:02d} {MESES[f[1] - 1]} {f[2]}"


def _datos(
    f: Ficticios,
    rng: np.random.Generator,
    usados: set[str],
    *,
    en_lista: bool = True,
    dv_valido: bool = True,
    siete_digitos: bool = False,
    con_k: bool = False,
) -> DatosCedula:
    p = f.persona(dv_valido=dv_valido, en_lista=en_lista, siete_digitos=siete_digitos)
    if con_k:
        p.rut_cuerpo, p.rut_dv = f.rut_con_k()
        p.rut_dv_valido = True
    while True:
        numero = str(int(rng.integers(100_000_000, 600_000_000)))
        if numero not in usados:
            usados.add(numero)
            break
    anio = int(rng.integers(2019, 2026))
    emision = (int(rng.integers(1, 29)), int(rng.integers(1, 13)), anio)
    vencimiento = (emision[0], emision[1], anio + 10)
    return DatosCedula(
        persona=p,
        numero=numero,
        emision=emision,
        vencimiento=vencimiento,
        profesion=str(rng.choice(PROFESIONES)),
        lugar_nacimiento=p.comuna.upper(),
    )


def _nivel_nombre(p: Persona) -> str:
    return "base" if p.en_lista else "fuera_de_alcance"


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------


def _fondo_tarjeta(rng: np.random.Generator, tono: tuple[int, int, int]) -> Image.Image:
    """Fondo claro con un patrón genérico de líneas sinusoidales entrelazadas (tipo guilloche)."""
    yy, xx = np.mgrid[0:ALTO, 0:ANCHO].astype(np.float32)
    grad = (xx / ANCHO) * 10 - (yy / ALTO) * 6
    base = np.array(tono, np.float32)[None, None, :] + grad[..., None]
    img = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    xs = np.arange(0, ANCHO + 4, 3, dtype=np.float64)
    fase = float(rng.uniform(0, 2 * math.pi))
    for i in range(34):
        amp = 18 + 6 * math.sin(i * 0.7)
        y0 = -40 + i * 21
        ys = y0 + amp * np.sin(xs / 57.0 + fase + i * 0.35) + 9 * np.sin(xs / 13.0 + i)
        color = (200, 214, 222) if i % 2 == 0 else (214, 204, 222)
        d.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=color, width=1)
    # rosetón genérico (curva de roseta, no es un emblema)
    cx, cy = ANCHO * 0.78, ALTO * 0.46
    t = np.linspace(0, 2 * math.pi, 1400)
    for k, r0 in ((7, 95), (9, 70), (11, 45)):
        r = r0 + 22 * np.sin(k * t)
        pts = list(zip((cx + r * np.cos(t)).tolist(), (cy + r * np.sin(t)).tolist(), strict=True))
        d.line(pts, fill=(205, 212, 228), width=1)
    return img


def _franja(lz: Lienzo, y0: int, y1: int) -> None:
    ImageDraw.Draw(lz.img).rectangle((0, y0, ANCHO, y1), fill=ROJO_FRANJA)
    tam = 24
    w = lz.ancho_texto(TEXTO_FRANJA, "sans_negrita", tam)
    base = (y0 + y1) / 2 + tam * 0.36
    lz.escribir((ANCHO - w) / 2, base, TEXTO_FRANJA, nombre_fuente="sans_negrita", tam=tam, color=(255, 255, 255))


def _etiqueta(lz: Lienzo, x: float, y: float, texto: str) -> None:
    lz.escribir(x, y, texto, nombre_fuente="sans_negrita", tam=14, color=TINTA_ETIQUETA)


def _valor_sensible(lz: Lienzo, x: float, y: float, texto: str, campo: str, tam: int = 22) -> None:
    lz.escribir(x, y, texto, nombre_fuente="sans", tam=tam, color=TINTA, etiquetas={"sensible": True, "campo": campo})


def _pegar_semitransparente(lz: Lienzo, img: Image.Image, x: int, y: int, ancho: int, alfa: float) -> tuple:
    """Pega una copia gris y semitransparente (foto 'fantasma'). Devuelve (sx, sy) de la escala usada."""
    alto = max(1, round(img.height * ancho / img.width))
    gris = img.convert("L").resize((ancho, alto), Image.Resampling.LANCZOS).convert("RGB")
    mascara = Image.new("L", (ancho, alto), int(round(255 * alfa)))
    lz.img.paste(gris, (x, y), mascara)
    return ancho / img.width, alto / img.height


def dibujar_frente(
    datos: DatosCedula, rostros: ProveedorRostros, rng: np.random.Generator, nivel_datos: str = "base"
) -> Lienzo:
    """Anverso de la cédula ficticia. ``nivel_datos`` fija el nivel de RUT y rostro principal."""
    p = datos.persona
    lz = Lienzo(_fondo_tarjeta(rng, (228, 238, 236)))
    d = ImageDraw.Draw(lz.img)

    # Encabezado
    lz.escribir(40, 52, "CÉDULA DE PRUEBA", nombre_fuente="sans_negrita", tam=34, color=(20, 60, 100))
    lz.escribir(
        40, 80, "IDENTIFICACIÓN FICTICIA PARA PRUEBAS DE SOFTWARE", nombre_fuente="sans", tam=15, color=(40, 70, 100)
    )
    d.line((40, 94, ANCHO - 40, 94), fill=(20, 60, 100), width=2)

    # Foto principal
    (rostro,) = rostros.tomar("frente", 1)
    ancho_foto = 240
    alto_foto = round(rostro.img.height * ancho_foto / rostro.img.width)
    d.rectangle((36, 108, 36 + ancho_foto + 8, 112 + alto_foto + 4), fill=(255, 255, 255))
    lz.pegar(
        rostro.img,
        40,
        112,
        ancho=ancho_foto,
        elementos=ProveedorRostros.elementos(rostro, nivel_datos, {"origen": "cedula"}),
    )

    # Campos
    x1, x2 = 310, 575
    etq_nombre = {"en_lista": p.en_lista}
    _etiqueta(lz, x1, 126, "APELLIDOS")
    for y, apellido, variante in ((160, p.apellido_p, "apellido_paterno"), (194, p.apellido_m, "apellido_materno")):
        lz.escribir(
            x1,
            y,
            apellido,
            tipo="nombre",
            nombre_fuente="sans_negrita",
            tam=28,
            color=TINTA,
            nivel=_nivel_nombre(p),
            etiquetas={**etq_nombre, "variante": variante},
        )
    _etiqueta(lz, x1, 228, "NOMBRES")
    lz.escribir(
        x1,
        262,
        p.nombres,
        tipo="nombre",
        nombre_fuente="sans_negrita",
        tam=28,
        color=TINTA,
        nivel=_nivel_nombre(p),
        etiquetas={**etq_nombre, "variante": "nombres"},
    )
    _etiqueta(lz, x1, 296, "NACIONALIDAD")
    lz.escribir(x1, 324, "CHILENA", tam=22, color=TINTA)
    _etiqueta(lz, x2, 296, "SEXO")
    _valor_sensible(lz, x2, 324, p.sexo, "sexo")
    _etiqueta(lz, x1, 358, "FECHA DE NACIMIENTO")
    _valor_sensible(lz, x1, 386, _fecha_tarjeta(datos.nacimiento), "fecha_nacimiento")
    _etiqueta(lz, x2, 358, "NÚMERO DOCUMENTO")
    _valor_sensible(lz, x2, 386, datos.numero_formateado, "numero_documento")
    _etiqueta(lz, x1, 420, "FECHA DE EMISIÓN")
    _valor_sensible(lz, x1, 448, _fecha_tarjeta(datos.emision), "fecha_emision")
    _etiqueta(lz, x2, 420, "FECHA DE VENCIMIENTO")
    lz.escribir(
        x2,
        448,
        _fecha_tarjeta(datos.vencimiento),
        tam=22,
        color=TINTA,
        etiquetas={"senuelo": "fecha", "campo": "fecha_vencimiento"},
    )

    # RUN grande bajo la foto
    rut = p.rut("puntos")
    x_run = 40
    lz.escribir(x_run, 496, "RUN", nombre_fuente="sans_negrita", tam=36, color=TINTA)
    x_run += lz.ancho_texto("RUN ", "sans_negrita", 36)
    lz.escribir(
        x_run,
        496,
        rut,
        tipo="rut",
        nombre_fuente="sans_negrita",
        tam=36,
        color=TINTA,
        nivel=nivel_datos if nivel_datos != "base" else FORMATOS_RUT["puntos"],
        etiquetas={"formato": "puntos", "dv_valido": p.rut_dv_valido},
    )

    # Firma (manuscrita simulada) con un trazo final
    firma = f"{p.nombres.split()[0][0]}. {p.apellido_p}"
    e_firma = lz.escribir_irregular(
        x2 + 10,
        528,
        firma,
        rng,
        tipo="firma",
        nombre_fuente="stix_cursiva",
        tam=40,
        color=(25, 35, 110),
        nivel="fuera_de_alcance",
        etiquetas={"campo": "firma"},
    )
    if e_firma is not None:
        bx0, by0, bx1, by1 = caja_envolvente(e_firma.poligono)
        xs = np.linspace(bx0 - 6, bx1 + 18, 60)
        ys = by1 + 2 + 5 * np.sin((xs - bx0) / 18.0)
        d.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=(25, 35, 110), width=2)
        tr = [(float(xs.min()), float(ys.min())), (float(xs.max()) + 1, float(ys.max()) + 1)]
        e_firma.poligono = rect(
            math.floor(min(bx0, tr[0][0]) - 1),
            math.floor(min(by0, tr[0][1]) - 1),
            math.ceil(max(bx1, tr[1][0]) + 1),
            math.ceil(max(by1, tr[1][1]) + 1),
        )
    lz.escribir(x2 + 10, 578, "FIRMA DEL TITULAR", nombre_fuente="sans", tam=12, color=TINTA_ETIQUETA)

    # Foto fantasma (copia pequeña semitransparente)
    fx, fy, fancho = 880, 128, 108
    sx, sy = _pegar_semitransparente(lz, rostro.img, fx, fy, fancho, alfa=0.38)
    (fantasma,) = ProveedorRostros.elementos(rostro, "estres", {"fantasma": True, "origen": "cedula"})
    m = np.array([[sx, 0, fx], [0, sy, fy], [0, 0, 1]], dtype=np.float64)
    lz.elementos.extend(transformar_elementos([fantasma], m))

    _franja(lz, 600, ALTO)
    return lz


# ---------------------------------------------------------------------------
# Reverso: zona legible por máquina (MRZ) y código QR
# ---------------------------------------------------------------------------


def _digito_mrz(s: str) -> str:
    pesos = (7, 3, 1)
    total = 0
    for i, c in enumerate(s):
        v = int(c) if c.isdigit() else (ord(c) - 55 if c.isalpha() else 0)
        total += v * pesos[i % 3]
    return str(total % 10)


def _mrz_nombre(p: Persona) -> str:
    def limpio(t: str) -> str:
        return sin_tildes(t).upper().replace("Ñ", "N").replace(" ", "<")

    s = f"{limpio(p.apellido_p)}<{limpio(p.apellido_m)}<<{limpio(p.nombres)}"
    return (s + "<" * 30)[:30]


def lineas_mrz(datos: DatosCedula) -> tuple[str, str, str, str, str]:
    """Devuelve (prefijo_linea1, run_mrz, sufijo_linea1, linea2, linea3) en formato TD1 de 30 caracteres."""
    p = datos.persona
    prefijo = f"INCHL{datos.numero}{_digito_mrz(datos.numero)}"
    run = f"{p.rut_cuerpo}<{p.rut_dv}"
    sufijo = "<" * (30 - len(prefijo) - len(run))
    dn, mn, an = datos.nacimiento
    dv, mv, av = datos.vencimiento
    nac = f"{an % 100:02d}{mn:02d}{dn:02d}"
    ven = f"{av % 100:02d}{mv:02d}{dv:02d}"
    cuerpo2 = f"{nac}{_digito_mrz(nac)}{p.sexo}{ven}{_digito_mrz(ven)}CHL"
    opcional2 = "<" * (29 - len(cuerpo2))
    # dígito compuesto TD1 (ICAO 9303): línea 1 desde la posición 6, y de la línea 2 las
    # fechas con sus dígitos y el campo opcional (sin sexo ni nacionalidad)
    compuesto = (prefijo + run + sufijo)[5:] + nac + _digito_mrz(nac) + ven + _digito_mrz(ven) + opcional2
    linea2 = cuerpo2 + opcional2 + _digito_mrz(compuesto)
    return prefijo, run, sufijo, linea2, _mrz_nombre(p)


def _qr(texto: str, modulo_px: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Imagen del QR (con zona blanca) y la caja de los módulos oscuros en coordenadas de la imagen."""
    q = cv2.QRCodeEncoder.create().encode(texto)
    q = np.pad(q, 2, constant_values=255)
    grande = np.kron(q, np.ones((modulo_px, modulo_px), np.uint8))
    ys, xs = np.nonzero(grande == 0)
    caja = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Image.fromarray(grande).convert("RGB"), caja


def dibujar_dorso(datos: DatosCedula, rng: np.random.Generator) -> Lienzo:
    p = datos.persona
    lz = Lienzo(_fondo_tarjeta(rng, (232, 236, 240)))
    d = ImageDraw.Draw(lz.img)

    _etiqueta(lz, 40, 48, "LUGAR DE NACIMIENTO")
    _valor_sensible(lz, 40, 80, datos.lugar_nacimiento, "lugar_nacimiento", tam=24)
    _etiqueta(lz, 40, 126, "PROFESIÓN")
    _valor_sensible(lz, 40, 158, datos.profesion, "profesion", tam=24)
    lz.escribir(40, 206, "Tarjeta generada para pruebas de software.", tam=16, color=TINTA_ETIQUETA)
    lz.escribir(40, 230, "No es válida como documento de identificación.", tam=16, color=TINTA_ETIQUETA)

    # QR con la URL de verificación (contiene el RUN)
    img_qr, (qx0, qy0, qx1, qy1) = _qr(datos.url_qr, 5)
    px, py = ANCHO - 40 - img_qr.width, 22
    lz.img.paste(img_qr, (px, py))
    lz.elementos.append(
        Elemento(
            tipo="qr",
            pagina=0,
            poligono=rect(px + qx0, py + qy0, px + qx1, py + qy1),
            valor=datos.url_qr,
            nivel="estres",
            capa="raster",
            etiquetas={"contiene": "rut", "modulo_px": 5},
        )
    )

    _franja(lz, 292, 336)

    # MRZ en fondo blanco
    d.rectangle((0, 380, ANCHO, 598), fill=(250, 250, 250))
    prefijo, run, sufijo, linea2, linea3 = lineas_mrz(datos)
    tam, x0 = 44, 62
    etq_mrz = {"sensible": True, "campo": "mrz"}
    y = 446
    lz.escribir(x0, y, prefijo, nombre_fuente="mono", tam=tam, color=TINTA, etiquetas={**etq_mrz, "linea": 1})
    x = x0 + lz.ancho_texto(prefijo, "mono", tam)
    lz.escribir(
        x,
        y,
        run,
        tipo="rut",
        nombre_fuente="mono",
        tam=tam,
        color=TINTA,
        nivel="estres",
        etiquetas={"formato": "mrz", "dv_valido": p.rut_dv_valido, "linea": 1},
    )
    x += lz.ancho_texto(run, "mono", tam)
    lz.escribir(x, y, sufijo, nombre_fuente="mono", tam=tam, color=TINTA, etiquetas={**etq_mrz, "linea": 1})
    lz.escribir(x0, y + 62, linea2, nombre_fuente="mono", tam=tam, color=TINTA, etiquetas={**etq_mrz, "linea": 2})
    # La tercera línea es el nombre en forma MRZ: es un dato personal (difícil de calzar con la lista).
    lz.escribir(
        x0,
        y + 124,
        linea3,
        tipo="nombre",
        nombre_fuente="mono",
        tam=tam,
        color=TINTA,
        nivel="estres" if p.en_lista else "fuera_de_alcance",
        etiquetas={"en_lista": p.en_lista, "variante": "mrz", "formato": "mrz", "linea": 3},
    )
    return lz


# ---------------------------------------------------------------------------
# Composiciones fotográficas
# ---------------------------------------------------------------------------


def _jitter(rng: np.random.Generator, quad: list[list[float]], px: float) -> list[list[float]]:
    return [[x + float(rng.uniform(-px, px)), y + float(rng.uniform(-px, px))] for x, y in quad]


def _quad_girado(cx: float, cy: float, grados: float, escala: float) -> list[list[float]]:
    """Cuadrilátero de la tarjeta con escorzo (borde superior más corto) y giro antihorario ``grados``."""
    w, h = ANCHO * escala / 2, ALTO * escala / 2
    local = [[-w * 0.93, -h], [w * 0.93, -h], [w, h], [-w, h]]
    t = math.radians(grados)
    c, s = math.cos(t), math.sin(t)
    return [[cx + x * c + y * s, cy - x * s + y * c] for x, y in local]


def _foto(lz: Lienzo, destino: list[list[float]], tam: tuple[int, int], rng: np.random.Generator) -> Lienzo:
    fondo = textura_mesa(tam[0], tam[1], rng)
    # sombra suave bajo la tarjeta
    sombra = Image.new("L", tam, 0)
    ImageDraw.Draw(sombra).polygon([(x + 10, y + 14) for x, y in destino], fill=110)
    sombra = desenfocar(sombra, 12)
    oscuro = Image.new("RGB", tam, (30, 20, 12))
    fondo.paste(oscuro, (0, 0), sombra)
    foto = lz.perspectiva(destino, fondo)
    img = iluminacion(foto.img, rng, 0.3)
    foto.img = desenfocar(img, 0.7)
    return foto


def _reflejo(img: Image.Image, centro: tuple[float, float], radios: tuple[float, float], fuerza: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - centro[0]) / radios[0]) ** 2 + ((yy - centro[1]) / radios[1]) ** 2)) * fuerza
    arr = arr + (255 - arr) * g[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _movimiento(img: Image.Image, largo: int, grados: float) -> Image.Image:
    k = np.zeros((largo, largo), np.float32)
    k[largo // 2, :] = 1
    rot = cv2.getRotationMatrix2D((largo / 2 - 0.5, largo / 2 - 0.5), grados, 1.0)
    k = cv2.warpAffine(k, rot, (largo, largo))
    k /= k.sum()
    return Image.fromarray(cv2.filter2D(np.asarray(img), -1, k))


def _ensanchar_por_movimiento(elementos: list[Elemento], largo: int, grados: float) -> None:
    """Agranda los polígonos de texto para que cubran la estela del desenfoque de movimiento.

    El desenfoque arrastra la tinta hasta ``(largo - 1) / 2`` px a cada lado en la dirección
    ``grados`` (antihoraria). Cada borde del cuadrilátero se desplaza hacia afuera según la
    proyección de ese arrastre sobre su normal; el cuadrilátero resultante contiene la suma de
    Minkowski del original con el segmento de arrastre. Los rostros no cambian: su caja es una
    región de la cara, no un contorno de tinta.
    """
    t = math.radians(grados)
    medio = (largo - 1) / 2
    arrastre = np.array([math.cos(t), -math.sin(t)]) * medio
    for e in elementos:
        if e.tipo == "rostro" or e.poligono is None or len(e.poligono) != 4:
            continue
        p = np.asarray(e.poligono, np.float64)
        centro = p.mean(axis=0)
        lineas = []
        for i in range(4):
            a, b = p[i], p[(i + 1) % 4]
            n = np.array([b[1] - a[1], a[0] - b[0]])
            n /= np.linalg.norm(n)
            if np.dot(n, a - centro) < 0:
                n = -n
            lineas.append((n, float(np.dot(n, a)) + abs(float(np.dot(n, arrastre))) + 0.5))
        nuevos = []
        for i in range(4):
            (n1, c1), (n2, c2) = lineas[i - 1], lineas[i]
            x, y = np.linalg.solve(np.array([n1, n2]), np.array([c1, c2]))
            nuevos.append([round(float(x), 3), round(float(y), 3)])
        e.poligono = nuevos


def _marcar(elementos: list[Elemento], **etiquetas: Any) -> None:
    for e in elementos:
        e.etiquetas.update(etiquetas)


def _a_estres(elementos: list[Elemento]) -> None:
    for e in elementos:
        if e.tipo != "texto" and e.nivel == "base":
            e.nivel = "estres"


# ---------------------------------------------------------------------------
# PDF con ambos lados
# ---------------------------------------------------------------------------


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _pdf_ambos_lados(ctx: Contexto, ruta: str, frente: Lienzo, dorso: Lienzo, datos: DatosCedula) -> Archivo:
    p = datos.persona
    doc = pymupdf.open()
    ancho_pag, alto_pag = 595.276, 841.89
    page = doc.new_page(width=ancho_pag, height=alto_pag)
    page.insert_font(fontname="dejavu", fontfile=str(ruta_fuente("sans")))
    elementos: list[Elemento] = []

    def texto(x: float, y: float, t: str, tam: float, tipo: str = "texto", **kw: Any) -> None:
        page.insert_text((x, y), t, fontname="dejavu", fontsize=tam, color=(0.1, 0.1, 0.1))
        quads = page.search_for(t, quads=True)
        assert len(quads) == 1, (t, quads)
        q = quads[0]
        pol = [[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y]]
        pol = [[round(float(a), 3), round(float(b), 3)] for a, b in pol]
        elementos.append(Elemento(tipo=tipo, pagina=0, poligono=pol, valor=t, capa="texto", **kw))

    texto(72, 72, "Copia simple de cédula de identidad (anverso y reverso)", 13)
    texto(72, 92, "Documento ficticio generado para pruebas del anonimizador.", 10)
    ancho_img = 400.0
    alto_img = ancho_img * ALTO / ANCHO
    x0 = (ancho_pag - ancho_img) / 2
    cajas = [(x0, 120.0, x0 + ancho_img, 120.0 + alto_img)]
    cajas.append((x0, cajas[0][3] + 36, x0 + ancho_img, cajas[0][3] + 36 + alto_img))
    for lz, caja in ((frente, cajas[0]), (dorso, cajas[1])):
        page.insert_image(pymupdf.Rect(*caja), stream=_png(lz.img))
        elementos.extend(mapear_a_rect(lz.elementos, lz.ancho, lz.alto, caja))
    y_titular = cajas[1][3] + 40
    texto(72, y_titular, "Titular:", 11)
    x_nombre = 72 + pymupdf.Font(fontfile=str(ruta_fuente("sans"))).text_length("Titular: ", fontsize=11)
    texto(
        x_nombre,
        y_titular,
        p.nombre_completo,
        11,
        tipo="nombre",
        nivel=_nivel_nombre(p),
        etiquetas={"en_lista": p.en_lista, "variante": "completo"},
    )
    texto(72, y_titular + 20, "Se extiende la presente copia para fines de prueba.", 10)
    doc.set_metadata({})
    doc.subset_fonts()
    destino = ctx.ruta(ruta)
    # no_new_id: sin /ID aleatorio en el trailer, para que el archivo sea idéntico en cada corrida
    doc.save(destino, garbage=3, deflate=True, no_new_id=True)
    doc.close()
    for e in elementos:
        e.pagina = 0
    return Archivo(
        id="ced_ambos_lados_pdf",
        ruta=ruta,
        formato="pdf",
        categoria=CATEGORIA,
        descripcion="PDF con las imágenes del anverso y reverso de una cédula ficticia y el nombre del titular.",
        paginas=[Pagina(indice=0, ancho=ancho_pag, alto=alto_pag, unidad="pt", rotacion=0)],
        elementos=elementos,
        etiquetas={"imagenes_incrustadas": 2},
    )


# ---------------------------------------------------------------------------
# Generador
# ---------------------------------------------------------------------------


def generar(ctx: Contexto) -> list[Archivo]:
    f = ctx.ficticios(MODULO)
    rng = ctx.rng(MODULO)
    usados: set[str] = set()
    archivos: list[Archivo] = []

    def guardar(lz: Lienzo, ruta: str, id: str, descripcion: str, formato: str, **etq: Any) -> None:
        opciones = {"quality": etq.pop("calidad")} if "calidad" in etq else {}
        guardar_imagen(lz.img, ctx.ruta(ruta), formato, **opciones)
        archivos.append(
            archivo_imagen(
                id=id,
                ruta=ruta,
                formato=formato,
                categoria=CATEGORIA,
                descripcion=descripcion,
                lienzo=lz,
                etiquetas=etq,
            )
        )

    # 1. Anverso plano
    d1 = _datos(f, rng, usados)
    frente = dibujar_frente(d1, ctx.rostros, rng)
    guardar(frente, "cedula/cedula_frente_plana.png", "ced_frente_plana", "Anverso plano de cédula ficticia.", "png")

    # 2. Reverso plano (persona fuera de la lista)
    d2 = _datos(f, rng, usados, en_lista=False)
    dorso = dibujar_dorso(d2, rng)
    guardar(dorso, "cedula/cedula_dorso_plana.png", "ced_dorso_plana", "Reverso plano: MRZ y QR con el RUN.", "png")

    # 3. Foto con perspectiva sobre una mesa
    d3 = _datos(f, rng, usados, siete_digitos=True)
    lz3 = dibujar_frente(d3, ctx.rostros, rng)
    destino3 = _jitter(rng, [[250, 230], [1330, 290], [1380, 960], [210, 915]], 12)
    foto3 = _foto(lz3, destino3, (1600, 1200), rng)
    guardar(
        foto3,
        "cedula/cedula_frente_foto_perspectiva.jpg",
        "ced_frente_foto_perspectiva",
        "Foto del anverso sobre una mesa, con perspectiva, luz desigual y leve desenfoque (JPEG 80).",
        "jpg",
        calidad=80,
        perspectiva=True,
    )

    # 4. Foto con perspectiva y giro de ~30 grados
    d4 = _datos(f, rng, usados, dv_valido=False)
    lz4 = dibujar_frente(d4, ctx.rostros, rng)
    grados = 30 + float(rng.uniform(-2, 2))
    destino4 = _jitter(rng, _quad_girado(850, 750, grados, 1.09), 6)
    foto4 = _foto(lz4, destino4, (1700, 1500), rng)
    for e in foto4.elementos:
        e.etiquetas["angulo"] = round(grados, 2)
    guardar(
        foto4,
        "cedula/cedula_frente_foto_girada.jpg",
        "ced_frente_foto_girada",
        f"Foto del anverso girada {grados:.0f} grados y con perspectiva (JPEG 80).",
        "jpg",
        calidad=80,
        perspectiva=True,
        angulo=round(grados, 2),
    )

    # 5. Foto del reverso
    d5 = _datos(f, rng, usados, con_k=True)
    lz5 = dibujar_dorso(d5, rng)
    destino5 = _jitter(rng, [[230, 280], [1360, 240], [1390, 930], [200, 960]], 12)
    foto5 = _foto(lz5, destino5, (1600, 1200), rng)
    guardar(
        foto5,
        "cedula/cedula_dorso_foto.jpg",
        "ced_dorso_foto",
        "Foto del reverso (MRZ y QR) sobre una mesa, con perspectiva (JPEG 80).",
        "jpg",
        calidad=80,
        perspectiva=True,
    )

    # 6. PDF con ambos lados (misma persona en anverso y reverso)
    d6 = _datos(f, rng, usados)
    frente6 = dibujar_frente(d6, ctx.rostros, rng)
    dorso6 = dibujar_dorso(d6, rng)
    archivos.append(_pdf_ambos_lados(ctx, "cedula/cedula_ambos_lados.pdf", frente6, dorso6, d6))

    # 7. Foto con reflejo y desenfoque de movimiento (estrés)
    d7 = _datos(f, rng, usados, en_lista=False)
    lz7 = dibujar_frente(d7, ctx.rostros, rng, nivel_datos="estres")
    destino7 = _jitter(rng, [[240, 260], [1340, 250], [1370, 950], [220, 930]], 10)
    foto7 = _foto(lz7, destino7, (1600, 1200), rng)
    nombres = [e for e in foto7.elementos if e.tipo == "nombre"]
    cx = float(np.mean([pt[0] for e in nombres for pt in e.poligono]))
    cy = float(np.mean([pt[1] for e in nombres for pt in e.poligono]))
    largo_mov, grados_mov = 11, 8
    foto7.img = _movimiento(_reflejo(foto7.img, (cx - 60, cy + 40), (260, 150), 0.82), largo_mov, grados_mov)
    _ensanchar_por_movimiento(foto7.elementos, largo_mov, grados_mov)
    _a_estres(foto7.elementos)
    _marcar(foto7.elementos, reflejo=True, desenfoque_movimiento=largo_mov)
    guardar(
        foto7,
        "cedula/cedula_con_reflejo.jpg",
        "ced_con_reflejo",
        "Foto del anverso con un reflejo fuerte sobre los nombres y desenfoque de movimiento.",
        "jpg",
        calidad=85,
        perspectiva=True,
        reflejo=True,
    )
    return archivos
