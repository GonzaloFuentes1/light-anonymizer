"""Pantallazos sintéticos: cliente de correo, chat de celular, planilla, formulario web y diálogo.

Las interfaces se dibujan con ``Lienzo`` con un aspecto genérico (sin marcas de productos reales).
Cada archivo usa personas distintas: los valores son canarios únicos en todo el conjunto, por eso
las variantes reescaladas (``planilla_reducida``, ``correo_escritorio_hidpi``) repiten el diseño
pero con otros datos.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from banco_pruebas.contexto import Contexto, archivo_imagen, guardar_imagen
from banco_pruebas.esquema import Archivo, Elemento
from banco_pruebas.ficticios import (
    FORMATOS_CORREO,
    FORMATOS_POR_CLASE,
    FORMATOS_RUT,
    FORMATOS_TELEFONO,
    FRASES_ADMINISTRATIVAS,
    Ficticios,
    Persona,
    Telefono,
)
from banco_pruebas.lienzo import Lienzo, transformar_elementos
from banco_pruebas.rostros import ProveedorRostros, Rostro

MODULO = "pantallazos"
CATEGORIA = "pantallazo"

Color = tuple[int, int, int]

CORREOS_BASE = [f for f, n in FORMATOS_CORREO.items() if n == "base"]
RUTS_BASE = [f for f, n in FORMATOS_RUT.items() if n == "base"]


# ---------------------------------------------------------------------------
# Segmentos de texto con tipo (una línea puede mezclar etiquetas neutras y datos)
# ---------------------------------------------------------------------------


@dataclass
class Seg:
    texto: str
    tipo: str = "texto"
    nivel: str = "base"
    etiquetas: dict[str, Any] = field(default_factory=dict)
    color: Color | None = None
    fuente: str | None = None


def t(texto: str, **etiquetas: Any) -> Seg:
    """Segmento de texto neutro."""
    return Seg(texto, etiquetas=etiquetas)


def nombre(p: Persona, variante: str = "completo", nivel: str | None = None) -> Seg:
    valor = p.nombre_completo if variante == "completo" else f"{p.nombres} {p.apellido_p}"
    return Seg(
        valor,
        "nombre",
        nivel or ("base" if p.en_lista else "fuera_de_alcance"),
        {"en_lista": p.en_lista, "variante": variante},
    )


def direccion(p: Persona) -> Seg:
    return Seg(p.direccion, "direccion", "base" if p.en_lista else "fuera_de_alcance", {"en_lista": p.en_lista})


def rut(p: Persona, formato: str = "puntos") -> Seg:
    return Seg(p.rut(formato), "rut", FORMATOS_RUT[formato], {"formato": formato, "dv_valido": p.rut_dv_valido})


def correo(p: Persona, formato: str = "punto") -> Seg:
    return Seg(p.correo(formato), "correo", FORMATOS_CORREO[formato], {"formato": formato})


def telefono(tel: Telefono, formato: str) -> Seg:
    return Seg(tel.formatear(formato), "telefono", FORMATOS_TELEFONO[formato], {"formato": formato, "clase": tel.clase})


def escribir_segs(
    lz: Lienzo,
    x: float,
    y: float,
    segs: list[Seg],
    *,
    tam: int,
    nombre_fuente: str = "sans",
    color: Color = (30, 30, 30),
) -> float:
    """Escribe los segmentos seguidos sobre la línea base ``y``. Devuelve la x final."""
    for s in segs:
        f_s = s.fuente or nombre_fuente
        limpio = s.texto.strip()
        if limpio:
            ini = len(s.texto) - len(s.texto.lstrip())
            x0 = x + lz.ancho_texto(s.texto[:ini], f_s, tam)
            lz.escribir(
                x0,
                y,
                limpio,
                tipo=s.tipo,
                nombre_fuente=f_s,
                tam=tam,
                color=s.color or color,
                nivel=s.nivel,
                etiquetas=s.etiquetas,
            )
        x += lz.ancho_texto(s.texto, f_s, tam)
    return x


def ancho_segs(lz: Lienzo, segs: list[Seg], tam: int, nombre_fuente: str = "sans") -> float:
    return sum(lz.ancho_texto(s.texto, s.fuente or nombre_fuente, tam) for s in segs)


def _tam_que_cabe(lz: Lienzo, segs: list[Seg], tam: int, ancho: float, nombre_fuente: str = "sans") -> int:
    """Mayor tamaño de letra (hasta ``tam``) con el que los segmentos caben en ``ancho``."""
    while tam > 6 and ancho_segs(lz, segs, tam, nombre_fuente) > ancho:
        tam -= 1
    return tam


def _a_estres(elementos: list[Elemento]) -> None:
    for e in elementos:
        if e.tipo != "texto" and e.nivel == "base":
            e.nivel = "estres"


def _elegir(rng: np.random.Generator, opciones: list[str]) -> str:
    return str(opciones[int(rng.integers(len(opciones)))])


def _formato_movil(rng: np.random.Generator) -> str:
    return _elegir(rng, [f for f in FORMATOS_POR_CLASE["movil"] if FORMATOS_TELEFONO[f] == "base"])


def _formato_fijo(rng: np.random.Generator, tel: Telefono) -> str:
    return _elegir(rng, [f for f in FORMATOS_POR_CLASE[tel.clase] if FORMATOS_TELEFONO[f] == "base"])


# ---------------------------------------------------------------------------
# Rostros recortados para avatares
# ---------------------------------------------------------------------------


def _recorte_centrado(r: Rostro, holgura: float = 1.5) -> tuple[Image.Image, list[Elemento]]:
    """Recorte cuadrado centrado en la cara (con bordes replicados si hace falta) y su elemento."""
    xs = [p[0] for p in r.caja]
    ys = [p[1] for p in r.caja]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    lado = int(round(max(max(xs) - min(xs), max(ys) - min(ys)) * holgura))
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    arr = np.asarray(r.img.convert("RGB"))
    pad = max(0, -x0, -y0, x0 + lado - arr.shape[1], y0 + lado - arr.shape[0])
    if pad:
        arr = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    recorte = Image.fromarray(arr[y0 + pad : y0 + pad + lado, x0 + pad : x0 + pad + lado].copy())
    m = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    return recorte, transformar_elementos(ProveedorRostros.elementos(r), m)


def _mascara_circular(lado: int, sobremuestreo: int = 4) -> Image.Image:
    grande = Image.new("L", (lado * sobremuestreo, lado * sobremuestreo), 0)
    ImageDraw.Draw(grande).ellipse((0, 0, lado * sobremuestreo - 1, lado * sobremuestreo - 1), fill=255)
    return grande.resize((lado, lado), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# 1. Cliente de correo de escritorio (escala 1.0 y 1.5)
# ---------------------------------------------------------------------------

ASUNTOS = [
    "Informe de avance mensual - septiembre",
    "Boleta de honorarios",
    "Reunión de coordinación del jueves",
    "Solicitud de antecedentes",
    "Rendición de gastos del taller",
    "Acta de la sesión de ayer",
    "Consulta sobre el convenio",
    "Certificado de recepción conforme",
    "Programa de talleres comunitarios",
    "Actualización de datos del proveedor",
]
HORAS = ["10:42", "09:15", "08:03", "Ayer", "Ayer", "lun", "dom", "vie", "jue", "mié"]


def _correo_escritorio(f: Ficticios, rng: np.random.Generator, escala: float) -> Lienzo:
    def e(v: float) -> float:
        return v * escala

    def z(v: float) -> int:
        return int(round(v * escala))

    W, H = z(1920), z(1080)
    lz = Lienzo.nuevo(W, H, (255, 255, 255))
    d = ImageDraw.Draw(lz.img)
    gris = (110, 115, 125)

    dueno = f.persona()
    remitentes = [f.persona(en_lista=(i not in (3, 7))) for i in range(10)]
    autor = remitentes[0]
    cc = [f.persona(), f.persona(en_lista=False)]

    # Barra de título
    d.rectangle((0, 0, W, z(36)), fill=(45, 48, 56))
    escribir_segs(
        lz,
        e(16),
        e(24),
        [t("Bandeja de entrada - "), correo(dueno, "punto"), t(" - Cliente de correo")],
        tam=z(13),
        color=(235, 235, 240),
    )
    for i, c in enumerate(((236, 95, 90), (245, 190, 80), (100, 200, 110))):
        cx = W - e(30 + 26 * i)
        d.ellipse((cx - e(7), e(11), cx + e(7), e(25)), fill=c)

    # Barra de herramientas
    d.rectangle((0, z(36), W, z(84)), fill=(243, 244, 246))
    x = e(16)
    for texto in ("Nuevo mensaje", "Responder", "Responder a todos", "Reenviar", "Archivar", "Eliminar"):
        w = lz.ancho_texto(texto, "sans", z(14)) + e(28)
        d.rounded_rectangle((x, e(46), x + w, e(74)), radius=z(5), fill=(255, 255, 255), outline=(205, 208, 214))
        lz.escribir(x + e(14), e(65), texto, tam=z(14), color=(40, 45, 55))
        x += w + e(10)
    d.rounded_rectangle(
        (W - e(440), e(46), W - e(20), e(74)), radius=z(5), fill=(255, 255, 255), outline=(205, 208, 214)
    )
    lz.escribir(W - e(424), e(65), "Buscar en el correo", tam=z(14), color=(150, 150, 158))

    # Barra lateral de carpetas
    d.rectangle((0, z(84), z(250), H - z(28)), fill=(248, 249, 251))
    # la barra lateral mide 250 px: nombre y correo del dueño se achican hasta caber
    tam_n = _tam_que_cabe(lz, [nombre(dueno)], z(14), e(222), "sans_negrita")
    escribir_segs(lz, e(20), e(118), [nombre(dueno)], tam=tam_n, nombre_fuente="sans_negrita")
    tam_c = _tam_que_cabe(lz, [correo(dueno, "punto")], z(13), e(222))
    escribir_segs(lz, e(20), e(138), [correo(dueno, "punto")], tam=tam_c, color=gris)
    carpetas = [("Bandeja de entrada", "12"), ("Destacados", ""), ("Enviados", ""), ("Borradores", "2"),
                ("Archivados", ""), ("Spam", "5"), ("Papelera", "")]  # fmt: skip
    for i, (carpeta, n) in enumerate(carpetas):
        y = e(186 + 34 * i)
        if i == 0:
            d.rounded_rectangle((e(10), y - e(22), e(240), y + e(10)), radius=z(6), fill=(220, 232, 250))
        lz.escribir(e(28), y, carpeta, nombre_fuente="sans_negrita" if i == 0 else "sans", tam=z(14))
        if n:
            lz.escribir(e(222) - lz.ancho_texto(n, "sans", z(13)), y, n, tam=z(13), color=gris)

    # Lista de mensajes
    x0l, x1l = z(250), z(800)
    d.line((x0l, z(84), x0l, H - z(28)), fill=(220, 222, 228), width=max(1, z(1)))
    d.line((x1l, z(84), x1l, H - z(28)), fill=(220, 222, 228), width=max(1, z(1)))
    lz.escribir(x0l + e(20), e(116), "Bandeja de entrada", nombre_fuente="sans_negrita", tam=z(15))
    lz.escribir(x1l - e(20) - lz.ancho_texto("Ordenar: fecha", "sans", z(13)), e(116), "Ordenar: fecha",
                tam=z(13), color=gris)  # fmt: skip
    formatos_lista = [_elegir(rng, CORREOS_BASE) for _ in remitentes]
    for i, (p, asunto, hora) in enumerate(zip(remitentes, ASUNTOS, HORAS, strict=True)):
        y = e(132 + 80 * i)
        if i == 0:
            d.rectangle((x0l + 1, y, x1l - 1, y + e(80)), fill=(220, 232, 250))
        d.line((x0l, y + e(80), x1l, y + e(80)), fill=(232, 234, 238), width=max(1, z(1)))
        escribir_segs(lz, x0l + e(20), y + e(24), [nombre(p)], tam=z(14), nombre_fuente="sans_negrita")
        lz.escribir(x1l - e(20) - lz.ancho_texto(hora, "sans", z(13)), y + e(24), hora, tam=z(13), color=gris,
                    etiquetas={"campo": "hora"})  # fmt: skip
        escribir_segs(lz, x0l + e(20), y + e(45), [correo(p, formatos_lista[i])], tam=z(13), color=gris)
        lz.escribir(x0l + e(20), y + e(66), asunto, tam=z(13), color=(50, 55, 65))

    # Mensaje abierto
    xm = x1l + e(32)
    lz.escribir(xm, e(128), ASUNTOS[0], nombre_fuente="sans_negrita", tam=z(20))
    fmt_autor = formatos_lista[0]
    y = e(170)
    escribir_segs(lz, xm, y, [t("De: "), nombre(autor), t(" <"), correo(autor, fmt_autor), t(">")], tam=z(14))
    escribir_segs(lz, xm, y + e(24), [t("Para: "), nombre(dueno), t(" <"), correo(dueno, "punto"), t(">")],
                  tam=z(14))  # fmt: skip
    fmt_cc = [_elegir(rng, CORREOS_BASE) for _ in cc]
    segs_cc = [t("CC: "), nombre(cc[0]), t(" <"), correo(cc[0], fmt_cc[0]), t(">, "), nombre(cc[1]), t(" <"),
               correo(cc[1], fmt_cc[1]), t(">")]  # fmt: skip
    escribir_segs(lz, xm, y + e(48), segs_cc, tam=_tam_que_cabe(lz, segs_cc, z(14), W - xm - e(24)))
    escribir_segs(lz, xm, y + e(72), [t("Fecha: "), Seg(f.fecha(), etiquetas={"senuelo": "fecha"}), t(" 10:42")],
                  tam=z(14), color=gris)  # fmt: skip
    # Adjunto
    ya = y + e(96)
    d.rounded_rectangle((xm, ya, xm + e(290), ya + e(40)), radius=z(6), fill=(246, 247, 249), outline=(210, 213, 220))
    d.rectangle((xm + e(10), ya + e(8), xm + e(32), ya + e(32)), fill=(200, 60, 60))
    lz.escribir(xm + e(42), ya + e(26), "informe_septiembre.pdf (245 KB)", tam=z(13))
    d.line((xm, ya + e(58), W - e(32), ya + e(58)), fill=(225, 227, 232), width=max(1, z(1)))

    tel_autor = autor.telefono
    fmt_tel = _formato_movil(rng)
    fmt_rut = _elegir(rng, RUTS_BASE)
    frases = [FRASES_ADMINISTRATIVAS[int(i)] for i in rng.choice(len(FRASES_ADMINISTRATIVAS), 2, replace=False)]
    lineas: list[list[Seg]] = [
        [t("Estimado equipo:")],
        [],
        [t("Junto con saludar, adjunto el informe de avance correspondiente al mes de septiembre.")],
        [t(frases[0])],
        [t(frases[1])],
        [],
        [t("Para efectos del pago de la boleta, mis datos son los siguientes:")],
        [t("RUT: "), rut(autor, fmt_rut)],
        [t("Teléfono: "), telefono(tel_autor, fmt_tel)],
        [t("Dirección: "), direccion(autor), t(", "), t(autor.comuna, campo="comuna")],
        [t("El monto bruto de la boleta es de "), Seg(f.monto(), etiquetas={"senuelo": "monto"}), t(".")],
        [],
        [t("Quedo atento(a) a cualquier consulta."), ],
        [t("Saludos cordiales,")],
        [],
        [t("--")],
        [nombre(autor)],
        [t("Profesional de apoyo, División de Planificación")],
        [t("Gobierno Regional Ficticio")],
        [t("Tel.: "), telefono(autor.telefono_fijo, _formato_fijo(rng, autor.telefono_fijo))],
        [correo(autor, "mayusculas" if fmt_autor != "mayusculas" else "punto")],
    ]  # fmt: skip
    y = ya + e(96)
    for i, segs in enumerate(lineas):
        es_firma = i >= 16
        escribir_segs(
            lz,
            xm,
            y,
            segs,
            tam=z(14 if es_firma else 15),
            nombre_fuente="sans_negrita" if i == 16 else "sans",
            color=(70, 75, 85) if es_firma else (30, 30, 30),
        )
        y += e(22 if es_firma else 26)

    # Barra de estado
    d.rectangle((0, H - z(28), W, H), fill=(236, 238, 242))
    lz.escribir(e(16), H - e(9), "Conectado - 12 mensajes sin leer", tam=z(12), color=gris)
    return lz


# ---------------------------------------------------------------------------
# 2. Chat en el celular
# ---------------------------------------------------------------------------


def _chat_movil(f: Ficticios, rng: np.random.Generator, rostros: ProveedorRostros) -> Lienzo:
    W, H = 1080, 2340
    lz = Lienzo.nuevo(W, H, (236, 229, 221))
    d = ImageDraw.Draw(lz.img)
    contacto = f.persona()
    yo = f.persona()
    pareja = f.persona(en_lista=False)

    # Barra de estado y encabezado
    d.rectangle((0, 0, W, 300), fill=(0, 105, 92))
    lz.escribir(48, 58, "10:42", nombre_fuente="sans_negrita", tam=34, color=(255, 255, 255),
                etiquetas={"campo": "hora"})  # fmt: skip
    lz.escribir(W - 48 - lz.ancho_texto("5G 87%", "sans", 32), 58, "5G 87%", tam=32, color=(255, 255, 255))
    d.line([(70, 200), (44, 176), (70, 152)], fill=(255, 255, 255), width=7)
    (r,) = rostros.tomar("frente", 1)
    recorte, elems = _recorte_centrado(r, 1.5)
    lado = 136
    lz.pegar(recorte, 96, 108, ancho=lado, elementos=elems, mascara=_mascara_circular(recorte.width))
    lz.elementos[-1].etiquetas.update({"avatar": True, "mascara": "circular"})
    escribir_segs(lz, 262, 176, [nombre(contacto)], tam=42, nombre_fuente="sans_negrita", color=(255, 255, 255))
    lz.escribir(262, 226, "en línea", tam=30, color=(210, 235, 230))

    # Chip de fecha
    ancho_hoy = lz.ancho_texto("HOY", "sans_negrita", 28) + 48
    d.rounded_rectangle(((W - ancho_hoy) / 2, 340, (W + ancho_hoy) / 2, 392), radius=16, fill=(221, 236, 244))
    lz.escribir((W - ancho_hoy) / 2 + 24, 377, "HOY", nombre_fuente="sans_negrita", tam=28, color=(80, 90, 100))

    fmt_rut_pareja = "sin_puntos"
    mensajes: list[tuple[bool, list[list[Seg]], str]] = [
        (False, [[t("Hola! Para el contrato de honorarios")], [t("necesito tus datos, porfa")]], "09:58"),
        (True, [[t("Hola! Claro. Mi RUT es "), rut(yo, "puntos")]], "10:01"),
        (True, [[t("Correo: "), correo(yo, "con_anio")]], "10:02"),
        (False, [[t("¿Y un teléfono de contacto?")]], "10:03"),
        (True, [[t("Al celular: "), telefono(yo.telefono, "movil_internacional")]], "10:03"),
        (True, [[t("Vivo en")], [direccion(yo)], [t(yo.comuna, campo="comuna")]], "10:04"),
        (False, [[t("¿Me pasas también el RUT de tu pareja")], [t("para la carga del seguro?")]], "10:05"),
        (True, [[t("Es "), rut(pareja, fmt_rut_pareja)], [nombre(pareja)]], "10:06"),
        (False, [[t("Perfecto. Cualquier cosa llámame")],
                 [t("a la oficina: "), telefono(contacto.telefono_fijo, _formato_fijo(rng, contacto.telefono_fijo))]],
         "10:07"),
    ]  # fmt: skip
    y = 430.0
    tam, alto_linea, pad = 40, 54, 24
    for propio, lineas, hora in mensajes:
        tams = []
        for segs in lineas:
            w = ancho_segs(lz, segs, tam)
            tams.append(tam if w <= 800 else int(tam * 800 / w))
        anchos = [ancho_segs(lz, segs, tl) for segs, tl in zip(lineas, tams, strict=True)]
        ancho_hora = lz.ancho_texto(hora, "sans", 26)
        w = max(max(anchos), anchos[-1] + ancho_hora + 24) + 2 * pad
        h = len(lineas) * alto_linea + 2 * pad + 22
        assert y + h < 2190, "el chat no cabe sobre la barra para escribir"
        x0 = W - 40 - w if propio else 40
        d.rounded_rectangle((x0, y, x0 + w, y + h), radius=26, fill=(217, 253, 211) if propio else (255, 255, 255))
        for i, (segs, tl) in enumerate(zip(lineas, tams, strict=True)):
            escribir_segs(lz, x0 + pad, y + pad + 40 + i * alto_linea, segs, tam=tl, color=(20, 20, 20))
        lz.escribir(x0 + w - pad - ancho_hora, y + h - 14, hora, tam=26, color=(110, 120, 115),
                    etiquetas={"campo": "hora"})  # fmt: skip
        y += h + 18

    # Barra para escribir
    d.rectangle((0, 2190, W, H), fill=(236, 229, 221))
    d.rounded_rectangle((30, 2210, W - 170, 2320), radius=55, fill=(255, 255, 255))
    lz.escribir(90, 2280, "Mensaje", tam=40, color=(140, 140, 140))
    d.ellipse((W - 150, 2210, W - 40, 2320), fill=(0, 140, 120))
    d.polygon([(W - 118, 2240), (W - 62, 2265), (W - 118, 2290), (W - 110, 2265)], fill=(255, 255, 255))
    return lz


# ---------------------------------------------------------------------------
# 3. Planilla de cálculo
# ---------------------------------------------------------------------------


def _planilla(f: Ficticios, rng: np.random.Generator) -> Lienzo:
    W, H = 1600, 900
    lz = Lienzo.nuevo(W, H, (255, 255, 255))
    d = ImageDraw.Draw(lz.img)
    gris = (100, 105, 112)

    # Menús y barra de fórmulas
    d.rectangle((0, 0, W, 34), fill=(245, 246, 248))
    lz.escribir(14, 23, "honorarios_2026.xlsx", nombre_fuente="sans_negrita", tam=13)
    x = 200
    for m in ("Archivo", "Editar", "Ver", "Insertar", "Formato", "Datos", "Herramientas", "Ayuda"):
        lz.escribir(x, 23, m, tam=13, color=(50, 55, 60))
        x += lz.ancho_texto(m, "sans", 13) + 22
    d.rectangle((0, 34, W, 72), fill=(250, 250, 251))
    d.line((0, 72, W, 72), fill=(210, 212, 216))

    personas = [f.persona(en_lista=(i not in (2, 6, 10))) for i in range(12)]
    fmt_rut = [_elegir(rng, RUTS_BASE) if i != 9 else "sin_guion" for i in range(12)]
    fmt_correo = [_elegir(rng, CORREOS_BASE) for _ in range(12)]
    fmt_tel = [_formato_movil(rng) if i % 4 else _formato_fijo(rng, p.telefono_fijo) for i, p in enumerate(personas)]

    d.rectangle((10, 42, 90, 64), fill=(255, 255, 255), outline=(200, 202, 206))
    lz.escribir(18, 58, "B5", tam=13)
    lz.escribir(104, 58, "fx", nombre_fuente="serif_cursiva", tam=13, color=gris)
    escribir_segs(lz, 130, 58, [rut(personas[3], fmt_rut[3])], tam=13)

    # Grilla
    y0, alto_fila, ancho_nf = 80, 30, 44
    cols = [("Nombre", 270), ("RUT", 130), ("Correo", 340), ("Teléfono", 170), ("Comuna", 140), ("Monto", 120)]
    cols += [("", 110)] * 5
    xs = [ancho_nf]
    for _, w in cols:
        xs.append(xs[-1] + w)
    d.rectangle((0, y0, W, y0 + 26), fill=(240, 241, 243))
    d.rectangle((0, y0, ancho_nf, H - 40), fill=(240, 241, 243))
    for j in range(len(cols)):
        letra = chr(ord("A") + j)
        cx = (xs[j] + xs[j + 1]) / 2
        if cx > W - 12:
            break
        lz.escribir(cx - lz.ancho_texto(letra, "sans", 12) / 2, y0 + 18, letra, tam=12, color=gris)
    y_filas = y0 + 26
    n_filas = (H - 40 - y_filas) // alto_fila
    for i in range(n_filas):
        yf = y_filas + i * alto_fila
        num = str(i + 1)
        lz.escribir(ancho_nf / 2 - lz.ancho_texto(num, "sans", 12) / 2, yf + 20, num, tam=12, color=gris)
    for xv in xs:
        d.line((xv, y0, xv, H - 40), fill=(218, 220, 224))
    for i in range(n_filas + 1):
        d.line((0, y_filas + i * alto_fila, W, y_filas + i * alto_fila), fill=(218, 220, 224))
    d.line((0, y0 + 26, W, y0 + 26), fill=(200, 202, 206))
    # celda seleccionada
    d.rectangle((xs[1], y_filas + 4 * alto_fila, xs[2], y_filas + 5 * alto_fila), outline=(30, 110, 220), width=2)

    # Encabezados y datos
    d.rectangle((xs[0] + 1, y_filas + 1, xs[6] - 1, y_filas + alto_fila - 1), fill=(226, 239, 218))
    for j, (titulo, _) in enumerate(cols[:6]):
        lz.escribir(xs[j] + 8, y_filas + 20, titulo, nombre_fuente="sans_negrita", tam=13)
    for i, p in enumerate(personas):
        yb = y_filas + (i + 1) * alto_fila + 20
        tel = p.telefono if fmt_tel[i].startswith(("movil", "antiguo_movil")) else p.telefono_fijo
        celdas = [
            nombre(p),
            rut(p, fmt_rut[i]),
            correo(p, fmt_correo[i]),
            telefono(tel, fmt_tel[i]),
            t(p.comuna, campo="comuna"),
        ]
        for j, s in enumerate(celdas):
            # la celda no se desborda: se achica la letra (como el ajuste de texto de una planilla)
            tam_celda = _tam_que_cabe(lz, [s], 13, cols[j][1] - 14)
            assert tam_celda >= 10, s.texto
            escribir_segs(lz, xs[j] + 8, yb, [s], tam=tam_celda)
        monto = f.monto()
        lz.escribir(xs[6] - 8 - lz.ancho_texto(monto, "sans", 13), yb, monto, tam=13, etiquetas={"senuelo": "monto"})

    # Pestañas de hojas
    d.rectangle((0, H - 40, W, H), fill=(245, 246, 248))
    for k, hoja in enumerate(("Honorarios", "Resumen", "Hoja3")):
        xh = 60 + k * 130
        if k == 0:
            d.rectangle((xh - 10, H - 40, xh + 110, H - 8), fill=(255, 255, 255), outline=(210, 212, 216))
        lz.escribir(xh, H - 18, hoja, nombre_fuente="sans_negrita" if k == 0 else "sans", tam=13)
    return lz


# ---------------------------------------------------------------------------
# 4. Formulario web
# ---------------------------------------------------------------------------


def _formulario_web(f: Ficticios, rng: np.random.Generator, rostros: ProveedorRostros) -> Lienzo:
    W, H = 1366, 768
    lz = Lienzo.nuevo(W, H, (244, 246, 249))
    d = ImageDraw.Draw(lz.img)
    p = f.persona()
    gris = (95, 100, 110)

    # Navegador genérico: pestaña y barra de dirección (la URL lleva el RUN como parámetro)
    d.rectangle((0, 0, W, 86), fill=(222, 225, 230))
    d.rounded_rectangle((12, 6, 300, 40), radius=8, fill=(255, 255, 255))
    lz.escribir(28, 29, "Mis datos - Portal de trámites", tam=13)
    d.rounded_rectangle((110, 46, W - 110, 80), radius=17, fill=(255, 255, 255))
    for i, simbolo in enumerate(("<", ">", "C")):
        lz.escribir(18 + 30 * i, 70, simbolo, nombre_fuente="sans_negrita", tam=15, color=gris)
    url_pre = "https://tramites.ejemplo.cl/perfil/editar?run="
    run_url = p.rut("sin_puntos")
    x_url = 130
    lz.escribir(
        x_url,
        69,
        url_pre + run_url,
        tipo="url",
        tam=14,
        color=(40, 40, 45),
        etiquetas={"contiene": "rut"},
    )
    lz.escribir(
        x_url + lz.ancho_texto(url_pre, "sans", 14),
        69,
        run_url,
        tipo="rut",
        tam=14,
        color=(40, 40, 45),
        nivel="estres",
        etiquetas={"formato": "sin_puntos", "dv_valido": p.rut_dv_valido, "en_url": True},
    )

    # Encabezado del sitio
    d.rectangle((0, 86, W, 146), fill=(28, 58, 105))
    lz.escribir(40, 124, "Portal de Trámites (ficticio)", nombre_fuente="sans_negrita", tam=20, color=(255, 255, 255))
    for i, m in enumerate(("Inicio", "Mis solicitudes", "Mis datos", "Cerrar sesión")):
        lz.escribir(760 + 140 * i, 122, m, tam=14, color=(220, 230, 245))

    # Tarjeta de perfil con foto
    d.rounded_rectangle((40, 176, 330, 520), radius=10, fill=(255, 255, 255), outline=(215, 218, 224))
    (r,) = rostros.tomar("frente", 1)
    ys = [q[1] for q in r.caja]
    ancho_foto = int(round(r.img.width * 120 / (max(ys) - min(ys))))  # cara de ~120 px de alto
    alto_foto = round(r.img.height * ancho_foto / r.img.width)
    xf = 185 - ancho_foto // 2
    d.rectangle((xf - 4, 196, xf + ancho_foto + 4, 200 + alto_foto + 4), fill=(225, 228, 234))
    lz.pegar(r.img, xf, 200, ancho=ancho_foto, elementos=ProveedorRostros.elementos(r, "base", {"origen": "perfil"}))
    yb = 200 + alto_foto + 38
    w_nombre = ancho_segs(lz, [nombre(p)], 15, "sans_negrita")
    tam_nombre = 15 if w_nombre <= 270 else int(15 * 270 / w_nombre)
    escribir_segs(lz, 185 - ancho_segs(lz, [nombre(p)], tam_nombre, "sans_negrita") / 2, yb, [nombre(p)],
                  tam=tam_nombre, nombre_fuente="sans_negrita")  # fmt: skip
    lz.escribir(185 - lz.ancho_texto("Cambiar foto", "sans", 13) / 2, yb + 28, "Cambiar foto", tam=13,
                color=(30, 100, 200))  # fmt: skip

    # Formulario
    d.rounded_rectangle((360, 176, 1326, 700), radius=10, fill=(255, 255, 255), outline=(215, 218, 224))
    lz.escribir(390, 218, "Mis datos personales", nombre_fuente="sans_negrita", tam=22)
    lz.escribir(390, 244, "Revise y actualice su información de contacto.", tam=14, color=gris)

    def campo(x: float, y: float, ancho: float, etiqueta: str, segs: list[Seg], flecha: bool = False) -> None:
        lz.escribir(x, y, etiqueta, nombre_fuente="sans_negrita", tam=13, color=(60, 65, 75))
        d.rounded_rectangle((x, y + 10, x + ancho, y + 50), radius=5, fill=(255, 255, 255), outline=(190, 195, 205))
        escribir_segs(lz, x + 12, y + 36, segs, tam=15)
        if flecha:
            d.polygon([(x + ancho - 26, y + 26), (x + ancho - 14, y + 26), (x + ancho - 20, y + 34)], fill=gris)

    fmt_correo = _elegir(rng, CORREOS_BASE)
    fmt_tel = _formato_movil(rng)
    campo(390, 290, 906, "Nombre completo", [nombre(p)])
    campo(390, 372, 440, "RUT", [rut(p, "puntos")])
    campo(856, 372, 440, "Fecha de nacimiento", [t(p.nacimiento, sensible=True, campo="fecha_nacimiento")])
    campo(390, 454, 440, "Correo electrónico", [correo(p, fmt_correo)])
    campo(856, 454, 440, "Teléfono", [telefono(p.telefono, fmt_tel)])
    campo(390, 536, 620, "Dirección", [direccion(p)])
    campo(1036, 536, 260, "Comuna", [t(p.comuna, campo="comuna")], flecha=True)
    d.rectangle((390, 612, 408, 630), fill=(30, 100, 200))
    d.line([(394, 621), (398, 626), (405, 615)], fill=(255, 255, 255), width=2)
    lz.escribir(418, 627, "Deseo recibir notificaciones por correo electrónico", tam=14)
    d.rounded_rectangle((1086, 640, 1296, 684), radius=6, fill=(30, 100, 200))
    lz.escribir(1116, 668, "Guardar cambios", nombre_fuente="sans_negrita", tam=15, color=(255, 255, 255))
    d.rounded_rectangle((960, 640, 1070, 684), radius=6, fill=(255, 255, 255), outline=(190, 195, 205))
    lz.escribir(982, 668, "Cancelar", tam=15)
    lz.escribir(40, 744, "Portal ficticio para pruebas de software. Ninguna persona es real.", tam=12, color=gris)
    return lz


# ---------------------------------------------------------------------------
# 5. Diálogo pequeño de propiedades
# ---------------------------------------------------------------------------


def _dialogo(f: Ficticios, rng: np.random.Generator) -> Lienzo:
    W, H = 600, 300
    lz = Lienzo.nuevo(W, H, (240, 240, 240))
    d = ImageDraw.Draw(lz.img)
    p = f.persona()
    tam = 11
    d.rectangle((0, 0, W, 26), fill=(252, 252, 252))
    d.line((0, 26, W, 26), fill=(205, 205, 205))
    lz.escribir(10, 18, "Propiedades del documento", nombre_fuente="sans_negrita", tam=12)
    lz.escribir(W - 22, 18, "x", nombre_fuente="sans", tam=13, color=(80, 80, 80))
    x = 10
    for i, pestana in enumerate(("General", "Descripción", "Seguridad", "Fuentes")):
        w = lz.ancho_texto(pestana, "sans", tam) + 20
        if i == 1:
            d.rectangle((x, 32, x + w, 52), fill=(255, 255, 255), outline=(200, 200, 200))
        lz.escribir(x + 10, 47, pestana, tam=tam)
        x += w + 2
    d.rectangle((10, 52, W - 10, 254), fill=(255, 255, 255), outline=(200, 200, 200))
    filas: list[tuple[str, list[Seg]]] = [
        ("Archivo:", [t("informe_avance_septiembre.pdf")]),
        ("Título:", [t("Informe de avance mensual")]),
        ("Autor:", [nombre(p)]),
        ("Correo del autor:", [correo(p, _elegir(rng, CORREOS_BASE))]),
        ("Asunto:", [t("Programa de fomento productivo")]),
        ("Palabras clave:", [t("informe, avance, honorarios")]),
        ("Creado:", [Seg(f.fecha(), etiquetas={"senuelo": "fecha"}), t(" 10:14")]),
        ("Modificado:", [Seg(f.fecha(), etiquetas={"senuelo": "fecha"}), t(" 16:51")]),
        ("Aplicación:", [t("Procesador de texto 7.2")]),
    ]
    for i, (etq, segs) in enumerate(filas):
        yb = 74 + i * 20
        lz.escribir(150 - lz.ancho_texto(etq, "sans", tam), yb, etq, tam=tam, color=(90, 90, 90))
        escribir_segs(lz, 160, yb, segs, tam=tam)
    for texto, x0 in (("Aceptar", 400), ("Cancelar", 494)):
        d.rectangle((x0, 264, x0 + 86, 290), fill=(250, 250, 250), outline=(170, 170, 170))
        lz.escribir(x0 + 43 - lz.ancho_texto(texto, "sans", tam) / 2, 281, texto, tam=tam)
    # Un diálogo tan pequeño se considera de estrés para todos sus datos
    _a_estres(lz.elementos)
    for e in lz.elementos:
        e.etiquetas["texto_pequeno"] = True
    return lz


# ---------------------------------------------------------------------------
# Generador
# ---------------------------------------------------------------------------


def generar(ctx: Contexto) -> list[Archivo]:
    f = ctx.ficticios(MODULO)
    rng = ctx.rng(MODULO)
    archivos: list[Archivo] = []

    def guardar(
        construir: Callable[[], Lienzo],
        ruta: str,
        id: str,
        formato: str,
        descripcion: str,
        opciones: dict[str, Any] | None = None,
        post: Callable[[Lienzo], Lienzo] | None = None,
        **etiquetas: Any,
    ) -> None:
        lz = construir()
        if post is not None:
            lz = post(lz)
        guardar_imagen(lz.img, ctx.ruta(ruta), formato, **(opciones or {}))
        archivos.append(
            archivo_imagen(
                id=id,
                ruta=ruta,
                formato=formato,
                categoria=CATEGORIA,
                descripcion=descripcion,
                lienzo=lz,
                etiquetas=etiquetas,
            )
        )

    guardar(
        lambda: _correo_escritorio(f, rng, 1.0),
        "pantallazos/correo_escritorio.png",
        "pant_correo_escritorio",
        "png",
        "Cliente de correo 1920x1080: lista de remitentes, cabecera De/Para/CC y cuerpo con RUT y teléfono.",
        escala=1.0,
    )
    guardar(
        lambda: _chat_movil(f, rng, ctx.rostros),
        "pantallazos/chat_movil.jpg",
        "pant_chat_movil",
        "jpg",
        "Chat de celular 1080x2340 (JPEG 75) con RUT, teléfono, correo y dirección en burbujas; avatar redondo.",
        opciones={"quality": 75},
    )
    guardar(
        lambda: _planilla(f, rng),
        "pantallazos/planilla.png",
        "pant_planilla",
        "png",
        "Planilla 1600x900 con 12 filas Nombre | RUT | Correo | Teléfono | Comuna | Monto (texto de 12-13 px).",
    )

    def reducir(lz: Lienzo) -> Lienzo:
        chico = lz.escalar(0.6)
        _a_estres(chico.elementos)
        for e in chico.elementos:
            e.etiquetas["escala"] = 0.6
            e.etiquetas["tam_px"] = round(e.etiquetas.get("tam_px", 13) * 0.6, 1)
        return chico

    guardar(
        lambda: _planilla(f, rng),
        "pantallazos/planilla_reducida.png",
        "pant_planilla_reducida",
        "png",
        "La misma planilla (con otras personas) reducida al 60 %: texto de unos 8 px.",
        post=reducir,
        escala=0.6,
    )
    guardar(
        lambda: _formulario_web(f, rng, ctx.rostros),
        "pantallazos/formulario_web.webp",
        "pant_formulario_web",
        "webp",
        "Formulario web 1366x768 con campos llenos, foto de perfil y el RUN en la URL.",
        opciones={"quality": 90},
    )
    guardar(
        lambda: _dialogo(f, rng),
        "pantallazos/dialogo_pequeno.png",
        "pant_dialogo_pequeno",
        "png",
        "Ventana 'Propiedades del documento' 600x300 con autor y correo en texto de 11 px.",
    )
    guardar(
        lambda: _correo_escritorio(f, rng, 1.5),
        "pantallazos/correo_escritorio_hidpi.png",
        "pant_correo_escritorio_hidpi",
        "png",
        "El cliente de correo redibujado al 150 % (2880x1620), con otras personas.",
        escala=1.5,
    )
    return archivos
