"""Reglas de contexto del prototipo: etiquetas, columnas de tablas, nombres completos, URL y montos.

Se aplican sobre "líneas" con caja (capa de texto de un PDF o líneas del OCR), en cualquier
unidad (puntos o píxeles), siempre que todas las líneas de una página usen la misma.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from banco_pruebas.ficticios import NOMBRES_F, NOMBRES_M


@dataclass
class Linea:
    texto: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def alto(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c)).casefold().strip()


# ---------------------------------------------------------------------------
# Etiquetas de campos personales
# ---------------------------------------------------------------------------

_ETQ_COLUMNA = (
    r"nombres?(?:\s+y\s+apellidos?)?|nombre\s+completo|apellidos?|rut|r\.?u\.?t\.?|run|c\.?\s?i\.?|cedula(?:\s+de\s+identidad)?"
    r"|correo(?:\s+electronico)?|e-?\s?mail|mail|telefono|fono|celular|movil|direccion|domicilio|firma(?:\s+del\s+titular)?"
)
_ETQ_SOLO_CON_DOS_PUNTOS = r"de|para|cc|cco|remitente|destinatario|asistente|solicitante|funcionari[oa]|prestador(?:a)?"
ETIQUETA_SOLA = re.compile(rf"^\s*(?:{_ETQ_COLUMNA})\s*:?\s*$")
ETIQUETA_VALOR = re.compile(rf"^\s*(?:{_ETQ_COLUMNA}|{_ETQ_SOLO_CON_DOS_PUNTOS})\s*:\s*(\S.*)$")
_ES_FIRMA = re.compile(r"^\s*firma")


def _tipo_por_etiqueta(etq: str) -> str:
    e = _norm(etq)
    if re.match(r"(rut|r\.?u\.?t|run|c\.?\s?i|cedula)", e):
        return "rut"
    if re.match(r"(correo|e-?\s?mail|mail)", e):
        return "correo"
    if re.match(r"(telefono|fono|celular|movil)", e):
        return "telefono"
    if re.match(r"(direccion|domicilio)", e):
        return "direccion"
    if _ES_FIRMA.match(e):
        return "firma"
    return "nombre"


def reglas_de_contexto(
    lineas: list[Linea], ancho_pagina: float, alto_pagina: float
) -> tuple[list[tuple[str, int, int, int]], list[tuple[str, float, float, float, float]]]:
    """Devuelve (tramos, rectángulos).

    tramos: (tipo, índice de línea, inicio, fin) dentro del texto de esa línea.
    rectángulos: (tipo, x0, y0, x1, y1) para zonas sin texto legible (columna de firmas).
    """
    tramos: list[tuple[str, int, int, int]] = []
    rects: list[tuple[str, float, float, float, float]] = []
    norm = [_norm(ln.texto) for ln in lineas]

    # 1. "Etiqueta: valor" en la misma línea
    for i, n in enumerate(norm):
        m = ETIQUETA_VALOR.match(n)
        if m and not ETIQUETA_SOLA.match(m.group(1)):
            etiqueta = n[: m.start(1)]
            tramos.append((_tipo_por_etiqueta(etiqueta), i, m.start(1), len(lineas[i].texto)))

    # 2. Etiquetas solas: encabezado de columna (fila con varios encabezados) o celda de formulario
    solas = [i for i, n in enumerate(norm) if ETIQUETA_SOLA.match(n)]
    for i in solas:
        h = lineas[i]
        misma_fila = [
            j
            for j, ln in enumerate(lineas)
            if j != i
            and _solapa_vertical(h, ln) > 0.4
            and len(norm[j].split()) <= 4
            and ln.x1 - ln.x0 < ancho_pagina * 0.5
        ]
        tipo = _tipo_por_etiqueta(norm[i])
        if len(misma_fila) >= 2:  # encabezado de tabla
            derecha = [lineas[j].x0 for j in misma_fila if lineas[j].x0 > h.x1]
            izquierda = [lineas[j].x1 for j in misma_fila if lineas[j].x1 < h.x0]
            x_ini = (max(izquierda) + h.x0) / 2 if izquierda else h.x0 - h.alto
            x_fin = (min(derecha) + h.x1) / 2 if derecha else min(ancho_pagina, h.x1 + 1.5 * (h.x1 - h.x0))
            celdas = _celdas_de_columna(lineas, h, x_ini, x_fin, ancho_pagina, alto_pagina)
            for j in celdas:
                tramos.append((tipo, j, 0, len(lineas[j].texto)))
            fondo = max((lineas[j].y1 for j in celdas), default=h.y1 + 6 * h.alto)
            if tipo == "firma" or not celdas:
                rects.append((tipo, x_ini, h.y1, x_fin, fondo + h.alto))
        else:  # celda de formulario: el valor está a la derecha en la misma fila
            candidatas = [
                j
                for j, ln in enumerate(lineas)
                if j != i and _solapa_vertical(h, ln) > 0.4 and ln.x0 >= h.x1 - 2 and ln.x0 - h.x1 < ancho_pagina * 0.45
            ]
            if candidatas:
                j = min(candidatas, key=lambda k: lineas[k].x0)
                if not ETIQUETA_SOLA.match(norm[j]):
                    tramos.append((tipo, j, 0, len(lineas[j].texto)))
    return tramos, rects


def _solapa_vertical(a: Linea, b: Linea) -> float:
    inter = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(0.0, inter) / max(1e-6, min(a.alto, b.alto))


def _celdas_de_columna(
    lineas: list[Linea], h: Linea, x_ini: float, x_fin: float, ancho: float, alto: float
) -> list[int]:
    """Líneas bajo el encabezado ``h`` dentro de la columna, hasta que la tabla termina."""
    debajo = sorted(
        (j for j, ln in enumerate(lineas) if ln.y0 > h.y1 - 1 and x_ini - 2 <= ln.cx <= x_fin + 2),
        key=lambda j: lineas[j].y0,
    )
    celdas: list[int] = []
    ultimo = h.y1
    salto_max = max(4 * h.alto, alto * 0.08)
    for j in debajo:
        ln = lineas[j]
        if ln.x1 - ln.x0 > ancho * 0.55 or ln.y0 - ultimo > salto_max:
            break
        if ETIQUETA_SOLA.match(_norm(ln.texto)):
            break
        celdas.append(j)
        ultimo = ln.y1
    return celdas


# ---------------------------------------------------------------------------
# Nombres completos
# ---------------------------------------------------------------------------

_EXTRA_NOMBRES = """
alicia andrea angela angelica alejandra alejandro alberto alfredo alvaro amanda ana andres antonio arturo barbara
beatriz bernardita blanca bruno camila carla carlos carmen carolina catalina cecilia cesar claudia claudio constanza
cristian cristina daniel daniela david denisse diego eduardo elena elizabeth emilia enrique ernesto esteban eugenia
eva fabian felipe fernanda fernando francisca francisco gabriel gabriela gloria gonzalo graciela guillermo gustavo
hector hernan hugo ignacia ignacio isabel ivan ivonne jaime javier javiera jessica joaquin jorge jose josefa juan
julia julio karen karina katherine laura leonardo lorena luis luisa manuel marcela marcelo marco margarita maria
mariana mario marta martin matias mauricio maximiliano miguel monica natalia nelson nicolas olga oscar pablo paola
patricia patricio paula paulina pedro rafael ramon raul rebeca ricardo roberto rocio rodrigo rosa rosario ruben
sandra sara sebastian sergio silvia sofia sonia susana tamara teresa tomas valentina valeria vanessa veronica
victor victoria viviana ximena yasna yolanda
"""
NOMBRES_PILA = frozenset(
    {_norm(p) for n in [*NOMBRES_F, *NOMBRES_M] for p in n.split()} | {p for p in _EXTRA_NOMBRES.split()}
)
_PALABRA_NOMBRE = re.compile(r"[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+")
_NO_NOMBRE = frozenset(
    _norm(p)
    for p in """
firma electronica avanzada funcionaria funcionario jefe jefa director directora departamento unidad gobierno regional
santiago region chile informe anexo acta fecha nombre institucion correo telefono division seccion gabinete asesor
asesora coordinador coordinadora encargado encargada cargo profesional servicio municipal municipalidad estadio
""".split()
)
_HONORIFICO = re.compile(r"(?i)\b(?:don|doña|sr\.?|sra\.?|srta\.?|señor|señora)\s+")


def nombres_por_diccionario(texto: str) -> list[tuple[int, int]]:
    """Secuencias de 2 a 5 palabras con mayúscula que empiezan con un nombre de pila conocido.

    Se aplica en líneas cortas (firmas, celdas, encabezados de correo) y después de "don/doña/Sr./Sra.".
    """
    tramos = []
    for linea in _lineas_con_offset(texto):
        inicio, contenido = linea
        palabras = list(_PALABRA_NOMBRE.finditer(contenido))
        corta = len(contenido.split()) <= 7
        honorificos = [m.end() for m in _HONORIFICO.finditer(contenido)]
        i = 0
        while i < len(palabras):
            p = palabras[i]
            permitido = corta or p.start() in honorificos
            if permitido and _norm(p.group(0)) in NOMBRES_PILA:
                j = i
                while (
                    j + 1 < len(palabras)
                    and contenido[palabras[j].end() : palabras[j + 1].start()].strip() == ""
                    and _norm(palabras[j + 1].group(0)) not in _NO_NOMBRE
                    and j + 1 - i < 5
                ):
                    j += 1
                if j > i or p.start() in honorificos:
                    tramos.append((inicio + p.start(), inicio + palabras[j].end()))
                i = j + 1
            else:
                i += 1
    return tramos


def expandir_nombre(texto: str, a: int, b: int) -> tuple[int, int]:
    """Amplía un nombre de la lista a las palabras con mayúscula vecinas de la misma línea (nombres de pila)."""
    for _ in range(4):
        m = re.search(r"([A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+)[ \t]+$", texto[max(0, a - 40) : a])
        if not m or _norm(m.group(1)) in _NO_NOMBRE:
            break
        a = a - (len(texto[max(0, a - 40) : a]) - m.start(1))
    for _ in range(4):
        m = re.match(r"[ \t]+([A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+)", texto[b : b + 40])
        if not m or _norm(m.group(1)) in _NO_NOMBRE:
            break
        b = b + m.end()
    return a, b


def _lineas_con_offset(texto: str) -> list[tuple[int, str]]:
    salida, pos = [], 0
    for linea in texto.split("\n"):
        salida.append((pos, linea))
        pos += len(linea) + 1
    return salida


# ---------------------------------------------------------------------------
# URL y montos
# ---------------------------------------------------------------------------

URL_PERSONAL = re.compile(
    r"(?i)(facebook|instagram|linkedin|twitter\.|//x\.com|tiktok|wa\.me|whatsapp|teams\.microsoft|zoom\.us|meet\.google"
    r"|drive\.google|docs\.google|dropbox|onedrive|1drv\.ms|sharepoint|wetransfer|forms\.gle|calendly)"
    r"|[?&](rut|run|id|token|key|pwd|p|user|usuario|email|mail|correo)="
)
_CONTINUACION_URL = re.compile(r"\n([\w\-./%?=&#~+:]+)(?=[ \t]*(?:\n|$))")


def completar_url(texto: str, a: int, b: int) -> tuple[int, int]:
    """Si la URL sigue en la línea siguiente (corte de línea del PDF), incluye ese trozo."""
    m = _CONTINUACION_URL.match(texto, b)
    if m and ("/" in m.group(1) or "-" in m.group(1)) and " " not in m.group(1):
        return a, m.end(1)
    return a, b


def es_monto(texto: str, a: int, b: int) -> bool:
    """Números de dinero o con agrupación de miles: no son teléfonos ni RUT."""
    antes = texto[max(0, a - 4) : a]
    despues = texto[b : b + 12].lower()
    valor = texto[a:b].strip()
    if "$" in antes or "us$" in antes.lower() or "clp" in antes.lower():
        return True
    if re.match(r"\s*(millones|mil\b|pesos|uf\b|%|usd)", despues):
        return True
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?", valor))
