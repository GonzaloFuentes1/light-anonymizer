"""Context rules of the prototype: labels, table columns, full names, URLs and amounts.

They are applied to "lines" with a box (text layer of a PDF or OCR lines), in any unit (points
or pixels), as long as every line of a page uses the same one.

The regular expressions match Spanish document text (labels such as "Nombre", "Correo",
"Teléfono"), so their content stays in Spanish.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from test_bench.fake_data import FEMALE_NAMES, MALE_NAMES


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    # False for vertical or tilted lines of a PDF's text layer (their box is not the line's shape).
    horizontal: bool = True

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c)).casefold().strip()


# ---------------------------------------------------------------------------
# Labels of personal fields
# ---------------------------------------------------------------------------

_COLUMN_LABELS = (
    r"nombres?(?:\s+y\s+apellidos?)?|nombre\s+completo|apellidos?|rut|r\.?u\.?t\.?|run|c\.?\s?i\.?|cedula(?:\s+de\s+identidad)?"
    r"|correo(?:\s+electronico)?|e-?\s?mail|mail|telefono|fono|celular|movil|direccion|domicilio|firma(?:\s+del\s+titular)?"
)
_LABELS_ONLY_WITH_COLON = r"de|para|cc|cco|remitente|destinatario|asistente|solicitante|funcionari[oa]|prestador(?:a)?"
LABEL_ONLY = re.compile(rf"^\s*(?:{_COLUMN_LABELS})\s*:?\s*$")
LABEL_VALUE = re.compile(rf"^\s*(?:{_COLUMN_LABELS}|{_LABELS_ONLY_WITH_COLON})\s*:\s*(\S.*)$")
_IS_SIGNATURE = re.compile(r"^\s*firma")


def _type_by_label(label: str) -> str:
    e = _norm(label)
    if re.match(r"(rut|r\.?u\.?t|run|c\.?\s?i|cedula)", e):
        return "rut"
    if re.match(r"(correo|e-?\s?mail|mail)", e):
        return "email"
    if re.match(r"(telefono|fono|celular|movil)", e):
        return "phone"
    if re.match(r"(direccion|domicilio)", e):
        return "address"
    if _IS_SIGNATURE.match(e):
        return "signature"
    return "name"


def context_rules(
    lines: list[Line], page_width: float, page_height: float
) -> tuple[list[tuple[str, int, int, int]], list[tuple[str, float, float, float, float]]]:
    """Returns (spans, rectangles).

    spans: (type, line index, start, end) inside the text of that line.
    rectangles: (type, x0, y0, x1, y1) for zones without legible text (signature column).
    """
    spans: list[tuple[str, int, int, int]] = []
    rects: list[tuple[str, float, float, float, float]] = []
    norm = [_norm(ln.text) for ln in lines]

    # 1. "Label: value" on the same line
    for i, n in enumerate(norm):
        m = LABEL_VALUE.match(n)
        if m and not LABEL_ONLY.match(m.group(1)):
            label = n[: m.start(1)]
            spans.append((_type_by_label(label), i, m.start(1), len(lines[i].text)))

    # 2. Standalone labels: column header (row with several headers) or form cell
    standalone = [i for i, n in enumerate(norm) if LABEL_ONLY.match(n)]
    for i in standalone:
        h = lines[i]
        same_row = [
            j
            for j, ln in enumerate(lines)
            if j != i
            and _vertical_overlap(h, ln) > 0.4
            and len(norm[j].split()) <= 4
            and ln.x1 - ln.x0 < page_width * 0.5
        ]
        type_ = _type_by_label(norm[i])
        if len(same_row) >= 2:  # table header
            right = [lines[j].x0 for j in same_row if lines[j].x0 > h.x1]
            left = [lines[j].x1 for j in same_row if lines[j].x1 < h.x0]
            x_start = (max(left) + h.x0) / 2 if left else h.x0 - h.height
            x_end = (min(right) + h.x1) / 2 if right else min(page_width, h.x1 + 1.5 * (h.x1 - h.x0))
            cells = _column_cells(lines, h, x_start, x_end, page_width, page_height)
            for j in cells:
                spans.append((type_, j, 0, len(lines[j].text)))
            bottom = max((lines[j].y1 for j in cells), default=h.y1 + 6 * h.height)
            if type_ == "signature" or not cells:
                rects.append((type_, x_start, h.y1, x_end, bottom + h.height))
        else:  # form cell: the value is to the right on the same row
            candidates = [
                j
                for j, ln in enumerate(lines)
                if j != i and _vertical_overlap(h, ln) > 0.4 and ln.x0 >= h.x1 - 2 and ln.x0 - h.x1 < page_width * 0.45
            ]
            if candidates:
                j = min(candidates, key=lambda k: lines[k].x0)
                if not LABEL_ONLY.match(norm[j]):
                    spans.append((type_, j, 0, len(lines[j].text)))
    return spans, rects


def _vertical_overlap(a: Line, b: Line) -> float:
    inter = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(0.0, inter) / max(1e-6, min(a.height, b.height))


def _column_cells(lines: list[Line], h: Line, x_start: float, x_end: float, width: float, height: float) -> list[int]:
    """Lines under the header ``h`` inside the column, until the table ends."""
    below = sorted(
        (j for j, ln in enumerate(lines) if ln.y0 > h.y1 - 1 and x_start - 2 <= ln.cx <= x_end + 2),
        key=lambda j: lines[j].y0,
    )
    cells: list[int] = []
    last = h.y1
    max_gap = max(4 * h.height, height * 0.08)
    for j in below:
        ln = lines[j]
        if ln.x1 - ln.x0 > width * 0.55 or ln.y0 - last > max_gap:
            break
        if LABEL_ONLY.match(_norm(ln.text)):
            break
        cells.append(j)
        last = ln.y1
    return cells


# ---------------------------------------------------------------------------
# Full names
# ---------------------------------------------------------------------------

_EXTRA_NAMES = """
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
GIVEN_NAMES = frozenset(
    {_norm(p) for n in [*FEMALE_NAMES, *MALE_NAMES] for p in n.split()} | {p for p in _EXTRA_NAMES.split()}
)
_NAME_WORD = re.compile(r"[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+")
_NOT_NAME = frozenset(
    _norm(p)
    for p in """
firma electronica avanzada funcionaria funcionario jefe jefa director directora departamento unidad gobierno regional
santiago region chile informe anexo acta fecha nombre institucion correo telefono division seccion gabinete asesor
asesora coordinador coordinadora encargado encargada cargo profesional servicio municipal municipalidad estadio
""".split()
)
_HONORIFIC = re.compile(r"(?i)\b(?:don|doña|sr\.?|sra\.?|srta\.?|señor|señora)\s+")


def names_by_dictionary(text: str) -> list[tuple[int, int]]:
    """Sequences of 2 to 5 capitalized words that start with a known given name.

    Applied to short lines (signatures, cells, e-mail headers) and after "don/doña/Sr./Sra.".
    """
    spans = []
    for line in _lines_with_offset(text):
        start, content = line
        words = list(_NAME_WORD.finditer(content))
        short = len(content.split()) <= 7
        honorifics = [m.end() for m in _HONORIFIC.finditer(content)]
        i = 0
        while i < len(words):
            p = words[i]
            allowed = short or p.start() in honorifics
            if allowed and _norm(p.group(0)) in GIVEN_NAMES:
                j = i
                while (
                    j + 1 < len(words)
                    and content[words[j].end() : words[j + 1].start()].strip() == ""
                    and _norm(words[j + 1].group(0)) not in _NOT_NAME
                    and j + 1 - i < 5
                ):
                    j += 1
                if j > i or p.start() in honorifics:
                    spans.append((start + p.start(), start + words[j].end()))
                i = j + 1
            else:
                i += 1
    return spans


def expand_name(text: str, a: int, b: int) -> tuple[int, int]:
    """Extends a name of the list to the neighbouring capitalized words of the same line (given names)."""
    for _ in range(4):
        m = re.search(r"([A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+)[ \t]+$", text[max(0, a - 40) : a])
        if not m or _norm(m.group(1)) in _NOT_NAME:
            break
        a = a - (len(text[max(0, a - 40) : a]) - m.start(1))
    for _ in range(4):
        m = re.match(r"[ \t]+([A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+)", text[b : b + 40])
        if not m or _norm(m.group(1)) in _NOT_NAME:
            break
        b = b + m.end()
    return a, b


def _lines_with_offset(text: str) -> list[tuple[int, str]]:
    output, pos = [], 0
    for line in text.split("\n"):
        output.append((pos, line))
        pos += len(line) + 1
    return output


# ---------------------------------------------------------------------------
# URLs and amounts
# ---------------------------------------------------------------------------

PERSONAL_URL = re.compile(
    r"(?i)(facebook|instagram|linkedin|twitter\.|//x\.com|tiktok|wa\.me|whatsapp|teams\.microsoft|zoom\.us|meet\.google"
    r"|drive\.google|docs\.google|dropbox|onedrive|1drv\.ms|sharepoint|wetransfer|forms\.gle|calendly)"
    r"|[?&](rut|run|id|token|key|pwd|p|user|usuario|email|mail|correo)="
)
_URL_CONTINUATION = re.compile(r"\n([\w\-./%?=&#~+:]+)(?=[ \t]*(?:\n|$))")


def complete_url(text: str, a: int, b: int) -> tuple[int, int]:
    """If the URL continues on the next line (PDF line break), includes that piece."""
    m = _URL_CONTINUATION.match(text, b)
    if m and ("/" in m.group(1) or "-" in m.group(1)) and " " not in m.group(1):
        return a, m.end(1)
    return a, b


def is_amount(text: str, a: int, b: int) -> bool:
    """Money figures or numbers with thousands grouping: they are neither phones nor RUTs."""
    before = text[max(0, a - 4) : a]
    after = text[b : b + 12].lower()
    value = text[a:b].strip()
    if "$" in before or "us$" in before.lower() or "clp" in before.lower():
        return True
    if re.match(r"\s*(millones|mil\b|pesos|uf\b|%|usd)", after):
        return True
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?", value))
