"""Context rules: "label: value" pairs, form cells and table columns of personal fields.

They are applied to "lines" with a box (text layer of a PDF or OCR lines), in any unit (points
or pixels), as long as every line of a page uses the same one.

The regular expressions match Spanish document text (labels such as "Nombre", "Correo",
"Teléfono"), so their content stays in Spanish.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anonymizer.engine.patterns import norm


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


_COLUMN_LABELS = (
    r"nombres?(?:\s+y\s+apellidos?)?|nombre\s+completo|apellidos?|rut|r\.?u\.?t\.?|run|c\.?\s?i\.?|cedula(?:\s+de\s+identidad)?"
    r"|correo(?:\s+electronico)?|e-?\s?mail|mail|telefono|fono|celular|movil|direccion|domicilio|firma(?:\s+del\s+titular)?"
)
_LABELS_ONLY_WITH_COLON = r"de|para|cc|cco|remitente|destinatario|asistente|solicitante|funcionari[oa]|prestador(?:a)?"
LABEL_ONLY = re.compile(rf"^\s*(?:{_COLUMN_LABELS})\s*:?\s*$")
LABEL_VALUE = re.compile(rf"^\s*(?:{_COLUMN_LABELS}|{_LABELS_ONLY_WITH_COLON})\s*:\s*(\S.*)$")
_IS_SIGNATURE = re.compile(r"^\s*firma")
# Types found only through the "names by context" group (the others belong to the patterns).
NAME_TYPES = ("name", "signature")


def type_by_label(label: str) -> str:
    e = norm(label)
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
    lines: list[Line], page_width: float, page_height: float, names: bool = True
) -> tuple[list[tuple[str, int, int, int]], list[tuple[str, float, float, float, float]]]:
    """Returns (spans, rectangles).

    spans: (type, line index, start, end) inside the text of that line.
    rectangles: (type, x0, y0, x1, y1) for zones without legible text (signature column).
    ``names=False`` leaves out names and signatures (``NAME_TYPES``): RUT, e-mail, phone and
    address columns and labels are still found.
    """
    spans: list[tuple[str, int, int, int]] = []
    rects: list[tuple[str, float, float, float, float]] = []
    normalized = [norm(ln.text) for ln in lines]

    # 1. "Label: value" on the same line
    for i, n in enumerate(normalized):
        m = LABEL_VALUE.match(n)
        if m and not LABEL_ONLY.match(m.group(1)):
            label = n[: m.start(1)]
            spans.append((type_by_label(label), i, m.start(1), len(lines[i].text)))

    # 2. Standalone labels: column header (row with several headers) or form cell
    standalone = [i for i, n in enumerate(normalized) if LABEL_ONLY.match(n)]
    for i in standalone:
        h = lines[i]
        same_row = [
            j
            for j, ln in enumerate(lines)
            if j != i
            and _vertical_overlap(h, ln) > 0.4
            and len(normalized[j].split()) <= 4
            and ln.x1 - ln.x0 < page_width * 0.5
        ]
        type_ = type_by_label(normalized[i])
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
                if not LABEL_ONLY.match(normalized[j]):
                    spans.append((type_, j, 0, len(lines[j].text)))
    if not names:
        spans = [s for s in spans if s[0] not in NAME_TYPES]
        rects = [r for r in rects if r[0] not in NAME_TYPES]
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
        if LABEL_ONLY.match(norm(ln.text)):
            break
        cells.append(j)
        last = ln.y1
    return cells
