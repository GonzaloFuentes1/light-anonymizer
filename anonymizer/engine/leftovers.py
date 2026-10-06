"""What a page's redaction may have left under its zones (decided 2026-10-06: "if unsure, that
page is exported as an image").

MuPDF's redaction, the letters' rectangles of D8 (``vectors.cover``) and the removal of stroked
paths (``strokes.remove``) take out of the file what lies under a zone whenever they can do it
with certainty. What they cannot (a letter a zone cuts, a stroke that crosses the edge of a zone or
that the content filter could not match, a shape painted with a pattern, a smooth shading, a stroke
that is also a clip) would stay in the file under the black box. ``check`` looks at the page after
its redaction and says whether anything of that kind is left; ``pdf.redact_page`` then exports the
page as an image (``pdf.rasterize``), which removes it for certain.

Everything drawn is suspect except page layout: a straight horizontal or vertical line stroked
no wider than ``LAYOUT_WIDTH`` (a table rule, an underline) and an axis-aligned rectangle, filled
or stroked no wider than that (a frame, a cell, a band, the black boxes of the redaction), unless
the rectangle lies whole inside a zone and is not black: then it is a mark the zone should have
removed (a QR module, a barcode bar). Any other path, clip or not, whose outline passes inside a
zone (a fill's edge inside the zone, a stroke whose painted width reaches it) is a leftover; the
inside of a filled shape that holds the zone whole, with no edge under it, hides nothing. Painted
paths that ``get_drawings`` does not list (shapes filled or stroked with a pattern) and smooth
shadings that touch a zone are leftovers too: they are found with MuPDF's content filter, like in
``strokes``. A page that takes longer than ``SECONDS`` to check is a leftover as well.

Coordinates: PyMuPDF's unrotated page space; call with ``PDF_LOCK`` held, on an unrotated page.
"""

from __future__ import annotations

import time

import numpy as np
import pymupdf

from anonymizer.engine import strokes

LAYOUT_WIDTH = 3.0  # points: a straight stroke up to this wide is page layout (a rule, a frame edge)
EDGE = 0.05  # points: an outline this close to a zone's edge (the black box's own edge) is not inside it
SECONDS = 15.0  # time budget of the check of one page
_AXIS = 0.5  # points: a line whose ends differ by less than this across is horizontal or vertical
_BLACK = (0.0, 0.0, 0.0)

# Why a page is exported as an image (Spanish: shown to the user and written in the audit report).
REASONS = {
    "shape": "letras o formas dibujadas bajo una zona no se pudieron quitar con certeza",
    "stroke": "un trazo dibujado bajo una zona no se pudo quitar con certeza",
    "clip": "un trazado de recorte bajo una zona no se pudo quitar con certeza",
    "pattern": "una trama o un degradado bajo una zona no se pudo quitar con certeza",
    "time": "la revisión de los dibujos de la página no terminó a tiempo",
}
_ORDER = tuple(REASONS)


def reasons_text(codes: list[str]) -> str:
    """The Spanish reasons of ``codes``, joined (``REASONS``)."""
    return "; ".join(REASONS[c] for c in _ORDER if c in codes)


class _OnlyShadings(pymupdf.mupdf.PdfSanitizeFilterOptions2):
    """MuPDF's culler callback that drops everything painted but smooth shadings (clips stay: they
    bound what a shading paints)."""

    _DROP = (
        pymupdf.mupdf.FZ_CULL_PATH_FILL,
        pymupdf.mupdf.FZ_CULL_PATH_STROKE,
        pymupdf.mupdf.FZ_CULL_PATH_FILL_STROKE,
        pymupdf.mupdf.FZ_CULL_GLYPH,
        pymupdf.mupdf.FZ_CULL_IMAGE,
    )

    def __init__(self):
        super().__init__()
        self.use_virtual_culler()

    def culler(self, ctx, bbox, kind):  # noqa: ARG002 - MuPDF's callback signature
        return 1 if kind in self._DROP else 0


def _shading_under(page: pymupdf.Page, zones: list[pymupdf.Rect]) -> bool:
    """A smooth shading (the ``sh`` operator) paints inside one of ``zones``. The content filter
    does not say where a shading paints (it fills the clip in effect), so a copy of the page with
    everything else dropped is rendered and looked at inside the zones."""
    with pymupdf.open() as tmp:
        tmp.insert_pdf(page.parent, from_page=page.number, to_page=page.number, links=False, annots=False)
        alone = tmp[0]
        strokes._filter(alone, _OnlyShadings(), update=True)
        for z in zones:
            inner = z + (EDGE, EDGE, -EDGE, -EDGE)
            if inner.is_empty:
                continue
            pix = alone.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=inner, alpha=False, annots=False)
            if pix.width and pix.height:
                a = np.frombuffer(pix.samples, np.uint8)
                if int(a.min()) < 250:
                    return True
    return False


def _cubic(p0, p1, p2, p3) -> np.ndarray:
    """Points along a cubic Bézier curve, about one per point of its control polygon's length."""
    c = np.array([[p.x, p.y] for p in (p0, p1, p2, p3)], np.float64)
    n = int(min(256, max(8, np.hypot(*np.diff(c, axis=0).T).sum())))
    t = np.linspace(0, 1, n + 1)[:, None]
    return (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t**2 * c[2] + t**3 * c[3]


def subpaths(items) -> list[tuple[str, np.ndarray]]:
    """The subpaths of a path's items: ("box", its 4 corners) for a rectangle or a quad, ("line",
    points) for a run of connected lines and curves (curves flattened), as MuPDF paints them."""
    out: list[tuple[str, list]] = []
    end = None
    for item in items:
        kind = item[0]
        if kind == "re":
            r = pymupdf.Rect(item[1])
            out.append(("box", [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]))
            end = None
            continue
        if kind == "qu":
            q = item[1]
            out.append(("quad", [(p.x, p.y) for p in (q.ul, q.ur, q.lr, q.ll, q.ul)]))
            end = None
            continue
        if kind == "l":
            points = [(item[1].x, item[1].y), (item[2].x, item[2].y)]
        elif kind == "c":
            points = [tuple(p) for p in _cubic(*item[1:5])]
        else:
            continue
        start = points[0]
        if end is None or abs(start[0] - end[0]) > 1e-3 or abs(start[1] - end[1]) > 1e-3:
            out.append(("line", [start]))
        out[-1][1].extend(points[1:])
        end = points[-1]
    return [(kind, np.asarray(pts, np.float64)) for kind, pts in out]


def _axis_box(kind: str, pts: np.ndarray) -> bool:
    """A rectangle with horizontal and vertical sides (a "re", or a quad or a closed run of four
    lines that is one)."""
    if kind == "box":
        return True
    if len(pts) != 5 or np.abs(pts[0] - pts[-1]).max() > 1e-3:
        return False
    d = np.diff(pts, axis=0)
    return bool(np.all(np.minimum(np.abs(d[:, 0]), np.abs(d[:, 1])) < _AXIS))


def _rule(kind: str, pts: np.ndarray) -> bool:
    """A single straight horizontal or vertical line."""
    if kind != "line" or len(pts) != 2:
        return False
    dx, dy = np.abs(pts[1] - pts[0])
    return bool(min(dx, dy) < _AXIS)


def _crosses(pts: np.ndarray, zone: tuple[float, float, float, float]) -> bool:
    """Some segment of the polyline ``pts`` passes inside the open rectangle ``zone``
    (Liang-Barsky clipping of every segment at once)."""
    x0, y0, x1, y1 = zone
    if x1 <= x0 or y1 <= y0 or len(pts) < 2:
        return False
    p, q = pts[:-1], pts[1:]
    d = q - p
    t0 = np.zeros(len(p))
    t1 = np.ones(len(p))
    ok = np.ones(len(p), bool)
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        dd, pp = d[:, axis], p[:, axis]
        flat = np.abs(dd) < 1e-12
        ok &= ~flat | ((pp > lo) & (pp < hi))
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (lo - pp) / dd
            b = (hi - pp) / dd
        enter = np.where(flat, -np.inf, np.minimum(a, b))
        leave = np.where(flat, np.inf, np.maximum(a, b))
        t0 = np.maximum(t0, enter)
        t1 = np.minimum(t1, leave)
    return bool(np.any(ok & (t0 < t1)))


def _inside(pts: np.ndarray, zone: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = zone
    return bool(pts[:, 0].min() >= x0 and pts[:, 1].min() >= y0 and pts[:, 0].max() <= x1 and pts[:, 1].max() <= y1)


def _path_reason(d: dict, zones: np.ndarray) -> str | None:
    """Why the path ``d`` (``get_drawings(extended=True)``) is a leftover under one of ``zones``
    (rows x0, y0, x1, y1), or None."""
    kind = d.get("type") or ""
    stroked = "s" in kind
    width = float(d.get("width") or 0) if stroked else 0.0
    reach = max(width, 0.5) / 2 if stroked else 0.0
    parts = subpaths(d.get("items") or ())
    if not parts:
        return None
    every = np.concatenate([pts for _, pts in parts])
    lo, hi = every.min(axis=0) - reach, every.max(axis=0) + reach
    near = zones[(zones[:, 0] < hi[0]) & (lo[0] < zones[:, 2]) & (zones[:, 1] < hi[1]) & (lo[1] < zones[:, 3])]
    if not len(near):
        return None
    why = "clip" if kind == "clip" else ("stroke" if kind == "s" else "shape")
    # The black boxes of the redaction (MuPDF paints them filled and stroked in black).
    black = tuple(d.get("fill") or ()) == _BLACK and (not stroked or tuple(d.get("color") or ()) == _BLACK)
    for sub_kind, pts in parts:
        rule, box = _rule(sub_kind, pts), _axis_box(sub_kind, pts)
        if (rule or box) and width <= LAYOUT_WIDTH:  # page layout
            if kind == "clip" or black or (rule and not stroked):  # a rectangular clip, a black box, nothing painted
                continue
            if any(_inside(pts, (z[0] - EDGE, z[1] - EDGE, z[2] + EDGE, z[3] + EDGE)) for z in near):
                return why  # whole inside a zone: a mark the zone should have removed
            continue
        for z in near:
            # A stroke paints half its width on each side of the outline; a fill only up to it.
            if _crosses(pts, (z[0] - reach + EDGE, z[1] - reach + EDGE, z[2] + reach - EDGE, z[3] + reach - EDGE)):
                return why
    return None


def _boxes(drawings: list[dict]) -> np.ndarray:
    """A rough box of every drawing (rows x0, y0, x1, y1), grown by its stroke: a clip's from its
    items, a group's empty."""
    out = np.full((len(drawings), 4), np.nan)
    for k, d in enumerate(drawings):
        kind = d.get("type") or ""
        if kind == "group":
            continue
        if "rect" in d:
            r = d["rect"]
            grow = max(float(d.get("width") or 0), 0.5) / 2 if "s" in kind else 0.0
            out[k] = (r.x0 - grow, r.y0 - grow, r.x1 + grow, r.y1 + grow)
            continue
        points = []
        for item in d.get("items") or ():
            if item[0] in ("re", "qu"):
                r = pymupdf.Rect(item[1]) if item[0] == "re" else item[1].rect
                points += [(r.x0, r.y0), (r.x1, r.y1)]
            else:
                points += [(p.x, p.y) for p in item[1:] if isinstance(p, pymupdf.Point)]
        if points:
            a = np.asarray(points, np.float64)
            out[k] = (*a.min(axis=0), *a.max(axis=0))
    return out


def check(page: pymupdf.Page, zones: list[pymupdf.Rect], deadline: float | None = None) -> list[str]:
    """Reasons (keys of ``REASONS``) why what is left on ``page`` after its redaction may still hold
    something under ``zones``; empty when nothing is. Call on the unrotated page, after its
    redaction, with ``PDF_LOCK`` held. Past ``deadline`` (monotonic; ``SECONDS`` from now by
    default) the answer is ["time"]."""
    zones = [pymupdf.Rect(z) for z in zones if not pymupdf.Rect(z).is_empty]
    if not zones:
        return []
    deadline = time.monotonic() + SECONDS if deadline is None else deadline
    table = np.array([[z.x0, z.y0, z.x1, z.y1] for z in zones], np.float64)
    found: set[str] = set()
    try:
        drawings = page.get_drawings(extended=True)
        strokes._check(deadline)
        boxes = _boxes(drawings)
        near = np.zeros(len(drawings), bool)
        for x0, y0, x1, y1 in table:  # NaN rows (groups) compare False
            near |= (boxes[:, 0] < x1) & (x0 < boxes[:, 2]) & (boxes[:, 1] < y1) & (y0 < boxes[:, 3])
        for n, k in enumerate(np.flatnonzero(near)):
            if n % 256 == 0:
                strokes._check(deadline)
            why = _path_reason(drawings[k], table)
            if why:
                found.add(why)
        # Painted paths get_drawings does not list (a pattern's fill or stroke) and smooth shadings.
        entries, clips = strokes.paths(page, deadline, drawings)
        probe = strokes._Culler()
        strokes._filter(page, probe, update=False)
        match = strokes._match(probe.calls, entries, ~page.transformation_matrix, deadline, clips)
    except TimeoutError:
        return ["time"]
    if strokes._unlisted(page, probe.calls, match, zones):
        found.add("pattern")
    if probe.other[pymupdf.mupdf.FZ_CULL_SHADING] and _shading_under(page, zones):
        found.add("pattern")
    return [c for c in _ORDER if c in found]
