"""What a page's redaction may have left under its zones (decided 2026-10-06: "if unsure, that
page is exported as an image").

MuPDF's redaction, the letters' rectangles of D8 (``vectors.cover``) and the removal of stroked
paths (``strokes.remove``) take out of the file what lies under a zone whenever they can do it
with certainty. What they cannot (a letter a zone cuts, a stroke that crosses the edge of a zone or
that the content filter could not match, a shape painted with a pattern, an image inside a soft
mask, data painted through a clip) would stay in the file under the black box. ``check`` looks at
the page after its redaction and says whether anything of that kind is left; ``pdf.redact_page``
then exports the page as an image (``pdf.rasterize``), which removes it for certain.

Two looks, the second a catch-all:

1. The drawings (``get_drawings(extended=True)``, judged where they are visible: their box cut to
   the clip that shows them). Page layout is left alone: a straight line (a run of collinear
   segments: a rule, an underline drawn in pieces) or an open run of horizontal and vertical
   segments (a cell's border) stroked no wider than ``LAYOUT_WIDTH``, and a rectangle, upright or
   tilted, filled or stroked no wider than that (a frame, a cell, a band, a tilted stamp's border,
   the black boxes of the redaction), unless it lies whole inside a zone and is not black (a QR
   module the zone should have removed). What such a line or rectangle hides under a zone is only
   where it goes on or ends. Any other path, or clip that is not a rectangle, whose outline passes
   inside a zone (a fill's edge, a stroke's painted width) is a leftover; the inside of a filled
   shape that holds the zone whole, with no edge under it, hides nothing. Painted paths that
   ``get_drawings`` does not list (shapes filled or stroked with a pattern) are found with MuPDF's
   content filter, like in ``strokes``: one that touches a zone is a leftover, unless it is drawn
   in the same place as a listed path (the filter reports subpaths a listing may join) or it is a
   rectangle that holds the zone whole under rectangular clips (a page's background pattern).
2. What is still painted inside the zones (``_ink_under``): on a copy of the page, the redaction's
   own black boxes and the page layout of step 1 are dropped with the content filter, and the
   inside of each zone is rendered. A plain background (a flat fill, a smooth gradient) shows no
   edge there; anything with sharp edges (an image a soft mask carries, data painted through a
   clip of small rectangles, a Type3 glyph drawn far from its box, a pattern) is a leftover.
   MuPDF's blanked image pixels render white.

A page that takes longer than ``SECONDS`` to check is a leftover as well.

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
_RIGHT = 0.02  # cosine: corners this close to 90 degrees make a rectangle
_COLLINEAR = 0.3  # points: a run of segments whose points stay this close to one line is a straight line
_AXIS = 0.3  # points: a segment whose ends differ by less than this across is horizontal or vertical
_LINE_UP = 0.6  # points: a filter call drawn this close to a listed path's place is that path
_BLACK = (0.0, 0.0, 0.0)
INK_ZOOM = 3.0  # the inside of the zones is rendered at 216 dpi
INK_INSET = 1.0  # points: the zone's border is left out (antialiasing of the box's own edge)
INK_STEP = 40  # a jump of this much (0-255) between neighbouring pixels is a sharp edge
INK_EDGES = 3  # this many sharp edges inside a zone are ink

# Why a page is exported as an image (Spanish: shown to the user and written in the audit report).
REASONS = {
    "shape": "letras o formas dibujadas bajo una zona no se pudieron quitar con certeza",
    "stroke": "un trazo dibujado bajo una zona no se pudo quitar con certeza",
    "clip": "un trazado de recorte bajo una zona no se pudo quitar con certeza",
    "pattern": "una trama o un degradado bajo una zona no se pudo quitar con certeza",
    "ink": "algo pintado bajo una zona (una imagen, un relleno o un texto dibujado) no se pudo quitar con certeza",
    "time": "la revisión de los dibujos de la página no terminó a tiempo",
}
_ORDER = tuple(REASONS)


def reasons_text(codes: list[str]) -> str:
    """The Spanish reasons of ``codes``, joined (``REASONS``)."""
    return "; ".join(REASONS[c] for c in _ORDER if c in codes)


# ---------------------------------------------------------------------------
# Geometry of a path
# ---------------------------------------------------------------------------


def _cubic(c: np.ndarray) -> np.ndarray:
    """Points along a cubic Bézier curve (4 control points), about one per point of its length."""
    n = int(min(256, max(8, np.hypot(*np.diff(c, axis=0).T).sum())))
    t = np.linspace(0, 1, n + 1)[:, None]
    return (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t**2 * c[2] + t**3 * c[3]


class _Sub:
    """A subpath: ``kind`` "box" (a rectangle or a quad: ``ctrl`` its closed corners) or "line" (a
    run of connected lines and curves: ``pieces`` their control points). ``ctrl`` holds every
    control point, whose box contains the subpath."""

    __slots__ = ("kind", "pieces", "ctrl", "_pts")

    def __init__(self, kind: str, pieces: list[np.ndarray]):
        self.kind = kind
        self.pieces = pieces
        self.ctrl = np.concatenate(pieces)
        self._pts = None

    @property
    def curved(self) -> bool:
        return any(len(p) == 4 for p in self.pieces) and self.kind == "line"

    def points(self) -> np.ndarray:
        """The subpath as a polyline (curves flattened)."""
        if self._pts is None:
            if self.kind != "line":
                self._pts = self.ctrl
            else:
                out = [self.pieces[0][:1]]
                for p in self.pieces:
                    out.append((_cubic(p) if len(p) == 4 else p)[1:])
                self._pts = np.concatenate(out)
        return self._pts


def subpaths(items) -> list[_Sub]:
    """The subpaths of a path's items, as MuPDF paints them. A line or curve that starts where the
    previous one ended continues its subpath (the listing does not keep the moves)."""
    out: list[_Sub] = []
    run: list[np.ndarray] = []
    end = None
    for item in items:
        kind = item[0]
        if kind in ("re", "qu"):
            if run:
                out.append(_Sub("line", run))
                run = []
            if kind == "re":
                r = pymupdf.Rect(item[1])
                corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
            else:
                q = item[1]
                corners = [(p.x, p.y) for p in (q.ul, q.ur, q.lr, q.ll, q.ul)]
            out.append(_Sub("box", [np.asarray(corners, np.float64)]))
            end = None
            continue
        if kind == "l":
            piece = np.array([[item[1].x, item[1].y], [item[2].x, item[2].y]], np.float64)
        elif kind == "c":
            piece = np.array([[p.x, p.y] for p in item[1:5]], np.float64)
        else:
            continue
        if run and (end is None or np.abs(piece[0] - end).max() > 1e-3):
            out.append(_Sub("line", run))
            run = []
        run.append(piece)
        end = piece[-1]
    if run:
        out.append(_Sub("line", run))
    return out


def _frame(sub: _Sub) -> bool:
    """A rectangle, upright or tilted (a "re", or a quad or a closed run of four lines whose
    corners are right angles)."""
    if sub.kind == "box":
        return True
    if sub.curved:
        return False
    pts = sub.points()
    if len(pts) != 5 or np.abs(pts[0] - pts[-1]).max() > 1e-3:
        return False
    d = np.diff(pts, axis=0)
    length = np.hypot(d[:, 0], d[:, 1])
    if length.min() <= 0:
        return False
    nxt = np.roll(d, -1, axis=0)
    cos = np.abs((d * nxt).sum(axis=1)) / (length * np.roll(length, -1))
    return bool(cos.max() < _RIGHT)


def _rounded_frame(sub: _Sub) -> bool:
    """A rectangle with rounded corners: a closed run of four horizontal and vertical lines and
    up to four small curves at its corners (each within a third of the frame's sides)."""
    if sub.kind != "line" or np.abs(sub.ctrl[0] - sub.ctrl[-1]).max() > 1e-3:
        return False
    lines = [p for p in sub.pieces if len(p) == 2]
    curves = [p for p in sub.pieces if len(p) == 4]
    if len(lines) != 4 or len(curves) > 4:
        return False
    if not all(min(abs(p[1, 0] - p[0, 0]), abs(p[1, 1] - p[0, 1])) < _AXIS for p in lines):
        return False
    lo, hi = sub.ctrl.min(axis=0), sub.ctrl.max(axis=0)
    w, h = hi - lo
    return all((c.max(axis=0) - c.min(axis=0) <= (w / 3 + 1e-6, h / 3 + 1e-6)).all() for c in curves)


def _rule(sub: _Sub) -> bool:
    """A straight line: lines only, every point within ``_COLLINEAR`` of the line through the two
    points farthest apart (an underline drawn in pieces is one)."""
    if sub.kind != "line" or sub.curved:
        return False
    pts = sub.points()
    a = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    b = pts[np.argmax(np.hypot(*(pts - a).T))]
    length = float(np.hypot(*(b - a)))
    if length == 0:
        return True
    cross = np.abs((b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])) / length
    return bool(cross.max() <= _COLLINEAR)


def _axis_open(sub: _Sub) -> bool:
    """An open run of horizontal and vertical lines (an L-shaped cell border)."""
    if sub.kind != "line" or sub.curved:
        return False
    pts = sub.points()
    if len(pts) < 3 or np.abs(pts[0] - pts[-1]).max() <= 1e-3:
        return False
    d = np.abs(np.diff(pts, axis=0))
    return bool(np.all(np.minimum(d[:, 0], d[:, 1]) < _AXIS))


def _crosses(pts: np.ndarray, zone) -> bool:
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


def _inside(pts: np.ndarray, zone) -> bool:
    x0, y0, x1, y1 = zone
    return bool(pts[:, 0].min() >= x0 and pts[:, 1].min() >= y0 and pts[:, 0].max() <= x1 and pts[:, 1].max() <= y1)


def _holds(pts: np.ndarray, zone) -> bool:
    """The box of ``pts`` holds ``zone`` whole."""
    return bool(pts[:, 0].min() <= zone[0] and pts[:, 1].min() <= zone[1] and pts[:, 0].max() >= zone[2]
                and pts[:, 1].max() >= zone[3])  # fmt: skip


def _cut(zone, visible) -> tuple[float, float, float, float]:
    """``zone`` cut to the visible box of a path (``visible`` None: the whole zone)."""
    if visible is None:
        return tuple(zone)
    return (max(zone[0], visible[0]), max(zone[1], visible[1]), min(zone[2], visible[2]), min(zone[3], visible[3]))


class _Judged:
    """What ``_judge`` says of one drawing: ``why`` (a key of ``REASONS``) when it is a leftover;
    ``layout`` when every subpath is page layout (rules, rectangles); ``holder`` when it is
    filled (not a black box of the redaction) and one of its subpaths holds a zone whole, with no
    edge under it (a background: a band, a cell, a logo's disc); ``subs`` its subpaths and
    ``reach`` its stroke's."""

    __slots__ = ("why", "layout", "holder", "subs", "reach")

    def __init__(self, why, layout, holder, subs, reach):
        self.why, self.layout, self.holder, self.subs, self.reach = why, layout, holder, subs, reach


def _redaction_box(d: dict) -> bool:
    """A black box of the redaction: MuPDF paints each one as a single rectangle, filled and
    stroked in black (a black fill alone is page content: it may paint data through a clip)."""
    items = d.get("items") or ()
    return (
        d.get("type") == "fs"
        and tuple(d.get("fill") or ()) == _BLACK
        and tuple(d.get("color") or ()) == _BLACK
        and len(items) == 1
        and items[0][0] in ("re", "qu")
    )


def _judge(d: dict, zones: np.ndarray, visible, deadline: float | None) -> _Judged:
    """Judges the drawing ``d`` (``get_drawings(extended=True)``) against ``zones`` (rows x0, y0,
    x1, y1), where it is ``visible`` (its box cut to its clip; None: everywhere)."""
    kind = d.get("type") or ""
    stroked = "s" in kind
    filled = "f" in kind
    width = float(d.get("width") or 0) if stroked else 0.0
    reach = max(width, 0.5) / 2 if stroked else 0.0
    subs = subpaths(d.get("items") or ())
    black = _redaction_box(d)
    why = None
    layout = bool(subs)
    holder = False
    tag = "clip" if kind == "clip" else ("stroke" if kind == "s" else "shape")
    for sub in subs:
        strokes._check(deadline)
        lo, hi = sub.ctrl.min(axis=0) - reach, sub.ctrl.max(axis=0) + reach
        near = zones[(zones[:, 0] < hi[0]) & (lo[0] < zones[:, 2]) & (zones[:, 1] < hi[1]) & (lo[1] < zones[:, 3])]
        if filled and not black and kind != "clip" and any(_holds(sub.ctrl, z) for z in near):
            holder = True
        frame = _frame(sub) or _rounded_frame(sub)
        straight = (_rule(sub) or (_axis_open(sub) and not filled)) and kind != "clip"
        if (frame or straight) and width <= LAYOUT_WIDTH:  # page layout
            if kind == "clip" or black or (straight and not stroked):  # a clip, a black box, nothing painted
                continue
            if why is None and any(
                _inside(sub.ctrl, (z[0] - EDGE, z[1] - EDGE, z[2] + EDGE, z[3] + EDGE)) for z in near
            ):
                why = tag  # whole inside a zone: a mark the zone should have removed
            continue
        layout = False
        if why is not None or not len(near):
            continue
        pts = sub.points()
        for z in near:
            # A stroke paints half its width on each side of the outline; a fill only up to it.
            grown = (z[0] - reach + EDGE, z[1] - reach + EDGE, z[2] + reach - EDGE, z[3] + reach - EDGE)
            if _crosses(pts, _cut(grown, visible)):
                why = tag
                break
    return _Judged(why, layout, holder and why is None, subs, reach)


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


def _plain_clips(drawings: list[dict]) -> dict[int, bool]:
    """For every clip of the listing (by position), whether it and every clip around it is a single
    rectangle: what is painted under it shows as a plain box."""
    out: dict[int, bool] = {}
    stack: list[tuple[int, bool]] = []
    for seq, d in enumerate(drawings):
        if (d.get("type") or "") != "clip":
            continue
        level = int(d.get("level") or 0)
        while stack and stack[-1][0] >= level:
            stack.pop()
        subs = subpaths(d.get("items") or ())
        plain = len(subs) == 1 and _frame(subs[0]) and (not stack or stack[-1][1])
        out[seq] = plain
        stack.append((level, plain))
    return out


def _lines_up(box, judged: dict[int, _Judged]) -> int | None:
    """The listed page-layout path drawn in the place of the filter call box ``box`` (page space),
    if any: the filter reports subpaths that the listing joins (an underline drawn in pieces)."""
    for k, j in judged.items():
        if not j.layout:
            continue
        for sub in j.subs:
            lo, hi = sub.ctrl.min(axis=0), sub.ctrl.max(axis=0)
            tol = j.reach + _LINE_UP
            if _frame(sub):
                if all(abs(a - b) <= tol for a, b in zip(box, (*lo, *hi), strict=True)):
                    return k
            elif box[0] >= lo[0] - tol and box[1] >= lo[1] - tol and box[2] <= hi[0] + tol and box[3] <= hi[1] + tol:
                return k
    return None


# ---------------------------------------------------------------------------
# What is still painted inside the zones
# ---------------------------------------------------------------------------


class _Dropper(strokes._Culler):
    """The content filter's culler that drops the painted paths ``drop`` (positions among the calls,
    as ``strokes._Culler`` counts them) and, with ``shadings``, every smooth shading."""

    def __init__(self, drop: set[int], shadings: bool):
        super().__init__(drop)
        self.shadings = shadings

    def culler(self, ctx, bbox, kind):  # noqa: ARG002 - MuPDF's callback signature
        if self.shadings and kind == pymupdf.mupdf.FZ_CULL_SHADING:
            return 1
        return super().culler(ctx, bbox, kind)


def _render_zones(page: pymupdf.Page, zones: list[pymupdf.Rect], drop: set[int], shadings: bool):
    """The inside of each zone (``INK_INSET`` in) rendered on a copy of the page from which the
    filter calls ``drop`` (and, with ``shadings``, the smooth shadings) were taken out; None when a
    black box of the redaction that fills a zone is still in the copy (the calls did not line up:
    nothing can be judged)."""
    with pymupdf.open() as tmp:
        tmp.insert_pdf(page.parent, from_page=page.number, to_page=page.number, links=False, annots=False)
        copy = tmp[0]
        strokes._filter(copy, _Dropper(set(drop), shadings), update=True)
        for d in copy.get_drawings():
            if _redaction_box(d) and any(
                all(abs(a - b) <= 0.05 for a, b in zip(d["rect"], z, strict=True)) for z in zones
            ):
                return None
        out = []
        for z in zones:
            inner = z + (INK_INSET, INK_INSET, -INK_INSET, -INK_INSET)
            if inner.is_empty or inner.width < 1 or inner.height < 1:
                continue
            pix = copy.get_pixmap(matrix=pymupdf.Matrix(INK_ZOOM, INK_ZOOM), clip=inner, alpha=False, annots=False)
            if pix.width and pix.height:
                out.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).astype(np.int16))
        return out


def _ink_under(page: pymupdf.Page, zones: list[pymupdf.Rect], layout: set[int], backgrounds: set[int]) -> bool:
    """Something is still painted inside a zone. Two renders of a copy of the page, inside the zones:
    without the redaction's black boxes and the page layout (``layout``), any sharp edge is ink
    (an image a soft mask carries, data painted through a clip, a glyph); without the backgrounds
    too (``backgrounds``: fills that hold a zone whole under rectangular clips, and smooth
    shadings), anything not white is ink (a uniform image or pattern under the box)."""
    edges = _render_zones(page, zones, layout, False)
    if edges is None:
        return True
    for a in edges:
        if a.shape[0] < 2 or a.shape[1] < 2:
            continue
        n = int((np.abs(np.diff(a, axis=1)).max(axis=2) > INK_STEP).sum())
        n += int((np.abs(np.diff(a, axis=0)).max(axis=2) > INK_STEP).sum())
        if n >= INK_EDGES:
            return True
    plain = _render_zones(page, zones, layout | backgrounds, True)
    if plain is None:
        return True
    return any(int((a.min(axis=2) < 255 - INK_STEP).sum()) >= INK_EDGES for a in plain)


def _soft_masks(doc: pymupdf.Document) -> bool:
    """The document paints through a soft mask somewhere (an ExtGState's /SMask dictionary): a
    fill that holds a zone may then show anything there."""
    cached = getattr(doc, "_anonymizer_soft_masks", None)
    if cached is None:
        cached = False
        for x in range(1, doc.xref_length()):
            try:
                source = doc.xref_object(x, compressed=True)
            except Exception:  # noqa: BLE001 - a broken object: no answer from it
                continue
            if "/Luminosity" in source or ("/SMask<<" in source.replace(" ", "")) or "/Alpha" in source:
                cached = True
                break
        doc._anonymizer_soft_masks = cached
    return cached


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


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
        entries, clips = strokes.paths(page, deadline, drawings)
        by_seq = {e.seq: (i, e) for i, e in enumerate(entries)}
        plain = _plain_clips(drawings)
        probe = strokes._Culler()
        strokes._filter(page, probe, update=False)
        match = strokes._match(probe.calls, entries, ~page.transformation_matrix, deadline, clips)
        boxes = _boxes(drawings)
        near = np.zeros(len(drawings), bool)
        for x0, y0, x1, y1 in table:  # NaN rows (groups) compare False
            near |= (boxes[:, 0] < x1) & (x0 < boxes[:, 2]) & (boxes[:, 1] < y1) & (y0 < boxes[:, 3])
        # Shapes painted with a pattern that hold the zones whole under rectangular clips (a page's
        # background pattern): judged by what they paint there, with their cells.
        holder_calls: set[int] = set()
        holder_clips: set[int] = set()
        unlisted = strokes._unlisted(page, probe.calls, match, zones)
        for n, box, _ in unlisted:
            owner = match.get(n)
            if owner is None or owner >= 0:
                continue
            seq = clips[-1 - owner][0]
            shape = subpaths(drawings[seq].get("items") or ())
            touched = [z for z in table if box[0] < z[2] and z[0] < box[2] and box[1] < z[3] and z[1] < box[3]]
            if (
                plain.get(seq)
                and len(shape) == 1
                and _frame(shape[0])
                and all(_holds(shape[0].ctrl, z) for z in touched)
            ):
                holder_calls.add(n)
                holder_clips.add(seq)
        judged: dict[int, _Judged] = {}
        for k in np.flatnonzero(near):
            strokes._check(deadline)
            d = drawings[k]
            if (d.get("type") or "") == "group":
                continue
            visible = None
            if k in by_seq:
                _, entry = by_seq[k]
                if not entry.visible:
                    continue  # clipped away: it shows nowhere, so the zone hides nothing of it
                if entry.clip in holder_clips:
                    continue  # a cell of a background pattern: judged by what it paints
                visible = entry.extent
            j = _judge(d, table, visible, deadline)
            judged[k] = j
            if j.why:
                found.add(j.why)
        for n, box, _ in unlisted:
            if n in holder_calls:
                continue
            k = _lines_up(box, judged)
            if k is None:
                found.add("pattern")
            else:
                match[n] = by_seq[k][0] if k in by_seq else match.get(n, -1)
        if not found:
            index = {i: seq for seq, (i, _) in by_seq.items()}
            masked = _soft_masks(page.parent)
            layout: set[int] = set()
            backgrounds: set[int] = set()
            for n, owner in match.items():
                if owner < 0 or probe.calls[n][0] not in strokes._PATH_KINDS:
                    continue
                seq = index.get(owner, -1)
                j = judged.get(seq)
                if j is None:
                    continue
                if j.layout and not j.holder:
                    layout.add(n)
                elif j.holder and not masked and (by_seq[seq][1].clip < 0 or plain.get(by_seq[seq][1].clip)):
                    backgrounds.add(n)
            strokes._check(deadline)
            if _ink_under(page, zones, layout, backgrounds):
                found.add("ink")
    except TimeoutError:
        return ["time"]
    return [c for c in _ORDER if c in found]
