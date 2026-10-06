"""Text converted to vector paths (D8): letters drawn as curves, with no text behind them.

PDFs exported from design tools, or a block pasted as an image of text and then vectorized, show
text that has no text layer: only OCR can read it. A scanned-looking page (almost no text layer)
is already read whole by OCR; on a page with a text layer, only the areas where many small filled
shapes sit with few characters of the text layer are rendered and read (``text_regions``). Pages
without them cost one listing of their drawings and are never rendered for this.

A drawing program may write one path per letter or one path for a whole line or block, with each
letter as a subpath; so the unit here is the subpath. MuPDF redacts line art subpath by subpath,
and the first redaction rectangle that touches a shape decides: the shape is removed only if that
rectangle covers all of it (measured with MuPDF 1.28: a letter first touched by a zone that cuts
it stays, even when another rectangle covers it whole). For a shape that is also stroked, the
rectangle must cover the box grown by the stroke: half the line width with round or bevel joins,
the miter limit (10) times the width with miter joins. So when a zone is applied
(``pdf.apply_page_zones``), each letter under it (``under``: its centre, or half of it, inside the
zone) first gets a small rectangle of its own (``cover``), in a pass of its own; only then the
zones are applied, deciding alone on everything else. Never a band across the zone, at most a few
points beyond it, and never into what the reviewer kept visible: a letter whose centre is in a
kept area belongs to that area. ``verify.glyph_leaks`` checks that no letter under an applied zone,
and no shape almost entirely inside one, is left.

Coordinates are PyMuPDF's unrotated page space, like the text layer. Every call needs
``PDF_LOCK`` held (PyMuPDF is not thread-safe).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import NamedTuple

import pymupdf

# A filled shape no larger than this on either side (points) can be a letter; larger ones are
# drawings, frames or backgrounds.
GLYPH_MAX_SIDE = 40.0
# An area is read by OCR when it has at least this many letter-like shapes...
MIN_GLYPHS = 8
# ...and fewer characters of the text layer than this share of its shapes.
MAX_CHARS_PER_GLYPH = 0.5
# A letter is under a zone when its centre is inside it, or at least this share of its box: it is
# then covered whole when the zone is applied (``cover``), and one left in the output is a leak.
UNDER_SHARE = 0.5
# A letter is covered only if that does not reach further than this beyond the zone (points); one
# that would is left, and the leak check asks for a larger zone.
COVER_MAX = 6.0
# Points added around a letter's rectangle: MuPDF does not count a shape as covered when the edges
# are exactly equal.
COVER_PAD = 0.02
# How far MuPDF looks beyond the box of a stroked shape: the miter limit (PDF default 10) times the
# line width with miter joins, half the width with round or bevel joins (measured, MuPDF 1.28).
MITER_LIMIT = 10.0
# A shape of any size with at least this share of its box inside an applied zone and still in the
# output is a leak (``verify.glyph_leaks``).
LEFT_SHARE = 0.9
_TOUCH = 1e-3  # points: a subpath goes on where the previous segment ended


class Shape(NamedTuple):
    """A filled subpath: its box and how far MuPDF's "covered" test reaches beyond it (``reach``: 0
    for a shape that is only filled, the stroke's reach for one that is also stroked)."""

    box: pymupdf.Rect
    reach: float


def _subpaths(items) -> list[tuple[list[tuple[float, float]], bool]]:
    """Splits the items of a drawing (``get_cdrawings``) into subpaths: (points, is_box). A new
    subpath starts where a segment does not begin at the end of the previous one; a rectangle or a
    quad is a subpath of its own (a box)."""
    out: list[tuple[list[tuple[float, float]], bool]] = []
    end = None
    for item in items:
        kind = item[0]
        if kind == "re":
            x0, y0, x1, y1 = item[1]
            out.append(([(x0, y0), (x1, y1)], True))
            end = None
            continue
        if kind == "qu":
            out.append(([tuple(p) for p in item[1]], True))
            end = None
            continue
        points = [tuple(p) for p in item[1:]]
        start = points[0]
        if end is None or abs(start[0] - end[0]) > _TOUCH or abs(start[1] - end[1]) > _TOUCH:
            out.append(([], False))
        out[-1][0].extend(points)
        end = points[-1]
    return out


def _reach(d: dict) -> float:
    if d.get("type") != "fs":
        return 0.0
    width = float(d.get("width") or 1.0)
    return MITER_LIMIT * width if not d.get("lineJoin") else width / 2


def shapes(page: pymupdf.Page, max_side: float | None = GLYPH_MAX_SIDE) -> list[Shape]:
    """The filled shapes of the page that are not plain boxes (a bullet square, a table cell, a QR
    module or the black box of a redaction are a single rectangle): a small path is one shape,
    whatever its subpaths (an "i" and its dot); a larger one gives its subpaths, as MuPDF redacts
    them, those up to ``max_side`` on each side (None: any size)."""
    out = []
    for d in page.get_cdrawings():
        if d.get("type") not in ("f", "fs"):
            continue
        reach = _reach(d)
        items = d.get("items") or ()
        x0, y0, x1, y1 = d["rect"]
        if 0 < x1 - x0 <= GLYPH_MAX_SIDE and 0 < y1 - y0 <= GLYPH_MAX_SIDE:  # one small path: one shape
            if not (len(items) == 1 and items[0][0] in ("re", "qu")):
                out.append(Shape(pymupdf.Rect(x0, y0, x1, y1), reach))
            continue
        for points, is_box in _subpaths(items):  # its subpaths, as MuPDF redacts them
            if is_box or not points:
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if (w > 0 or h > 0) and (max_side is None or (0 < w <= max_side and 0 < h <= max_side)):
                out.append(Shape(pymupdf.Rect(min(xs), min(ys), max(xs), max(ys)), reach))
    return out


def glyph_paths(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Boxes of the letter-like shapes of the page (``shapes``): filled, letter-sized, not boxes."""
    return [s.box for s in shapes(page)]


def clusters(rects: list[pymupdf.Rect]) -> list[tuple[pymupdf.Rect, int]]:
    """Groups boxes closer than about one letter height (words, lines and blocks of drawn text).

    Returns each group's box (grown by that margin) and how many boxes it joins. Linear in the
    number of boxes: each one is only compared with those in the same cells of a grid.
    """
    grown = []
    for r in rects:
        m = max(2.0, r.height)
        grown.append((r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m))
    parent = list(range(len(grown)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cell = 3 * GLYPH_MAX_SIDE  # a grown box spans at most two cells per side
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x0, y0, x1, y1) in enumerate(grown):
        for cx in range(math.floor(x0 / cell), math.floor(x1 / cell) + 1):
            for cy in range(math.floor(y0 / cell), math.floor(y1 / cell) + 1):
                bucket = grid[(cx, cy)]
                for j in bucket:
                    a, b = root(i), root(j)
                    if a != b:
                        o = grown[j]
                        if x0 < o[2] and o[0] < x1 and y0 < o[3] and o[1] < y1:
                            parent[a] = b
                bucket.append(i)
    groups: dict[int, list] = {}
    for i, (x0, y0, x1, y1) in enumerate(grown):
        g = groups.setdefault(root(i), [x0, y0, x1, y1, 0])
        g[:] = [min(g[0], x0), min(g[1], y0), max(g[2], x1), max(g[3], y1), g[4] + 1]
    return [(pymupdf.Rect(g[:4]), g[4]) for g in groups.values()]


def text_regions(
    page: pymupdf.Page, char_boxes: list[pymupdf.Rect | None], glyphs: list[pymupdf.Rect] | None = None
) -> list[pymupdf.Rect]:
    """Areas of the page with text drawn as paths: many letter-like shapes and few characters of the
    text layer (``char_boxes``, the boxes of ``pdf.chars``). ``glyphs``: ``glyph_paths(page)``, if
    already listed. Overlapping areas are joined, so none is read twice."""
    glyphs = glyph_paths(page) if glyphs is None else glyphs
    if len(glyphs) < MIN_GLYPHS:
        return []
    centers = [
        pymupdf.Point((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2) for b in char_boxes if b is not None and not b.is_empty
    ]
    regions: list[pymupdf.Rect] = []
    for rect, count in clusters(glyphs):
        if count < MIN_GLYPHS:
            continue
        chars = sum(1 for c in centers if rect.contains(c))
        if chars < count * MAX_CHARS_PER_GLYPH:
            regions.append(rect)
    changed = True
    while changed:  # few areas: a plain pairwise join, until no two overlap
        changed = False
        joined: list[pymupdf.Rect] = []
        for r in regions:
            for k, o in enumerate(joined):
                if r.intersects(o):
                    joined[k] = o | r  # a new Rect: pymupdf.Rect has no in-place union
                    changed = True
                    break
            else:
                joined.append(pymupdf.Rect(r))
        regions = joined
    return regions


def join_overlapping(rects: list[pymupdf.Rect], keep: list[pymupdf.Rect] | tuple = ()) -> list[pymupdf.Rect]:
    """``rects`` with every group of overlapping rectangles replaced by its union, so that no shape
    covered by one of them is first touched by a neighbour that cuts it. Two rectangles whose union
    would enter an area in ``keep`` stay apart: only those, not the whole group."""
    out = [pymupdf.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        joined: list[pymupdf.Rect] = []
        for r in out:
            for k, o in enumerate(joined):
                if r.intersects(o) and not any((o | r).intersects(a) for a in keep):
                    joined[k] = o | r
                    changed = True
                    break
            else:
                joined.append(r)
        out = joined
    return out


def letters(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """The boxes of the letter-like shapes of the page, filled or filled and stroked: what applying a
    zone removes, and what the leak check looks for under the zones applied."""
    return glyph_paths(page)


def centre(r: pymupdf.Rect) -> pymupdf.Point:
    return pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)


def kept(g: pymupdf.Rect, keep: list[pymupdf.Rect] | tuple) -> bool:
    """The letter ``g`` belongs to an area left visible: its centre is in one of ``keep``."""
    c = centre(g)
    return any(k.contains(c) for k in keep)


def left_over(page: pymupdf.Page, rect: pymupdf.Rect) -> list[pymupdf.Rect]:
    """Filled shapes of any size (``shapes`` without a size limit, plus boxes other than the black
    ones of the redactions) with at least ``LEFT_SHARE`` of their box inside ``rect``: after a zone
    is applied, none should be left."""
    boxes = [s.box for s in shapes(page, max_side=None)]
    for d in page.get_cdrawings():
        if d.get("type") in ("f", "fs") and tuple(d.get("fill") or ()) != (0.0, 0.0, 0.0):
            for points, is_box in _subpaths(d.get("items") or ()):
                if is_box and points:
                    xs, ys = [q[0] for q in points], [q[1] for q in points]
                    boxes.append(pymupdf.Rect(min(xs), min(ys), max(xs), max(ys)))
    out = []
    for box in boxes:
        inter = box & rect
        if box.get_area() > 0 and not inter.is_empty and inter.get_area() >= LEFT_SHARE * box.get_area():
            out.append(box)
    return out


def under(rect: pymupdf.Rect, glyphs: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """The letters of ``rect``: those whose centre is inside it, or at least ``UNDER_SHARE`` of their
    box. A letter of the next line that a zone only grazes is not one of them."""
    out = []
    for g in glyphs:
        centre = pymupdf.Point((g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2)
        inter = g & rect
        if rect.contains(centre) or (not inter.is_empty and inter.get_area() >= UNDER_SHARE * max(g.get_area(), 1e-6)):
            out.append(g)
    return out


def cover(rect: pymupdf.Rect, letters: list[Shape], keep: list[pymupdf.Rect] | tuple = ()) -> list[pymupdf.Rect]:
    """One small rectangle per letter under ``rect`` (``under``), so that applying the zone removes
    them whole: an OCR box or a drawn zone a little tighter than a letter would otherwise leave that
    letter in the file, under the black box. Nothing else is covered.

    A letter whose centre is in an area of ``keep`` (what the reviewer left visible) belongs to that
    area: it is not covered, nor a leak. A letter that reaches more than ``COVER_MAX`` beyond the
    zone, or whose rectangle (grown by its stroke's reach, up to ``COVER_MAX``) would enter a kept
    area, is not covered: the leak check reports it."""
    out = []
    limit = rect + (-COVER_MAX, -COVER_MAX, COVER_MAX, COVER_MAX)
    under_rect = {tuple(g) for g in under(rect, [s.box for s in letters])}
    for s in letters:
        g = s.box
        if tuple(g) not in under_rect or kept(g, keep) or not limit.contains(g):
            continue
        grow = min(s.reach, COVER_MAX) + COVER_PAD
        r = g + (-grow, -grow, grow, grow)
        if any(r.intersects(k) for k in keep):
            continue
        out.append(r)
    return out
