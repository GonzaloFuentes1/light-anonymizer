"""Text converted to vector paths (D8): letters drawn as curves, with no text behind them.

PDFs exported from design tools, or a block pasted as an image of text and then vectorized, show
text that has no text layer: only OCR can read it. A scanned-looking page (almost no text layer)
is already read whole by OCR; on a page with a text layer, only the areas where many small filled
shapes sit with few characters of the text layer are rendered and read (``text_regions``). Pages
without them cost one listing of their drawings and are never rendered for this.

A drawing program may write one path per letter or one path for a whole line or block, with each
letter as a subpath; so the unit here is the subpath. MuPDF redacts line art subpath by subpath:
it removes a subpath only when the redaction covers all of it (``snap`` grows a zone to do so, and
``verify.glyph_leaks`` checks that none is left under an applied zone).

Coordinates are PyMuPDF's unrotated page space, like the text layer. Every call needs
``PDF_LOCK`` held (PyMuPDF is not thread-safe).
"""

from __future__ import annotations

import math
from collections import defaultdict

import pymupdf

# A filled shape no larger than this on either side (points) can be a letter; larger ones are
# drawings, frames or backgrounds.
GLYPH_MAX_SIDE = 40.0
# An area is read by OCR when it has at least this many letter-like shapes...
MIN_GLYPHS = 8
# ...and fewer characters of the text layer than this share of its shapes.
MAX_CHARS_PER_GLYPH = 0.5
# A letter-like shape is under a zone when at least this share of its box is inside the zone: it is
# then covered whole when the zone is applied (``snap``), and one left in the output is a leak.
UNDER_SHARE = 0.02
_TOUCH = 1e-3  # points: a subpath goes on where the previous segment ended


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


def glyph_paths(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Boxes of the filled, letter-sized subpaths of the page that are not plain boxes (a bullet
    square, a table cell, a QR module or the black box of a redaction are a single rectangle)."""
    rects = []
    for d in page.get_cdrawings():
        if d.get("type") not in ("f", "fs"):
            continue
        x0, y0, x1, y1 = d["rect"]
        if 0 < x1 - x0 <= GLYPH_MAX_SIDE and 0 < y1 - y0 <= GLYPH_MAX_SIDE:  # one small path: one box
            items = d.get("items") or ()
            if not (len(items) == 1 and items[0][0] in ("re", "qu")):
                rects.append(pymupdf.Rect(x0, y0, x1, y1))
            continue
        for points, is_box in _subpaths(d.get("items") or ()):  # a large path: its letter-sized subpaths
            if is_box or not points:
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if 0 < w <= GLYPH_MAX_SIDE and 0 < h <= GLYPH_MAX_SIDE:
                rects.append(pymupdf.Rect(min(xs), min(ys), max(xs), max(ys)))
    return rects


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
    while changed:  # few areas: a plain pairwise join
        changed = False
        joined: list[pymupdf.Rect] = []
        for r in regions:
            for o in joined:
                if r.intersects(o):
                    o |= r
                    changed = True
                    break
            else:
                joined.append(pymupdf.Rect(r))
        regions = joined
    return regions


def under(rect: pymupdf.Rect, glyphs: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """The letter-like shapes with at least ``UNDER_SHARE`` of their box inside ``rect``."""
    out = []
    for g in glyphs:
        inter = g & rect
        if not inter.is_empty and inter.get_area() >= UNDER_SHARE * max(g.get_area(), 1e-6):
            out.append(g)
    return out


def snap(rect: pymupdf.Rect, glyphs: list[pymupdf.Rect]) -> pymupdf.Rect:
    """``rect`` grown to cover every letter-like shape under it (``under``), so that applying it
    removes them: an OCR box or a drawn zone a little tighter than a letter would otherwise leave
    that letter in the file, under the black box."""
    out = pymupdf.Rect(rect)
    for g in under(rect, glyphs):
        out |= g
    return out
