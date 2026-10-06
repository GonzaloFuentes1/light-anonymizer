"""Text converted to vector paths (D8): letters drawn as curves, with no text behind them.

PDFs exported from design tools, or a block pasted as an image of text and then vectorized, show
text that has no text layer: only OCR can read it. A scanned-looking page (almost no text layer)
is already read whole by OCR; on a page with a text layer, only the areas where many small filled
paths sit with few characters of the text layer are rendered and read (``text_regions``). Pages
without them cost one listing of their drawings and are never rendered for this.

Coordinates are PyMuPDF's unrotated page space, like the text layer. Every call needs
``PDF_LOCK`` held (PyMuPDF is not thread-safe).
"""

from __future__ import annotations

import pymupdf

# A filled path no larger than this on either side (points) can be a letter; larger ones are
# drawings, frames or backgrounds.
GLYPH_MAX_SIDE = 40.0
# An area is read by OCR when it has at least this many letter-like paths...
MIN_GLYPHS = 8
# ...and fewer characters of the text layer than this share of its paths.
MAX_CHARS_PER_GLYPH = 0.5


def glyph_paths(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Boxes of the filled, letter-sized paths of the page that are not plain boxes (a bullet square,
    a table cell, a QR module or the black box of a redaction are a single rectangle)."""
    rects = []
    for d in page.get_cdrawings():
        if d.get("type") not in ("f", "fs"):
            continue
        x0, y0, x1, y1 = d["rect"]
        if not (0 < x1 - x0 <= GLYPH_MAX_SIDE and 0 < y1 - y0 <= GLYPH_MAX_SIDE):
            continue
        items = d.get("items") or ()
        if len(items) == 1 and items[0][0] in ("re", "qu"):
            continue
        rects.append(pymupdf.Rect(x0, y0, x1, y1))
    return rects


def clusters(rects: list[pymupdf.Rect]) -> list[tuple[pymupdf.Rect, int]]:
    """Groups boxes closer than about one letter height (words, lines and blocks of drawn text).

    Returns each group's box (grown by that margin) and how many boxes it joins.
    """
    groups: list[list] = []  # [x0, y0, x1, y1, count]
    for r in rects:
        m = max(2.0, r.height)
        groups.append([r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m, 1])
    changed = True
    while changed:
        changed = False
        merged: list[list] = []
        for g in groups:
            for o in merged:
                if g[0] < o[2] and o[0] < g[2] and g[1] < o[3] and o[1] < g[3]:
                    o[:] = [min(o[0], g[0]), min(o[1], g[1]), max(o[2], g[2]), max(o[3], g[3]), o[4] + g[4]]
                    changed = True
                    break
            else:
                merged.append(g)
        groups = merged
    return [(pymupdf.Rect(g[:4]), g[4]) for g in groups]


def text_regions(
    page: pymupdf.Page, char_boxes: list[pymupdf.Rect | None], glyphs: list[pymupdf.Rect] | None = None
) -> list[pymupdf.Rect]:
    """Areas of the page with text drawn as paths: many letter-like paths and few characters of the
    text layer (``char_boxes``, the boxes of ``pdf.chars``). ``glyphs``: ``glyph_paths(page)``, if
    already listed."""
    glyphs = glyph_paths(page) if glyphs is None else glyphs
    if len(glyphs) < MIN_GLYPHS:
        return []
    centers = [
        pymupdf.Point((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2) for b in char_boxes if b is not None and not b.is_empty
    ]
    regions = []
    for rect, count in clusters(glyphs):
        if count < MIN_GLYPHS:
            continue
        chars = sum(1 for c in centers if rect.contains(c))
        if chars < count * MAX_CHARS_PER_GLYPH:
            regions.append(rect)
    return regions


def snap(rect: pymupdf.Rect, glyphs: list[pymupdf.Rect], min_share: float = 0.25) -> pymupdf.Rect:
    """``rect`` grown to cover every letter-like path it covers at least ``min_share`` of.

    MuPDF removes a path only when the redaction covers all of it: an OCR box a little tighter than
    a letter would leave that letter's path in the file, under the black box.
    """
    out = pymupdf.Rect(rect)
    for g in glyphs:
        inter = g & rect
        if not inter.is_empty and inter.get_area() >= min_share * max(g.get_area(), 1e-6):
            out |= g
    return out
