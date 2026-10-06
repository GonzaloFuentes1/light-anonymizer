"""Stroked vector paths under redaction zones: which ones leave the file, removing them, and the leak check.

MuPDF's redaction removes the *filled* paths a zone covers but keeps every *stroked* one: a
signature drawn with a pen tool stayed in the file under the black box. Its option to remove every
path a zone touches also removes page frames, table shading and background bands. Stroked paths
are therefore handled here, apart from MuPDF's redaction:

- ``plan`` decides with the box of each stroked path cut to the clip that shows it. A path whose
  box lies inside a zone (one point of margin: the redaction and the leak check use the same) goes.
  Page layout (rectangles, straight rules) and closed convex outlines (rings, ovals, rounded
  frames: stamps, radio buttons) that cross a zone stay. A pen stroke (a curve, or a polyline that
  is not a grid) that crosses the edge of a zone where drawings are the data (a signature, a zone
  drawn by the reviewer) goes whole when most of its visible length lies inside the zones, and the
  reviewer must enlarge the zone when it turns back like handwriting with most of it outside. Any
  other stroke with at least 80 % of its visible length under the zones also needs a larger zone.
- ``remove`` takes them out of the page's content with MuPDF's content filter. The filter only
  reports, in content order, the box of each painted path (grown for its stroke), so its calls are
  lined up with ``page.get_drawings()`` (same order, consistent boxes, found through an index of
  box centers) and only the calls matched to a planned path are dropped: never a glyph, an image,
  a Type3 glyph procedure (the filter does not enter them) or a path it cannot match. Then the
  page's drawings, text and images are compared with what was expected; on any other difference
  the page is put back as it was.
- ``leaks`` runs ``plan`` again over the exported file with the same zones: a stroke that should
  have left and is still there, or one the zone must cover entirely, blocks the export with a
  message that says what to do.

Coordinates: PyMuPDF's unrotated page space (that of ``get_drawings`` and ``add_redact_annot``).
The filter's boxes are in PDF user space (``~page.transformation_matrix``). This module uses
private PyMuPDF bindings (``_make_PdfFilterOptions``, ``_as_pdf_page``) and MuPDF's culler
callback: ``self_test`` checks at startup that they still behave as expected.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import pymupdf

from anonymizer.engine.common import bbox_of
from anonymizer.engine.model import TYPE_LABELS, Finding, Leak
from anonymizer.engine.signatures import closed_convex, path_shape

log = logging.getLogger(__name__)

mupdf = pymupdf.mupdf
_FILL, _STROKE, _FILL_STROKE = mupdf.FZ_CULL_PATH_FILL, mupdf.FZ_CULL_PATH_STROKE, mupdf.FZ_CULL_PATH_FILL_STROKE
_PATH_KINDS = (_FILL, _STROKE, _FILL_STROKE)

# Zones whose content may be a drawing: a pen stroke that crosses their edge matters.
DRAWN_TYPES = ("signature", "manual")
MOSTLY = 0.6  # share of a pen stroke's visible length under a drawing zone for it to go whole
UNDER = 0.8  # share of any other stroke's visible length under the zones for the zone to be enlarged
MARGIN = 1.0  # points: a path this close to a zone's edge counts as inside it
REMOVE_SECONDS = 15.0  # lining the filter up with the page gives up after this (the export is then blocked)
_TOLERANCE = 0.5  # points: the filter's boxes and get_drawings' rectangles agree within this

Box = tuple[float, float, float, float]


class Entry(NamedTuple):
    """A path of ``get_drawings`` with its box and drawn extent (box grown by half the line width),
    both cut to the clip that shows it (None: clipped away), and the clips in effect."""

    d: dict
    box: Box | None
    extent: Box | None
    clips: tuple[int, ...]  # indexes into the page's clips (``paths``)


class Plan(NamedTuple):
    """What ``plan`` decided for the stroked paths of a page."""

    paths: list[Entry]
    remove: set[int]  # paths that leave the file
    whole: set[int]  # among them, those removed whole although part of them lies outside the zones
    enlarge: list[tuple[int, int]]  # (path, zone index): the zone must cover the whole stroke
    calls: list[tuple[int, Box]]  # the paths the content filter paints, in order (``_Culler``)
    other: Counter  # the other things it paints (glyphs, images...), by kind


# ---------------------------------------------------------------------------
# Zones and paths
# ---------------------------------------------------------------------------


def zones_by_page(
    findings: Iterable[Finding], to_page: list[pymupdf.Matrix]
) -> dict[int, list[tuple[pymupdf.Rect, Finding]]]:
    """The redaction rectangle of each finding (its bounding box, in unrotated page space), by page.

    The redaction and the leak check both use these exact rectangles.
    """
    out: dict[int, list[tuple[pymupdf.Rect, Finding]]] = {}
    for f in findings:
        if 0 <= f.page < len(to_page):
            r = (pymupdf.Rect(*bbox_of(f.polygon)) * to_page[f.page]).normalize()
            out.setdefault(f.page, []).append((r, f))
    return out


def is_pen_stroke(items) -> bool:
    """A curve, or a polyline of six segments or more that is not mostly axis-parallel (the
    signature detector's criterion), and not a closed convex outline (a ring, an oval, a rounded
    frame): what a pen leaves, unlike rules, frames, grids and stamps."""
    curves, segments, _, straight, _ = path_shape(items)
    return (curves >= 1 or (segments >= 6 and straight < 0.6)) and not closed_convex(items)


def _turns_back(items) -> bool:
    """Handwriting: the pen turns back left or right (loops), or swings up and down steeply again
    and again (a zigzag scrawl). A chart's curve goes one way and rises or falls gently."""
    pts = []
    for item in items:
        if item[0] == "l":
            pts += [item[1], item[2]]
        elif item[0] == "c":
            pts += list(item[1:5])
    if len(pts) < 3:
        return False
    steps = np.diff(np.array([[p.x, p.y] for p in pts], np.float64), axis=0)
    steps = steps[np.hypot(steps[:, 0], steps[:, 1]) > 0.3]
    if len(steps) < 2:
        return False

    def reversals(values: np.ndarray) -> int:
        signs = np.sign(values[np.abs(values) > 0.3])
        return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0

    steep = np.abs(steps[:, 1]).sum() / max(np.abs(steps[:, 0]).sum(), 1e-6)
    return reversals(steps[:, 0]) >= 2 or (reversals(steps[:, 1]) >= 3 and steep >= 1.5)


def _is_layout(items) -> bool:
    """Rectangles and straight horizontal or vertical rules: the page's frames, tables and underlines."""
    kinds = [item[0] for item in items]
    if all(k in ("re", "qu") for k in kinds):
        return True
    if kinds == ["l"]:
        p, q = items[0][1], items[0][2]
        return abs(p.x - q.x) < 0.5 or abs(p.y - q.y) < 0.5
    return False


def _intersect(a: Box, b: Box) -> Box | None:
    r = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return r if r[0] <= r[2] and r[1] <= r[3] else None


def _inside(r: Box, zone: pymupdf.Rect, margin: float = 0.0) -> bool:
    return (
        zone.x0 - margin <= r[0] and zone.y0 - margin <= r[1] and r[2] <= zone.x1 + margin and r[3] <= zone.y1 + margin
    )


def _touches(r: Box, zone: pymupdf.Rect) -> bool:
    return r[0] < zone.x1 and zone.x0 < r[2] and r[1] < zone.y1 and zone.y0 < r[3]


def paths(page: pymupdf.Page) -> tuple[list[Entry], list[Box | None]]:
    """The page's paths (``get_drawings``, in content order) and the scissor of every clip."""
    out: list[Entry] = []
    scissors: list[Box | None] = []
    stack: list[tuple[int, int, Box | None]] = []  # (level, clip index, scissor cut to the outer clips)
    for d in page.get_drawings(extended=True):
        kind = d.get("type") or ""
        level = int(d.get("level") or 0)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if kind == "clip":
            s = d.get("scissor")
            scissor = None if s is None else (s.x0, s.y0, s.x1, s.y1)
            scissors.append(scissor)
            if stack and scissor is not None and stack[-1][2] is not None:
                scissor = _intersect(scissor, stack[-1][2])
            stack.append((level, len(scissors) - 1, scissor))
            continue
        if kind == "group" or "rect" not in d:
            continue
        half = max(float(d.get("width") or 0), 0.5) / 2 if "s" in kind else 0.0
        r = d["rect"]
        box: Box | None = (r.x0, r.y0, r.x1, r.y1)
        drawn: Box | None = (r.x0 - half, r.y0 - half, r.x1 + half, r.y1 + half)
        if stack and stack[-1][2] is not None:
            box = _intersect(box, stack[-1][2])
            drawn = _intersect(drawn, stack[-1][2]) if box is not None else None
        out.append(Entry(d, box, drawn, tuple(c for _, c, _ in stack)))
    return out, scissors


def _samples(items, visible: Box) -> np.ndarray:
    """Points along the curves and segments of a path (about one per point of length) that lie in ``visible``."""
    pts = []
    for item in items:
        if item[0] == "l":
            ctrl = [item[1], item[2]]
        elif item[0] == "c":
            ctrl = list(item[1:5])
        else:
            continue
        c = np.array([[p.x, p.y] for p in ctrl], np.float64)
        n = int(min(200, max(4, np.hypot(*np.diff(c, axis=0).T).sum())))
        t = np.linspace(0, 1, n)[:, None]
        if len(c) == 2:
            pts.append(c[0] + t * (c[1] - c[0]))
        else:
            pts.append((1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t**2 * c[2] + t**3 * c[3])
    if not pts:
        return np.zeros((0, 2))
    p = np.concatenate(pts)
    keep = (p[:, 0] >= visible[0]) & (p[:, 0] <= visible[2]) & (p[:, 1] >= visible[1]) & (p[:, 1] <= visible[3])
    return p[keep]


def _share_inside(items, visible: Box, zones: list[pymupdf.Rect]) -> float:
    p = _samples(items, visible)
    if not len(p):
        return 0.0
    inside = np.zeros(len(p), bool)
    for z in zones:
        inside |= (p[:, 0] >= z.x0) & (p[:, 0] <= z.x1) & (p[:, 1] >= z.y0) & (p[:, 1] <= z.y1)
    return float(inside.mean())


def _pattern_cells(
    found: list[Entry], scissors: list[Box | None], calls: list[tuple[int, Box]], match: dict[int, int], to_user
) -> set[int]:
    """The paths ``get_drawings`` lists for the cells of tiling patterns.

    A shape filled (or stroked) with a pattern is a path the content filter paints and
    ``get_drawings`` does not list; instead it lists a clip with that shape's box and, under it, the
    pattern's cell, which the filter never reports. Those cells are not paths of the page: the
    redaction neither plans for them nor reports them.
    """
    unexplained = [calls[n][1] for n in range(len(calls)) if n not in match]
    if not unexplained:
        return set()
    index: dict[tuple[int, int], list[Box]] = {}
    for b in unexplained:
        index.setdefault((round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)), []).append(b)
    pattern_clips = set()
    for c, scissor in enumerate(scissors):
        if scissor is None:
            continue
        u = (pymupdf.Rect(scissor) * to_user).normalize()
        cx, cy = round((u.x0 + u.x1) / 2), round((u.y0 + u.y1) / 2)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for b in index.get((cx + dx, cy + dy), ()):
                    grown = (u.x0 - b[0], u.y0 - b[1], b[2] - u.x1, b[3] - u.y1)
                    if min(grown) >= -1 and max(grown) - min(grown) <= 1:
                        pattern_clips.add(c)
    matched = set(match.values())
    return {k for k, e in enumerate(found) if k not in matched and pattern_clips.intersection(e.clips)}


def plan(page: pymupdf.Page, zones: list[pymupdf.Rect], drawn: list[pymupdf.Rect] = ()) -> Plan:
    """Which stroked paths of the page leave the file, and which ones the zones must cover entirely.

    ``drawn``: the zones (among ``zones``) whose content may be a drawing (signatures, zones drawn
    by the reviewer). The content filter runs once without changing anything, to know which paths
    the page itself paints (``_pattern_cells``).
    """
    found, scissors = paths(page)
    remove: set[int] = set()
    whole: set[int] = set()
    enlarge: list[tuple[int, int]] = []
    zones = list(zones)
    probe = _Culler()
    if not any(
        "s" in (e.d.get("type") or "") and e.extent is not None and any(_touches(e.extent, z) for z in zones)
        for e in found
    ):
        return Plan(found, remove, whole, enlarge, probe.calls, probe.other)  # no stroke near a zone
    _filter(page, probe, update=False)
    to_user = ~page.transformation_matrix
    try:
        match = _match(probe.calls, found, to_user, time.monotonic() + REMOVE_SECONDS)
        cells = _pattern_cells(found, scissors, probe.calls, match, to_user)
    except TimeoutError:
        log.warning("page %d: lining up %d painted paths took too long", page.number, len(probe.calls))
        cells = set()
    drawing = [any(z == d for d in drawn) for z in zones]
    for i, (d, box, extent, _) in enumerate(found):
        if "s" not in (d.get("type") or "") or box is None or i in cells:
            continue
        if any(_inside(box, z, MARGIN) for z in zones):
            remove.add(i)
            continue
        hit = [k for k, z in enumerate(zones) if _touches(extent, z)]
        if not hit:
            continue
        items = d.get("items") or []
        if _is_layout(items) or closed_convex(items):
            continue
        share = _share_inside(items, extent, zones)
        drawing_hit = [k for k in hit if drawing[k]]
        if drawing_hit and is_pen_stroke(items):
            if share >= MOSTLY:
                remove.add(i)
                whole.add(i)
                continue
            if share > 0 and _turns_back(items):
                enlarge.append((i, drawing_hit[0]))
                continue
        if share >= UNDER:
            enlarge.append((i, hit[0]))
    return Plan(found, remove, whole, enlarge, probe.calls, probe.other)


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


class _Culler(mupdf.PdfSanitizeFilterOptions2):
    """MuPDF's culler callback: records the painted paths (``drop`` None) or drops some of them by
    their position among the painted paths."""

    def __init__(self, drop: set[int] | None = None):
        super().__init__()
        self.drop = drop
        self.calls: list[tuple[int, Box]] = []
        self.other: Counter = Counter()  # glyphs, images, shadings and clips painted: never dropped
        self.use_virtual_culler()

    def culler(self, ctx, bbox, kind):  # noqa: ARG002 - MuPDF's callback signature
        if kind not in _PATH_KINDS:
            self.other[int(kind)] += 1
            return 0
        n = len(self.calls)
        r = mupdf.FzRect(bbox)
        self.calls.append((int(kind), (r.x0, r.y0, r.x1, r.y1)))
        return 1 if self.drop is not None and n in self.drop else 0


def _filter(page: pymupdf.Page, culler: _Culler, update: bool) -> None:
    # recurse=0: Type3 glyph procedures (shared by every page that uses the font) and the cells of
    # patterns are never entered; instance_forms=1: each use of a form is filtered as its own copy,
    # so dropping a path in one use does not touch the others.
    options = pymupdf._make_PdfFilterOptions(
        recurse=0, instance_forms=1, sanitize=1, no_update=0 if update else 1, sopts=culler
    )
    pdf_page = pymupdf._as_pdf_page(page.this)
    mupdf.pdf_filter_page_contents(pdf_page.doc(), pdf_page, options)


def _fits(kind: int, box: Box, path: tuple[str, Box, float]) -> bool:
    """The filter's call and the path can be the same: kinds agree, and the call's box is the path's
    box (a fill), or the path's box grown evenly on every side (a stroke: MuPDF grows it by half the
    line width, or by the width times the miter limit for mitred joins, whatever the limit)."""
    kind_text, u, width = path
    if kind == _FILL:
        return "f" in kind_text and all(abs(a - b) <= _TOLERANCE for a, b in zip(box, u, strict=True))
    if (kind == _STROKE and "s" not in kind_text) or (kind == _FILL_STROKE and kind_text != "fs"):
        return False
    grown = (u[0] - box[0], u[1] - box[1], box[2] - u[2], box[3] - u[3])
    low, high = min(grown), max(grown)
    return low >= -_TOLERANCE and high - low <= _TOLERANCE and high <= max(width, 1.0) * 1000


def _match(calls: list[tuple[int, Box]], found: list[Entry], to_user, deadline: float | None = None) -> dict[int, int]:
    """Filter call -> the path of ``found`` it paints, for every call that matches one.

    Both lists are in content order. Each call is matched to the first path at or after the last
    match whose box fits it, looked up by its center (a stroke's box grows evenly, so the center
    stays): a path that ``get_drawings`` lists and the filter does not report (a Type3 glyph, a
    pattern's cell) is skipped, and a call that matches no path (a pattern fill) is left out,
    without scanning the page for it. A fill followed by a stroke of the same path is one "fs" path
    in ``get_drawings`` and two calls here. Raises ``TimeoutError`` after ``deadline`` (monotonic).
    """
    info: dict[int, tuple[str, Box, float]] = {}
    buckets: dict[tuple[int, int], list[int]] = {}
    for k, entry in enumerate(found):
        d = entry[0]
        if entry[1] is None:  # clipped away: whatever paints it, it shows nothing
            continue
        u = (pymupdf.Rect(d["rect"]) * to_user).normalize()
        info[k] = (d.get("type") or "", (u.x0, u.y0, u.x1, u.y1), float(d.get("width") or 0))
        buckets.setdefault((round((u.x0 + u.x1) / 2), round((u.y0 + u.y1) / 2)), []).append(k)
    match: dict[int, int] = {}
    j = 0
    for n, (kind, box) in enumerate(calls):
        if deadline is not None and n % 256 == 0 and time.monotonic() > deadline:
            raise TimeoutError("lining up the content filter took too long")
        cx, cy = round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)
        best: int | None = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = buckets.get((cx + dx, cy + dy))
                if not bucket:
                    continue
                for k in bucket[bisect_left(bucket, j) :]:
                    if best is not None and k >= best:
                        break
                    if _fits(kind, box, info[k]):
                        best = k
                        break
        if best is None:
            continue
        match[n] = best
        j = best if kind == _FILL and info[best][0] == "fs" else best + 1
    return match


def _align(
    calls: list[tuple[int, Box]], found: list[Entry], targets: set[int], to_user, deadline: float | None = None
) -> dict[int, int]:
    """Filter calls to drop -> the path each one paints, for the paths in ``targets`` (``_match``)."""
    return {n: k for n, k in _match(calls, found, to_user, deadline).items() if k in targets}


def _key(d: dict) -> tuple:
    """What identifies a path when comparing a page before and after the filter (which rewrites the
    content, so coordinates are compared to a hundredth of a point)."""
    x0, y0, x1, y1 = d["rect"]  # a Rect (get_drawings) or a tuple (get_cdrawings)
    return (
        d.get("type"),
        len(d.get("items") or ()),
        round(x0, 2),
        round(y0, 2),
        round(x1, 2),
        round(y1, 2),
        round(d.get("width") or 0, 2),
        d.get("color"),
        d.get("fill"),
    )


def _paths_and_text(page: pymupdf.Page, found: list[Entry] | None = None) -> tuple[Counter, str]:
    """The page's paths and text. ``found``: its paths already listed (``paths``)."""
    drawn = [entry[0] for entry in found] if found is not None else page.get_cdrawings()
    return Counter(map(_key, drawn)), page.get_text()


def remove(
    page: pymupdf.Page, zones: list[pymupdf.Rect], drawn: list[pymupdf.Rect] = (), whole: list[Box] | None = None
) -> int:
    """Removes from the page's content the stroked paths ``plan`` says must leave. Returns how many.

    Call on an unrotated page. ``whole``: receives the drawn extent of each path removed whole
    although part of it lay outside the zones (it changes what the page shows there). When the
    result is not exactly the page minus those paths (same other drawings, text and images), or
    lining the filter up takes longer than ``REMOVE_SECONDS``, the page is left as it was and 0 is
    returned: the leak check then finds the paths still there and blocks the export.
    """
    planned = plan(page, zones, drawn)
    if not planned.remove:
        return 0
    to_user = ~page.transformation_matrix
    try:
        drop = _align(planned.calls, planned.paths, planned.remove, to_user, time.monotonic() + REMOVE_SECONDS)
    except TimeoutError:
        log.warning(
            "page %d: lining up %d painted paths took too long; nothing removed", page.number, len(planned.calls)
        )
        return 0
    if not drop:
        return 0
    doc = page.parent
    saved = {key: doc.xref_get_key(page.xref, key) for key in ("Contents", "Resources")}
    drawings, text = _paths_and_text(page, planned.paths)
    removed = set(drop.values())
    expected = drawings - Counter(_key(planned.paths[k][0]) for k in removed)
    culler = _Culler(set(drop))
    _filter(page, culler, update=True)
    # The filter painted the same paths, glyphs and images, and the page shows exactly what was
    # expected: the same paths but the removed ones, the same text.
    same_calls = [c[0] for c in culler.calls] == [c[0] for c in planned.calls] and culler.other == planned.other
    if same_calls and _paths_and_text(page) == (expected, text):
        if whole is not None:
            whole += [planned.paths[k][2] for k in sorted(removed & planned.whole) if planned.paths[k][2]]
        return len(removed)
    log.warning("page %d: removing drawn strokes changed something else; the page is left as it was", page.number)
    for key, (kind, value) in saved.items():
        doc.xref_set_key(page.xref, key, "null" if kind == "null" else value)
    if _paths_and_text(page) != (drawings, text):
        raise RuntimeError("the page could not be restored after a failed stroke removal")
    return 0


def _self_test_document() -> pymupdf.Document:
    """Two pages: a curve in a form placed under a zone on page 1 and outside it on page 2, a frame
    around the zone, and a Type3 glyph drawn with a stroke on both pages."""
    doc = pymupdf.open()
    form = doc.get_new_xref()
    doc.update_object(form, "<< /Type /XObject /Subtype /Form /BBox [0 0 100 40] >>")
    doc.update_stream(form, b"0 0 0.5 RG 1 w 10 20 m 30 40 60 0 90 20 c S")
    proc = doc.get_new_xref()
    doc.update_object(proc, "<<>>")
    doc.update_stream(proc, b"500 0 d0 20 w 0 0 m 100 400 300 -100 400 200 c S")
    font = doc.get_new_xref()
    doc.update_object(
        font,
        f"<< /Type /Font /Subtype /Type3 /FontBBox [0 -200 500 500] /FontMatrix [0.001 0 0 0.001 0 0] "
        f"/CharProcs << /a {proc} 0 R >> /Encoding << /Type /Encoding /Differences [97 /a] >> "
        f"/FirstChar 97 /LastChar 97 /Widths [500] /Resources << >> >>",
    )
    for _ in range(2):
        page = doc.new_page(width=200, height=200)
        contents = doc.get_new_xref()
        doc.update_object(contents, "<<>>")
        doc.update_stream(
            contents,
            b"q 1 0 0 1 50 80 cm /F1 Do Q 0 0 0 RG 1 w 40 70 120 60 re S BT /T3 20 Tf 20 20 Td (a) Tj ET",
        )
        doc.xref_set_key(page.xref, "Contents", f"{contents} 0 R")
        doc.xref_set_key(page.xref, "Resources", f"<< /XObject << /F1 {form} 0 R >> /Font << /T3 {font} 0 R >> >>")
    return doc


def self_test() -> bool:
    """The content filter still removes exactly a stroke under a zone: not the frame around it, not
    the same form's use on another page, not a Type3 glyph drawn with a stroke.

    It relies on private PyMuPDF bindings: a PyMuPDF update could break it, so the engine checks it
    at startup and refuses to run without it.
    """

    def described(page: pymupdf.Page) -> list[tuple]:
        return sorted(
            (d["type"], "".join(item[0] for item in d["items"]), tuple(round(v, 2) for v in d["rect"]))
            for d in page.get_drawings()
        )

    try:
        with _self_test_document() as doc:
            other = described(doc[1])
            glyph = [k for k in described(doc[0]) if k[1] == "c" and k[2][1] > 150]
            removed = remove(doc[0], [pymupdf.Rect(55, 75, 145, 125)])
            left = described(doc[0])
            ok = (
                removed == 1
                and [(k[0], k[1]) for k in left] == [("s", "c"), ("s", "re")]  # the glyph and the frame
                and [k for k in left if k[1] == "c"] == glyph
                and described(doc[1]) == other
            )
        return ok
    except Exception:  # noqa: BLE001 - any failure means the filter cannot be trusted
        log.exception("the stroke removal self-test failed")
        return False


# ---------------------------------------------------------------------------
# Leak check
# ---------------------------------------------------------------------------


def _label(f: Finding) -> str:
    return "Zona dibujada" if f.type == "manual" else TYPE_LABELS.get(f.type, f.type)


def leaks(doc: pymupdf.Document, active: list[Finding]) -> list[Leak]:
    """Strokes still in an exported PDF that should have left with the zones of ``active``, or that
    the zones must cover entirely (the same rectangles and rules as the redaction). Call with
    ``PDF_LOCK`` held."""
    out: list[Leak] = []
    to_page = [pymupdf.Matrix(page.derotation_matrix) for page in doc]
    for n, items in sorted(zones_by_page(active, to_page).items()):
        zones = [r for r, _ in items]
        planned = plan(doc[n], zones, [r for r, f in items if f.type in DRAWN_TYPES])
        reported: set[tuple[str, str]] = set()
        for k in sorted(planned.remove):
            d, box = planned.paths[k].d, planned.paths[k].box
            if _is_layout(d.get("items") or []):  # frames, rules and the black boxes themselves
                continue
            holder = next((f for r, f in items if _inside(box, r, MARGIN)), None) or next(
                (f for r, f in items if _touches(box, r)), items[0][1]
            )
            if (holder.id, "kept") in reported:
                continue
            reported.add((holder.id, "kept"))
            out.append(
                Leak(
                    page=n,
                    type=holder.type,
                    message=f"{_label(holder)}: un trazo dibujado sigue en el archivo bajo la zona de la página "
                    f"{n + 1} y no se pudo quitar sin alterar el resto de la página. No publiques este archivo: "
                    "publica esa página escaneada o impresa como imagen.",
                    finding_id=holder.id,
                )
            )
        for _, z in planned.enlarge:
            f = items[z][1]
            if (f.id, "enlarge") in reported:
                continue
            reported.add((f.id, "enlarge"))
            out.append(
                Leak(
                    page=n,
                    type=f.type,
                    message=f"{_label(f)}: un trazo dibujado cruza el borde de la zona de la página {n + 1} y la parte "
                    "tapada sigue en el archivo; agranda la zona para cubrirlo entero.",
                    finding_id=f.id,
                )
            )
    return out
