"""Stroked vector paths under redaction zones: which ones leave the file, removing them, and the leak check.

MuPDF's redaction removes the *filled* paths a zone covers but keeps every *stroked* one: a
signature drawn with a pen tool stayed in the file under the black box. Its option to remove every
path a zone touches also removes page frames, table shading and background bands. Stroked paths
are therefore handled here, apart from MuPDF's redaction.

The rule is that of the whole tool: recall over precision. The stroke logic stays silent only when
it is certain that a stroke is not under an applied zone, or that it left the file; whenever a
classification would be a guess, the export is blocked with a message that says what to do.

- ``plan`` judges every stroked path by its box, cut to the clip that shows it (a path clipped
  away entirely is judged by where it is drawn: its data is still in the file). A path whose box
  lies inside a zone (one point of margin) leaves the file. Straight rules and rectangles (page
  layout) stay, and so do closed convex outlines (rings, ovals, rounded frames: stamps, frames)
  that lie mostly outside the zone. Any other curve or polyline that crosses the edge of a zone
  where drawings are the data (a signature, a zone drawn by the reviewer) leaves whole when at
  least 60 % of its visible length lies inside the zones, and blocks the export from 20 %: the
  reviewer must enlarge the zone. Any other stroke with at least 80 % under the zones blocks too.
  Nothing is exempted for being a pattern's cell or a glyph: ``get_drawings`` lists them as paths,
  and what the redaction cannot remove (MuPDF removes a pattern fill or a glyph it covers) blocks.
- ``remove`` takes them out of the page's content with MuPDF's content filter. The filter only
  reports, in content order, the box of each painted path (grown for its stroke), so its calls are
  lined up with ``page.get_drawings(extended=True)``, clips included (a shape filled with a pattern
  is listed as a clip with its box), through an index of boxes; only the calls matched to a planned
  path are dropped. The filter never enters Type3 glyph procedures or pattern cells. Then the page's
  paths and text, and what the filter painted, are compared with what was expected; on any other
  difference the page is put back as it was.
- ``leaks`` runs ``plan`` again over the exported file with the same zones and rules: a stroke that
  should have left and is still there, or one a zone must cover entirely, blocks the export.

Each page has a time budget (``REMOVE_SECONDS``) for the whole stage; past it nothing is removed
and the export is blocked. Coordinates: PyMuPDF's unrotated page space (that of ``get_drawings``
and ``add_redact_annot``); the filter's boxes are in PDF user space (``~page.transformation_matrix``).
This module uses private PyMuPDF bindings (``_make_PdfFilterOptions``, ``_as_pdf_page``) and
MuPDF's culler callback: ``self_test`` checks at startup that they still behave as expected, and
PyMuPDF is pinned below 1.29.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_left, bisect_right
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
_PATH_KINDS = (_FILL, _STROKE, _FILL_STROKE)  # the only calls ever dropped
# What each call of the filter paints: a path ("fill", "stroke", "fs"), a clip without painting
# ("clip", from "W n"), or a path that also clips ("W f", "W S", "W B": painted, never dropped).
_CLASS = {
    _FILL: "fill",
    _STROKE: "stroke",
    _FILL_STROKE: "fs",
    mupdf.FZ_CULL_CLIP_PATH_DROP: "clip",
    mupdf.FZ_CULL_CLIP_PATH_FILL: "fill",
    mupdf.FZ_CULL_CLIP_PATH_STROKE: "stroke",
    mupdf.FZ_CULL_CLIP_PATH_FILL_STROKE: "fs",
}
_CLIPPING = (mupdf.FZ_CULL_CLIP_PATH_FILL, mupdf.FZ_CULL_CLIP_PATH_STROKE, mupdf.FZ_CULL_CLIP_PATH_FILL_STROKE)

# Zones whose content may be a drawing: a stroke that crosses their edge matters.
DRAWN_TYPES = ("signature", "manual")
MOSTLY = 0.6  # share of a stroke's visible length under a drawing zone for it to go whole
CROSSING = 0.2  # share under a drawing zone from which a crossing stroke blocks the export
UNDER = 0.8  # share under any zone from which any other stroke blocks the export
AROUND = 0.5  # a closed convex outline with less than this under the zones is a frame or stamp around it
MARGIN = 1.0  # points: a path this close to a zone's edge counts as inside it
REMOVE_SECONDS = 15.0  # time budget of the stroke stage of one page (the export is blocked past it)
_TOLERANCE = 0.5  # points: the filter's boxes and get_drawings' rectangles agree within this

Box = tuple[float, float, float, float]


class Entry(NamedTuple):
    """A path of ``get_drawings``: its box and drawn extent (box grown by half the line width), cut
    to the clip that shows it; for a path clipped away entirely (``visible`` False), where it is
    drawn. ``seq``: its position in the extended listing (clips included); ``clip``: the position of
    the innermost clip it is under (-1: none)."""

    d: dict
    box: Box
    extent: Box
    visible: bool
    seq: int
    clip: int = -1


class Plan(NamedTuple):
    """What ``plan`` decided for the stroked paths of a page."""

    paths: list[Entry]
    remove: set[int]  # paths that leave the file
    whole: set[int]  # among them, those removed whole although part of them lies outside the zones
    enlarge: list[tuple[int, int]]  # (path, zone index): the zone must cover the whole stroke
    clips: list[tuple[int, Box]]  # (seq, scissor) of every clip of the listing


def _check(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("the stroke stage took too long")


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


def _pen_shape(items) -> bool:
    """A curve, or a polyline of six segments or more that is not mostly axis-parallel (the
    signature detector's criterion)."""
    curves, segments, _, straight, _ = path_shape(items)
    return curves >= 1 or (segments >= 6 and straight < 0.6)


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


def paths(page: pymupdf.Page, deadline: float | None = None) -> tuple[list[Entry], list[tuple[int, Box]]]:
    """The page's paths (``get_drawings``, in content order) and its clips, as (seq, scissor)."""
    out: list[Entry] = []
    clips: list[tuple[int, Box]] = []
    stack: list[tuple[int, Box | None, int]] = []  # (level, scissor cut to the outer clips, seq)
    for seq, d in enumerate(page.get_drawings(extended=True)):
        if seq % 1024 == 0:
            _check(deadline)
        kind = d.get("type") or ""
        level = int(d.get("level") or 0)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if kind == "clip":
            s = d.get("scissor")
            scissor = None if s is None else (s.x0, s.y0, s.x1, s.y1)
            if scissor is not None:
                clips.append((seq, scissor))
            if stack and scissor is not None and stack[-1][1] is not None:
                scissor = _intersect(scissor, stack[-1][1])
            stack.append((level, scissor, seq))
            continue
        if kind == "group" or "rect" not in d:
            continue
        half = max(float(d.get("width") or 0), 0.5) / 2 if "s" in kind else 0.0
        r = d["rect"]
        box: Box = (r.x0, r.y0, r.x1, r.y1)
        extent: Box = (r.x0 - half, r.y0 - half, r.x1 + half, r.y1 + half)
        visible = True
        if stack and stack[-1][1] is not None:
            cut = _intersect(box, stack[-1][1])
            if cut is None:  # clipped away: judged by where it is drawn
                visible = False
            else:
                box, extent = cut, _intersect(extent, stack[-1][1]) or cut
        out.append(Entry(d, box, extent, visible, seq, stack[-1][2] if stack else -1))
    return out, clips


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


def plan(
    page: pymupdf.Page, zones: list[pymupdf.Rect], drawn: list[pymupdf.Rect] = (), deadline: float | None = None
) -> Plan:
    """Which stroked paths of the page leave the file, and which ones the zones must cover entirely.

    ``drawn``: the zones (among ``zones``) whose content may be a drawing (signatures, zones drawn
    by the reviewer). Raises ``TimeoutError`` past ``deadline`` (monotonic).
    """
    found, clips = paths(page, deadline)
    remove: set[int] = set()
    whole: set[int] = set()
    enlarge: list[tuple[int, int]] = []
    zones = list(zones)
    drawing = [any(z == d for d in drawn) for z in zones]
    for i, (d, box, extent, _, _, _) in enumerate(found):
        if i % 256 == 0:
            _check(deadline)
        if "s" not in (d.get("type") or ""):
            continue
        if any(_inside(box, z, MARGIN) for z in zones):
            remove.add(i)
            continue
        hit = [k for k, z in enumerate(zones) if _touches(extent, z)]
        if not hit:
            continue
        items = d.get("items") or []
        if _is_layout(items):
            continue
        share = _share_inside(items, extent, zones)
        convex = closed_convex(items)
        if convex and share < AROUND:
            continue  # a frame or a stamp around the zone
        drawing_hit = [k for k in hit if drawing[k]]
        if drawing_hit and (convex or _pen_shape(items)):
            if share >= MOSTLY:
                remove.add(i)
                whole.add(i)
                continue
            if share >= CROSSING:
                enlarge.append((i, drawing_hit[0]))
                continue
        if share >= UNDER:
            enlarge.append((i, hit[0]))
    return Plan(found, remove, whole, enlarge, clips)


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
        if kind not in _CLASS:
            self.other[int(kind)] += 1
            return 0
        n = len(self.calls)
        r = mupdf.FzRect(bbox)
        self.calls.append((int(kind), (r.x0, r.y0, r.x1, r.y1)))
        return 1 if self.drop is not None and n in self.drop and kind in _PATH_KINDS else 0


def _filter(page: pymupdf.Page, culler: _Culler, update: bool) -> None:
    # recurse=0: Type3 glyph procedures (shared by every page that uses the font) and the cells of
    # patterns are never entered; instance_forms=1: each use of a form is filtered as its own copy,
    # so dropping a path in one use does not touch the others.
    options = pymupdf._make_PdfFilterOptions(
        recurse=0, instance_forms=1, sanitize=1, no_update=0 if update else 1, sopts=culler
    )
    pdf_page = pymupdf._as_pdf_page(page.this)
    mupdf.pdf_filter_page_contents(pdf_page.doc(), pdf_page, options)


def _fits(call: str, box: Box, kind_text: str, u: Box) -> bool:
    """The filter's call (``_CLASS``) and a listed path (or clip, ``kind_text`` "clip") can be the
    same: kinds agree, and the call's box is the path's box (a fill, a clip), or the path's box grown
    evenly on every side (a stroke: MuPDF grows it by half the line width, or by the width times the
    miter limit for mitred joins; the miter limit is not listed, so any even growth fits)."""
    if call == "clip":
        return kind_text == "clip" and all(abs(a - b) <= _TOLERANCE for a, b in zip(box, u, strict=True))
    if call == "fill":
        if "f" not in kind_text and kind_text != "clip":
            return False
        return all(abs(a - b) <= _TOLERANCE for a, b in zip(box, u, strict=True))
    if kind_text != "clip" and ((call == "stroke" and "s" not in kind_text) or (call == "fs" and kind_text != "fs")):
        return False
    grown = (u[0] - box[0], u[1] - box[1], box[2] - u[2], box[3] - u[3])
    return min(grown) >= -_TOLERANCE and max(grown) - min(grown) <= _TOLERANCE


def _subpaths(items) -> list[Box]:
    """Boxes of the subpaths of a path (the filter reports each subpath as a call of its own): a new
    subpath starts where a segment does not continue the last one, and every rectangle is one."""
    boxes: list[list[float]] = []
    last = None
    for item in items:
        kind = item[0]
        if kind == "re":
            r = item[1]
            boxes.append([r.x0, r.y0, r.x1, r.y1])
            last = None
            continue
        if kind == "qu":
            q = item[1]
            xs, ys = [q.ul.x, q.ur.x, q.ll.x, q.lr.x], [q.ul.y, q.ur.y, q.ll.y, q.lr.y]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            last = None
            continue
        pts = item[1:]
        if last is None or abs(pts[0].x - last.x) > 0.01 or abs(pts[0].y - last.y) > 0.01:
            boxes.append([pts[0].x, pts[0].y, pts[0].x, pts[0].y])
        b = boxes[-1]
        for pt in pts:
            b[0], b[1], b[2], b[3] = min(b[0], pt.x), min(b[1], pt.y), max(b[2], pt.x), max(b[3], pt.y)
        last = pts[-1]
    return [(b[0], b[1], b[2], b[3]) for b in boxes]


def _match(
    calls: list[tuple[int, Box]],
    found: list[Entry],
    to_user,
    deadline: float | None = None,
    clips: list[tuple[int, Box]] = (),
) -> dict[int, int]:
    """Filter call -> the path of ``found`` it paints (an index), or the clip of ``clips`` it matches
    (-1 - its index), for every call that matches one.

    The filter reports one call per subpath, in content order; ``get_drawings`` lists whole paths
    and clips, in the same order. A clip without painting ("W n") takes its clip entry; a path that
    also clips takes its subpath and the clip entry of the path's box. A shape filled or stroked with
    a pattern is listed only as a clip with its box (its cell's paths are listed under it and never
    reported by the filter), so its call takes that clip. Each call goes to the first candidate after
    the last match whose box fits; a path listed and never reported (a Type3 glyph, a pattern's cell)
    is passed over, and a call that fits nothing is left out. A fill followed by a stroke of the same
    path is one "fs" path in ``get_drawings`` and two series of calls here. Candidates come from an
    index of box centers and sizes (a stroke grows its box evenly, which keeps its center and the
    difference of its sides), never from scanning the page.
    """
    fills: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}  # (cx, cy, w) -> [(seq, sub, id)]
    strokes_: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}  # (cx, cy, w - h) -> ...
    info: list[tuple[int, str, Box]] = []  # id -> (path index or -1 - clip index, kind, box in user space)
    whole: dict[int, Box] = {}  # path index -> its box in user space

    def user(b: Box) -> Box:
        r = (pymupdf.Rect(b) * to_user).normalize()
        return (r.x0, r.y0, r.x1, r.y1)

    def add(owner: int, seq: int, sub: int, kind_text: str, u: Box) -> None:
        info.append((owner, kind_text, u))
        key = len(info) - 1
        cx, cy = round((u[0] + u[2]) / 2), round((u[1] + u[3]) / 2)
        if "f" in kind_text or kind_text == "clip":
            fills.setdefault((cx, cy, round(u[2] - u[0])), []).append((seq, sub, key))
        if "s" in kind_text or kind_text == "clip":
            strokes_.setdefault((cx, cy, round((u[2] - u[0]) - (u[3] - u[1]))), []).append((seq, sub, key))

    for k, entry in enumerate(found):
        kind_text = entry.d.get("type") or ""
        whole[k] = user(tuple(entry.d["rect"]))
        for sub, b in enumerate(_subpaths(entry.d.get("items") or ()) or [tuple(entry.d["rect"])]):
            add(k, entry.seq, sub, kind_text, user(b))
    for c, (seq, scissor) in enumerate(clips):
        add(-1 - c, seq, 0, "clip", user(scissor))
    for index in (fills, strokes_):
        for bucket in index.values():
            bucket.sort()

    def first(call: str, box: Box, after: tuple[int, int], accept) -> tuple[int, int, int] | None:
        cx, cy = round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)
        if call in ("fill", "clip"):
            index, size = fills, round(box[2] - box[0])
        else:
            index, size = strokes_, round((box[2] - box[0]) - (box[3] - box[1]))
        best: tuple[int, int, int] | None = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ds in (-1, 0, 1):
                    bucket = index.get((cx + dx, cy + dy, size + ds))
                    if not bucket:
                        continue
                    for seq, sub, key in bucket[bisect_left(bucket, (after[0], after[1], -1)) :]:
                        if best is not None and (seq, sub) >= best[:2]:
                            break
                        owner, kind_text, u = info[key]
                        if accept(owner) and _fits(call, box, kind_text, u):
                            best = (seq, sub, key)
                            break
        return best

    used: set[int] = set()  # clip entries already taken
    painted = sorted(entry.seq for entry in found)  # positions of the painted paths in the listing
    match: dict[int, int] = {}
    j = (0, 0)  # position after the last match
    for n, (kind, box) in enumerate(calls):
        if n % 256 == 0:
            _check(deadline)
        call = _CLASS.get(kind)
        if call is None:
            continue
        if call == "clip":
            best = first(call, box, j, lambda o: o < 0 and o not in used)
        elif kind in _CLIPPING:
            best = first(call, box, j, lambda o: o >= 0)
        else:
            best = first(call, box, j, lambda o: o >= 0 or o not in used)
            if best is not None and info[best[2]][0] < 0:
                # A clip with the box of the next painted path (a form's bounding-box clip around
                # its filled and stroked frame) is not a pattern's clip: that path takes the call.
                path = first(call, box, j, lambda o: o >= 0)
                after = bisect_right(painted, best[0])
                if path is not None and (after == len(painted) or painted[after] >= path[0]):
                    best = path
        if best is None:
            continue
        seq, sub, key = best
        owner, kind_text, _ = info[key]
        match[n] = owner
        if owner < 0:
            used.add(owner)
        # A fill of an "fs" path stays at the path: the stroke of its subpaths follows.
        j = (seq, 0) if (owner >= 0 and call == "fill" and kind_text == "fs") else (seq, sub + 1)
        if kind in _CLIPPING and owner >= 0:  # the clip entry this path sets, listed next to it
            clip = first("clip", whole[owner], (seq - 2, 0), lambda o: o < 0 and o not in used)
            if clip is not None and clip[0] <= seq + 2:
                used.add(info[clip[2]][0])
                j = max(j, (clip[0], 1))
    return match


def _unlisted(
    page: pymupdf.Page, calls: list[tuple[int, Box]], match: dict[int, int], zones: list[pymupdf.Rect]
) -> list[tuple[int, Box, bool]]:
    """Painted paths ``get_drawings`` does not list (a shape filled or stroked with a pattern, listed
    only as a clip; anything it does not list at all) whose box touches a zone: (call, box in page
    space, whether the box lies inside a zone). Their box is the filter's, grown for the stroke, so
    "inside" is certain."""
    to_page = page.transformation_matrix
    out = []
    for n, (kind, box) in enumerate(calls):
        if _CLASS.get(kind) in (None, "clip") or match.get(n, -1) >= 0:
            continue
        r = (pymupdf.Rect(box) * to_page).normalize()
        b = (r.x0, r.y0, r.x1, r.y1)
        if any(_touches(b, z) for z in zones):
            out.append((n, b, any(_inside(b, z, MARGIN) for z in zones)))
    return out


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
    drawn = [entry.d for entry in found] if found is not None else page.get_cdrawings()
    return Counter(map(_key, drawn)), page.get_text()


def remove(
    page: pymupdf.Page, zones: list[pymupdf.Rect], drawn: list[pymupdf.Rect] = (), whole: list[Box] | None = None
) -> int:
    """Removes from the page's content the stroked paths ``plan`` says must leave. Returns how many.

    Call on an unrotated page. ``whole``: receives the drawn extent of each path removed whole
    although part of it lay outside the zones (it changes what the page shows there). When the
    result is not exactly the page minus those paths (same other paths and text, the filter painting
    the same things), or the stage takes longer than ``REMOVE_SECONDS``, the page is left as it was
    and 0 is returned: the leak check then finds the paths still there and blocks the export.
    """
    deadline = time.monotonic() + REMOVE_SECONDS
    try:
        planned = plan(page, zones, drawn, deadline)
        if not planned.remove:
            return 0
        probe = _Culler()
        _filter(page, probe, update=False)
        to_user = ~page.transformation_matrix
        match = _match(probe.calls, planned.paths, to_user, deadline, planned.clips)
    except TimeoutError:
        log.warning("page %d: the stroke stage took too long; nothing removed", page.number)
        return 0
    drop = {n: k for n, k in match.items() if k in planned.remove and probe.calls[n][0] in _PATH_KINDS}
    # A shape painted with a pattern, or a path get_drawings does not list, drawn entirely inside a
    # zone: its painted box is certain, so it goes too (with the pattern's clip and cell).
    gone_clips = set()
    for n, _, inside in _unlisted(page, probe.calls, match, zones):
        if inside and probe.calls[n][0] in _PATH_KINDS:
            drop[n] = match.get(n, -(10**9))
            if match.get(n, 0) < 0:
                gone_clips.add(planned.clips[-1 - match[n]][0])
    if not drop:
        return 0
    doc = page.parent
    saved = {key: doc.xref_get_key(page.xref, key) for key in ("Contents", "Resources")}
    drawings, text = _paths_and_text(page, planned.paths)
    removed = {k for k in drop.values() if k >= 0}
    gone = removed | {k for k, e in enumerate(planned.paths) if e.clip in gone_clips}
    expected = drawings - Counter(_key(planned.paths[k].d) for k in gone)
    culler = _Culler(set(drop))
    _filter(page, culler, update=True)
    # The filter painted the same paths, glyphs and images, and the page shows exactly what was
    # expected: the same paths but the removed ones, the same text.
    same_calls = [c[0] for c in culler.calls] == [c[0] for c in probe.calls] and culler.other == probe.other
    if same_calls and _paths_and_text(page) == (expected, text):
        if whole is not None:
            whole += [planned.paths[k].extent for k in sorted(removed & planned.whole)]
        return len(removed) + sum(1 for k in drop.values() if k < 0)
    log.warning("page %d: removing drawn strokes changed something else; the page is left as it was", page.number)
    for key, (kind, value) in saved.items():
        doc.xref_set_key(page.xref, key, "null" if kind == "null" else value)
    if _paths_and_text(page) != (drawings, text):
        raise RuntimeError("the page could not be restored after a failed stroke removal")
    return 0


def _self_test_document() -> pymupdf.Document:
    """Two pages: a curve in a form, placed under a zone (55, 75, 145, 125) on page 1 and at the same
    place on page 2, a frame around the zone, and a Type3 glyph drawn with a stroke inside the zone."""
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
            b"q 1 0 0 1 50 80 cm /F1 Do Q 0 0 0 RG 1 w 40 70 120 60 re S BT /T3 20 Tf 120 80 Td (a) Tj ET",
        )
        doc.xref_set_key(page.xref, "Contents", f"{contents} 0 R")
        doc.xref_set_key(page.xref, "Resources", f"<< /XObject << /F1 {form} 0 R >> /Font << /T3 {font} 0 R >> >>")
    return doc


def self_test() -> bool:
    """The content filter still removes exactly a stroke under a zone: not the frame around it, not
    the same form's use on another page, not a Type3 glyph drawn with a stroke inside the zone (the
    filter must not enter glyph procedures, which every page using the font shares: it would report
    one more painted path).

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
            zone = pymupdf.Rect(55, 75, 145, 125)
            other = described(doc[1])
            before = described(doc[0])
            glyph = [k for k in before if k[1] == "c" and _inside(k[2], zone)]
            probe = _Culler()
            _filter(doc[0], probe, update=False)
            removed = remove(doc[0], [zone])
            left = described(doc[0])
            ok = (
                len(glyph) == 2  # the form's curve and the glyph, both under the zone
                and sum(1 for k, _ in probe.calls if k in _PATH_KINDS) == 2  # the curve and the frame: no glyph entered
                and removed == 1
                and sorted((k[0], k[1]) for k in left) == [("s", "c"), ("s", "re")]  # the glyph and the frame
                and [k for k in left if k[1] == "c"] == [k for k in glyph if k[2][0] > 100]
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
    the zones must cover entirely (the same rectangles and rules as the redaction), and painted paths
    ``get_drawings`` does not list under a zone. Call with ``PDF_LOCK`` held; each page is looked at
    unrotated, like in the redaction (only in memory)."""
    out: list[Leak] = []
    to_page = [pymupdf.Matrix(page.derotation_matrix) for page in doc]
    for n, items in sorted(zones_by_page(active, to_page).items()):
        page = doc[n]
        rotation = page.rotation
        if rotation:
            page.set_rotation(0)
        try:
            out += _page_leaks(page, n, items)
        finally:
            if rotation:
                page.set_rotation(rotation)
    return out


def _page_leaks(page: pymupdf.Page, n: int, items: list[tuple[pymupdf.Rect, Finding]]) -> list[Leak]:
    zones = [r for r, _ in items]
    deadline = time.monotonic() + REMOVE_SECONDS
    try:
        planned = plan(page, zones, [r for r, f in items if f.type in DRAWN_TYPES], deadline)
        probe = _Culler()
        _filter(page, probe, update=False)
        match = _match(probe.calls, planned.paths, ~page.transformation_matrix, deadline, planned.clips)
    except TimeoutError:
        f = items[0][1]
        message = (
            f"La revisión de los trazos dibujados de la página {n + 1} no terminó a tiempo (la página tiene demasiados "
            "trazos). No publiques este archivo: publica esa página escaneada o impresa como imagen."
        )
        return [Leak(page=n, type=f.type, message=message, finding_id=f.id)]
    out: list[Leak] = []
    reported: set[tuple[str, str]] = set()

    def report(f: Finding, why: str, message: str) -> None:
        if (f.id, why) not in reported:
            reported.add((f.id, why))
            out.append(Leak(page=n, type=f.type, message=message, finding_id=f.id))

    for _, box, _ in _unlisted(page, probe.calls, match, zones):
        holder = next((f for r, f in items if _touches(box, r)), items[0][1])
        report(
            holder,
            "unlisted",
            f"{_label(holder)}: en la página {n + 1} queda bajo la zona un trazo o relleno con trama que no se puede "
            "revisar ni quitar por partes. Agranda la zona para cubrirlo entero o publica esa página escaneada o "
            "impresa como imagen.",
        )
    for k in sorted(planned.remove):
        d, box = planned.paths[k].d, planned.paths[k].box
        if _is_layout(d.get("items") or []):  # frames, rules and the black boxes themselves
            continue
        holder = next((f for r, f in items if _inside(box, r, MARGIN)), None) or next(
            (f for r, f in items if _touches(box, r)), items[0][1]
        )
        report(
            holder,
            "kept",
            f"{_label(holder)}: un trazo dibujado sigue en el archivo bajo la zona de la página {n + 1} y no se pudo "
            "quitar sin alterar el resto de la página. No publiques este archivo: publica esa página escaneada o "
            "impresa como imagen.",
        )
    for _, z in planned.enlarge:
        f = items[z][1]
        report(
            f,
            "enlarge",
            f"{_label(f)}: un trazo dibujado cruza el borde de la zona de la página {n + 1} y la parte tapada sigue en "
            "el archivo; agranda la zona para cubrirlo entero.",
        )
    return out
