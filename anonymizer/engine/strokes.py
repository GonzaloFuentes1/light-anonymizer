"""Stroked vector paths under redaction zones: which ones leave the file, removing them, and the leak check.

MuPDF's redaction removes the *filled* paths a zone covers but keeps every *stroked* one: a
signature drawn with a pen tool stayed in the file under the black box. Its option to remove every
path a zone touches also removes page frames, table shading and background bands. Stroked paths
are therefore handled here, apart from MuPDF's redaction:

- ``plan`` decides with the drawn extent of each stroked path (its box grown by half its line
  width, cut to the clip that shows it). A path drawn entirely inside a zone goes. A pen stroke (a
  curve, or a polyline that is not a grid) that crosses the edge of a zone where drawings are the
  data (a signature, a zone drawn by the reviewer) goes whole when most of its visible length lies
  inside the zones, and is otherwise a problem for the reviewer: the zone must be enlarged.
- ``remove`` takes them out of the page's content with MuPDF's content filter. The filter only
  reports, in content order, the box of each painted path (grown for its stroke), so its calls are
  lined up with ``page.get_drawings()`` (same order, consistent boxes) and only the calls matched
  to a planned path are dropped: never a glyph, an image, a Type3 glyph procedure (the filter does
  not enter them) or a path it cannot match. Then the page's drawings, text and images are compared
  with what was expected; on any other difference the page is put back as it was.
- ``leaks`` runs ``plan`` again over the exported file with the same zones: a pen stroke that
  should have left and is still there, or one that crosses the edge of a drawing zone, blocks the
  export with a message that says what to do.

Coordinates: PyMuPDF's unrotated page space (that of ``get_drawings`` and ``add_redact_annot``).
The filter's boxes are in PDF user space (``~page.transformation_matrix``). This module uses
private PyMuPDF bindings (``_make_PdfFilterOptions``, ``_as_pdf_page``) and MuPDF's culler
callback: ``self_test`` checks at startup that they still behave as expected.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pymupdf

from anonymizer.engine.common import bbox_of
from anonymizer.engine.model import TYPE_LABELS, Finding, Leak
from anonymizer.engine.signatures import path_shape

log = logging.getLogger(__name__)

mupdf = pymupdf.mupdf
_FILL, _STROKE, _FILL_STROKE = mupdf.FZ_CULL_PATH_FILL, mupdf.FZ_CULL_PATH_STROKE, mupdf.FZ_CULL_PATH_FILL_STROKE
_PATH_KINDS = (_FILL, _STROKE, _FILL_STROKE)

# Zones whose content may be a drawing: a pen stroke that crosses their edge matters.
DRAWN_TYPES = ("signature", "manual")
MOSTLY = 0.6  # share of a crossing pen stroke's visible length inside the zones for it to go whole
_TOLERANCE = 0.5  # points: the filter's boxes and get_drawings' rectangles agree within this

Box = tuple[float, float, float, float]


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
    signature detector's criterion): what a pen leaves, unlike rules, frames and grids."""
    curves, segments, _, straight, _ = path_shape(items)
    return curves >= 1 or (segments >= 6 and straight < 0.6)


def _intersect(a: Box, b: Box) -> Box | None:
    r = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return r if r[0] <= r[2] and r[1] <= r[3] else None


def _inside(r: Box, zone: pymupdf.Rect) -> bool:
    return zone.x0 <= r[0] and zone.y0 <= r[1] and r[2] <= zone.x1 and r[3] <= zone.y1


def _touches(r: Box, zone: pymupdf.Rect) -> bool:
    return r[0] < zone.x1 and zone.x0 < r[2] and r[1] < zone.y1 and zone.y0 < r[3]


def paths(page: pymupdf.Page) -> list[tuple[dict, Box | None]]:
    """The page's paths (``get_drawings``, in content order), each with its drawn extent: its box
    grown by half its line width (strokes) and cut to the clip that shows it (None: clipped away)."""
    out: list[tuple[dict, Box | None]] = []
    clips: list[tuple[int, Box | None]] = []  # (level, scissor) of the clips in effect
    for d in page.get_drawings(extended=True):
        kind = d.get("type") or ""
        level = int(d.get("level") or 0)
        while clips and clips[-1][0] >= level:
            clips.pop()
        if kind == "clip":
            s = d.get("scissor")
            scissor = None if s is None else (s.x0, s.y0, s.x1, s.y1)
            if clips and scissor is not None and clips[-1][1] is not None:
                scissor = _intersect(scissor, clips[-1][1])
            clips.append((level, scissor))
            continue
        if kind == "group" or "rect" not in d:
            continue
        half = max(float(d.get("width") or 0), 0.5) / 2 if "s" in kind else 0.0
        r = d["rect"]
        drawn: Box | None = (r.x0 - half, r.y0 - half, r.x1 + half, r.y1 + half)
        if clips and clips[-1][1] is not None and drawn is not None:
            drawn = _intersect(drawn, clips[-1][1])
        out.append((d, drawn))
    return out


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
    page: pymupdf.Page, zones: list[pymupdf.Rect], drawn: list[pymupdf.Rect] = ()
) -> tuple[list[tuple[dict, Box | None]], set[int], list[tuple[int, int]]]:
    """Which stroked paths of the page must leave the file.

    Returns the page's paths (``paths``), the indexes of those to remove (drawn entirely inside a
    zone; or pen strokes that cross a zone of ``drawn`` with most of their visible length inside the
    zones), and the pen strokes that cross a zone of ``drawn`` with most of it outside, as (path
    index, index in ``drawn``): the reviewer must enlarge that zone.
    """
    found = paths(page)
    remove: set[int] = set()
    crossing: list[tuple[int, int]] = []
    for i, (d, visible) in enumerate(found):
        if "s" not in (d.get("type") or "") or visible is None:
            continue
        if any(_inside(visible, z) for z in zones):
            remove.add(i)
            continue
        hit = [k for k, z in enumerate(drawn) if _touches(visible, z)]
        if not hit or not is_pen_stroke(d["items"]):
            continue
        share = _share_inside(d["items"], visible, list(zones))
        if share >= MOSTLY:
            remove.add(i)
        elif share > 0:
            crossing.append((i, hit[0]))
    return found, remove, crossing


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
        self.use_virtual_culler()

    def culler(self, ctx, bbox, kind):  # noqa: ARG002 - MuPDF's callback signature
        if kind not in _PATH_KINDS:
            return 0
        n = len(self.calls)
        r = mupdf.FzRect(bbox)
        self.calls.append((int(kind), (r.x0, r.y0, r.x1, r.y1)))
        return 1 if self.drop is not None and n in self.drop else 0


def _filter(page: pymupdf.Page, culler: _Culler, update: bool) -> None:
    # recurse=0: Type3 glyph procedures (shared by every page that uses the font) are never
    # entered; instance_forms=1: each use of a form is filtered as its own copy, so dropping a path
    # in one use does not touch the others.
    options = pymupdf._make_PdfFilterOptions(
        recurse=0, instance_forms=1, sanitize=1, no_update=0 if update else 1, sopts=culler
    )
    pdf_page = pymupdf._as_pdf_page(page.this)
    mupdf.pdf_filter_page_contents(pdf_page.doc(), pdf_page, options)


def _matches(kind: int, box: Box, d: dict, to_user: pymupdf.Matrix) -> bool:
    """The filter's call and the path ``d`` can be the same: kinds agree, and the call's box is the
    path's box, grown evenly on every side for a stroke (MuPDF grows it by half the line width, or
    by the width times the miter limit for mitred joins)."""
    t = d.get("type") or ""
    u = (pymupdf.Rect(d["rect"]) * to_user).normalize()
    if kind == _FILL:
        return "f" in t and all(abs(a - b) <= _TOLERANCE for a, b in zip(box, (u.x0, u.y0, u.x1, u.y1), strict=True))
    if (kind == _STROKE and "s" not in t) or (kind == _FILL_STROKE and t != "fs"):
        return False
    grown = (u.x0 - box[0], u.y0 - box[1], box[2] - u.x1, box[3] - u.y1)
    width = float(d.get("width") or 0)
    return min(grown) >= -_TOLERANCE and max(grown) - min(grown) <= _TOLERANCE and max(grown) <= 20 * max(width, 1) + 1


def _align(
    calls: list[tuple[int, Box]], found: list[tuple[dict, Box | None]], targets: set[int], to_user: pymupdf.Matrix
) -> dict[int, int]:
    """Filter calls to drop -> the path each one paints, for the paths in ``targets``.

    Both lists are in content order. A path that ``get_drawings`` lists and the filter does not
    report (a Type3 glyph) is skipped; a call that matches no path is kept. A fill followed by a
    stroke of the same path is one "fs" path in ``get_drawings`` and two calls here.
    """
    drop: dict[int, int] = {}
    j = 0
    for n, (kind, box) in enumerate(calls):
        k = j
        while k < len(found) and not _matches(kind, box, found[k][0], to_user):
            k += 1
        if k == len(found):
            continue
        if k in targets:
            drop[n] = k
        j = k if kind == _FILL and found[k][0].get("type") == "fs" else k + 1
    return drop


def _key(d: dict) -> tuple:
    def rounded(values):
        return None if values is None else tuple(round(float(v), 2) for v in values)

    return (
        d.get("type"),
        tuple(item[0] for item in d.get("items") or ()),
        rounded(d["rect"]),
        round(float(d.get("width") or 0), 2),
        rounded(d.get("color")),
        rounded(d.get("fill")),
    )


def _state(page: pymupdf.Page) -> tuple[Counter, str, int]:
    return Counter(_key(d) for d in page.get_drawings()), page.get_text(), len(page.get_image_info())


def remove(page: pymupdf.Page, zones: list[pymupdf.Rect], drawn: list[pymupdf.Rect] = ()) -> int:
    """Removes from the page's content the stroked paths ``plan`` says must leave. Returns how many.

    Call on an unrotated page. When the result is not exactly the page minus those paths (same
    other drawings, text and images), the page is put back as it was and 0 is returned: the leak
    check then finds the paths still there and blocks the export.
    """
    found, targets, _ = plan(page, zones, drawn)
    if not targets:
        return 0
    to_user = ~page.transformation_matrix
    probe = _Culler()
    _filter(page, probe, update=False)
    drop = _align(probe.calls, found, targets, to_user)
    if not drop:
        return 0
    doc = page.parent
    saved = {key: doc.xref_get_key(page.xref, key) for key in ("Contents", "Resources")}
    drawings, text, images = _state(page)
    expected = drawings - Counter(_key(found[k][0]) for k in set(drop.values()))
    culler = _Culler(set(drop))
    _filter(page, culler, update=True)
    if len(culler.calls) == len(probe.calls) and _state(page) == (expected, text, images):
        return len(set(drop.values()))
    log.warning("page %d: removing drawn strokes changed something else; the page is left as it was", page.number)
    for key, (kind, value) in saved.items():
        doc.xref_set_key(page.xref, key, "null" if kind == "null" else value)
    if _state(page) != (drawings, text, images):
        raise RuntimeError("the page could not be restored after a failed stroke removal")
    return 0


def self_test() -> bool:
    """The content filter still removes exactly a stroke under a zone (and keeps a frame around it).

    It relies on private PyMuPDF bindings: a PyMuPDF update could break it, so the engine checks it
    at startup and refuses to run without it.
    """
    try:
        with pymupdf.open() as doc:
            page = doc.new_page(width=200, height=200)
            page.draw_bezier((60, 100), (80, 60), (120, 140), (140, 100), color=(0, 0, 0.5), width=1)
            page.draw_rect(pymupdf.Rect(50, 50, 150, 150), color=(0, 0, 0), width=1)
            removed = remove(page, [pymupdf.Rect(55, 55, 145, 145)])
            left = [(d["type"], "".join(i[0] for i in d["items"])) for d in page.get_drawings()]
        return removed == 1 and left == [("s", "re")]
    except Exception:  # noqa: BLE001 - any failure means the filter cannot be trusted
        log.exception("the stroke removal self-test failed")
        return False


# ---------------------------------------------------------------------------
# Leak check
# ---------------------------------------------------------------------------


def _label(f: Finding) -> str:
    return "Zona dibujada" if f.type == "manual" else TYPE_LABELS.get(f.type, f.type)


def leaks(doc: pymupdf.Document, active: list[Finding]) -> list[Leak]:
    """Pen strokes still in an exported PDF that should have left with the zones of ``active`` (the
    same rectangles and rules the redaction used). Call with ``PDF_LOCK`` held."""
    out: list[Leak] = []
    to_page = [pymupdf.Matrix(page.derotation_matrix) for page in doc]
    for n, items in sorted(zones_by_page(active, to_page).items()):
        page = doc[n]
        zones = [r for r, _ in items]
        drawn_items = [(r, f) for r, f in items if f.type in DRAWN_TYPES]
        found, remove_, crossing = plan(page, zones, [r for r, _ in drawn_items])
        reported: set[tuple[str, str]] = set()
        for k in sorted(remove_):
            d, visible = found[k]
            if not is_pen_stroke(d["items"]):
                continue
            holder = next((f for r, f in items if _inside(visible, r)), None) or next(
                (f for r, f in items if _touches(visible, r)), items[0][1]
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
        for _, z in crossing:
            f = drawn_items[z][1]
            if (f.id, "crossing") in reported:
                continue
            reported.add((f.id, "crossing"))
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
