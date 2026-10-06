"""Handwritten and drawn signatures, found by rules over ink strokes, keywords and lines.

No pretrained signature detector passed the license review (PLAN.md, section 5.7: every one we
found is trained on data whose copyright is unclear, such as Tobacco800), so these rules look for
what a signature leaves on a page:

- Raster (scanned pages, photos, images inside PDFs): pen strokes, that is, ink that printed text
  does not explain (OCR did not read it confidently, or it is much larger than the letters of the
  line that covers it), long, thin and curved, and neither a straight rule, a ring or a frame (a
  stamp), a grid (a table, a QR code), a filled shape nor part of a texture (a photo). A stroke
  counts near a signature keyword ("Firma", "Firmado", "V°B°", "p.p."), over a signature line of a
  document or inside a small image placed on a text page; next to a keyword, letters written apart
  count too. A stroke that OCR read as part of a line of text (a title in a script font) needs a
  keyword, or a signature line with the signer's name or role on its other side. On a plain sheet
  of paper a large curly stroke that OCR did not read counts by itself. In a table, the column
  under the header "Firma" is a zone whatever it holds. An image repeated at the same place on
  several pages (a letterhead emblem) is never a signature by itself. A zone covers the stroke and the
  marks that touch it (the letters of a name it crosses, their accents); a stamp pressed on the
  signature joins it.
- Vector (text pages of a PDF): clusters of curved stroked paths near a keyword or a signature
  line, or long and curly enough to be one by themselves; closed convex outlines (rings, ovals,
  rounded boxes: seals, radio buttons) are never part of one. Pages whose text reads vertically
  in PDF space (a landscape page stored as portrait plus /Rotate) are analyzed transposed.

Every finding is doubtful (``DOUBT_SIGNATURE``): the reviewer checks it against the original.
Regions are rectangles in the frame of the line that anchors them, so tilted photos and pages
scanned sideways work the same as upright pages.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from anonymizer.engine.common import Zone
from anonymizer.engine.ocr import rotate_points
from anonymizer.engine.patterns import norm

DETECTOR = "signatures"
DOUBT_SIGNATURE = "Posible firma: revisa el original"

# Keywords, on ``norm`` text (lowercase, no accents; "º" becomes "o", "°" stays): firma, firmado,
# firmante..., V°B° / VºBº / Vo.Bo. / V.B. (visto bueno) and p.p. (por poder).
_KEYWORD = re.compile(
    r"(?<![a-z0-9])(?:firm(?:a|as|ado|ada|ados|adas|ante|antes)|v\s?[°o]\s?\.?\s?b\s?[°o]\.?|v\.\s?b\.|p\.\s?p\.)"
    r"(?![a-z0-9])"
)
_KEYWORD_MAX_WORDS = 8  # a keyword in running text ("el acta se firma en dos ejemplares") anchors nothing
_HEADER = re.compile(r"^\s*firmas?\b")  # header of a signature column

WORK_SIDE = 2000  # larger rasters are analyzed downscaled to this long side
PRINTED_SCORE = 0.8  # OCR lines at least this confident explain their ink as printed text
_MIN_AREA = 6  # pixels: specks of noise are ignored


def is_keyword_line(text: str) -> bool:
    """A short line with a signature keyword: "Firma del titular", "V°B° Jefatura", "p.p. Director"."""
    n = norm(text or "")
    return len(n.split()) <= _KEYWORD_MAX_WORDS and bool(_KEYWORD.search(n))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """A rectangle centered on ``c`` with axes ``u`` (along) and ``v`` (across), half sizes ``a`` and ``b``."""

    c: np.ndarray
    u: np.ndarray
    a: float
    b: float

    @property
    def v(self) -> np.ndarray:
        return np.array([-self.u[1], self.u[0]])

    def st(self, points: np.ndarray) -> np.ndarray:
        """Coordinates of ``points`` (N x 2) along and across the frame."""
        d = np.asarray(points, np.float64).reshape(-1, 2) - self.c
        return np.stack([d @ self.u, d @ self.v], axis=1)

    def polygon(self, s0: float, s1: float, t0: float, t1: float) -> np.ndarray:
        return np.array([self.c + s * self.u + t * self.v for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))])


def line_frame(polygon) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Center, direction (unit), length and height of a text line's quadrilateral.

    The direction is the longer side: OCR passes in other orientations return the corners in a
    different order, so the order of the corners is not trusted.
    """
    p = np.asarray(polygon, np.float64).reshape(-1, 2)
    e1, e2 = p[1] - p[0], p[3] - p[0]
    n1, n2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    if n1 >= n2:
        u, length, height = (e1 / n1 if n1 else np.array([1.0, 0.0])), n1, n2
    else:
        u, length, height = e2 / n2, n2, n1
    return p.mean(axis=0), u, length, height


def to_turn(points: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Maps points of an image of width ``w`` and height ``h`` into ``np.rot90(img, k)`` (inverse of
    ``ocr.rotate_points``)."""
    x, y = points[:, 0], points[:, 1]
    if k == 0:
        return points
    if k == 1:
        return np.stack([y, w - x], axis=1)
    if k == 2:
        return np.stack([w - x, h - y], axis=1)
    return np.stack([h - y, x], axis=1)


def _area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """The convex hulls of two polygons share at least 20 % of the smaller one."""
    a = cv2.convexHull(np.asarray(a, np.float32)).reshape(-1, 2)
    b = cv2.convexHull(np.asarray(b, np.float32)).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    return inter > 0.2 * max(1e-6, min(_area(a.astype(np.float64)), _area(b.astype(np.float64))))


def _hull(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    points = np.concatenate([np.asarray(a, np.float64), np.asarray(b, np.float64)]).astype(np.float32)
    return cv2.convexHull(points).reshape(-1, 2).astype(np.float64)


def merge(found: list[tuple[np.ndarray, float]]) -> list[tuple[np.ndarray, float]]:
    """Joins zones that overlap (at least 20 % of the smaller one) into the convex hull of both."""
    zones = [(np.asarray(p, np.float64), s) for p, s in found]
    changed = True
    while changed:
        changed = False
        out: list[tuple[np.ndarray, float]] = []
        for pol, score in zones:
            for i, (other, other_score) in enumerate(out):
                if _overlap(pol, other):
                    out[i] = (_hull(pol, other), max(score, other_score))
                    changed = True
                    break
            else:
                out.append((pol, score))
        zones = out
    return zones


def merge_zones(zones: list[Zone]) -> list[Zone]:
    """Joins overlapping signature zones (from any detector, such as the context rule for a
    "Firma" column) into one doubtful zone with the detector of the largest and the texts of all;
    a signature zone that overlaps no other, and every other zone, is kept as it is."""
    while True:  # a joined zone can reach zones it did not touch before: repeat until stable
        out: list[Zone] = []
        for z in zones:
            if z.type != "signature":
                out.append(z)
                continue
            for i, other in enumerate(out):
                if other.type == "signature" and _overlap(z.polygon, other.polygon):
                    big = max((z, other), key=lambda q: _area(np.asarray(q.polygon, np.float64)))
                    text = " ".join(dict.fromkeys(t for t in (other.text, z.text) if t))
                    out[i] = Zone("signature", _hull(z.polygon, other.polygon), text, big.detector,
                                  max(float(z.score), float(other.score)), DOUBT_SIGNATURE)  # fmt: skip
                    break
            else:
                out.append(z)
        if len(out) == len(zones):
            return out
        zones = out


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    polygon: np.ndarray  # working pixels
    text: str
    score: float
    turn: int


class _Page:
    """The ink of one raster: masks, connected strokes and what printed text explains."""

    def __init__(self, img: np.ndarray, lines: list[_Line], faces: list[np.ndarray]):
        self.h, self.w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.bg = _background(gray)
        diff = self.bg.astype(np.int16) - gray.astype(np.int16)
        self.ink = (diff > np.maximum(40, self.bg * 0.25)).astype(np.uint8)
        confident = [ln for ln in lines if ln.score >= PRINTED_SCORE]
        heights = [line_frame(ln.polygon)[3] for ln in confident]
        self.text_h = float(np.clip(np.median(heights), 8, 150)) if heights else self._text_height_from_ink()
        length = max(25, int(3 * self.text_h))
        self.rules_h = cv2.morphologyEx(
            self.ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        )
        self.rules_v = cv2.morphologyEx(
            self.ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        )
        rules = cv2.dilate(self.rules_h | self.rules_v, np.ones((3, 3), np.uint8))
        strokes = self.ink & (rules == 0).astype(np.uint8)
        self.n, self.labels, self.stats, self.centroids = cv2.connectedComponentsWithStats(strokes, connectivity=8)
        n = self.n
        area = self.stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        self.area = area
        self.extent = np.maximum(self.stats[:, cv2.CC_STAT_WIDTH], self.stats[:, cv2.CC_STAT_HEIGHT]).astype(np.float64)
        # Printed: mostly inside a confident OCR line, and not much larger than that line's letters.
        height_map = np.zeros((self.h, self.w), np.float32)
        for ln in sorted(confident, key=lambda ln: line_frame(ln.polygon)[3]):
            cv2.fillPoly(height_map, [np.round(ln.polygon).astype(np.int32)], float(line_frame(ln.polygon)[3]))
        inside = height_map > 0
        in_labels = self.labels[inside]
        covered = np.bincount(in_labels, minlength=n).astype(np.float64)
        tallest = np.zeros(n, np.float32)
        np.maximum.at(tallest, in_labels, height_map[inside])
        self.printed = (covered >= 0.6 * area) & (self.extent <= 1.8 * tallest)
        self.printed[0] = False
        self.in_line = covered / np.maximum(area, 1)  # share of each mark inside a confident OCR line
        any_map = np.zeros((self.h, self.w), np.uint8)
        for ln in lines:
            cv2.fillPoly(any_map, [np.round(ln.polygon).astype(np.int32)], 1)
        # Share inside any line OCR read, however unsure: a stroke OCR took for text.
        self.in_any = np.bincount(self.labels[any_map > 0], minlength=n) / np.maximum(area, 1)
        self.confident = confident
        keyword_map = np.zeros((self.h, self.w), np.uint8)
        for ln in lines:
            if is_keyword_line(ln.text):
                cv2.fillPoly(keyword_map, [np.round(ln.polygon).astype(np.int32)], 1)
        # The marks of a keyword label ("Firma del titular") stay out of the zone: the label is not data.
        self.keyword = np.bincount(self.labels[keyword_map > 0], minlength=n) >= 0.6 * area
        self.keyword[0] = False
        self.face = np.zeros(n, bool)
        if faces:
            face_mask = np.zeros((self.h, self.w), np.uint8)
            cv2.fillPoly(face_mask, [np.round(f).astype(np.int32) for f in faces], 1)
            self.face = np.bincount(self.labels[face_mask > 0], minlength=n) > 0.2 * area
        x, y, w, h = (self.stats[:, k].astype(np.float64) for k in range(4))
        self.corners = np.stack([np.stack(c, axis=1) for c in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))], axis=1)
        self._shapes: dict[int, tuple[float, ...]] = {}

    def _text_height_from_ink(self) -> float:
        """Letter height when there is no OCR line: from the size of the small ink components."""
        _, _, stats, _ = cv2.connectedComponentsWithStats(self.ink, connectivity=8)
        hs = stats[1:, cv2.CC_STAT_HEIGHT]
        small = hs[(stats[1:, cv2.CC_STAT_AREA] >= 10) & (hs >= 4) & (hs <= max(8, 0.08 * self.h))]
        return float(np.clip(np.median(small) * 1.4, 10, 120)) if small.size >= 5 else 16.0

    @property
    def doclike(self) -> bool:
        """Mostly light background with sparse ink: a page, a card or a signature, not a photo."""
        return float(np.median(self.bg)) >= 170 and float(self.ink.mean()) <= 0.2

    @property
    def paper(self) -> bool:
        """A sheet of paper and nothing else (a scanned page, a signature image): light, sparse ink and
        an even background. A photo, even a bright one, or a card on a table is not."""
        median = float(np.median(self.bg))
        return self.doclike and float((np.abs(self.bg.astype(np.float32) - median) <= 25).mean()) >= 0.7

    # -- strokes ---------------------------------------------------------

    def _shape(self, i: int) -> tuple[float, ...]:
        """Shape of a mark: stroke width, curliness (drawn length over extent), thickness across its main
        axis, whether it is a hollow convex outline (a ring or a frame: a stamp), and the share of its
        outline made of long straight runs (rules, frames) and of axis-parallel runs (QR codes, grids)."""
        if i in self._shapes:
            return self._shapes[i]
        x, y, w, h, a = (int(v) for v in self.stats[i])
        sub = (self.labels[y : y + h, x : x + w] == i).astype(np.uint8)
        contours, hierarchy = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        length = max(1.0, perimeter / 2)
        outer = [c for c, hh in zip(contours, hierarchy[0], strict=False) if hh[3] < 0] or list(contours)
        big = max(outer, key=cv2.contourArea)
        outer_area = max(1.0, cv2.contourArea(big))
        hull_area = max(1.0, cv2.contourArea(cv2.convexHull(big)))
        (_, _), (rw, rh), _ = cv2.minAreaRect(big)
        hollow = outer_area / hull_area > 0.85 and a / outer_area < 0.35
        long_run = max(3 * self.text_h, 40.0)  # a table border or a stamp's frame, not a pen's swing
        total = straight = axis = 0.0
        for c in contours:
            pts = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(np.float64)
            if len(pts) < 2:
                continue
            d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
            runs = np.hypot(d[:, 0], d[:, 1])
            total += runs.sum()
            straight += runs[runs >= long_run].sum()
            axis += runs[((np.abs(d[:, 0]) <= 1) | (np.abs(d[:, 1]) <= 1)) & (runs >= 3)].sum()
        total = max(total, 1.0)
        shape = (a / length, length / max(w, h, 1), float(min(rw, rh)), hollow, straight / total, axis / total)
        self._shapes[i] = shape
        return shape

    def _structural(self, i: int) -> bool:
        """A ring or frame, a grid or a straight mark: part of a stamp, a table or a QR code."""
        x, y, w, h = (int(v) for v in self.stats[i, :4])
        if self.area[i] / max(1, w * h) > 0.45:
            return True
        width, curly, across, hollow, straight, axis = self._shape(i)
        return hollow or straight > 0.5 or axis > 0.6 or (curly < 1.2 and across <= 3 * width)

    def is_stroke(self, i: int, min_extent: float, max_extent: float, lone: bool = False) -> bool:
        """A handwriting stroke: unexplained by print, long, thin, curved, and not a straight rule, a ring
        or a frame (stamps), a grid of squares (QR codes, tables) or part of a texture (a photo).

        ``lone``: no keyword or line anchors it, so it must be curlier and lie outside every line of
        text that OCR read, however unsure (letters run together by blur, a bold or script font).
        """
        if self.printed[i] or self.face[i] or self.area[i] < _MIN_AREA:
            return False
        if not min_extent <= self.extent[i] <= max_extent:
            return False
        x, y, w, h = (int(v) for v in self.stats[i, :4])
        if x <= 1 or y <= 1 or x + w >= self.w - 1 or y + h >= self.h - 1:
            return False
        if self.area[i] / max(1, w * h) > 0.45:
            return False
        width, curly, across, hollow, straight, axis = self._shape(i)
        if hollow or width > max(0.35 * self.text_h, 3.0) or (curly < 1.2 and across <= 3 * width):
            return False
        if straight > 0.5 or axis > 0.6:
            return False
        if lone and (curly < 1.6 or self.in_any[i] >= 0.4):
            return False
        m = int(self.text_h)
        around = self.ink[max(0, y - m) : y + h + m, max(0, x - m) : x + w + m]
        return float(around.mean()) <= 0.3  # sparse ink around it: not a photo or a texture

    def _blob_is_handwriting(self, members: np.ndarray, frame: Frame) -> bool:
        """Several unexplained marks that together look like a word (a signature in separate letters)."""
        if members.size < 3:
            return False
        st = frame.st(self.corners[members].reshape(-1, 2))
        span = st[:, 0].max() - st[:, 0].min()
        return (
            span >= 2.5 * self.text_h
            and float(self.area[members].sum()) >= 0.3 * self.text_h**2
            and float(self.extent[members].mean()) >= 0.5 * self.text_h
        )

    def zones_in(
        self,
        frame: Frame,
        min_extent: float,
        max_extent: float,
        weak: bool,
        lone: bool = False,
        signed: frozenset[int] | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        """Signature zones inside ``frame``: clusters of unexplained ink that hold a stroke (or, with
        ``weak``, look like handwriting), grown over the printed marks that touch them.

        ``signed`` (a signature line): the sides of the line (-1, +1 across the frame) with a printed
        line next to it (the signer's name or role). A stroke that OCR read as text only counts on the
        side opposite such a line: a title in a script font over a decorative rule does not.
        """
        st = frame.st(self.corners.reshape(-1, 2)).reshape(-1, 4, 2)
        s0, s1 = st[:, :, 0].min(axis=1), st[:, :, 0].max(axis=1)
        t0, t1 = st[:, :, 1].min(axis=1), st[:, :, 1].max(axis=1)
        center = frame.st(self.centroids)
        inside = (np.abs(center[:, 0]) <= frame.a) & (np.abs(center[:, 1]) <= frame.b) & (self.area >= _MIN_AREA)
        inside[0] = False
        ids = np.flatnonzero(inside & ~self.printed & ~self.face)
        if not ids.size:
            return []
        seeds = {int(i) for i in ids if self.is_stroke(int(i), min_extent, max_extent, lone)}
        if signed is not None:
            across = frame.st(self.centroids)[:, 1]
            seeds = {i for i in seeds if self.in_any[i] < 0.4 or (-1 if across[i] > 0 else 1) in signed}
        if not seeds and not weak:
            return []
        # Large marks that are not strokes (stamp rings and frames, grids) do not join a signature, and
        # neither do the marks inside a ring or frame (the text of a stamp).
        large = [int(i) for i in ids if i not in seeds and self.extent[i] >= min_extent and self._structural(int(i))]
        frames = [self.stats[i, :4] for i in large if self._shape(i)[3]]
        ids = np.array(
            [i for i in ids if i not in large and (i in seeds or not any(
                fx < self.centroids[i][0] < fx + fw and fy < self.centroids[i][1] < fy + fh for fx, fy, fw, fh in frames
            ))],
            np.int64,
        )  # fmt: skip
        if not ids.size:
            return []
        # Clusters of the unexplained ink: marks closer than half a letter belong together.
        box = frame.polygon(-frame.a, frame.a, -frame.b, frame.b)
        x0, y0 = np.maximum(0, np.floor(box.min(axis=0)).astype(int))
        x1, y1 = np.minimum([self.w, self.h], np.ceil(box.max(axis=0)).astype(int) + 1)
        sub = self.labels[y0:y1, x0:x1]
        mask = np.isin(sub, ids).astype(np.uint8)
        gap = max(2, int(round(0.5 * self.text_h)))
        joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1)))
        _, blobs = cv2.connectedComponents(joined, connectivity=8)
        blob_of = np.zeros(self.n, np.int32)
        on = mask > 0
        blob_of[sub[on]] = blobs[on]
        others = np.flatnonzero(inside & ~self.keyword)
        out = []
        for b in np.unique(blob_of[ids]):
            members = ids[blob_of[ids] == b]
            seeded = bool(seeds.intersection(members.tolist()))
            if not seeded and not (weak and self._blob_is_handwriting(members, frame)):
                continue
            r = [s0[members].min(), s1[members].max(), t0[members].min(), t1[members].max()]
            # The marks that touch it: first any mark within one and a half letters (the letters of a
            # name it crosses), then the loose marks next to those (their accents and dots). Printed
            # lines further away (the signer's name and role under the line) are left visible.
            reach, bound = 0.6 * self.text_h, 1.5 * self.text_h
            first = others[
                (s0[others] >= r[0] - bound) & (s1[others] <= r[1] + bound)
                & (t0[others] >= r[2] - bound) & (t1[others] <= r[3] + bound)
            ]  # fmt: skip
            for pool in (first, ids):
                near = pool[
                    (s0[pool] <= r[1] + reach) & (s1[pool] >= r[0] - reach)
                    & (t0[pool] <= r[3] + reach) & (t1[pool] >= r[2] - reach)
                ]  # fmt: skip
                r = [min(r[0], s0[near].min(initial=r[0])), max(r[1], s1[near].max(initial=r[1])),
                     min(r[2], t0[near].min(initial=r[2])), max(r[3], t1[near].max(initial=r[3]))]  # fmt: skip
            m = max(2.0, 0.15 * self.text_h)
            r = [max(-frame.a, r[0] - m), min(frame.a, r[1] + m), max(-frame.b, r[2] - m), min(frame.b, r[3] + m)]
            out.append((frame.polygon(*r), 0.6 if seeded else 0.4))
        return out

    # -- anchors ---------------------------------------------------------

    def rule_frames(self) -> list[tuple[Frame, frozenset[int]]]:
        """Signature lines: straight rules a few words long that are not part of a table grid, on a
        page or a card (in a photo, the straight edges of things are not lines), each with the sides
        that have a printed line next to it (``zones_in``)."""
        frames: list[tuple[Frame, frozenset[int]]] = []
        if not self.doclike:
            return frames
        for rules, across, horizontal in ((self.rules_h, self.rules_v, True), (self.rules_v, self.rules_h, False)):
            n, _, stats, _ = cv2.connectedComponentsWithStats(rules, connectivity=8)
            side = self.w if horizontal else self.h
            for x, y, w, h, _ in stats[1:]:
                length, thick = (w, h) if horizontal else (h, w)
                if not (max(5 * self.text_h, 0.06 * side) <= length <= 0.7 * side) or thick > max(5, 0.3 * self.text_h):
                    continue
                if across[max(0, y - 4) : y + h + 4, max(0, x - 4) : x + w + 4].any():
                    continue  # crossed by a rule the other way: a table
                u = np.array([1.0, 0.0]) if horizontal else np.array([0.0, 1.0])
                frame = Frame(np.array([x + w / 2, y + h / 2]), u, 0.6 * length, min(5 * self.text_h, 0.6 * length))
                frames.append((frame, self._signed_sides(frame, length / 2)))
        return frames

    def _signed_sides(self, frame: Frame, half: float) -> frozenset[int]:
        """Sides of a rule with a confident printed line within two and a half letters of it."""
        sides = set()
        for ln in self.confident:
            st = frame.st(ln.polygon)
            if st[:, 0].max() < -half or st[:, 0].min() > half:
                continue
            t0, t1 = st[:, 1].min(), st[:, 1].max()
            if 0 < t0 <= 2.5 * self.text_h:
                sides.add(1)
            elif -2.5 * self.text_h <= t1 < 0:
                sides.add(-1)
        return frozenset(sides)

    def keyword_frames(self, lines: list[_Line]) -> list[Frame]:
        frames: list[Frame] = []
        for ln in lines:
            if not is_keyword_line(ln.text):
                continue
            c, u, length, height = line_frame(ln.polygon)
            if length < 2:
                continue
            big = max(height, self.text_h)
            frame = Frame(c, u, length / 2 + max(length, 10 * big), height / 2 + 6 * big)
            if not any(np.hypot(*(f.c - c)) < height and abs(float(f.u @ u)) > 0.9 for f in frames):
                frames.append(frame)  # the same label read in several orientations counts once
        return frames


def _background(gray: np.ndarray) -> np.ndarray:
    """Paper brightness around every pixel: a maximum filter (wider than any pen stroke), smoothed."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.blur(cv2.dilate(small, np.ones((7, 7), np.uint8)), (7, 7))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _column_zones(lines: list[_Line], w: int, h: int) -> list[tuple[np.ndarray, float]]:
    """The column under a "Firma" header of a table, down to the table's last row, in every orientation."""
    out = []
    for k in (0, 1, 3):
        size = (w, h) if k % 2 == 0 else (h, w)
        boxes = []
        for ln in lines:
            if ln.turn != k:
                continue
            p = to_turn(ln.polygon, k, w, h)
            (x0, y0), (x1, y1) = p.min(axis=0), p.max(axis=0)
            if x1 - x0 >= y1 - y0:  # read across this orientation, not along it
                boxes.append((float(x0), float(y0), float(x1), float(y1), norm(ln.text)))
        for i, (hx0, hy0, hx1, hy1, text) in enumerate(boxes):
            if not _HEADER.match(text) or len(text.split()) > 4:
                continue
            hh, hw = hy1 - hy0, hx1 - hx0
            row = [
                b for j, b in enumerate(boxes)
                if j != i and min(b[3], hy1) - max(b[1], hy0) > 0.4 * min(hh, b[3] - b[1])
                and len(b[4].split()) <= 4 and b[2] - b[0] < 0.5 * size[0]
            ]  # fmt: skip
            if sum(1 for b in row if not _HEADER.match(b[4])) < 2:
                continue  # "Firma entrega" next to "Firma recibe" is not a table
            left = [b[2] for b in row if b[2] < hx0]
            right = [b[0] for b in row if b[0] > hx1]
            xs = (max(left) + hx0) / 2 if left else hx0 - hh
            xe = (min(right) + hx1) / 2 if right else min(size[0], hx1 + max(1.5 * hw, 8 * hh))
            # Rows: groups of lines under the header with cells in at least two of its columns. The
            # same table read upside down (OCR flips it) has the header under the rows: no row then.
            heads = sorted([*row, boxes[i]], key=lambda b: b[0])
            mids = [(a[2] + b[0]) / 2 for a, b in zip(heads, heads[1:], strict=False)]
            table_x0, table_x1 = heads[0][0] - hh, max(xe, max(b[2] for b in heads)) + hh
            below = sorted((b for b in boxes if b[1] > hy1 - 1 and table_x0 <= (b[0] + b[2]) / 2 <= table_x1),
                           key=lambda b: b[1])  # fmt: skip
            groups: list[list[tuple]] = []
            for b in below:
                if groups and b[1] - groups[-1][0][1] <= 0.6 * hh:
                    groups[-1].append(b)
                else:
                    groups.append([b])
            last, bottom, tops = hy1, hy1, []
            for group in groups:
                if group[0][1] - last > max(4 * hh, 0.08 * size[1]):
                    break
                if len({int(np.searchsorted(mids, (b[0] + b[2]) / 2)) for b in group}) < 2:
                    break
                tops.append(group[0][1])
                last = bottom = max(bottom, max(b[3] for b in group))
            if not tops:
                continue  # no rows under the header
            pitch = float(np.median(np.diff(tops))) if len(tops) >= 2 else 3 * hh
            bottom = min(size[1], bottom + 0.6 * pitch)
            corners = rotate_points(np.array([[xs, hy1], [xe, hy1], [xe, bottom], [xs, bottom]]), k, w, h)
            out.append((corners, 0.5))
    return out


def _column_frame(corners: np.ndarray, reach: float) -> Frame:
    """The frame of a column zone, wider by ``reach``: signatures overflow their cell."""
    along, across = corners[1] - corners[0], corners[3] - corners[0]
    length = float(np.hypot(*along))
    u = along / length if length else np.array([1.0, 0.0])
    return Frame(corners.mean(axis=0), u, length / 2 + reach, float(np.hypot(*across)) / 2)


def detect_raster(
    bgr: np.ndarray, lines: Sequence[Any] = (), *, faces: Sequence[Any] = (), whole: bool = False, lone: bool = True
) -> list[Zone]:
    """Signature zones of a BGR image, in its pixels.

    ``lines``: OCR lines (``ocr.OcrLine``: polygon, text, score, turn), plus any text-layer line of a
    PDF mapped to these pixels (score 1, turn 0). ``faces``: polygons of the faces found, left out.
    ``whole``: the image itself may be a signature (a small image placed on a text page).
    ``lone=False``: a stroke needs a keyword, a line or a column (an image repeated on every page).
    """
    h0, w0 = bgr.shape[:2]
    if min(h0, w0) < 16:
        return []
    f = min(1.0, WORK_SIDE / max(h0, w0))
    img = cv2.resize(bgr, (round(w0 * f), round(h0 * f)), interpolation=cv2.INTER_AREA) if f < 1 else bgr
    work = [
        _Line(np.asarray(ln.polygon, np.float64).reshape(-1, 2) * f, str(ln.text), float(ln.score), int(getattr(ln, "turn", 0)))
        for ln in lines
        if np.asarray(ln.polygon).size >= 8
    ]  # fmt: skip
    page = _Page(img, work, [np.asarray(p, np.float64).reshape(-1, 2) * f for p in faces])
    found = _column_zones(work, page.w, page.h)
    anchored = max(20.0, 3 * page.text_h)
    largest = 0.5 * max(page.w, page.h)  # a mark half the page long is a frame or a chart, not a signature
    columns = [_column_frame(corners, 4 * page.text_h) for corners, _ in found]
    # Keyword labels and table columns allow a signature in separate letters; a bare line needs a stroke.
    for frame in page.keyword_frames(work) + columns:
        found += page.zones_in(frame, anchored, largest, weak=True)
    for frame, signed in page.rule_frames():
        found += page.zones_in(frame, anchored, largest, weak=False, signed=signed)
    if whole and page.paper:
        frame = Frame(np.array([page.w / 2, page.h / 2]), np.array([1.0, 0.0]), page.w / 2, page.h / 2)
        found += page.zones_in(frame, max(12.0, 0.25 * min(page.w, page.h)), float("inf"), weak=True)
    if lone and page.paper:  # a large curly stroke on its own, on a sheet of paper
        lone = max(40.0, 4 * page.text_h)
        taken = merge(found)
        for i in np.flatnonzero(page.extent >= lone):
            if not page.is_stroke(int(i), lone, largest, lone=True):
                continue
            if any(
                all(
                    cv2.pointPolygonTest(p.astype(np.float32), (float(x), float(y)), False) >= 0
                    for x, y in page.corners[i]
                )
                for p, _ in taken
            ):
                continue  # already inside a zone
            x, y, w, h = page.stats[i, :4]
            frame = Frame(np.array([x + w / 2, y + h / 2]), np.array([1.0, 0.0]), w / 2 + 4 * page.text_h,
                          h / 2 + 4 * page.text_h)  # fmt: skip
            found += page.zones_in(frame, lone, largest, weak=False, lone=True)
    return [Zone("signature", pol / f, "", DETECTOR, score, DOUBT_SIGNATURE) for pol, score in merge(found)]


# ---------------------------------------------------------------------------
# Vector paths of a PDF text page
# ---------------------------------------------------------------------------


def path_shape(items) -> tuple[int, int, float, float, int]:
    """Curves, line segments, drawn length, the share of axis-parallel segments and how many times
    the pen turns back left or right (handwriting does it at every letter; a chart never does)."""
    curves = lines = 0
    length = 0.0
    straight = 0
    xs: list[float] = []
    for item in items:
        kind = item[0]
        if kind == "l":
            p, q = item[1], item[2]
            dx, dy = q.x - p.x, q.y - p.y
            lines += 1
            length += float(np.hypot(dx, dy))
            straight += abs(dx) < 0.5 or abs(dy) < 0.5
            xs += [p.x, q.x]
        elif kind == "c":
            p0, p1, p2, p3 = item[1:5]
            chord = np.hypot(p3.x - p0.x, p3.y - p0.y)
            hull = sum(np.hypot(b.x - a.x, b.y - a.y) for a, b in ((p0, p1), (p1, p2), (p2, p3)))
            curves += 1
            length += float(chord + hull) / 2
            xs += [p0.x, p1.x, p2.x, p3.x]
    steps = np.diff(np.asarray(xs, np.float64)) if len(xs) > 1 else np.zeros(0)
    signs = np.sign(steps[np.abs(steps) > 0.3])
    turns = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    return curves, lines, length, straight / max(1, lines), turns


def _closed_convex(items) -> bool:
    """Every subpath is a closed outline that bulges out everywhere: a ring, an oval or a rounded box
    (seals, stamps, radio buttons), never a pen's stroke."""
    subpaths: list[list[tuple[float, float]]] = []
    for item in items:
        if item[0] == "l":
            pts = [item[1], item[2]]
        elif item[0] == "c":
            p0, p1, p2, p3 = item[1:5]
            pts = [
                ((1 - t) ** 3 * p0.x + 3 * (1 - t) ** 2 * t * p1.x + 3 * (1 - t) * t**2 * p2.x + t**3 * p3.x,
                 (1 - t) ** 3 * p0.y + 3 * (1 - t) ** 2 * t * p1.y + 3 * (1 - t) * t**2 * p2.y + t**3 * p3.y)
                for t in np.linspace(0, 1, 9)
            ]  # fmt: skip
            pts = [type(p0)(x, y) for x, y in pts]
        else:
            return False
        if not subpaths or np.hypot(pts[0].x - subpaths[-1][-1][0], pts[0].y - subpaths[-1][-1][1]) > 0.5:
            subpaths.append([])
        subpaths[-1] += [(p.x, p.y) for p in pts]
    for sub in subpaths:
        poly = np.asarray(sub, np.float64)
        size = float(np.ptp(poly, axis=0).max()) if len(poly) else 0.0
        if len(poly) < 6 or size <= 0 or np.hypot(*(poly[0] - poly[-1])) > max(0.5, 0.05 * size):
            return False
        hull = cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        if _area(hull) <= 0 or _area(poly) / _area(hull) < 0.9:
            return False
    return bool(subpaths)


def _reads_vertically(lines: Sequence[Any]) -> bool:
    """Most of the text runs top to bottom in this space (a page turned by /Rotate, a sideways table)."""
    across = along = 0
    for ln in lines:
        n = len(ln.text.strip())
        if n < 3:
            continue
        if ln.y1 - ln.y0 > ln.x1 - ln.x0 and not getattr(ln, "horizontal", False):
            along += n
        else:
            across += n
    return along > across


def _transposed(d: dict) -> dict:
    """A path of ``get_drawings`` with x and y swapped."""
    import pymupdf

    def swap(p):
        return pymupdf.Point(p.y, p.x)

    items = []
    for item in d.get("items") or []:
        if item[0] in ("l", "c"):
            items.append((item[0], *(swap(p) for p in item[1:])))
        elif item[0] == "re":
            r = item[1]
            items.append(("re", pymupdf.Rect(r.y0, r.x0, r.y1, r.x1), *item[2:]))
        else:
            items.append(item)
    r = d["rect"]
    return {**d, "rect": pymupdf.Rect(r.y0, r.x0, r.y1, r.x1), "items": items}


def _text_columns(lines: Sequence[Any], width: float, height: float) -> list[tuple[float, ...]]:
    """The column under a "Firma" header of a table of the text layer (``_column_zones``)."""
    work = [
        _Line(np.array([[ln.x0, ln.y0], [ln.x1, ln.y0], [ln.x1, ln.y1], [ln.x0, ln.y1]], np.float64), ln.text, 1.0, 0)
        for ln in lines
    ]
    out = []
    for corners, score in _column_zones(work, round(width), round(height)):
        (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
        out.append((float(x0), float(y0), float(x1), float(y1), score))
    return out


def detect_vector(
    drawings: list[dict], lines: Sequence[Any], width: float, height: float, *, columns: bool = False
) -> list[tuple[float, ...]]:
    """Signature zones ``(x0, y0, x1, y1, score)`` among the vector paths of a text page.

    ``drawings``: ``page.get_drawings()``; ``lines``: the text layer's lines (``context.Line``) in the
    same space; ``width``/``height``: the page size in that space. ``columns``: also the column under
    a "Firma" header of a table (the context rule for names finds it when that group is on). When
    most of the text reads vertically in this space, the page is analyzed transposed, where it reads
    across, and the zones are turned back.
    """
    if _reads_vertically(lines):
        from anonymizer.engine.context import Line

        turned = [Line(ln.text, ln.y0, ln.x0, ln.y1, ln.x1, True) for ln in lines]
        found = _detect_vector([_transposed(d) for d in drawings if d.get("rect") is not None], turned, height,
                               width, columns)  # fmt: skip
        return [(y0, x0, y1, x1, score) for x0, y0, x1, y1, score in found]
    return _detect_vector(drawings, lines, width, height, columns)


def _detect_vector(
    drawings: list[dict], lines: Sequence[Any], width: float, height: float, columns: bool
) -> list[tuple[float, ...]]:
    out: list[tuple[float, ...]] = _text_columns(lines, width, height) if columns else []
    heights = [ln.y1 - ln.y0 for ln in lines if getattr(ln, "horizontal", True) and ln.y1 > ln.y0]
    text_h = float(np.clip(np.median(heights), 4, 40)) if heights else 10.0
    paths = []
    rules = []
    for d in drawings:
        r = d.get("rect")
        items = d.get("items") or []
        if r is None or not items:
            continue
        curves, segments, length, straight, turns = path_shape(items)
        if not curves and segments == 1 and r.height <= 2 and 5 * text_h <= r.width <= 0.7 * width:
            rules.append((r.x0, r.y0, r.x1, r.y1))  # a signature line
        boxes = sum(1 for item in items if item[0] in ("re", "qu"))
        if boxes and not curves:
            if boxes == 1 and r.height <= 2 and 5 * text_h <= r.width <= 0.7 * width:
                rules.append((r.x0, r.y0, r.x1, r.y1))
            continue
        if r.width > 0.6 * width and r.height > 0.3 * height:
            continue
        if not (curves >= 2 or (segments >= 6 and straight < 0.6)) or max(r.width, r.height) < 0.3 * text_h:
            continue
        if _closed_convex(items):
            continue
        stroked = "s" in (d.get("type") or "") and d.get("color") is not None
        paths.append((r.x0, r.y0, r.x1, r.y1, curves, segments, length, stroked, float(d.get("width") or 0), turns))
    if not paths:
        return out
    regions = []
    for ln in lines:
        if is_keyword_line(ln.text):
            big = max(ln.y1 - ln.y0, text_h)
            length = ln.x1 - ln.x0
            regions.append(
                (ln.x0 - max(length, 10 * big), ln.y0 - 6 * big, ln.x1 + max(length, 10 * big), ln.y1 + 6 * big)
            )
        elif len(ln.text.strip()) >= 10 and set(ln.text.strip()) <= set("_.-… "):
            rules.append((ln.x0, ln.y0, ln.x1, ln.y1))  # "______________" typed as a signature line
    for x0, y0, x1, y1 in rules:
        reach = min(5 * text_h, 0.6 * (x1 - x0))
        regions.append((x0 - 0.1 * (x1 - x0), y0 - reach, x1 + 0.1 * (x1 - x0), y1 + reach))
    # Clusters: paths closer than most of a letter's height belong together.
    box = np.array([p[:4] for p in paths], np.float64)
    gap = 0.8 * text_h
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if len(paths) <= 4000:
        near = (
            (box[:, None, 0] <= box[None, :, 2] + gap) & (box[None, :, 0] <= box[:, None, 2] + gap)
            & (box[:, None, 1] <= box[None, :, 3] + gap) & (box[None, :, 1] <= box[:, None, 3] + gap)
        )  # fmt: skip
        for i, j in zip(*np.nonzero(np.triu(near, 1)), strict=False):
            parent[find(int(i))] = find(int(j))
    clusters: dict[int, list[int]] = {}
    for i in range(len(paths)):
        clusters.setdefault(find(i), []).append(i)
    for members in clusters.values():
        b = box[members]
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        extent = max(x1 - x0, y1 - y0)
        if extent < 2 * text_h or y1 - y0 > 10 * text_h or x1 - x0 > 0.8 * width:
            continue
        curves = sum(paths[i][4] for i in members)
        segments = sum(paths[i][5] for i in members)
        length = sum(paths[i][6] for i in members)
        stroked = sum(paths[i][6] for i in members if paths[i][7])
        turns = sum(paths[i][9] for i in members)
        curly = length / max(extent, 1e-6)
        anchored = any(x0 <= rx1 and rx0 <= x1 and y0 <= ry1 and ry0 <= y1 for rx0, ry0, rx1, ry1 in regions)
        if stroked >= 0.5 * length:
            ok = (anchored and (curves >= 4 or segments >= 12) and curly >= 1.5 and turns >= 2) or (
                curves + segments / 3 >= 12 and curly >= 1.8 and turns >= 6
            )
        else:  # filled outlines (a signature traced into shapes): only next to a keyword or a line
            biggest = max(max(paths[i][2] - paths[i][0], paths[i][3] - paths[i][1]) for i in members)
            ok = anchored and biggest >= 2 * text_h and curves >= 8 and len(members) <= 15
        if ok:
            m = max(paths[i][8] for i in members) / 2 + 1.0
            out.append((float(x0 - m), float(y0 - m), float(x1 + m), float(y1 + m), 0.6 if anchored else 0.5))
    return out
