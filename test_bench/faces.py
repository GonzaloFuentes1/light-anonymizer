"""Faces for the test dataset, with their ground-truth box and their core (eyes, nose and mouth).

Sources with a documented free license (see ``SOURCES`` and LICENSES.md). The images are
downloaded once to ``test_data/cache/faces`` and verified by SHA-256. If there is neither a
connection nor a cache, drawn faces are used (``synthetic_drawn``): they serve to test the
generation flow, but not to measure recall, and the manifest records it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

from test_bench.canvas import rect
from test_bench.schema import Element, Polygon

POSES = ("front", "three_quarter", "profile")


@dataclass
class Face:
    id: str
    img: Image.Image  # RGB
    box: Polygon  # whole face: forehead to chin, ear to ear (coordinates of ``img``)
    core: Polygon  # eyes, nose and mouth: must always end up covered
    pose: str
    source: str  # key in SOURCES
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    """Real photo with several people and their annotated face boxes."""

    id: str
    img: Image.Image
    faces: list[tuple[Polygon, Polygon | None]]  # (box, core or None)
    source: str
    tags: dict[str, Any] = field(default_factory=dict)


# Source registry: id -> license and attribution data. Completed from the source research
# (Face Research Lab London Set, Open Images, etc.).
SOURCES: dict[str, dict[str, Any]] = {
    "synthetic_drawn": {
        "description": "Rostros esquemáticos dibujados por el propio generador (sin personas reales).",
        "license": "Generado por el proyecto (mismo licenciamiento que el repositorio).",
        "attribution": "",
        "valid_for_recall": False,
    },
}


class FaceProvider:
    def __init__(self, cache_dir: Path, seed: int, allow_download: bool = True) -> None:
        self.cache_dir = cache_dir
        self.rng = np.random.default_rng([seed, 7])
        self.allow_download = allow_download
        self._faces: dict[str, list[Face]] = {p: [] for p in POSES}
        self._scenes: list[Scene] = []
        self._indices: dict[str, int] = {p: 0 for p in POSES}
        self._used: set[str] = set()
        self.synthetic_only = True
        self._load()

    # -- loading ---------------------------------------------------------------

    def _load(self) -> None:
        real = self._load_real()
        if real:
            self.synthetic_only = False
            return
        for i in range(12):
            for pose in POSES:
                self._faces[pose].append(_drawn_face(f"dib{i:02d}_{pose}", pose, self.rng))

    def _load_real(self) -> bool:
        """Downloads/verifies and loads the real sources. Returns False if none is available.

        Standalone faces (``take``) come only from the Face Research Lab London Set, whose
        subjects signed consent. Photos of identifiable real people (Open Images, public-domain
        official portraits) are only used as scenes as they are, never pasted into fictitious
        documents.
        """
        if not _CATALOG.exists():
            return False
        for e in json.loads(_CATALOG.read_text(encoding="utf-8")):
            if e.get("optional"):
                continue
            path = self._ensure(e)
            if path is None:
                continue
            SOURCES[e["id"]] = {
                "description": e.get("note") or e.get("subject") or e["kind"],
                "license": e["license"],
                "attribution": e["attribution"],
                "url": _url(e),
                "sha256": e["sha256"],
                "valid_for_recall": True,
            }
            img = Image.open(path).convert("RGB")
            kind = e["kind"]
            if kind.startswith("frl_"):
                if kind == "frl_front":
                    box, core = _gt_from_tem(path.with_suffix(".tem"))
                else:
                    box, core = rect(*e["faces"][0]), rect(*e["core"][0])
                pose = "front" if kind == "frl_front" else ("profile" if "profile" in kind else "three_quarter")
                f = Face(e["id"], img, box, core, pose, e["id"], {"subject": e.get("subject"), "gt": e.get("gt")})
                self._faces[pose].append(_crop_portrait(f))
            else:
                self._scenes.append(
                    Scene(
                        id=e["id"],
                        img=img,
                        faces=[(rect(*b), None) for b in e["faces"]],
                        source=e["id"],
                        tags={"source_kind": kind, "flags": e.get("flags"), "gt": e.get("gt")},
                    )
                )
        return bool(self._faces["front"])

    def _ensure(self, e: dict[str, Any]) -> Path | None:
        """Verified local path of the image (and its .tem); downloads it if missing and allowed."""
        ext = Path(e.get("zip_entry") or urlparse(_url(e)).path).suffix or ".jpg"
        dest = self.cache_dir / f"{e['id']}{ext}"
        tem = dest.with_suffix(".tem") if e.get("tem_entry") else None
        if _verified(dest, e["sha256"]) and (tem is None or _verified(tem, e["tem_sha256"])):
            return dest
        if not self.allow_download:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            if e.get("zip_entry"):
                local_zip = self.cache_dir / "zips" / f"{Path(urlparse(e['url']).path).name}.zip"
                if not local_zip.exists() or hashlib.md5(local_zip.read_bytes()).hexdigest() != e.get("zip_md5"):
                    _download(e["url"], local_zip)
                with zipfile.ZipFile(local_zip) as z:
                    dest.write_bytes(z.read(e["zip_entry"]))
                    if tem is not None:
                        tem.write_bytes(z.read(e["tem_entry"]))
            else:
                _download(_url(e), dest)
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            print(f"  aviso: no se pudo obtener {e['id']}: {error}")
            return None
        if not _verified(dest, e["sha256"]):
            print(f"  aviso: {e['id']} no coincide con su SHA-256; se descarta")
            dest.unlink(missing_ok=True)
            return None
        return dest

    # -- use -------------------------------------------------------------------

    def take(self, pose: str = "front", n: int = 1) -> list[Face]:
        """Hands out ``n`` faces of the requested pose, cycling deterministically through the catalog."""
        faces = self._faces[pose] or self._faces["front"]
        out = []
        for _ in range(n):
            f = faces[self._indices[pose] % len(faces)]
            self._indices[pose] += 1
            self._used.add(f.source)
            out.append(copy.copy(f))
        return out

    def scenes(self) -> list[Scene]:
        for s in self._scenes:
            self._used.add(s.source)
        return list(self._scenes)

    @staticmethod
    def elements(f: Face, level: str = "base", tags: dict[str, Any] | None = None) -> list[Element]:
        """Face element in ``f.img`` coordinates (ready for ``Canvas.paste``)."""
        return [
            Element(
                type="face",
                page=0,
                polygon=copy.deepcopy(f.box),
                core=copy.deepcopy(f.core),
                value=None,
                level=level,
                layer="raster",
                tags={"pose": f.pose, "face_source": f.source, "face_id": f.id, **(tags or {})},
            )
        ]

    def used_sources(self) -> list[dict[str, Any]]:
        return [{"id": s, **SOURCES[s]} for s in sorted(self._used)]


_CATALOG = Path(__file__).with_name("face_sources.json")
_USER_AGENT = "light-anonymizer-test-bench/0.1 (CENIA; test data download)"

# Point groups of the 189-point .tem templates (Psychomorph/WebMorph).
_CONTOUR = [*range(109, 115), *range(125, 134)]
_BROWS = range(71, 87)
_CHIN = range(125, 134)
_CORE = [*range(18, 50), *range(50, 71), *range(87, 109)]  # eyes and eyelids, nose, mouth


def _url(e: dict[str, Any]) -> str:
    return str(e["url"]).split()[0]


def _verified(path: Path, sha256: str) -> bool:
    return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as f:
        shutil.copyfileobj(response, f)
    temporary.replace(dest)


def _gt_from_tem(path: Path) -> tuple[Polygon, Polygon]:
    """Face box (brows to chin, contour to contour) and core (eyes, nose, mouth)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    n = int(lines[0])
    p = np.array([[float(v) for v in line.split()[:2]] for line in lines[1 : n + 1]])
    x0, x1 = p[_CONTOUR, 0].min(), p[_CONTOUR, 0].max()
    y0, y1 = p[list(_BROWS), 1].min(), p[list(_CHIN), 1].max()
    nx0, ny0 = p[_CORE].min(axis=0)
    nx1, ny1 = p[_CORE].max(axis=0)
    margin = 0.04 * (x1 - x0)
    return rect(x0, y0, x1, y1), rect(nx0 - margin, ny0 - margin, nx1 + margin, ny1 + margin)


def _crop_portrait(f: Face, final_width: int = 480) -> Face:
    """Head-and-shoulders crop (ID-photo style) with the ground truth translated."""
    xs, ys = [q[0] for q in f.box], [q[1] for q in f.box]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    w, h = bx1 - bx0, by1 - by0
    cx0 = int(max(0, bx0 - 0.45 * w))
    cy0 = int(max(0, by0 - 0.7 * h))
    cx1 = int(min(f.img.width, bx1 + 0.45 * w))
    cy1 = int(min(f.img.height, by1 + 0.45 * h))
    img = f.img.crop((cx0, cy0, cx1, cy1))
    scale = min(1.0, final_width / img.width)
    if scale < 1.0:
        img = img.resize((final_width, max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
    sx, sy = img.width / (cx1 - cx0), img.height / (cy1 - cy0)

    def move(pol: Polygon) -> Polygon:
        return [[round((x - cx0) * sx, 3), round((y - cy0) * sy, 3)] for x, y in pol]

    return Face(f.id, img, move(f.box), move(f.core), f.pose, f.source, dict(f.tags))


def _drawn_face(id: str, pose: str, rng: np.random.Generator) -> Face:
    """Schematic face (oval, eyes, nose, mouth, hair). Only for testing the flow."""
    w, h = 300, 380
    background = tuple(int(c) for c in rng.integers(150, 230, 3))
    img = Image.new("RGB", (w, h), background)
    d = ImageDraw.Draw(img)
    skin = tuple(int(c) for c in rng.choice([[241, 194, 167], [198, 134, 99], [141, 85, 54], [224, 172, 105]]))
    hair = tuple(int(c) for c in rng.choice([[30, 20, 15], [90, 60, 30], [160, 120, 60], [120, 120, 120]]))
    shift = {"front": 0, "three_quarter": 28, "profile": 55}[pose]
    x0, y0, x1, y1 = 60, 60, 240, 320
    d.ellipse((x0 - 10, y0 - 25, x1 + 10, y0 + 120), fill=hair)
    d.ellipse((x0, y0, x1, y1), fill=skin)
    cx = (x0 + x1) / 2 + shift
    for dx in (-40, 40):
        if pose == "profile" and dx < 0:
            continue
        d.ellipse((cx + dx - 14, 160, cx + dx + 14, 176), fill=(255, 255, 255))
        d.ellipse((cx + dx - 6, 162, cx + dx + 6, 174), fill=(40, 30, 20))
    d.polygon([(cx, 180), (cx - 12, 230), (cx + 10, 232)], fill=tuple(max(0, c - 30) for c in skin))
    d.arc((cx - 35, 240, cx + 35, 280), 10, 170, fill=(150, 50, 50), width=5)
    return Face(
        id=id,
        img=img,
        box=rect(x0, y0 - 20, x1, y1),
        core=rect(cx - 60, 150, cx + 60, 285),
        pose=pose,
        source="synthetic_drawn",
    )
