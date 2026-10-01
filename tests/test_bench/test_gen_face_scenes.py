"""Invariants of the face generator: files, ground-truth geometry and levels."""

import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from test_bench.canvas import Canvas, bounding_box, rect
from test_bench.context import Context
from test_bench.faces import FaceProvider, Scene
from test_bench.fake_data import FakeData
from test_bench.generators import face_scenes
from test_bench.schema import Manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_IDS = (
    [f"rost_retrato_frente_{i:02d}" for i in range(1, 5)]
    + ["rost_retrato_tres_cuartos_01", "rost_retrato_tres_cuartos_02", "rost_perfil_01", "rost_perfil_02"]
    + [f"rost_retrato_rotado_{g}" for g in (90, 180, 270, 15, 45)]
    + [f"rost_grupo_compuesto_{i:02d}" for i in (1, 2, 3)]
    + ["rost_afiche_impreso", "rost_blanco_y_negro", "rost_baja_resolucion", "rost_oclusion", "rost_espejado"]
)


def _context(root: Path) -> Context:
    return Context(
        root=root,
        seed=33,
        fake=FakeData(33),
        faces=FaceProvider(REPO_ROOT / "test_data" / "cache" / "faces", 33, allow_download=False),
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    ctx = _context(tmp_path_factory.mktemp("faces") / "generated")
    start = time.perf_counter()
    files = face_scenes.generate(ctx)
    return ctx, files, time.perf_counter() - start


def _open(ctx: Context, file_entry) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(ctx.root / file_entry.path))


def _mask(polygon, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(polygon, np.float64) - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def _by_id(files):
    return {a.id: a for a in files}


def test_ids_paths_and_manifest(generated):
    ctx, files, _ = generated
    ids = [a.id for a in files]
    assert len(ids) == len(set(ids))
    assert set(EXPECTED_IDS) <= set(ids)
    # without real scenes (only drawn faces) there must be no scene files
    if not ctx.faces.scenes():
        assert not any(i.startswith("rost_escena_real_") for i in ids)
    man = Manifest(root=str(ctx.root), seed=33)
    for a in files:
        assert a.id.startswith("rost_")
        assert a.category == "faces"
        assert a.format == "jpg" and a.path.startswith("faces/") and a.path.endswith(".jpg")
        man.add(a)
    assert man.validate() == []


def test_files_open_with_declared_size(generated):
    ctx, files, _ = generated
    for a in files:
        img = _open(ctx, a)
        assert img.size == (a.pages[0].width, a.pages[0].height), a.id
        assert not img.info.get("exif"), f"{a.id}: must not carry EXIF"
    bw = _by_id(files)["rost_blanco_y_negro"]
    assert Image.open(ctx.root / bw.path).mode == "L"


def test_single_portraits_between_400_and_800(generated):
    ctx, files, _ = generated
    for a in files:
        if a.id.startswith(("rost_retrato_", "rost_perfil_", "rost_blanco", "rost_oclusion", "rost_espejado")):
            assert 400 <= max(a.pages[0].width, a.pages[0].height) <= 800, a.id


def test_polygons_inside_image(generated):
    _, files, _ = generated
    for a in files:
        w, h = a.pages[0].width, a.pages[0].height
        for e in a.elements:
            assert e.layer == "raster" and e.page == 0
            for poly in (e.polygon, e.core):
                if poly is None:
                    continue
                x0, y0, x1, y1 = bounding_box(poly)
                assert x0 >= -0.5 and y0 >= -0.5 and x1 <= w + 0.5 and y1 <= h + 0.5, (a.id, e.type, poly)


def test_faces_well_formed_and_core_with_content(generated):
    ctx, files, _ = generated
    for a in files:
        arr = np.asarray(_open(ctx, a).convert("L")).astype(np.float32)
        for e in a.elements:
            if e.type != "face":
                continue
            assert e.value is None and e.layer == "raster"
            assert len(e.polygon) == 4
            for key in ("pose", "face_source", "height_px", "angle"):
                assert key in e.tags, (a.id, key)
            if a.id.startswith("rost_escena_real_"):
                # real scenes may come without an annotated core
                assert (e.core is None) == e.tags["no_core"], a.id
            else:
                assert e.core is not None, a.id
            if e.core is not None:
                pix = arr[_mask(e.core, arr.shape)]
                assert pix.size > 0 and pix.std() > 5, (a.id, e.tags)
            box_pix = arr[_mask(e.polygon, arr.shape)]
            assert box_pix.std() > 5, a.id


def test_text_with_ink(generated):
    ctx, files, _ = generated
    for a in files:
        arr = np.asarray(_open(ctx, a).convert("L")).astype(np.float32)
        for e in a.elements:
            if e.type == "face":
                continue
            pix = arr[_mask(e.polygon, arr.shape)]
            assert pix.size > 0 and pix.std() > 12, (a.id, e.value)


def test_counts_per_file(generated):
    _, files, _ = generated
    by_id = _by_id(files)
    for i in EXPECTED_IDS:
        n_faces = sum(e.type == "face" for e in by_id[i].elements)
        if i.startswith("rost_grupo_compuesto_"):
            assert 5 <= n_faces <= 12, i
        else:
            assert n_faces == 1, i
    poster = Counter(e.type for e in by_id["rost_afiche_impreso"].elements)
    assert poster["email"] == 1 and poster["phone"] == 1 and poster["name"] == 2
    assert poster["text"] >= 8
    elements = by_id["rost_afiche_impreso"].elements
    assert any(e.tags.get("decoy") == "date" for e in elements)
    name_levels = sorted(e.level for e in elements if e.type == "name")
    assert name_levels == ["base", "out_of_scope"]
    for e in elements:
        if e.type == "email":
            assert "format" in e.tags
        if e.type == "phone":
            assert {"format", "kind"} <= set(e.tags)


def test_levels(generated):
    _, files, _ = generated
    by_id = _by_id(files)
    stress = {"rost_perfil_01", "rost_perfil_02", "rost_baja_resolucion", "rost_oclusion"}
    for i in EXPECTED_IDS:
        if i.startswith("rost_grupo_") or i == "rost_afiche_impreso":
            continue
        (e,) = [e for e in by_id[i].elements if e.type == "face"]
        assert e.level == ("stress" if i in stress else "base"), i
    for i in (1, 2, 3):
        faces = [e for e in by_id[f"rost_grupo_compuesto_{i:02d}"].elements if e.type == "face"]
        heights = [e.tags["height_px"] for e in faces]
        assert min(heights) < 40 and max(heights) >= 80
        assert any(e.tags["occluded_fraction"] > 0 for e in faces), "there must be overlapping faces"
        for e in faces:
            assert 18 <= e.tags["height_px"] <= 205
            expected = "stress" if e.tags["height_px"] < 40 or e.tags["pose"] == "profile" else "base"
            assert e.level == expected
            assert e.tags["occluded_fraction"] <= 0.3


def test_angles(generated):
    _, files, _ = generated
    by_id = _by_id(files)
    for g in (90, 180, 270, 15, 45):
        (e,) = by_id[f"rost_retrato_rotado_{g}"].elements
        assert e.tags["angle"] == pytest.approx(g)
    assert all(e.tags["angle"] == 90 for e in by_id["rost_grupo_compuesto_03"].elements)
    (e,) = by_id["rost_espejado"].elements
    assert e.tags.get("mirror") == "horizontal"


def test_rotation_90_matches_pixels(generated):
    """The exact 90° rotation must bring the face crop to the same position as the GT."""
    ctx, files, _ = generated
    by_id = _by_id(files)
    rotated = np.asarray(_open(ctx, by_id["rost_retrato_rotado_90"]).convert("L")).astype(np.float32)
    upright = np.asarray(_open(ctx, by_id["rost_retrato_rotado_180"]).convert("L")).astype(np.float32)
    # 180 -> 90: rotate 270 more (both come from the same face with different scale)
    back = np.rot90(upright, k=-1)
    (e90,) = by_id["rost_retrato_rotado_90"].elements
    (e180,) = by_id["rost_retrato_rotado_180"].elements
    x0, y0, x1, y1 = (int(round(v)) for v in bounding_box(e90.core))
    h180, _ = upright.shape
    # core of the 180 one taken to the frame rotated 270 (clockwise 90): (x, y) -> (h - y, x)
    pts = np.asarray(e180.core)
    xr0, yr0 = (h180 - pts[:, 1]).min(), pts[:, 0].min()
    xr1, yr1 = (h180 - pts[:, 1]).max(), pts[:, 0].max()
    a = rotated[y0:y1, x0:x1]
    b = back[int(round(yr0)) : int(round(yr1)), int(round(xr0)) : int(round(xr1))]
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(a - b).mean() < 12


def test_real_scenes(ctx):
    """With scenes available: they are rescaled to <= 1600 px and the boxes are transformed alike."""
    sample_faces = ctx.faces.take("front", 3)
    cv = Canvas.new(2400, 1500, (180, 175, 160))
    faces = []
    for r, (x, width) in zip(sample_faces, [(100, 600), (900, 300), (1600, 30)], strict=True):
        (e,) = cv.paste(r.img, x, 300, width=width, elements=FaceProvider.elements(r))
        faces.append((e.polygon, e.core))
    faces[1] = (faces[1][0], None)  # a face without an annotated core
    faces.append((rect(2380, 20, 2420, 80), None))  # box that goes past the border (it gets clipped)
    ctx.faces._scenes = [Scene(id="demo/01", img=cv.img, faces=faces, source="synthetic_drawn")]
    files = face_scenes._real_scenes(ctx, ctx.rng("prueba"))
    (a,) = files
    assert a.id == "rost_escena_real_demo_01" and a.path == "faces/escena_real_demo_01.jpg"
    assert (a.pages[0].width, a.pages[0].height) == (1600, 1000)
    img = _open(ctx, a)
    assert img.size == (1600, 1000)
    assert len(a.elements) == 4
    f = 1600 / 2400
    for e, (box, core) in zip(a.elements, faces, strict=True):
        assert e.type == "face" and e.value is None and e.layer == "raster"
        cx0, cy0, cx1, cy1 = bounding_box(box)
        ex0, ey0, ex1, ey1 = bounding_box(e.polygon)
        assert ex0 == pytest.approx(cx0 * f, abs=0.01) and ey1 == pytest.approx(cy1 * f, abs=0.01)
        assert ex1 <= 1600 + 1e-6
        assert (e.core is None) == (core is None)
        expected = "base" if (ey1 - ey0) >= 24 else "stress"
        assert e.level == expected
    assert [e.level for e in a.elements][:3] == ["base", "base", "stress"]


def test_time(generated):
    _, _, seconds = generated
    # Timing depends on the machine load: warn, do not fail (timings are reported separately).
    if seconds > 90:
        warnings.warn(f"slow generation: {seconds:.1f} s", stacklevel=1)


def test_deterministic(generated, tmp_path):
    """Two runs with the same seed produce the same bytes and the same ground truth."""
    ctx, files, _ = generated
    ctx2 = _context(tmp_path / "other")
    files2 = face_scenes.generate(ctx2)
    assert [a.id for a in files] == [a.id for a in files2]
    for a, b in zip(files, files2, strict=True):
        assert (ctx.root / a.path).read_bytes() == (ctx2.root / b.path).read_bytes(), a.id
        assert [(e.type, e.value, e.level, e.polygon, e.core) for e in a.elements] == [
            (e.type, e.value, e.level, e.polygon, e.core) for e in b.elements
        ], a.id


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _rectify(img: np.ndarray, polygon, side: int = 64) -> np.ndarray:
    """Maps the quadrilateral (continuous coordinates) to a square of ``side`` px."""
    dest = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    m = cv2.getPerspectiveTransform(np.float32(polygon) - 0.5, dest)
    return cv2.warpPerspective(img, m, (side, side), flags=cv2.INTER_AREA)


def test_core_matches_source_photo(generated):
    """Independent geometry check: the core rectified from the output must look like the core
    of the source photo (rotations, scales, mirror and perspective included), and much less so
    if the polygon is shifted by 10 % of its width."""
    ctx, files, _ = generated
    source = {r.id: r for faces in ctx.faces._faces.values() for r in faces}
    checked = 0
    for a in files:
        if a.id in ("rost_oclusion", "rost_baja_resolucion"):
            continue
        arr = np.asarray(_open(ctx, a).convert("L")).astype(np.float32)
        for e in a.elements:
            if e.type != "face" or e.tags.get("face_id") not in source or e.tags["height_px"] < 40:
                continue
            if e.tags.get("occluded_fraction", 0) > 0:
                continue
            r = source[e.tags["face_id"]]
            template = _rectify(np.asarray(r.img.convert("L")).astype(np.float32), r.core)
            output = _rectify(arr, e.core)
            p = np.asarray(e.core, np.float64)
            shifted = p + 0.1 * (p[1] - p[0])
            assert _ncc(template, output) > 0.85, (a.id, e.tags)
            assert _ncc(template, output) > _ncc(template, _rectify(arr, shifted)) + 0.1, (a.id, e.tags)
            checked += 1
    assert checked >= 20
