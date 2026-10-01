"""Invariants of the ``images`` generator: readable files, geometry inside the page, polygons
over ink and metadata (EXIF, XMP, IPTC, PNG, TIFF) really written into the file."""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import piexif
import piexif.helper
import pytest
from PIL import Image, ImageOps, ImageSequence, IptcImagePlugin

from test_bench.context import Context
from test_bench.faces import FaceProvider
from test_bench.fake_data import FakeData
from test_bench.generators import images
from test_bench.schema import FileEntry, Manifest
from test_bench.visualize import visible_pages

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[FileEntry]]:
    root = tmp_path_factory.mktemp("images") / "generated"
    ctx = Context(
        root=root,
        seed=33,
        fake=FakeData(33),
        faces=FaceProvider(REPO_ROOT / "test_data" / "cache" / "faces", 33, allow_download=False),
    )
    return root, images.generate(ctx)


def _by_id(files: list[FileEntry]) -> dict[str, FileEntry]:
    return {a.id: a for a in files}


def _utf8(value: str) -> str:
    """Pillow decodes ASCII tags as Latin-1; the file stores UTF-8 (like exiftool)."""
    return value.encode("latin-1").decode("utf-8")


def _mask(polygon, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(polygon, np.float64) - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def test_manifest_consistent(generated):
    root, files = generated
    man = Manifest(root=str(root), seed=33)
    for a in files:
        man.add(a)
    assert man.validate() == []
    assert all(a.id.startswith("img_") for a in files)
    cats = Counter(a.category for a in files)
    assert cats["rotated_image"] == 17 and cats["exif"] == 8 and cats["tiff"] == 2
    paths = {a.path for a in files}
    for stem in ("nota", "tarjeta"):
        for ang in (0, 90, 180, 270, 15, 45):
            assert any(r.startswith(f"rotated_images/{stem}_{ang}.") for r in paths)


def test_counts_by_type(generated):
    _, files = generated
    n = Counter((e.type, e.level) for a in files for e in a.elements)
    assert n[("rut", "base")] >= 12 and n[("email", "base")] >= 20 and n[("phone", "base")] >= 25
    assert n[("rut", "stress")] >= 3 and n[("phone", "stress")] >= 3
    assert n[("name", "out_of_scope")] >= 3 and n[("name", "base")] >= 10
    assert n[("face", "base")] == 2
    assert n[("url", "base")] >= 6 and n[("address", "base")] >= 4
    decoys = Counter(e.tags.get("decoy") for a in files for e in a.elements if e.type == "text")
    assert decoys["date"] >= 10 and decoys["amount"] >= 2
    # the base 45 degrees are recorded as rotated quadrilaterals
    a45 = _by_id(files)["img_rot_nota_45"]
    e = next(e for e in a45.elements if e.type == "rut")
    p = np.asarray(e.polygon)
    assert len(p) == 4 and abs(p[:, 0].min() - p[:, 0].max()) > 20 and len({round(x) for x in p[:, 1]}) == 4


def test_geometry_and_ink(generated):
    root, files = generated
    for a in files:
        pages = visible_pages(root / a.path, a.format)
        assert len(pages) == len(a.pages), a.id
        for pg, img in zip(a.pages, pages, strict=True):
            assert (pg.width, pg.height) == img.size, a.id
        grays = [np.asarray(img.convert("L"), np.float32) for img in pages]
        for e in a.elements:
            assert e.layer == "raster"
            pg = a.pages[e.page]
            p = np.asarray(e.polygon)
            assert p[:, 0].min() >= -0.5 and p[:, 1].min() >= -0.5, (a.id, e.value)
            assert p[:, 0].max() <= pg.width + 0.5 and p[:, 1].max() <= pg.height + 0.5, (a.id, e.value)
            pix = grays[e.page][_mask(e.polygon, grays[e.page].shape)]
            assert pix.size > 0, (a.id, e.value)
            contrast = np.percentile(pix, 95) - np.percentile(pix, 5)
            minimum = 20 if "bajo_contraste" in a.id else 40
            assert contrast > minimum, (a.id, e.type, e.value, contrast)
            if e.type == "face":
                n = np.asarray(e.core)
                assert n[:, 0].min() >= p[:, 0].min() and n[:, 0].max() <= p[:, 0].max()


def _fit(gray: np.ndarray, polygon, threshold: float, ring: float = 4.0) -> tuple[float, list[bool]]:
    """Independent check of how tightly a text quadrilateral fits, using only the final pixels.

    In the local frame of the quadrilateral (u axis along the line, v along the height) it
    measures what fraction of the ink falls in an outer ring of ``ring`` px, and whether there is
    ink touching each of the four edges (tight box, not a loose one).
    """
    p = np.asarray(polygon, np.float64)
    o, eu, ev = p[0], p[1] - p[0], p[3] - p[0]
    length, height = np.linalg.norm(eu), np.linalg.norm(ev)
    u, v = eu / length, ev / height
    x0, y0 = np.maximum(np.floor(p.min(0) - ring - 1).astype(int), 0)
    x1, y1 = np.ceil(p.max(0) + ring + 1).astype(int)
    x1, y1 = min(x1, gray.shape[1]), min(y1, gray.shape[0])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx, cy = xx + 0.5 - o[0], yy + 0.5 - o[1]
    uu, vv = cx * u[0] + cy * u[1], cx * v[0] + cy * v[1]
    sub = gray[y0:y1, x0:x1]
    inside = (uu >= 0) & (uu <= length) & (vv >= 0) & (vv <= height)
    outside = (uu >= -ring) & (uu <= length + ring) & (vv >= -ring) & (vv <= height + ring) & ~inside
    ink = np.abs(sub - np.median(sub[outside])) > threshold
    ti, to = int(ink[inside].sum()), int(ink[outside].sum())
    b = 2.5
    edges = [
        bool(ink[inside & (uu < b)].any()),
        bool(ink[inside & (uu > length - b)].any()),
        bool(ink[inside & (vv < b)].any()),
        bool(ink[inside & (vv > height - b)].any()),
    ]
    return to / max(1, ti + to), edges


def test_boxes_fit_the_ink(generated):
    """Each text box contains its ink (almost nothing is left outside) and touches it on all four sides."""
    root, files = generated
    for a in files:
        grays = [np.asarray(img.convert("L"), np.float32) for img in visible_pages(root / a.path, a.format)]
        threshold = 20 if "bajo_contraste" in a.id else 45
        for e in a.elements:
            if e.type == "face":
                continue
            leak, edges = _fit(grays[e.page], e.polygon, threshold)
            assert leak < 0.12, (a.id, e.type, e.value, leak)
            assert all(edges), (a.id, e.type, e.value, edges)


def test_tags_for_breakdowns(generated):
    """Every element carries angle and degradation (the report breaks recall down by them)."""
    _, files = generated
    for a in files:
        for e in a.elements:
            assert "degradation" in e.tags, (a.id, e.type, e.value)
            if e.type != "face":
                assert "angle" in e.tags and "size_px" in e.tags, (a.id, e.value)
    # the commune after the address is recorded exactly as written (with the comma)
    communes = [e for a in files for e in a.elements if e.type == "text" and e.value.startswith(", ")]
    assert len(communes) == 10  # 8 cards in rotated_image, the WEBP one and page 1 of the TIFF


def test_stress_and_output_formats(generated):
    _, files = generated
    formats = Counter(e.tags.get("format") for a in files for e in a.elements)
    assert any(formats[f] for f in ("no_hyphen", "commas", "inner_spaces"))
    assert any(formats[f] for f in ("old_mobile_09", "old_mobile_8", "old_regional_0"))
    assert any(formats[f] for f in ("spelled_arroba", "at", "spaces"))
    by_angle: dict[int, set[str]] = {}
    for a in files:
        if a.category == "rotated_image" and not a.tags["stress"]:
            by_angle.setdefault(a.tags["angle"], set()).add(a.format)
    assert all(len(f) == 2 for f in by_angle.values()), by_angle  # note and card in different formats


def test_unique_values_in_image(generated):
    _, files = generated
    seen: set[tuple[str, str]] = set()
    for a in files:
        for e in a.elements:
            if e.type in ("rut", "email", "phone"):
                key = (e.type, e.value)
                assert key not in seen, key
                seen.add(key)
    canaries = [m.value for a in files for m in a.sensitive_metadata if m.value]
    visible = {e.value for a in files for e in a.elements}
    assert not set(canaries) & visible  # metadata canaries are not reused in the content


def _in_bytes(data: bytes, value: str) -> bool:
    return any(value.encode(c) in data for c in ("utf-8", "latin-1", "utf-16-le", "utf-16-be") if _encodes(value, c))


def _encodes(value: str, encoding: str) -> bool:
    try:
        value.encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def test_canaries_present_in_bytes(generated):
    root, files = generated
    for a in files:
        data = (root / a.path).read_bytes()
        for m in a.sensitive_metadata:
            if m.value:
                assert _in_bytes(data, m.value), (a.id, m.location, m.value)


def test_orientation_6_full_exif(generated):
    root, files = generated
    a = _by_id(files)["img_exif_orientacion_6"]
    path = root / a.path
    meta = {m.location: m for m in a.sensitive_metadata}
    ex = piexif.load(str(path))
    assert ex["0th"][piexif.ImageIFD.Orientation] == 6
    assert ex["0th"][piexif.ImageIFD.Artist].decode("utf-8") == meta["exif.artist"].value
    assert meta["exif.image_description"].value in ex["0th"][piexif.ImageIFD.ImageDescription].decode()
    assert piexif.helper.UserComment.load(ex["Exif"][piexif.ExifIFD.UserComment]) == meta["exif.user_comment"].value
    xp = bytes(ex["0th"][piexif.ImageIFD.XPAuthor]).decode("utf-16-le").rstrip("\x00")
    assert xp == meta["exif.xp_author"].value
    assert meta["exif.copyright"].value in ex["0th"][piexif.ImageIFD.Copyright].decode()
    assert ex["GPS"][piexif.GPSIFD.GPSLatitudeRef] == b"S"
    lat = sum(n / d / 60**i for i, (n, d) in enumerate(ex["GPS"][piexif.GPSIFD.GPSLatitude]))
    assert lat == pytest.approx(-meta["exif.gps"].tags["lat"], abs=1e-4)
    assert ex["1st"][piexif.ImageIFD.Compression] == 6  # IFD1 declares a JPEG thumbnail
    thumbnail = Image.open(io.BytesIO(ex["thumbnail"]))
    assert max(thumbnail.size) <= 160 and thumbnail.width < thumbnail.height  # stored unrotated (portrait)
    with Image.open(path) as im:
        assert im.size == (a.pages[0].height, a.pages[0].width)
        assert meta["xmp.dc:creator"].value in im.info["xmp"].decode("utf-8")
        assert ImageOps.exif_transpose(im).size == (a.pages[0].width, a.pages[0].height)


@pytest.mark.parametrize("orientation", [3, 8])
def test_orientations_with_gps(generated, orientation):
    root, files = generated
    a = _by_id(files)[f"img_exif_orientacion_{orientation}"]
    ex = piexif.load(str(root / a.path))
    assert ex["0th"][piexif.ImageIFD.Orientation] == orientation
    assert piexif.GPSIFD.GPSLatitude in ex["GPS"]
    with Image.open(root / a.path) as im:
        expected = (a.pages[0].width, a.pages[0].height)
        assert im.size == (expected if orientation == 3 else expected[::-1])


def test_wrong_orientation(generated):
    root, files = generated
    a = _by_id(files)["img_exif_orientacion_incorrecta"]
    ex = piexif.load(str(root / a.path))
    assert ex["0th"][piexif.ImageIFD.Orientation] == 6
    with Image.open(root / a.path) as im:
        assert im.size == (a.pages[0].height, a.pages[0].width)
    assert a.tags["wrong_exif"] and all(e.tags["wrong_exif"] for e in a.elements)
    # in the displayed geometry the text ends up sideways: boxes are taller than wide
    for e in a.elements:
        if e.type in ("rut", "email", "name"):
            p = np.asarray(e.polygon)
            assert np.ptp(p[:, 1]) > np.ptp(p[:, 0])


def test_webp_exif_xmp(generated):
    root, files = generated
    a = _by_id(files)["img_exif_webp_gps"]
    meta = {m.location: m for m in a.sensitive_metadata}
    with Image.open(root / a.path) as im:
        assert im.format == "WEBP"
        exif = im.getexif()
        assert _utf8(exif[315]) == meta["exif.artist"].value
        assert exif.get_ifd(0x8825)
        assert meta["xmp.dc:creator"].value in im.info["xmp"].decode("utf-8")


def test_png_text_xmp_exif(generated):
    root, files = generated
    a = _by_id(files)["img_exif_png_metadatos"]
    meta = {m.location: m for m in a.sensitive_metadata}
    with Image.open(root / a.path) as im:
        assert im.info["Author"] == meta["png.text.Author"].value
        assert im.info["Comment"] == meta["png.text.Comment"].value
        assert meta["png.text.Description"].value in im.info["Description"]
        assert meta["xmp.dc:creator"].value in im.info["XML:com.adobe.xmp"]
        assert im.getexif().get_ifd(0x8825)
    assert b"eXIf" in (root / a.path).read_bytes()


def test_face_gps_and_iptc(generated):
    root, files = generated
    by_id = _by_id(files)
    a = by_id["img_exif_rostro_gps"]
    ex = piexif.load(str(root / a.path))
    assert ex["GPS"] and ex["0th"][piexif.ImageIFD.Artist].decode() == a.sensitive_metadata[1].value
    assert [e.type for e in a.elements] == ["face"]
    b = by_id["img_exif_iptc"]
    meta = {(m.location, m.tags["type"]): m for m in b.sensitive_metadata}
    with Image.open(root / b.path) as im:
        iptc = IptcImagePlugin.getiptcinfo(im)
        assert iptc[(2, 80)].decode("utf-8") == meta[("iptc.byline", "name")].value
        assert meta[("iptc.caption", "rut")].value in iptc[(2, 120)].decode("utf-8")
        im.load()


def test_tiff_multipage(generated):
    root, files = generated
    a = _by_id(files)["img_tiff_multipagina"]
    meta = {m.location: m for m in a.sensitive_metadata}
    with Image.open(root / a.path) as im:
        assert im.n_frames == 3
        frames = [(f.size, f.info["compression"], dict(f.tag_v2)) for f in ImageSequence.Iterator(im)]
    assert [m[0] for m in frames] == [(p.width, p.height) for p in a.pages]
    assert len({m[0] for m in frames}) == 3
    assert [m[1] for m in frames] == ["tiff_lzw", "tiff_adobe_deflate", "jpeg"]
    tags = frames[0][2]
    assert _utf8(tags[315]) == meta["tiff.artist"].value
    assert meta["tiff.image_description"].value in _utf8(tags[270])
    assert meta["tiff.document_name"].value in _utf8(tags[269])
    assert meta["tiff.page_name"].value in _utf8(tags[285])
    assert 305 in tags and 315 not in frames[1][2]
    assert Counter(e.page for e in a.elements)[2] == 3  # face + caption on page 2
    assert any(e.type == "face" and e.page == 2 for e in a.elements)


def test_tiff_bw_scan(generated):
    root, files = generated
    a = _by_id(files)["img_tiff_escaneo_bn"]
    with Image.open(root / a.path) as im:
        assert im.mode == "1" and im.info["compression"] == "group4"
        assert tuple(round(v) for v in im.info["dpi"]) == (200, 200)
    types = Counter(e.type for e in a.elements)
    assert types["rut"] == 2 and types["email"] == 2 and types["phone"] == 2
    assert all(e.level in ("base", "out_of_scope") for e in a.elements)
