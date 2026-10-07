"""Regression tests from the adversarial review of D14 (pages exported as an image): what must never
stay in the file under a black box, what must not make an ordinary page an image, the text check
of pages exported as an image, the search of the file's own strings and the time budgets.
Invented documents only.

The oracle is the review's: after the redaction, the redaction's own black boxes are taken out of
the output and what is still painted under each zone is rendered."""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from anonymizer.engine import common, faces, leftovers, ocr, pdf, qr
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import AnalyzedFile, Finding
from anonymizer.engine.real import RealEngine

W = H = 300
_BOX = re.compile(
    rb"q\s+\d[\d. ]* \d[\d. ]* \d[\d. ]* \d[\d. ]* re\s+h\s+0 0 0 RG 0 0 0 rg B\s+Q|q 0 0 0 rg 0 0 0 RG [-\d. ]+ re B Q"
)


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr(ocr, "read_lines", lambda bgr, min_side=0, check=None: [])
    monkeypatch.setattr(faces, "detect", lambda bgr, threshold=0.5, check=None: [])
    monkeypatch.setattr(qr, "detect", lambda bgr: [])


def new_stream(doc, data: bytes, dict_src: str = "<<>>") -> int:
    x = doc.get_new_xref()
    doc.update_object(x, dict_src)
    doc.update_stream(x, data)
    return x


def build(path: Path, content: bytes, resources: str = "<<>>", setup=None) -> Path:
    """One page with raw ``content``; ``setup(doc)`` returns names -> xref for ``resources``."""
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    subs = setup(doc) if setup else {}
    doc.xref_set_key(page.xref, "Contents", f"{new_stream(doc, content)} 0 R")
    doc.xref_set_key(page.xref, "Resources", resources.format(**subs) if subs else resources)
    doc.save(path)
    doc.close()
    return path


def secret_left(src: Path, out: Path, zones, colour: str) -> tuple[int, int]:
    """Pixels of the secret colour under the zones, in the source and in the output once its black
    boxes are taken out (a page exported as an image keeps them: it shows black)."""
    revealed = out.with_name(out.stem + "_rev.pdf")
    with pymupdf.open(out) as doc:
        for page in doc:
            for x in page.get_contents():
                doc.update_stream(x, _BOX.sub(b"", doc.xref_stream(x)))
        doc.save(revealed)

    def count(path):
        total = 0
        with pymupdf.open(path) as doc:
            for z in zones:
                pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=pymupdf.Rect(z) + (1, 1, -1, -1), alpha=False)
                a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
                if colour == "dark":
                    total += int((a.max(axis=2) < 100).sum())
                elif colour == "red":
                    total += int(((a[:, :, 0] > 180) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)).sum())
                else:
                    total += int(((a[:, :, 2] > 180) & (a[:, :, 0] < 90) & (a[:, :, 1] < 90)).sum())
        return total

    return count(src), count(revealed)


def redact(tmp_path: Path, name: str, content: bytes, resources: str, setup, zones, colour: str):
    src = build(tmp_path / f"{name}.pdf", content, resources, setup)
    out = tmp_path / f"{name}_out.pdf"
    outcome = pdf.redact(str(src), str(out), {0: [pymupdf.Rect(z) for z in zones]})[0]
    before, after = secret_left(src, out, zones, colour)
    return outcome, before, after


# ---------------------------------------------------------------------------
# Never silent: images and data under a zone (review I1, I2, Type3)
# ---------------------------------------------------------------------------

N = 40
CHECK = bytes(255 if ((i // 4) + (j // 4)) % 2 else 0 for j in range(N) for i in range(N))
RED = b"\xff\x00\x00" * N * N


def red_img(doc):
    return {
        "im": new_stream(
            doc, RED, f"<</Type/XObject/Subtype/Image/Width {N}/Height {N}/ColorSpace/DeviceRGB/BitsPerComponent 8>>"
        )
    }


def mask_img(doc):
    bits = bytes([0x00] * (N // 8) * N)
    return {
        "im": new_stream(
            doc, bits, f"<</Type/XObject/Subtype/Image/Width {N}/Height {N}/ImageMask true/BitsPerComponent 1>>"
        )
    }


def smask_img(doc):
    sm = new_stream(
        doc, CHECK, f"<</Type/XObject/Subtype/Image/Width {N}/Height {N}/ColorSpace/DeviceGray/BitsPerComponent 8>>"
    )
    im = new_stream(
        doc,
        RED,
        f"<</Type/XObject/Subtype/Image/Width {N}/Height {N}/ColorSpace/DeviceRGB/BitsPerComponent 8/SMask {sm} 0 R>>",
    )
    return {"im": im}


def pattern_img(doc):
    im = red_img(doc)["im"]
    pat = new_stream(
        doc,
        b"q 40 0 0 40 0 0 cm /Im0 Do Q",
        f"<</Type/Pattern/PatternType 1/PaintType 1/TilingType 1/BBox[0 0 40 40]/XStep 40/YStep 40/Resources<</XObject<</Im0 {im} 0 R>>>>>>",
    )
    return {"pat": pat}


def softmask_group_img(doc):
    im = new_stream(
        doc, CHECK, f"<</Type/XObject/Subtype/Image/Width {N}/Height {N}/ColorSpace/DeviceGray/BitsPerComponent 8>>"
    )
    grp = new_stream(
        doc,
        b"q 40 0 0 40 100 130 cm /Im0 Do Q",
        f"<</Type/XObject/Subtype/Form/BBox[0 0 300 300]/Group<</S/Transparency/CS/DeviceGray>>/Resources<</XObject<</Im0 {im} 0 R>>>>>>",
    )
    return {"grp": grp}


def softmask_group_path(doc):
    body = b"1 g 105 135 m 135 135 135 165 120 165 c 105 165 105 150 115 150 c f"
    return {
        "grp": new_stream(
            doc, body, "<</Type/XObject/Subtype/Form/BBox[0 0 300 300]/Group<</S/Transparency/CS/DeviceGray>>>>"
        )
    }


def type3_img(doc):
    proc = new_stream(doc, b"1000 0 d0 q 1 0 0 rg 1000 0 0 1000 0 0 cm /Im0 Do Q")
    im = red_img(doc)["im"]
    font = doc.get_new_xref()
    doc.update_object(
        font,
        f"<</Type/Font/Subtype/Type3/FontBBox[0 0 1000 1000]/FontMatrix[0.001 0 0 0.001 0 0]/CharProcs<</a {proc} 0 R>>/Encoding<</Type/Encoding/Differences[65/a]>>/FirstChar 65/LastChar 65/Widths[1000]/Resources<</XObject<</Im0 {im} 0 R>>>>>>",
    )
    return {"f": font}


def type3_far(doc):
    # a Type3 glyph drawn far to the right of its own box (d1 says 0..1000)
    proc = new_stream(doc, b"1000 0 0 0 1000 1000 d1 1500 0 m 2500 0 l 2500 1000 2000 1000 2000 500 c f")
    font = doc.get_new_xref()
    doc.update_object(
        font,
        f"<</Type/Font/Subtype/Type3/FontBBox[0 0 1000 1000]/FontMatrix[0.001 0 0 0.001 0 0]/CharProcs<</a {proc} 0 R>>/Encoding<</Type/Encoding/Differences[65/a]>>/FirstChar 65/LastChar 65/Widths[1000]>>",
    )
    return {"f": font}


def form_img(doc):
    im = red_img(doc)["im"]
    return {
        "fm": new_stream(
            doc,
            b"q 40 0 0 40 0 0 cm /Im0 Do Q",
            f"<</Type/XObject/Subtype/Form/BBox[0 0 40 40]/Resources<</XObject<</Im0 {im} 0 R>>>>>>",
        )
    }


INLINE = f"q 40 0 0 40 100 130 cm BI /W {N} /H {N} /CS /RGB /BPC 8 ID ".encode() + RED + b"\nEI Q"
GS = "<</ExtGState<</GS1<</SMask<</S/Luminosity/G {grp} 0 R>>>>>>>>"
IMAGES = {
    "i1_image": (b"q 40 0 0 40 100 130 cm /Im0 Do Q", "<</XObject<</Im0 {im} 0 R>>>>", red_img, False),
    "i2_stencil_mask": (b"q 1 0 0 rg 40 0 0 40 100 130 cm /Im0 Do Q", "<</XObject<</Im0 {im} 0 R>>>>", mask_img, False),
    "i3_smask_image": (b"q 40 0 0 40 100 130 cm /Im0 Do Q", "<</XObject<</Im0 {im} 0 R>>>>", smask_img, False),
    "i4_inline_image": (INLINE, "<<>>", None, False),
    "i5_pattern_image": (b"/Pattern cs /P0 scn 100 130 40 40 re f", "<</Pattern<</P0 {pat} 0 R>>>>", pattern_img, True),
    "i6_softmask_group_image": (b"q /GS1 gs 1 0 0 rg 90 120 120 60 re f Q", GS, softmask_group_img, True),
    "i6b_softmask_group_path": (b"q /GS1 gs 1 0 0 rg 90 120 120 60 re f Q", GS, softmask_group_path, True),
    "i7_type3_image": (b"BT /F1 40 Tf 100 130 Td (A) Tj ET", "<</Font<</F1 {f} 0 R>>>>", type3_img, False),
    "i8_form_image": (b"q 1 0 0 1 100 130 cm /Fm0 Do Q", "<</XObject<</Fm0 {fm} 0 R>>>>", form_img, False),
    "i9_type3_far": (b"1 0 0 rg BT /F1 20 Tf 80 140 Td (A) Tj ET", "<</Font<</F1 {f} 0 R>>>>", type3_far, True),
}


@pytest.mark.parametrize("name", list(IMAGES))
def test_no_image_or_glyph_survives_under_a_zone(tmp_path, name):
    content, resources, setup, image_page = IMAGES[name]
    outcome, before, after = redact(tmp_path, name, content, resources, setup, [(110, 140, 130, 160)], "red")
    assert before > 1000  # the secret is there to begin with
    assert after == 0, (name, outcome.reasons)
    if image_page:  # what the redaction cannot take out: the page goes as an image
        assert outcome.rasterized, name


random.seed(7)
GRID = [(i, j) for i in range(12) for j in range(12) if random.random() < 0.5]


def modules(op: str = "re") -> str:
    out = []
    for i, j in GRID:
        x, y = 100 + 3 * i, 130 + 3 * j
        out.append(f"{x} {y} 3 3 re" if op == "re" else f"{x} {y} m {x + 3} {y} l {x + 3} {y + 3} l {x} {y + 3} l h")
    return " ".join(out)


RULE = "20 40 260 0.6 re"
VECTORS = {
    "v1_black_modules": (f"0 0 0 rg {modules()} {RULE} f", "dark"),
    "v1b_gray_modules": (f"0.25 0.25 0.25 rg {modules()} {RULE} f", "dark"),
    "v1c_cmyk_black": (f"0 0 0 1 k {modules()} {RULE} f", "dark"),
    "v1d_black_fill_stroke": (f"0 0 0 rg 0 0 0 RG 0.2 w {modules()} {RULE} B", "dark"),
    "v2_modules_as_lines": (f"0 0 0 rg {modules('l')} {RULE} f", "dark"),
    "v3_clip_of_modules": (f"q {modules()} W n 0 0 0 rg 0 0 300 300 re f Q", "dark"),
    "v3b_clip_of_modules_red": (f"q {modules()} W n 1 0 0 rg 0 0 300 300 re f Q", "red"),
    "v5_one_module_a_path": (
        " ".join(f"0 0 0 rg {100 + 3 * i} {130 + 3 * j} 3 3 re f" for i, j in GRID) + f" {RULE} f",
        "dark",
    ),
    "v6_segments_fill_stroke": (
        "0 0 0 rg 0 0 0 RG 1 w "
        + " ".join(f"{100 + 3 * i} {130 + 3 * j} m {100 + 3 * i + 2.5} {130 + 3 * j} l" for i, j in GRID)
        + " 20 40 m 280 40 l B",
        "dark",
    ),
    "v6b_segments_stroke": (
        "0 0 0 RG 1 w "
        + " ".join(f"{100 + 3 * i} {130 + 3 * j} m {100 + 3 * i + 2.5} {130 + 3 * j} l" for i, j in GRID)
        + " 20 40 m 280 40 l S",
        "dark",
    ),
}


@pytest.mark.parametrize("name", list(VECTORS))
def test_no_module_survives_under_a_zone(tmp_path, name):
    content, colour = VECTORS[name]
    outcome, before, after = redact(tmp_path, name, content.encode(), "<<>>", None, [(98, 132, 138, 172)], colour)
    assert before > 1000
    if outcome.rasterized:  # one image of the redacted page: nothing but its pixels, black in the zone
        with pymupdf.open(tmp_path / f"{name}_out.pdf") as doc:
            page = doc[0]
            assert not page.get_drawings() and not page.get_text().strip() and len(page.get_images()) == 1
            pix = page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=pymupdf.Rect(99, 133, 137, 171), alpha=False)
            assert max(pix.samples) < 60
    else:
        assert after == 0, (name, outcome.reasons)
    if name.startswith("v3"):  # the clip of modules, or what it paints, makes the page an image
        assert outcome.rasterized and outcome.reasons in (["clip"], ["ink"], ["clip", "ink"])


def test_a_barcode_crossing_the_zone_leaves_nothing_under_it(tmp_path):
    bars = " ".join(f"{90 + 2 * k} 140 {1 + (k * 7) % 3 * 0.5} 30 re" for k in range(30))
    outcome, before, after = redact(
        tmp_path, "v4", f"0 0 0 rg {bars} f".encode(), "<<>>", None, [(110, 128, 140, 162)], "dark"
    )
    assert before > 1000 and after == 0, outcome.reasons


# ---------------------------------------------------------------------------
# Ordinary layout and backgrounds do not make a page an image (review I4 and minor)
# ---------------------------------------------------------------------------

TEXT = b"BT /F1 10 Tf 105 162 Td (Ana Maria Soto Perez) Tj ET "
HELV = "<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>"
LAYOUT = {
    "underline_single": b"0 0 1 RG 0.6 w 105 158 m 215 158 l S",
    "underline_in_pieces": b"0 0 1 RG 1 w 105 157.5 m 130 157.5 l 160 157.5 l 190 157.5 l 215 157.5 l S",
    "underline_joined_subpaths": b"0 0 1 RG 1 w 105 157.5 m 160 157.5 l 160 157.5 m 215 157.5 l S",
    "underline_two_paths": b"0 0 1 RG 1 w 105 157.5 m 160 157.5 l S 160 157.5 m 215 157.5 l S",
    "rounded_frame_tight": (b"0.3 0.3 0.3 RG 0.75 w 104 141 m 214 141 l 218 141 220 143 220 147 c 220 168 l "
                            b"220 170.5 218 172 214 172 c 104 172 l 100 172 98 170.5 98 168 c 98 147 l 98 143 100 141 104 141 c h S"),
    "cell_corner_polyline": b"0 0 0 RG 0.5 w 100 175 m 100 158 l 230 158 l S",
    "dashed_underline": b"0 0 0 RG [2 1] 0 d 0.6 w 105 158 m 215 158 l S",
    "filled_rect_underline": b"0 0 1 rg 105 157.4 110 0.8 re f",
    "logo_curve_touching": b"0.1 0.3 0.6 rg 60 150 m 60 175 104 175 104 150 c f",
}  # fmt: skip


@pytest.mark.parametrize("name", list(LAYOUT))
def test_ordinary_layout_next_to_a_name_keeps_the_page_vector(tmp_path, name):
    outcome, _, after = redact(tmp_path, name, TEXT + LAYOUT[name], HELV, None, [(103, 127, 218, 143)], "red")
    assert not outcome.rasterized, (name, outcome.reasons)
    assert after == 0


def shading(doc):
    sh = doc.get_new_xref()
    doc.update_object(
        sh,
        "<</ShadingType 2/ColorSpace/DeviceRGB/Coords[0 0 0 300]/Function<</FunctionType 2/Domain[0 1]/C0[1 1 1]/C1[0.85 0.9 1]/N 1>>/Extend[true true]>>",
    )
    return {"sh": sh}


def tiling(doc):
    return {"pat": new_stream(doc, b"0.85 0.9 1 RG 0.3 w 0 0 m 6 6 l S 0 6 m 6 0 l S",
                              "<</Type/Pattern/PatternType 1/PaintType 1/TilingType 1/BBox[0 0 6 6]/XStep 6/YStep 6/Resources<<>>>>")}  # fmt: skip


@pytest.mark.parametrize(
    "name, content, resources, setup, image",
    [
        (
            "gradient",
            b"q /Sh0 sh Q",
            "<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>/Shading<</Sh0 {sh} 0 R>>>>",
            shading,
            False,
        ),
        ("flat_band", b"0.85 0.9 1 rg 0 0 300 300 re f", HELV, None, False),
        # A hatched background shows edges under the zone however light it is: it may be data, and the
        # page is an image (the closing review: light data must not pass).
        (
            "light_hatch",
            b"/Pattern cs /P0 scn 0 0 300 300 re f",
            "<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>/Pattern<</P0 {pat} 0 R>>>>",
            tiling,
            True,
        ),
    ],
)
def test_a_background_that_holds_the_zone(tmp_path, name, content, resources, setup, image):
    outcome, _, _ = redact(tmp_path, name, content + b" 0 g " + TEXT, resources, setup, [(103, 127, 218, 143)], "blue")
    assert outcome.rasterized is image, (name, outcome.reasons)


# ---------------------------------------------------------------------------
# A page exported as an image still has its text checked (review I3)
# ---------------------------------------------------------------------------


def _manual(x0, y0, x1, y1) -> Finding:
    return Finding(id="m1", file_id="n", page=0, type="manual", detector="reviewer", status="added",
                   polygon=common.rect_polygon(x0, y0, x1, y1))  # fmt: skip


def test_data_nobody_marked_blocks_a_page_exported_as_an_image(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Solicitud de prueba con datos inventados.", fontsize=11)
        page.insert_text((72, 130), "Correo de contacto: ana.soto@correo-inventado.cl", fontsize=11)
        page.insert_text((72, 160), "RUT del solicitante: 12.345.678-5", fontsize=11)
        page.draw_circle((200, 400), 30, color=None, fill=(0.2, 0.3, 0.7))  # a logo
        doc.save(tmp_path / "net.pdf")
    for label, zone in (("word", (72, 88, 130, 104)), ("half_logo", (150, 360, 200, 440))):
        file = AnalyzedFile(id=label, name="net.pdf", path=str(tmp_path / "net.pdf"), kind="pdf",
                            findings=[_manual(*zone)], status="confirmed")  # fmt: skip
        result = RealEngine().export(file, str(tmp_path / label))
        assert not result.exported, label  # the detectors missed the e-mail and the RUT
        messages = " ".join(leak.message for leak in result.leaks)
        assert "un RUT legible" in messages and "un correo legible" in messages, (label, messages)


def test_a_page_exported_as_an_image_with_nothing_unmarked_exports(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Texto neutro de relleno.", fontsize=11)
        page.draw_circle((200, 400), 30, color=None, fill=(0.2, 0.3, 0.7))
        doc.save(tmp_path / "logo.pdf")
    file = AnalyzedFile(id="l", name="logo.pdf", path=str(tmp_path / "logo.pdf"), kind="pdf",
                        findings=[_manual(150, 360, 200, 440)], status="confirmed")  # fmt: skip
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert result.exported and result.rasterized_pages, [leak.message for leak in result.leaks]


# ---------------------------------------------------------------------------
# The file's own strings (PLAN section 7)
# ---------------------------------------------------------------------------


def test_a_value_left_in_the_strings_of_the_file_blocks_the_export(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Nombre: Ana Prueba Soto", fontsize=11)
        doc.xref_set_key(page.xref, "LastModifiedBy", "(Ana Prueba Soto)")  # a key nobody cleans
        doc.save(tmp_path / "s.pdf")
    with pymupdf.open(tmp_path / "s.pdf") as doc:
        hit = doc[0].search_for("Ana Prueba Soto")[0]
    finding = Finding(id="n1", file_id="s", page=0, type="name", polygon=common.rect_polygon(*hit),
                      text="Ana Prueba Soto", detector="name_list")  # fmt: skip
    file = AnalyzedFile(id="s", name="s.pdf", path=str(tmp_path / "s.pdf"), kind="pdf", findings=[finding],
                        status="confirmed")  # fmt: skip
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert not result.exported
    assert any("dentro del archivo" in leak.message and leak.finding_id == "n1" for leak in result.leaks)


# ---------------------------------------------------------------------------
# Time budgets and the after's cache
# ---------------------------------------------------------------------------


def test_the_check_keeps_to_its_budget_on_a_page_of_long_paths(tmp_path):
    rng = np.random.default_rng(9)
    ops = ["0.2 0.2 0.2 RG 0.3 w"]
    for _ in range(60):
        for _ in range(1000):
            x, y = rng.uniform(20, 280, 2)
            if 130 < x < 170 and 130 < y < 170:
                x += 60
            ops.append(f"{x:.1f} {y:.1f} m {x + 1:.1f} {y + 2:.1f} {x + 2:.1f} {y - 1:.1f} {x + 3:.1f} {y + 1:.1f} c")
        ops.append("S")
    src = build(tmp_path / "long.pdf", " ".join(ops).encode())
    with PDF_LOCK, pymupdf.open(src) as doc:
        started = time.monotonic()
        found = leftovers.check(doc[0], [pymupdf.Rect(145, 145, 155, 155)], deadline=started + 1.0)
        assert time.monotonic() - started < 4.0
    assert found in ([], ["time"])


def test_the_after_is_redacted_once_per_set_of_zones(tmp_path, monkeypatch):
    path = build(tmp_path / "a.pdf", TEXT + b"0.1 0.3 0.6 rg 60 150 m 60 175 104 175 104 150 c f", HELV)
    calls = []
    real = pdf.redact_page

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pdf, "redact_page", counting)
    engine = RealEngine()
    file = AnalyzedFile(id="a", name="a.pdf", path=str(path), kind="pdf", findings=[_manual(103, 127, 218, 143)],
                        status="ready")  # fmt: skip
    first = engine.render_result(file, 0, 1.0, file.findings)
    engine.render_result(file, 0, 1.5, file.findings)
    assert engine.render_result(file, 0, 1.0, file.findings) == first
    assert len(calls) == 1
    file.findings.append(Finding(id="m2", file_id="a", page=0, type="manual", detector="reviewer", status="added",
                                 polygon=common.rect_polygon(20, 20, 60, 40)))  # fmt: skip
    engine.render_result(file, 0, 1.0, file.findings)
    assert len(calls) == 2


def test_text_outside_the_crop_box_does_not_make_the_page_an_image(tmp_path):
    # The test set's stress page has a name and a RUT written outside its crop box: the zone lies
    # where nothing shows; only what shows is rendered.
    content = b"BT /F1 10 Tf 40 285 Td (Patricia Prueba Rojas) Tj ET BT /F1 10 Tf 60 150 Td (Texto neutro) Tj ET"
    src = build(tmp_path / "off.pdf", content, HELV)
    with pymupdf.open(src) as doc:
        doc[0].set_cropbox(pymupdf.Rect(30, 30, 270, 270))
        doc.save(tmp_path / "cropped.pdf")
    with pymupdf.open(tmp_path / "cropped.pdf") as doc:
        text, boxes, _ = pdf.chars(doc[0])  # text outside the crop box too
    start = text.index("Patricia")
    zone = pymupdf.Rect(boxes[start]) | pymupdf.Rect(boxes[start + len("Patricia Prueba Rojas") - 1])
    zone += (-1, -1, 1, 1)
    assert zone.y1 < 0  # above the crop box
    out = tmp_path / "off_out.pdf"
    outcome = pdf.redact(str(tmp_path / "cropped.pdf"), str(out), {0: [zone]})[0]
    assert not outcome.rasterized, outcome.reasons
    with pymupdf.open(out) as doc:
        assert "Patricia" not in pdf.chars(doc[0])[0]


# ---------------------------------------------------------------------------
# Closing re-review: data riding on layout, look-alike boxes, light data (full export)
# ---------------------------------------------------------------------------

MODS = " ".join(f"{100 + 3 * i} {130 + 3 * j} 3 3 re" for i, j in GRID)
THIN = " ".join(f"{100 + 3 * i} {130 + 3 * j} 0.05 3 re" for i, j in GRID)
SOFT = "<</ExtGState<</GS1<</SMask<</S/Luminosity/G {grp} 0 R>>>>>>>>"
CLOSING = {
    # a layout rectangle that neither holds the zone nor lies in it, painting a QR through a clip
    "a1a_clip_modules_fill_qr_box": (f"q {MODS} W n 0 0 0 rg 100 130 36 36 re f Q", "<<>>", None, (104, 138, 150, 166), "dark"),
    "a1b_clip_modules_red_qr_box": (f"q {MODS} W n 1 0 0 rg 100 130 36 36 re f Q", "<<>>", None, (104, 138, 150, 166), "red"),
    # the same rectangle through a soft mask
    "a1d_softmask_rect_partial": ("q /GS1 gs 1 0 0 rg 100 130 40 40 re f Q", SOFT, softmask_group_img, (105, 135, 150, 165), "red"),
    # a stroked rule through the module clip
    "a1e_clip_modules_stroked_rule": (f"q {MODS} W n 0 0 0 RG 3 w 90 148 m 160 148 l S Q", "<<>>", None, (104, 138, 150, 166), "dark"),
    # the document's own box, painted exactly like MuPDF's, through a clip of modules
    "a2b_own_black_fs_box_clip_zin": (f"q {MODS} W n 0 0 0 rg 0 0 0 RG 0 0 300 300 re B Q", "<<>>", None, (102, 138, 132, 166), "dark"),
    # light data: a grey QR, a light soft-mask checkerboard, sub-pixel modules
    "a3a_light_qr_086": (f"q {MODS} W n 0.86 g 0 0 300 300 re f Q", "<<>>", None, (102, 138, 132, 166), "light"),
    "a5a_softmask_light_087": ("q /GS1 gs 0.87 g 90 120 120 60 re f Q", SOFT, softmask_group_img, (110, 140, 130, 160), "light"),
    "a5b_thin_modules_005": (f"q {THIN} W n 0 g 0 0 300 300 re f Q", "<<>>", None, (102, 138, 132, 166), "light"),
}  # fmt: skip


def _secret_pixels(path: Path, zone, colour: str, zoom: float = 16) -> int:
    with pymupdf.open(path) as doc:
        pix = doc[0].get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(zone) + (1, 1, -1, -1), alpha=False
        )
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
    if colour == "red":
        return int(((a[:, :, 0] > 150) & (a[:, :, 1] < 120)).sum())
    if colour == "dark":
        return int((a.max(axis=2) < 100).sum())
    return int((a.min(axis=2) < 250).sum())


@pytest.mark.parametrize("name", list(CLOSING))
def test_no_data_rides_on_layout_or_hides_in_light_colours(tmp_path, name):
    content, resources, setup, zone, colour = CLOSING[name]
    src = build(tmp_path / f"{name}.pdf", content.encode(), resources, setup)
    assert _secret_pixels(src, zone, colour) > 0
    file = AnalyzedFile(id="c", name=src.name, path=str(src), kind="pdf", findings=[_manual(*zone)], status="confirmed")
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    out = Path(result.output_path)
    if result.rasterized_pages:  # one image: nothing but its pixels, black in the zone
        with pymupdf.open(out) as doc:
            page = doc[0]
            assert not page.get_drawings() and not page.get_text().strip() and len(page.get_images()) == 1
        inner = pymupdf.Rect(zone) + (1, 1, -1, -1)
        with pymupdf.open(out) as doc:
            assert max(doc[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=inner, alpha=False).samples) < 60
        return
    stripped = tmp_path / "stripped.pdf"
    with pymupdf.open(out) as doc:
        for x in doc[0].get_contents():
            doc.update_stream(x, _BOX.sub(b"", doc.xref_stream(x)))
        doc.save(stripped)
    assert _secret_pixels(stripped, zone, colour) == 0, name
