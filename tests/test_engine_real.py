"""The real engine against the engine contract (invented data only)."""

from __future__ import annotations

import importlib.util
import io
import threading
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from anonymizer.engine import common, faces, get_engine, verify
from anonymizer.engine.fake import FakeEngine
from anonymizer.engine.model import AnalyzedFile, Finding
from anonymizer.engine.patterns import rut_check_digit
from anonymizer.engine.real import RealEngine
from anonymizer.engine.text import DOUBT_CONTEXT_NAME, DOUBT_OCR, DOUBT_RUT
from test_bench.canvas import font

needs_ocr = pytest.mark.skipif(
    importlib.util.find_spec("rapidocr") is None or not faces.available(), reason="needs the OCR and face models"
)

VALID_RUT = f"12.345.678-{rut_check_digit('12345678')}"
BAD_RUT = "11.111.111-2"
EMAIL = "ana.prueba@ejemplo.cl"
PHONE = "+56 9 8123 4567"
SIGNER = "Ana Luisa Soto"  # starts with a known given name: found by the dictionary


def make_pdf(path: Path, rotation: int = 0, extra: list[str] | None = None) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    lines = [
        "Informe ficticio de prueba para el motor de anonimización, con texto neutro de relleno.",
        f"RUT: {VALID_RUT}",
        f"Correo: {EMAIL}",
        f"Teléfono: {PHONE}",
        f"Segundo RUT {BAD_RUT} de la contraparte",
        SIGNER,
        *(extra or []),
    ]
    for i, line in enumerate(lines):
        page.insert_text((72, 100 + 22 * i), line, fontsize=11)
    page.set_rotation(rotation)
    doc.set_metadata({"author": "Persona Inventada", "title": "Documento ficticio"})
    doc.save(path)
    doc.close()
    return path


def analyzed(path: Path, names: list[str] | None = None, engine: RealEngine | None = None, **kw) -> AnalyzedFile:
    engine = engine or RealEngine()
    file = AnalyzedFile(id="f1", name=path.name, path=str(path), **kw)
    engine.analyze(file, names or [])
    return file


def output_text(path: str) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_findings_are_in_view_space(tmp_path, rotation):
    path = make_pdf(tmp_path / "a.pdf", rotation)
    file = analyzed(path)
    assert file.status == "ready" and file.kind == "pdf"
    page = file.pages[0]
    assert (page.width, page.height) == ((842, 595) if rotation in (90, 270) else (595, 842))
    rut = next(f for f in file.findings if f.type == "rut" and f.text == VALID_RUT)
    with pymupdf.open(path) as doc:
        expected = doc[0].search_for(VALID_RUT)[0] * doc[0].rotation_matrix
    x0, y0, x1, y1 = common.bbox_of(rut.polygon)
    assert abs((x0 + x1) / 2 - (expected.x0 + expected.x1) / 2) < 3
    assert abs((y0 + y1) / 2 - (expected.y0 + expected.y1) / 2) < 3
    assert 0 <= x0 < x1 <= page.width and 0 <= y0 < y1 <= page.height


def test_types_detectors_and_doubts(tmp_path):
    file = analyzed(make_pdf(tmp_path / "a.pdf"))
    by_text = {f.text: f for f in file.findings}
    assert by_text[VALID_RUT].detector == "regex" and not by_text[VALID_RUT].doubtful
    assert by_text[BAD_RUT].doubtful and by_text[BAD_RUT].doubt_reason == DOUBT_RUT
    assert by_text[EMAIL].type == "email" and by_text[PHONE].type == "phone"
    signer = by_text[SIGNER]
    assert signer.type == "name" and signer.detector == "context"
    assert signer.doubtful and signer.doubt_reason == DOUBT_CONTEXT_NAME
    assert all(f.history and f.history[0].action == "proposed" for f in file.findings)


def test_name_on_the_list_is_not_doubtful(tmp_path):
    file = analyzed(make_pdf(tmp_path / "a.pdf"), [SIGNER])
    signer = [f for f in file.findings if f.text == SIGNER]
    assert signer and not any(f.doubtful for f in signer)
    assert {f.detector for f in signer} == {"name_list"}


def test_same_name_elsewhere_on_the_page_is_redacted_too(tmp_path):
    # In a long line the dictionary rule does not apply: the name is found because it was found above.
    long_line = f"En la reunión del comité de evaluación del proyecto participó {SIGNER} como representante."
    file = analyzed(make_pdf(tmp_path / "a.pdf", extra=[long_line]))
    assert len([f for f in file.findings if f.text == SIGNER]) == 2
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    assert "Soto" not in output_text(result.output_path)


def test_progress_and_cancel(tmp_path):
    path = make_pdf(tmp_path / "a.pdf")
    steps: list[tuple[float, str]] = []
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(path))
    RealEngine().analyze(file, [], progress=lambda fraction, step: steps.append((fraction, step)))
    assert "Leyendo el texto de la página 1 de 1" in [s for _, s in steps]
    assert [f for f, _ in steps] == sorted(f for f, _ in steps)
    cancel = threading.Event()
    cancel.set()
    file = AnalyzedFile(id="f2", name="a.pdf", path=str(path))
    RealEngine().analyze(file, [], cancel=cancel)
    assert file.status == "cancelled" and not file.findings


def test_errors_by_content(tmp_path):
    (tmp_path / "empty.pdf").write_bytes(b"")
    (tmp_path / "doc.pdf").write_bytes(b"PK\x03\x04 not a pdf")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.7\n garbage garbage")
    (tmp_path / "bad.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Contenido ficticio")
    doc.save(tmp_path / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="clave", owner_pw="clave")
    doc.close()
    (tmp_path / "foto.heic").write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64)
    expected = {"empty.pdf": "empty", "doc.pdf": "format", "bad.pdf": "corrupt", "bad.jpg": "corrupt"}
    expected["locked.pdf"] = "password"
    expected["foto.heic"] = "heic"  # D5: not supported, with its own message
    # A generic HEIF brand counts only with a HEIC brand among the compatible ones; an AVIF does not.
    (tmp_path / "foto.heif").write_bytes(b"\x00\x00\x00\x1cftypmif1\x00\x00\x00\x00mif1heicmiaf" + b"\x00" * 64)
    (tmp_path / "foto.avif").write_bytes(b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf" + b"\x00" * 64)
    (tmp_path / "otra.avif").write_bytes(b"\x00\x00\x00\x1cftypmif1\x00\x00\x00\x00mif1avifmiaf" + b"\x00" * 64)
    expected |= {"foto.heif": "heic", "foto.avif": "format", "otra.avif": "format"}
    for name, code in expected.items():
        file = analyzed(tmp_path / name)
        assert (file.status, file.error) == ("error", code), name
        assert file.error_message


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", [0, 90])
def test_export_applies_active_findings_only(tmp_path, rotation):
    file = analyzed(make_pdf(tmp_path / "a.pdf", rotation))
    kept = next(f for f in file.findings if f.text == EMAIL)
    kept.status = "removed"
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    assert result.removed_by_reviewer == 1
    text = output_text(result.output_path)
    assert EMAIL in text  # the reviewer kept it: not a leak
    assert VALID_RUT not in text and "8123" not in text and BAD_RUT not in text
    assert "Informe ficticio" in text
    with pymupdf.open(result.output_path) as doc:
        assert not any(v for k, v in doc.metadata.items() if k not in ("format", "encryption"))
        assert doc[0].rotation == rotation


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_manual_finding_drawn_in_view_space(tmp_path, rotation):
    path = make_pdf(tmp_path / "a.pdf", rotation)
    file = analyzed(path)
    for f in file.findings:  # the reviewer keeps every proposal visible and draws one zone
        f.status = "removed"
    with pymupdf.open(path) as doc:
        target = doc[0].search_for("Informe ficticio")[0] * doc[0].rotation_matrix
    target.normalize()
    polygon = common.rect_polygon(target.x0 - 2, target.y0 - 2, target.x1 + 2, target.y1 + 2)
    file.findings.append(
        Finding(id="m1", file_id="f1", page=0, type="manual", polygon=polygon, detector="reviewer", status="added")
    )
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert result.exported
    text = output_text(result.output_path)
    assert "Informe ficticio" not in text and VALID_RUT in text


def test_leak_blocks_export_and_never_overwrites(tmp_path):
    engine = RealEngine()
    file = analyzed(make_pdf(tmp_path / "a.pdf"), engine=engine)
    out = tmp_path / "out"
    first = engine.export(file, str(out))
    second = engine.export(file, str(out))
    assert Path(first.output_path).name == "a.pdf" and Path(second.output_path).name == "a (2).pdf"
    rut = next(f for f in file.findings if f.text == VALID_RUT)
    rut.polygon = [[0, 0], [5, 0], [5, 5], [0, 5]]  # its zone no longer covers the text
    blocked = engine.export(file, str(tmp_path / "blocked"))
    assert not blocked.exported and blocked.output_path is None
    assert any("RUT sigue legible en la página 1" in leak.message for leak in blocked.leaks)
    assert not any((tmp_path / "blocked").iterdir())


def test_unmarked_pattern_in_the_output_is_a_leak(tmp_path):
    file = analyzed(make_pdf(tmp_path / "a.pdf"))
    file.findings = [f for f in file.findings if f.type != "phone"]  # as if the phone was never proposed
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert not result.exported
    assert any(leak.type == "phone" and "no estaba marcado" in leak.message for leak in result.leaks)


NEUTRAL_SCAN = "Acta ficticia escaneada con capa de texto invisible, solo para pruebas."


def _image_page_pdf(path: Path, img: Image.Image, invisible: list[str] | None = None) -> Path:
    """A PDF page that is one image (a scan); ``invisible`` adds an OCR layer on top (render mode 3)."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.getvalue())
    for i, line in enumerate(invisible or []):
        page.insert_text((72, 100 + 22 * i), line, fontsize=11, render_mode=3)
    doc.save(path)
    doc.close()
    return path


@needs_ocr
def test_scan_is_exported_as_an_image_without_a_text_layer(tmp_path):
    # D9: a scanned page goes out as the redacted image it came in as; no OCR text layer is added.
    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate([f"RUT: {VALID_RUT}", f"Correo: {EMAIL}", "Texto neutro de la nota"]):
        draw.text((120, 160 + 70 * i), line, font=font("sans", 36), fill=(0, 0, 0))
    engine = RealEngine()
    file = analyzed(_image_page_pdf(tmp_path / "escaneo.pdf", img), engine=engine)
    assert file.pages[0].scanned and {"rut", "email"} <= {f.type for f in file.findings}
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        assert doc[0].get_text().strip() == ""
        assert not doc[0].get_texttrace()
        assert len(doc[0].get_images()) == 1


def test_sandwich_ocr_layer_keeps_only_the_original_text_that_was_not_redacted(tmp_path):
    # D9: in a scan with an invisible OCR layer, the redacted spans leave that layer too; what is
    # left is the original's own text, still invisible, and nothing new is written.
    lines = [NEUTRAL_SCAN, f"RUT: {VALID_RUT}", f"Correo: {EMAIL}"]
    path = _image_page_pdf(tmp_path / "sandwich.pdf", Image.new("RGB", (850, 1100), "white"), lines)
    options = {"ocr": False, "faces": False, "qr": False}  # only the text layer matters here
    engine = RealEngine()
    file = analyzed(path, engine=engine, options=options)
    assert {"rut", "email"} <= {f.type for f in file.findings}
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        spans = doc[0].get_texttrace()
        text = doc[0].get_text()
    assert VALID_RUT not in text and EMAIL not in text
    assert NEUTRAL_SCAN in text and "RUT:" in text
    assert spans and {span["type"] for span in spans} == {3}  # still invisible


def test_image_metadata_check(tmp_path):
    img = Image.new("RGB", (40, 30), "white")
    exif = Image.Exif()
    exif[315] = "Persona Inventada"
    img.save(tmp_path / "with.jpg", exif=exif)
    img.save(tmp_path / "clean.png")
    assert verify.image_leaks(tmp_path / "with.jpg")
    assert not verify.image_leaks(tmp_path / "clean.png")


# ---------------------------------------------------------------------------
# images (OCR and faces)
# ---------------------------------------------------------------------------


def _text_image(lines: list[str], size=(760, 760)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((30, 30 + 60 * i), line, font=font("sans", 32), fill=(0, 0, 0))
    return img


@needs_ocr
def test_image_with_exif_orientation(tmp_path):
    upright = _text_image([f"Correo: {EMAIL}", "Texto neutro de la nota"])
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated: shown rotated 90° clockwise back to upright
    upright.transpose(Image.Transpose.ROTATE_90).save(tmp_path / "nota.jpg", exif=exif, quality=95)
    engine = RealEngine()
    file = analyzed(tmp_path / "nota.jpg", engine=engine)
    assert file.status == "ready" and file.kind == "image"
    assert (file.pages[0].width, file.pages[0].height) == upright.size  # view space: after EXIF
    email = next(f for f in file.findings if f.type == "email")
    assert email.detector == "ocr" and email.score is not None
    x0, y0, x1, y1 = common.bbox_of(email.polygon)
    assert y1 < 120 and x0 < 100  # first line, at the top left of the upright image
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported
    with Image.open(result.output_path) as out:
        assert out.size == upright.size and not len(out.getexif())
        assert out.convert("L").getpixel((int((x0 + x1) / 2), int((y0 + y1) / 2))) < 40


@needs_ocr
def test_all_text_mode_turns_every_line_into_a_finding(tmp_path):
    _text_image([f"Correo: {EMAIL}", "Texto neutro de la nota"]).save(tmp_path / "nota.png")
    normal = analyzed(tmp_path / "nota.png")
    every = analyzed(tmp_path / "nota.png", all_text=True)
    assert not any(f.type == "text" for f in normal.findings)
    assert any(f.type == "text" and "neutro" in (f.text or "") for f in every.findings)


@needs_ocr
def test_low_confidence_ocr_is_doubtful():
    from anonymizer.engine import raster

    assert raster._line_doubt("email", EMAIL, 0.5, "ocr", ()) == DOUBT_OCR
    assert raster._line_doubt("email", EMAIL, 0.9, "ocr", ()) is None


def test_small_faces_are_doubtful():
    import numpy as np

    box = np.array([[0, 0], [30, 0], [30, 30], [0, 30]], float)
    big = np.array([[0, 0], [90, 0], [90, 90], [0, 90]], float)
    small = faces.merge([("face", box, "", "faces", 0.95)])
    assert small[0].doubt is not None
    assert faces.merge([("face", big, "", "faces", 0.95)])[0].doubt is None
    assert faces.merge([("face", big, "", "faces", 0.6)])[0].doubt is not None


# ---------------------------------------------------------------------------
# render and engine selection
# ---------------------------------------------------------------------------


def test_render_page_in_view_space(tmp_path):
    file = analyzed(make_pdf(tmp_path / "a.pdf", 90))
    png = RealEngine().render_page(file, 0, 0.5)
    with Image.open(io.BytesIO(png)) as img:
        assert img.size == (421, 298)


def test_get_engine_falls_back_when_models_are_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("ANONYMIZER_ENGINE", raising=False)
    monkeypatch.setattr(faces, "YUNET_MODEL", tmp_path / "missing.onnx")
    assert isinstance(get_engine(), FakeEngine)
    monkeypatch.setenv("ANONYMIZER_ENGINE", "fake")
    assert isinstance(get_engine(), FakeEngine)


@needs_ocr
def test_get_engine_defaults_to_real(monkeypatch):
    monkeypatch.delenv("ANONYMIZER_ENGINE", raising=False)
    assert isinstance(get_engine(), RealEngine)


def test_api_flow_with_the_real_engine(tmp_path):
    from anonymizer.api.server import create_app
    from tests.live_client import LiveClient
    from tests.test_api import TOKEN, upload, wait_status

    app = create_app(RealEngine(), TOKEN)
    with LiveClient(app) as client:
        client.headers["X-Session-Token"] = TOKEN
        file_id = upload(client, "informe.pdf", make_pdf(tmp_path / "a.pdf", 90).read_bytes())
        assert client.get("/api/state").json()["engine"] == "real"
        client.post("/api/process", json={"file_ids": [file_id]})
        assert wait_status(client, file_id)["status"] == "ready"
        file = client.get(f"/api/files/{file_id}").json()
        assert any(f["doubtful"] for f in file["findings"])
        signer = next(f for f in file["findings"] if f["text"] == SIGNER)
        client.patch(f"/api/files/{file_id}/findings/{signer['id']}", json={"action": "remove"})
        assert client.post(f"/api/files/{file_id}/confirm").status_code == 200
        dest = tmp_path / "salida"
        r = client.post("/api/export", json={"dest_dir": str(dest), "audit_pdf": False, "audit_json": True})
        assert r.status_code == 200, r.text
        exported = dest / "informe.pdf"
        text = output_text(str(exported))
        assert SIGNER in text and VALID_RUT not in text and EMAIL not in text


def test_tilted_line_read_in_several_orientations_is_one_finding():
    import numpy as np

    from anonymizer.engine import raster
    from anonymizer.engine.common import Zone

    # The same tilted line (a photo, a crooked scan) as the OCR passes at 0°, 90° and 270° return it:
    # nearly the same quadrilateral, starting at a different corner.
    base = np.array([[100.0, 100.0], [500.0, 130.0], [497.0, 170.0], [97.0, 140.0]])
    readings = [base, np.roll(base + 1.5, 1, axis=0), np.roll(base - 1.0, 2, axis=0)]
    zones = [
        Zone("rut", pol, "RUT 1", "ocr", 0.9 - 0.1 * i, None if i else "Lectura dudosa")
        for i, pol in enumerate(readings)
    ]
    zones.append(Zone("rut", base + [0, 200], "RUT 2", "ocr", 0.9))  # another line
    zones.append(Zone("face", base, "", "faces", 0.9))  # another detector: never joined
    out = raster.dedup(zones)
    assert [z.type for z in out] == ["rut", "rut", "face"]
    joined = np.asarray(out[0].polygon, np.float64)
    for pol in readings:  # covers every reading
        for x, y in pol:
            assert cv2_inside(joined, x, y)
    assert out[0].doubt is None and out[0].score == pytest.approx(0.9)


def cv2_inside(polygon, x: float, y: float) -> bool:
    import cv2
    import numpy as np

    return cv2.pointPolygonTest(np.asarray(polygon, np.float32), (float(x), float(y)), True) >= -0.01


def test_power_throttling_is_turned_off_on_windows():
    import sys

    assert common.disable_power_throttling() is (sys.platform == "win32")


# ---------------------------------------------------------------------------
# review findings: geometry, leak check, no network, safe output
# ---------------------------------------------------------------------------


def _boxed_pdf(path: Path, rotation: int, box: str) -> Path:
    """A text page whose CropBox or MediaBox does not start at (0, 0), rotated."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=600)
    if box == "mediabox":
        doc.xref_set_key(page.xref, "MediaBox", "[100 200 500 800]")
        page = doc.reload_page(page)
    else:
        page.set_cropbox(pymupdf.Rect(20, 30, 380, 580))
    page.insert_text((60, 300), f"RUT: {VALID_RUT}", fontsize=12)
    page.insert_text((60, 500), "Texto neutro de relleno para la prueba del motor, sin datos.", fontsize=10)
    page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    return path


def _gray(path: str, page: int = 0):
    import numpy as np

    with pymupdf.open(path) as doc:
        pix = doc[page].get_pixmap(alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].mean(axis=2)


@pytest.mark.parametrize("box", ["cropbox", "mediabox"])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_zone_drawn_in_view_space_lands_there_on_rotated_cropped_pages(tmp_path, rotation, box):
    import hashlib

    path = _boxed_pdf(tmp_path / "a.pdf", rotation, box)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    engine = RealEngine()
    file = analyzed(path, engine=engine)
    w, h = file.pages[0].width, file.pages[0].height
    assert any(f.type == "rut" for f in file.findings)
    x0, y0, x1, y1 = w * 0.1, h * 0.05, w * 0.4, h * 0.15  # blank paper, top left as the user sees it
    file.findings.append(
        Finding(id="m1", file_id="f1", page=0, type="manual", polygon=common.rect_polygon(x0, y0, x1, y1),
                detector="reviewer", status="added")
    )  # fmt: skip
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    gray = _gray(result.output_path)
    assert gray.shape == (round(h), round(w))
    assert gray[int(y0) + 2 : int(y1) - 2, int(x0) + 2 : int(x1) - 2].mean() < 10  # black where it was drawn
    assert gray[int(h * 0.85) : int(h * 0.95), int(w * 0.6) : int(w * 0.9)].mean() > 245  # nothing elsewhere
    assert VALID_RUT not in output_text(result.output_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before  # the working copy is never modified


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_zone_drawn_on_an_image_with_exif_orientation(tmp_path, orientation):
    import numpy as np

    transpose = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270}
    upright = Image.new("RGB", (300, 200), "white")
    upright.paste((255, 0, 0), (200, 120, 280, 180))  # a red mark at the bottom right, as the user sees it
    exif = Image.Exif()
    exif[0x0112] = orientation
    upright.transpose(transpose[orientation]).save(tmp_path / "foto.png", exif=exif.tobytes())
    file = AnalyzedFile(id="f1", name="foto.png", path=str(tmp_path / "foto.png"), kind="image")
    with Image.open(io.BytesIO(RealEngine().render_page(file, 0))) as view:
        assert view.size == (300, 200) and view.convert("RGB").getpixel((240, 150)) == (255, 0, 0)
    file.findings = [
        Finding(id="m1", file_id="f1", page=0, type="manual", polygon=common.rect_polygon(20, 20, 120, 80),
                detector="reviewer", status="added"),
        Finding(id="k1", file_id="f1", page=0, type="face", polygon=common.rect_polygon(200, 120, 280, 180),
                detector="faces", status="removed"),
    ]  # fmt: skip
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert result.exported
    with Image.open(result.output_path) as out:
        pixels = np.array(out.convert("RGB"))
    assert pixels.shape[:2] == (200, 300)
    assert pixels[22:78, 22:118].max() == 0  # the drawn zone, in the same place
    assert tuple(pixels[150, 240]) == (255, 0, 0)  # what the reviewer kept is untouched
    assert pixels[150, 60].min() == 255


def _skip_zones(monkeypatch, skip):
    """Makes the PDF redaction skip the zones of the findings in ``skip`` (as a faulty apply would)."""
    from anonymizer.engine import pdf

    original = pdf.redact
    boxes = [common.bbox_of(f.polygon) for f in skip]

    def faulty(source, dest, rects_by_page, keep_by_page=None, drawn_by_page=None):
        kept = {
            n: [r for r in rects if not any(abs(r.x0 - b[0]) < 0.01 and abs(r.y0 - b[1]) < 0.01 for b in boxes)]
            for n, rects in rects_by_page.items()
        }
        return original(source, dest, kept, keep_by_page, drawn_by_page)

    monkeypatch.setattr(pdf, "redact", faulty)


def test_leak_check_when_the_same_text_was_kept_in_another_place(tmp_path, monkeypatch):
    path = make_pdf(tmp_path / "a.pdf", extra=[f"Repetido al final: {VALID_RUT}"])
    engine = RealEngine()
    file = analyzed(path, engine=engine)
    first, second = sorted((f for f in file.findings if f.text == VALID_RUT), key=lambda f: f.polygon[0][1])
    first.status = "removed"  # the reviewer keeps the first one visible
    clean = engine.export(file, str(tmp_path / "clean"))
    assert clean.exported, [leak.message for leak in clean.leaks]  # the kept one is not a leak
    assert output_text(clean.output_path).count(VALID_RUT) == 1
    _skip_zones(monkeypatch, [second])  # the second one's zone is not applied
    blocked = engine.export(file, str(tmp_path / "blocked"))
    assert not blocked.exported
    assert any(leak.finding_id == second.id for leak in blocked.leaks)
    assert not any((tmp_path / "blocked").iterdir())


def test_zone_without_text_that_was_not_applied_blocks_the_export(tmp_path, monkeypatch):
    path = make_pdf(tmp_path / "a.pdf")
    engine = RealEngine()
    file = analyzed(path, engine=engine)
    drawn = Finding(id="m1", file_id="f1", page=0, type="manual", polygon=common.rect_polygon(300, 500, 400, 560),
                    detector="reviewer", status="added")  # fmt: skip
    file.findings.append(drawn)
    _skip_zones(monkeypatch, [drawn])
    result = engine.export(file, str(tmp_path / "out"))
    assert not result.exported
    leak = next(leak for leak in result.leaks if leak.finding_id == "m1")
    assert leak.message == "Una zona marcada en la página 1 (Agregada) no quedó tapada por completo."


def test_image_zone_that_was_not_applied_blocks_the_export(tmp_path, monkeypatch):
    from anonymizer.engine import image

    Image.new("RGB", (200, 100), "white").save(tmp_path / "a.png")
    file = AnalyzedFile(id="f1", name="a.png", path=str(tmp_path / "a.png"), kind="image")
    file.findings = [Finding(id="q1", file_id="f1", page=0, type="qr", polygon=common.rect_polygon(10, 10, 60, 60),
                             detector="qr", status="proposed")]  # fmt: skip
    original = image.redact
    monkeypatch.setattr(image, "redact", lambda source, polygons, folder: original(source, {}, folder))
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert not result.exported and result.leaks[0].finding_id == "q1"
    assert not any((tmp_path / "out").iterdir())


def test_output_is_never_left_half_written(tmp_path, monkeypatch):
    staged = tmp_path / "staged.pdf"
    staged.write_bytes(b"%PDF-1.7 contenido")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "a.pdf").write_bytes(b"anterior")
    assert common.publish(staged, dest, "a.pdf") == dest / "a (2).pdf"
    assert (dest / "a.pdf").read_bytes() == b"anterior"

    def broken_copy(source, target):
        Path(target).write_bytes(b"%PDF-1.7 con")
        raise OSError("disco lleno")

    monkeypatch.setattr(common.shutil, "copyfile", broken_copy)
    with pytest.raises(OSError):
        common.publish(staged, dest, "b.pdf")
    assert sorted(p.name for p in dest.iterdir()) == ["a (2).pdf", "a.pdf"]


def test_ocr_models_are_local(monkeypatch):
    from anonymizer.engine import ocr

    paths = ocr.model_paths()
    if paths is None:
        pytest.skip("rapidocr is not installed")
    assert set(paths) == {"Det", "Cls", "Rec"}
    monkeypatch.setitem(ocr.MODEL_FILES, "Rec", "missing.onnx")
    assert not ocr.available()  # a missing model is reported, never downloaded


@needs_ocr
def test_ocr_starts_without_network(monkeypatch):
    import socket

    import numpy as np

    from anonymizer.engine import ocr

    attempts = []

    def deny(*args, **kwargs):
        attempts.append(args)
        raise OSError("sin red en la prueba")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    reader = ocr.engine.__wrapped__()  # a new engine, not the cached one
    img = np.array(_text_image([EMAIL], size=(700, 120)))[:, :, ::-1].copy()
    assert any(EMAIL in txt for txt in reader(img).txts or ())
    assert not attempts


# ---------------------------------------------------------------------------
# after: image helpers (spec 4.1)
# ---------------------------------------------------------------------------


def _tiff(path: Path, sizes) -> Path:
    frames = []
    for i, (w, h) in enumerate(sizes):
        img = Image.new("RGB", (w, h), (255, 255, 255))
        ImageDraw.Draw(img).rectangle((10, 10, 60 + 10 * i, 40), fill=(200, 30 * i, 0))
        frames.append(img)
    frames[0].save(path, "TIFF", save_all=True, append_images=frames[1:], compression="tiff_deflate")
    return path


def test_image_frame_equals_the_frames_generator(tmp_path):
    import numpy as np

    from anonymizer.engine import image

    path = _tiff(tmp_path / "multi.tif", [(300, 200), (240, 320), (500, 260)])
    every = list(image.frames(str(path)))
    for n in range(3):
        assert np.array_equal(np.array(image.frame(str(path), n)), np.array(every[n]))
    with pytest.raises(IndexError):
        image.frame(str(path), 3)


def test_image_frame_applies_exif_orientation(tmp_path):
    from anonymizer.engine import image

    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (300, 200), "white").save(tmp_path / "o6.png", exif=exif.tobytes())
    assert image.frame(str(tmp_path / "o6.png"), 0).size == (200, 300)


def test_redact_frame_fills_the_grown_polygon_only():
    import numpy as np

    from anonymizer.engine import image

    arr = np.full((100, 200, 3), 255, np.uint8)
    out = image.redact_frame(arr, [[[50, 40], [150, 40], [150, 60], [50, 60]]])
    assert out is arr
    assert arr[45:56, 55:146].max() == 0
    assert arr[5:20, 5:40].min() == 255


# ---------------------------------------------------------------------------
# after: PDF helpers (spec 4.1)
# ---------------------------------------------------------------------------


def test_redaction_rects_move_view_boxes_to_the_unrotated_page(tmp_path):
    from anonymizer.engine import pdf
    from anonymizer.engine.locks import PDF_LOCK

    path = _boxed_pdf(tmp_path / "r.pdf", 90, "cropbox")
    with PDF_LOCK, pymupdf.open(path) as doc:
        rects = pdf.redaction_rects(doc, {0: [common.rect_polygon(10, 20, 110, 60)], 5: [common.rect_polygon(0, 0, 1, 1)]})
        assert list(rects) == [0]
        expected = (pymupdf.Rect(10, 20, 110, 60) * pymupdf.Matrix(doc[0].derotation_matrix)).normalize()
        assert rects[0] == [expected]


def test_redact_page_removes_text_and_annotations_of_that_page_only(tmp_path):
    from anonymizer.engine import pdf
    from anonymizer.engine.locks import PDF_LOCK

    path = make_pdf(tmp_path / "a.pdf")
    with pymupdf.open(path) as doc:
        doc[0].add_freetext_annot(pymupdf.Rect(300, 600, 500, 640), "Nota de Persona Inventada")
        doc.save(tmp_path / "b.pdf")
    with PDF_LOCK, pymupdf.open(tmp_path / "b.pdf") as doc:
        hit = doc[0].search_for(EMAIL)[0]
        pdf.redact_page(doc, 0, [hit])
        assert EMAIL not in doc[0].get_text()
        assert VALID_RUT in doc[0].get_text()
        assert not list(doc[0].annots() or [])


# ---------------------------------------------------------------------------
# after: parity with the export (spec 9.1)
# ---------------------------------------------------------------------------


def _pixels(png: bytes):
    import numpy as np

    with Image.open(io.BytesIO(png)) as img:
        return np.array(img.convert("RGB"))


def _exported_render(engine, file: AnalyzedFile, page: int, zoom: float, tmp_path: Path):
    result = engine.export(file, str(tmp_path / f"out{page}{zoom}"))
    assert result.exported, [leak.message for leak in result.leaks]
    out = AnalyzedFile(id="o", name=Path(result.output_path).name, path=result.output_path, kind=file.kind)
    return _pixels(engine.render_page(out, page, zoom))


def _manual(page: int, x0, y0, x1, y1, fid="m", status="added") -> Finding:
    return Finding(id=f"{fid}{page}", file_id="f1", page=page, type="manual",
                   polygon=common.rect_polygon(x0, y0, x1, y1), detector="reviewer", status=status)  # fmt: skip


def _assert_parity(engine, file, page, zoom, tmp_path, lossy=False):
    import numpy as np

    after = _pixels(engine.render_result(file, page, zoom, list(file.findings)))
    exported = _exported_render(engine, file, page, zoom, tmp_path)
    assert after.shape == exported.shape
    before = _pixels(engine.render_page(file, page, zoom))  # the before column: same size, rows align
    assert before.shape[:2] == after.shape[:2]
    if lossy:
        assert np.abs(after.astype(int) - exported.astype(int)).mean() < 2
    else:
        assert np.array_equal(after, exported)


def _file(path: Path, kind: str, findings) -> AnalyzedFile:
    return AnalyzedFile(id="f1", name=path.name, path=str(path), kind=kind, status="ready", findings=findings)


def test_after_equals_export_text_page(tmp_path):
    file = analyzed(make_pdf(tmp_path / "a.pdf"))
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)


def test_after_equals_export_page_with_annotation_and_no_findings(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        page.insert_text((40, 60), "Texto neutro de relleno.", fontsize=11)
        page.add_freetext_annot(pymupdf.Rect(40, 100, 300, 140), "Nota de Persona Inventada")
        doc.save(tmp_path / "n.pdf")
    _assert_parity(RealEngine(), _file(tmp_path / "n.pdf", "pdf", []), 0, 1.0, tmp_path)


@pytest.mark.parametrize("zoom", [1.0, 1.9])
def test_after_equals_export_scanned_page(tmp_path, zoom):
    scan = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(scan)
    draw.text((100, 200), f"RUT {VALID_RUT}", fill=(0, 0, 0), font=font("sans", 40))
    buf = io.BytesIO()
    scan.save(buf, "JPEG", quality=85)
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=buf.getvalue())
        doc.save(tmp_path / "s.pdf")
    file = _file(tmp_path / "s.pdf", "pdf", [_manual(0, 40, 85, 320, 125)])
    _assert_parity(RealEngine(), file, 0, zoom, tmp_path)


def test_after_equals_export_rotated_cropped_page(tmp_path):
    path = _boxed_pdf(tmp_path / "b.pdf", 90, "cropbox")
    file = analyzed(path)
    file.findings.append(_manual(0, 30, 30, 200, 80))
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)


def test_after_hides_an_annotation_with_a_name(tmp_path):
    path = make_pdf(tmp_path / "a.pdf")
    with pymupdf.open(path) as doc:
        doc[0].add_freetext_annot(pymupdf.Rect(300, 600, 560, 640), "Persona Inventada Rojas")
        doc.save(tmp_path / "an.pdf")
    file = analyzed(tmp_path / "an.pdf")
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)


def _shared_resource_pdf(path: Path, kind: str) -> Path:
    """Two pages that draw the same Form XObject (letterhead) or the same image XObject."""
    with pymupdf.open() as src:
        head = src.new_page(width=400, height=80)
        head.draw_rect(pymupdf.Rect(0, 0, 400, 80), color=(0, 0, 0), fill=(0.8, 0.8, 0.9))
        head.insert_text((20, 40), "Membrete institucional de prueba", fontsize=12)
        src.save(path.with_suffix(".head.pdf"))
    img = Image.new("RGB", (400, 80), "white")
    ImageDraw.Draw(img).text((20, 30), f"RUT {VALID_RUT}", fill=(0, 0, 0), font=font("sans", 20))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    with pymupdf.open() as doc, pymupdf.open(path.with_suffix(".head.pdf")) as head_doc:
        for _ in range(2):
            page = doc.new_page(width=400, height=300)
            if kind == "form":
                page.show_pdf_page(pymupdf.Rect(0, 0, 400, 80), head_doc, 0)
            else:
                page.insert_image(pymupdf.Rect(0, 0, 400, 80), stream=buf.getvalue())
        doc.save(path)
    return path


@pytest.mark.parametrize("kind", ["form", "image"])
def test_after_equals_export_with_a_resource_shared_by_two_pages(tmp_path, kind):
    path = _shared_resource_pdf(tmp_path / "shared.pdf", kind)
    engine = RealEngine()
    file = _file(path, "pdf", [_manual(1, 10, 10, 200, 70)])
    with pymupdf.open(path) as doc:  # guard: both pages really share the resource
        if kind == "image":
            shared = {img[0] for img in doc[0].get_images()} & {img[0] for img in doc[1].get_images()}
        else:
            shared = {x[0] for x in doc[0].get_xobjects()} & {x[0] for x in doc[1].get_xobjects()}
    assert shared
    for page in (0, 1):
        _assert_parity(engine, file, page, 1.0, tmp_path)
    # guard: the finding does change page 1, and only page 1
    assert not (_pixels(engine.render_result(file, 1, 1.0, file.findings)) == _pixels(engine.render_page(file, 1, 1.0))).all()
    assert (_pixels(engine.render_result(file, 0, 1.0, file.findings)) == _pixels(engine.render_page(file, 0, 1.0))).all()


def test_after_reveals_and_redacts_a_hidden_layer(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        xref = doc.add_ocg("Capa oculta", on=False)
        page.insert_text((40, 60), f"RUT {VALID_RUT}", fontsize=12, oc=xref)
        page.insert_text((40, 200), "Texto neutro visible.", fontsize=12)
        doc.save(tmp_path / "ocg.pdf")
    file = analyzed(tmp_path / "ocg.pdf")
    assert any(f.type == "rut" for f in file.findings)
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)


@pytest.mark.parametrize("mode", ["exif6", "rgba", "palette"])
def test_after_equals_export_png(tmp_path, mode):
    img = Image.new("RGBA" if mode == "rgba" else "RGB", (300, 200), (255, 255, 255, 255) if mode == "rgba" else "white")
    ImageDraw.Draw(img).rectangle((20, 20, 120, 80), fill=(10, 10, 10) if mode != "rgba" else (10, 10, 10, 128))
    kwargs = {}
    if mode == "exif6":
        exif = Image.Exif()
        exif[0x0112] = 6
        kwargs["exif"] = exif.tobytes()
    if mode == "palette":
        img = img.convert("P")
    img.save(tmp_path / "p.png", **kwargs)
    file = _file(tmp_path / "p.png", "image", [_manual(0, 15, 15, 130, 90)])
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)
    _assert_parity(RealEngine(), file, 0, 0.62, tmp_path)


def test_after_equals_export_tiff_page_2(tmp_path):
    path = _tiff(tmp_path / "t.tif", [(300, 200), (240, 320), (500, 260)])
    file = _file(path, "image", [_manual(1, 10, 10, 100, 50), _manual(2, 5, 5, 50, 50)])
    _assert_parity(RealEngine(), file, 1, 1.0, tmp_path)


@pytest.mark.parametrize("fmt,ext", [("JPEG", "jpg"), ("WEBP", "webp")])
def test_after_close_to_export_lossy(tmp_path, fmt, ext):
    img = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(img).text((20, 80), f"RUT {VALID_RUT}", fill=(0, 0, 0), font=font("sans", 18))
    img.save(tmp_path / f"f.{ext}", fmt, quality=92)
    file = _file(tmp_path / f"f.{ext}", "image", [_manual(0, 15, 70, 250, 110)])
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path, lossy=True)


def test_after_ignores_removed_suggested_and_other_pages(tmp_path):
    import numpy as np

    path = _tiff(tmp_path / "t.tif", [(300, 200), (300, 200)])
    engine = RealEngine()
    file = _file(path, "image", [])
    plain = _pixels(engine.render_result(file, 0, 1.0, []))
    kept = [
        _manual(0, 10, 10, 100, 50, fid="r", status="removed"),
        _manual(0, 10, 60, 100, 90, fid="s", status="suggested"),
        _manual(1, 10, 10, 100, 50, fid="o"),
    ]
    assert np.array_equal(_pixels(engine.render_result(file, 0, 1.0, kept)), plain)


def test_redacted_route_equals_export_and_writes_nothing(tmp_path, monkeypatch):
    import hashlib
    import tempfile

    import numpy as np

    from anonymizer.api.server import create_app
    from tests.live_client import LiveClient
    from tests.test_api import TOKEN, upload, wait_status

    app = create_app(RealEngine(), TOKEN)
    with LiveClient(app) as client:
        client.headers["X-Session-Token"] = TOKEN
        file_id = upload(client, "informe.pdf", make_pdf(tmp_path / "a.pdf").read_bytes())
        client.post("/api/process", json={"file_ids": [file_id]})
        assert wait_status(client, file_id)["status"] == "ready"
        file = client.get(f"/api/files/{file_id}").json()
        signer = next(f for f in file["findings"] if f["text"] == SIGNER)
        client.patch(f"/api/files/{file_id}/findings/{signer['id']}", json={"action": "remove", "reason": "Otro motivo"})
        session = app.state.session
        working = Path(session.get(file_id).path)
        digest = hashlib.sha256(working.read_bytes()).hexdigest()
        listing = sorted((p.name, p.stat().st_mtime_ns) for p in Path(session.dir).iterdir())
        empty = tmp_path / "tmp"
        empty.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(empty))
        monkeypatch.setenv("TMP", str(empty))
        monkeypatch.setenv("TEMP", str(empty))
        after = client.get(f"/api/files/{file_id}/pages/0.png?redacted=true").content
        assert list(empty.iterdir()) == []
        assert sorted((p.name, p.stat().st_mtime_ns) for p in Path(session.dir).iterdir()) == listing
        assert hashlib.sha256(working.read_bytes()).hexdigest() == digest
        monkeypatch.undo()
        client.post(f"/api/files/{file_id}/confirm")
        dest = tmp_path / "salida"
        client.post("/api/export", json={"dest_dir": str(dest), "audit_pdf": False, "audit_json": False})
        out = AnalyzedFile(id="o", name="informe.pdf", path=str(dest / "informe.pdf"), kind="pdf")
        assert np.array_equal(_pixels(after), _pixels(RealEngine().render_page(out, 0, 1.0)))


def _disc_pdf(path: Path, rotation: int = 0) -> Path:
    """A text page with a filled disc: a zone over half of it leaves the rest of its outline under
    the zone, so the page is exported as an image (decided 2026-10-06)."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Texto neutro de relleno que sigue visible.", fontsize=11)
        page.draw_circle((200, 400), 30, color=None, fill=(0.2, 0.3, 0.7))
        page.set_cropbox(pymupdf.Rect(10, 20, 585, 830))
        if rotation:
            page.set_rotation(rotation)
        doc.save(path)
    return path


@pytest.mark.parametrize("rotation", [0, 90])
@pytest.mark.parametrize("zoom", [1.0, 1.9])
def test_after_equals_export_page_exported_as_an_image(tmp_path, rotation, zoom):
    path = _disc_pdf(tmp_path / "d.pdf", rotation)
    with pymupdf.open(path) as doc:
        zone = (pymupdf.Rect(150, 360, 200, 440) * doc[0].rotation_matrix).normalize()
    engine = RealEngine()
    file = _file(path, "pdf", [_manual(0, *zone)])
    info: dict = {}
    engine.render_result(file, 0, zoom, list(file.findings), info=info)
    assert info["as_image"] and info["reason"]
    _assert_parity(engine, file, 0, zoom, tmp_path)


def test_redacted_route_says_when_a_page_will_be_an_image(tmp_path):
    from urllib.parse import unquote

    from anonymizer.api.server import create_app
    from anonymizer.engine import leftovers
    from tests.live_client import LiveClient
    from tests.test_api import TOKEN, upload, wait_status

    app = create_app(RealEngine(), TOKEN)
    with LiveClient(app) as client:
        client.headers["X-Session-Token"] = TOKEN
        file_id = upload(client, "disco.pdf", _disc_pdf(tmp_path / "d.pdf").read_bytes())
        client.post("/api/process", json={"file_ids": [file_id]})
        assert wait_status(client, file_id)["status"] == "ready"
        plain = client.get(f"/api/files/{file_id}/pages/0.png?redacted=true")
        assert plain.status_code == 200 and "x-page-as-image" not in plain.headers
        polygon = common.rect_polygon(140, 340, 190, 420)  # over half the disc (view space, cropped page)
        assert client.post(f"/api/files/{file_id}/findings", json={"page": 0, "polygon": polygon}).status_code < 300
        after = client.get(f"/api/files/{file_id}/pages/0.png?redacted=true")
        assert after.headers["x-page-as-image"] == "1"
        assert unquote(after.headers["x-page-as-image-reason"]) == leftovers.REASONS["shape"]
        before = client.get(f"/api/files/{file_id}/pages/0.png")
        assert "x-page-as-image" not in before.headers
