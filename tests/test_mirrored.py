"""D6: mirrored images (a photo taken with a front camera) are read mirrored only when the normal
pass finds no legible text. Invented data only."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from anonymizer.engine import common, faces, ocr
from anonymizer.engine.model import AnalyzedFile
from anonymizer.engine.real import RealEngine
from test_bench.canvas import font

needs_ocr = pytest.mark.skipif(
    importlib.util.find_spec("rapidocr") is None or not faces.available(), reason="needs the OCR and face models"
)

EMAIL = "ana.prueba@ejemplo.cl"


def line(text: str, score: float) -> ocr.OcrLine:
    return ocr.OcrLine(np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]]), text, score, 0)


def test_when_to_read_mirrored():
    assert not ocr.needs_mirror([])  # nothing that looks like text: a plain photo
    assert not ocr.needs_mirror([line("ab", 0.6), line(":3", 0.8)])  # specks, not text
    assert not ocr.needs_mirror([line("Nombre: Ana", 0.99), line("ʇxǝʇ", 0.7)])  # legible text
    assert ocr.needs_mirror([line("DLO2 DE 2OTICILNLE", 0.76), line("g: 8 0e", 0.81)])  # garbage only
    assert ocr.needs_mirror([line("Fecha", 0.88)])  # below the bar for legible


class MarkReader:
    """Stands in for RapidOCR: "reads" the black mark of the image legibly only when it is a wide bar
    in the right half (what the bar of a left-aligned line looks like after mirroring)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        ys, xs = np.nonzero(img[:, :, 0] < 128)
        if not len(xs):
            return SimpleNamespace(boxes=None, txts=None, scores=None)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        legible = (x1 - x0) > (y1 - y0) and x0 > img.shape[1] / 2
        text, score = (f"Correo: {EMAIL}", 0.97) if legible else ("ʇxǝʇ ǝpɐɹɹǝ", 0.7)
        return SimpleNamespace(boxes=[box], txts=[text], scores=[score])


def mark_image(x0: int, x1: int) -> np.ndarray:
    img = np.full((300, 600, 3), 255, np.uint8)
    img[40:70, x0:x1] = 0
    return img


@pytest.mark.parametrize(
    "img,calls,legible",
    [
        (mark_image(20, 260), 6, True),  # mirrored text: read again, mirrored
        (mark_image(340, 580), 3, True),  # legible as it is: no second pass
        (np.full((300, 600, 3), 255, np.uint8), 3, False),  # nothing to read: no second pass
    ],
)
def test_mirrored_pass_only_without_legible_text(monkeypatch, img, calls, legible):
    reader = MarkReader()
    monkeypatch.setattr(ocr, "engine", lambda: reader)
    lines = ocr.read_lines(img)
    assert reader.calls == calls
    read = [ln for ln in lines if ln.score >= 0.9]
    assert bool(read) is legible
    if read:
        # A line read in the mirrored image is mapped back onto the mark of the original.
        (x0, y0), (x1, y1) = read[0].polygon.min(axis=0), read[0].polygon.max(axis=0)
        expected = (20, 40, 260, 70) if calls == 6 else (340, 40, 580, 70)
        assert (round(x0), round(y0), round(x1), round(y1)) == expected


@needs_ocr
def test_photo_taken_with_a_front_camera(tmp_path):
    img = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(["Nota de la reunión", f"Correo: {EMAIL}", "Gracias por asistir"]):
        draw.text((40, 40 + 110 * i), text, font=font("sans", 44), fill=(0, 0, 0))
    upright = img.copy()
    img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(tmp_path / "selfie.png")
    engine = RealEngine()
    file = common_analyze(engine, tmp_path / "selfie.png")
    email = next(f for f in file.findings if f.type == "email")
    x0, y0, x1, y1 = common.bbox_of(email.polygon)
    # In the mirrored photo the line ends near the right edge (it started at x = 40 in the upright one).
    assert x1 > 820 and 130 < (y0 + y1) / 2 < 200
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with Image.open(result.output_path) as out:
        assert out.convert("L").getpixel((int((x0 + x1) / 2), int((y0 + y1) / 2))) < 40
    # The same photo, upright: read once, nothing changes.
    upright.save(tmp_path / "nota.png")
    assert any(f.type == "email" for f in common_analyze(engine, tmp_path / "nota.png").findings)


def common_analyze(engine: RealEngine, path):
    file = AnalyzedFile(id="f1", name=path.name, path=str(path))
    engine.analyze(file, [])
    assert file.status == "ready", file.error
    return file
