"""Selectable detection groups, stage timings, the time estimate and D12 (other URLs), on the real
engine. OCR, faces and QR are replaced by recorders where the test is about skipping work, so most
of these tests need no models. Invented data only."""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image

from anonymizer.engine import common, context, estimate, faces, ocr, qr, raster
from anonymizer.engine.common import Zone
from anonymizer.engine.model import DETECTION_GROUPS, AnalyzedFile, DetectionOptions
from anonymizer.engine.patterns import rut_check_digit
from anonymizer.engine.real import RealEngine

VALID_RUT = f"12.345.678-{rut_check_digit('12345678')}"
EMAIL = "ana.prueba@ejemplo.cl"
PHONE = "+56 9 8123 4567"
SIGNER = "Ana Luisa Soto"  # short line starting with a known given name: found by context
LISTED = "Zoila Quintanilla Brito"  # not a dictionary given name: only the list finds it
OTHER_URL = "https://www.goreficticio.cl/noticias/2026/informe-anual"
PERSONAL_URL = "https://www.facebook.com/ana.prueba.inventada"
# Pages with less than 50 characters count as scanned: every test page carries this line.
NEUTRAL = "Informe ficticio de prueba para el motor de anonimización, con texto neutro de relleno."


def text_pdf(path: Path, lines: list[str], pages: int = 1) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        for i, line in enumerate([NEUTRAL, *lines]):
            page.insert_text((72, 100 + 22 * i), line, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def scanned_pdf(path: Path) -> Path:
    """One page that is only an image (no text layer): a scan."""
    img = Image.new("RGB", (850, 1100), "white")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)
    doc.close()
    return path


def analyze(path: Path, options: DetectionOptions | None = None, names: list[str] | None = None) -> AnalyzedFile:
    file = AnalyzedFile(id="f1", name=path.name, path=str(path), options=(options or DetectionOptions()).to_dict())
    RealEngine().analyze(file, names or [])
    assert file.status == "ready", (file.error, file.error_message)
    return file


@pytest.fixture
def recorders(monkeypatch):
    """OCR returns one line with an e-mail; faces and QR record their calls (and find nothing)."""
    calls = {"ocr": 0, "faces": 0, "qr": 0}
    line = ocr.OcrLine(
        np.array([[100.0, 100.0], [900.0, 100.0], [900.0, 140.0], [100.0, 140.0]]), f"Correo: {EMAIL}", 0.95, 0
    )

    def read_lines(bgr, min_side=0, check=None):
        calls["ocr"] += 1
        if check is not None:
            check()
        return [line]

    def detect_faces(bgr, threshold=0.5, check=None):
        calls["faces"] += 1
        return []

    def detect_qr(bgr):
        calls["qr"] += 1
        return []

    monkeypatch.setattr(ocr, "read_lines", read_lines)
    monkeypatch.setattr(faces, "detect", detect_faces)
    monkeypatch.setattr(qr, "detect", detect_qr)
    return calls


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def test_groups_and_defaults():
    keys = [g.key for g in DETECTION_GROUPS]
    assert keys == ["patterns", "urls_personal", "urls_other", "names_list", "names_context", "ocr", "faces", "qr"]
    defaults = DetectionOptions().to_dict()
    assert defaults == {g.key: g.default for g in DETECTION_GROUPS}
    assert defaults["urls_other"] is False and all(v for k, v in defaults.items() if k != "urls_other")
    for g in DETECTION_GROUPS:
        assert g.label and g.description and g.short
        assert g.warning or g.locked or not g.detection  # every switch that can hide data warns
    assert [g.key for g in DETECTION_GROUPS if g.locked] == ["patterns"]


def test_patterns_cannot_be_turned_off(tmp_path):
    assert DetectionOptions.from_dict({"patterns": False}).patterns is True
    assert DetectionOptions().replace(patterns=False).patterns is True
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(text_pdf(tmp_path / "a.pdf", [f"RUT: {VALID_RUT}"])))
    file.options = {"patterns": False, "faces": False}
    RealEngine().analyze(file, [])
    assert file.options["patterns"] is True and file.options["faces"] is False
    assert any(f.type == "rut" and f.text == VALID_RUT for f in file.findings)


def test_ocr_off_reads_no_pixels(tmp_path, recorders):
    path = scanned_pdf(tmp_path / "scan.pdf")
    on = analyze(path)
    assert recorders["ocr"] > 0 and any(f.type == "email" and f.detector == "ocr" for f in on.findings)
    assert on.pages[0].scanned is True
    recorders["ocr"] = 0
    off = analyze(path, DetectionOptions(ocr=False))
    assert recorders["ocr"] == 0  # no OCR pass at all, not just hidden results
    assert not off.findings and "ocr" not in off.timings
    assert recorders["faces"] > 0 and recorders["qr"] > 0  # faces and QR still run on the scan


def test_faces_and_qr_off_are_not_called(tmp_path, recorders):
    Image.new("RGB", (640, 480), "white").save(tmp_path / "foto.png")
    file = analyze(tmp_path / "foto.png", DetectionOptions(faces=False, qr=False))
    assert recorders == {"ocr": 1, "faces": 0, "qr": 0}
    assert set(file.timings) >= {"render", "ocr", "analyze"} and not {"faces", "qr"} & set(file.timings)


def test_nothing_rendered_when_ocr_faces_and_qr_are_off(tmp_path, recorders, monkeypatch):
    rendered = []
    original = pymupdf.Page.get_pixmap
    monkeypatch.setattr(
        pymupdf.Page, "get_pixmap", lambda self, *a, **kw: rendered.append(1) or original(self, *a, **kw)
    )
    file = analyze(scanned_pdf(tmp_path / "scan.pdf"), DetectionOptions(ocr=False, faces=False, qr=False))
    assert not rendered and recorders == {"ocr": 0, "faces": 0, "qr": 0}
    assert set(file.timings) == {"text", "analyze"}


def test_name_list_off(tmp_path):
    line = f"En la reunión del comité de evaluación del proyecto regional participó {LISTED} como asesora."
    path = text_pdf(tmp_path / "a.pdf", [line, SIGNER])
    on = analyze(path, names=[LISTED])
    assert any(f.detector == "name_list" and LISTED in (f.text or "") for f in on.findings)
    off = analyze(path, DetectionOptions(names_list=False), names=[LISTED])
    assert not any(LISTED in (f.text or "") for f in off.findings)
    signer = [f for f in off.findings if f.text == SIGNER]
    assert signer and signer[0].detector == "context" and signer[0].doubtful  # context names stay


def test_context_names_off_keeps_the_other_context_rules(tmp_path):
    path = text_pdf(tmp_path / "a.pdf", [SIGNER, "Nombre: Pedro Inventado", "Domicilio: Pasaje Ficticio 123", EMAIL])
    on = analyze(path)
    assert {f.text for f in on.findings if f.type == "name"} >= {SIGNER, "Pedro Inventado"}
    off = analyze(path, DetectionOptions(names_context=False))
    assert not any(f.type == "name" for f in off.findings)
    assert any(f.type == "address" and f.detector == "context" for f in off.findings)
    assert any(f.type == "email" and f.text == EMAIL for f in off.findings)


def test_context_rules_without_names():
    L = context.Line
    lines = [L("Nombre:", 0, 0, 60, 10), L("Pedro Inventado", 70, 0, 200, 10), L("Firma", 0, 40, 40, 50),
             L("Correo", 300, 40, 360, 50), L("Fono", 400, 40, 440, 50), L(EMAIL, 300, 55, 420, 65)]  # fmt: skip
    spans, rects = context.context_rules(lines, 600, 800)
    assert {s[0] for s in spans} & {"name"} and {s[0] for s in spans} & {"email"}
    spans, rects = context.context_rules(lines, 600, 800, names=False)
    assert not {s[0] for s in spans} & set(context.NAME_TYPES) and not [r for r in rects if r[0] == "signature"]
    assert "email" in {s[0] for s in spans}


def test_personal_urls_off(tmp_path):
    path = text_pdf(tmp_path / "a.pdf", [f"Perfil: {PERSONAL_URL}", f"Noticia: {OTHER_URL}"])
    on = analyze(path)
    assert {f.text for f in on.findings if f.type == "url"} == {PERSONAL_URL, OTHER_URL}
    off = analyze(path, DetectionOptions(urls_personal=False))
    assert {f.text for f in off.findings if f.type == "url"} == {OTHER_URL}  # the other URL is always found


def test_everything_on_keeps_every_finding_applied(tmp_path):
    lines = [f"RUT: {VALID_RUT}", f"Correo: {EMAIL}", f"Teléfono: {PHONE}", SIGNER, f"Ver {OTHER_URL}",
             f"Perfil {PERSONAL_URL}"]  # fmt: skip
    path = text_pdf(tmp_path / "a.pdf", lines)
    every = analyze(path, DetectionOptions.everything())
    default = analyze(path)

    def shape(file):
        return sorted((f.page, f.type, f.text, f.detector, f.doubtful, str(f.polygon)) for f in file.findings)

    assert shape(every) == shape(default)  # the same detection: D12 only changes the starting status
    assert all(f.active and f.status == "proposed" for f in every.findings)
    assert {f.status for f in default.findings if f.optional} == {"suggested"}


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------


def test_stage_timings(tmp_path, recorders):
    text = analyze(text_pdf(tmp_path / "a.pdf", [f"RUT: {VALID_RUT}"]))
    assert set(text.timings) == {"text", "analyze"}
    assert 0 <= text.timings["text"] <= text.timings["analyze"]
    scan = analyze(scanned_pdf(tmp_path / "scan.pdf"))
    assert set(scan.timings) == {"text", "render", "ocr", "faces", "qr", "analyze"}
    assert sum(v for k, v in scan.timings.items() if k != "analyze") <= scan.timings["analyze"] + 0.01


def test_waiting_for_a_lock_is_not_counted():
    lock = threading.Lock()
    lock.acquire()
    threading.Timer(0.3, lock.release).start()
    clock = common.StageClock()
    started = time.perf_counter()
    with clock.running(), common.stage("ocr"), common.waiting_for(lock):
        time.sleep(0.02)
    assert time.perf_counter() - started >= 0.29
    assert 0.01 <= clock.seconds["ocr"] < 0.2
    with common.stage("faces"):  # no clock running in this thread: nothing is recorded
        pass
    assert "faces" not in clock.seconds


# ---------------------------------------------------------------------------
# profile and cost model
# ---------------------------------------------------------------------------


def test_profile_is_fast_and_correct(tmp_path):
    logo = io.BytesIO()
    Image.new("RGB", (240, 120), (20, 60, 140)).save(logo, "PNG")
    scan = io.BytesIO()
    Image.new("RGB", (400, 520), "white").save(scan, "PNG")
    doc = pymupdf.open()
    xref = 0
    for n in range(30):
        page = doc.new_page(width=595, height=842)
        if n < 10:  # scanned: only an image
            page.insert_image(page.rect, stream=scan.getvalue())
            continue
        page.insert_text(
            (72, 200), "Texto neutro de una página del informe ficticio, con varias palabras.", fontsize=11
        )
        if n < 20:  # the same logo on ten pages
            rect = pymupdf.Rect(72, 40, 192, 100)
            xref = page.insert_image(rect, xref=xref) if xref else page.insert_image(rect, stream=logo.getvalue())
    doc.save(tmp_path / "mixed.pdf")
    doc.close()
    started = time.perf_counter()
    facts = estimate.profile(str(tmp_path / "mixed.pdf"))
    assert time.perf_counter() - started < 3
    assert facts["kind"] == "pdf" and facts["pages"] == 30
    assert facts["scanned_pages"] == 10 and facts["raster_pages"] == 20
    assert facts["regions"] == 1 and 0 < facts["regions_mp"] < 0.2  # the logo is read once
    frames = [Image.new("RGB", (1000, 500), "white"), Image.new("RGB", (400, 300), "white")]
    frames[0].save(tmp_path / "two.tif", save_all=True, append_images=frames[1:])
    facts = estimate.profile(str(tmp_path / "two.tif"))
    assert facts["kind"] == "image" and facts["frames"] == 2
    assert facts["megapixels"] == pytest.approx(0.62, abs=0.001)
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.7 roto")
    assert estimate.profile(str(tmp_path / "bad.pdf"))["pages"] == 0  # an estimate never fails


def test_cost_model_learns_and_saves_only_numbers(tmp_path):
    path = tmp_path / "estimates.json"
    model = estimate.CostModel.load(path)
    assert not model.calibrated
    facts = {**estimate.empty_profile("pdf"), "pages": 2, "scanned_pages": 2, "raster_pages": 2}
    before = model.stages(facts)
    assert before["ocr"] == pytest.approx(2 * estimate.DEFAULT_RATES["ocr"]["scanned_page"])
    assert model.update(facts, {"ocr": before["ocr"] * 2, "faces": 0.5, "analyze": 99})
    after = model.stages(facts)
    assert before["ocr"] < after["ocr"] < before["ocr"] * 2  # moved towards the measurement, smoothed
    assert after["faces"] < before["faces"]
    assert after["qr"] == before["qr"]  # a stage that did not run is not touched
    saved = json.loads(path.read_text(encoding="utf-8"))  # only numbers: no names, paths or content
    assert set(saved) == {"version", "samples", "rates"} and saved["samples"] == 1
    assert all(isinstance(v, float) for rates in saved["rates"].values() for v in rates.values())
    again = estimate.CostModel.load(path)
    assert again.calibrated and again.stages(facts)["ocr"] == pytest.approx(after["ocr"], rel=1e-4)
    path.write_text("{no es json", encoding="utf-8")
    assert estimate.CostModel.load(path).stages(facts) == pytest.approx(before)  # broken file: defaults


def test_estimate_total_follows_the_options():
    stages = {"text": 0.1, "render": 0.3, "ocr": 14.0, "faces": 1.8, "qr": 0.2}
    groups = estimate.by_group(stages)
    assert groups["ocr"] == 14.0 and groups["patterns"] == 0.1 and groups["names_list"] == 0.0
    every = estimate.total(stages, DetectionOptions())
    assert every == pytest.approx(16.4)
    assert estimate.total(stages, DetectionOptions(ocr=False)) == pytest.approx(2.4)
    assert estimate.total(stages, DetectionOptions(ocr=False, faces=False, qr=False)) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# D12: personal URLs and other URLs
# ---------------------------------------------------------------------------


def test_other_url_is_optional_and_left_visible_on_export(tmp_path):
    path = text_pdf(tmp_path / "a.pdf", [f"RUT: {VALID_RUT}", f"Más información en {OTHER_URL}", OTHER_URL])
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(path))
    engine.analyze(file, [])
    urls = [f for f in file.findings if f.type == "url"]
    assert len(urls) == 2 and all(f.optional and f.status == "suggested" and not f.active for f in urls)
    assert all(f.history[0].action == "suggested" for f in urls)
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]  # left visible on purpose: never a leak
    assert result.removed_by_reviewer == 0 and result.redactions_applied == 1
    with pymupdf.open(result.output_path) as doc:
        text = doc[0].get_text()
    assert OTHER_URL in text.replace("\n", "") and VALID_RUT not in text
    urls[0].status = "proposed"  # the reviewer applies one copy
    result = engine.export(file, str(tmp_path / "out2"))
    assert result.exported and result.redactions_applied == 2
    with pymupdf.open(result.output_path) as doc:
        assert doc[0].get_text().replace("\n", "").count(OTHER_URL) == 1


def test_other_urls_applied_from_the_start(tmp_path):
    file = analyze(text_pdf(tmp_path / "a.pdf", [f"Noticia: {OTHER_URL}"]), DetectionOptions(urls_other=True))
    url = next(f for f in file.findings if f.type == "url")
    assert url.optional and url.status == "proposed" and url.active


@pytest.mark.parametrize(
    "url",
    [
        PERSONAL_URL,
        "https://www.goreficticio.cl/ficha?rut=12345678-5",
        "https://www.goreficticio.cl/equipo/zoila-quintanilla-brito",
        "https://teams.microsoft.com/l/meetup-join/reunion-inventada",
    ],
)
def test_personal_url_is_never_optional(tmp_path, url):
    file = analyze(text_pdf(tmp_path / "a.pdf", [f"Enlace: {url}"]), names=["Zoila Quintanilla Brito"])
    found = [f for f in file.findings if f.type == "url"]
    assert found and not any(f.optional for f in found) and all(f.status == "proposed" for f in found)


def test_url_with_a_name_found_elsewhere_is_not_optional(tmp_path):
    # The signer is found by context; the same name inside a URL makes that URL personal.
    lines = [SIGNER, "Su ficha está en https://www.goreficticio.cl/equipo/ana-luisa-soto"]
    file = analyze(text_pdf(tmp_path / "a.pdf", lines))
    url = next(f for f in file.findings if f.type == "url")
    assert not url.optional and url.status == "proposed"


def test_url_under_a_name_label_is_not_optional(tmp_path):
    file = analyze(text_pdf(tmp_path / "a.pdf", [f"Nombre: {OTHER_URL}"]))
    url = next(f for f in file.findings if f.type == "url")
    assert not url.optional


def test_ocr_zone_that_is_also_a_name_is_not_optional():
    pol = np.array([[0.0, 0.0], [300.0, 0.0], [300.0, 20.0], [0.0, 20.0]])
    zones = [Zone("url", pol, OTHER_URL, "ocr", 0.9, None, True), Zone("name", pol, OTHER_URL, "context", 0.9)]
    merged = raster.dedup(zones)
    assert len(merged) == 1 and merged[0].optional is False
    assert raster.dedup([zones[0]])[0].optional is True


def test_ocr_line_with_only_an_other_url_is_optional(recorders, monkeypatch):
    lines = [
        ocr.OcrLine(np.array([[10.0, 10.0], [600.0, 10.0], [600.0, 40.0], [10.0, 40.0]]), OTHER_URL, 0.95, 0),
        ocr.OcrLine(
            np.array([[10.0, 60.0], [600.0, 60.0], [600.0, 90.0], [10.0, 90.0]]), f"{OTHER_URL} {EMAIL}", 0.95, 0
        ),
    ]
    monkeypatch.setattr(ocr, "read_lines", lambda bgr, min_side=0, check=None: lines)
    rgb = np.full((200, 700, 3), 255, np.uint8)
    zones = raster.detect_in_image(rgb, ())
    by_text = {z.text: z for z in zones}
    assert by_text[OTHER_URL].optional is True
    assert by_text[f"{OTHER_URL} {EMAIL}"].optional is False  # it also covers an e-mail
    every_line = raster.detect_in_image(rgb, (), all_text=True)
    assert not any(z.optional for z in every_line)  # "censurar todo el texto" leaves nothing out


def test_ocr_url_with_a_name_found_elsewhere_is_not_optional(tmp_path, recorders, monkeypatch):
    # A name signed on a text page, or read by OCR on another line, makes the URL that carries it
    # personal, also when the URL was read by OCR (a scanned page or a photo).
    def line(y: float, text: str) -> ocr.OcrLine:
        return ocr.OcrLine(np.array([[50.0, y], [700.0, y], [700.0, y + 30], [50.0, y + 30]]), text, 0.95, 0)

    lines = [
        line(50, "https://www.goreficticio.cl/equipo/ana-luisa-soto"),
        line(150, "Nombre: Pedro Inventado Rojas"),
        line(250, "https://www.goreficticio.cl/equipo/pedro_inventado_rojas"),
        line(350, OTHER_URL),
    ]
    monkeypatch.setattr(ocr, "read_lines", lambda bgr, min_side=0, check=None: lines)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), NEUTRAL, fontsize=11)
    page.insert_text((72, 130), SIGNER, fontsize=11)
    doc.save(tmp_path / "text.pdf")
    doc.close()
    scan = pymupdf.open(scanned_pdf(tmp_path / "scan.pdf"))
    mixed = pymupdf.open(tmp_path / "text.pdf")
    mixed.insert_pdf(scan)
    mixed.save(tmp_path / "mixed.pdf")
    mixed.close()
    scan.close()
    file = analyze(tmp_path / "mixed.pdf")
    urls = {f.text: f for f in file.findings if f.type == "url"}
    for text in (lines[0].text, lines[2].text):
        assert not urls[text].optional and urls[text].status == "proposed"
        assert urls[text].history[0].action == "proposed"
    assert urls[OTHER_URL].optional and urls[OTHER_URL].status == "suggested"
    # In a photo only what that photo says counts: the signer of the PDF is not there.
    Image.new("RGB", (800, 400), "white").save(tmp_path / "foto.png")
    photo = {f.text: f for f in analyze(tmp_path / "foto.png").findings if f.type == "url"}
    assert photo[lines[0].text].optional and not photo[lines[2].text].optional


def test_text_url_with_underscores_and_a_name_found_elsewhere_is_not_optional(tmp_path):
    lines = [SIGNER, "Ficha: https://www.goreficticio.cl/equipo/ana_luisa_soto"]
    url = next(f for f in analyze(text_pdf(tmp_path / "a.pdf", lines)).findings if f.type == "url")
    assert not url.optional and url.status == "proposed"


@pytest.mark.parametrize(
    "content",
    [
        '{"version": 1, "samples": Infinity, "rates": {}}',
        '{"version": 1, "samples": 1e999, "rates": {}}',
        '{"version": 1, "samples": NaN, "rates": {}}',
        '{"version": 1, "samples": 3, "rates": {"ocr": {"scanned_page": true}}}',
        '{"version": 1, "samples": 3, "rates": {"ocr": {"scanned_page": Infinity}}}',
        '{"version": 1, "samples": 3, "rates": {"ocr": "rapido"}}',
        '{"version": 1, "samples": 3, "rates": []}',
        "[1, 2, 3]",
        "null",
        "",
    ],
)
def test_cost_model_survives_a_broken_file(tmp_path, content):
    path = tmp_path / "estimates.json"
    path.write_text(content, encoding="utf-8")
    model = estimate.CostModel.load(path)  # never raises: the app must always start
    assert model.rates == estimate.DEFAULT_RATES
    assert all(isinstance(v, float) for rates in model.rates.values() for v in rates.values())
    path.write_bytes(b"\xff\xfe\x00")
    assert estimate.CostModel.load(path).rates == estimate.DEFAULT_RATES


def test_estimate_total_never_grows_when_a_group_is_turned_off():
    stages = {"text": 0.4, "render": 1.3, "ocr": 21.0, "faces": 2.2, "qr": 0.3}
    keys = [g.key for g in DETECTION_GROUPS if not g.locked]
    for bits in range(2 ** len(keys)):
        options = DetectionOptions.from_dict({k: bool(bits >> i & 1) for i, k in enumerate(keys)})
        seconds = estimate.total(stages, options)
        for key in keys:
            if getattr(options, key):
                assert estimate.total(stages, options.replace(**{key: False})) <= seconds


def test_profile_takes_the_pdf_lock_one_page_at_a_time(tmp_path, monkeypatch):
    held = []

    class Recorder:
        def __enter__(self):
            held.append(time.perf_counter())

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(estimate, "PDF_LOCK", Recorder())
    facts = estimate.profile(str(text_pdf(tmp_path / "a.pdf", ["Texto neutro."], pages=6)))
    assert facts["pages"] == 6 and len(held) >= 6  # analyses can use PyMuPDF between two pages
