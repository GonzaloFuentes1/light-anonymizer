"""D10: the exceptions list (RUTs of institutions, phones, 600 and 800 numbers). Invented data only."""

from __future__ import annotations

import numpy as np
import pymupdf
import pytest
from PIL import Image

from anonymizer.engine import exceptions, ocr, raster
from anonymizer.engine.model import AnalyzedFile, DetectionOptions, Finding, HistoryEntry
from anonymizer.engine.patterns import phones
from anonymizer.engine.real import RealEngine
from anonymizer.engine.text import dedup_spans, detect_spans

INSTITUTION_RUT = "72.123.456-8"  # invented, valid check digit
OTHER_RUT = "12.345.678-5"
CALL_CENTER = "600 123 4567"
TOLL_FREE = "800 123 456"
OTHER_URL = "https://www.goreficticio.cl/noticias/2026/informe-anual"


@pytest.mark.parametrize(
    "entry,rut,phone",
    [
        ("72.123.456-8", "721234568", None),
        ("72123456-8", "721234568", None),
        ("  72 123 456 - 8 ", "721234568", None),
        ("21.098.765-k", "21098765K", None),
        ("21098765K", "21098765K", None),
        ("600 123 4567", None, "6001234567"),
        ("600-123-4567", None, "6001234567"),
        ("+56 600 123 4567", None, "6001234567"),
        ("800 123 456", None, "800123456"),
        ("+56 2 2345 6789", None, "223456789"),
        ("(41) 221 3456", None, "412213456"),
        ("0056 9 8123 4567", None, "981234567"),
        ("721234568", "721234568", "721234568"),  # bare digits: either
        ("600.123.4567", None, "6001234567"),  # dots in a 600 number: not a RUT
    ],
)
def test_entries_are_normalized(entry, rut, phone):
    assert exceptions.parse(entry) == (rut, phone)


@pytest.mark.parametrize("entry", ["", "Gobierno Regional", "123", "72.123", "abc-1", "www.goreficticio.cl"])
def test_entries_that_are_not_a_rut_or_a_phone(entry):
    assert exceptions.parse(entry) == (None, None)


def test_clean_keeps_one_of_each_value_and_reports_the_invalid_ones():
    kept, invalid = exceptions.clean(
        ["72.123.456-8", "72123456-8", " 600 123 4567 ", "+56 600 123 4567", "", "Mesa central"]
    )
    assert kept == ["72.123.456-8", "600 123 4567"]
    assert invalid == ["Mesa central"]


def test_matching_ignores_the_format():
    exc = exceptions.Exceptions.from_entries(["72.123.456-8", "600 123 4567", "800 123 456"])
    for value in ("72.123.456-8", "72123456-8", "72.123.456‐8", "72,123,456-8", "721234568"):
        assert exc.matches(value, "rut"), value
    for value in ("600 123 4567", "+56 600 123 4567", "600-123-4567", "800 123 456", "+56 800 123 456"):
        assert exc.matches(value, "phone"), value
    assert not exc.matches(OTHER_RUT, "rut") and not exc.matches("600 123 4568", "phone")
    assert not exceptions.Exceptions.from_entries([]).matches(INSTITUTION_RUT, "rut")


def test_a_rut_and_a_phone_with_the_same_digits_are_different_values():
    phone = exceptions.Exceptions.from_entries(["+56 2 2345 6789"])
    assert phone.matches("(2) 2345 6789", "phone") and not phone.matches("22.345.678-9", "rut")
    rut = exceptions.Exceptions.from_entries(["61.234.567-8"])
    assert rut.matches("61234567-8", "rut") and not rut.matches("(61) 234 5678", "phone")
    bare = exceptions.Exceptions.from_entries(["223456789"])  # written without format: either
    assert bare.matches("22.345.678-9", "rut") and bare.matches("2 2345 6789", "phone")


def test_600_numbers_are_phones():
    # D10: they are still censored by default, in every format (with dashes they were not found).
    for text in ("Mesa central 600-123-4567", "Llame al 600 123 4567", "Fono +56 600 123 4567"):
        assert phones(text), text
    # Written with spaces, a 600 number has the shape of a RUT without a dash, but no RUT has that
    # 9-digit body: it is a phone, and not a doubtful RUT.
    spans = dedup_spans(detect_spans("Mesa central 600 123 4567", ()))
    assert [(s[0], s[1], s[2]) for s in spans] == [("phone", 13, 25)]


def finding(type_: str, text: str, detector: str = "regex", **kw) -> Finding:
    return Finding(
        id=text[:12],
        file_id="f1",
        page=0,
        type=type_,
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        text=text,
        detector=detector,
        history=[HistoryEntry(at="2026-10-05T10:00:00+00:00", action="proposed")],
        **kw,
    )


def test_apply_turns_listed_values_into_suggestions():
    findings = [
        finding("rut", INSTITUTION_RUT),
        finding("rut", OTHER_RUT),
        finding("phone", CALL_CENTER),
        finding("phone", f"+56 {TOLL_FREE}"),
        finding("rut", f"RUT: {INSTITUTION_RUT}", detector="ocr"),  # an OCR line with only that datum
        finding("rut", f"RUT: {INSTITUTION_RUT} Fono: {CALL_CENTER}", detector="ocr"),
        finding("rut", f"RUT: {INSTITUTION_RUT} {OTHER_URL}", detector="ocr"),
        finding("rut", f"RUT {INSTITUTION_RUT} Fono {OTHER_RUT}", detector="ocr"),  # another RUT: applied
        finding("name", f"Ana Luisa Soto {INSTITUTION_RUT}", detector="ocr"),  # a name: never an exception
        finding("rut", f"Ana Luisa Soto {INSTITUTION_RUT}", detector="ocr"),
        finding("rut", "72 123 456 8"),  # a value read by a context rule, without its dash
    ]
    changed = exceptions.apply(findings, ["72.123.456-8", CALL_CENTER, TOLL_FREE], ())
    excepted = [f for f in findings if f.optional]
    assert changed == len(excepted) == 7
    assert [f.text for f in findings if not f.optional] == [
        OTHER_RUT,
        f"RUT {INSTITUTION_RUT} Fono {OTHER_RUT}",
        f"Ana Luisa Soto {INSTITUTION_RUT}",
        f"Ana Luisa Soto {INSTITUTION_RUT}",
    ]
    for f in excepted:
        assert f.status == "suggested" and f.optional_reason == "exception" and not f.active
        assert [h.action for h in f.history] == ["suggested"]
    for f in findings:
        if not f.optional:
            assert f.status == "proposed" and f.optional_reason is None


def test_apply_leaves_other_urls_and_reviewer_zones_alone():
    url = finding("url", OTHER_URL, optional=True, optional_reason="url", status="suggested")
    drawn = finding("manual", "", detector="reviewer", status="added")
    face = finding("face", "")
    assert exceptions.apply([url, drawn, face], [INSTITUTION_RUT], ()) == 0
    assert url.optional_reason == "url" and drawn.status == "added" and not face.optional


def test_a_name_from_the_list_next_to_the_value_is_never_an_exception():
    line = finding("rut", f"Zoila Quintanilla Brito {INSTITUTION_RUT}", detector="ocr")
    assert exceptions.apply([line], [INSTITUTION_RUT], ("Zoila Quintanilla Brito",)) == 0
    assert line.status == "proposed"


def test_real_engine_marks_listed_values_before_the_file_is_ready(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    lines = [
        "Informe ficticio de prueba para el motor de anonimización, con texto neutro de relleno.",
        f"RUT del servicio: {INSTITUTION_RUT}",
        "Mesa central: 600-123-4567",
        f"RUT de la persona: {OTHER_RUT}",
    ]
    for i, line in enumerate(lines):
        page.insert_text((72, 100 + 22 * i), line, fontsize=11)
    doc.save(tmp_path / "a.pdf")
    doc.close()
    options = DetectionOptions(ocr=False, faces=False, qr=False).to_dict()
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(tmp_path / "a.pdf"), options=options)
    file.exceptions = ["72123456-8", "+56 600 123 4567"]
    RealEngine().analyze(file, [])
    assert file.status == "ready"
    by_text = {f.text: f for f in file.findings}
    assert by_text[INSTITUTION_RUT].status == "suggested" and by_text["600-123-4567"].status == "suggested"
    assert by_text["600-123-4567"].type == "phone"
    assert by_text[OTHER_RUT].status == "proposed" and not by_text[OTHER_RUT].optional
    # Without the list, the same values are censored by default.
    file.exceptions = []
    RealEngine().analyze(file, [])
    assert all(f.status == "proposed" for f in file.findings)


def ocr_line(text: str, y: float = 100, x0: float = 50, x1: float = 900) -> ocr.OcrLine:
    return ocr.OcrLine(np.array([[x0, y], [x1, y], [x1, y + 30], [x0, y + 30]], np.float64), text, 0.98, 0)


def ocr_findings(lines: list[ocr.OcrLine], entries: list[str], all_text: bool = False) -> list[Finding]:
    """The findings of OCR lines, as the engine builds them, with ``entries`` applied."""
    zones = raster.dedup(raster._text_zones(lines, (), True, all_text, DetectionOptions(), 1000, 1000))
    findings = [finding(z.type, z.text, detector=z.detector) for z in zones]
    for i, f in enumerate(findings):
        f.id = f"z{i}"
    exceptions.apply(findings, entries, ())
    return findings


def test_a_line_with_anything_else_than_the_listed_value_and_its_label_stays_censored():
    # A name found only by a context rule ("Solicitante:", a "Nombre" column) on the same OCR line
    # as the listed number: the context zone keeps its type when it is merged, and blocks the list.
    cases = {
        "Solicitante: Quintanilla Brito 600 123 4567": [ocr_line("Solicitante: Quintanilla Brito 600 123 4567")],
        "Quintanilla Brito 600 123 4567": [
            ocr_line("Nombre", y=50, x0=50, x1=150),
            ocr_line("Cargo", y=50, x0=400, x1=480),
            ocr_line("Unidad", y=50, x0=600, x1=700),
            ocr_line("Quintanilla Brito 600 123 4567", y=100, x0=40, x1=380),
        ],
        "Telefono: 600 123 4567 / 223 4567": [ocr_line("Telefono: 600 123 4567 / 223 4567")],  # an old number
        "Llame a la secretaria Zoila al 600 123 4567": [ocr_line("Llame a la secretaria Zoila al 600 123 4567")],
    }
    for text, lines in cases.items():
        findings = ocr_findings(lines, [CALL_CENTER])
        line = next(f for f in findings if f.text == text)
        assert line.status == "proposed" and not line.optional, text
    for text in ("Solicitante: Quintanilla Brito 600 123 4567", "Quintanilla Brito 600 123 4567"):
        merged = next(f for f in ocr_findings(cases[text], []) if f.text == text)
        assert merged.type == "name", text


def test_a_line_with_only_labels_and_listed_values_is_an_exception():
    for text in (
        f"Mesa central: {CALL_CENTER}",
        f"Línea gratuita {TOLL_FREE}",
        f"RUT N° {INSTITUTION_RUT}",
        f"RUT: {INSTITUTION_RUT}",
        f"Fono: {CALL_CENTER}.",
    ):
        (line,) = ocr_findings([ocr_line(text)], [CALL_CENTER, TOLL_FREE, INSTITUTION_RUT])
        assert line.status == "suggested" and line.optional_reason == "exception", text


@pytest.mark.parametrize(
    "text",
    [
        f"Contacto: Rut Mesa {CALL_CENTER}",  # a person called Rut Mesa, not two labels
        f"Fono Mesa {CALL_CENTER}",  # Mesa is a surname: as a label it needs a colon
        f"Rut {INSTITUTION_RUT}",  # the given name Rut, not the uppercase label RUT
        f"J U A N P E R E Z {CALL_CENTER}",  # a name written letter by letter
        f"J. P. {CALL_CENTER}",  # initials
        f"Whatsapp {CALL_CENTER} · Cel. E. Rut",  # words after the value
        f"Fono: {CALL_CENTER} Ana",
    ],
)
def test_names_and_initials_next_to_a_listed_value_keep_it_censored(text):
    (line,) = ocr_findings([ocr_line(text)], [CALL_CENTER, INSTITUTION_RUT])
    assert line.status == "proposed" and not line.optional, text


def test_a_context_value_with_another_number_stays_censored(tmp_path):
    # The text layer: the value of "Telefono:" holds the listed number and a direct line the
    # patterns do not take; the whole value is one zone, so it is never an exception.
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    lines = [
        "Informe ficticio de prueba para el motor de anonimizacion, con texto neutro de relleno.",
        "Telefono: mesa central 600 123 4567, anexo directo (2) 234 5678",
        "Funcionario: Quintanilla Brito, 600 123 4567",
        f"Fono: {CALL_CENTER}",
    ]
    for i, line in enumerate(lines):
        page.insert_text((60, 100 + 22 * i), line, fontsize=10)
    doc.save(tmp_path / "ctx.pdf")
    doc.close()
    options = DetectionOptions(ocr=False, faces=False, qr=False).to_dict()
    file = AnalyzedFile(id="c", name="ctx.pdf", path=str(tmp_path / "ctx.pdf"), options=options)
    file.exceptions = [CALL_CENTER]
    engine = RealEngine()
    engine.analyze(file, [])
    by_text = {f.text: f for f in file.findings}
    assert by_text["mesa central 600 123 4567, anexo directo (2) 234 5678"].status == "proposed"
    assert by_text["Quintanilla Brito, 600 123 4567"].status == "proposed"
    listed = [f for f in file.findings if f.status == "suggested"]
    assert [f.text for f in listed] == [CALL_CENTER] and listed[0].type == "phone" and not listed[0].doubtful
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported
    with pymupdf.open(result.output_path) as out:
        text = out[0].get_text()
    assert "234 5678" not in text and "Quintanilla" not in text and f"Fono: {CALL_CENTER}" in text


def test_the_list_does_not_apply_when_every_line_is_censored(tmp_path, monkeypatch):
    # "Censurar todo el texto de esta imagen": every line is censored, listed values too.
    line = ocr_line(f"RUT {INSTITUTION_RUT}")
    monkeypatch.setattr(ocr, "read_lines", lambda bgr, min_side=0, check=None, mirror=True: [line])
    Image.new("RGB", (1000, 300), "white").save(tmp_path / "foto.png")
    options = DetectionOptions(faces=False, qr=False).to_dict()
    for all_text, status in ((False, "suggested"), (True, "proposed")):
        path = str(tmp_path / "foto.png")
        file = AnalyzedFile(id="i", name="foto.png", path=path, options=options, all_text=all_text)
        file.exceptions = [INSTITUTION_RUT]
        RealEngine().analyze(file, [])
        assert [f.status for f in file.findings] == [status], all_text
