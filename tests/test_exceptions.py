"""D10: the exceptions list (RUTs of institutions, phones, 600 and 800 numbers). Invented data only."""

from __future__ import annotations

import pymupdf
import pytest

from anonymizer.engine import exceptions
from anonymizer.engine.model import AnalyzedFile, DetectionOptions, Finding, HistoryEntry
from anonymizer.engine.patterns import phones
from anonymizer.engine.real import RealEngine

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
        assert exc.matches(value), value
    for value in ("600 123 4567", "+56 600 123 4567", "600-123-4567", "800 123 456", "+56 800 123 456"):
        assert exc.matches(value), value
    assert not exc.matches(OTHER_RUT) and not exc.matches("600 123 4568")
    assert not exceptions.Exceptions.from_entries([]).matches(INSTITUTION_RUT)


def test_600_numbers_are_phones():
    # D10: they are still censored by default, in every format (with dashes they were not found).
    for text in ("Mesa central 600-123-4567", "Llame al 600 123 4567", "Fono +56 600 123 4567"):
        assert phones(text), text


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
