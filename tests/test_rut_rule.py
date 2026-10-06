"""RUTs decided 2026-10-06: a bare run of digits whose check digit does not match and with no RUT
label before it is only suggested (shown, not applied); a "rut" finding always holds a RUT-shaped
number. Invented data only."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from anonymizer.engine import audit, raster
from anonymizer.engine.common import Zone
from anonymizer.engine.model import AnalyzedFile, ExportResult
from anonymizer.engine.patterns import rut_check_digit
from anonymizer.engine.real import RealEngine
from anonymizer.engine.text import DOUBT_RUT, context_type, rut_suggested
from tests.test_engine_real import _text_image, needs_ocr

BARE_BAD = "123456780"  # 12.345.678-0: the check digit should be 5
BARE_GOOD = f"12345678{rut_check_digit('12345678')}"
DOTTED_BAD = "12.345.678-0"


def at(text: str, value: str) -> tuple[int, int]:
    a = text.index(value)
    return a, a + len(value)


@pytest.mark.parametrize(
    "line, expected",
    [
        (f"Folio {BARE_BAD} del expediente", True),
        (f"RUT: {BARE_BAD}", False),
        (f"rut {BARE_BAD}", False),
        (f"R.U.T. {BARE_BAD}", False),
        (f"RUN {BARE_BAD}", False),
        (f"Rol Único Tributario {BARE_BAD}", False),
        (f"Ruta 5, kilómetro 12, código {BARE_BAD}", True),  # "Ruta" is not a label
        (f"RUT del contribuyente según el formulario adjunto número {BARE_BAD}", True),  # over 30 characters before
        (f"RUT:\nFolio {BARE_BAD}", True),  # a label on another line does not count
        (f"Folio {BARE_GOOD}", False),  # a valid one is a RUT
        (f"Folio {DOTTED_BAD}", False),  # written as a RUT: applied and doubtful
        ("Folio 12345678-0", False),
    ],
)
def test_which_ruts_are_only_suggested(line, expected):
    value = re.search(r"\d[\d.-]{7,}\d", line).group(0)
    assert rut_suggested(line, *at(line, value)) is expected


def test_a_rut_finding_holds_a_rut_shaped_number():
    assert context_type("rut", "", ()) is None
    assert context_type("rut", "—", ()) is None
    assert context_type("rut", "N° 5", ()) is None  # one digit
    assert context_type("rut", "ana.prueba@ejemplo.cl", ()) == "email"
    assert context_type("rut", DOTTED_BAD, ()) == "rut"
    assert context_type("rut", "l2.345.678-5", (), ocr=True) == "rut"  # an OCR "l" for a 1
    assert context_type("name", "Ana", ()) == "name"


def _pdf(path: Path, lines: list[str]) -> Path:
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        for i, line in enumerate(lines):
            page.insert_text((72, 100 + 22 * i), line, fontsize=11)
        doc.save(path)
    return path


def _ruts(file: AnalyzedFile) -> dict[str, tuple]:
    return {f.text: (f.status, f.optional, f.optional_reason, f.doubt_reason) for f in file.findings if f.type == "rut"}


def test_text_layer_ruts(tmp_path):
    path = _pdf(tmp_path / "a.pdf", [
        "Informe ficticio con datos inventados.",
        f"Folio {BARE_BAD} del expediente",
        f"RUT: 98765432{rut_check_digit('98765432')}",
        f"Contraparte RUT {BARE_BAD[:-1]}1",
        f"Segundo número {DOTTED_BAD} de la contraparte",
        f"Folio {BARE_GOOD} del expediente",
    ])  # fmt: skip
    file = AnalyzedFile(id="r", name="a.pdf", path=str(path))
    RealEngine().analyze(file, [])
    ruts = _ruts(file)
    assert ruts[BARE_BAD] == ("suggested", True, "rut", DOUBT_RUT)
    assert ruts[f"{BARE_BAD[:-1]}1"] == ("proposed", False, None, DOUBT_RUT)  # labelled: applied, doubtful
    assert ruts[DOTTED_BAD] == ("proposed", False, None, DOUBT_RUT)  # formatted: applied, doubtful
    assert ruts[BARE_GOOD] == ("proposed", False, None, None)


def test_a_bare_rut_in_a_rut_column_is_labelled(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        for x, label in ((72, "Nombre"), (250, "RUT"), (400, "Comuna")):
            page.insert_text((x, 100), label, fontsize=11)
        for x, value in ((72, "Persona Inventada"), (250, BARE_BAD), (400, "Concepcion")):
            page.insert_text((x, 122), value, fontsize=11)
        doc.save(tmp_path / "t.pdf")
    file = AnalyzedFile(id="t", name="t.pdf", path=str(tmp_path / "t.pdf"))
    RealEngine().analyze(file, [])
    assert _ruts(file)[BARE_BAD][:2] == ("proposed", False)


def test_an_empty_rut_field_is_no_finding(tmp_path):
    path = _pdf(tmp_path / "e.pdf", ["Formulario ficticio de prueba.", "RUT: -", "Nombre: Persona Inventada",
                                     "RUT: N° 5", "R.U.T.: ana.prueba@ejemplo.cl"])  # fmt: skip
    file = AnalyzedFile(id="e", name="e.pdf", path=str(path))
    RealEngine().analyze(file, [])
    assert [f for f in file.findings if f.type == "rut"] == []
    assert any(f.type == "email" for f in file.findings)


def test_a_reading_without_the_number_does_not_clear_a_rut_doubt():
    box = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], np.float64)
    zones = [
        Zone("rut", box, f"Folio {DOTTED_BAD}", "ocr", 0.95, DOUBT_RUT),
        Zone("rut", box + 0.5, "RUT:", "context", 0.95, None),  # another pass, without the number
    ]
    (merged,) = raster.dedup(zones)
    assert merged.doubt == DOUBT_RUT
    valid = Zone("rut", box + 0.5, f"Folio 12.345.678-{rut_check_digit('12345678')}", "ocr", 0.95, None)
    (merged,) = raster.dedup([zones[0], valid])
    assert merged.doubt is None  # a reading with a valid check digit settles it


@needs_ocr
def test_ocr_ruts(tmp_path):
    img = _text_image([f"Folio {BARE_BAD}", f"RUT {BARE_BAD[:-1]}1", "Texto neutro de la nota"])
    img.save(tmp_path / "n.png")
    file = AnalyzedFile(id="o", name="n.png", path=str(tmp_path / "n.png"))
    RealEngine().analyze(file, [])
    by_line = {f.text: f for f in file.findings if f.type == "rut"}
    folio = next(f for t, f in by_line.items() if "Folio" in t)
    labelled = next(f for t, f in by_line.items() if "RUT" in t.upper())
    assert (folio.status, folio.optional_reason, folio.doubtful) == ("suggested", "rut", True)
    assert (labelled.status, labelled.optional, labelled.doubtful) == ("proposed", False, True)
    file_all = AnalyzedFile(id="p", name="n.png", path=str(tmp_path / "n.png"), all_text=True)
    RealEngine().analyze(file_all, [])
    assert not any(f.optional for f in file_all.findings)  # "censurar todo el texto": nothing optional


def test_doubtful_ruts_can_be_applied_and_are_audited(tmp_path):
    from anonymizer.api.server import create_app
    from tests.live_client import LiveClient
    from tests.test_api import TOKEN, upload, wait_status

    path = _pdf(
        tmp_path / "a.pdf", ["Informe ficticio.", f"Folio {BARE_BAD} del expediente", f"Otro folio {BARE_BAD[:-1]}9"]
    )
    app = create_app(RealEngine(), TOKEN)
    with LiveClient(app) as client:
        client.headers["X-Session-Token"] = TOKEN
        file_id = upload(client, "a.pdf", path.read_bytes())
        client.post("/api/process", json={"file_ids": [file_id]})
        summary = wait_status(client, file_id)
        assert summary["counts"]["suggested_ruts"] == 2
        file = client.get(f"/api/files/{file_id}").json()
        first = next(f for f in file["findings"] if f["optional_reason"] == "rut")
        r = client.patch(f"/api/files/{file_id}/findings/{first['id']}", json={"action": "apply"})
        assert r.status_code == 200 and r.json()["status"] == "proposed"
        r = client.post(f"/api/files/{file_id}/findings/apply-optional", json={"reason": "rut"})
        assert len(r.json()["applied"]) == 1
    session_file = app.state.session.get(file_id)
    result = ExportResult(file_id=file_id, output_path=None, leaks=[], redactions_applied=0, removed_by_reviewer=0,
                          exported=False, message="")  # fmt: skip
    record = audit.build_report([session_file], [result])["files"][0]
    assert record["doubtful_ruts"]["applied"] == 2 and record["doubtful_ruts"]["left_visible"] == 0
    assert record["other_urls"]["items"] == []
    for f in session_file.findings:
        if f.optional_reason == "rut":
            f.status = "suggested"
    html = audit._file_html(session_file, result)
    assert "RUT dudosos sin puntos ni guion" in html and BARE_BAD in html


def test_the_review_shows_doubtful_ruts_with_the_other_suggestions():
    js = (Path(__file__).parents[1] / "anonymizer" / "ui" / "app.js").read_text(encoding="utf-8")
    for text in ("Censurar todos los RUT dudosos", "RUT dudoso sin puntos ni guion", "RUT dudosos: números sin puntos",
                 'applyAllOptional("rut")'):  # fmt: skip
        assert text in js
