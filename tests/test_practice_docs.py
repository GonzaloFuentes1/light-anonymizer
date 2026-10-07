"""The practice documents of the user manual and the usability test (scripts/generate_practice_docs.py)
still show what the tasks need. Invented data only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from anonymizer.engine import faces
from anonymizer.engine.model import AnalyzedFile
from anonymizer.engine.real import RealEngine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_practice_docs as practice  # noqa: E402

needs_ocr = pytest.mark.skipif(
    importlib.util.find_spec("rapidocr") is None or not faces.available(), reason="needs the OCR and face models"
)


@pytest.fixture(scope="module")
def docs(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("practice")
    assert practice.main(["--output", str(folder)]) == 0
    return folder


def analyzed(path: Path) -> AnalyzedFile:
    file = AnalyzedFile(id="f1", name=path.name, path=str(path))
    RealEngine().analyze(file, [])
    assert file.status == "ready", file.error_message
    return file


def test_same_bytes_on_every_run(docs, tmp_path):
    assert practice.main(["--output", str(tmp_path)]) == 0
    for name in ("acta_con_timbre.pdf", "oficio_fotografiado.jpg"):
        assert (tmp_path / name).read_bytes() == (docs / name).read_bytes()


def test_acta_has_a_doubtful_rut_another_link_and_a_page_exported_as_an_image(docs):
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="acta_con_timbre.pdf", path=str(docs / "acta_con_timbre.pdf"))
    engine.analyze(file, [])
    suggested = {f.optional_reason for f in file.findings if f.status == "suggested"}
    assert suggested == {"rut", "url"}
    assert any(f.type == "name" and f.doubtful for f in file.findings)
    info: dict = {}
    engine.render_result(file, 0, 1.0, [f for f in file.findings if f.active], info)
    assert info.get("as_image"), "the stamp over the signer's name no longer makes the page an image"


@needs_ocr
def test_the_photographed_letter_has_a_signature_the_app_misses(docs):
    file = analyzed(docs / "oficio_fotografiado.jpg")
    types = {f.type for f in file.findings}
    assert "signature" not in types, "the app now finds this signature: the usability task needs another"
    assert {"name", "email"} <= types
