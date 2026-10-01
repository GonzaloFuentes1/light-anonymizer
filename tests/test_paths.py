"""Resource paths in development and inside the packaged executable (sys.frozen / sys._MEIPASS)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from anonymizer import paths
from anonymizer.api import server
from anonymizer.app import write_url_file
from anonymizer.engine import faces, ocr

ROOT = Path(__file__).resolve().parents[1]


def test_development_paths_point_into_the_repository():
    assert not paths.is_frozen()
    assert paths.resource_root() == ROOT
    assert server.UI_DIR == ROOT / "anonymizer" / "ui"
    assert (server.UI_DIR / "index.html").is_file()
    # The license texts an "Acerca de" screen can show (bundled at the same relative paths).
    assert paths.resource("LICENSE").is_file() and paths.resource("LICENSES.md").is_file()
    if "ANONYMIZER_MODELS_DIR" not in os.environ:
        assert faces.MODELS_DIR == ROOT / "models"
    model_paths = ocr.model_paths()
    assert model_paths is not None
    assert all(path.parent.parent.name == "rapidocr" for path in model_paths.values())


def test_frozen_paths_point_into_the_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.is_frozen()
    assert paths.resource_root() == tmp_path
    assert paths.resource("anonymizer", "ui") == tmp_path / "anonymizer" / "ui"
    assert paths.package_dir("rapidocr") == tmp_path / "rapidocr"
    assert paths.package_dir("no_such_package_for_this_test") is None
    model_paths = ocr.model_paths()
    assert model_paths is not None
    assert {path.parent for path in model_paths.values()} == {tmp_path / "rapidocr" / "models"}
    assert not ocr.available()  # the models are not in this fake bundle


def test_url_file_has_url_token_and_pid(tmp_path):
    target = tmp_path / "sub" / "url.json"
    write_url_file(target, "http://127.0.0.1:1234/?k=abc", "secret")
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "url": "http://127.0.0.1:1234/?k=abc",
        "token": "secret",
        "pid": os.getpid(),
    }
    assert [p.name for p in target.parent.iterdir()] == ["url.json"]  # no partial file left
