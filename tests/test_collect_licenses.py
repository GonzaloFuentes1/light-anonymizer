"""THIRD_PARTY_LICENSES/ of the executable (scripts/collect_licenses.py), on a fake bundle folder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_collector():
    spec = importlib.util.spec_from_file_location("collect_licenses", ROOT / "scripts" / "collect_licenses.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("collect_licenses", module)
    spec.loader.exec_module(module)
    return module


def fake_bundle(tmp_path: Path, modules: list[str]) -> tuple[Path, Path]:
    app = tmp_path / "LightAnonymizer"
    internal = app / "_internal"
    for folder in ("numpy", "Shapely.libs", "anonymizer", "models"):
        (internal / folder).mkdir(parents=True)
    (internal / "_cffi_backend.cp312-win_amd64.pyd").write_bytes(b"")
    (internal / "python312.dll").write_bytes(b"")
    toc = tmp_path / "PYZ-00.toc"
    toc.write_text(repr(("PYZ-00.pyz", [(m, None, "PYMODULE") for m in modules])), encoding="utf-8")
    return app, toc


def test_every_bundled_package_gets_its_license_texts(tmp_path):
    collector = load_collector()
    app, toc = fake_bundle(tmp_path, ["fastapi", "fastapi.routing", "rapidocr", "json", "anonymizer.app"])
    assert collector.collect(app, toc) == []
    out = app / "THIRD_PARTY_LICENSES"
    folders = {p.name.rsplit("-", 1)[0] for p in out.iterdir() if p.is_dir()}
    assert {"numpy", "shapely", "cffi", "fastapi", "rapidocr", "python", "models", "fonts"} <= folders
    assert "light-anonymizer" not in folders  # the project itself is covered by LICENSE.txt
    assert "pymupdf" not in folders  # not in this fake bundle
    shapely = next(out.glob("shapely-*"))
    assert (shapely / "LICENSE_GEOS").is_file()  # LGPL text of the bundled GEOS
    rapidocr = next(out.glob("rapidocr-*"))
    assert (rapidocr / "LICENSE").is_file()  # from packaging/licenses: the wheel has none
    index = (out / "INDEX.txt").read_text(encoding="utf-8")
    assert "fastapi" in index and "shapely" in index and "LGPL-2.1" in index
    assert (next(out.glob("python-*")) / "LICENSE.txt").is_file()


def test_a_package_without_any_license_text_fails_the_build(tmp_path, monkeypatch):
    collector = load_collector()
    app, toc = fake_bundle(tmp_path, ["fastapi"])
    monkeypatch.setattr(collector, "license_files", lambda dist: [])
    monkeypatch.setattr(collector, "VENDORED", tmp_path / "no_vendored_texts")
    problems = collector.collect(app, toc)
    assert any(p.startswith("fastapi ") for p in problems)
