"""Static checks of the desktop UI (anonymizer/ui): local assets only, valid JavaScript."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "anonymizer" / "ui"
REMOTE = re.compile(r"(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)


def read(name: str) -> str:
    return (UI / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["index.html", "app.css", "app.js"])
def test_no_remote_urls(name):
    text = read(name)
    assert "http://" not in text and "https://" not in text, f"{name} references a remote URL"
    assert not REMOTE.search(text), f"{name} references a protocol-relative remote URL"


def test_index_references_only_existing_local_assets():
    html = read("index.html")
    refs = re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", html)
    assert "app.css" in refs and "app.js" in refs
    for ref in refs:
        if ref.startswith(("data:", "#")):
            continue
        assert not ref.startswith(("/", "\\")) and ":" not in ref, f"asset must be relative and local: {ref}"
        assert (UI / ref).is_file(), f"missing asset: {ref}"


def test_index_has_session_token_meta_and_csp():
    html = read("index.html")
    assert re.search(r"""<meta\s+name=["']session-token["']""", html)
    assert "Content-Security-Policy" in html and "connect-src 'self'" in html


def test_fonts_are_bundled_with_their_licenses():
    css = read("app.css")
    urls = re.findall(r"""url\(["']?([^"')]+)["']?\)""", css)
    fonts = [u for u in urls if u.startswith("fonts/")]
    assert len(fonts) == 5, fonts
    for url in fonts:
        path = UI / url
        assert path.is_file() and path.stat().st_size > 10_000, f"missing font file: {url}"
    for license_file in ("OFL-AtkinsonHyperlegible.txt", "OFL-IBMPlexMono.txt"):
        text = (UI / "fonts" / license_file).read_text(encoding="utf-8")
        assert "SIL Open Font License" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_app_js_syntax():
    result = subprocess.run(
        ["node", "--check", str(UI / "app.js")], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name,strings",
    [
        (
            "index.html",
            [
                "Qué buscar",
                "Acerca de",
                "Copiar enlace",
                "Ver licencia completa",
                "Software libre: puedes usarlo, estudiarlo, modificarlo y compartirlo según la licencia GNU AGPL v3 "
                "o posterior. Se entrega sin ninguna garantía.",
                "Funciona sin conexión: no envía tus documentos ni datos a ningún lado.",
            ],
        ),
        (
            "app.js",
            [
                "Tiempo estimado: ",
                "depende del computador",
                "Apagaste ",
                "menos de 1 s",
                "Listo en ",
                "En este archivo no se buscaron: ",
                "Revisa esas partes a mano.",
                "En estos archivos no se buscó todo.",
                "Otros enlaces (sin censurar)",
                "Censurar todos los otros enlaces",
                "No censurar",
                "sin censurar",
                "Las fotos HEIC (por ejemplo de iPhone) todavía no se pueden abrir.",
            ],
        ),
        ("app.css", [".zone.suggested", ".result .zone.suggested", ".detect", ".license", "td .warntxt"]),
    ],
)
def test_new_spanish_strings_are_present(name, strings):
    text = read(name)
    for s in strings:
        assert s in text, f"{name} is missing {s!r}"


def test_the_source_url_comes_from_the_api():
    # The UI may not hardcode external addresses (the CSP blocks them anyway): GET /api/about serves it.
    for name in ("index.html", "app.js"):
        assert "github" not in read(name).lower()
