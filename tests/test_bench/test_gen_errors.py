"""The errors generator: each file fails the way its expected error code says."""

import io
import zipfile

import pymupdf
import pytest
from PIL import Image, UnidentifiedImageError

from test_bench.generators import errors
from test_bench.schema import Manifest


@pytest.fixture
def generated(ctx):
    files = errors.generate(ctx)
    man = Manifest(root=str(ctx.root), seed=33)
    for a in files:
        man.add(a)
    return ctx.root, man, {a.id: a for a in files}


def test_entries(generated):
    root, man, files = generated
    expected = {
        "err_protegido_contrasena": ("errors/protegido_contrasena.pdf", "error:password"),
        "err_corrupto": ("errors/corrupto.pdf", "error:corrupt"),
        "err_vacio": ("errors/vacio.pdf", "error:empty"),
        "err_no_es_pdf": ("errors/no_es_pdf.pdf", "error:format"),
        "err_imagen_corrupta": ("errors/imagen_corrupta.jpg", "error:corrupt"),
        "err_no_es_imagen": ("errors/no_es_imagen.png", "error:format"),
        "err_documento_docx": ("errors/documento.docx", "error:format"),
    }
    assert {i: (a.path, a.expected) for i, a in files.items()} == expected
    for a in files.values():
        assert (root / a.path).exists() and a.sha256
        assert a.category == "errors" and a.pages == [] and a.elements == [] and a.sensitive_metadata == []
    assert man.validate() == []
    # The manifest with empty pages is saved and loaded again without problems.
    loaded = Manifest.load(man.save())
    assert [a.pages for a in loaded.files] == [[]] * len(expected)


def test_password_protected(generated):
    root, _, files = generated
    a = files["err_protegido_contrasena"]
    with pymupdf.open(root / a.path) as d:
        assert d.needs_pass
        assert d.authenticate(a.tags["user_password"])
        assert "AES" in d.metadata["encryption"] and "256" in d.metadata["encryption"]
        assert d[0].get_text().strip()


def test_corrupt_pdf(generated):
    root, _, files = generated
    a = files["err_corrupto"]
    data = (root / a.path).read_bytes()
    assert data.startswith(b"%PDF-") and b"%%EOF" not in data and b"startxref" not in data
    assert len(data) == a.tags["truncated_bytes"] == int(a.tags["original_bytes"] * 0.4)
    try:
        d = pymupdf.open(root / a.path)
    except (pymupdf.FileDataError, RuntimeError):
        return
    with d:
        assert d.is_repaired and d.page_count == 0


def test_empty_and_not_pdf(generated):
    root, _, files = generated
    assert (root / files["err_vacio"].path).stat().st_size == 0
    not_pdf = root / files["err_no_es_pdf"].path
    assert not not_pdf.read_bytes().startswith(b"%PDF")
    not_pdf.read_bytes().decode("utf-8")
    for path in (root / files["err_vacio"].path, not_pdf):
        with pytest.raises((pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError)):
            pymupdf.open(path)


def test_broken_images(generated):
    root, _, files = generated
    jpg = (root / files["err_imagen_corrupta"].path).read_bytes()
    assert jpg.startswith(b"\xff\xd8") and not jpg.endswith(b"\xff\xd9")
    img = Image.open(io.BytesIO(jpg))
    assert img.format == "JPEG" and img.size == (900, 600)
    with pytest.raises(OSError, match="truncated"):
        img.load()
    with pytest.raises(UnidentifiedImageError):
        Image.open(root / files["err_no_es_imagen"].path)


def test_docx_with_garbage(generated):
    root, _, files = generated
    a = files["err_documento_docx"]
    assert a.format == "docx"
    with zipfile.ZipFile(root / a.path) as z:
        assert z.testzip() is None
        assert z.namelist() == ["[Content_Types].xml", "word/document.xml"]
        assert not z.read("word/document.xml").startswith(b"<?xml")


def test_deterministic(ctx, tmp_path):
    from test_bench.context import Context

    first = {a.id: (ctx.root / a.path) for a in errors.generate(ctx)}
    other = Context(root=tmp_path / "other", seed=33, fake=ctx.fake, faces=ctx.faces)
    second = {a.id: (other.root / a.path) for a in errors.generate(other)}
    for i, path in first.items():
        if i == "err_protegido_contrasena":  # AES encryption uses a random salt and IV
            continue
        assert path.read_bytes() == second[i].read_bytes(), i
