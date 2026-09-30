"""El generador errores: cada archivo falla de la forma que dice su código de error esperado."""

import io
import zipfile

import pymupdf
import pytest
from PIL import Image, UnidentifiedImageError

from banco_pruebas.esquema import Manifiesto
from banco_pruebas.generadores import errores


@pytest.fixture
def generado(ctx):
    archivos = errores.generar(ctx)
    man = Manifiesto(raiz=str(ctx.raiz), semilla=33)
    for a in archivos:
        man.agregar(a)
    return ctx.raiz, man, {a.id: a for a in archivos}


def test_registros(generado):
    raiz, man, archivos = generado
    esperados = {
        "err_protegido_contrasena": ("errores/protegido_contrasena.pdf", "error:contrasena"),
        "err_corrupto": ("errores/corrupto.pdf", "error:corrupto"),
        "err_vacio": ("errores/vacio.pdf", "error:vacio"),
        "err_no_es_pdf": ("errores/no_es_pdf.pdf", "error:formato"),
        "err_imagen_corrupta": ("errores/imagen_corrupta.jpg", "error:corrupto"),
        "err_no_es_imagen": ("errores/no_es_imagen.png", "error:formato"),
        "err_documento_docx": ("errores/documento.docx", "error:formato"),
    }
    assert {i: (a.ruta, a.esperado) for i, a in archivos.items()} == esperados
    for a in archivos.values():
        assert (raiz / a.ruta).exists() and a.sha256
        assert a.categoria == "errores" and a.paginas == [] and a.elementos == [] and a.metadatos_sensibles == []
    assert man.validar() == []
    # El manifiesto con páginas vacías se guarda y se vuelve a cargar sin problemas.
    cargado = Manifiesto.cargar(man.guardar())
    assert [a.paginas for a in cargado.archivos] == [[]] * len(esperados)


def test_protegido_con_contrasena(generado):
    raiz, _, archivos = generado
    a = archivos["err_protegido_contrasena"]
    with pymupdf.open(raiz / a.ruta) as d:
        assert d.needs_pass
        assert d.authenticate(a.etiquetas["contrasena_usuario"])
        assert "AES" in d.metadata["encryption"] and "256" in d.metadata["encryption"]
        assert d[0].get_text().strip()


def test_pdf_corrupto(generado):
    raiz, _, archivos = generado
    a = archivos["err_corrupto"]
    datos = (raiz / a.ruta).read_bytes()
    assert datos.startswith(b"%PDF-") and b"%%EOF" not in datos and b"startxref" not in datos
    assert len(datos) == a.etiquetas["bytes_truncado"] == int(a.etiquetas["bytes_originales"] * 0.4)
    try:
        d = pymupdf.open(raiz / a.ruta)
    except (pymupdf.FileDataError, RuntimeError):
        return
    with d:
        assert d.is_repaired and d.page_count == 0


def test_vacio_y_no_es_pdf(generado):
    raiz, _, archivos = generado
    assert (raiz / archivos["err_vacio"].ruta).stat().st_size == 0
    no_pdf = raiz / archivos["err_no_es_pdf"].ruta
    assert not no_pdf.read_bytes().startswith(b"%PDF")
    no_pdf.read_bytes().decode("utf-8")
    for ruta in (raiz / archivos["err_vacio"].ruta, no_pdf):
        with pytest.raises((pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError)):
            pymupdf.open(ruta)


def test_imagenes_rotas(generado):
    raiz, _, archivos = generado
    jpg = (raiz / archivos["err_imagen_corrupta"].ruta).read_bytes()
    assert jpg.startswith(b"\xff\xd8") and not jpg.endswith(b"\xff\xd9")
    img = Image.open(io.BytesIO(jpg))
    assert img.format == "JPEG" and img.size == (900, 600)
    with pytest.raises(OSError, match="truncated"):
        img.load()
    with pytest.raises(UnidentifiedImageError):
        Image.open(raiz / archivos["err_no_es_imagen"].ruta)


def test_docx_con_basura(generado):
    raiz, _, archivos = generado
    a = archivos["err_documento_docx"]
    assert a.formato == "docx"
    with zipfile.ZipFile(raiz / a.ruta) as z:
        assert z.testzip() is None
        assert z.namelist() == ["[Content_Types].xml", "word/document.xml"]
        assert not z.read("word/document.xml").startswith(b"<?xml")


def test_determinista(ctx, tmp_path):
    from banco_pruebas.contexto import Contexto

    primero = {a.id: (ctx.raiz / a.ruta) for a in errores.generar(ctx)}
    otro = Contexto(raiz=tmp_path / "otro", semilla=33, fict=ctx.fict, rostros=ctx.rostros)
    segundo = {a.id: (otro.raiz / a.ruta) for a in errores.generar(otro)}
    for i, ruta in primero.items():
        if i == "err_protegido_contrasena":  # el cifrado AES usa sal e IV aleatorios
            continue
        assert ruta.read_bytes() == segundo[i].read_bytes(), i
