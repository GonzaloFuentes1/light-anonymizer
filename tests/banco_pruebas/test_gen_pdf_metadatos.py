"""El generador pdf_metadatos: la verdad de terreno calza con el PDF y cada estructura escondida existe."""

import re
import time
import warnings
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pypdfium2
import pytest

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Archivo, Manifiesto
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.generadores import pdf_metadatos
from banco_pruebas.lienzo import caja_envolvente
from banco_pruebas.rostros import ProveedorRostros

RAIZ_REPO = Path(__file__).resolve().parents[2]
ESCALA = 2.0


def _contexto(raiz: Path) -> Contexto:
    return Contexto(
        raiz=raiz,
        semilla=33,
        fict=Ficticios(33),
        rostros=ProveedorRostros(RAIZ_REPO / "datos_prueba" / "cache" / "rostros", 33, permitir_descarga=False),
    )


@pytest.fixture(scope="module")
def generado(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("pdfm") / "generado"
    ctx = _contexto(raiz)
    inicio = time.perf_counter()
    archivos = pdf_metadatos.generar(ctx)
    duracion = time.perf_counter() - inicio
    man = Manifiesto(raiz=str(raiz), semilla=33)
    for a in archivos:
        man.agregar(a)
    return raiz, man, {a.id: a for a in archivos}, duracion


def _ruta(raiz: Path, a: Archivo) -> Path:
    return raiz / a.ruta


def _flujos_descomprimidos(datos: bytes) -> list[bytes]:
    """Todos los flujos del archivo (de todas las revisiones), descomprimidos cuando se puede."""
    salida = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", datos, re.S):
        crudo = m.group(1)
        try:
            salida.append(zlib.decompress(crudo))
        except zlib.error:
            salida.append(crudo)
    return salida


def _caja(poligono) -> pymupdf.Rect:
    return pymupdf.Rect(*caja_envolvente(poligono))


def _mascara(poligono, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(poligono, np.float64) * ESCALA - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def test_archivos_ids_y_paginas(generado):
    raiz, man, archivos, _ = generado
    assert set(archivos) == {
        "pdfm_metadatos_completos",
        "pdfm_revision_incremental",
        "pdfm_redaccion_falsa",
        "pdfm_solo_permisos",
    }
    assert man.validar() == []
    for a in archivos.values():
        assert a.categoria == "pdf_metadatos" and a.formato == "pdf" and a.esperado == "procesar"
        assert a.ruta.startswith("pdf_metadatos/") and a.sha256
        with pymupdf.open(_ruta(raiz, a)) as d:
            assert not d.needs_pass
            assert d.page_count == len(a.paginas) == 1
            p = d[0]
            assert (a.paginas[0].ancho, a.paginas[0].alto, a.paginas[0].rotacion) == (
                p.cropbox.width,
                p.cropbox.height,
                p.rotation,
            )


def test_poligonos_dentro_de_la_pagina(generado):
    _, _, archivos, _ = generado
    for a in archivos.values():
        pag = a.paginas[0]
        for e in a.elementos:
            assert e.poligono is not None and len(e.poligono) == 4
            if e.capa == "oculto":
                continue
            x0, y0, x1, y1 = caja_envolvente(e.poligono)
            assert -0.5 <= x0 < x1 <= pag.ancho + 0.5, (a.id, e.valor)
            assert -0.5 <= y0 < y1 <= pag.alto + 0.5, (a.id, e.valor)


def test_valores_de_la_capa_de_texto_se_encuentran_en_su_lugar(generado):
    raiz, _, archivos, _ = generado
    for a in archivos.values():
        with pymupdf.open(_ruta(raiz, a)) as d:
            p = d[0]
            for e in a.elementos:
                if e.capa != "texto":
                    continue
                caja = _caja(e.poligono)
                encontrados = [q.rect for q in p.search_for(e.valor, quads=True)]
                assert any(abs(r.x0 - caja.x0) < 0.5 and abs(r.x1 - caja.x1) < 0.5 for r in encontrados), (
                    a.id,
                    e.valor,
                )
                assert any(abs(r.y0 - caja.y0) < 0.5 and abs(r.y1 - caja.y1) < 0.5 for r in encontrados)


def test_poligonos_visibles_tienen_tinta(generado):
    """Texto normal: píxeles no uniformes. Texto bajo un tapado negro: el polígono se ve negro."""
    raiz, _, archivos, _ = generado
    for a in archivos.values():
        with pymupdf.open(_ruta(raiz, a)) as d:
            pix = d[0].get_pixmap(matrix=pymupdf.Matrix(ESCALA, ESCALA), alpha=False)
        gris = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).mean(axis=2)
        for e in a.elementos:
            if e.capa == "oculto":
                continue
            valores = gris[_mascara(e.poligono, gris.shape)]
            assert valores.size > 20, (a.id, e.valor)
            if e.etiquetas.get("tapado") in ("rectangulo_dibujado", "anotacion_cuadrada"):
                assert valores.mean() < 40, (a.id, e.valor)
            else:
                assert valores.min() < 110 and valores.std() > 20, (a.id, e.valor)


def test_conteos_por_tipo(generado):
    _, _, archivos, _ = generado
    conteo = {i: Counter(e.tipo for e in a.elementos) for i, a in archivos.items()}
    completos = conteo["pdfm_metadatos_completos"]
    assert completos["nombre"] == 2 and completos["rut"] == 2 and completos["correo"] == 3
    assert completos["telefono"] == 1 and completos["direccion"] == 1 and completos["texto"] >= 20
    assert conteo["pdfm_revision_incremental"]["rut"] == 1 and conteo["pdfm_revision_incremental"]["nombre"] == 2
    falsa = conteo["pdfm_redaccion_falsa"]
    assert (falsa["nombre"], falsa["rut"], falsa["correo"], falsa["telefono"]) == (2, 1, 1, 1)
    permisos = conteo["pdfm_solo_permisos"]
    assert (permisos["nombre"], permisos["rut"], permisos["direccion"], permisos["telefono"]) == (2, 2, 2, 2)
    todos = [e for a in archivos.values() for e in a.elementos]
    assert any(e.nivel == "fuera_de_alcance" and e.tipo == "nombre" for e in todos)
    assert all(e.capa == "texto" for e in todos if e.capa != "oculto")
    ocultos = Counter(e.tipo for e in todos if e.capa == "oculto")
    assert ocultos == {"rut": 1, "texto": 1}  # el RUT de la capa apagada y su etiqueta
    assert sum(e.tipo == "texto" and e.etiquetas.get("estructura") == "anotacion_freetext" for e in todos) == 1
    senuelos = Counter(e.etiquetas.get("senuelo") for e in todos if e.tipo == "texto")
    assert senuelos["monto"] == 3 and senuelos["fecha"] == 4


def test_canarios_de_metadatos_unicos(generado):
    _, _, archivos, _ = generado
    valores = [m.valor for a in archivos.values() for m in a.metadatos_sensibles]
    assert all(valores) and len(valores) == len(set(valores)) == 17


def test_metadatos_completos_estructuras(generado):
    raiz, _, archivos, _ = generado
    a = archivos["pdfm_metadatos_completos"]
    ruta = _ruta(raiz, a)
    datos = ruta.read_bytes()
    flujos = _flujos_descomprimidos(datos)
    por_donde: dict[str, list] = {}
    for m in a.metadatos_sensibles:
        por_donde.setdefault(m.donde, []).append(m)
    assert set(por_donde) == {
        "pdf.info.author",
        "pdf.info.title",
        "pdf.info.subject",
        "pdf.info.keywords",
        "pdf.info.creator",
        "pdf.xmp",
        "pdf.anotacion",
        "pdf.adjunto",
        "pdf.ocg",
        "pdf.formulario",
        "pdf.marcador",
        "pdf.javascript",
    }
    with pymupdf.open(ruta) as d:
        info = d.metadata
        for campo in ("author", "title", "subject", "keywords", "creator"):
            (m,) = por_donde[f"pdf.info.{campo}"]
            assert m.valor in info[campo]
        assert info["creator"].startswith("Microsoft Word - informe_") and info["creator"].endswith(".docx")

        xmp = d.get_xml_metadata()
        assert "<dc:creator>" in xmp and "gore:rutFuncionario" in xmp
        assert all(m.valor in xmp for m in por_donde["pdf.xmp"])

        p = d[0]
        anotaciones = {x.type[1]: x for x in p.annots()}
        assert {"Text", "FreeText", "Highlight", "FileAttachment"} <= set(anotaciones)
        contenidos = {k: x.info["content"] for k, x in anotaciones.items()}
        for m in por_donde["pdf.anotacion"]:
            assert m.valor in contenidos[m.etiquetas["anotacion"]]
        assert anotaciones["Highlight"].popup_xref > 0

        assert d.embfile_count() == 1 and d.embfile_names() == ["datos_contacto.txt"]
        embebido = d.embfile_get(0).decode()
        archivo_anot = anotaciones["FileAttachment"].get_file().decode()
        for m in por_donde["pdf.adjunto"]:
            assert m.valor in (embebido if m.etiquetas["forma"] == "embfile" else archivo_anot)

        ocgs = {v["name"]: v for v in d.get_ocgs().values()}
        assert "Notas internas" in ocgs and ocgs["Notas internas"]["on"] is False
        (m_ocg,) = por_donde["pdf.ocg"]
        (oculto,) = [e for e in a.elementos if e.capa == "oculto" and e.tipo != "texto"]
        assert oculto.valor == m_ocg.valor and oculto.tipo == "rut"
        assert m_ocg.valor not in p.get_text()  # la capa apagada no se ve ni se extrae con PyMuPDF
        assert any(m_ocg.valor.encode() in f for f in flujos)  # pero está en el flujo, como literal

        # Con la capa encendida, search_for encuentra cada texto oculto justo en su polígono.
        d.xref_set_key(d.pdf_catalog(), "OCProperties/D/OFF", "[]")
        with pymupdf.open("pdf", d.tobytes()) as encendida:
            p_on = encendida[0]
            for e in (e for e in a.elementos if e.capa == "oculto"):
                (r,) = p_on.search_for(e.valor)
                caja = _caja(e.poligono)
                assert max(abs(r.x0 - caja.x0), abs(r.x1 - caja.x1), abs(r.y0 - caja.y0), abs(r.y1 - caja.y1)) < 0.5

        widgets = list(p.widgets())
        (m_form,) = por_donde["pdf.formulario"]
        assert [(w.field_name, w.field_value) for w in widgets] == [("correo_notificacion", m_form.valor)]

        (m_toc,) = por_donde["pdf.marcador"]
        assert any(m_toc.valor in titulo for _, titulo, _ in d.get_toc())

        (m_js,) = por_donde["pdf.javascript"]
        nombres = d.xref_get_key(d.pdf_catalog(), "Names/JavaScript")
        assert nombres[0] == "dict" and "(contacto)" in nombres[1]
        assert f'"{m_js.valor}"'.encode() in datos

    # pdfium sí extrae el texto de la capa apagada: el canario es detectable por extracción de texto.
    pdf = pypdfium2.PdfDocument(str(ruta))
    try:
        assert m_ocg.valor in pdf[0].get_textpage().get_text_range()
    finally:
        pdf.close()


def test_revision_incremental(generado):
    raiz, _, archivos, _ = generado
    a = archivos["pdfm_revision_incremental"]
    datos = _ruta(raiz, a).read_bytes()
    (m,) = a.metadatos_sensibles
    assert m.donde == "pdf.revision_anterior"
    assert datos.count(b"%%EOF") == 2 and datos.count(b"startxref") == 2
    with pymupdf.open(_ruta(raiz, a)) as d:
        actual = d[0].get_text()
    assert m.valor not in actual
    assert all(e.valor in actual for e in a.elementos)
    # La revisión 1 (hasta el primer %%EOF) es un PDF válido y todavía contiene la línea borrada.
    fin_rev1 = datos.index(b"%%EOF") + len(b"%%EOF")
    with pymupdf.open(stream=datos[:fin_rev1], filetype="pdf") as viejo:
        assert m.etiquetas["linea"] in viejo[0].get_text()
    # Y se recupera de los bytes (flujos descomprimidos) como cadena literal.
    assert any(m.valor.encode() in f for f in _flujos_descomprimidos(datos))


def test_redaccion_falsa(generado):
    raiz, _, archivos, _ = generado
    a = archivos["pdfm_redaccion_falsa"]
    tapados = {e.etiquetas["tapado"]: e for e in a.elementos if "tapado" in e.etiquetas}
    assert set(Counter(e.etiquetas["tapado"] for e in a.elementos if "tapado" in e.etiquetas).items()) == {
        ("rectangulo_dibujado", 2),
        ("redact_sin_aplicar", 1),
        ("anotacion_cuadrada", 1),
    }
    assert all(e.capa == "texto" and e.nivel == "base" for e in a.elementos if "tapado" in e.etiquetas)
    with pymupdf.open(_ruta(raiz, a)) as d:
        p = d[0]
        texto = p.get_text()
        for e in a.elementos:
            assert e.valor in texto  # todo sigue siendo extraíble
        negros = [dr["rect"] for dr in p.get_drawings() if dr.get("fill") == (0.0, 0.0, 0.0)]
        for e in a.elementos:
            if e.etiquetas.get("tapado") == "rectangulo_dibujado":
                assert any(r.contains(_caja(e.poligono)) for r in negros), e.valor
        anotaciones = {x.type[1]: x for x in p.annots()}
        assert anotaciones["Redact"].rect.contains(_caja(tapados["redact_sin_aplicar"].poligono))
        cuadro = anotaciones["Square"]
        assert cuadro.colors["fill"] == [0.0, 0.0, 0.0]
        assert cuadro.rect.contains(_caja(tapados["anotacion_cuadrada"].poligono))


def test_solo_permisos(generado):
    raiz, _, archivos, _ = generado
    a = archivos["pdfm_solo_permisos"]
    assert _ruta(raiz, a).read_bytes().count(b"/Encrypt") >= 1
    with pymupdf.open(_ruta(raiz, a)) as d:
        assert not d.needs_pass
        assert "AES" in d.metadata["encryption"] and "256" in d.metadata["encryption"]
        assert not d.permissions & pymupdf.PDF_PERM_COPY
        assert not d.permissions & pymupdf.PDF_PERM_PRINT
        texto = d[0].get_text()
    assert all(e.valor in texto for e in a.elementos)


def test_tiempo_y_determinismo(generado, tmp_path):
    _, _, archivos, duracion = generado
    # El tiempo depende de la carga del equipo: se avisa, no se falla (los tiempos se reportan aparte).
    if duracion > 90:
        warnings.warn(f"generación lenta: {duracion:.1f} s", stacklevel=1)
    otra = pdf_metadatos.generar(_contexto(tmp_path / "otra"))

    def sin_hash(a: Archivo) -> dict:
        datos = asdict(a)
        datos.pop("sha256")
        for e in datos["elementos"]:
            e.pop("id")
        for m in datos["metadatos_sensibles"]:
            m.pop("id")
        return datos

    assert [sin_hash(a) for a in otra] == [sin_hash(a) for a in archivos.values()]

    # Bytes idénticos (sin /ID aleatorio ni fechas de la máquina), salvo el archivo cifrado con
    # AES, cuya sal y vector de inicialización son aleatorios: ahí se compara el texto y el sha256 cambia.
    raiz, _, _, _ = generado
    for a in otra:
        viejo, nuevo = (raiz / a.ruta).read_bytes(), (tmp_path / "otra" / a.ruta).read_bytes()
        if a.etiquetas.get("cifrado"):
            with pymupdf.open(stream=viejo) as d1, pymupdf.open(stream=nuevo) as d2:
                assert [p.get_text() for p in d1] == [p.get_text() for p in d2]
        else:
            assert viejo == nuevo, a.id
            assert b"D:2026" not in viejo or all(
                f in (pdf_metadatos.FECHA_CREACION[:16].encode(), pdf_metadatos.FECHA_MODIFICACION[:16].encode())
                for f in re.findall(rb"D:\d{14}", viejo)
            ), a.id
