"""Invariantes del generador de PDF escaneados: la verdad de terreno tiene que calzar con cada página."""

from __future__ import annotations

import io
import math
import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Archivo, Manifiesto
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.generadores import pdf_escaneado
from banco_pruebas.lienzo import caja_envolvente
from banco_pruebas.rostros import ProveedorRostros

RAIZ_REPO = Path(__file__).resolve().parents[2]
ESCALA = 2.0  # 144 ppp para revisar píxeles


@pytest.fixture(scope="module")
def generado(tmp_path_factory: pytest.TempPathFactory) -> tuple[Contexto, list[Archivo], float]:
    ctx = _contexto(tmp_path_factory.mktemp("pdf_escaneado") / "generado")
    inicio = time.perf_counter()
    archivos = pdf_escaneado.generar(ctx)
    return ctx, archivos, time.perf_counter() - inicio


def _por_id(archivos: list[Archivo], id: str) -> Archivo:
    return next(a for a in archivos if a.id == "pdfe_" + id)


def _mascara(poligono, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(poligono, np.float64) * ESCALA - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def _centro(poligono) -> tuple[float, float]:
    x0, y0, x1, y1 = caja_envolvente(poligono)
    return (x0 + x1) / 2, (y0 + y1) / 2


def test_archivos_existen_y_abren(generado):
    ctx, archivos, dt = generado
    assert len(archivos) == 8
    assert len({a.id for a in archivos}) == 8
    # El tiempo depende de la carga del equipo: se avisa, no se falla (los tiempos se reportan aparte).
    if dt > 90:
        warnings.warn(f"generación lenta: {dt:.1f} s", stacklevel=1)
    for a in archivos:
        assert a.id.startswith("pdfe_") and a.categoria == "pdf_escaneado" and a.formato == "pdf"
        ruta = ctx.raiz / a.ruta
        assert ruta.exists() and a.ruta.startswith("pdf_escaneado/")
        with pymupdf.open(ruta) as doc:
            assert doc.page_count == len(a.paginas)
            for pag, page in zip(a.paginas, doc, strict=True):
                assert (pag.ancho, pag.alto, pag.rotacion) == (page.rect.width, page.rect.height, page.rotation)
                assert (pag.ancho, pag.alto) == (595.0, 842.0) and pag.unidad == "pt"


def test_manifiesto_valido(generado):
    ctx, archivos, _ = generado
    man = Manifiesto(raiz=str(ctx.raiz), semilla=33)
    for a in archivos:
        man.agregar(a)
    assert man.validar() == []


def test_poligonos_dentro_de_la_pagina(generado):
    _, archivos, _ = generado
    for a in archivos:
        for e in a.elementos:
            assert e.poligono is not None and len(e.poligono) == 4
            if e.capa == "oculto":
                continue
            pag = a.paginas[e.pagina]
            x0, y0, x1, y1 = caja_envolvente(e.poligono)
            assert x0 >= -0.5 and y0 >= -0.5 and x1 <= pag.ancho + 0.5 and y1 <= pag.alto + 0.5, (a.id, e)
            if e.nucleo:
                nx0, ny0, nx1, ny1 = caja_envolvente(e.nucleo)
                assert x0 <= nx0 and y0 <= ny0 and nx1 <= x1 and ny1 <= y1


def test_paginas_escaneadas_sin_texto(generado):
    """Las páginas escaneadas no tienen capa de texto (salvo la OCR invisible del sándwich) y son una imagen."""
    ctx, archivos, _ = generado
    for a in archivos:
        with pymupdf.open(ctx.raiz / a.ruta) as doc:
            for i, page in enumerate(doc):
                capas = {e.capa for e in a.elementos if e.pagina == i}
                if "texto" in capas:
                    assert not page.get_images()
                    continue
                assert capas <= {"raster", "oculto"}
                info = page.get_image_info()
                assert len(info) == 1 and pymupdf.Rect(info[0]["bbox"]) == page.rect
                if "oculto" in capas:
                    traza = page.get_texttrace()
                    assert traza and {s["type"] for s in traza} == {3}, "la capa OCR debe ser invisible"
                else:
                    assert page.get_text().strip() == ""


def test_capa_texto_y_oculta_se_encuentran_con_search_for(generado):
    ctx, archivos, _ = generado
    revisados = Counter()
    for a in archivos:
        with pymupdf.open(ctx.raiz / a.ruta) as doc:
            for e in a.elementos:
                if e.capa not in ("texto", "oculto"):
                    continue
                hits = doc[e.pagina].search_for(e.valor, quads=True)
                assert hits, (a.id, e.valor)
                d = min(
                    math.dist(_centro([[p.x, p.y] for p in (q.ul, q.ur, q.lr, q.ll)]), _centro(e.poligono))
                    for q in hits
                )
                assert d < 0.01, (a.id, e.valor, d)
                revisados[e.capa] += 1
    assert revisados["texto"] >= 20 and revisados["oculto"] >= 30


def test_sandwich_registra_cada_dato_dos_veces(generado):
    _, archivos, _ = generado
    a = _por_id(archivos, "sandwich_ocr")
    raster = [e for e in a.elementos if e.capa == "raster"]
    ocultos = [e for e in a.elementos if e.capa == "oculto"]
    assert len(raster) == len(ocultos)
    for r, o in zip(raster, ocultos, strict=True):
        assert (r.tipo, r.valor, r.nivel) == (o.tipo, o.valor, o.nivel)
        assert r.etiquetas["sandwich"] and o.etiquetas["sandwich"]
        # la capa invisible queda encima del texto de la imagen (como la deja el OCR del escáner)
        assert math.dist(_centro(r.poligono), _centro(o.poligono)) < 4, (r.valor, r.poligono, o.poligono)
    assert {e.tipo for e in ocultos} >= {"rut", "correo", "telefono", "nombre", "direccion"}


def test_poligonos_raster_tienen_tinta(generado):
    ctx, archivos, _ = generado
    for a in archivos:
        with pymupdf.open(ctx.raiz / a.ruta) as doc:
            for i, page in enumerate(doc):
                elementos = [e for e in a.elementos if e.pagina == i and e.capa == "raster"]
                if not elementos:
                    continue
                pix = page.get_pixmap(matrix=pymupdf.Matrix(ESCALA, ESCALA), colorspace=pymupdf.csGRAY)
                gris = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                for e in elementos:
                    valores = gris[_mascara(e.poligono, gris.shape)]
                    assert valores.size > 0, (a.id, e.valor)
                    if e.tipo == "rostro":
                        assert valores.std() > 15
                        continue
                    fondo = float(np.percentile(valores, 95))
                    contraste = fondo - float(np.percentile(valores, 3))
                    assert contraste > 60, (a.id, e.tipo, e.valor, contraste)
                    # el polígono no es solo fondo: una fracción visible es tinta
                    assert (valores < fondo - 50).mean() > 0.04, (a.id, e.tipo, e.valor)


def test_conteos_y_niveles(generado):
    _, archivos, _ = generado
    elementos = [e for a in archivos for e in a.elementos]
    tipos = Counter(e.tipo for e in elementos)
    assert tipos["rut"] >= 20 and tipos["telefono"] >= 20 and tipos["correo"] >= 15
    assert tipos["nombre"] >= 18 and tipos["direccion"] >= 6 and tipos["firma"] >= 3
    assert tipos["rostro"] == 1 and tipos["texto"] >= 200
    senuelos = Counter(e.etiquetas.get("senuelo") for e in elementos if e.tipo == "texto")
    assert senuelos["monto"] >= 8 and senuelos["fecha"] >= 10
    for e in elementos:
        if e.tipo in ("nombre", "direccion") and not e.etiquetas["en_lista"]:
            assert e.nivel == "fuera_de_alcance"
        if e.tipo == "firma":
            assert e.nivel == "fuera_de_alcance"
        if e.tipo == "rut":
            assert {"formato", "dv_valido"} <= e.etiquetas.keys()
        if e.tipo == "telefono":
            assert {"formato", "clase"} <= e.etiquetas.keys()
        if e.tipo == "correo":
            assert "formato" in e.etiquetas
    assert any(e.tipo == "nombre" and e.nivel == "fuera_de_alcance" for e in elementos)
    assert any(e.tipo == "rut" and not e.etiquetas["dv_valido"] for e in elementos)
    (rostro,) = [e for e in elementos if e.tipo == "rostro"]
    assert rostro.nucleo is not None and rostro.nivel == "base"


def test_fax_binarizado_y_en_estres(generado):
    ctx, archivos, _ = generado
    a = _por_id(archivos, "fax_150dpi")
    assert {e.nivel for e in a.elementos} <= {"estres", "fuera_de_alcance"}
    with pymupdf.open(ctx.raiz / a.ruta) as doc:
        (img,) = doc[0].get_images(full=True)
        assert img[4] == 1 and img[2] == int(round(8.27 * 150))  # 1 bit por componente, 150 ppp


def test_timbre_firma_y_nota(generado):
    _, archivos, _ = generado
    a = _por_id(archivos, "timbre_y_firma")
    timbre = [e for e in a.elementos if e.etiquetas.get("timbre")]
    assert {e.tipo for e in timbre} >= {"nombre", "rut", "texto"}
    for e in timbre:
        if e.tipo != "texto":
            assert e.nivel in ("estres", "fuera_de_alcance")
        assert abs(e.etiquetas["angulo"] % 360 - 20.5) < 0.01 or abs(e.etiquetas["angulo"] % 360 - 353.5) < 0.01
    (firma,) = [e for e in a.elementos if e.tipo == "firma"]
    assert firma.nivel == "fuera_de_alcance"
    manuscritos = [e for e in a.elementos if e.tipo == "telefono" and e.etiquetas.get("manuscrito")]
    assert len(manuscritos) == 1 and manuscritos[0].nivel == "estres"


def test_escaneo_invertido_giros(generado):
    _, archivos, _ = generado
    a = _por_id(archivos, "escaneo_invertido")
    angulos = {i: {round(e.etiquetas["angulo"] % 360, 1) for e in a.elementos if e.pagina == i} for i in (0, 1)}
    assert angulos[0] == {180.6}
    assert angulos[1] == {89.2}
    x0, y0, x1, y1 = caja_envolvente(next(e for e in a.elementos if e.pagina == 1 and e.tipo == "rut").poligono)
    assert (y1 - y0) > 2 * (x1 - x0), "en la planilla girada los datos corren en vertical"


def _contexto(raiz: Path) -> Contexto:
    return Contexto(
        raiz=raiz,
        semilla=33,
        fict=Ficticios(33),
        rostros=ProveedorRostros(RAIZ_REPO / "datos_prueba" / "cache" / "rostros", 33, permitir_descarga=False),
    )


def test_pdf_determinista(tmp_path):
    """Dos corridas con la misma semilla dan bytes idénticos (sin /ID aleatorio en el trailer)."""
    bytes_por_corrida = []
    for n in (1, 2):
        ctx = _contexto(tmp_path / f"corrida{n}")
        a = pdf_escaneado._gen_fax(ctx, ctx.ficticios(pdf_escaneado.MODULO), ctx.rng(pdf_escaneado.MODULO))
        bytes_por_corrida.append((ctx.raiz / a.ruta).read_bytes())
    assert bytes_por_corrida[0] == bytes_por_corrida[1]


def _error_extension(gris: np.ndarray, poligono: np.ndarray) -> float | None:
    """Distancia (px) entre los bordes del cuadrilátero y la extensión de la tinta, en el marco del texto.

    Cálculo independiente del generador: se mide la tinta real de la imagen incrustada, a lo
    largo de la línea base (dentro del alto) y en perpendicular (dentro del largo).
    """
    p0, p1, p3 = poligono[0], poligono[1], poligono[3]
    largo, alto = float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(p3 - p0))
    u, v = (p1 - p0) / largo, (p3 - p0) / alto
    m = max(4.0, 0.6 * alto)
    x0, y0 = np.maximum(np.floor(poligono.min(0) - m), 0).astype(int)
    x1, y1 = np.ceil(poligono.max(0) + m).astype(int)
    sub = gris[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0 : y0 + sub.shape[0], x0 : x0 + sub.shape[1]] + 0.5
    d = np.stack([xx - p0[0], yy - p0[1]], -1)
    cu, cv = d @ u, d @ v
    dentro = (cu >= 0) & (cu <= largo) & (cv >= 0) & (cv <= alto)
    if not dentro.any():
        return None
    umbral = (np.percentile(sub, 90) + np.percentile(sub[dentro], 2)) / 2
    tinta = sub < umbral
    sel_v = tinta & (cu >= 0) & (cu <= largo) & (cv >= -0.35 * alto) & (cv <= 1.35 * alto)
    sel_u = tinta & (cv >= 0) & (cv <= alto) & (cu >= -0.12 * alto) & (cu <= largo + 0.12 * alto)
    if sel_v.sum() < 3 or sel_u.sum() < 3:
        return None
    vs, us = np.sort(cv[sel_v]), np.sort(cu[sel_u])  # se descarta un píxel en cada extremo (ruido)
    return float(max(abs(vs[1]), abs(vs[-2] - alto), abs(us[1]), abs(us[-2] - largo)))


def test_poligonos_raster_ajustados_a_la_tinta(generado):
    """Los polígonos raster calzan con la tinta de la imagen de página (mediana < 1 pt por archivo)."""
    ctx, archivos, _ = generado
    for a in archivos:
        errores = []
        with pymupdf.open(ctx.raiz / a.ruta) as doc:
            for i, page in enumerate(doc):
                elementos = [e for e in a.elementos if e.pagina == i and e.capa == "raster" and e.tipo != "rostro"]
                if not elementos:
                    continue
                datos = doc.extract_image(page.get_images()[0][0])["image"]
                gris = np.asarray(Image.open(io.BytesIO(datos)).convert("L"), np.float32)
                escala = np.array([gris.shape[1] / page.rect.width, gris.shape[0] / page.rect.height])
                for e in elementos:
                    err = _error_extension(gris, np.asarray(e.poligono, np.float64) * escala)
                    assert err is not None, (a.id, e.valor)
                    errores.append(err / escala[0])
        assert float(np.median(errores)) < 1.0, (a.id, np.median(errores))
        assert float(np.percentile(errores, 90)) < 3.5, (a.id, np.percentile(errores, 90))
