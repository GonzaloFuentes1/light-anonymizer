"""Invariantes del generador de cédulas ficticias: archivos válidos y verdad de terreno que calza."""

from __future__ import annotations

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
from banco_pruebas.esquema import Archivo, Elemento, Manifiesto
from banco_pruebas.ficticios import Ficticios, digito_verificador
from banco_pruebas.generadores import cedula
from banco_pruebas.lienzo import caja_envolvente
from banco_pruebas.rostros import ProveedorRostros

RAIZ_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generado(tmp_path_factory: pytest.TempPathFactory) -> tuple[Contexto, list[Archivo], float]:
    raiz = tmp_path_factory.mktemp("cedula") / "generado"
    ctx = Contexto(
        raiz=raiz,
        semilla=33,
        fict=Ficticios(33),
        rostros=ProveedorRostros(RAIZ_REPO / "datos_prueba" / "cache" / "rostros", 33, permitir_descarga=False),
    )
    inicio = time.perf_counter()
    archivos = cedula.generar(ctx)
    return ctx, archivos, time.perf_counter() - inicio


def _mascara(poligono, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(poligono, np.float64) - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def _pagina_gris(ctx: Contexto, a: Archivo) -> tuple[np.ndarray, float]:
    """Página en gris y escala (px por unidad del manifiesto)."""
    ruta = ctx.raiz / a.ruta
    if a.formato == "pdf":
        doc = pymupdf.open(ruta)
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        return np.asarray(img.convert("L")), 2.0
    return np.asarray(Image.open(ruta).convert("L")), 1.0


def test_tiempo_y_archivos(generado):
    ctx, archivos, segundos = generado
    # El tiempo depende de la carga del equipo: se avisa, no se falla (los tiempos se reportan aparte).
    if segundos > 90:
        warnings.warn(f"generación lenta: {segundos:.1f} s", stacklevel=1)
    esperados = {
        "cedula/cedula_frente_plana.png",
        "cedula/cedula_dorso_plana.png",
        "cedula/cedula_frente_foto_perspectiva.jpg",
        "cedula/cedula_frente_foto_girada.jpg",
        "cedula/cedula_dorso_foto.jpg",
        "cedula/cedula_ambos_lados.pdf",
        "cedula/cedula_con_reflejo.jpg",
    }
    assert {a.ruta for a in archivos} == esperados
    assert len({a.id for a in archivos}) == len(archivos)
    for a in archivos:
        assert a.id.startswith("ced_") and a.categoria == "cedula"
        ruta = ctx.raiz / a.ruta
        assert ruta.exists()
        if a.formato == "pdf":
            doc = pymupdf.open(ruta)
            assert len(doc) == len(a.paginas) == 1
            pag = doc[0]
            assert (pag.rect.width, pag.rect.height) == pytest.approx((a.paginas[0].ancho, a.paginas[0].alto))
            assert pag.rotation == a.paginas[0].rotacion
            assert len(pag.get_images()) == 2
            doc.close()
        else:
            img = Image.open(ruta)
            assert img.size == (a.paginas[0].ancho, a.paginas[0].alto)
            assert not img.getexif()
    planas = [a for a in archivos if "plana" in a.ruta]
    for a in planas:
        assert (a.paginas[0].ancho, a.paginas[0].alto) == (1027, 648)


def test_manifiesto_valido(generado):
    ctx, archivos, _ = generado
    man = Manifiesto(raiz=str(ctx.raiz), semilla=33)
    for a in archivos:
        man.agregar(a)
    assert man.validar() == []


def test_poligonos_dentro_y_con_contenido(generado):
    ctx, archivos, _ = generado
    for a in archivos:
        gris, escala = _pagina_gris(ctx, a)
        pag = a.paginas[0]
        for e in a.elementos:
            assert e.poligono is not None and len(e.poligono) == 4
            if e.capa == "oculto":
                continue
            x0, y0, x1, y1 = caja_envolvente(e.poligono)
            assert x0 >= -1 and y0 >= -1 and x1 <= pag.ancho + 1 and y1 <= pag.alto + 1, (a.id, e)
            m = _mascara([[x * escala, y * escala] for x, y in e.poligono], gris.shape)
            assert m.sum() > 0
            assert gris[m].std() > 2, (a.id, e.tipo, e.valor)
            if e.nucleo is not None:
                nx0, ny0, nx1, ny1 = caja_envolvente(e.nucleo)
                assert x0 - 1 <= nx0 and y0 - 1 <= ny0 and nx1 <= x1 + 1 and ny1 <= y1 + 1


def test_texto_plano_tiene_tinta_oscura(generado):
    ctx, archivos, _ = generado
    for a in archivos:
        if "plana" not in a.ruta:
            continue
        gris = np.asarray(Image.open(ctx.raiz / a.ruta).convert("L"))
        for e in a.elementos:
            if e.tipo in ("texto", "rut", "nombre"):
                m = _mascara(e.poligono, gris.shape)
                assert gris[m].min() < 130, (a.id, e.valor)


def test_capa_texto_en_pdf(generado):
    ctx, archivos, _ = generado
    (a,) = [a for a in archivos if a.formato == "pdf"]
    doc = pymupdf.open(ctx.raiz / a.ruta)
    pag = doc[0]
    en_texto = [e for e in a.elementos if e.capa == "texto"]
    assert any(e.tipo == "nombre" for e in en_texto)
    for e in en_texto:
        hits = pag.search_for(e.valor)
        assert hits, e.valor
        x0, y0, x1, y1 = caja_envolvente(e.poligono)
        assert any(abs(h.x0 - x0) < 1 and abs(h.y0 - y0) < 1 and abs(h.x1 - x1) < 1 for h in hits)
    raster = [e for e in a.elementos if e.capa == "raster"]
    assert Counter(e.tipo for e in raster)["rostro"] == 2
    doc.close()


def test_conteos_por_tipo(generado):
    _, archivos, _ = generado
    total = Counter(e.tipo for a in archivos for e in a.elementos)
    # 5 anversos (plana, perspectiva, girada, pdf, reflejo) y 3 reversos (plana, foto, pdf)
    assert total["rostro"] == 10
    assert total["firma"] == 5
    assert total["qr"] == 3
    assert total["rut"] == 8
    assert total["nombre"] == 5 * 3 + 3 + 1
    for a in archivos:
        c = Counter(e.tipo for e in a.elementos)
        assert c["texto"] >= 8, a.id
        fantasmas = [e for e in a.elementos if e.tipo == "rostro" and e.etiquetas.get("fantasma")]
        assert all(e.nivel == "estres" for e in fantasmas)
        assert all(e.nucleo is not None for e in a.elementos if e.tipo == "rostro")
        assert all(e.nivel == "fuera_de_alcance" for e in a.elementos if e.tipo == "firma")
        assert all(e.nivel == "estres" for e in a.elementos if e.tipo == "qr")
        for e in a.elementos:
            if e.tipo == "nombre" and not e.etiquetas["en_lista"]:
                assert e.nivel == "fuera_de_alcance"
            if e.etiquetas.get("formato") == "mrz":
                assert e.nivel in ("estres", "fuera_de_alcance")
    (reflejo,) = [a for a in archivos if a.id == "ced_con_reflejo"]
    assert all(e.nivel != "base" for e in reflejo.elementos if e.tipo != "texto")
    base = [e for a in archivos for e in a.elementos if e.tipo == "rut" and e.nivel == "base"]
    assert len(base) == 4  # plana, perspectiva, girada y pdf


def test_tinta_no_escapa_de_los_poligonos(generado):
    """Comprobación independiente de la geometría (también con perspectiva, giro y desenfoque):
    alrededor de cada texto oscuro sobre fondo claro no queda tinta fuera de todo polígono."""
    ctx, archivos, _ = generado
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anillo_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    for a in archivos:
        gris, escala = _pagina_gris(ctx, a)
        oscuro = gris < 105
        union = np.zeros(gris.shape, bool)
        for e in a.elementos:
            union |= _mascara([[x * escala, y * escala] for x, y in e.poligono], gris.shape)
        union = cv2.dilate(union.astype(np.uint8), nucleo).astype(bool)
        for e in a.elementos:
            if e.tipo in ("rostro", "qr"):
                continue
            pol = [[x * escala, y * escala] for x, y in e.poligono]
            # ventana local alrededor del polígono (más rápido que trabajar con la imagen entera)
            bx0, by0, bx1, by1 = caja_envolvente(pol)
            x0, y0 = max(0, int(bx0) - 10), max(0, int(by0) - 10)
            x1, y1 = min(gris.shape[1], int(bx1) + 11), min(gris.shape[0], int(by1) + 11)
            sub = (slice(y0, y1), slice(x0, x1))
            m = _mascara([[x - x0, y - y0] for x, y in pol], (y1 - y0, x1 - x0))
            anillo = cv2.dilate(m.astype(np.uint8), anillo_k).astype(bool) & ~union[sub]
            if not anillo.any() or np.median(gris[sub][anillo]) < 150:
                continue  # texto claro sobre fondo oscuro (franja roja)
            dentro = int((oscuro[sub] & m).sum())
            fuera = int((oscuro[sub] & anillo).sum())
            assert fuera <= max(8, 0.03 * dentro), (a.id, e.tipo, e.valor, dentro, fuera)


def test_pdf_determinista(generado, tmp_path):
    ctx, _, _ = generado
    f = Ficticios(5).derivar("prueba")
    rng = np.random.default_rng(5)
    datos = cedula._datos(f, rng, set())
    frente = cedula.dibujar_frente(datos, ctx.rostros, rng)
    dorso = cedula.dibujar_dorso(datos, rng)
    ctx2 = Contexto(raiz=tmp_path, semilla=5, fict=f, rostros=ctx.rostros)
    cedula._pdf_ambos_lados(ctx2, "a.pdf", frente, dorso, datos)
    cedula._pdf_ambos_lados(ctx2, "b.pdf", frente, dorso, datos)
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()


def test_ensanchar_por_movimiento():
    """El polígono crece medio largo del núcleo (más 0,5 px) en la dirección del arrastre."""
    e = Elemento(tipo="rut", pagina=0, poligono=[[0, 0], [10, 0], [10, 10], [0, 10]], valor="x")
    cedula._ensanchar_por_movimiento([e], 11, 0)
    assert caja_envolvente(e.poligono) == pytest.approx((-5.5, -0.5, 15.5, 10.5))
    e = Elemento(tipo="rut", pagina=0, poligono=[[0, 0], [10, 0], [10, 10], [0, 10]], valor="x")
    cedula._ensanchar_por_movimiento([e], 11, 90)
    assert caja_envolvente(e.poligono) == pytest.approx((-0.5, -5.5, 10.5, 15.5))
    r = Elemento(tipo="rostro", pagina=0, poligono=[[0, 0], [10, 0], [10, 10], [0, 10]])
    cedula._ensanchar_por_movimiento([r], 11, 0)
    assert r.poligono == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_reflejo_cubre_la_estela_del_movimiento(generado):
    """En la foto con desenfoque de movimiento, la tinta tenue (no solo la oscura) queda dentro."""
    ctx, archivos, _ = generado
    (a,) = [a for a in archivos if a.id == "ced_con_reflejo"]
    gris = np.asarray(Image.open(ctx.raiz / a.ruta).convert("L")).astype(np.float64)
    (run,) = [e for e in a.elementos if e.tipo == "rut"]
    m = _mascara(run.poligono, gris.shape)
    anillo = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
    fondo = np.median(gris[anillo])
    assert (gris[anillo] < fondo - 40).sum() <= 0.01 * anillo.sum()


def test_qr_se_decodifica_dentro_del_poligono(generado):
    ctx, archivos, _ = generado
    (a,) = [a for a in archivos if a.id == "ced_dorso_plana"]
    (qr,) = [e for e in a.elementos if e.tipo == "qr"]
    img = cv2.imread(str(ctx.raiz / a.ruta))
    texto, puntos, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert texto == qr.valor
    assert "RUN=" in texto and "type=CEDULA" in texto
    x0, y0, x1, y1 = caja_envolvente(qr.poligono)
    for x, y in puntos.reshape(-1, 2):
        assert x0 - 3 <= x <= x1 + 3 and y0 - 3 <= y <= y1 + 3


def test_mrz(generado):
    _, archivos, _ = generado
    (a,) = [a for a in archivos if a.id == "ced_dorso_plana"]
    mrz = [e for e in a.elementos if e.etiquetas.get("campo") == "mrz" or e.etiquetas.get("formato") == "mrz"]
    run = [e for e in mrz if e.tipo == "rut"]
    assert len(run) == 1
    cuerpo, dv = run[0].valor.split("<")
    assert (digito_verificador(int(cuerpo)) == dv) == run[0].etiquetas["dv_valido"]
    lineas = {}
    for e in mrz:
        lineas.setdefault(e.etiquetas["linea"], []).append(e)
    assert sorted(lineas) == [1, 2, 3]
    linea1 = "".join(e.valor for e in sorted(lineas[1], key=lambda e: e.poligono[0][0]))
    assert len(linea1) == 30 and run[0].valor in linea1
    assert all(len(e.valor) == 30 for e in lineas[2] + lineas[3])
    assert cedula._digito_mrz("520727") == "3"  # ejemplo de ICAO 9303
    # dígito compuesto TD1: l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    (l2,) = [e.valor for e in lineas[2]]
    assert l2[29] == cedula._digito_mrz(linea1[5:] + l2[0:7] + l2[8:15] + l2[18:29])
