"""La verdad de terreno tiene que calzar con los píxeles: si no, toda métrica posterior es falsa."""

import cv2
import numpy as np
import pytest
from PIL import Image

from banco_pruebas.lienzo import Lienzo, area_poligono, caja_envolvente, rect, textura_mesa


def _tinta(img: Image.Image, umbral: int = 200) -> np.ndarray:
    return np.asarray(img.convert("L")) < umbral


def _mascara(poligono, shape, holgura: float = 0.0) -> np.ndarray:
    """Máscara del polígono (dilatada ``holgura`` píxeles para tolerar el antialias)."""
    m = np.zeros(shape, np.uint8)
    p = np.asarray(poligono, np.float64) - 0.5  # coordenadas continuas -> centros de píxel
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    if holgura:
        k = int(np.ceil(holgura)) * 2 + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m.astype(bool)


def _fraccion_tinta_fuera(lienzo: Lienzo, holgura: float = 1.5) -> float:
    tinta = _tinta(lienzo.img)
    dentro = np.zeros_like(tinta)
    for e in lienzo.elementos:
        dentro |= _mascara(e.poligono, tinta.shape, holgura)
    return float((tinta & ~dentro).sum()) / max(1, int(tinta.sum()))


@pytest.mark.parametrize("x,y", [(40, 80), (40.4, 80.7), (13.5, 61.25)])
def test_caja_de_tinta_exacta(x, y):
    lz = Lienzo.nuevo(700, 160)
    e = lz.escribir(x, y, "RUT: 15.782.334-9 Peña", tam=36)
    ys, xs = np.nonzero(np.asarray(lz.img.convert("L")) < 255)
    x0, y0, x1, y1 = caja_envolvente(e.poligono)
    assert abs(x0 - xs.min()) <= 1 and abs(x1 - (xs.max() + 1)) <= 1
    assert abs(y0 - ys.min()) <= 1 and abs(y1 - (ys.max() + 1)) <= 1


@pytest.mark.parametrize("grados", [90, 180, 270, 15, 45, -30, 137])
def test_rotacion_conserva_geometria(grados):
    lz = Lienzo.nuevo(900, 500)
    lz.escribir(60, 120, "ana.rojas@ejemplo.cl", tam=40)
    lz.escribir(300, 400, "+56 9 8123 4567", tam=30)
    rot = lz.rotar(grados)
    assert _fraccion_tinta_fuera(rot) < 0.01
    for a, b in zip(lz.elementos, rot.elementos, strict=True):
        assert area_poligono(b.poligono) == pytest.approx(area_poligono(a.poligono), rel=1e-4)


def test_rotacion_90_es_exacta_en_pixeles():
    lz = Lienzo.nuevo(400, 300)
    lz.escribir(50, 100, "12.345.678-5", tam=30)
    for g in (90, 180, 270):
        rot = lz.rotar(g)
        ys, xs = np.nonzero(_tinta(rot.img, 255))
        x0, y0, x1, y1 = caja_envolvente(rot.elementos[0].poligono)
        assert (x0, y0, x1, y1) == pytest.approx((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), abs=1e-6)


def test_espejo_y_escala():
    lz = Lienzo.nuevo(500, 200)
    lz.escribir(30, 90, "Peña Muñoz", tam=40)
    assert _fraccion_tinta_fuera(lz.espejar()) < 0.01
    assert _fraccion_tinta_fuera(lz.escalar(0.37), holgura=1.5) < 0.02


def test_texto_rotado_tipo_timbre():
    lz = Lienzo.nuevo(800, 800)
    for g in (15, 45, 90, 200):
        lz.escribir_rotado(400, 400, "RUN 9.876.543-3", g, tam=32)
        assert _fraccion_tinta_fuera(lz) < 0.01


def test_perspectiva():
    rng = np.random.default_rng(0)
    lz = Lienzo.nuevo(600, 380, fondo=(250, 250, 250))
    lz.escribir(40, 100, "RUN 12.345.678-5", tam=40)
    lz.escribir(40, 250, "ANA MARÍA ROJAS PEÑA", tam=32)
    fondo = Image.new("RGB", (1200, 900), (255, 255, 255))
    destino = [[210, 140], [930, 205], [880, 700], [150, 610]]
    foto = lz.perspectiva(destino, fondo)
    assert _fraccion_tinta_fuera(foto, holgura=2) < 0.02
    # sobre un fondo con textura también debe quedar dentro del cuadrilátero de la tarjeta
    foto2 = lz.perspectiva(destino, textura_mesa(1200, 900, rng))
    assert all(0 <= x <= 1200 and 0 <= y <= 900 for e in foto2.elementos for x, y in e.poligono)


def test_pegar_con_escala():
    lz = Lienzo.nuevo(600, 600)
    cuadro = Image.new("RGB", (100, 80), (0, 0, 0))
    from banco_pruebas.esquema import Elemento

    elem = Elemento(tipo="rostro", pagina=0, poligono=rect(0, 0, 100, 80))
    (e,) = lz.pegar(cuadro, 123.4, 77.8, ancho=250, elementos=[elem])
    ys, xs = np.nonzero(_tinta(lz.img))
    assert caja_envolvente(e.poligono) == pytest.approx((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), abs=1)
