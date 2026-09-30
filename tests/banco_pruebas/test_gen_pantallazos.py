"""Invariantes del generador de pantallazos: tamaños, verdad de terreno ajustada y sin textos encimados."""

from __future__ import annotations

import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Archivo, Manifiesto
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.generadores import pantallazos
from banco_pruebas.lienzo import caja_envolvente
from banco_pruebas.rostros import ProveedorRostros

RAIZ_REPO = Path(__file__).resolve().parents[2]

TAMANOS = {
    "pantallazos/correo_escritorio.png": (1920, 1080),
    "pantallazos/chat_movil.jpg": (1080, 2340),
    "pantallazos/planilla.png": (1600, 900),
    "pantallazos/planilla_reducida.png": (960, 540),
    "pantallazos/formulario_web.webp": (1366, 768),
    "pantallazos/dialogo_pequeno.png": (600, 300),
    "pantallazos/correo_escritorio_hidpi.png": (2880, 1620),
}


@pytest.fixture(scope="module")
def generado(tmp_path_factory: pytest.TempPathFactory) -> tuple[Contexto, dict[str, Archivo], float]:
    raiz = tmp_path_factory.mktemp("pantallazos") / "generado"
    ctx = Contexto(
        raiz=raiz,
        semilla=33,
        fict=Ficticios(33),
        rostros=ProveedorRostros(RAIZ_REPO / "datos_prueba" / "cache" / "rostros", 33, permitir_descarga=False),
    )
    inicio = time.perf_counter()
    archivos = pantallazos.generar(ctx)
    return ctx, {a.ruta: a for a in archivos}, time.perf_counter() - inicio


def _mascara(poligono, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(poligono, np.float64) - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def test_archivos_y_tamanos(generado):
    ctx, archivos, segundos = generado
    # El tiempo depende de la carga del equipo: se avisa, no se falla (los tiempos se reportan aparte).
    if segundos > 90:
        warnings.warn(f"generación lenta: {segundos:.1f} s", stacklevel=1)
    assert set(archivos) == set(TAMANOS)
    assert len({a.id for a in archivos.values()}) == len(archivos)
    for ruta, a in archivos.items():
        assert a.id.startswith("pant_") and a.categoria == "pantallazo"
        img = Image.open(ctx.raiz / ruta)
        img.load()
        assert img.size == TAMANOS[ruta] == (a.paginas[0].ancho, a.paginas[0].alto)
        assert a.formato == ruta.rsplit(".", 1)[1]
        assert not img.getexif()


def test_manifiesto_valido(generado):
    ctx, archivos, _ = generado
    man = Manifiesto(raiz=str(ctx.raiz), semilla=33)
    for a in archivos.values():
        man.agregar(a)
    assert man.validar() == []


def test_poligonos_dentro_y_con_tinta(generado):
    ctx, archivos, _ = generado
    for ruta, a in archivos.items():
        gris = np.asarray(Image.open(ctx.raiz / ruta).convert("L"))
        w, h = TAMANOS[ruta]
        for e in a.elementos:
            assert e.capa == "raster" and e.poligono is not None
            x0, y0, x1, y1 = caja_envolvente(e.poligono)
            assert x0 >= -1 and y0 >= -1 and x1 <= w + 1 and y1 <= h + 1, (a.id, e)
            m = _mascara(e.poligono, gris.shape)
            assert m.sum() > 0
            assert gris[m].std() > 4, (a.id, e.tipo, e.valor)


def test_tinta_no_escapa_de_los_poligonos(generado):
    """Comprobación independiente de la geometría (también tras escalar al 60 % y al 150 %):
    alrededor de cada texto oscuro sobre fondo claro no queda tinta fuera de todo polígono."""
    ctx, archivos, _ = generado
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anillo_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    for ruta, a in archivos.items():
        rgb = np.asarray(Image.open(ctx.raiz / ruta).convert("RGB")).astype(np.int16)
        gris = rgb.mean(axis=2)
        # tinta: oscura y gris (el borde azul de la celda seleccionada no cuenta)
        oscuro = (gris < 105) & (rgb.max(axis=2) - rgb.min(axis=2) < 60)
        union = np.zeros(gris.shape, bool)
        for e in a.elementos:
            union |= _mascara(e.poligono, gris.shape)
        union = cv2.dilate(union.astype(np.uint8), nucleo).astype(bool)
        for e in a.elementos:
            if e.tipo == "rostro":
                continue
            # ventana local alrededor del polígono (más rápido que trabajar con la imagen entera)
            bx0, by0, bx1, by1 = caja_envolvente(e.poligono)
            x0, y0 = max(0, int(bx0) - 10), max(0, int(by0) - 10)
            x1, y1 = min(gris.shape[1], int(bx1) + 11), min(gris.shape[0], int(by1) + 11)
            sub = (slice(y0, y1), slice(x0, x1))
            m = _mascara([[x - x0, y - y0] for x, y in e.poligono], (y1 - y0, x1 - x0))
            anillo = cv2.dilate(m.astype(np.uint8), anillo_k).astype(bool) & ~union[sub]
            g_sub, oscuro_sub = gris[sub], oscuro[sub]
            if not anillo.any() or np.median(g_sub[anillo]) < 150:
                continue  # texto claro sobre una barra oscura
            dentro = int((oscuro_sub & m).sum())
            fuera = int((oscuro_sub & anillo).sum())
            assert fuera <= max(8, 0.03 * dentro), (a.id, e.tipo, e.valor, dentro, fuera)


def test_textos_no_se_enciman(generado):
    """Ningún texto se desborda sobre otro (salvo el RUT anidado dentro de la URL)."""
    _, archivos, _ = generado
    for a in archivos.values():
        cajas = [(e, caja_envolvente(e.poligono)) for e in a.elementos if e.tipo != "rostro"]
        for i, (e1, c1) in enumerate(cajas):
            for e2, c2 in cajas[i + 1 :]:
                if {e1.tipo, e2.tipo} == {"url", "rut"}:
                    continue
                ix = min(c1[2], c2[2]) - max(c1[0], c2[0])
                iy = min(c1[3], c2[3]) - max(c1[1], c2[1])
                if ix <= 0 or iy <= 0:
                    continue
                menor = min((c[2] - c[0]) * (c[3] - c[1]) for c in (c1, c2))
                assert ix * iy < 0.05 * menor, (a.id, e1.valor, e2.valor)


def test_conteos_y_niveles(generado):
    _, archivos, _ = generado
    for ruta in ("pantallazos/correo_escritorio.png", "pantallazos/correo_escritorio_hidpi.png"):
        c = Counter(e.tipo for e in archivos[ruta].elementos)
        assert c["rut"] == 1 and c["telefono"] == 2 and c["direccion"] == 1
        assert c["correo"] >= 15 and c["nombre"] >= 15 and c["texto"] >= 50
    chat = Counter(e.tipo for e in archivos["pantallazos/chat_movil.jpg"].elementos)
    assert chat["rut"] == 2 and chat["telefono"] == 2 and chat["correo"] == 1 and chat["direccion"] == 1
    assert chat["rostro"] == 1
    for ruta in ("pantallazos/planilla.png", "pantallazos/planilla_reducida.png"):
        el = archivos[ruta].elementos
        c = Counter(e.tipo for e in el)
        assert c["nombre"] == 12 and c["correo"] == 12 and c["telefono"] == 12 and c["rut"] == 13
        assert sum(1 for e in el if e.etiquetas.get("senuelo") == "monto") == 12
        assert any(e.nivel == "fuera_de_alcance" for e in el if e.tipo == "nombre")
    for ruta in ("pantallazos/planilla_reducida.png", "pantallazos/dialogo_pequeno.png"):
        assert all(e.nivel != "base" for e in archivos[ruta].elementos if e.tipo != "texto")
    form = Counter(e.tipo for e in archivos["pantallazos/formulario_web.webp"].elementos)
    assert form["url"] == 1 and form["rostro"] == 1 and form["rut"] == 2 and form["direccion"] == 1
    dialogo = Counter(e.tipo for e in archivos["pantallazos/dialogo_pequeno.png"].elementos)
    assert dialogo["nombre"] == 1 and dialogo["correo"] == 1
    for a in archivos.values():
        for e in a.elementos:
            if e.tipo in ("nombre", "direccion") and not e.etiquetas["en_lista"]:
                assert e.nivel == "fuera_de_alcance"


def test_tamano_de_letra(generado):
    _, archivos, _ = generado
    for e in archivos["pantallazos/correo_escritorio.png"].elementos:
        assert 12 <= e.etiquetas["tam_px"] <= 20
    for e in archivos["pantallazos/planilla.png"].elementos:
        assert 10 <= e.etiquetas["tam_px"] <= 13
    reducida = [e.etiquetas["tam_px"] for e in archivos["pantallazos/planilla_reducida.png"].elementos]
    assert max(reducida) <= 8
    hidpi = [e.etiquetas["tam_px"] for e in archivos["pantallazos/correo_escritorio_hidpi.png"].elementos]
    assert min(hidpi) >= 15  # 150 % de 12 px, con algún correo largo achicado para caber


def test_rostros_del_tamano_pedido(generado):
    _, archivos, _ = generado
    (avatar,) = [e for e in archivos["pantallazos/chat_movil.jpg"].elementos if e.tipo == "rostro"]
    (perfil,) = [e for e in archivos["pantallazos/formulario_web.webp"].elementos if e.tipo == "rostro"]
    for e, alto in ((avatar, 90), (perfil, 120)):
        x0, y0, x1, y1 = caja_envolvente(e.poligono)
        assert alto * 0.8 <= y1 - y0 <= alto * 1.2
        assert e.nucleo is not None and e.nivel == "base"
