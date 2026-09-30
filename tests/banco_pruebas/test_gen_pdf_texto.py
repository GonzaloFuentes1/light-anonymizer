"""Invariantes del generador ``pdf_texto``: archivos válidos y verdad de terreno que calza con el PDF."""

import re
import time
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Archivo, Manifiesto
from banco_pruebas.ficticios import FORMATOS_CORREO, FORMATOS_RUT, FORMATOS_TELEFONO, Ficticios, sin_tildes
from banco_pruebas.generadores import pdf_texto
from banco_pruebas.lienzo import caja_envolvente
from banco_pruebas.rostros import ProveedorRostros
from banco_pruebas.visualizar import ESCALA_PDF, a_visual

RAIZ_REPO = Path(__file__).resolve().parents[2]


def _contexto(raiz: Path) -> Contexto:
    return Contexto(
        raiz=raiz,
        semilla=33,
        fict=Ficticios(33),
        rostros=ProveedorRostros(RAIZ_REPO / "datos_prueba" / "cache" / "rostros", 33, permitir_descarga=False),
    )


@pytest.fixture(scope="module")
def generado(tmp_path_factory):
    ctx = _contexto(tmp_path_factory.mktemp("pdf_texto") / "generado")
    inicio = time.perf_counter()
    archivos = pdf_texto.generar(ctx)
    return ctx, archivos, time.perf_counter() - inicio


@pytest.fixture(scope="module")
def archivos(generado) -> list[Archivo]:
    return generado[1]


@pytest.fixture(scope="module")
def raiz(generado) -> Path:
    return generado[0].raiz


def _por_id(archivos: list[Archivo], id: str) -> Archivo:
    return next(a for a in archivos if a.id == id)


@lru_cache(maxsize=64)
def _render(ruta: str, pagina: int) -> np.ndarray:
    doc = pymupdf.open(ruta)
    pix = doc[pagina].get_pixmap(matrix=pymupdf.Matrix(ESCALA_PDF, ESCALA_PDF), alpha=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    doc.close()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _pixeles(archivo: Archivo, raiz: Path, e) -> np.ndarray:
    gris = _render(str(raiz / archivo.ruta), e.pagina)
    pts = np.asarray(a_visual(archivo, e.pagina, e.poligono), np.float64) - 0.5
    x0, y0 = np.floor(pts.min(axis=0)).astype(int).clip(0)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 2
    recorte = gris[y0:y1, x0:x1]
    mascara = np.zeros(recorte.shape, np.uint8)
    cv2.fillPoly(mascara, [np.round((pts - [x0, y0]) * 16).astype(np.int32)], 1, shift=4)
    return recorte[mascara.astype(bool)]


def _dentro(e, pagina, holgura: float = 0.5) -> bool:
    x0, y0, x1, y1 = caja_envolvente(e.poligono)
    return x0 >= -holgura and y0 >= -holgura and x1 <= pagina.ancho + holgura and y1 <= pagina.alto + holgura


# ---------------------------------------------------------------------------


def test_tiempo_de_generacion(generado):
    # El tiempo depende de la carga del equipo: se avisa, no se falla.
    if generado[2] > 90:
        warnings.warn(f"generación lenta: {generado[2]:.1f} s", stacklevel=1)


def test_manifiesto_valido(generado, archivos):
    ctx = generado[0]
    ids = [a.id for a in archivos]
    assert len(ids) == len(set(ids)) == 12
    assert all(i.startswith("pdft_") for i in ids)
    assert {a.categoria for a in archivos} == {"pdf_texto"}
    man = Manifiesto(raiz=str(ctx.raiz), semilla=33)
    for a in archivos:
        man.agregar(a)
    assert man.validar() == []
    assert all(a.sha256 for a in man.archivos)


def test_paginas_coinciden_con_el_pdf(archivos, raiz):
    for a in archivos:
        doc = pymupdf.open(raiz / a.ruta)
        assert doc.page_count == len(a.paginas), a.id
        for pag, page in zip(a.paginas, doc, strict=True):
            assert pag.unidad == "pt"
            assert pag.ancho == pytest.approx(page.cropbox.width) and pag.alto == pytest.approx(page.cropbox.height)
            assert pag.rotacion == page.rotation
        for e in a.elementos:
            assert 0 <= e.pagina < doc.page_count
            assert e.poligono is not None and len(e.poligono) == 4, e
        doc.close()


def test_poligonos_dentro_de_la_pagina(archivos):
    for a in archivos:
        for e in a.elementos:
            pagina = a.paginas[e.pagina]
            if e.capa == "oculto" and e.etiquetas.get("oculto") == "fuera_del_cropbox":
                assert not _dentro(e, pagina), (a.id, e.valor)
            else:
                assert _dentro(e, pagina), (a.id, e.valor, e.poligono)


def test_capa_de_texto_se_encuentra_con_search_for(archivos, raiz):
    revisados = 0
    for a in archivos:
        doc = pymupdf.open(raiz / a.ruta)
        for e in a.elementos:
            if e.capa not in ("texto", "oculto"):
                continue
            page = doc[e.pagina]
            if "espaciado_pt" in e.etiquetas:
                # letras sueltas: el texto dentro del polígono, sin espacios, es el valor
                x0, y0, x1, y1 = caja_envolvente(e.poligono)
                extraido = page.get_text("text", clip=pymupdf.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1))
                assert "".join(extraido.split()) == e.valor
                continue
            tp = pdf_texto.textpage_completa(page)
            quads = page.search_for(e.valor, quads=True, textpage=tp)
            distancias = [pdf_texto._distancia(pdf_texto._quad_a_poligono(q), e.poligono) for q in quads]
            assert distancias and min(distancias) <= 1.0, (a.id, e.valor, e.etiquetas.get("gt"))
            revisados += 1
        doc.close()
    assert revisados > 800


def test_elementos_visibles_tienen_tinta(archivos, raiz):
    for a in archivos:
        for e in a.elementos:
            if e.capa == "oculto":
                continue
            px = _pixeles(a, raiz, e)
            assert px.size > 0, (a.id, e.valor)
            if e.tipo == "rostro":
                assert px.std() > 10, (a.id, e.etiquetas)
            else:
                assert px.min() < 150, (a.id, e.tipo, e.valor)


def test_poligonos_ajustados_a_la_tinta(archivos, raiz):
    """La tinta de cada dato queda dentro de su polígono: ningún trazo que lo toca se sale más de 1,5 pt.

    Se descuentan los demás polígonos de la página (texto vecino), los trazos que no tocan el
    polígono (puntuación contigua) y las líneas rectas largas (bordes de tabla, líneas de firma).
    Cubre también la página con /Rotate, el texto girado y las imágenes incrustadas.
    """
    revisados = 0
    for a in archivos:
        for pagina in range(len(a.paginas)):
            gris = _render(str(raiz / a.ruta), pagina)
            elementos = [e for e in a.elementos if e.pagina == pagina and e.capa != "oculto"]
            visuales = [np.asarray(a_visual(a, pagina, e.poligono), np.float64) - 0.5 for e in elementos]
            for e, pts in zip(elementos, visuales, strict=True):
                if e.tipo in ("texto", "rostro", "qr"):
                    continue
                # ventana local con margen amplio (para reconocer las líneas largas) y franja de 3 px = 1,5 pt
                x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 32
                x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 32
                x0, y0 = max(x0, 0), max(y0, 0)
                ventana = gris[y0:y1, x0:x1]

                def mascara(q: np.ndarray, forma=ventana.shape, origen=(x0, y0)) -> np.ndarray:
                    m = np.zeros(forma, np.uint8)
                    cv2.fillPoly(m, [np.round((q - origen) * 16).astype(np.int32)], 1, shift=4)
                    return m.astype(bool)

                propia = mascara(pts)
                franja = cv2.dilate(propia.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~propia
                for otra in visuales:
                    if otra is not pts:
                        franja &= ~mascara(otra)
                fondo = int(np.median(ventana[franja | propia]))
                tinta = (np.abs(ventana.astype(int) - fondo) > 80).astype(np.uint8)
                lineas = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, np.ones((1, 40), np.uint8)) | cv2.morphologyEx(
                    tinta, cv2.MORPH_OPEN, np.ones((40, 1), np.uint8)
                )
                tinta = tinta.astype(bool) & ~lineas.astype(bool)
                # solo cuentan los trazos que tocan el polígono y se salen de él (no la puntuación vecina)
                _, etiquetas = cv2.connectedComponents(tinta.astype(np.uint8), connectivity=8)
                tocan = np.isin(etiquetas, np.unique(etiquetas[tinta & propia]))
                dentro, fuera = int((tinta & propia).sum()), int((tinta & tocan & franja).sum())
                assert dentro > 0 and fuera <= 0.02 * dentro, (a.id, e.tipo, e.valor, dentro, fuera)
                revisados += 1
    assert revisados > 250


def test_oculto_no_se_ve(archivos, raiz):
    a = _por_id(archivos, "pdft_estres_capa_texto")
    modos = Counter()
    for e in a.elementos:
        if e.capa != "oculto":
            continue
        modos[e.etiquetas["oculto"]] += 1
        if e.etiquetas["oculto"] == "fuera_del_cropbox":
            continue
        px = _pixeles(a, raiz, e)
        assert int(px.max()) - int(px.min()) < 12, (e.valor, e.etiquetas["oculto"])
    assert set(modos) == {"blanco_sobre_blanco", "render_mode_3", "bajo_imagen", "fuera_del_cropbox"}
    fuera = [e for e in a.elementos if e.etiquetas.get("oculto") == "fuera_del_cropbox" and e.tipo != "texto"]
    assert {e.tipo for e in fuera} == {"nombre", "rut", "correo"}
    assert all(e.nivel in ("estres", "fuera_de_alcance") for e in fuera)


def test_texto_vectorizado_no_tiene_texto(archivos, raiz):
    a = _por_id(archivos, "pdft_texto_vectorizado")
    doc = pymupdf.open(raiz / a.ruta)
    assert doc[0].get_text().strip() == ""
    assert len(doc[0].get_drawings()) > 100
    assert {e.capa for e in a.elementos} == {"vector"}
    assert {"rut", "correo", "telefono", "nombre", "direccion"} <= {e.tipo for e in a.elementos}
    doc.close()


def test_conteos_por_tipo(archivos):
    c = Counter((e.tipo, e.nivel) for a in archivos for e in a.elementos)
    tipos = Counter(e.tipo for a in archivos for e in a.elementos)
    assert tipos["rut"] >= 55 and tipos["correo"] >= 55 and tipos["telefono"] >= 50
    assert tipos["url"] >= 12 and tipos["direccion"] >= 12 and tipos["texto"] >= 500
    assert tipos["rostro"] == 3 and tipos["qr"] == 1
    assert c[("nombre", "fuera_de_alcance")] >= 2 and c[("direccion", "fuera_de_alcance")] >= 1
    assert c[("nombre", "estres")] >= 4
    senuelos = Counter(e.etiquetas.get("senuelo") for a in archivos for e in a.elementos if e.tipo == "texto")
    assert senuelos["monto"] >= 15 and senuelos["fecha"] >= 30


def test_texto_neutro_no_contiene_nombres(archivos):
    """Ningún texto registrado como neutro lleva un nombre o apellido de las personas del conjunto."""

    def palabras(texto: str) -> set[str]:
        return {sin_tildes(w).lower() for w in re.findall(r"\w{3,}", texto)}

    tokens = set().union(*(palabras(e.valor) for a in archivos for e in a.elementos if e.tipo == "nombre"))
    tokens -= {"los", "las", "del"}
    for a in archivos:
        for e in a.elementos:
            if e.tipo == "texto":
                assert not palabras(e.valor) & tokens, (a.id, e.valor)
    saludos = [e for e in _por_id(archivos, "pdft_correo_impreso").elementos
               if e.tipo == "nombre" and e.etiquetas["variante"] == "solo_nombre"]  # fmt: skip
    assert len(saludos) == 2 and all(e.nivel == "estres" for e in saludos)


def test_cobertura_de_formatos_en_honorarios(archivos):
    elementos = [e for a in archivos if a.id.startswith("pdft_honorarios_") for e in a.elementos]
    for tipo, formatos in (("rut", FORMATOS_RUT), ("telefono", FORMATOS_TELEFONO), ("correo", FORMATOS_CORREO)):
        usados = Counter(e.etiquetas["formato"] for e in elementos if e.tipo == tipo)
        for formato, nivel in formatos.items():
            if nivel == "base":
                assert usados[formato] >= 2, (tipo, formato)
    ruts = [e for e in elementos if e.tipo == "rut"]
    assert any(e.etiquetas["formato"] == "k_minuscula" and e.valor.endswith("-k") for e in ruts)
    assert any(e.valor.endswith("-K") for e in ruts)
    assert any(e.etiquetas.get("siete_digitos") for e in ruts)
    assert sum(not e.etiquetas["dv_valido"] for e in ruts) >= 3
    variantes = Counter(e.etiquetas["variante"] for e in elementos if e.tipo == "nombre")
    assert {"exacta", "sin_tildes", "mayusculas", "apellidos_nombres", "parcial"} <= set(variantes)
    assert all(e.nivel == "estres" for e in elementos if e.tipo == "nombre" and e.etiquetas["variante"] == "parcial"
               and e.etiquetas["en_lista"])  # fmt: skip
    assert {e.etiquetas["clase"] for e in elementos if e.tipo == "telefono"} == {"movil", "santiago", "regional"}


def test_replica_del_cuaderno(generado, archivos, raiz):
    ctx = generado[0]
    a = _por_id(archivos, "pdft_cuaderno")
    doc = pymupdf.open(raiz / a.ruta)
    lineas = [x for x in doc[0].get_text().splitlines() if x.strip()]
    assert lineas == [x for x in pdf_texto.TEXTO_CUADERNO.splitlines() if x.strip()]
    doc.close()
    valores = {(e.tipo, e.valor) for e in a.elementos}
    assert ("rut", "15.782.334-9") in valores and ("rut", "9.876.543-3") in valores
    assert ("url", "https://www.ejemplo.cl/rendiciones") in valores
    (malo,) = [e for e in a.elementos if e.valor == "15.782.334-9"]
    assert malo.etiquetas["dv_valido"] is False
    lista = ctx.fict.lista_nombres()
    assert "Ana Maria Rojas Pena" in lista and "Pasaje Los Alerces 442, depto 31" in lista


def test_imagenes_incrustadas(archivos, raiz):
    a = _por_id(archivos, "pdft_mixto_imagenes")
    doc = pymupdf.open(raiz / a.ruta)
    compartidos = {x[0] for x in doc[1].get_images(full=True) if len(doc[1].get_image_rects(x[0])) == 2}
    assert len(compartidos) == 1
    con_smask = [x for p in doc for x in p.get_images(full=True) if x[1] > 0]
    assert con_smask, "falta la imagen con transparencia (SMask)"
    rostros = [e for e in a.elementos if e.tipo == "rostro"]
    assert len(rostros) == 3 and all(e.nucleo for e in rostros)
    (qr,) = [e for e in a.elementos if e.tipo == "qr"]
    gris = _render(str(raiz / a.ruta), qr.pagina)
    x0, y0, x1, y1 = (int(v * ESCALA_PDF) for v in caja_envolvente(qr.poligono))
    recorte = cv2.copyMakeBorder(gris[y0:y1, x0:x1], 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    texto, _, _ = cv2.QRCodeDetector().detectAndDecode(recorte)
    assert texto == qr.valor and qr.nivel == "estres"
    raster = Counter(e.tipo for e in a.elementos if e.capa == "raster")
    assert raster["correo"] >= 1 and raster["telefono"] >= 3 and raster["rut"] >= 1
    doc.close()


def test_pagina_rotada_se_ve_horizontal(archivos):
    a = _por_id(archivos, "pdft_pagina_rotada")
    assert [p.rotacion for p in a.paginas] == [0, 90]
    datos = [e for e in a.elementos if e.pagina == 1 and e.tipo != "texto"]
    assert len(datos) >= 35
    for e in datos:
        pts = np.asarray(a_visual(a, 1, e.poligono))
        ancho, alto = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        assert ancho > alto, e.valor


def test_texto_girado_cubre_los_angulos(archivos):
    a = _por_id(archivos, "pdft_texto_girado")
    angulos = {e.etiquetas["angulo"] for e in a.elementos if e.tipo != "texto"}
    assert {0, 90, 180, 270, 30.0} <= angulos
    for e in a.elementos:
        assert e.etiquetas["gt"] == "search_for", e.valor


def test_generacion_determinista(archivos, tmp_path):
    otra = pdf_texto.generar(_contexto(tmp_path / "otra"))
    firma = [(a.id, [(e.tipo, e.valor, e.nivel, e.poligono) for e in a.elementos]) for a in archivos]
    assert firma == [(a.id, [(e.tipo, e.valor, e.nivel, e.poligono) for e in a.elementos]) for a in otra]
