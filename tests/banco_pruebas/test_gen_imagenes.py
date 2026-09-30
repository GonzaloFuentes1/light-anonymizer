"""Invariantes del generador ``imagenes``: archivos legibles, geometría dentro de la página, polígonos
sobre tinta y metadatos (EXIF, XMP, IPTC, PNG, TIFF) realmente escritos en el archivo."""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import piexif
import piexif.helper
import pytest
from PIL import Image, ImageOps, ImageSequence, IptcImagePlugin

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Archivo, Manifiesto
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.generadores import imagenes
from banco_pruebas.rostros import ProveedorRostros
from banco_pruebas.visualizar import paginas_visibles

RAIZ_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generado(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Archivo]]:
    raiz = tmp_path_factory.mktemp("imagenes") / "generado"
    ctx = Contexto(
        raiz=raiz,
        semilla=33,
        fict=Ficticios(33),
        rostros=ProveedorRostros(RAIZ_REPO / "datos_prueba" / "cache" / "rostros", 33, permitir_descarga=False),
    )
    return raiz, imagenes.generar(ctx)


def _por_id(archivos: list[Archivo]) -> dict[str, Archivo]:
    return {a.id: a for a in archivos}


def _utf8(valor: str) -> str:
    """Pillow decodifica las etiquetas ASCII como Latin-1; el archivo guarda UTF-8 (como exiftool)."""
    return valor.encode("latin-1").decode("utf-8")


def _mascara(poligono, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(poligono, np.float64) - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def test_manifiesto_consistente(generado):
    raiz, archivos = generado
    man = Manifiesto(raiz=str(raiz), semilla=33)
    for a in archivos:
        man.agregar(a)
    assert man.validar() == []
    assert all(a.id.startswith("img_") for a in archivos)
    cats = Counter(a.categoria for a in archivos)
    assert cats["imagen_rotada"] == 17 and cats["exif"] == 8 and cats["tiff"] == 2
    rutas = {a.ruta for a in archivos}
    for diseno in ("nota", "tarjeta"):
        for ang in (0, 90, 180, 270, 15, 45):
            assert any(r.startswith(f"imagenes_rotadas/{diseno}_{ang}.") for r in rutas)


def test_conteos_por_tipo(generado):
    _, archivos = generado
    n = Counter((e.tipo, e.nivel) for a in archivos for e in a.elementos)
    assert n[("rut", "base")] >= 12 and n[("correo", "base")] >= 20 and n[("telefono", "base")] >= 25
    assert n[("rut", "estres")] >= 3 and n[("telefono", "estres")] >= 3
    assert n[("nombre", "fuera_de_alcance")] >= 3 and n[("nombre", "base")] >= 10
    assert n[("rostro", "base")] == 2
    assert n[("url", "base")] >= 6 and n[("direccion", "base")] >= 4
    senuelos = Counter(e.etiquetas.get("senuelo") for a in archivos for e in a.elementos if e.tipo == "texto")
    assert senuelos["fecha"] >= 10 and senuelos["monto"] >= 2
    # los 45 grados de base se registran como cuadriláteros girados
    a45 = _por_id(archivos)["img_rot_nota_45"]
    e = next(e for e in a45.elementos if e.tipo == "rut")
    p = np.asarray(e.poligono)
    assert len(p) == 4 and abs(p[:, 0].min() - p[:, 0].max()) > 20 and len({round(x) for x in p[:, 1]}) == 4


def test_geometria_y_tinta(generado):
    raiz, archivos = generado
    for a in archivos:
        paginas = paginas_visibles(raiz / a.ruta, a.formato)
        assert len(paginas) == len(a.paginas), a.id
        for pag, img in zip(a.paginas, paginas, strict=True):
            assert (pag.ancho, pag.alto) == img.size, a.id
        grises = [np.asarray(img.convert("L"), np.float32) for img in paginas]
        for e in a.elementos:
            assert e.capa == "raster"
            pag = a.paginas[e.pagina]
            p = np.asarray(e.poligono)
            assert p[:, 0].min() >= -0.5 and p[:, 1].min() >= -0.5, (a.id, e.valor)
            assert p[:, 0].max() <= pag.ancho + 0.5 and p[:, 1].max() <= pag.alto + 0.5, (a.id, e.valor)
            pix = grises[e.pagina][_mascara(e.poligono, grises[e.pagina].shape)]
            assert pix.size > 0, (a.id, e.valor)
            contraste = np.percentile(pix, 95) - np.percentile(pix, 5)
            minimo = 20 if "bajo_contraste" in a.id else 40
            assert contraste > minimo, (a.id, e.tipo, e.valor, contraste)
            if e.tipo == "rostro":
                n = np.asarray(e.nucleo)
                assert n[:, 0].min() >= p[:, 0].min() and n[:, 0].max() <= p[:, 0].max()


def _ajuste(gris: np.ndarray, poligono, umbral: float, anillo: float = 4.0) -> tuple[float, list[bool]]:
    """Chequeo independiente del ajuste de un cuadrilátero de texto, solo con los píxeles finales.

    En el marco local del cuadrilátero (eje u a lo largo de la línea, v a lo alto) mide qué
    fracción de la tinta cae en un anillo exterior de ``anillo`` px, y si hay tinta pegada a
    cada uno de los cuatro bordes (caja ajustada, no holgada).
    """
    p = np.asarray(poligono, np.float64)
    o, eu, ev = p[0], p[1] - p[0], p[3] - p[0]
    largo, alto = np.linalg.norm(eu), np.linalg.norm(ev)
    u, v = eu / largo, ev / alto
    x0, y0 = np.maximum(np.floor(p.min(0) - anillo - 1).astype(int), 0)
    x1, y1 = np.ceil(p.max(0) + anillo + 1).astype(int)
    x1, y1 = min(x1, gris.shape[1]), min(y1, gris.shape[0])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx, cy = xx + 0.5 - o[0], yy + 0.5 - o[1]
    uu, vv = cx * u[0] + cy * u[1], cx * v[0] + cy * v[1]
    sub = gris[y0:y1, x0:x1]
    dentro = (uu >= 0) & (uu <= largo) & (vv >= 0) & (vv <= alto)
    fuera = (uu >= -anillo) & (uu <= largo + anillo) & (vv >= -anillo) & (vv <= alto + anillo) & ~dentro
    tinta = np.abs(sub - np.median(sub[fuera])) > umbral
    ti, ta = int(tinta[dentro].sum()), int(tinta[fuera].sum())
    b = 2.5
    bordes = [
        bool(tinta[dentro & (uu < b)].any()),
        bool(tinta[dentro & (uu > largo - b)].any()),
        bool(tinta[dentro & (vv < b)].any()),
        bool(tinta[dentro & (vv > alto - b)].any()),
    ]
    return ta / max(1, ti + ta), bordes


def test_cajas_ajustadas_a_la_tinta(generado):
    """Cada caja de texto contiene su tinta (casi nada queda afuera) y la toca por los cuatro lados."""
    raiz, archivos = generado
    for a in archivos:
        grises = [np.asarray(img.convert("L"), np.float32) for img in paginas_visibles(raiz / a.ruta, a.formato)]
        umbral = 20 if "bajo_contraste" in a.id else 45
        for e in a.elementos:
            if e.tipo == "rostro":
                continue
            fuga, bordes = _ajuste(grises[e.pagina], e.poligono, umbral)
            assert fuga < 0.12, (a.id, e.tipo, e.valor, fuga)
            assert all(bordes), (a.id, e.tipo, e.valor, bordes)


def test_etiquetas_para_desgloses(generado):
    """Todo elemento lleva ángulo y degradación (el informe desglosa el recall por ellos)."""
    _, archivos = generado
    for a in archivos:
        for e in a.elementos:
            assert "degradacion" in e.etiquetas, (a.id, e.tipo, e.valor)
            if e.tipo != "rostro":
                assert "angulo" in e.etiquetas and "tam_px" in e.etiquetas, (a.id, e.valor)
    # la comuna tras la dirección se registra tal como está escrita (con la coma)
    comunas = [e for a in archivos for e in a.elementos if e.tipo == "texto" and e.valor.startswith(", ")]
    assert len(comunas) == 10  # 8 tarjetas en imagen_rotada, la del WEBP y la página 1 del TIFF


def test_formatos_de_estres_y_de_salida(generado):
    _, archivos = generado
    formatos = Counter(e.etiquetas.get("formato") for a in archivos for e in a.elementos)
    assert any(formatos[f] for f in ("sin_guion", "comas", "espacios_internos"))
    assert any(formatos[f] for f in ("antiguo_movil_09", "antiguo_movil_8", "antiguo_regional_0"))
    assert any(formatos[f] for f in ("arroba_texto", "at", "espacios"))
    por_angulo: dict[int, set[str]] = {}
    for a in archivos:
        if a.categoria == "imagen_rotada" and not a.etiquetas["estres"]:
            por_angulo.setdefault(a.etiquetas["angulo"], set()).add(a.formato)
    assert all(len(f) == 2 for f in por_angulo.values()), por_angulo  # nota y tarjeta en formatos distintos


def test_valores_unicos_y_en_imagen(generado):
    _, archivos = generado
    vistos: set[tuple[str, str]] = set()
    for a in archivos:
        for e in a.elementos:
            if e.tipo in ("rut", "correo", "telefono"):
                clave = (e.tipo, e.valor)
                assert clave not in vistos, clave
                vistos.add(clave)
    canarios = [m.valor for a in archivos for m in a.metadatos_sensibles if m.valor]
    visibles = {e.valor for a in archivos for e in a.elementos}
    assert not set(canarios) & visibles  # los canarios de metadatos no se reutilizan en el contenido


def _en_bytes(datos: bytes, valor: str) -> bool:
    return any(valor.encode(c) in datos for c in ("utf-8", "latin-1", "utf-16-le", "utf-16-be") if _codifica(valor, c))


def _codifica(valor: str, codificacion: str) -> bool:
    try:
        valor.encode(codificacion)
    except UnicodeEncodeError:
        return False
    return True


def test_canarios_presentes_en_bytes(generado):
    raiz, archivos = generado
    for a in archivos:
        datos = (raiz / a.ruta).read_bytes()
        for m in a.metadatos_sensibles:
            if m.valor:
                assert _en_bytes(datos, m.valor), (a.id, m.donde, m.valor)


def test_orientacion_6_exif_completo(generado):
    raiz, archivos = generado
    a = _por_id(archivos)["img_exif_orientacion_6"]
    ruta = raiz / a.ruta
    meta = {m.donde: m for m in a.metadatos_sensibles}
    ex = piexif.load(str(ruta))
    assert ex["0th"][piexif.ImageIFD.Orientation] == 6
    assert ex["0th"][piexif.ImageIFD.Artist].decode("utf-8") == meta["exif.artist"].valor
    assert meta["exif.image_description"].valor in ex["0th"][piexif.ImageIFD.ImageDescription].decode()
    assert piexif.helper.UserComment.load(ex["Exif"][piexif.ExifIFD.UserComment]) == meta["exif.user_comment"].valor
    xp = bytes(ex["0th"][piexif.ImageIFD.XPAuthor]).decode("utf-16-le").rstrip("\x00")
    assert xp == meta["exif.xp_author"].valor
    assert meta["exif.copyright"].valor in ex["0th"][piexif.ImageIFD.Copyright].decode()
    assert ex["GPS"][piexif.GPSIFD.GPSLatitudeRef] == b"S"
    lat = sum(n / d / 60**i for i, (n, d) in enumerate(ex["GPS"][piexif.GPSIFD.GPSLatitude]))
    assert lat == pytest.approx(-meta["exif.gps"].etiquetas["lat"], abs=1e-4)
    assert ex["1st"][piexif.ImageIFD.Compression] == 6  # IFD1 declara una miniatura JPEG
    miniatura = Image.open(io.BytesIO(ex["thumbnail"]))
    assert max(miniatura.size) <= 160 and miniatura.width < miniatura.height  # guardada sin girar (vertical)
    with Image.open(ruta) as im:
        assert im.size == (a.paginas[0].alto, a.paginas[0].ancho)
        assert meta["xmp.dc:creator"].valor in im.info["xmp"].decode("utf-8")
        assert ImageOps.exif_transpose(im).size == (a.paginas[0].ancho, a.paginas[0].alto)


@pytest.mark.parametrize("orientacion", [3, 8])
def test_orientaciones_con_gps(generado, orientacion):
    raiz, archivos = generado
    a = _por_id(archivos)[f"img_exif_orientacion_{orientacion}"]
    ex = piexif.load(str(raiz / a.ruta))
    assert ex["0th"][piexif.ImageIFD.Orientation] == orientacion
    assert piexif.GPSIFD.GPSLatitude in ex["GPS"]
    with Image.open(raiz / a.ruta) as im:
        esperado = (a.paginas[0].ancho, a.paginas[0].alto)
        assert im.size == (esperado if orientacion == 3 else esperado[::-1])


def test_orientacion_incorrecta(generado):
    raiz, archivos = generado
    a = _por_id(archivos)["img_exif_orientacion_incorrecta"]
    ex = piexif.load(str(raiz / a.ruta))
    assert ex["0th"][piexif.ImageIFD.Orientation] == 6
    with Image.open(raiz / a.ruta) as im:
        assert im.size == (a.paginas[0].alto, a.paginas[0].ancho)
    assert a.etiquetas["exif_incorrecto"] and all(e.etiquetas["exif_incorrecto"] for e in a.elementos)
    # en la geometría mostrada el texto queda de lado: las cajas son más altas que anchas
    for e in a.elementos:
        if e.tipo in ("rut", "correo", "nombre"):
            p = np.asarray(e.poligono)
            assert np.ptp(p[:, 1]) > np.ptp(p[:, 0])


def test_webp_exif_xmp(generado):
    raiz, archivos = generado
    a = _por_id(archivos)["img_exif_webp_gps"]
    meta = {m.donde: m for m in a.metadatos_sensibles}
    with Image.open(raiz / a.ruta) as im:
        assert im.format == "WEBP"
        exif = im.getexif()
        assert _utf8(exif[315]) == meta["exif.artist"].valor
        assert exif.get_ifd(0x8825)
        assert meta["xmp.dc:creator"].valor in im.info["xmp"].decode("utf-8")


def test_png_texto_xmp_exif(generado):
    raiz, archivos = generado
    a = _por_id(archivos)["img_exif_png_metadatos"]
    meta = {m.donde: m for m in a.metadatos_sensibles}
    with Image.open(raiz / a.ruta) as im:
        assert im.info["Author"] == meta["png.texto.Author"].valor
        assert im.info["Comment"] == meta["png.texto.Comment"].valor
        assert meta["png.texto.Description"].valor in im.info["Description"]
        assert meta["xmp.dc:creator"].valor in im.info["XML:com.adobe.xmp"]
        assert im.getexif().get_ifd(0x8825)
    assert b"eXIf" in (raiz / a.ruta).read_bytes()


def test_rostro_gps_e_iptc(generado):
    raiz, archivos = generado
    por_id = _por_id(archivos)
    a = por_id["img_exif_rostro_gps"]
    ex = piexif.load(str(raiz / a.ruta))
    assert ex["GPS"] and ex["0th"][piexif.ImageIFD.Artist].decode() == a.metadatos_sensibles[1].valor
    assert [e.tipo for e in a.elementos] == ["rostro"]
    b = por_id["img_exif_iptc"]
    meta = {(m.donde, m.etiquetas["tipo"]): m for m in b.metadatos_sensibles}
    with Image.open(raiz / b.ruta) as im:
        iptc = IptcImagePlugin.getiptcinfo(im)
        assert iptc[(2, 80)].decode("utf-8") == meta[("iptc.byline", "nombre")].valor
        assert meta[("iptc.caption", "rut")].valor in iptc[(2, 120)].decode("utf-8")
        im.load()


def test_tiff_multipagina(generado):
    raiz, archivos = generado
    a = _por_id(archivos)["img_tiff_multipagina"]
    meta = {m.donde: m for m in a.metadatos_sensibles}
    with Image.open(raiz / a.ruta) as im:
        assert im.n_frames == 3
        marcos = [(f.size, f.info["compression"], dict(f.tag_v2)) for f in ImageSequence.Iterator(im)]
    assert [m[0] for m in marcos] == [(p.ancho, p.alto) for p in a.paginas]
    assert len({m[0] for m in marcos}) == 3
    assert [m[1] for m in marcos] == ["tiff_lzw", "tiff_adobe_deflate", "jpeg"]
    tags = marcos[0][2]
    assert _utf8(tags[315]) == meta["tiff.artist"].valor
    assert meta["tiff.image_description"].valor in _utf8(tags[270])
    assert meta["tiff.document_name"].valor in _utf8(tags[269])
    assert meta["tiff.page_name"].valor in _utf8(tags[285])
    assert 305 in tags and 315 not in marcos[1][2]
    assert Counter(e.pagina for e in a.elementos)[2] == 3  # rostro + leyenda en la página 2
    assert any(e.tipo == "rostro" and e.pagina == 2 for e in a.elementos)


def test_tiff_escaneo_bn(generado):
    raiz, archivos = generado
    a = _por_id(archivos)["img_tiff_escaneo_bn"]
    with Image.open(raiz / a.ruta) as im:
        assert im.mode == "1" and im.info["compression"] == "group4"
        assert tuple(round(v) for v in im.info["dpi"]) == (200, 200)
    tipos = Counter(e.tipo for e in a.elementos)
    assert tipos["rut"] == 2 and tipos["correo"] == 2 and tipos["telefono"] == 2
    assert all(e.nivel in ("base", "fuera_de_alcance") for e in a.elementos)
