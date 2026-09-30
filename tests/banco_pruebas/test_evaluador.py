"""El evaluador tiene que detectar cada tipo de fuga y aceptar una censura bien hecha.

Cada caso se construye aquí mismo (sin depender de los generadores): un PDF con texto, una
imagen con EXIF, un PDF con una imagen tapada por un rectángulo y un PDF con texto vectorizado.
Para cada uno se prueba la salida sin censurar (todo es fuga), una censura aparente (tapar
sin borrar: falla la comprobación específica) y una censura correcta (todo censurado).
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import piexif
import pymupdf
import pytest
from PIL import Image, ImageFilter, ImageOps

from banco_pruebas.esquema import (
    Archivo,
    Censura,
    Elemento,
    InformeCensura,
    Manifiesto,
    Metadato,
    Pagina,
    ResultadoArchivo,
)
from banco_pruebas.evaluacion import cobertura as cob
from banco_pruebas.evaluacion import evaluar, evaluar_archivo
from banco_pruebas.evaluacion.buscar import Pajar, agujas, textos_de_bytes
from banco_pruebas.evaluacion.pdf import cadenas_pdf
from banco_pruebas.evaluacion.reporte import escribir
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.lienzo import Lienzo, rect


@pytest.fixture
def f() -> Ficticios:
    return Ficticios(33).derivar("test_evaluador")


# ---------------------------------------------------------------------------
# utilidades de los casos
# ---------------------------------------------------------------------------


def _poligono(q: pymupdf.Quad) -> list[list[float]]:
    return [[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y]]


def _buscar(pagina: pymupdf.Page, texto: str) -> list[list[float]]:
    quads = pagina.search_for(texto, quads=True)
    assert quads, f"no se encontró {texto!r}"
    q = quads[0]
    for extra in quads[1:]:  # un valor partido en varias cajas por el motor: se unen
        q = pymupdf.Quad(q.rect | extra.rect)
    return _poligono(q)


def _censuras(archivo: Archivo, tipos_excluidos=("texto",)) -> list[Censura]:
    return [
        Censura(pagina=e.pagina, poligono=e.poligono, tipo=e.tipo, detector="oraculo")
        for e in archivo.elementos
        if e.tipo not in tipos_excluidos and e.poligono is not None and e.capa != "oculto"
    ]


def _resultado(archivo: Archivo, censuras: list[Censura] | None = None, **kw) -> ResultadoArchivo:
    return ResultadoArchivo(
        entrada=archivo.ruta,
        salida=kw.pop("salida", archivo.ruta),
        censuras=_censuras(archivo) if censuras is None else censuras,
        tiempo_s=kw.pop("tiempo_s", 0.1),
        **kw,
    )


def _por_tipo(ev, tipo: str):
    return [e for e in ev.elementos if e.tipo == tipo]


def _guardar(doc: pymupdf.Document, ruta: Path, **kw) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    doc.save(ruta, garbage=kw.pop("garbage", 4), deflate=True, **kw)


# ---------------------------------------------------------------------------
# 1. PDF con capa de texto
# ---------------------------------------------------------------------------


def _pdf_texto(f: Ficticios, dir_: Path, rotacion: int = 0) -> Archivo:
    p = f.persona()
    fuera = f.persona(en_lista=False)
    rut, correo, tel = p.rut("puntos"), p.correo("punto"), p.telefono.formatear("movil_internacional")
    monto = f.monto()
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    lineas = [
        (72, 100, "Informe mensual de honorarios"),
        (72, 140, f"Nombre: {p.nombre_completo}"),
        (72, 170, f"RUT: {rut}"),
        (72, 200, f"Correo: {correo}"),
        (72, 230, f"Teléfono: {tel}"),
        (72, 260, f"Monto bruto: {monto}"),
        (72, 290, f"Contraparte: {fuera.nombre_completo}"),
    ]
    for x, y, t in lineas:
        pg.insert_text((x, y), t, fontsize=12)
    # texto oculto: blanco sobre blanco y fuera de la página
    oculto_rut = f.persona().rut("sin_puntos")
    pg.insert_text((72, 400), oculto_rut, fontsize=12, color=(1, 1, 1))
    oculto_correo = f.persona().correo("inicial")
    pg.insert_text((72, 900), oculto_correo, fontsize=12)
    if rotacion:
        pg.set_rotation(rotacion)
    elementos = [
        Elemento(
            "texto", 0, _buscar(pg, "Informe mensual de honorarios"), "Informe mensual de honorarios", capa="texto"
        ),
        Elemento("texto", 0, _buscar(pg, "RUT:"), "RUT:", capa="texto"),
        Elemento(
            "nombre", 0, _buscar(pg, p.nombre_completo), p.nombre_completo, capa="texto", etiquetas={"en_lista": True}
        ),
        Elemento("rut", 0, _buscar(pg, rut), rut, capa="texto", etiquetas={"formato": "puntos", "dv_valido": True}),
        Elemento("correo", 0, _buscar(pg, correo), correo, capa="texto", etiquetas={"formato": "punto"}),
        Elemento("telefono", 0, _buscar(pg, tel), tel, capa="texto", etiquetas={"formato": "movil_internacional"}),
        Elemento("texto", 0, _buscar(pg, monto), monto, capa="texto", etiquetas={"senuelo": "monto"}),
        Elemento(
            "nombre",
            0,
            _buscar(pg, fuera.nombre_completo),
            fuera.nombre_completo,
            nivel="fuera_de_alcance",
            capa="texto",
            etiquetas={"en_lista": False},
        ),
        Elemento("rut", 0, _buscar(pg, oculto_rut), oculto_rut, capa="oculto", etiquetas={"formato": "sin_puntos"}),
        Elemento("correo", 0, [[72, 890], [200, 890], [200, 905], [72, 905]], oculto_correo, capa="oculto"),
    ]
    ruta = "pdf/texto.pdf" if not rotacion else f"pdf/texto_rot{rotacion}.pdf"
    _guardar(doc, dir_ / ruta)
    return Archivo(
        id=f"t_pdf_texto_{rotacion}",
        ruta=ruta,
        formato="pdf",
        categoria="pdf_texto",
        descripcion="prueba",
        paginas=[Pagina(0, pg.cropbox.width, pg.cropbox.height, "pt", pg.rotation)],
        elementos=elementos,
    )


def _redactar(entrada: Path, salida: Path, archivo: Archivo, relleno=(0, 0, 0), incluir_oculto: bool = True) -> None:
    doc = pymupdf.open(entrada)
    for e in archivo.elementos:
        if e.tipo == "texto" or e.poligono is None or (e.capa == "oculto" and not incluir_oculto):
            continue
        r = pymupdf.Quad(
            *[pymupdf.Point(*pt) for pt in (e.poligono[0], e.poligono[1], e.poligono[3], e.poligono[2])]
        ).rect
        doc[e.pagina].add_redact_annot(r, fill=relleno)
    for pg in doc:
        pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
    doc.scrub()
    _guardar(doc, salida)


@pytest.mark.parametrize("rotacion", [0, 90, 270])
def test_pdf_texto_identidad_todo_es_fuga(tmp_path, f, rotacion):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_texto(f, ent, rotacion)
    shutil.copytree(ent, sal)
    ev = evaluar_archivo(a, _resultado(a, censuras=[]), sal)
    assert ev.estado == "procesado" and not ev.geometria_distinta
    for e in ev.elementos:
        if e.tipo == "texto":
            assert e.estado == "neutro" and e.extraible is True
            continue
        assert e.estado == "fuga", e
        if e.capa == "texto":
            assert set(e.fallas) >= {"C", "T", "B", "P"}, e.fallas
        else:
            assert set(e.fallas) == {"T", "B"}, e.fallas


@pytest.mark.parametrize("rotacion", [0, 90])
def test_pdf_texto_oraculo_todo_censurado(tmp_path, f, rotacion):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_texto(f, ent, rotacion)
    _redactar(ent / a.ruta, sal / a.ruta, a)
    ev = evaluar_archivo(a, _resultado(a), sal)
    for e in ev.elementos:
        if e.tipo == "texto":
            continue
        assert e.estado == "censurado", (e.id, e.fallas, e.detalle)
        assert e.detectado
    # el texto neutro sigue extraíble (conservación)
    assert all(e.extraible for e in ev.elementos if e.tipo == "texto")


def test_pdf_texto_rectangulo_encima_falla_t_y_b(tmp_path, f):
    """Dibujar un rectángulo negro sin borrar el texto: se ve censurado, pero el texto sigue ahí."""
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_texto(f, ent)
    doc = pymupdf.open(ent / a.ruta)
    for e in a.elementos:
        if e.tipo != "texto" and e.capa == "texto":
            doc[0].draw_rect(pymupdf.Rect(*cob.caja(e.poligono)), color=None, fill=(0, 0, 0))
    _guardar(doc, sal / a.ruta)
    ev = evaluar_archivo(a, _resultado(a), sal)
    for e in ev.elementos:
        if e.tipo != "texto" and e.capa == "texto":
            assert e.comprobaciones["C"] and e.comprobaciones["P"], e
            assert e.fallas == ["T", "B"], e.fallas


def test_pdf_texto_zonas_sin_pintar_falla_p(tmp_path, f):
    """Texto borrado de la capa de texto, pero la página quedó como imagen con el dato visible."""
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_texto(f, ent)
    src = pymupdf.open(ent / a.ruta)
    pix = src[0].get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
    doc = pymupdf.open()
    pg = doc.new_page(width=src[0].rect.width, height=src[0].rect.height)
    pg.insert_image(pg.rect, stream=pix.tobytes("png"))
    _guardar(doc, sal / a.ruta)
    ev = evaluar_archivo(a, _resultado(a), sal)
    for e in ev.elementos:
        if e.tipo != "texto" and e.capa == "texto":
            assert e.comprobaciones["T"] and e.comprobaciones["B"], (e.id, e.detalle)
            assert e.fallas == ["P"], (e.id, e.fallas, e.detalle)


def test_pdf_texto_oculto_no_borrado(tmp_path, f):
    """Censura de lo visible sin eliminar el texto oculto: fuga T/B solo en la capa oculta."""
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_texto(f, ent)
    _redactar(ent / a.ruta, sal / a.ruta, a, incluir_oculto=False)
    ev = evaluar_archivo(a, _resultado(a), sal)
    for e in ev.elementos:
        if e.capa == "oculto":
            assert e.estado == "fuga" and "T" in e.fallas, (e.id, e.fallas)
            assert not e.detectado
        elif e.tipo != "texto":
            assert e.estado == "censurado", (e.id, e.fallas)


def test_pdf_capa_opcional_apagada_se_extrae(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    rut = f.persona().rut("puntos")
    doc = pymupdf.open()
    pg = doc.new_page(width=400, height=400)
    ocg = doc.add_ocg("borrador", on=False)
    pg.insert_text((50, 100), rut, fontsize=12, oc=ocg)
    _guardar(doc, ent / "ocg.pdf")
    a = Archivo(
        "t_ocg",
        "ocg.pdf",
        "pdf",
        "pdf_metadatos",
        "",
        [Pagina(0, 400, 400, "pt")],
        [Elemento("rut", 0, rect(50, 90, 150, 104), rut, capa="oculto")],
    )
    shutil.copytree(ent, sal)
    ev = evaluar_archivo(a, _resultado(a, censuras=[]), sal)
    assert ev.elementos[0].estado == "fuga"
    assert any(h.startswith("texto_pymupdf") for h in ev.elementos[0].detalle["T"])


# ---------------------------------------------------------------------------
# 2. Imagen con EXIF (orientación, GPS, miniatura, autor)
# ---------------------------------------------------------------------------


def _jpeg_exif(f: Ficticios, dir_: Path) -> tuple[Archivo, Image.Image]:
    p = f.persona()
    lz = Lienzo.nuevo(640, 360)
    lz.escribir(30, 60, "Comprobante de atención", tam=26)
    lz.linea(30, 130, [("RUT: ", "texto"), (p.rut("puntos"), "rut")], tam=28)
    lz.escribir(30, 200, p.correo("punto"), tipo="correo", tam=24)
    lz.escribir_rotado(460, 280, p.telefono.formatear("movil_nacional"), 15, tipo="telefono", tam=22)
    visible = lz.img
    # se guarda girada con orientación EXIF 6: al aplicar el EXIF queda como ``visible``
    almacenada = visible.transpose(Image.Transpose.ROTATE_90)
    autor, descripcion = f.persona().nombre_completo, f.persona().correo("guion_bajo")
    miniatura = io.BytesIO()
    almacenada.resize((80, 142)).save(miniatura, "JPEG")
    exif = {
        "0th": {
            piexif.ImageIFD.Orientation: 6,
            piexif.ImageIFD.Artist: autor.encode("utf-8"),
            piexif.ImageIFD.XPComment: descripcion.encode("utf-16-le"),
            piexif.ImageIFD.Software: b"Editor 1.0",
        },
        "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"S", piexif.GPSIFD.GPSLatitude: ((36, 1), (49, 1), (0, 1))},
        "1st": {piexif.ImageIFD.JPEGInterchangeFormat: 0, piexif.ImageIFD.JPEGInterchangeFormatLength: 0},
        "thumbnail": miniatura.getvalue(),
    }
    ruta = dir_ / "img" / "exif.jpg"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    almacenada.save(ruta, "JPEG", quality=92, exif=piexif.dump(exif))
    assert ImageOps.exif_transpose(Image.open(ruta)).size == visible.size
    a = Archivo(
        "t_jpg_exif",
        "img/exif.jpg",
        "jpg",
        "imagenes",
        "",
        [Pagina(0, visible.width, visible.height, "px")],
        lz.elementos,
        [
            Metadato("exif.artist", autor),
            Metadato("exif.xpcomment", descripcion),
            Metadato("exif.gps", None),
            Metadato("exif.miniatura", None),
        ],
    )
    Manifiesto(raiz=str(dir_), semilla=33).agregar(a)  # asigna ids
    return a, visible


def _pintar(img: Image.Image, poligonos, color=(0, 0, 0)) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).copy()
    for pol in poligonos:
        m = np.zeros(arr.shape[:2], np.uint8)
        cob.rellenar(m, pol)
        arr[m.astype(bool)] = color
    return Image.fromarray(arr)


def test_imagen_identidad_fuga_de_pixeles_y_metadatos(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a, _ = _jpeg_exif(f, ent)
    shutil.copytree(ent, sal)
    ev = evaluar_archivo(a, _resultado(a, censuras=[]), sal)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.estado == "fuga" and set(e.fallas) == {"C", "P"}, (e.id, e.fallas)
    assert all(m.estado == "fuga" for m in ev.metadatos), [(m.donde, m.motivo) for m in ev.metadatos]


def test_imagen_tapada_en_el_informe_pero_no_en_los_pixeles(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a, _ = _jpeg_exif(f, ent)
    shutil.copytree(ent, sal)
    ev = evaluar_archivo(a, _resultado(a), sal)  # zonas correctas, archivo sin tocar
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.detectado and e.fallas == ["P"], (e.id, e.fallas)


def test_imagen_desenfocada_no_es_censura(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a, visible = _jpeg_exif(f, ent)
    arr = np.asarray(visible).copy()
    borrosa = np.asarray(visible.filter(ImageFilter.GaussianBlur(3)))
    for e in a.elementos:
        if e.tipo != "texto":
            m = np.zeros(arr.shape[:2], np.uint8)
            cob.rellenar(m, e.poligono)
            arr[m.astype(bool)] = borrosa[m.astype(bool)]
    (sal / "img").mkdir(parents=True)
    Image.fromarray(arr).save(sal / a.ruta, "JPEG", quality=92)
    ev = evaluar_archivo(a, _resultado(a), sal)
    assert all(m.estado == "eliminado" for m in ev.metadatos)
    assert any(e.fallas == ["P"] for e in ev.elementos if e.tipo != "texto")


def test_imagen_oraculo_censurada_y_sin_metadatos(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a, visible = _jpeg_exif(f, ent)
    limpia = _pintar(visible, [e.poligono for e in a.elementos if e.tipo != "texto"])
    (sal / "img").mkdir(parents=True)
    limpia.save(sal / a.ruta, "JPEG", quality=90)  # JPEG: el borde de la zona tiene artefactos
    ev = evaluar_archivo(a, _resultado(a), sal)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.estado == "censurado", (e.id, e.fallas, e.detalle)
    assert all(m.estado == "eliminado" for m in ev.metadatos), [(m.donde, m.motivo) for m in ev.metadatos]


def test_imagen_con_otro_tamano_es_geometria_distinta(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a, visible = _jpeg_exif(f, ent)
    (sal / "img").mkdir(parents=True)
    _pintar(visible, [e.poligono for e in a.elementos]).resize((320, 180)).save(sal / a.ruta, "JPEG")
    ev = evaluar_archivo(a, _resultado(a), sal)
    assert ev.geometria_distinta
    assert all("geometria_distinta" in e.fallas for e in ev.elementos if e.tipo != "texto")


def test_png_texto_comprimido_y_tiff(tmp_path, f):
    """Canarios en zTXt/iTXt (comprimidos: no se ven en los bytes) y en etiquetas TIFF."""
    from PIL import PngImagePlugin, TiffImagePlugin

    ent = tmp_path / "ent"
    ent.mkdir()
    nombre, correo, artista = f.persona().nombre_completo, f.persona().correo("punto"), f.persona().nombre_completo
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", nombre, zip=True)
    info.add_itxt("Comment", correo, zip=True)
    Image.new("RGB", (50, 40), "white").save(ent / "a.png", pnginfo=info)
    assert nombre.encode() not in (ent / "a.png").read_bytes()
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[315] = artista
    Image.new("RGB", (50, 40), "white").save(ent / "a.tiff", tiffinfo=ifd)
    archivos = [
        Archivo(
            "t_png",
            "a.png",
            "png",
            "imagenes",
            "",
            [Pagina(0, 50, 40, "px")],
            [],
            [Metadato("png.texto.Author", nombre), Metadato("png.texto.Comment", correo)],
        ),  # fmt: skip
        Archivo(
            "t_tiff", "a.tiff", "tiff", "tiff", "", [Pagina(0, 50, 40, "px")], [], [Metadato("tiff.artist", artista)]
        ),  # fmt: skip
    ]
    for a in archivos:
        ev = evaluar_archivo(a, _resultado(a, censuras=[]), ent)
        assert all(m.estado == "fuga" for m in ev.metadatos), [(m.donde, m.motivo) for m in ev.metadatos]
    limpia = tmp_path / "sal"
    limpia.mkdir()
    Image.new("RGB", (50, 40), "white").save(limpia / "a.png")
    Image.new("RGB", (50, 40), "white").save(limpia / "a.tiff")
    for a in archivos:
        ev = evaluar_archivo(a, _resultado(a, censuras=[]), limpia)
        assert all(m.estado == "eliminado" for m in ev.metadatos), [(m.donde, m.motivo) for m in ev.metadatos]


# ---------------------------------------------------------------------------
# 3. PDF con imagen incrustada (escaneo con texto y un rostro) tapada por un rectángulo
# ---------------------------------------------------------------------------


def _pdf_con_imagen(f: Ficticios, dir_: Path) -> Archivo:
    p = f.persona()
    rut = p.rut("puntos")
    lz = Lienzo.nuevo(800, 400, fondo=(250, 250, 247))
    lz.linea(40, 80, [("RUN ", "texto"), (rut, "rut")], tam=36)
    rng = np.random.default_rng(3)
    cara = Image.fromarray((rng.random((160, 128, 3)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
    lz.img.paste(cara, (600, 150))
    buf = io.BytesIO()
    lz.img.save(buf, "PNG")
    doc = pymupdf.open()
    pg = doc.new_page(width=600, height=300)
    destino = pymupdf.Rect(0, 0, 600, 300)  # 800x400 px -> 600x300 pt (0,75 pt/px)
    pg.insert_image(destino, stream=buf.getvalue())
    esc = 600 / 800
    elementos = [
        Elemento(e.tipo, 0, [[x * esc, y * esc] for x, y in e.poligono], e.valor, capa="raster") for e in lz.elementos
    ]
    elementos.append(
        Elemento(
            "rostro",
            0,
            rect(600 * esc, 150 * esc, 728 * esc, 310 * esc),
            None,
            capa="raster",
            nucleo=rect(630 * esc, 190 * esc, 698 * esc, 280 * esc),
            etiquetas={"pose": "frente"},
        )
    )
    _guardar(doc, dir_ / "escaneo.pdf")
    return Archivo("t_pdf_img", "escaneo.pdf", "pdf", "pdf_escaneado", "", [Pagina(0, 600, 300, "pt")], elementos)


def test_pdf_imagen_tapada_con_rectangulo_falla_i(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_con_imagen(f, ent)
    doc = pymupdf.open(ent / a.ruta)
    for e in a.elementos:
        if e.tipo != "texto":
            doc[0].draw_rect(pymupdf.Rect(*cob.caja(e.poligono)), color=None, fill=(0, 0, 0))
    _guardar(doc, sal / a.ruta)
    ev = evaluar_archivo(a, _resultado(a), sal, ent)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.comprobaciones["C"] and e.comprobaciones["P"], (e.id, e.detalle)
            # la imagen original sigue en la página (I) y con sus mismos píxeles (O)
            assert e.fallas == ["I", "O"], (e.id, e.fallas, e.detalle)


def test_pdf_imagen_identidad_y_oraculo(tmp_path, f):
    ent, sal, sal2 = tmp_path / "ent", tmp_path / "sal", tmp_path / "sal2"
    a = _pdf_con_imagen(f, ent)
    shutil.copytree(ent, sal)
    ev = evaluar_archivo(a, _resultado(a, censuras=[]), sal, ent)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert set(e.fallas) == {"C", "P", "I", "O"}, (e.id, e.fallas)
    _redactar(ent / a.ruta, sal2 / a.ruta, a)
    ev = evaluar_archivo(a, _resultado(a), sal2, ent)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.estado == "censurado", (e.id, e.fallas, e.detalle)
            assert e.comprobaciones["O"] is True


@pytest.mark.parametrize("modo", ["huerfana", "incremental"])
def test_pdf_imagen_original_huerfana_falla_o(tmp_path, f, modo):
    """Redacción correcta de los píxeles, pero la imagen original queda en el archivo: sin
    recolectar basura (objeto huérfano) o en la revisión anterior de un guardado incremental."""
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_con_imagen(f, ent)
    if modo == "huerfana":
        doc = pymupdf.open(ent / a.ruta)
        destino = sal / a.ruta
    else:
        sal.mkdir()
        shutil.copy(ent / a.ruta, sal / a.ruta)
        doc = pymupdf.open(sal / a.ruta)
        destino = None
    for e in a.elementos:
        if e.tipo != "texto":
            doc[0].add_redact_annot(pymupdf.Rect(*cob.caja(e.poligono)), fill=(0, 0, 0))
    doc[0].apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
    if destino is None:
        doc.saveIncr()
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)
        doc.save(destino, garbage=0)
    doc.close()
    ev = evaluar_archivo(a, _resultado(a), sal, ent)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.comprobaciones["I"] and e.comprobaciones["P"], (e.id, e.detalle)
            assert e.fallas == ["O"], (e.id, e.fallas, e.detalle)
    # sin la carpeta de entrada, O no se puede evaluar y queda una advertencia
    ev = evaluar_archivo(a, _resultado(a), sal)
    assert any("comprobación O" in x for x in ev.advertencias)


def test_rostro_regla_nucleo_y_caja(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_con_imagen(f, ent)
    rostro = next(e for e in a.elementos if e.tipo == "rostro")
    x0, y0, x1, y1 = cob.caja(rostro.poligono)
    shutil.copytree(ent, sal)
    solo_nucleo = [Censura(0, rostro.nucleo, "rostro", "prueba")]
    ev = evaluar_archivo(a, _resultado(a, censuras=solo_nucleo), sal)
    r = _por_tipo(ev, "rostro")[0]
    assert r.cobertura_nucleo == pytest.approx(1.0) and r.cobertura < 0.8 and not r.detectado
    casi = [Censura(0, rect(x0, y0 + 0.1 * (y1 - y0), x1, y1), "rostro", "prueba")]  # 90 % de la cara
    ev = evaluar_archivo(a, _resultado(a, censuras=casi), sal)
    assert _por_tipo(ev, "rostro")[0].detectado


# ---------------------------------------------------------------------------
# 4. PDF con texto vectorizado (glifos como trazos)
# ---------------------------------------------------------------------------


def _pdf_vector(f: Ficticios, dir_: Path) -> Archivo:
    p = f.persona()
    rut, tel = (
        p.rut("sin_puntos"),
        p.telefono_fijo.formatear(
            "santiago_internacional" if p.telefono_fijo.clase == "santiago" else "regional_internacional"
        ),
    )
    doc = pymupdf.open()
    pg = doc.new_page(width=420, height=300)
    pg.insert_text((40, 80), "Certificado", fontsize=16)
    pg.insert_text((40, 130), f"RUT {rut}", fontsize=14)
    pg.insert_text((40, 170), f"Fono {tel}", fontsize=14)
    elementos = [
        Elemento("texto", 0, _buscar(pg, "Certificado"), "Certificado", capa="vector"),
        Elemento("rut", 0, _buscar(pg, rut), rut, capa="vector"),
        Elemento("telefono", 0, _buscar(pg, tel), tel, capa="vector"),
    ]
    svg = pg.get_svg_image(text_as_path=True)
    vec = pymupdf.open("pdf", pymupdf.open(stream=svg.encode(), filetype="svg").convert_to_pdf())
    assert vec[0].get_text().strip() == ""
    _guardar(vec, dir_ / "vector.pdf")
    return Archivo("t_pdf_vec", "vector.pdf", "pdf", "pdf_texto", "", [Pagina(0, 420, 300, "pt")], elementos)


def test_pdf_vectorial(tmp_path, f):
    ent = tmp_path / "ent"
    a = _pdf_vector(f, ent)
    # identidad: quedan los glifos
    ev = evaluar_archivo(a, _resultado(a, censuras=[]), ent)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert set(e.fallas) == {"C", "P", "V"}, (e.id, e.fallas)
    # rectángulo negro encima de los trazos: se ve bien pero los glifos siguen en el archivo
    tapado = tmp_path / "tapado"
    doc = pymupdf.open(ent / a.ruta)
    for e in a.elementos:
        if e.tipo != "texto":
            doc[0].draw_rect(pymupdf.Rect(*cob.caja(e.poligono)), color=None, fill=(0, 0, 0))
    _guardar(doc, tapado / a.ruta)
    ev = evaluar_archivo(a, _resultado(a), tapado)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.fallas == ["V"], (e.id, e.fallas, e.detalle)
    # redacción que elimina los trazos cubiertos
    limpio = tmp_path / "limpio"
    doc = pymupdf.open(ent / a.ruta)
    for e in a.elementos:
        if e.tipo != "texto":
            x0, y0, x1, y1 = cob.caja(e.poligono)
            doc[0].add_redact_annot(pymupdf.Rect(x0 - 1, y0 - 2, x1 + 1, y1 + 2), fill=(0, 0, 0))
    doc[0].apply_redactions(graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
    _guardar(doc, limpio / a.ruta)
    ev = evaluar_archivo(a, _resultado(a), limpio)
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.estado == "censurado", (e.id, e.fallas, e.detalle)


# ---------------------------------------------------------------------------
# 5. Metadatos PDF y revisiones anteriores
# ---------------------------------------------------------------------------


def test_pdf_metadatos_y_revision_anterior(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    autor, nota, adjunto, marcador = (f.persona().nombre_completo for _ in range(4))
    rut_previo = f.persona().rut("puntos")
    doc = pymupdf.open()
    pg = doc.new_page(width=300, height=300)
    pg.insert_text((40, 60), rut_previo, fontsize=11)
    ent.mkdir()
    doc.set_metadata({"author": autor, "producer": "Generador"})
    pg.add_text_annot((100, 100), nota)
    doc.embfile_add("respaldo.txt", f"Responsable: {adjunto}".encode("utf-16"))
    doc.set_toc([[1, marcador, 1]])
    doc.save(ent / "meta.pdf")
    # guardado incremental que borra el texto: la versión anterior sigue en el archivo
    doc = pymupdf.open(ent / "meta.pdf")
    doc[0].add_redact_annot(doc[0].search_for(rut_previo)[0])
    doc[0].apply_redactions()
    doc.save(ent / "meta.pdf", incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
    a = Archivo(
        "t_pdf_meta",
        "meta.pdf",
        "pdf",
        "pdf_metadatos",
        "",
        [Pagina(0, 300, 300, "pt")],
        [],
        [
            Metadato("pdf.info.author", autor),
            Metadato("pdf.anotacion", nota),
            Metadato("pdf.adjunto", adjunto),
            Metadato("pdf.marcador", marcador),
            Metadato("pdf.revision_anterior", rut_previo, {"tipo": "rut"}),
        ],
    )
    shutil.copytree(ent, sal)
    ev = evaluar_archivo(a, _resultado(a, censuras=[]), sal)
    assert all(m.estado == "fuga" for m in ev.metadatos), [(m.donde, m.motivo) for m in ev.metadatos]
    revision = ev.metadatos[-1]
    assert any("canario" in x for x in revision.motivo) and any("estructura" in x for x in revision.motivo)
    assert any("Producer" in adv or "producer" in adv for adv in ev.advertencias)
    # limpieza completa: sin metadatos, anotaciones, adjuntos, marcadores ni revisiones
    limpio = tmp_path / "limpio"
    doc = pymupdf.open(ent / "meta.pdf")
    doc.set_metadata({})
    doc.del_xml_metadata()
    for pagina in doc:
        for anot in list(pagina.annots()):
            pagina.delete_annot(anot)
    for nombre in doc.embfile_names():
        doc.embfile_del(nombre)
    doc.set_toc([])
    _guardar(doc, limpio / "meta.pdf")
    ev = evaluar_archivo(a, _resultado(a, censuras=[]), limpio)
    assert all(m.estado == "eliminado" for m in ev.metadatos), [(m.donde, m.motivo) for m in ev.metadatos]


# ---------------------------------------------------------------------------
# 6. Búsqueda de canarios y fragmentos
# ---------------------------------------------------------------------------


def test_fragmentos_criticos():
    partes = {a.parte: a.forma for a in agujas("rut", "12.345.678-5")}
    assert partes == {"valor": "123456785", "cuerpo_rut": "12345678"}
    assert {a.parte: a.forma for a in agujas("telefono", "+56 9 8123 4567")}["ultimos7"] == "1234567"
    assert {a.parte: a.forma for a in agujas("correo", "Ana.Rojas@ejemplo.cl")}["local@"] == "ana.rojas@"
    assert {a.parte: a.forma for a in agujas("correo", "ana.rojas [arroba] ejemplo.cl")}[
        "local@"
    ] == "ana.rojas[arroba]"
    assert {a.parte: a.forma for a in agujas("nombre", "Ana María Rojas Peña")}["apellidos"] == "rojas pena"
    assert {a.parte: a.forma for a in agujas("nombre", "ROJAS PEÑA, Ana María")}["apellidos"] == "rojas pena"
    calle = {a.parte: a.forma for a in agujas("direccion", "Pasaje Los Alerces 1234, depto 56")}["calle_numero"]
    assert calle == "pasaje los alerces 1234"


@pytest.mark.parametrize(
    "texto",
    ["RUT 12,345,678-5", "12 345 678 - 5", "12.345.678–5", "1 2 . 3 4 5 . 6 7 8 - 5", "N°12345678K otro"],
)
def test_rut_en_texto_con_separadores(texto):
    pajar = Pajar()
    pajar.agregar("t", texto)
    assert pajar.contiene(agujas("rut", "12.345.678-5"))


def test_sin_falsos_positivos_por_unir_numeros():
    pajar = Pajar()
    pajar.agregar("t", "Folio 1234 del 5 de mayo; monto $ 678.000")
    assert not pajar.contiene(agujas("rut", "12.345.678-5"))


def test_numeros_de_un_flujo_no_se_unen():
    """En bytes crudos (operadores de contenido) no se unen dígitos: se busca el valor tal como se escribió."""
    tel = "+56 9 8123 4567"
    pajar = Pajar()
    pajar.agregar("flujos", "q 1 0 0 1 98 12 34567 cm Q", binario=True)
    assert not pajar.contiene(agujas("telefono", tel))
    pajar.agregar("flujos", "BT (+56 9 8123 4567) Tj ET", binario=True)
    assert pajar.contiene(agujas("telefono", tel))
    rut = Pajar()
    rut.agregar("flujos", "<x:rut>12.345.678</x:rut>", binario=True)  # cuerpo sin DV, tal como se escribió
    assert rut.contiene(agujas("rut", "12.345.678-5"))


def test_bytes_en_varias_codificaciones():
    valor = "Ana María Rojas Peña"
    for cod in ("utf-8", "latin-1", "utf-16-le", "utf-16-be"):
        datos = b"\x00\x01basura" + valor.encode(cod) + b"\xff\xfe"
        pajar = Pajar()
        pajar.agregar("b", textos_de_bytes(datos))
        assert pajar.contiene(agujas("nombre", valor)), cod


def test_cadenas_pdf_unidas_en_tj():
    flujo = rb"BT /F1 12 Tf 72 700 Td [(12.3) -20 (45.6) 15 (78-5)] TJ ET BT (Pe\361a) Tj ET <FEFF00410042>"
    textos = cadenas_pdf(flujo)
    assert "12.345.678-5" in textos and "Peña" in textos and "AB" in textos


def test_cobertura_rasterizada():
    pol = [[10, 10], [110, 10], [110, 30], [10, 30]]
    assert cob.cobertura(pol, [pol], 8) == pytest.approx(1.0)
    assert cob.cobertura(pol, [rect(10, 10, 60, 30)], 8) == pytest.approx(0.5, abs=0.01)
    assert cob.cobertura(pol, [rect(10, 10, 60, 30), rect(55, 10, 110, 30)], 8) == pytest.approx(1.0)
    girado = [[0, 0], [10, 10], [0, 20], [-10, 10]]
    assert cob.cobertura(girado, [rect(-10, 0, 10, 20)], 8) == pytest.approx(1.0)
    assert cob.cobertura(rect(-10, 0, 10, 20), [girado], 8) == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# 7. Conjunto completo: errores esperados, archivos faltantes, agregación y reporte
# ---------------------------------------------------------------------------


def test_conjunto_completo_y_reporte(tmp_path, f):
    ent = tmp_path / "ent"
    man = Manifiesto(raiz=str(ent), semilla=33)
    a_pdf = man.agregar(_pdf_texto(f, ent))
    a_img, visible = _jpeg_exif(f, ent)
    a_img.id = "t_jpg_exif2"
    man.agregar(a_img)
    (ent / "clave.pdf").write_bytes(b"%PDF-1.7 cifrado")
    a_err = man.agregar(Archivo("t_err", "clave.pdf", "pdf", "errores", "", [], esperado="error:contrasena"))
    a_err2 = man.agregar(Archivo("t_err2", "vacio.pdf", "pdf", "errores", "", [], esperado="error:vacio"))
    ruta_man = man.guardar()
    man = Manifiesto.cargar(ruta_man)

    # sistema que censura bien el PDF y la imagen, y rechaza bien un error
    sal = tmp_path / "sal"
    _redactar(ent / a_pdf.ruta, sal / a_pdf.ruta, a_pdf)
    (sal / "img").mkdir(parents=True, exist_ok=True)
    _pintar(visible, [e.poligono for e in a_img.elementos if e.tipo != "texto"]).save(sal / a_img.ruta, "PNG")
    informe = InformeCensura(
        sistema="prueba",
        resultados=[
            _resultado(a_pdf),
            _resultado(a_img),
            ResultadoArchivo(entrada=a_err.ruta, salida=None, error="contrasena: el archivo pide clave"),
            ResultadoArchivo(entrada=a_err2.ruta, salida=None, error="vacio"),
        ],
        detalles={"equipo": "CPU de prueba"},
    )
    r = evaluar(man, informe, sal, procesos=1)
    assert r["veredicto"]["aprobado"], r["veredicto"]
    assert r["resumen"]["fugas"] == 0 and r["resumen"]["recall"] == 1.0
    assert r["errores_esperados"]["correctos"] == 2
    ruta_json, ruta_md = escribir(r, tmp_path / "reporte")
    assert json.loads(ruta_json.read_text(encoding="utf-8"))["veredicto"]["aprobado"]
    assert "APROBADO" in ruta_md.read_text(encoding="utf-8")

    # identidad: todo es fuga, sin resultado para la imagen y un error esperado no rechazado
    ident = tmp_path / "ident"
    shutil.copytree(ent, ident)
    informe = InformeCensura(
        sistema="identidad",
        resultados=[_resultado(a_pdf, censuras=[]), ResultadoArchivo(entrada=a_err.ruta, salida=a_err.ruta)],
    )
    r = evaluar(man, informe, ident, procesos=2)
    v = r["veredicto"]
    assert not v["aprobado"]
    # los metadatos de la imagen no cuentan: no hay salida que publicar
    assert [c["cumple"] for c in v["criterios"]] == [False, True, False, True, True]
    assert r["resumen"]["fugas_criticas"] == 5  # 3 visibles + 2 ocultas
    assert r["procesables_no_procesados"][0]["archivo"] == "t_jpg_exif2"
    assert {x["estado"] for x in r["errores_esperados"]["archivos"]} == {"no_rechazado", "sin_resultado"}
    img_elems = [e for e in r["elementos"] if e["archivo"] == "t_jpg_exif2" and e["tipo"] != "texto"]
    assert img_elems and all(e["estado"] == "no_procesado" for e in img_elems)
    _, ruta_md = escribir(r, tmp_path / "reporte_ident")
    md = ruta_md.read_text(encoding="utf-8")
    assert "NO APROBADO" in md and "Detalle de fugas" in md
    for e in a_pdf.elementos:
        if e.tipo in ("rut", "correo", "telefono") and e.capa == "texto":
            assert e.valor in md


# ---------------------------------------------------------------------------
# 8. Controles contra la entrada, páginas giradas re-guardadas y lugares de metadatos
# ---------------------------------------------------------------------------


def _png_bajo_contraste(f: Ficticios, dir_: Path) -> Archivo:
    """Un RUT gris muy claro: con la tolerancia de P (40 niveles) la zona parece uniforme."""
    lz = Lienzo.nuevo(420, 120)
    lz.escribir(20, 70, f.persona().rut("puntos"), tipo="rut", tam=34, color=(222, 222, 222))
    ruta = dir_ / "bajo_contraste.png"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    lz.img.save(ruta)
    a = Archivo("t_png_bajo", "bajo_contraste.png", "png", "imagenes", "", [Pagina(0, 420, 120, "px")], lz.elementos)
    Manifiesto(raiz=str(dir_), semilla=33).agregar(a)
    return a


def test_imagen_bajo_contraste_sin_cambios_falla_p(tmp_path, f):
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _png_bajo_contraste(f, ent)
    shutil.copytree(ent, sal)
    # sin la entrada, P no distingue el dato tenue de una zona pintada (límite de la tolerancia)
    ev = evaluar_archivo(a, _resultado(a), sal)
    assert ev.elementos[0].comprobaciones["P"] is True
    # con la entrada, los píxeles iguales al original delatan que no se censuró
    ev = evaluar_archivo(a, _resultado(a), sal, ent)
    e = ev.elementos[0]
    assert e.fallas == ["P"] and e.detalle["correlacion_original"] > 0.99, (e.fallas, e.detalle)
    # recomprimido en JPEG sin tocar la zona: sigue siendo el original
    Image.open(ent / a.ruta).save(sal / a.ruta, "JPEG", quality=70)
    assert evaluar_archivo(a, _resultado(a), sal, ent).elementos[0].fallas == ["P"]
    # zona rellenada (aunque sea de un color parecido al fondo): censurado
    _pintar(Image.open(ent / a.ruta), [a.elementos[0].poligono], (240, 240, 240)).save(sal / a.ruta, "PNG")
    e = evaluar_archivo(a, _resultado(a), sal, ent).elementos[0]
    assert e.estado == "censurado", (e.fallas, e.detalle)


def _quitar_rotacion(doc: pymupdf.Document, indice: int = 0) -> None:
    """Deja la página con /Rotate 0 y el contenido girado dentro del flujo: se ve igual que antes."""
    pg = doc[indice]
    assert pg.rotation == 90
    ancho, alto = pg.mediabox.width, pg.mediabox.height
    pg.clean_contents()
    xref = pg.get_contents()[0]
    # giro horario de 90°: (x, y) -> (y, ancho - x) en el espacio de usuario PDF
    doc.update_stream(xref, f"q 0 -1 1 0 0 {ancho:g} cm\n".encode() + doc.xref_stream(xref) + b"\nQ")
    pg.set_rotation(0)
    pg.set_mediabox(pymupdf.Rect(0, 0, alto, ancho))


def test_pdf_girado_reguardado_sin_rotate(tmp_path, f):
    """La salida dibuja la página girada 90° en una hoja sin /Rotate (ancho y alto intercambiados):
    la página visible es la misma, así que no es geometría distinta."""
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_texto(f, ent, rotacion=90)
    limpio = tmp_path / "limpio"
    _redactar(ent / a.ruta, limpio / a.ruta, a)
    doc = pymupdf.open(limpio / a.ruta)
    antes = doc[0].get_pixmap(alpha=False).samples
    _quitar_rotacion(doc)
    assert doc[0].rotation == 0 and doc[0].rect.width > doc[0].rect.height
    assert doc[0].get_pixmap(alpha=False).samples == antes  # misma página visible
    _guardar(doc, sal / a.ruta)
    ev = evaluar_archivo(a, _resultado(a), sal, ent)
    assert not ev.geometria_distinta
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.estado == "censurado", (e.id, e.fallas, e.detalle)
    # la entrada tal cual pero en una hoja de otro tamaño: geometría distinta
    doc = pymupdf.open()
    doc.new_page(width=500, height=500).show_pdf_page(pymupdf.Rect(0, 0, 500, 500), pymupdf.open(limpio / a.ruta), 0)
    _guardar(doc, sal / a.ruta)
    ev = evaluar_archivo(a, _resultado(a), sal, ent)
    assert ev.geometria_distinta
    assert all("geometria_distinta" in e.fallas for e in ev.elementos if e.tipo != "texto" and e.capa != "oculto")


def test_pdf_girado_rectangulo_sobre_vector(tmp_path, f):
    """V también funciona si la salida guardó la página girada sin /Rotate."""
    ent, sal = tmp_path / "ent", tmp_path / "sal"
    a = _pdf_vector(f, ent)
    src = pymupdf.open(stream=(ent / a.ruta).read_bytes(), filetype="pdf")
    src[0].set_rotation(90)
    _guardar(src, ent / a.ruta)
    a.paginas[0].rotacion = 90
    matriz = src[0].rotation_matrix
    _quitar_rotacion(src)
    pg = src[0]
    for e in a.elementos:
        if e.tipo != "texto":
            vis = [pymupdf.Point(x, y) * matriz for x, y in e.poligono]
            pg.draw_rect(pymupdf.Rect(vis[0], vis[2]).normalize(), color=None, fill=(0, 0, 0))
    _guardar(src, sal / a.ruta)
    ev = evaluar_archivo(a, _resultado(a), sal, ent)
    assert not ev.geometria_distinta
    for e in ev.elementos:
        if e.tipo != "texto":
            assert e.fallas == ["V"], (e.id, e.fallas, e.detalle)


def test_lugares_de_metadatos_con_contenedor():
    from banco_pruebas.evaluacion.imagen import estructura_tiff
    from banco_pruebas.evaluacion.nucleo import estructura_presente

    est = {"exif.gps": True, "xmp": True, "tiff.270": True}
    assert estructura_presente("png.exif.gps", est) is True
    assert estructura_presente("webp.xmp", est) is True
    assert estructura_presente("exif.miniatura", est) is False
    assert estructura_tiff("tiff.image_description") == "tiff.270"
    assert estructura_presente("tiff.image_description", est) is True
    assert estructura_presente("otro.lugar", est) is None
