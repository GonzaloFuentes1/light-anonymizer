"""Pruebas de las tres líneas base (identidad, oráculo, cuaderno) con un conjunto mínimo armado aquí."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import piexif
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageOps

from banco_pruebas import visualizar
from banco_pruebas.esquema import Archivo, Elemento, InformeCensura, Manifiesto, Pagina, normalizar
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.lienzo import fuente
from banco_pruebas.linea_base import ejecutar, main
from banco_pruebas.lineas_base import cuaderno, oraculo

# Documento de prueba del cuaderno (celda 6), copiado tal cual.
TEXTO_CUADERNO = """INFORME DE HONORARIOS - AGOSTO 2026

Nombre: Ana Maria Rojas Pena
RUT: 15.782.334-9
Correo: ana.rojas@ejemplo.cl
Telefono: +56 9 8123 4567
Direccion: Pasaje Los Alerces 442, depto 31

Producto 1: Informe de avance del programa
Monto bruto: $ 1.450.000

Contraparte: Jefatura de la unidad
RUT contraparte: 9.876.543-3
Sitio: https://www.ejemplo.cl/rendiciones
"""

HALLAZGOS_CUADERNO = [
    (1, "rut", "15.782.334-9"),
    (1, "rut", "9.876.543-3"),
    (1, "correo", "ana.rojas@ejemplo.cl"),
    (1, "telefono", "+56 9 8123 4567"),
    (1, "url", "https://www.ejemplo.cl/rendiciones"),
]


# ---------------------------------------------------------------------------
# Construcción del conjunto mínimo
# ---------------------------------------------------------------------------


def _quad(q: pymupdf.Quad) -> list[list[float]]:
    return [[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y]]


def _elementos_texto(pagina: pymupdf.Page, indice: int, datos: list[tuple[str, str, dict]]) -> list[Elemento]:
    """Registra cada (tipo, valor, etiquetas) con el cuadrilátero que da ``search_for``."""
    salida = []
    for tipo, valor, etiquetas in datos:
        quads = pagina.search_for(valor, quads=True)
        assert quads, valor
        salida.append(
            Elemento(tipo=tipo, pagina=indice, poligono=_quad(quads[0]), valor=valor, capa="texto", etiquetas=etiquetas)
        )
    return salida


def _pdf_texto(raiz: Path, f: Ficticios) -> Archivo:
    p = f.persona()
    rut, correo, fono = p.rut("puntos"), p.correo("punto"), p.telefono.formatear("movil_internacional")
    doc = pymupdf.open()
    pag = doc.new_page(width=595, height=842)
    y = 90
    lineas = [f"Nombre: {p.nombre_completo}", f"RUT: {rut}", f"Correo: {correo}", f"Telefono: {fono}"]
    for linea in lineas:
        pag.insert_text((60, y), linea, fontsize=11, fontname="helv")
        y += 20
    # Firma como imagen incrustada (capa raster dentro del PDF).
    firma = Image.new("RGB", (300, 100), "white")
    d = ImageDraw.Draw(firma)
    d.line([(10, 80), (80, 20), (150, 70), (220, 15), (290, 60)], fill=(10, 10, 60), width=6)
    buf = io.BytesIO()
    firma.save(buf, "PNG")
    caja = pymupdf.Rect(300, 300, 450, 350)
    pag.insert_image(caja, stream=buf.getvalue())
    doc.set_metadata({"author": p.nombre_completo, "title": "informe"})
    doc.set_xml_metadata(f"<x:xmpmeta xmlns:x='adobe:ns:meta/'><dc>{p.nombre_completo}</dc></x:xmpmeta>")
    elementos = _elementos_texto(
        pag,
        0,
        [
            ("texto", "Nombre:", {}),
            ("nombre", p.nombre_completo, {"en_lista": True}),
            ("texto", "RUT:", {}),
            ("rut", rut, {"formato": "puntos", "dv_valido": True}),
            ("correo", correo, {"formato": "punto"}),
            ("telefono", fono, {"formato": "movil_internacional", "clase": "movil"}),
        ],
    )
    elementos.append(
        Elemento(
            tipo="firma",
            pagina=0,
            poligono=[[300, 300], [450, 300], [450, 350], [300, 350]],
            nivel="fuera_de_alcance",
            capa="raster",
        )
    )
    (raiz / "pdf").mkdir(parents=True, exist_ok=True)
    doc.save(raiz / "pdf/texto.pdf")
    doc.close()
    return Archivo(
        id="lb_texto",
        ruta="pdf/texto.pdf",
        formato="pdf",
        categoria="pdf_texto",
        descripcion="PDF con capa de texto y firma incrustada",
        paginas=[Pagina(0, 595, 842, "pt")],
        elementos=elementos,
    )


def _pdf_rotado(raiz: Path, f: Ficticios) -> Archivo:
    doc = pymupdf.open()
    paginas, elementos = [], []
    for i, rotacion in enumerate((90, 270, 180)):
        p = f.persona()
        rut = p.rut("sin_puntos")
        pag = doc.new_page(width=612, height=792)
        pag.insert_text((72, 100 + 40 * i), f"RUT: {rut}", fontsize=12, fontname="helv")
        pag.insert_text((72, 300), "Texto neutro de la pagina", fontsize=12, fontname="helv")
        pag.set_rotation(rotacion)
        elementos += _elementos_texto(
            pag,
            i,
            [
                ("rut", rut, {"formato": "sin_puntos", "dv_valido": True}),
                ("texto", "Texto neutro de la pagina", {}),
            ],
        )
        paginas.append(Pagina(i, 612, 792, "pt", rotacion))
    doc.save(raiz / "pdf/rotada.pdf")
    doc.close()
    return Archivo(
        id="lb_rotada",
        ruta="pdf/rotada.pdf",
        formato="pdf",
        categoria="pdf_texto",
        descripcion="páginas con /Rotate 90, 270 y 180",
        paginas=paginas,
        elementos=elementos,
    )


def _dibujar_dato(img: Image.Image, texto: str, x: int, y: int, tam: int = 28) -> list[list[float]]:
    d = ImageDraw.Draw(img)
    f = fuente("sans", tam)
    d.text((x, y), texto, fill=(0, 0, 0), font=f)
    x0, y0, x1, y1 = d.textbbox((x, y), texto, font=f)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _jpeg_exif(raiz: Path, f: Ficticios) -> Archivo:
    p = f.persona()
    rut, correo = p.rut("puntos"), p.correo("inicial")
    visible = Image.new("RGB", (640, 420), (235, 235, 230))
    poli_rut = _dibujar_dato(visible, f"RUT {rut}", 40, 60)
    poli_correo = _dibujar_dato(visible, correo, 40, 300, 22)
    guardada = visible.transpose(Image.Transpose.ROTATE_90)  # EXIF 6 la vuelve a su posición al mostrarla
    exif = piexif.dump(
        {
            "0th": {piexif.ImageIFD.Orientation: 6, piexif.ImageIFD.Artist: p.nombre_completo.encode()},
            "GPS": {
                piexif.GPSIFD.GPSLatitudeRef: b"S",
                piexif.GPSIFD.GPSLatitude: ((33, 1), (26, 1), (0, 1)),
                piexif.GPSIFD.GPSLongitudeRef: b"W",
                piexif.GPSIFD.GPSLongitude: ((70, 1), (39, 1), (0, 1)),
            },
        }
    )
    (raiz / "img").mkdir(parents=True, exist_ok=True)
    guardada.save(raiz / "img/exif6.jpg", "JPEG", quality=92, exif=exif)
    return Archivo(
        id="lb_exif6",
        ruta="img/exif6.jpg",
        formato="jpg",
        categoria="imagenes",
        descripcion="JPEG con orientación EXIF 6 y GPS",
        paginas=[Pagina(0, 640, 420, "px")],
        elementos=[
            Elemento(tipo="rut", pagina=0, poligono=poli_rut, valor=rut, etiquetas={"formato": "puntos"}),
            Elemento(tipo="correo", pagina=0, poligono=poli_correo, valor=correo, etiquetas={"formato": "inicial"}),
        ],
    )


def _tiff_multipagina(raiz: Path, f: Ficticios) -> Archivo:
    cuadros, paginas, elementos = [], [], []
    for i, tam in enumerate(((500, 300), (360, 520))):
        img = Image.new("RGB", tam, "white")
        fono = f.telefono("movil").formatear("movil_nacional")
        elementos.append(
            Elemento(
                tipo="telefono",
                pagina=i,
                poligono=_dibujar_dato(img, fono, 30, 40 + 60 * i),
                valor=fono,
                etiquetas={"formato": "movil_nacional", "clase": "movil"},
            )
        )
        cuadros.append(img)
        paginas.append(Pagina(i, tam[0], tam[1], "px"))
    cuadros[0].save(
        raiz / "img/multi.tiff", "TIFF", save_all=True, append_images=cuadros[1:], tiffinfo={315: "Autor Ficticio"}
    )
    return Archivo(
        id="lb_tiff",
        ruta="img/multi.tiff",
        formato="tiff",
        categoria="tiff",
        descripcion="TIFF de dos páginas",
        paginas=paginas,
        elementos=elementos,
    )


def _pdf_clave(raiz: Path) -> Archivo:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Documento protegido con clave de usuario", fontname="helv")
    doc.save(raiz / "pdf/clave.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="usuario", owner_pw="dueno")
    doc.close()
    return Archivo(
        id="lb_clave",
        ruta="pdf/clave.pdf",
        formato="pdf",
        categoria="errores",
        descripcion="PDF con clave de usuario",
        paginas=[Pagina(0, 595, 842, "pt")],
        esperado="error:contrasena",
    )


def _pdf_permisos(raiz: Path, f: Ficticios) -> Archivo:
    """Solo clave de dueño (restricciones de permisos): se abre sin clave y debe procesarse."""
    p = f.persona()
    correo = p.correo("guion_bajo")
    doc = pymupdf.open()
    pag = doc.new_page(width=595, height=842)
    pag.insert_text((72, 120), f"Contacto: {correo}", fontsize=10, fontname="helv")
    elementos = _elementos_texto(pag, 0, [("correo", correo, {"formato": "guion_bajo"})])
    doc.save(raiz / "pdf/permisos.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_128, owner_pw="dueno", permissions=0)
    doc.close()
    return Archivo(
        id="lb_permisos",
        ruta="pdf/permisos.pdf",
        formato="pdf",
        categoria="pdf_texto",
        descripcion="PDF con clave de dueño solamente",
        paginas=[Pagina(0, 595, 842, "pt")],
        elementos=elementos,
    )


def _pdf_escaneo(raiz: Path) -> Archivo:
    img = Image.new("RGB", (850, 1100), "white")
    ImageDraw.Draw(img).rectangle([100, 100, 700, 140], fill=(40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    doc = pymupdf.open()
    pag = doc.new_page(width=612, height=792)
    pag.insert_image(pag.rect, stream=buf.getvalue())
    doc.save(raiz / "pdf/escaneo.pdf")
    doc.close()
    return Archivo(
        id="lb_escaneo",
        ruta="pdf/escaneo.pdf",
        formato="pdf",
        categoria="pdf_escaneado",
        descripcion="PDF solo imagen",
        paginas=[Pagina(0, 612, 792, "pt")],
    )


def _pdf_cuaderno(raiz: Path) -> Archivo:
    doc = pymupdf.open()
    pagina = doc.new_page()
    pagina.insert_text((60, 70), TEXTO_CUADERNO, fontsize=11, fontname="helv")
    doc.save(raiz / "pdf/cuaderno.pdf")
    doc.close()
    with pymupdf.open(raiz / "pdf/cuaderno.pdf") as doc:
        elementos = _elementos_texto(
            doc[0], 0, [("rut", "15.782.334-9", {}), ("rut", "9.876.543-3", {}), ("nombre", "Ana Maria Rojas Pena", {})]
        )
        ancho, alto = doc[0].rect.width, doc[0].rect.height
    return Archivo(
        id="lb_cuaderno",
        ruta="pdf/cuaderno.pdf",
        formato="pdf",
        categoria="pdf_texto",
        descripcion="documento de prueba del cuaderno",
        paginas=[Pagina(0, ancho, alto, "pt")],
        elementos=elementos,
    )


@pytest.fixture(scope="module")
def conjunto(tmp_path_factory: pytest.TempPathFactory) -> Manifiesto:
    raiz = tmp_path_factory.mktemp("conjunto")
    f = Ficticios(33).derivar("test_lineas_base")
    man = Manifiesto(raiz=str(raiz), semilla=33)
    for a in (
        _pdf_texto(raiz, f),
        _pdf_rotado(raiz, f),
        _jpeg_exif(raiz, f),
        _tiff_multipagina(raiz, f),
        _pdf_clave(raiz),
        _pdf_permisos(raiz, f),
        _pdf_escaneo(raiz),
        _pdf_cuaderno(raiz),
    ):
        man.agregar(a)
    man.lista_nombres = ["Ana Maria Rojas Pena", *(p.nombre_completo for p in f.personas)]
    assert man.validar() == []
    man.guardar()
    return man


def _por_id(man: Manifiesto, id: str) -> Archivo:
    return next(a for a in man.archivos if a.id == id)


def _resultado(informe: InformeCensura, ruta: str):
    return next(r for r in informe.resultados if r.entrada == ruta)


def _fraccion_negra(img: Image.Image, puntos: list[tuple[float, float]], umbral: int = 40) -> float:
    mascara = Image.new("L", img.size, 0)
    ImageDraw.Draw(mascara).polygon(puntos, fill=255)
    m = np.asarray(mascara) > 0
    assert m.any()
    oscuro = np.asarray(img.convert("RGB")).max(axis=2) < umbral
    return float(oscuro[m].mean())


# ---------------------------------------------------------------------------
# identidad
# ---------------------------------------------------------------------------


def test_identidad_copia_byte_a_byte(conjunto: Manifiesto, tmp_path: Path) -> None:
    informe = ejecutar("identidad", conjunto, tmp_path)
    assert len(informe.resultados) == len(conjunto.archivos)
    for a, r in zip(conjunto.archivos, informe.resultados, strict=True):
        assert r.error is None and r.censuras == []
        assert r.salida == a.ruta
        entrada = (Path(conjunto.raiz) / a.ruta).read_bytes()
        assert (tmp_path / "archivos" / r.salida).read_bytes() == entrada


# ---------------------------------------------------------------------------
# oráculo
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corrida_oraculo(conjunto: Manifiesto, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, InformeCensura]:
    salida = tmp_path_factory.mktemp("oraculo")
    return salida, ejecutar("oraculo", conjunto, salida)


def test_oraculo_errores_esperados(conjunto: Manifiesto, corrida_oraculo) -> None:
    salida, informe = corrida_oraculo
    r = _resultado(informe, "pdf/clave.pdf")
    assert r.error == "contrasena" and r.salida is None
    assert not (salida / "archivos/pdf/clave.pdf").exists()
    for a in conjunto.archivos:
        if a.esperado == "procesar":
            assert _resultado(informe, a.ruta).error is None, a.ruta


def test_oraculo_pdf_sin_texto_ni_metadatos(conjunto: Manifiesto, corrida_oraculo) -> None:
    salida, informe = corrida_oraculo
    for a in conjunto.archivos:
        if a.formato != "pdf" or a.esperado != "procesar":
            continue
        r = _resultado(informe, a.ruta)
        ruta = salida / "archivos" / r.salida
        with pymupdf.open(Path(conjunto.raiz) / a.ruta) as original, pymupdf.open(ruta) as doc:
            assert doc.page_count == original.page_count == r.paginas_procesadas
            for p_o, p_s in zip(original, doc, strict=True):
                assert p_s.rotation == 0
                assert abs(p_s.rect.width - p_o.rect.width) < 0.01
                assert abs(p_s.rect.height - p_o.rect.height) < 0.01
                assert p_s.get_text().strip() == ""
            assert all(not v for k, v in doc.metadata.items() if k not in ("format", "encryption"))
            assert doc.get_xml_metadata() == ""
            assert not doc.is_encrypted
        datos = ruta.read_bytes()
        for e in a.elementos:
            if e.valor:
                assert e.valor.encode() not in datos
        # Las censuras reportadas son los polígonos de la verdad de terreno.
        objetivos = [e for e in a.elementos if e.tipo != "texto"]
        assert [c.poligono for c in r.censuras] == [e.poligono for e in objetivos]
        assert {c.detector for c in r.censuras} <= {"oraculo"}


def test_oraculo_pdf_pixeles_negros(conjunto: Manifiesto, corrida_oraculo) -> None:
    """A 144 ppp y en orientación visible, cada polígono de dato personal queda negro; el texto neutro no."""
    salida, _ = corrida_oraculo
    for id in ("lb_texto", "lb_rotada", "lb_cuaderno", "lb_permisos"):
        a = _por_id(conjunto, id)
        paginas = visualizar.paginas_visibles(salida / "archivos" / a.ruta, "pdf")
        originales = visualizar.paginas_visibles(Path(conjunto.raiz) / a.ruta, "pdf")
        for e in a.elementos:
            pts = visualizar.a_visual(a, e.pagina, e.poligono)
            if e.tipo == "texto":
                assert _fraccion_negra(paginas[e.pagina], pts) < 0.5, (id, e.valor)
            else:
                assert _fraccion_negra(paginas[e.pagina], pts) >= 0.99, (id, e.valor)
                # En el original el polígono tiene tinta, pero no es un bloque negro (el mapeo es el correcto).
                assert _fraccion_negra(originales[e.pagina], pts) < 0.9, (id, e.valor)


def test_oraculo_jpeg_exif(conjunto: Manifiesto, corrida_oraculo) -> None:
    salida, informe = corrida_oraculo
    a = _por_id(conjunto, "lb_exif6")
    ruta = salida / "archivos" / a.ruta
    with Image.open(ruta) as img:
        assert img.format == "JPEG"
        assert img.size == (640, 420)  # geometría ya corregida
        assert len(img.getexif()) == 0
        assert "exif" not in img.info and "icc_profile" not in img.info
        assert ImageOps.exif_transpose(img).size == (640, 420)
        img.load()
        for e in a.elementos:
            assert _fraccion_negra(img, [tuple(p) for p in e.poligono]) >= 0.99
    datos = ruta.read_bytes()
    assert b"Exif" not in datos
    assert a.elementos[0].valor.encode() not in datos
    assert len(_resultado(informe, a.ruta).censuras) == 2


def test_oraculo_tiff_multipagina(conjunto: Manifiesto, corrida_oraculo) -> None:
    salida, informe = corrida_oraculo
    a = _por_id(conjunto, "lb_tiff")
    ruta = salida / "archivos" / a.ruta
    paginas = visualizar.paginas_visibles(ruta, "tiff")
    assert [p.size for p in paginas] == [(pg.ancho, pg.alto) for pg in a.paginas]
    for e in a.elementos:
        assert _fraccion_negra(paginas[e.pagina], [tuple(p) for p in e.poligono]) >= 0.99
    with Image.open(ruta) as img:
        assert img.n_frames == 2
        assert 315 not in img.tag_v2  # sin Artist
    assert b"Autor Ficticio" not in ruta.read_bytes()
    assert _resultado(informe, a.ruta).paginas_procesadas == 2


def test_oraculo_mapeo_igual_al_del_visualizador(conjunto: Manifiesto) -> None:
    """La transformación del oráculo (rotation_matrix) coincide con ``visualizar.a_visual``."""
    a = _por_id(conjunto, "lb_rotada")
    with pymupdf.open(Path(conjunto.raiz) / a.ruta) as doc:
        for e in a.elementos:
            pag = doc[e.pagina]
            m = pag.rotation_matrix * pymupdf.Matrix(visualizar.ESCALA_PDF, visualizar.ESCALA_PDF)
            propio = [tuple(pymupdf.Point(x, y) * m) for x, y in e.poligono]
            ref = visualizar.a_visual(a, e.pagina, e.poligono)
            assert np.allclose(propio, ref, atol=1e-3)


def test_oraculo_objetivos_excluyen_texto_y_oculto() -> None:
    a = Archivo(
        id="x",
        ruta="x.pdf",
        formato="pdf",
        categoria="c",
        descripcion="",
        paginas=[Pagina(0, 100, 100, "pt")],
        elementos=[
            Elemento(tipo="texto", pagina=0, poligono=[[0, 0], [1, 0], [1, 1]], capa="texto"),
            Elemento(tipo="rut", pagina=0, poligono=[[0, 0], [1, 0], [1, 1]], capa="oculto"),
            Elemento(tipo="rut", pagina=0, poligono=[[0, 0], [1, 0], [1, 1]], capa="vector"),
            Elemento(tipo="rostro", pagina=0, poligono=None),
        ],
    )
    assert [e.capa for e in oraculo.objetivos(a)] == ["vector"]


# ---------------------------------------------------------------------------
# cuaderno
# ---------------------------------------------------------------------------


def test_cuaderno_reproduce_celda_12(conjunto: Manifiesto) -> None:
    ruta = Path(conjunto.raiz) / "pdf/cuaderno.pdf"
    assert cuaderno.tiene_texto(ruta)
    assert cuaderno.detectar(ruta) == HALLAZGOS_CUADERNO
    assert cuaderno.detectar_nombres(ruta) == [(1, "nombre", "Ana Maria Rojas Pena")]


def test_cuaderno_corrida(conjunto: Manifiesto, tmp_path: Path) -> None:
    informe = ejecutar("cuaderno", conjunto, tmp_path)
    r = _resultado(informe, "pdf/cuaderno.pdf")
    assert r.error is None and r.salida == "pdf/cuaderno.pdf"
    tipos = sorted(c.tipo for c in r.censuras)
    assert tipos == ["correo", "nombre", "rut", "rut", "telefono", "url"]
    assert {c.detector for c in r.censuras} == {"regex", "lista_nombres"}
    with pymupdf.open(tmp_path / "archivos" / r.salida) as doc:
        texto = "".join(p.get_text() for p in doc)
    assert not re.findall(cuaderno.PATRONES["rut"], texto)
    assert "15.782.334-9" not in texto and "9.876.543-3" not in texto
    assert "Ana Maria Rojas Pena" not in texto
    assert "Monto bruto" in texto  # el texto neutro se conserva
    verif = informe.detalles["verificacion_cuaderno"]["pdf/cuaderno.pdf"]
    assert verif == {"detectados": 6, "fugas_segun_cuaderno": 0, "barrido_final": {}}

    # PDF de texto del conjunto: el RUT con puntos y el nombre de la lista se censuran.
    a = _por_id(conjunto, "lb_texto")
    r = _resultado(informe, a.ruta)
    assert r.error is None
    with pymupdf.open(tmp_path / "archivos" / a.ruta) as doc:
        texto = normalizar("rut", doc[0].get_text())
        assert doc.metadata["author"]  # el cuaderno no limpia metadatos
    rut = next(e for e in a.elementos if e.tipo == "rut")
    assert normalizar("rut", rut.valor) not in texto

    assert _resultado(informe, "img/exif6.jpg").error == "no_soportado"
    assert _resultado(informe, "img/multi.tiff").error == "no_soportado"
    assert _resultado(informe, "pdf/escaneo.pdf").error == "sin_texto"
    assert _resultado(informe, "pdf/clave.pdf").error.startswith("excepcion:")
    for r in informe.resultados:
        if r.error:
            assert r.salida is None and not (tmp_path / "archivos" / r.entrada).exists()


def test_cli_escribe_informe(conjunto: Manifiesto, tmp_path: Path) -> None:
    manifiesto = Path(conjunto.raiz) / "manifiesto.json"
    for sistema in ("identidad", "cuaderno"):
        salida = tmp_path / sistema
        assert main([sistema, "--manifiesto", str(manifiesto), "--salida", str(salida)]) == 0
        informe = InformeCensura.cargar(salida / "informe.json")
        assert informe.sistema == sistema
        assert [r.entrada for r in informe.resultados] == [a.ruta for a in conjunto.archivos]
        assert all(r.tiempo_s is not None for r in informe.resultados)
        # Una segunda corrida sobre la misma carpeta reemplaza la anterior.
        assert main([sistema, "--manifiesto", str(manifiesto), "--salida", str(salida)]) == 0
