"""Prototipo rápido del motor: un adelanto de la fase 1 para ver resultados reales.

Detecta de verdad (no usa la verdad de terreno salvo la lista de nombres, que es la que el
usuario le entregaría a la aplicación):

- patrones ampliados de RUT, correo, teléfono y URL sobre la capa de texto, con la geometría
  de cada carácter (sin volver a buscar el texto);
- nombres y direcciones de la lista, sin distinguir tildes ni mayúsculas, en cualquier orden;
- OCR (RapidOCR, PP-OCRv6) en 0°, 90° y 270° (el clasificador de la línea cubre 180°) sobre
  imágenes y sobre páginas PDF con imágenes o sin texto;
- rostros (YuNet) en 4 orientaciones y 2 escalas, con la caja ampliada un 20 %;
- códigos QR.

Censura real: en PDF, ``apply_redactions`` con borrado de píxeles, limpieza de metadatos,
anotaciones, adjuntos, capas y marcadores, y reescritura completa del archivo; en imágenes,
relleno sólido y archivo nuevo sin metadatos.

Limitaciones conocidas del prototipo: usa PyMuPDF (licencia AGPL, decisión D1 pendiente; solo
para pruebas locales), censura la línea completa de OCR cuando contiene un dato, no hace
verificación de fugas propia (la mide el evaluador) y no está optimizado.
"""

from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError  # noqa: E402

from banco_pruebas.esquema import Archivo, Censura, Manifiesto, ResultadoArchivo  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
MODELO_YUNET = RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"
PPP_OCR = 200

# ---------------------------------------------------------------------------
# Patrones
# ---------------------------------------------------------------------------

_GUION = r"[-‐‑‒–—−]"
RUT = re.compile(rf"(?<![\d.,])\d{{1,3}}(?:[.,·\s]?\d{{3}}){{2}}\s?{_GUION}?\s?[\dkK](?![\w@])")
CORREO = re.compile(
    r"[\w.+'-]+\s?(?:@|＠|©|\[at\]|\(at\)|\[arroba\]|\(arroba\)|\sarroba\s)\s?[\w-]+(?:\s?[.,]\s?[\w-]+)*\s?[.,]\s?[a-z]{2,4}\b",
    re.IGNORECASE,
)
URL = re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE)
# Dígitos con separadores de una misma línea; el punto solo entre dígitos (no el punto final de una frase).
_CORRIDA = re.compile(r"[+(]?\d(?:[\d \t()+‐‑–—-]|\.(?=\d)){5,24}\d")
_ETIQUETA_FONO = re.compile(r"(?i)(fono|tel[eé]?f?|cel|m[oó]vil|whats|wsp|fax|contacto)[^\n]{0,20}$")
_CONFUSIONES = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8", "Z": "2"}
)


def _rut_valido_por_forma(m: re.Match[str]) -> bool:
    """Evita tomar como RUT montos o fechas sin guion: sin guion exige 8-10 caracteres seguidos."""
    texto = m.group(0)
    if re.search(_GUION, texto):
        return True
    return bool(re.fullmatch(r"\d{7,9}[\dkK]", re.sub(r"\s", "", texto))) and "." not in texto


def telefonos(texto: str) -> list[tuple[int, int]]:
    salida = []
    for m in _CORRIDA.finditer(texto):
        d = re.sub(r"\D", "", m.group(0))
        if d.startswith("0056"):
            d = d[4:]
        elif d.startswith("56") and len(d) >= 10:
            d = d[2:]
        if len(d) == 10 and d.startswith("0"):
            d = d[1:]
        ok = len(d) == 9 and d[0] in "23456789"
        if not ok and len(d) == 8 and _ETIQUETA_FONO.search(texto[max(0, m.start() - 25) : m.start()]):
            ok = True
        if ok:
            salida.append((m.start(), m.end()))
    return salida


def _sin_tildes(c: str) -> str:
    base = unicodedata.normalize("NFKD", c)
    base = "".join(x for x in base if not unicodedata.combining(x))
    return (base[:1] or c).casefold()


def normalizar_1a1(texto: str) -> str:
    """Normaliza carácter por carácter (mismo largo), para poder volver a las posiciones originales."""
    return "".join(_sin_tildes(c) if c.strip() else " " for c in texto)


@lru_cache(maxsize=64)
def _patrones_lista(lista: tuple[str, ...]) -> list[tuple[str, re.Pattern[str]]]:
    salida = []
    for entrada in lista:
        partes = normalizar_1a1(entrada).split()
        if not partes:
            continue
        variantes = {" ".join(partes)}
        es_direccion = any(ch.isdigit() for ch in entrada)
        if not es_direccion and len(partes) >= 3:
            apellidos, nombres = partes[-2:], partes[:-2]
            variantes |= {
                " ".join(apellidos + nombres),
                " ".join(apellidos) + ", " + " ".join(nombres),
                " ".join([nombres[0], apellidos[0]]),
                " ".join(apellidos),
            }
        if es_direccion:
            m = re.match(r"^(\D*?\d+)", " ".join(partes))
            if m:
                variantes.add(m.group(1))
        for v in variantes:
            cuerpo = r"[\s,]+".join(re.escape(t.strip(",")) for t in v.split())
            salida.append(("direccion" if es_direccion else "nombre", re.compile(rf"(?<!\w){cuerpo}(?!\w)")))
    return salida


@lru_cache(maxsize=64)
def _palabras_lista(lista: tuple[str, ...]) -> frozenset[str]:
    """Nombres de pila y apellidos sueltos de las personas de la lista (sin direcciones)."""
    return frozenset(
        p for e in lista if not any(ch.isdigit() for ch in e) for p in normalizar_1a1(e).split() if len(p) >= 3
    )


def buscar(texto: str, lista: tuple[str, ...], ocr: bool = False) -> list[tuple[str, int, int]]:
    """Tramos (tipo, inicio, fin) con datos personales en ``texto``."""
    hallados: list[tuple[str, int, int]] = []
    variantes = [texto]
    if ocr:
        variantes.append(
            re.sub(
                r"\S+",
                lambda m: (
                    m.group(0).translate(_CONFUSIONES)
                    if sum(c.isdigit() for c in m.group(0)) >= len(m.group(0)) / 2
                    else m.group(0)
                ),
                texto,
            )
        )
    for t in variantes:
        hallados += [("rut", m.start(), m.end()) for m in RUT.finditer(t) if _rut_valido_por_forma(m)]
        hallados += [("telefono", a, b) for a, b in telefonos(t)]
    hallados += [("correo", m.start(), m.end()) for m in CORREO.finditer(texto)]
    hallados += [("url", m.start(), m.end()) for m in URL.finditer(texto)]
    norm = normalizar_1a1(texto)
    for tipo, patron in _patrones_lista(lista):
        hallados += [(tipo, m.start(), m.end()) for m in patron.finditer(norm)]
    return hallados


# ---------------------------------------------------------------------------
# Detectores de imagen
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _ocr():
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from rapidocr import RapidOCR

    return RapidOCR(params={"Global.log_level": "critical", "EngineConfig.onnxruntime.intra_op_num_threads": 4})


def _rotar_puntos(pts: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Lleva puntos de ``np.rot90(img, k)`` a la imagen original de ancho ``w`` y alto ``h``."""
    x, y = pts[:, 0], pts[:, 1]
    if k == 0:
        return pts
    if k == 1:
        return np.stack([w - y, x], axis=1)
    if k == 2:
        return np.stack([w - x, h - y], axis=1)
    return np.stack([y, h - x], axis=1)


def detectar_en_imagen(rgb: np.ndarray, lista: tuple[str, ...]) -> list[tuple[str, np.ndarray, str, str, float]]:
    """Zonas (tipo, polígono Nx2 en píxeles, texto, detector, score) en una imagen RGB."""
    h, w = rgb.shape[:2]
    zonas: list[tuple[str, np.ndarray, str, str, float]] = []
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    motor = _ocr()
    lineas: list[tuple[np.ndarray, str, float]] = []
    for k in (0, 1, 3):
        rot = np.ascontiguousarray(np.rot90(bgr, k))
        r = motor(rot)
        if r.boxes is None or r.txts is None:
            continue
        for caja, txt, score in zip(r.boxes, r.txts, r.scores, strict=False):
            pol = _rotar_puntos(np.asarray(caja, np.float64), k, w, h)
            lineas.append((pol, txt, float(score)))
            tipos = {t for t, _, _ in buscar(txt, lista, ocr=True)}
            if tipos:
                zonas.append((sorted(tipos)[0], pol, txt, "ocr", float(score)))
    # Documentos de identidad: apellidos y nombres vienen en líneas sueltas bajo sus etiquetas.
    if any(re.search(r"apellido|nombres", normalizar_1a1(t)) for _, t, _ in lineas):
        palabras = _palabras_lista(lista)
        for pol, txt, score in lineas:
            tokens = normalizar_1a1(txt).split()
            if tokens and len(tokens) <= 3 and all(tk in palabras for tk in tokens):
                zonas.append(("nombre", pol, txt, "ocr", score))
    zonas += _rostros(bgr)
    zonas += _qr(bgr)
    return zonas


@lru_cache(maxsize=1)
def _yunet():
    return cv2.FaceDetectorYN.create(str(MODELO_YUNET), "", (320, 320), 0.5, 0.3, 5000)


def _rostros(bgr: np.ndarray) -> list[tuple[str, np.ndarray, str, str, float]]:
    h, w = bgr.shape[:2]
    det = _yunet()
    salida = []
    for k in range(4):
        rot = np.ascontiguousarray(np.rot90(bgr, k))
        rh, rw = rot.shape[:2]
        for lado in sorted({640, 1280, max(rh, rw)}):
            if lado > max(rh, rw) * 1.01 and lado != 640:
                continue
            f = lado / max(rh, rw)
            img = cv2.resize(rot, (max(1, round(rw * f)), max(1, round(rh * f))))
            det.setInputSize((img.shape[1], img.shape[0]))
            _, caras = det.detect(img)
            for c in caras if caras is not None else []:
                x, y, cw, ch, score = c[0] / f, c[1] / f, c[2] / f, c[3] / f, float(c[14])
                mx, my = 0.2 * cw, 0.2 * ch
                pts = np.array(
                    [
                        [x - mx, y - my * 1.5],
                        [x + cw + mx, y - my * 1.5],
                        [x + cw + mx, y + ch + my],
                        [x - mx, y + ch + my],
                    ]
                )
                pol = _rotar_puntos(pts, k, w, h)
                salida.append(("rostro", pol, "", "yunet", score))
    return salida


def _qr(bgr: np.ndarray) -> list[tuple[str, np.ndarray, str, str, float]]:
    try:
        ok, textos, puntos, _ = cv2.QRCodeDetector().detectAndDecodeMulti(bgr)
    except cv2.error:
        return []
    if not ok or puntos is None:
        return []
    salida = []
    for pts, txt in zip(puntos, textos, strict=False):
        centro = pts.mean(axis=0)
        salida.append(("qr", (pts - centro) * 1.15 + centro, txt or "", "qr", 1.0))
    return salida


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

_CLAVES_CATALOGO = ("Names", "OpenAction", "AA", "AcroForm", "OCProperties", "Outlines", "Metadata", "PageLabels",
                    "StructTreeRoot", "MarkInfo", "PieceInfo")  # fmt: skip
_CLAVES_PAGINA = ("AA", "PieceInfo", "Thumb", "Metadata")


def _caracteres(page: pymupdf.Page) -> tuple[str, list[pymupdf.Rect | None]]:
    tp = page.get_textpage(clip=pymupdf.INFINITE_RECT(), flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_MEDIABOX_CLIP)
    datos = page.get_text("rawdict", textpage=tp)
    texto: list[str] = []
    cajas: list[pymupdf.Rect | None] = []
    for bloque in datos["blocks"]:
        for linea in bloque.get("lines", []):
            for span in linea["spans"]:
                for ch in span["chars"]:
                    texto.append(ch["c"])
                    cajas.append(pymupdf.Rect(ch["bbox"]))
            texto.append("\n")
            cajas.append(None)
    return "".join(texto), cajas


def _rects_de_tramo(cajas: list[pymupdf.Rect | None], a: int, b: int) -> list[pymupdf.Rect]:
    rects: list[pymupdf.Rect] = []
    actual: pymupdf.Rect | None = None
    for c in cajas[a:b]:
        if c is None:
            if actual is not None:
                rects.append(actual)
            actual = None
            continue
        actual = pymupdf.Rect(c) if actual is None else actual | c
    if actual is not None:
        rects.append(actual)
    return [r + (-1, -1, 1, 1) for r in rects if not r.is_empty]


def _poligono(r: pymupdf.Rect) -> list[list[float]]:
    return [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]]


def _procesar_pdf(entrada: Path, destino: Path, lista: tuple[str, ...]) -> tuple[list[Censura], int]:
    doc = pymupdf.open(entrada)
    if doc.needs_pass:
        doc.close()
        raise PermissionError("contrasena")
    # Revelar capas ocultas: su texto debe detectarse y eliminarse.
    doc.xref_set_key(doc.pdf_catalog(), "OCProperties", "null")
    censuras: list[Censura] = []
    for n, page in enumerate(doc):
        texto, cajas = _caracteres(page)
        for tipo, a, b in buscar(texto, lista):
            for r in _rects_de_tramo(cajas, a, b):
                page.add_redact_annot(r, fill=(0, 0, 0))
                censuras.append(Censura(pagina=n, poligono=_poligono(r), tipo=tipo, detector="regex", texto=texto[a:b]))
        con_imagenes = bool(page.get_images()) or len(texto.strip()) < 50
        if con_imagenes:
            zoom = PPP_OCR / 72
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
            inversa = pymupdf.Matrix(1 / zoom, 1 / zoom) * page.derotation_matrix
            for tipo, pol, txt, detector, score in detectar_en_imagen(rgb, lista):
                x0, y0 = pol.min(axis=0)
                x1, y1 = pol.max(axis=0)
                r = (pymupdf.Rect(x0, y0, x1, y1) * inversa) + (-1, -1, 1, 1)
                page.add_redact_annot(r, fill=(0, 0, 0))
                censuras.append(
                    Censura(pagina=n, poligono=_poligono(r), tipo=tipo, detector=detector, texto=txt, score=score)
                )
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
        for annot in list(page.annots() or []):
            page.delete_annot(annot)
        for w in list(page.widgets() or []):
            page.delete_widget(w)
        for clave in _CLAVES_PAGINA:
            doc.xref_set_key(page.xref, clave, "null")
    # Limpieza del documento
    for nombre in list(doc.embfile_names()):
        doc.embfile_del(nombre)
    doc.set_toc([])
    doc.set_metadata({})
    doc.del_xml_metadata()
    for clave in _CLAVES_CATALOGO:
        doc.xref_set_key(doc.pdf_catalog(), clave, "null")
    destino.parent.mkdir(parents=True, exist_ok=True)
    doc.save(destino, garbage=4, deflate=True, clean=True)
    n_paginas = doc.page_count
    doc.close()
    return censuras, n_paginas


# ---------------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------------

_FORMATOS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF"}


def _procesar_imagen(entrada: Path, destino: Path, lista: tuple[str, ...]) -> tuple[list[Censura], int]:
    img = Image.open(entrada)
    img.load()
    cuadros = [ImageOps.exif_transpose(f.copy()).convert("RGB") for f in ImageSequence.Iterator(img)]
    censuras: list[Censura] = []
    salidas = []
    for n, cuadro in enumerate(cuadros):
        arr = np.array(cuadro)
        for tipo, pol, txt, detector, score in detectar_en_imagen(arr, lista):
            centro = pol.mean(axis=0)
            pol2 = (pol - centro) * 1.04 + centro
            cv2.fillPoly(arr, [np.round(pol2).astype(np.int32)], (0, 0, 0))
            censuras.append(
                Censura(pagina=n, poligono=pol.round(2).tolist(), tipo=tipo, detector=detector, texto=txt, score=score)
            )
        salidas.append(Image.fromarray(arr))
    destino.parent.mkdir(parents=True, exist_ok=True)
    formato = _FORMATOS[entrada.suffix.lower()]
    opciones: dict[str, Any] = {
        "JPEG": {"quality": 92},
        "WEBP": {"quality": 92},
        "PNG": {},
        "TIFF": {"compression": "tiff_deflate"},
    }[formato]
    if len(salidas) > 1:
        salidas[0].save(destino, formato, save_all=True, append_images=salidas[1:], **opciones)
    else:
        salidas[0].save(destino, formato, **opciones)
    return censuras, len(salidas)


def _tipo_real(ruta: Path) -> str:
    cabeza = ruta.read_bytes()[:12]
    if not cabeza:
        return "vacio"
    if cabeza.startswith(b"%PDF"):
        return "pdf"
    if cabeza.startswith((b"\xff\xd8\xff", b"\x89PNG", b"II*\x00", b"MM\x00*")) or (
        cabeza[:4] == b"RIFF" and cabeza[8:12] == b"WEBP"
    ):
        return "imagen"
    return "formato"


def procesar(archivo: Archivo, manifiesto: Manifiesto, carpeta: Path, detalles: dict[str, Any]) -> ResultadoArchivo:
    entrada = Path(manifiesto.raiz) / archivo.ruta
    destino = carpeta / archivo.ruta
    lista = tuple(manifiesto.lista_nombres)
    tipo = _tipo_real(entrada)
    if tipo in ("vacio", "formato"):
        return ResultadoArchivo(entrada=archivo.ruta, salida=None, error=tipo)
    try:
        if tipo == "pdf":
            censuras, n = _procesar_pdf(entrada, destino, lista)
        else:
            censuras, n = _procesar_imagen(entrada, destino, lista)
    except PermissionError:
        return ResultadoArchivo(entrada=archivo.ruta, salida=None, error="contrasena")
    except (UnidentifiedImageError, OSError, RuntimeError, pymupdf.FileDataError, ValueError):
        destino.unlink(missing_ok=True)
        return ResultadoArchivo(
            entrada=archivo.ruta,
            salida=None,
            error="corrupto",
        )
    return ResultadoArchivo(entrada=archivo.ruta, salida=archivo.ruta, censuras=censuras, paginas_procesadas=n)
