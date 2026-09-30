"""Lectura de un PDF de salida para el evaluador: texto, bytes, píxeles, imágenes, trazos y metadatos.

Todas las coordenadas de entrada están en el espacio de página *sin rotar* de PyMuPDF (el de
``page.search_for``); ``get_image_info`` y ``get_drawings`` entregan ese mismo espacio.
Solo el dibujo de la página (comprobación P) está en orientación visible, a 144 ppp.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pypdfium2 as pdfium

from banco_pruebas.evaluacion.buscar import Pajar, decodificar_cadena, textos_de_bytes

ESCALA = 2.0  # 144 ppp, igual que banco_pruebas.visualizar

# Texto sin recortar a la página: TEXT_MEDIABOX_CLIP apagado y clip infinito (con la bandera
# apagada PyMuPDF igual recorta a page.rect si no se le pasa un clip explícito).
FLAGS_TEXTO = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_MEDIABOX_CLIP

pymupdf.TOOLS.mupdf_display_errors(False)


# ---------------------------------------------------------------------------
# Cadenas PDF: literales (...) y hexadecimales <...>, uniendo las piezas de un arreglo TJ
# ---------------------------------------------------------------------------

_CADENA = re.compile(
    rb"\((?:[^()\\]++|\\.|\((?:[^()\\]++|\\.|\((?:[^()\\]++|\\.)*+\))*+\))*+\)|<([0-9A-Fa-f\s]*)>",
    re.S,
)
_ESCAPE = re.compile(rb"\\([0-7]{1,3}|\r\n|[\s\S])")
_ESCAPES_SIMPLES = {
    b"n": b"\n",
    b"r": b"\r",
    b"t": b"\t",
    b"b": b"\b",
    b"f": b"\f",
    b"\r\n": b"",
    b"\r": b"",
    b"\n": b"",
}
_HUECO_TJ = re.compile(rb"^[\s\d.+\-]*$")


def _desescapar(m: re.Match[bytes]) -> bytes:
    s = m.group(1)
    if s in _ESCAPES_SIMPLES:
        return _ESCAPES_SIMPLES[s]
    if s[:1].isdigit():
        return bytes([int(s, 8) & 0xFF])
    return s


def _pieza(m: re.Match[bytes]) -> bytes:
    if m.group(1) is not None:
        h = re.sub(rb"\s", b"", m.group(1))
        if len(h) % 2:
            h += b"0"
        try:
            return bytes.fromhex(h.decode("ascii"))
        except ValueError:
            return b""
    return _ESCAPE.sub(_desescapar, m.group()[1:-1])


def cadenas_pdf(datos: bytes) -> list[str]:
    """Textos de las cadenas de un objeto o flujo de contenido PDF, decodificadas.

    Las cadenas separadas solo por números (ajustes de espaciado dentro de un ``TJ``) se unen,
    para recuperar ``[(12.3) -20 (45.678-5)] TJ`` como ``12.345.678-5``.
    """
    grupos: list[list[bytes]] = []
    fin_anterior = None
    for m in _CADENA.finditer(datos):
        pieza = _pieza(m)
        if fin_anterior is not None and _HUECO_TJ.match(datos[fin_anterior : m.start()]):
            grupos[-1].append(pieza)
        else:
            grupos.append([pieza])
        fin_anterior = m.end()
    salida = []
    for g in grupos:
        if not any(g):
            continue
        salida.append("".join(decodificar_cadena(p)[0] for p in g if p))
        if any(b"\x00" in p for p in g):
            salida.append("".join(p.decode("utf-16-be", "ignore") for p in g))
    return salida


# ---------------------------------------------------------------------------
# Flujos crudos (incluye revisiones anteriores y objetos huérfanos)
# ---------------------------------------------------------------------------

_INICIO_FLUJO = re.compile(rb"(?<!end)stream\r?\n")
_MARCAS_NO_TEXTO = (b"/Image", b"/Length1", b"/Length2", b"/FontFile", b"/Type1C", b"/CIDFontType0C", b"/OpenType")


def _es_flujo_no_texto(diccionario: bytes) -> bool:
    return any(k in diccionario for k in _MARCAS_NO_TEXTO)


def flujos_crudos(datos: bytes, imagenes: list[bytes] | None = None) -> tuple[list[bytes], bytes]:
    """Flujos encontrados recorriendo los bytes (sin usar la tabla xref) y el archivo sin sus flujos.

    Así se recuperan objetos de revisiones anteriores que la xref actual ya no referencia.
    Se descomprimen los flujos Flate; los que no tienen filtro se entregan tal cual; los de
    imagen y fuente se omiten (sus píxeles se revisan con las comprobaciones I y O). Si se pasa
    ``imagenes``, ahí se agregan los cuerpos crudos (sin decodificar) de los flujos de imagen.
    """
    flujos: list[bytes] = []
    resto: list[bytes] = []
    pos = 0
    for m in _INICIO_FLUJO.finditer(datos):
        if m.start() < pos:
            continue
        fin = datos.find(b"endstream", m.end())
        if fin < 0:
            break
        inicio_dic = datos.rfind(b"obj", max(0, m.start() - 4096), m.start())
        diccionario = datos[inicio_dic if inicio_dic >= 0 else max(0, m.start() - 4096) : m.start()]
        resto.append(datos[pos : m.end()])
        pos = fin
        cuerpo = datos[m.end() : fin]
        if _es_flujo_no_texto(diccionario):
            if imagenes is not None and b"/Image" in diccionario:
                imagenes.append(cuerpo)
            continue
        if b"/FlateDecode" in diccionario or b"/Fl " in diccionario or b"/Fl]" in diccionario:
            try:
                flujos.append(zlib.decompressobj().decompress(cuerpo, 64 << 20))
            except zlib.error:
                pass
        elif b"/Filter" not in diccionario:
            flujos.append(cuerpo)
    resto.append(datos[pos:])
    return flujos, b"\n".join(resto)


# ---------------------------------------------------------------------------
# Imágenes colocadas en la página
# ---------------------------------------------------------------------------


@dataclass
class ImagenColocada:
    xref: int
    bbox: tuple[float, float, float, float]
    transform: pymupdf.Matrix  # lleva el cuadrado unitario de la imagen a la página
    ancho: int
    alto: int
    pixeles: np.ndarray | None = None  # (alto, ancho, canales) o None si no se pudo decodificar

    def a_imagen(self, poligono: list[list[float]]) -> list[list[float]]:
        """Polígono de la página -> píxeles de la imagen (inversa de la matriz de colocación)."""
        inv = ~self.transform
        salida = []
        for x, y in poligono:
            p = pymupdf.Point(x, y) * inv
            salida.append([p.x * self.ancho, p.y * self.alto])
        return salida


def huella(datos: bytes | np.ndarray) -> bytes:
    """Huella de un flujo crudo o de los píxeles decodificados de una imagen."""
    if isinstance(datos, np.ndarray):
        return hashlib.blake2b(repr(datos.shape).encode() + datos.tobytes(), digest_size=16).digest()
    return hashlib.blake2b(datos.rstrip(b"\r\n"), digest_size=16).digest()


def pixmap_a_np(pix: pymupdf.Pixmap) -> np.ndarray:
    """Pixmap -> arreglo (alto, ancho, canales) en RGB o gris, sin alfa. CMYK se convierte a RGB."""
    if pix.n == pix.alpha:  # máscara de esténcil: solo alfa
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        return arr[:, :, :1].copy()
    if pix.colorspace is not None and pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).copy()


# ---------------------------------------------------------------------------
# Documento de salida
# ---------------------------------------------------------------------------


class SalidaPdf:
    """Envoltorio perezoso (con caché) de un PDF de salida."""

    def __init__(self, ruta: Path) -> None:
        self.ruta = ruta
        self.datos = ruta.read_bytes()
        self.doc = pymupdf.open(stream=self.datos, filetype="pdf")
        if self.doc.needs_pass:
            raise ValueError("el PDF de salida está protegido con contraseña")
        self._render: dict[int, np.ndarray] = {}
        self._imagenes: dict[int, list[ImagenColocada]] = {}
        self._pix_xref: dict[int, np.ndarray | None] = {}
        self._curvas: dict[int, np.ndarray] = {}
        self._geometria: list[tuple[float, float, int]] | None = None
        self.advertencias_lectura: list[str] = []

    def cerrar(self) -> None:
        self.doc.close()

    # -- geometría -----------------------------------------------------------

    def geometria(self) -> list[tuple[float, float, int]]:
        """(ancho, alto, rotación) de cada página, en el espacio sin rotar."""
        if self._geometria is None:
            self._geometria = [(p.cropbox.width, p.cropbox.height, p.rotation) for p in self.doc]
        return self._geometria

    # -- texto (comprobación T) ---------------------------------------------

    @cached_property
    def texto_pymupdf(self) -> list[str]:
        """Texto de cada página con PyMuPDF, sin recortar a la página y con todas las capas visibles."""
        doc = self.doc
        copia = None
        try:
            if self.doc.get_ocgs():
                # Las capas opcionales apagadas no se extraen: se quita /OCProperties en una copia.
                copia = pymupdf.open(stream=self.datos, filetype="pdf")
                copia.xref_set_key(copia.pdf_catalog(), "OCProperties", "null")
                copia = pymupdf.open(stream=copia.tobytes(), filetype="pdf")
                doc = copia
        except Exception as ex:  # noqa: BLE001 - se usa el original
            self.advertencias_lectura.append(f"no se pudieron encender las capas opcionales: {ex}")
        textos = []
        for p in doc:
            try:
                textos.append(p.get_text("text", flags=FLAGS_TEXTO, clip=pymupdf.INFINITE_RECT()))
            except Exception as ex:  # noqa: BLE001
                self.advertencias_lectura.append(f"PyMuPDF no extrajo texto de la página {p.number}: {ex}")
                textos.append("")
        if copia is not None:
            copia.close()
        return textos

    @cached_property
    def texto_pdfium(self) -> list[str]:
        textos = []
        try:
            pdf = pdfium.PdfDocument(self.datos)
        except Exception as ex:  # noqa: BLE001
            self.advertencias_lectura.append(f"pdfium no abrió el archivo: {ex}")
            return textos
        try:
            for i in range(len(pdf)):
                pagina = pdf[i]
                tp = pagina.get_textpage()
                textos.append(tp.get_text_range())
                tp.close()
                pagina.close()
        except Exception as ex:  # noqa: BLE001
            self.advertencias_lectura.append(f"pdfium falló extrayendo texto: {ex}")
        finally:
            pdf.close()
        return textos

    @cached_property
    def pajar_texto(self) -> Pajar:
        pajar = Pajar()
        pajar.agregar("texto_pymupdf", self.texto_pymupdf)
        pajar.agregar("texto_pdfium", self.texto_pdfium)
        return pajar

    # -- bytes (comprobación B) ---------------------------------------------

    @cached_property
    def _flujos(self) -> tuple[list[bytes], bytes, list[bytes]]:
        """(flujos de texto descomprimidos, archivo sin flujos, cuerpos crudos de flujos de imagen)."""
        imagenes: list[bytes] = []
        crudos, sin_flujos = flujos_crudos(self.datos, imagenes)
        return crudos, sin_flujos, imagenes

    @cached_property
    def pajar_bytes(self) -> Pajar:
        """Bytes del archivo, flujos descomprimidos y cadenas decodificadas de todos los objetos."""
        pajar = Pajar()
        crudos, sin_flujos, _ = self._flujos
        pajar.agregar("bytes", textos_de_bytes(sin_flujos), binario=True)
        pajar.agregar("bytes_cadenas", cadenas_pdf(sin_flujos))
        vistos: set[bytes] = set()

        def agregar_flujo(origen: str, datos: bytes) -> None:
            h = hashlib.blake2b(datos, digest_size=16).digest()
            if h in vistos:
                return
            vistos.add(h)
            pajar.agregar(origen, textos_de_bytes(datos), binario=True)
            pajar.agregar(origen + "_cadenas", cadenas_pdf(datos))

        doc = self.doc
        for xref in range(1, doc.xref_length()):
            try:
                fuente = doc.xref_object(xref, compressed=False)
            except Exception:  # noqa: BLE001 - objeto dañado
                continue
            pajar.agregar("objetos_cadenas", cadenas_pdf(fuente.encode("latin-1", "replace")))
            pajar.agregar("objetos", fuente, binario=True)
            if not doc.xref_is_stream(xref):
                continue
            if _es_flujo_no_texto(fuente.encode("latin-1", "replace")):
                continue
            try:
                datos = doc.xref_stream(xref)
            except Exception:  # noqa: BLE001
                datos = None
            if datos:
                agregar_flujo("flujos", datos)
        for datos in crudos:
            agregar_flujo("flujos", datos)
        return pajar

    # -- píxeles (comprobación P) --------------------------------------------

    def render(self, indice: int) -> np.ndarray:
        """Página ``indice`` dibujada a 144 ppp en su orientación visible (RGB)."""
        if indice not in self._render:
            pix = self.doc[indice].get_pixmap(matrix=pymupdf.Matrix(ESCALA, ESCALA), alpha=False)
            self._render[indice] = pixmap_a_np(pix)
        return self._render[indice]

    # -- imágenes (comprobación I) -------------------------------------------

    def _pixeles_xref(self, xref: int) -> np.ndarray | None:
        if xref not in self._pix_xref:
            try:
                self._pix_xref[xref] = pixmap_a_np(pymupdf.Pixmap(self.doc, xref))
            except Exception as ex:  # noqa: BLE001
                self.advertencias_lectura.append(f"no se pudo decodificar la imagen xref {xref}: {ex}")
                self._pix_xref[xref] = None
        return self._pix_xref[xref]

    def imagenes(self, indice: int) -> list[ImagenColocada]:
        """Imágenes dibujadas en la página (incluidas las de XObjects de formulario y en línea)."""
        if indice in self._imagenes:
            return self._imagenes[indice]
        pagina = self.doc[indice]
        salida: list[ImagenColocada] = []
        en_linea: list[dict[str, Any]] | None = None
        for info in pagina.get_image_info(xrefs=True):
            m = pymupdf.Matrix(info["transform"])
            if abs(m.a * m.d - m.b * m.c) < 1e-9:
                continue
            img = ImagenColocada(
                xref=info.get("xref", 0),
                bbox=tuple(info["bbox"]),
                transform=m,
                ancho=int(info["width"]),
                alto=int(info["height"]),
            )
            if img.xref > 0:
                img.pixeles = self._pixeles_xref(img.xref)
            else:
                if en_linea is None:
                    try:
                        dic = pagina.get_text("dict", flags=pymupdf.TEXT_PRESERVE_IMAGES, clip=pymupdf.INFINITE_RECT())
                        en_linea = [b for b in dic["blocks"] if b.get("type") == 1]
                    except Exception:  # noqa: BLE001
                        en_linea = []
                for b in en_linea:
                    if np.allclose(tuple(pymupdf.Matrix(b["transform"])), tuple(m), atol=0.05):
                        try:
                            img.pixeles = pixmap_a_np(pymupdf.Pixmap(b["image"]))
                        except Exception:  # noqa: BLE001
                            img.pixeles = None
                        break
            if img.pixeles is not None:
                img.alto, img.ancho = img.pixeles.shape[:2]
            salida.append(img)
        self._imagenes[indice] = salida
        return salida

    # -- imágenes originales (comprobación O) ---------------------------------

    def flujo_crudo(self, xref: int) -> bytes | None:
        """Flujo de una imagen tal como está guardado (comprimido)."""
        try:
            return self.doc.xref_stream_raw(xref)
        except Exception:  # noqa: BLE001
            return None

    @cached_property
    def inventario_imagenes(self) -> tuple[dict[tuple[int, int], list[int]], set[bytes]]:
        """Todas las imágenes del archivo, dibujadas o huérfanas: xrefs por (ancho, alto) y huellas
        de sus flujos crudos (incluidos los de revisiones anteriores que la xref ya no referencia)."""
        doc = self.doc
        por_tamano: dict[tuple[int, int], list[int]] = {}
        crudos: set[bytes] = set()
        for xref in range(1, doc.xref_length()):
            try:
                if doc.xref_get_key(xref, "Subtype")[1] != "/Image":
                    continue
                w = int(float(doc.xref_get_key(xref, "Width")[1]))
                h = int(float(doc.xref_get_key(xref, "Height")[1]))
            except Exception:  # noqa: BLE001 - objeto dañado o sin dimensiones
                continue
            por_tamano.setdefault((w, h), []).append(xref)
            crudo = self.flujo_crudo(xref)
            if crudo:
                crudos.add(huella(crudo))
        crudos.update(huella(c) for c in self._flujos[2] if c)
        return por_tamano, crudos

    def contiene_imagen(self, pixeles: np.ndarray, crudo: bytes | None) -> str | None:
        """Dónde aparece intacta una imagen (mismo flujo o mismos píxeles), o ``None`` si no aparece."""
        por_tamano, crudos = self.inventario_imagenes
        if crudo and huella(crudo) in crudos:
            return "flujo"
        alto, ancho = pixeles.shape[:2]
        objetivo = huella(pixeles)
        for xref in por_tamano.get((ancho, alto), []):
            p = self._pixeles_xref(xref)
            if p is not None and p.shape == pixeles.shape and huella(p) == objetivo:
                return f"xref {xref}"
        return None

    # -- trazos vectoriales (comprobación V) ---------------------------------

    def curvas(self, indice: int) -> np.ndarray:
        """Puntos (N, 2) muestreados sobre todas las curvas de Bézier de la página."""
        if indice not in self._curvas:
            puntos = []
            t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])[:, None]
            for camino in self.doc[indice].get_drawings():
                for item in camino.get("items", []):
                    if item[0] != "c":
                        continue
                    p0, p1, p2, p3 = (np.array([p.x, p.y]) for p in item[1:5])
                    puntos.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3)
            self._curvas[indice] = np.vstack(puntos) if puntos else np.zeros((0, 2))
        return self._curvas[indice]

    # -- metadatos y estructuras ---------------------------------------------

    @cached_property
    def revisiones(self) -> int:
        """Número de revisiones (guardados incrementales) del archivo."""
        eof = len(re.findall(rb"%%EOF", self.datos))
        linealizado = b"/Linearized" in self.datos[:2048]
        n = eof - (1 if linealizado and eof > 1 else 0)
        try:
            n = max(n, int(self.doc.version_count))
        except Exception:  # noqa: BLE001
            pass
        return max(1, n)

    @cached_property
    def metadatos(self) -> tuple[list[tuple[str, str]], dict[str, Any], list[str]]:
        """Lectores estructurados: (textos por lugar, estructuras presentes, advertencias)."""
        doc = self.doc
        textos: list[tuple[str, str]] = []
        est: dict[str, Any] = {}
        adv: list[str] = []

        # diccionario /Info (incluye claves propias) y XMP
        info = {k: v for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")}
        for k, v in info.items():
            textos.append((f"pdf.info.{k}", str(v)))
        try:
            tipo, valor = doc.xref_get_key(-1, "Info")
            if tipo == "xref":
                ix = int(valor.split()[0])
                for k in doc.xref_get_keys(ix):
                    t2, v2 = doc.xref_get_key(ix, k)
                    if v2:
                        textos.append((f"pdf.info.{k.lower()}", v2))
                        info.setdefault(k.lower(), v2)
        except Exception:  # noqa: BLE001
            pass
        est["pdf.info"] = sorted(info)
        for k in info:
            est[f"pdf.info.{k.lower()}"] = True
        if info:
            adv.append("diccionario /Info con: " + ", ".join(f"{k}={str(v)[:40]!r}" for k, v in sorted(info.items())))
        xmp = doc.get_xml_metadata() or ""
        est["pdf.xmp"] = bool(xmp.strip())
        if xmp.strip():
            textos.append(("pdf.xmp", xmp))
            adv.append(f"metadatos XMP ({len(xmp)} caracteres)")

        # anotaciones (incluidas las emergentes), formularios y adjuntos en anotaciones
        n_anot, n_widget_valor = 0, 0
        for pagina in doc:
            for item in pagina.annot_xrefs():
                xref, tipo_anot = item[0], item[1]
                if tipo_anot not in (pymupdf.PDF_ANNOT_LINK, pymupdf.PDF_ANNOT_WIDGET):
                    n_anot += 1
                for clave in ("Contents", "T", "Subj", "RC", "NM"):
                    try:
                        t2, v2 = doc.xref_get_key(xref, clave)
                    except Exception:  # noqa: BLE001
                        continue
                    if t2 in ("string", "text") and v2:
                        textos.append(("pdf.anotacion", v2))
                    elif t2 == "xref":
                        try:
                            textos.extend(
                                ("pdf.anotacion", t) for t in textos_de_bytes(doc.xref_stream(int(v2.split()[0])))
                            )
                        except Exception:  # noqa: BLE001
                            pass
            for anot in pagina.annots() or []:
                inf = anot.info
                textos.extend(("pdf.anotacion", inf.get(k, "")) for k in ("content", "title", "subject", "name"))
                if anot.type[0] == pymupdf.PDF_ANNOT_FILE_ATTACHMENT:
                    try:
                        textos.extend(("pdf.adjunto", t) for t in textos_de_bytes(anot.get_file()))
                        est["pdf.adjunto"] = True
                    except Exception:  # noqa: BLE001
                        pass
            for w in pagina.widgets() or []:
                valores = [w.field_value, w.field_label, w.field_name] + list(w.choice_values or [])
                for v in valores:
                    if v not in (None, "", "Off"):
                        textos.append(("pdf.formulario", str(v)))
                if w.field_value not in (None, "", "Off"):
                    n_widget_valor += 1
        est["pdf.anotacion"] = n_anot
        est["pdf.formulario"] = n_widget_valor
        if n_anot:
            adv.append(f"{n_anot} anotaciones")
        if n_widget_valor:
            adv.append(f"{n_widget_valor} campos de formulario con valor")

        # archivos incrustados
        try:
            nombres = doc.embfile_names()
        except Exception:  # noqa: BLE001
            nombres = []
        for nombre in nombres:
            textos.append(("pdf.adjunto", nombre))
            try:
                inf = doc.embfile_info(nombre)
                textos.extend(("pdf.adjunto", str(inf.get(k, ""))) for k in ("filename", "ufilename", "description"))
                textos.extend(("pdf.adjunto", t) for t in textos_de_bytes(doc.embfile_get(nombre)))
            except Exception:  # noqa: BLE001
                pass
        if nombres:
            est["pdf.adjunto"] = True
            adv.append(f"{len(nombres)} archivos incrustados")
        est.setdefault("pdf.adjunto", False)

        # capas opcionales
        try:
            ocgs = doc.get_ocgs() or {}
        except Exception:  # noqa: BLE001
            ocgs = {}
        for o in ocgs.values():
            textos.append(("pdf.ocg", str(o.get("name", ""))))
        est["pdf.ocg"] = len(ocgs)
        if ocgs:
            adv.append(f"{len(ocgs)} capas opcionales")

        # marcadores
        try:
            toc = doc.get_toc(simple=True)
        except Exception:  # noqa: BLE001
            toc = []
        for entrada in toc:
            textos.append(("pdf.marcador", str(entrada[1])))
        est["pdf.marcador"] = len(toc)
        if toc:
            adv.append(f"{len(toc)} marcadores")

        # JavaScript: cualquier objeto con /JS
        n_js = 0
        for xref in range(1, doc.xref_length()):
            try:
                t2, v2 = doc.xref_get_key(xref, "JS")
            except Exception:  # noqa: BLE001
                continue
            if t2 == "null":
                continue
            n_js += 1
            if t2 == "xref":
                try:
                    textos.extend(("pdf.javascript", t) for t in textos_de_bytes(doc.xref_stream(int(v2.split()[0]))))
                except Exception:  # noqa: BLE001
                    pass
            else:
                textos.append(("pdf.javascript", v2))
        est["pdf.javascript"] = n_js
        if n_js:
            adv.append(f"{n_js} acciones JavaScript")

        est["pdf.revision_anterior"] = self.revisiones > 1
        if self.revisiones > 1:
            adv.append(f"{self.revisiones} revisiones (guardado incremental)")
        return [(d, t) for d, t in textos if t], est, adv
