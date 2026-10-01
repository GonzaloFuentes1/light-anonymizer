"""Reading an output PDF for the evaluator: text, bytes, pixels, images, strokes and metadata.

All input coordinates are in PyMuPDF's *unrotated* page space (the one of
``page.search_for``); ``get_image_info`` and ``get_drawings`` return that same space.
Only the page rendering (check P) is in visible orientation, at 144 dpi.
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

from test_bench.evaluation.search import Haystack, decode_string, texts_from_bytes

SCALE = 2.0  # 144 dpi, the same as test_bench.visualize

# Text not clipped to the page: TEXT_MEDIABOX_CLIP off and infinite clip (with the flag off
# PyMuPDF still clips to page.rect unless an explicit clip is passed).
TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_MEDIABOX_CLIP

pymupdf.TOOLS.mupdf_display_errors(False)


# ---------------------------------------------------------------------------
# PDF strings: literal (...) and hexadecimal <...>, joining the pieces of a TJ array
# ---------------------------------------------------------------------------

_STRING = re.compile(
    rb"\((?:[^()\\]++|\\.|\((?:[^()\\]++|\\.|\((?:[^()\\]++|\\.)*+\))*+\))*+\)|<([0-9A-Fa-f\s]*)>",
    re.S,
)
_ESCAPE = re.compile(rb"\\([0-7]{1,3}|\r\n|[\s\S])")
_SIMPLE_ESCAPES = {
    b"n": b"\n",
    b"r": b"\r",
    b"t": b"\t",
    b"b": b"\b",
    b"f": b"\f",
    b"\r\n": b"",
    b"\r": b"",
    b"\n": b"",
}
_TJ_GAP = re.compile(rb"^[\s\d.+\-]*$")


def _unescape(m: re.Match[bytes]) -> bytes:
    s = m.group(1)
    if s in _SIMPLE_ESCAPES:
        return _SIMPLE_ESCAPES[s]
    if s[:1].isdigit():
        return bytes([int(s, 8) & 0xFF])
    return s


def _piece(m: re.Match[bytes]) -> bytes:
    if m.group(1) is not None:
        h = re.sub(rb"\s", b"", m.group(1))
        if len(h) % 2:
            h += b"0"
        try:
            return bytes.fromhex(h.decode("ascii"))
        except ValueError:
            return b""
    return _ESCAPE.sub(_unescape, m.group()[1:-1])


def pdf_strings(data: bytes) -> list[str]:
    """Texts of the strings of a PDF object or content stream, decoded.

    Strings separated only by numbers (spacing adjustments inside a ``TJ``) are joined, to
    recover ``[(12.3) -20 (45.678-5)] TJ`` as ``12.345.678-5``.
    """
    groups: list[list[bytes]] = []
    previous_end = None
    for m in _STRING.finditer(data):
        piece = _piece(m)
        if previous_end is not None and _TJ_GAP.match(data[previous_end : m.start()]):
            groups[-1].append(piece)
        else:
            groups.append([piece])
        previous_end = m.end()
    out = []
    for g in groups:
        if not any(g):
            continue
        out.append("".join(decode_string(p)[0] for p in g if p))
        if any(b"\x00" in p for p in g):
            out.append("".join(p.decode("utf-16-be", "ignore") for p in g))
    return out


# ---------------------------------------------------------------------------
# Raw streams (includes previous revisions and orphan objects)
# ---------------------------------------------------------------------------

_STREAM_START = re.compile(rb"(?<!end)stream\r?\n")
_NON_TEXT_MARKERS = (b"/Image", b"/Length1", b"/Length2", b"/FontFile", b"/Type1C", b"/CIDFontType0C", b"/OpenType")


def _is_non_text_stream(dictionary: bytes) -> bool:
    return any(k in dictionary for k in _NON_TEXT_MARKERS)


def raw_streams(data: bytes, images: list[bytes] | None = None) -> tuple[list[bytes], bytes]:
    """Streams found by walking the bytes (without using the xref table) and the file without its streams.

    This recovers objects of previous revisions that the current xref no longer references.
    Flate streams are decompressed; unfiltered ones are returned as they are; image and font
    streams are skipped (their pixels are checked with checks I and O). If ``images`` is
    given, the raw (undecoded) bodies of the image streams are appended to it.
    """
    streams: list[bytes] = []
    rest: list[bytes] = []
    pos = 0
    for m in _STREAM_START.finditer(data):
        if m.start() < pos:
            continue
        end = data.find(b"endstream", m.end())
        if end < 0:
            break
        dict_start = data.rfind(b"obj", max(0, m.start() - 4096), m.start())
        dictionary = data[dict_start if dict_start >= 0 else max(0, m.start() - 4096) : m.start()]
        rest.append(data[pos : m.end()])
        pos = end
        body = data[m.end() : end]
        if _is_non_text_stream(dictionary):
            if images is not None and b"/Image" in dictionary:
                images.append(body)
            continue
        if b"/FlateDecode" in dictionary or b"/Fl " in dictionary or b"/Fl]" in dictionary:
            try:
                streams.append(zlib.decompressobj().decompress(body, 64 << 20))
            except zlib.error:
                pass
        elif b"/Filter" not in dictionary:
            streams.append(body)
    rest.append(data[pos:])
    return streams, b"\n".join(rest)


# ---------------------------------------------------------------------------
# Images placed on the page
# ---------------------------------------------------------------------------


@dataclass
class PlacedImage:
    xref: int
    bbox: tuple[float, float, float, float]
    transform: pymupdf.Matrix  # maps the unit square of the image to the page
    width: int
    height: int
    pixels: np.ndarray | None = None  # (height, width, channels) or None if it could not be decoded

    def to_image(self, polygon: list[list[float]]) -> list[list[float]]:
        """Page polygon -> image pixels (inverse of the placement matrix)."""
        inv = ~self.transform
        out = []
        for x, y in polygon:
            p = pymupdf.Point(x, y) * inv
            out.append([p.x * self.width, p.y * self.height])
        return out


def fingerprint(data: bytes | np.ndarray) -> bytes:
    """Fingerprint of a raw stream or of the decoded pixels of an image."""
    if isinstance(data, np.ndarray):
        return hashlib.blake2b(repr(data.shape).encode() + data.tobytes(), digest_size=16).digest()
    return hashlib.blake2b(data.rstrip(b"\r\n"), digest_size=16).digest()


def pixmap_to_np(pix: pymupdf.Pixmap) -> np.ndarray:
    """Pixmap -> array (height, width, channels) in RGB or gray, without alpha. CMYK is converted to RGB."""
    if pix.n == pix.alpha:  # stencil mask: alpha only
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        return arr[:, :, :1].copy()
    if pix.colorspace is not None and pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).copy()


# ---------------------------------------------------------------------------
# Output document
# ---------------------------------------------------------------------------


class PdfOutput:
    """Lazy (cached) wrapper of an output PDF."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        self.doc = pymupdf.open(stream=self.data, filetype="pdf")
        if self.doc.needs_pass:
            raise ValueError("el PDF de salida está protegido con contraseña")
        self._render: dict[int, np.ndarray] = {}
        self._images: dict[int, list[PlacedImage]] = {}
        self._pix_xref: dict[int, np.ndarray | None] = {}
        self._curves: dict[int, np.ndarray] = {}
        self._geometry: list[tuple[float, float, int]] | None = None
        self.read_warnings: list[str] = []

    def close(self) -> None:
        self.doc.close()

    # -- geometry ------------------------------------------------------------

    def geometry(self) -> list[tuple[float, float, int]]:
        """(width, height, rotation) of every page, in the unrotated space."""
        if self._geometry is None:
            self._geometry = [(p.cropbox.width, p.cropbox.height, p.rotation) for p in self.doc]
        return self._geometry

    # -- text (check T) ------------------------------------------------------

    @cached_property
    def text_pymupdf(self) -> list[str]:
        """Text of every page with PyMuPDF, not clipped to the page and with every layer visible."""
        doc = self.doc
        copy = None
        try:
            if self.doc.get_ocgs():
                # Optional layers that are off are not extracted: /OCProperties is removed in a copy.
                copy = pymupdf.open(stream=self.data, filetype="pdf")
                copy.xref_set_key(copy.pdf_catalog(), "OCProperties", "null")
                copy = pymupdf.open(stream=copy.tobytes(), filetype="pdf")
                doc = copy
        except Exception as ex:  # noqa: BLE001 - the original is used
            self.read_warnings.append(f"no se pudieron encender las capas opcionales: {ex}")
        texts = []
        for p in doc:
            try:
                texts.append(p.get_text("text", flags=TEXT_FLAGS, clip=pymupdf.INFINITE_RECT()))
            except Exception as ex:  # noqa: BLE001
                self.read_warnings.append(f"PyMuPDF no extrajo texto de la página {p.number}: {ex}")
                texts.append("")
        if copy is not None:
            copy.close()
        return texts

    @cached_property
    def text_pdfium(self) -> list[str]:
        texts = []
        try:
            pdf = pdfium.PdfDocument(self.data)
        except Exception as ex:  # noqa: BLE001
            self.read_warnings.append(f"pdfium no abrió el archivo: {ex}")
            return texts
        try:
            for i in range(len(pdf)):
                page = pdf[i]
                tp = page.get_textpage()
                texts.append(tp.get_text_range())
                tp.close()
                page.close()
        except Exception as ex:  # noqa: BLE001
            self.read_warnings.append(f"pdfium falló extrayendo texto: {ex}")
        finally:
            pdf.close()
        return texts

    @cached_property
    def text_haystack(self) -> Haystack:
        haystack = Haystack()
        haystack.add("text_pymupdf", self.text_pymupdf)
        haystack.add("text_pdfium", self.text_pdfium)
        return haystack

    # -- bytes (check B) -----------------------------------------------------

    @cached_property
    def _streams(self) -> tuple[list[bytes], bytes, list[bytes]]:
        """(decompressed text streams, file without streams, raw bodies of image streams)."""
        images: list[bytes] = []
        raw, without_streams = raw_streams(self.data, images)
        return raw, without_streams, images

    @cached_property
    def bytes_haystack(self) -> Haystack:
        """File bytes, decompressed streams and decoded strings of every object."""
        haystack = Haystack()
        raw, without_streams, _ = self._streams
        haystack.add("bytes", texts_from_bytes(without_streams), binary=True)
        haystack.add("bytes_strings", pdf_strings(without_streams))
        seen: set[bytes] = set()

        def add_stream(origin: str, data: bytes) -> None:
            h = hashlib.blake2b(data, digest_size=16).digest()
            if h in seen:
                return
            seen.add(h)
            haystack.add(origin, texts_from_bytes(data), binary=True)
            haystack.add(origin + "_strings", pdf_strings(data))

        doc = self.doc
        for xref in range(1, doc.xref_length()):
            try:
                source = doc.xref_object(xref, compressed=False)
            except Exception:  # noqa: BLE001 - damaged object
                continue
            haystack.add("object_strings", pdf_strings(source.encode("latin-1", "replace")))
            haystack.add("objects", source, binary=True)
            if not doc.xref_is_stream(xref):
                continue
            if _is_non_text_stream(source.encode("latin-1", "replace")):
                continue
            try:
                data = doc.xref_stream(xref)
            except Exception:  # noqa: BLE001
                data = None
            if data:
                add_stream("streams", data)
        for data in raw:
            add_stream("streams", data)
        return haystack

    # -- pixels (check P) ----------------------------------------------------

    def render(self, index: int) -> np.ndarray:
        """Page ``index`` rendered at 144 dpi in its visible orientation (RGB)."""
        if index not in self._render:
            pix = self.doc[index].get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE), alpha=False)
            self._render[index] = pixmap_to_np(pix)
        return self._render[index]

    # -- images (check I) ----------------------------------------------------

    def _xref_pixels(self, xref: int) -> np.ndarray | None:
        if xref not in self._pix_xref:
            try:
                self._pix_xref[xref] = pixmap_to_np(pymupdf.Pixmap(self.doc, xref))
            except Exception as ex:  # noqa: BLE001
                self.read_warnings.append(f"no se pudo decodificar la imagen xref {xref}: {ex}")
                self._pix_xref[xref] = None
        return self._pix_xref[xref]

    def images(self, index: int) -> list[PlacedImage]:
        """Images drawn on the page (including those of form XObjects and inline ones)."""
        if index in self._images:
            return self._images[index]
        page = self.doc[index]
        out: list[PlacedImage] = []
        inline: list[dict[str, Any]] | None = None
        for info in page.get_image_info(xrefs=True):
            m = pymupdf.Matrix(info["transform"])
            if abs(m.a * m.d - m.b * m.c) < 1e-9:
                continue
            img = PlacedImage(
                xref=info.get("xref", 0),
                bbox=tuple(info["bbox"]),
                transform=m,
                width=int(info["width"]),
                height=int(info["height"]),
            )
            if img.xref > 0:
                img.pixels = self._xref_pixels(img.xref)
            else:
                if inline is None:
                    try:
                        dic = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_IMAGES, clip=pymupdf.INFINITE_RECT())
                        inline = [b for b in dic["blocks"] if b.get("type") == 1]
                    except Exception:  # noqa: BLE001
                        inline = []
                for b in inline:
                    if np.allclose(tuple(pymupdf.Matrix(b["transform"])), tuple(m), atol=0.05):
                        try:
                            img.pixels = pixmap_to_np(pymupdf.Pixmap(b["image"]))
                        except Exception:  # noqa: BLE001
                            img.pixels = None
                        break
            if img.pixels is not None:
                img.height, img.width = img.pixels.shape[:2]
            out.append(img)
        self._images[index] = out
        return out

    # -- original images (check O) -------------------------------------------

    def raw_stream(self, xref: int) -> bytes | None:
        """Stream of an image as it is stored (compressed)."""
        try:
            return self.doc.xref_stream_raw(xref)
        except Exception:  # noqa: BLE001
            return None

    @cached_property
    def image_inventory(self) -> tuple[dict[tuple[int, int], list[int]], set[bytes]]:
        """Every image of the file, drawn or orphan: xrefs by (width, height) and fingerprints of
        their raw streams (including those of previous revisions the xref no longer references)."""
        doc = self.doc
        by_size: dict[tuple[int, int], list[int]] = {}
        raw: set[bytes] = set()
        for xref in range(1, doc.xref_length()):
            try:
                if doc.xref_get_key(xref, "Subtype")[1] != "/Image":
                    continue
                w = int(float(doc.xref_get_key(xref, "Width")[1]))
                h = int(float(doc.xref_get_key(xref, "Height")[1]))
            except Exception:  # noqa: BLE001 - damaged object or without dimensions
                continue
            by_size.setdefault((w, h), []).append(xref)
            stream = self.raw_stream(xref)
            if stream:
                raw.add(fingerprint(stream))
        raw.update(fingerprint(c) for c in self._streams[2] if c)
        return by_size, raw

    def contains_image(self, pixels: np.ndarray, raw: bytes | None) -> str | None:
        """Where an image appears intact (same stream or same pixels), or ``None`` if it does not."""
        by_size, raw_prints = self.image_inventory
        if raw and fingerprint(raw) in raw_prints:
            return "stream"
        height, width = pixels.shape[:2]
        target = fingerprint(pixels)
        for xref in by_size.get((width, height), []):
            p = self._xref_pixels(xref)
            if p is not None and p.shape == pixels.shape and fingerprint(p) == target:
                return f"xref {xref}"
        return None

    # -- vector strokes (check V) --------------------------------------------

    def curves(self, index: int) -> np.ndarray:
        """Points (N, 2) sampled over every Bézier curve of the page."""
        if index not in self._curves:
            points = []
            t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])[:, None]
            for path in self.doc[index].get_drawings():
                for item in path.get("items", []):
                    if item[0] != "c":
                        continue
                    p0, p1, p2, p3 = (np.array([p.x, p.y]) for p in item[1:5])
                    points.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3)
            self._curves[index] = np.vstack(points) if points else np.zeros((0, 2))
        return self._curves[index]

    # -- metadata and structures ---------------------------------------------

    @cached_property
    def revisions(self) -> int:
        """Number of revisions (incremental saves) of the file."""
        eof = len(re.findall(rb"%%EOF", self.data))
        linearized = b"/Linearized" in self.data[:2048]
        n = eof - (1 if linearized and eof > 1 else 0)
        try:
            n = max(n, int(self.doc.version_count))
        except Exception:  # noqa: BLE001
            pass
        return max(1, n)

    @cached_property
    def metadata(self) -> tuple[list[tuple[str, str]], dict[str, Any], list[str]]:
        """Structured readers: (texts by location, structures present, warnings)."""
        doc = self.doc
        texts: list[tuple[str, str]] = []
        st: dict[str, Any] = {}
        warn: list[str] = []

        # /Info dictionary (including custom keys) and XMP
        info = {k: v for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")}
        for k, v in info.items():
            texts.append((f"pdf.info.{k}", str(v)))
        try:
            kind, value = doc.xref_get_key(-1, "Info")
            if kind == "xref":
                ix = int(value.split()[0])
                for k in doc.xref_get_keys(ix):
                    t2, v2 = doc.xref_get_key(ix, k)
                    if v2:
                        texts.append((f"pdf.info.{k.lower()}", v2))
                        info.setdefault(k.lower(), v2)
        except Exception:  # noqa: BLE001
            pass
        st["pdf.info"] = sorted(info)
        for k in info:
            st[f"pdf.info.{k.lower()}"] = True
        if info:
            warn.append("diccionario /Info con: " + ", ".join(f"{k}={str(v)[:40]!r}" for k, v in sorted(info.items())))
        xmp = doc.get_xml_metadata() or ""
        st["pdf.xmp"] = bool(xmp.strip())
        if xmp.strip():
            texts.append(("pdf.xmp", xmp))
            warn.append(f"metadatos XMP ({len(xmp)} caracteres)")

        # annotations (including popups), forms and attachments in annotations
        n_annots, n_widget_values = 0, 0
        for page in doc:
            for item in page.annot_xrefs():
                xref, annot_type = item[0], item[1]
                if annot_type not in (pymupdf.PDF_ANNOT_LINK, pymupdf.PDF_ANNOT_WIDGET):
                    n_annots += 1
                for key in ("Contents", "T", "Subj", "RC", "NM"):
                    try:
                        t2, v2 = doc.xref_get_key(xref, key)
                    except Exception:  # noqa: BLE001
                        continue
                    if t2 in ("string", "text") and v2:
                        texts.append(("pdf.annotation", v2))
                    elif t2 == "xref":
                        try:
                            texts.extend(
                                ("pdf.annotation", t) for t in texts_from_bytes(doc.xref_stream(int(v2.split()[0])))
                            )
                        except Exception:  # noqa: BLE001
                            pass
            for annot in page.annots() or []:
                inf = annot.info
                texts.extend(("pdf.annotation", inf.get(k, "")) for k in ("content", "title", "subject", "name"))
                if annot.type[0] == pymupdf.PDF_ANNOT_FILE_ATTACHMENT:
                    try:
                        texts.extend(("pdf.attachment", t) for t in texts_from_bytes(annot.get_file()))
                        st["pdf.attachment"] = True
                    except Exception:  # noqa: BLE001
                        pass
            for w in page.widgets() or []:
                values = [w.field_value, w.field_label, w.field_name] + list(w.choice_values or [])
                for v in values:
                    if v not in (None, "", "Off"):
                        texts.append(("pdf.form", str(v)))
                if w.field_value not in (None, "", "Off"):
                    n_widget_values += 1
        st["pdf.annotation"] = n_annots
        st["pdf.form"] = n_widget_values
        if n_annots:
            warn.append(f"{n_annots} anotaciones")
        if n_widget_values:
            warn.append(f"{n_widget_values} campos de formulario con valor")

        # embedded files
        try:
            names = doc.embfile_names()
        except Exception:  # noqa: BLE001
            names = []
        for name in names:
            texts.append(("pdf.attachment", name))
            try:
                inf = doc.embfile_info(name)
                texts.extend(("pdf.attachment", str(inf.get(k, ""))) for k in ("filename", "ufilename", "description"))
                texts.extend(("pdf.attachment", t) for t in texts_from_bytes(doc.embfile_get(name)))
            except Exception:  # noqa: BLE001
                pass
        if names:
            st["pdf.attachment"] = True
            warn.append(f"{len(names)} archivos incrustados")
        st.setdefault("pdf.attachment", False)

        # optional layers
        try:
            ocgs = doc.get_ocgs() or {}
        except Exception:  # noqa: BLE001
            ocgs = {}
        for o in ocgs.values():
            texts.append(("pdf.ocg", str(o.get("name", ""))))
        st["pdf.ocg"] = len(ocgs)
        if ocgs:
            warn.append(f"{len(ocgs)} capas opcionales")

        # bookmarks
        try:
            toc = doc.get_toc(simple=True)
        except Exception:  # noqa: BLE001
            toc = []
        for entry in toc:
            texts.append(("pdf.bookmark", str(entry[1])))
        st["pdf.bookmark"] = len(toc)
        if toc:
            warn.append(f"{len(toc)} marcadores")

        # JavaScript: any object with /JS
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
                    texts.extend(("pdf.javascript", t) for t in texts_from_bytes(doc.xref_stream(int(v2.split()[0]))))
                except Exception:  # noqa: BLE001
                    pass
            else:
                texts.append(("pdf.javascript", v2))
        st["pdf.javascript"] = n_js
        if n_js:
            warn.append(f"{n_js} acciones JavaScript")

        st["pdf.previous_revision"] = self.revisions > 1
        if self.revisions > 1:
            warn.append(f"{self.revisions} revisiones (guardado incremental)")
        return [(d, t) for d, t in texts if t], st, warn
