"""Reading an output image for the evaluator: pixels (with EXIF applied) and metadata.

Metadata is read with our own container parsers (JPEG segments, PNG and WEBP chunks) plus
``piexif`` for EXIF, without depending on what PIL decides to expose.
"""

from __future__ import annotations

import struct
import zlib
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import piexif
from PIL import Image, ImageOps, ImageSequence

from test_bench.evaluation.search import Haystack, decode_string, texts_from_bytes

Image.MAX_IMAGE_PIXELS = None

# TIFF tags that may carry personal data (and their name in ``MetadataEntry.location``).
TIFF_TAGS = {
    270: "description",
    315: "artist",
    269: "document",
    285: "page",
    305: "software",
    306: "date",
    316: "host_computer",
    33432: "copyright",
    700: "xmp",
    33723: "iptc",
    34665: "exif",
    34853: "gps",
}
_TIFF_ALIASES = {
    "imagedescription": 270,
    "description": 270,
    "artist": 315,
    "artista": 315,
    "documentname": 269,
    "pagename": 285,
    "hostcomputer": 316,
    "datetime": 306,
    "copyright": 33432,
    "software": 305,
    "gps": 34853,
    "gpsinfo": 34853,
    "exif": 34665,
    "xmp": 700,
    "iptc": 33723,
}


def format_from_signature(data: bytes) -> str:
    """Real format according to the first bytes (not to the extension)."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if b"%PDF" in data[:1024]:
        return "pdf"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypmsf1"):
        return "heic"
    return "unknown"


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------


def _value_text(v: Any) -> list[str]:
    if isinstance(v, bytes):
        if v[:8] in (b"UNICODE\x00", b"ASCII\x00\x00\x00", b"JIS\x00\x00\x00\x00\x00"):
            body = v[8:]
            return [
                body.decode("utf-16-be", "ignore"),
                body.decode("utf-16-le", "ignore"),
                body.decode("latin-1"),
            ]
        return decode_string(v) + ([v.decode("utf-16-le", "ignore")] if b"\x00" in v else [])
    if isinstance(v, tuple) and v and all(isinstance(x, int) and 0 <= x < 256 for x in v):
        return _value_text(bytes(v))
    return [str(v)]


def read_exif(data: bytes, texts: list[tuple[str, str]], st: dict[str, Any], location: str = "exif") -> None:
    """Reads an EXIF block (TIFF bytes, with or without the ``Exif\\0\\0`` prefix) with piexif."""
    if data.startswith(b"Exif\x00\x00"):
        data = data[6:]
    st["exif"] = True
    try:
        exif = piexif.load(data)
    except Exception:  # noqa: BLE001 - damaged EXIF: the byte search is enough
        texts.extend((location, t) for t in texts_from_bytes(data))
        st.setdefault("exif.tags", []).append("unreadable")
        return
    for ifd in ("0th", "Exif", "GPS", "Interop", "1st"):
        for tag, v in (exif.get(ifd) or {}).items():
            name = piexif.TAGS.get(ifd, {}).get(tag, {}).get("name", str(tag))
            st.setdefault("exif.tags", []).append(f"{ifd}.{name}")
            texts.extend((f"{location}.{name.lower()}", t) for t in _value_text(v))
    if exif.get("GPS"):
        st["exif.gps"] = True
    if exif.get("thumbnail") or exif.get("1st"):
        st["exif.thumbnail"] = True


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def _iptc(data: bytes, texts: list[tuple[str, str]]) -> None:
    """IPTC-NAA records (0x1C, record, dataset, length, data) inside an APP13/8BIM block."""
    i = 0
    while True:
        i = data.find(b"\x1c", i)
        if i < 0 or i + 5 > len(data):
            break
        length = struct.unpack(">H", data[i + 3 : i + 5])[0]
        if length & 0x8000 or data[i + 1] not in range(1, 10):
            i += 1
            continue
        texts.extend(("iptc", t) for t in decode_string(data[i + 5 : i + 5 + length]))
        i += 5 + length


def read_jpeg(data: bytes, texts: list[tuple[str, str]], st: dict[str, Any]) -> bytes:
    """Walks the JPEG segments; returns the bytes that are not compressed image data."""
    i = 2
    non_pixels = [data[:2]]
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:
            break
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        body = data[i + 4 : i + 2 + length]
        non_pixels.append(data[i : i + 2 + length])
        if marker == 0xE1 and body.startswith(b"Exif\x00\x00"):
            read_exif(body, texts, st)
        elif marker == 0xE1 and b"http://ns.adobe.com/" in body[:40]:
            st["xmp"] = True
            texts.append(("xmp", body.split(b"\x00", 1)[-1].decode("utf-8", "ignore")))
        elif marker == 0xED:
            st["iptc"] = True
            _iptc(body, texts)
            texts.extend(("iptc", t) for t in texts_from_bytes(body))
        elif marker == 0xFE:
            st["com"] = True
            texts.extend(("com", t) for t in decode_string(body))
        elif marker == 0xE2 and body.startswith(b"ICC_PROFILE"):
            st["icc"] = True
        elif 0xE0 <= marker <= 0xEF and marker != 0xE0:
            st.setdefault("jpeg.app", []).append(f"APP{marker - 0xE0}")
            texts.extend((f"jpeg.app{marker - 0xE0}", t) for t in texts_from_bytes(body))
        i += 2 + length
        if marker == 0xDA:
            # image data up to the EOI; whatever comes after the EOI is checked as bytes
            eoi = data.rfind(b"\xff\xd9")
            if eoi > i:
                non_pixels.append(data[eoi + 2 :])
                if eoi + 2 < len(data):
                    st["jpeg.data_after_eoi"] = len(data) - eoi - 2
            break
    return b"\n".join(non_pixels)


def read_png(data: bytes, texts: list[tuple[str, str]], st: dict[str, Any]) -> bytes:
    i = 8
    non_pixels = []
    keys: list[str] = []
    while i + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[i : i + 8])
        body = data[i + 8 : i + 8 + length]
        i += 12 + length
        if kind == b"IDAT":
            continue
        non_pixels.append(kind + body)
        try:
            if kind == b"tEXt":
                k, _, v = body.partition(b"\x00")
                keys.append(k.decode("latin-1"))
                texts.append((f"png.text.{k.decode('latin-1')}", v.decode("latin-1")))
            elif kind == b"zTXt":
                k, _, v = body.partition(b"\x00")
                keys.append(k.decode("latin-1"))
                texts.append((f"png.text.{k.decode('latin-1')}", zlib.decompress(v[1:]).decode("latin-1")))
            elif kind == b"iTXt":
                k, _, rest = body.partition(b"\x00")
                compressed = rest[0] == 1
                _, _, rest = rest[2:].partition(b"\x00")  # language
                translated, _, text = rest.partition(b"\x00")  # translated key
                if compressed:
                    text = zlib.decompress(text)
                keys.append(k.decode("latin-1"))
                texts.append((f"png.text.{k.decode('latin-1')}", text.decode("utf-8", "ignore")))
                texts.append((f"png.text.{k.decode('latin-1')}", translated.decode("utf-8", "ignore")))
                if k == b"XML:com.adobe.xmp":
                    st["xmp"] = True
            elif kind == b"eXIf":
                read_exif(body, texts, st)
            elif kind == b"iCCP":
                st["icc"] = True
            elif kind[:1].islower() and kind not in (b"pHYs", b"gAMA", b"cHRM", b"sRGB", b"bKGD", b"tRNS", b"sBIT"):
                st.setdefault("png.other", []).append(kind.decode("latin-1"))
                texts.extend((f"png.{kind.decode('latin-1')}", t) for t in texts_from_bytes(body))
        except Exception:  # noqa: BLE001 - damaged chunk: the byte search remains
            texts.extend(("png", t) for t in texts_from_bytes(body))
        if kind == b"IEND":
            non_pixels.append(data[i:])
            if i < len(data):
                st["png.data_after_iend"] = len(data) - i
            break
    if keys:
        st["png.text"] = keys
    return b"\n".join(non_pixels)


def read_webp(data: bytes, texts: list[tuple[str, str]], st: dict[str, Any]) -> bytes:
    i = 12
    non_pixels = []
    end = min(len(data), 8 + struct.unpack("<I", data[4:8])[0])
    while i + 8 <= end:
        kind, length = struct.unpack("<4sI", data[i : i + 8])
        body = data[i + 8 : i + 8 + length]
        i += 8 + length + (length & 1)
        if kind in (b"VP8 ", b"VP8L", b"ALPH", b"ANMF"):
            continue
        non_pixels.append(kind + body)
        if kind == b"EXIF":
            read_exif(body, texts, st)
        elif kind == b"XMP ":
            st["xmp"] = True
            texts.append(("xmp", body.decode("utf-8", "ignore")))
        elif kind == b"ICCP":
            st["icc"] = True
        elif kind not in (b"VP8X", b"ANIM"):
            st.setdefault("webp.other", []).append(kind.decode("latin-1"))
            texts.extend((f"webp.{kind.decode('latin-1').strip()}", t) for t in texts_from_bytes(body))
    non_pixels.append(data[end:])
    if end < len(data):
        st["webp.data_after_riff"] = len(data) - end
    return b"\n".join(non_pixels)


def read_tiff(img: Image.Image, data: bytes, texts: list[tuple[str, str]], st: dict[str, Any]) -> bytes:
    """Sensitive tags of every TIFF page; returns the bytes without the image strips/tiles."""
    ranges: list[tuple[int, int]] = []
    for n, frame in enumerate(ImageSequence.Iterator(img)):
        tags = frame.tag_v2
        for tag, name in TIFF_TAGS.items():
            if tag not in tags:
                continue
            st[f"tiff.{tag}"] = True
            st.setdefault("tiff.tags", []).append(f"p{n}.{name}")
            v = tags[tag]
            if tag == 34665:
                for k, x in frame.getexif().get_ifd(0x8769).items():
                    texts.extend((f"tiff.exif.{k}", t) for t in _value_text(x))
            elif tag == 34853:
                gps = frame.getexif().get_ifd(0x8825)
                if gps:
                    st["exif.gps"] = True
                    texts.extend(("tiff.gps", t) for x in gps.values() for t in _value_text(x))
            else:
                texts.extend((f"tiff.{name}", t) for t in _value_text(v))
                if tag == 700:
                    st["xmp"] = True
                elif tag == 33723:
                    st["iptc"] = True
                    _iptc(v if isinstance(v, bytes) else bytes(v) if isinstance(v, tuple) else b"", texts)
        for off_tag, cnt_tag in ((273, 279), (324, 325)):
            if off_tag in tags and cnt_tag in tags:
                offs, cnts = tags[off_tag], tags[cnt_tag]
                offs = offs if isinstance(offs, tuple) else (offs,)
                cnts = cnts if isinstance(cnts, tuple) else (cnts,)
                ranges.extend((int(o), int(o) + int(c)) for o, c in zip(offs, cnts, strict=False))
    if not ranges:
        return data
    parts, pos = [], 0
    for a, b in sorted(ranges):
        if a > pos:
            parts.append(data[pos:a])
        pos = max(pos, b)
    parts.append(data[pos:])
    return b"\n".join(parts)


class ImageOutput:
    """Output image: frames in the displayed geometry and metadata readers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        self.format = format_from_signature(self.data)
        self.img = Image.open(path)
        self.img.load()
        self.read_warnings: list[str] = []

    def close(self) -> None:
        self.img.close()

    @cached_property
    def frames(self) -> list[np.ndarray]:
        """Every frame (page of a TIFF) with the EXIF orientation applied, in RGB (no alpha)."""
        out = []
        for frame in ImageSequence.Iterator(self.img):
            c = ImageOps.exif_transpose(frame.copy())
            if c.mode in ("RGBA", "LA", "PA") or (c.mode == "P" and "transparency" in c.info):
                # the pixels under the transparency are still in the file: alpha is ignored
                c = Image.merge("RGB", c.convert("RGBA").split()[:3])
            elif c.mode != "RGB":
                c = c.convert("RGB")
            out.append(np.asarray(c))
        return out

    def geometry(self) -> list[tuple[int, int]]:
        return [(c.shape[1], c.shape[0]) for c in self.frames]

    @cached_property
    def metadata(self) -> tuple[list[tuple[str, str]], dict[str, Any], list[str], bytes]:
        """(texts by location, structures present, warnings, bytes that are not pixels)."""
        texts: list[tuple[str, str]] = []
        st: dict[str, Any] = {}
        try:
            if self.format == "jpg":
                rest = read_jpeg(self.data, texts, st)
            elif self.format == "png":
                rest = read_png(self.data, texts, st)
            elif self.format == "webp":
                rest = read_webp(self.data, texts, st)
            elif self.format == "tiff":
                rest = read_tiff(self.img, self.data, texts, st)
            else:
                rest = self.data
        except Exception as ex:  # noqa: BLE001 - damaged container: everything is checked as bytes
            self.read_warnings.append(f"no se pudo recorrer el contenedor: {ex}")
            rest = self.data
        # EXIF that PIL finds on its own (e.g. in TIFF or formats not walked)
        if "exif" not in st and self.format != "tiff":
            try:
                raw = self.img.info.get("exif")
                if raw:
                    read_exif(raw, texts, st)
            except Exception:  # noqa: BLE001
                pass
        warn = []
        if st.get("exif"):
            tags = st.get("exif.tags", [])
            warn.append(f"EXIF presente ({len(tags)} etiquetas: {', '.join(tags[:8])})")
        if st.get("exif.gps"):
            warn.append("bloque GPS presente")
        if st.get("exif.thumbnail"):
            warn.append("miniatura EXIF presente")
        for key, label in (
            ("xmp", "XMP"),
            ("iptc", "IPTC"),
            ("com", "comentario JPEG"),
            ("icc", "perfil de color ICC"),
        ):
            if st.get(key):
                warn.append(f"{label} presente")
        if st.get("png.text"):
            warn.append("bloques de texto PNG: " + ", ".join(st["png.text"]))
        if st.get("tiff.tags"):
            warn.append("etiquetas TIFF: " + ", ".join(st["tiff.tags"][:12]))
        for k in ("jpeg.app", "png.other", "webp.other"):
            if st.get(k):
                warn.append(f"{k}: {', '.join(st[k])}")
        for k in ("jpeg.data_after_eoi", "png.data_after_iend", "webp.data_after_riff"):
            if st.get(k):
                warn.append(f"{st[k]} bytes después del fin de la imagen")
        return [(d, t) for d, t in texts if t], st, warn, rest

    @cached_property
    def bytes_haystack(self) -> Haystack:
        haystack = Haystack()
        haystack.add("bytes", texts_from_bytes(self.metadata[3]), binary=True)
        return haystack


def tiff_structure(location: str) -> str | None:
    """Key of ``structures`` for a ``MetadataEntry.location`` of the form ``tiff.<name>``."""
    name = location.split(".", 1)[1].lower() if "." in location else ""
    if name.isdigit():
        return f"tiff.{int(name)}"
    for tag, n in TIFF_TAGS.items():
        if n == name:
            return f"tiff.{tag}"
    if name.replace("_", "") in _TIFF_ALIASES:
        return f"tiff.{_TIFF_ALIASES[name.replace('_', '')]}"
    return None
