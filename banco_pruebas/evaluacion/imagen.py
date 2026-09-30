"""Lectura de una imagen de salida para el evaluador: píxeles (con EXIF aplicado) y metadatos.

Los metadatos se leen con parsers propios de los contenedores (segmentos JPEG, fragmentos PNG
y WEBP) más ``piexif`` para el EXIF, sin depender de lo que PIL decida exponer.
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

from banco_pruebas.evaluacion.buscar import Pajar, decodificar_cadena, textos_de_bytes

Image.MAX_IMAGE_PIXELS = None

# Etiquetas TIFF que pueden llevar datos personales (y su nombre en ``Metadato.donde``).
ETIQUETAS_TIFF = {
    270: "descripcion",
    315: "artista",
    269: "documento",
    285: "pagina",
    305: "software",
    306: "fecha",
    316: "equipo",
    33432: "copyright",
    700: "xmp",
    33723: "iptc",
    34665: "exif",
    34853: "gps",
}
_ALIAS_TIFF = {
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


def formato_por_firma(datos: bytes) -> str:
    """Formato real según los primeros bytes (no según la extensión)."""
    if datos[:3] == b"\xff\xd8\xff":
        return "jpg"
    if datos[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if datos[:4] == b"RIFF" and datos[8:12] == b"WEBP":
        return "webp"
    if datos[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if b"%PDF" in datos[:1024]:
        return "pdf"
    if datos[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypmsf1"):
        return "heic"
    return "desconocido"


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------


def _texto_valor(v: Any) -> list[str]:
    if isinstance(v, bytes):
        if v[:8] in (b"UNICODE\x00", b"ASCII\x00\x00\x00", b"JIS\x00\x00\x00\x00\x00"):
            cuerpo = v[8:]
            return [
                cuerpo.decode("utf-16-be", "ignore"),
                cuerpo.decode("utf-16-le", "ignore"),
                cuerpo.decode("latin-1"),
            ]
        return decodificar_cadena(v) + ([v.decode("utf-16-le", "ignore")] if b"\x00" in v else [])
    if isinstance(v, tuple) and v and all(isinstance(x, int) and 0 <= x < 256 for x in v):
        return _texto_valor(bytes(v))
    return [str(v)]


def leer_exif(datos: bytes, textos: list[tuple[str, str]], est: dict[str, Any], donde: str = "exif") -> None:
    """Lee un bloque EXIF (bytes TIFF, con o sin el prefijo ``Exif\\0\\0``) con piexif."""
    if datos.startswith(b"Exif\x00\x00"):
        datos = datos[6:]
    est["exif"] = True
    try:
        exif = piexif.load(datos)
    except Exception:  # noqa: BLE001 - EXIF dañado: basta la búsqueda en bytes
        textos.extend((donde, t) for t in textos_de_bytes(datos))
        est.setdefault("exif.etiquetas", []).append("ilegible")
        return
    for ifd in ("0th", "Exif", "GPS", "Interop", "1st"):
        for tag, v in (exif.get(ifd) or {}).items():
            nombre = piexif.TAGS.get(ifd, {}).get(tag, {}).get("name", str(tag))
            est.setdefault("exif.etiquetas", []).append(f"{ifd}.{nombre}")
            textos.extend((f"{donde}.{nombre.lower()}", t) for t in _texto_valor(v))
    if exif.get("GPS"):
        est["exif.gps"] = True
    if exif.get("thumbnail") or exif.get("1st"):
        est["exif.miniatura"] = True


# ---------------------------------------------------------------------------
# Contenedores
# ---------------------------------------------------------------------------


def _iptc(datos: bytes, textos: list[tuple[str, str]]) -> None:
    """Registros IPTC-NAA (0x1C, registro, dataset, largo, datos) dentro de un bloque APP13/8BIM."""
    i = 0
    while True:
        i = datos.find(b"\x1c", i)
        if i < 0 or i + 5 > len(datos):
            break
        largo = struct.unpack(">H", datos[i + 3 : i + 5])[0]
        if largo & 0x8000 or datos[i + 1] not in range(1, 10):
            i += 1
            continue
        textos.extend(("iptc", t) for t in decodificar_cadena(datos[i + 5 : i + 5 + largo]))
        i += 5 + largo


def leer_jpeg(datos: bytes, textos: list[tuple[str, str]], est: dict[str, Any]) -> bytes:
    """Recorre los segmentos del JPEG; devuelve los bytes que no son datos de imagen comprimidos."""
    i = 2
    no_pixeles = [datos[:2]]
    while i + 4 <= len(datos):
        if datos[i] != 0xFF:
            i += 1
            continue
        marca = datos[i + 1]
        if marca == 0xFF:
            i += 1
            continue
        if marca in (0xD8, 0x01) or 0xD0 <= marca <= 0xD7:
            i += 2
            continue
        if marca == 0xD9:
            break
        largo = struct.unpack(">H", datos[i + 2 : i + 4])[0]
        cuerpo = datos[i + 4 : i + 2 + largo]
        no_pixeles.append(datos[i : i + 2 + largo])
        if marca == 0xE1 and cuerpo.startswith(b"Exif\x00\x00"):
            leer_exif(cuerpo, textos, est)
        elif marca == 0xE1 and b"http://ns.adobe.com/" in cuerpo[:40]:
            est["xmp"] = True
            textos.append(("xmp", cuerpo.split(b"\x00", 1)[-1].decode("utf-8", "ignore")))
        elif marca == 0xED:
            est["iptc"] = True
            _iptc(cuerpo, textos)
            textos.extend(("iptc", t) for t in textos_de_bytes(cuerpo))
        elif marca == 0xFE:
            est["com"] = True
            textos.extend(("com", t) for t in decodificar_cadena(cuerpo))
        elif marca == 0xE2 and cuerpo.startswith(b"ICC_PROFILE"):
            est["icc"] = True
        elif 0xE0 <= marca <= 0xEF and marca != 0xE0:
            est.setdefault("jpeg.app", []).append(f"APP{marca - 0xE0}")
            textos.extend((f"jpeg.app{marca - 0xE0}", t) for t in textos_de_bytes(cuerpo))
        i += 2 + largo
        if marca == 0xDA:
            # datos de imagen hasta el EOI; lo que venga después del EOI se revisa en bytes
            eoi = datos.rfind(b"\xff\xd9")
            if eoi > i:
                no_pixeles.append(datos[eoi + 2 :])
                if eoi + 2 < len(datos):
                    est["jpeg.datos_tras_eoi"] = len(datos) - eoi - 2
            break
    return b"\n".join(no_pixeles)


def leer_png(datos: bytes, textos: list[tuple[str, str]], est: dict[str, Any]) -> bytes:
    i = 8
    no_pixeles = []
    claves: list[str] = []
    while i + 8 <= len(datos):
        largo, tipo = struct.unpack(">I4s", datos[i : i + 8])
        cuerpo = datos[i + 8 : i + 8 + largo]
        i += 12 + largo
        if tipo == b"IDAT":
            continue
        no_pixeles.append(tipo + cuerpo)
        try:
            if tipo == b"tEXt":
                k, _, v = cuerpo.partition(b"\x00")
                claves.append(k.decode("latin-1"))
                textos.append((f"png.texto.{k.decode('latin-1')}", v.decode("latin-1")))
            elif tipo == b"zTXt":
                k, _, v = cuerpo.partition(b"\x00")
                claves.append(k.decode("latin-1"))
                textos.append((f"png.texto.{k.decode('latin-1')}", zlib.decompress(v[1:]).decode("latin-1")))
            elif tipo == b"iTXt":
                k, _, resto = cuerpo.partition(b"\x00")
                comprimido = resto[0] == 1
                _, _, resto = resto[2:].partition(b"\x00")  # idioma
                trad, _, texto = resto.partition(b"\x00")  # clave traducida
                if comprimido:
                    texto = zlib.decompress(texto)
                claves.append(k.decode("latin-1"))
                textos.append((f"png.texto.{k.decode('latin-1')}", texto.decode("utf-8", "ignore")))
                textos.append((f"png.texto.{k.decode('latin-1')}", trad.decode("utf-8", "ignore")))
                if k == b"XML:com.adobe.xmp":
                    est["xmp"] = True
            elif tipo == b"eXIf":
                leer_exif(cuerpo, textos, est)
            elif tipo == b"iCCP":
                est["icc"] = True
            elif tipo[:1].islower() and tipo not in (b"pHYs", b"gAMA", b"cHRM", b"sRGB", b"bKGD", b"tRNS", b"sBIT"):
                est.setdefault("png.otros", []).append(tipo.decode("latin-1"))
                textos.extend((f"png.{tipo.decode('latin-1')}", t) for t in textos_de_bytes(cuerpo))
        except Exception:  # noqa: BLE001 - fragmento dañado: queda la búsqueda en bytes
            textos.extend(("png", t) for t in textos_de_bytes(cuerpo))
        if tipo == b"IEND":
            no_pixeles.append(datos[i:])
            if i < len(datos):
                est["png.datos_tras_iend"] = len(datos) - i
            break
    if claves:
        est["png.texto"] = claves
    return b"\n".join(no_pixeles)


def leer_webp(datos: bytes, textos: list[tuple[str, str]], est: dict[str, Any]) -> bytes:
    i = 12
    no_pixeles = []
    fin = min(len(datos), 8 + struct.unpack("<I", datos[4:8])[0])
    while i + 8 <= fin:
        tipo, largo = struct.unpack("<4sI", datos[i : i + 8])
        cuerpo = datos[i + 8 : i + 8 + largo]
        i += 8 + largo + (largo & 1)
        if tipo in (b"VP8 ", b"VP8L", b"ALPH", b"ANMF"):
            continue
        no_pixeles.append(tipo + cuerpo)
        if tipo == b"EXIF":
            leer_exif(cuerpo, textos, est)
        elif tipo == b"XMP ":
            est["xmp"] = True
            textos.append(("xmp", cuerpo.decode("utf-8", "ignore")))
        elif tipo == b"ICCP":
            est["icc"] = True
        elif tipo not in (b"VP8X", b"ANIM"):
            est.setdefault("webp.otros", []).append(tipo.decode("latin-1"))
            textos.extend((f"webp.{tipo.decode('latin-1').strip()}", t) for t in textos_de_bytes(cuerpo))
    no_pixeles.append(datos[fin:])
    if fin < len(datos):
        est["webp.datos_tras_riff"] = len(datos) - fin
    return b"\n".join(no_pixeles)


def leer_tiff(img: Image.Image, datos: bytes, textos: list[tuple[str, str]], est: dict[str, Any]) -> bytes:
    """Etiquetas sensibles de cada página del TIFF; devuelve los bytes sin las tiras/teselas de imagen."""
    rangos: list[tuple[int, int]] = []
    for n, cuadro in enumerate(ImageSequence.Iterator(img)):
        tags = cuadro.tag_v2
        for tag, nombre in ETIQUETAS_TIFF.items():
            if tag not in tags:
                continue
            est[f"tiff.{tag}"] = True
            est.setdefault("tiff.etiquetas", []).append(f"p{n}.{nombre}")
            v = tags[tag]
            if tag == 34665:
                for k, x in cuadro.getexif().get_ifd(0x8769).items():
                    textos.extend((f"tiff.exif.{k}", t) for t in _texto_valor(x))
            elif tag == 34853:
                gps = cuadro.getexif().get_ifd(0x8825)
                if gps:
                    est["exif.gps"] = True
                    textos.extend(("tiff.gps", t) for x in gps.values() for t in _texto_valor(x))
            else:
                textos.extend((f"tiff.{nombre}", t) for t in _texto_valor(v))
                if tag == 700:
                    est["xmp"] = True
                elif tag == 33723:
                    est["iptc"] = True
                    _iptc(v if isinstance(v, bytes) else bytes(v) if isinstance(v, tuple) else b"", textos)
        for off_tag, cnt_tag in ((273, 279), (324, 325)):
            if off_tag in tags and cnt_tag in tags:
                offs, cnts = tags[off_tag], tags[cnt_tag]
                offs = offs if isinstance(offs, tuple) else (offs,)
                cnts = cnts if isinstance(cnts, tuple) else (cnts,)
                rangos.extend((int(o), int(o) + int(c)) for o, c in zip(offs, cnts, strict=False))
    if not rangos:
        return datos
    partes, pos = [], 0
    for a, b in sorted(rangos):
        if a > pos:
            partes.append(datos[pos:a])
        pos = max(pos, b)
    partes.append(datos[pos:])
    return b"\n".join(partes)


class SalidaImagen:
    """Imagen de salida: cuadros en la geometría mostrada y lectores de metadatos."""

    def __init__(self, ruta: Path) -> None:
        self.ruta = ruta
        self.datos = ruta.read_bytes()
        self.formato = formato_por_firma(self.datos)
        self.img = Image.open(ruta)
        self.img.load()
        self.advertencias_lectura: list[str] = []

    def cerrar(self) -> None:
        self.img.close()

    @cached_property
    def cuadros(self) -> list[np.ndarray]:
        """Cada cuadro (página de un TIFF) con la orientación EXIF aplicada, en RGB (sin alfa)."""
        salida = []
        for cuadro in ImageSequence.Iterator(self.img):
            c = ImageOps.exif_transpose(cuadro.copy())
            if c.mode in ("RGBA", "LA", "PA") or (c.mode == "P" and "transparency" in c.info):
                # los píxeles bajo la transparencia siguen en el archivo: se ignora el alfa
                c = Image.merge("RGB", c.convert("RGBA").split()[:3])
            elif c.mode != "RGB":
                c = c.convert("RGB")
            salida.append(np.asarray(c))
        return salida

    def geometria(self) -> list[tuple[int, int]]:
        return [(c.shape[1], c.shape[0]) for c in self.cuadros]

    @cached_property
    def metadatos(self) -> tuple[list[tuple[str, str]], dict[str, Any], list[str], bytes]:
        """(textos por lugar, estructuras presentes, advertencias, bytes que no son píxeles)."""
        textos: list[tuple[str, str]] = []
        est: dict[str, Any] = {}
        try:
            if self.formato == "jpg":
                resto = leer_jpeg(self.datos, textos, est)
            elif self.formato == "png":
                resto = leer_png(self.datos, textos, est)
            elif self.formato == "webp":
                resto = leer_webp(self.datos, textos, est)
            elif self.formato == "tiff":
                resto = leer_tiff(self.img, self.datos, textos, est)
            else:
                resto = self.datos
        except Exception as ex:  # noqa: BLE001 - contenedor dañado: se revisa todo en bytes
            self.advertencias_lectura.append(f"no se pudo recorrer el contenedor: {ex}")
            resto = self.datos
        # EXIF que PIL encuentre por su cuenta (p. ej. en TIFF o formatos no recorridos)
        if "exif" not in est and self.formato != "tiff":
            try:
                crudo = self.img.info.get("exif")
                if crudo:
                    leer_exif(crudo, textos, est)
            except Exception:  # noqa: BLE001
                pass
        adv = []
        if est.get("exif"):
            etiquetas = est.get("exif.etiquetas", [])
            adv.append(f"EXIF presente ({len(etiquetas)} etiquetas: {', '.join(etiquetas[:8])})")
        if est.get("exif.gps"):
            adv.append("bloque GPS presente")
        if est.get("exif.miniatura"):
            adv.append("miniatura EXIF presente")
        for clave, texto in (
            ("xmp", "XMP"),
            ("iptc", "IPTC"),
            ("com", "comentario JPEG"),
            ("icc", "perfil de color ICC"),
        ):
            if est.get(clave):
                adv.append(f"{texto} presente")
        if est.get("png.texto"):
            adv.append("bloques de texto PNG: " + ", ".join(est["png.texto"]))
        if est.get("tiff.etiquetas"):
            adv.append("etiquetas TIFF: " + ", ".join(est["tiff.etiquetas"][:12]))
        for k in ("jpeg.app", "png.otros", "webp.otros"):
            if est.get(k):
                adv.append(f"{k}: {', '.join(est[k])}")
        for k in ("jpeg.datos_tras_eoi", "png.datos_tras_iend", "webp.datos_tras_riff"):
            if est.get(k):
                adv.append(f"{est[k]} bytes después del fin de la imagen")
        return [(d, t) for d, t in textos if t], est, adv, resto

    @cached_property
    def pajar_bytes(self) -> Pajar:
        pajar = Pajar()
        pajar.agregar("bytes", textos_de_bytes(self.metadatos[3]), binario=True)
        return pajar


def estructura_tiff(donde: str) -> str | None:
    """Clave de ``estructuras`` para un ``Metadato.donde`` del tipo ``tiff.<nombre>``."""
    nombre = donde.split(".", 1)[1].lower() if "." in donde else ""
    if nombre.isdigit():
        return f"tiff.{int(nombre)}"
    for tag, n in ETIQUETAS_TIFF.items():
        if n == nombre:
            return f"tiff.{tag}"
    if nombre.replace("_", "") in _ALIAS_TIFF:
        return f"tiff.{_ALIAS_TIFF[nombre.replace('_', '')]}"
    return None
