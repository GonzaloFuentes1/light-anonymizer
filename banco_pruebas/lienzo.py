"""Dibujo de texto e imágenes con registro exacto de la geometría de cada elemento.

Todas las coordenadas son continuas: el píxel (i, j) ocupa [i, i+1) x [j, j+1). Las matrices
de transformación se expresan en esas coordenadas; al llamar a OpenCV (que usa el centro del
píxel como coordenada entera) se convierten con ``_a_convencion_cv``.
"""

from __future__ import annotations

import copy
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from banco_pruebas.esquema import Elemento, Poligono

# ---------------------------------------------------------------------------
# Fuentes (DejaVu y STIX vienen con matplotlib; licencias libres, ver LICENCIAS.md)
# ---------------------------------------------------------------------------

_DIR_FUENTES = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"

FUENTES = {
    "sans": "DejaVuSans.ttf",
    "sans_negrita": "DejaVuSans-Bold.ttf",
    "sans_oblicua": "DejaVuSans-Oblique.ttf",
    "serif": "DejaVuSerif.ttf",
    "serif_negrita": "DejaVuSerif-Bold.ttf",
    "serif_cursiva": "DejaVuSerif-Italic.ttf",
    "mono": "DejaVuSansMono.ttf",
    "mono_negrita": "DejaVuSansMono-Bold.ttf",
    "stix": "STIXGeneral.ttf",
    "stix_cursiva": "STIXGeneralItalic.ttf",
}


def ruta_fuente(nombre: str) -> Path:
    return _DIR_FUENTES / FUENTES[nombre]


@lru_cache(maxsize=256)
def fuente(nombre: str, tam: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(ruta_fuente(nombre)), tam)


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------


def rect(x0: float, y0: float, x1: float, y1: float) -> Poligono:
    """Rectángulo alineado a los ejes como polígono de 4 puntos (horario)."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def transformar_puntos(poligono: Poligono, m: np.ndarray) -> Poligono:
    """Aplica una matriz 3x3 (afín u homografía) en coordenadas continuas."""
    p = np.asarray(poligono, dtype=np.float64)
    h = np.hstack([p, np.ones((len(p), 1))]) @ m.T
    h = h[:, :2] / h[:, 2:3]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in h]


def transformar_elementos(elementos: Iterable[Elemento], m: np.ndarray) -> list[Elemento]:
    salida = []
    for e in elementos:
        e2 = copy.deepcopy(e)
        if e2.poligono is not None:
            e2.poligono = transformar_puntos(e2.poligono, m)
        if e2.nucleo is not None:
            e2.nucleo = transformar_puntos(e2.nucleo, m)
        salida.append(e2)
    return salida


def _traslacion(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def _a_convencion_cv(m: np.ndarray) -> np.ndarray:
    """Matriz en coordenadas continuas -> matriz para cv2 (centros de píxel en enteros)."""
    return _traslacion(-0.5, -0.5) @ m @ _traslacion(0.5, 0.5)


def matriz_escala(sx: float, sy: float | None = None) -> np.ndarray:
    sy = sx if sy is None else sy
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def matriz_rotacion(ancho: float, alto: float, grados: float, expandir: bool = True) -> tuple[np.ndarray, int, int]:
    """Rotación antihoraria (visual) en torno al centro. Devuelve (matriz, ancho_nuevo, alto_nuevo)."""
    t = math.radians(grados)
    c, s = math.cos(t), math.sin(t)
    cx, cy = ancho / 2, alto / 2
    if expandir:
        nuevo_w = abs(ancho * c) + abs(alto * s)
        nuevo_h = abs(ancho * s) + abs(alto * c)
        nuevo_w_i, nuevo_h_i = int(math.ceil(nuevo_w - 1e-6)), int(math.ceil(nuevo_h - 1e-6))
    else:
        nuevo_w_i, nuevo_h_i = int(ancho), int(alto)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
    m = _traslacion(nuevo_w_i / 2, nuevo_h_i / 2) @ rot @ _traslacion(-cx, -cy)
    return m, nuevo_w_i, nuevo_h_i


def area_poligono(poligono: Poligono) -> float:
    p = np.asarray(poligono, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def caja_envolvente(poligono: Poligono) -> tuple[float, float, float, float]:
    p = np.asarray(poligono, dtype=np.float64)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


# ---------------------------------------------------------------------------
# Lienzo
# ---------------------------------------------------------------------------


@dataclass
class Lienzo:
    """Imagen RGB más la lista de elementos (con geometría exacta) dibujados en ella."""

    img: Image.Image
    elementos: list[Elemento] = field(default_factory=list)

    @classmethod
    def nuevo(cls, ancho: int, alto: int, fondo: tuple[int, int, int] = (255, 255, 255)) -> Lienzo:
        return cls(Image.new("RGB", (ancho, alto), fondo))

    @property
    def ancho(self) -> int:
        return self.img.width

    @property
    def alto(self) -> int:
        return self.img.height

    # -- texto ---------------------------------------------------------------

    def escribir(
        self,
        x: float,
        y: float,
        texto: str,
        *,
        tipo: str = "texto",
        valor: str | None = None,
        nombre_fuente: str = "sans",
        tam: int = 24,
        color: tuple[int, ...] = (25, 25, 25),
        nivel: str = "base",
        etiquetas: dict[str, Any] | None = None,
        anchor: str = "ls",
        registrar: bool = True,
    ) -> Elemento | None:
        """Escribe ``texto`` con su línea base en (x, y) y registra la caja exacta de tinta.

        ``valor`` es el dato a buscar en la salida; por defecto es el mismo texto.
        """
        f = fuente(nombre_fuente, tam)
        draw = ImageDraw.Draw(self.img)
        draw.text((x, y), texto, font=f, fill=color, anchor=anchor)
        if not registrar:
            return None
        caja = _caja_tinta(texto, f, x, y, anchor)
        if caja is None:
            return None
        e = Elemento(
            tipo=tipo,
            pagina=0,
            poligono=rect(*caja),
            valor=valor if valor is not None else texto,
            nivel=nivel,
            capa="raster",
            etiquetas={"fuente": nombre_fuente, "tam_px": tam, "angulo": 0, **(etiquetas or {})},
        )
        self.elementos.append(e)
        return e

    def linea(
        self,
        x: float,
        y: float,
        partes: list[tuple[str, str]],
        **kwargs: Any,
    ) -> list[Elemento]:
        """Escribe segmentos consecutivos en la misma línea base: ``[(texto, tipo), ...]``.

        Los segmentos de tipo ``"texto"`` son etiquetas neutras ("RUT: "); los demás, datos.
        Devuelve los elementos registrados.
        """
        f = fuente(kwargs.get("nombre_fuente", "sans"), kwargs.get("tam", 24))
        registrados = []
        for texto, tipo in partes:
            e = self.escribir(x, y, texto, tipo=tipo, **kwargs)
            if e is not None:
                registrados.append(e)
            x += ImageDraw.Draw(self.img).textlength(texto, font=f)
        return registrados

    def ancho_texto(self, texto: str, nombre_fuente: str = "sans", tam: int = 24) -> float:
        return ImageDraw.Draw(self.img).textlength(texto, font=fuente(nombre_fuente, tam))

    def escribir_rotado(
        self,
        cx: float,
        cy: float,
        texto: str,
        grados: float,
        *,
        tipo: str = "texto",
        valor: str | None = None,
        nombre_fuente: str = "sans",
        tam: int = 24,
        color: tuple[int, int, int] = (25, 25, 25),
        nivel: str = "base",
        etiquetas: dict[str, Any] | None = None,
    ) -> Elemento:
        """Escribe texto girado ``grados`` (antihorario) centrado en (cx, cy), como un timbre."""
        f = fuente(nombre_fuente, tam)
        margen = tam
        x0, t, r, b = f.getbbox(texto, anchor="ls")
        w, h = int(r - x0 + 2 * margen), int(b - t + 2 * margen)
        parche = Lienzo(Image.new("RGBA", (w, h), (0, 0, 0, 0)))
        e = parche.escribir(
            margen - x0,
            margen - t,
            texto,
            tipo=tipo,
            valor=valor,
            nombre_fuente=nombre_fuente,
            tam=tam,
            color=(*color, 255),
            nivel=nivel,
            etiquetas=etiquetas,
        )
        assert e is not None
        m, nw, nh = matriz_rotacion(w, h, grados)
        girado = cv2.warpAffine(
            np.asarray(parche.img),
            _a_convencion_cv(m)[:2],
            (nw, nh),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        girado_img = Image.fromarray(girado, "RGBA")
        px, py = int(round(cx - nw / 2)), int(round(cy - nh / 2))
        self.img.paste(girado_img.convert("RGB"), (px, py), girado_img.getchannel("A"))
        m = _traslacion(px, py) @ m
        (e2,) = transformar_elementos([e], m)
        e2.etiquetas["angulo"] = grados
        self.elementos.append(e2)
        return e2

    def escribir_irregular(
        self,
        x: float,
        y: float,
        texto: str,
        rng: np.random.Generator,
        *,
        tipo: str = "texto",
        valor: str | None = None,
        nombre_fuente: str = "sans_oblicua",
        tam: int = 28,
        color: tuple[int, int, int] = (20, 30, 90),
        nivel: str = "estres",
        etiquetas: dict[str, Any] | None = None,
        jitter: float = 0.12,
    ) -> Elemento | None:
        """Texto con letras de tamaño, giro y posición irregulares (imita escritura a mano)."""
        f0 = fuente(nombre_fuente, tam)
        cajas = []
        for ch in texto:
            if ch == " ":
                x += f0.getlength(" ")
                continue
            tam_c = max(8, int(tam * (1 + rng.uniform(-jitter, jitter))))
            f = fuente(nombre_fuente, tam_c)
            x0, t, r, b = f.getbbox(ch, anchor="ls")
            m = tam_c
            parche = Image.new("RGBA", (int(r - x0 + 2 * m), int(b - t + 2 * m)), (0, 0, 0, 0))
            ImageDraw.Draw(parche).text((m - x0, m - t), ch, font=f, fill=(*color, 255), anchor="ls")
            parche = parche.rotate(float(rng.uniform(-12, 12)), resample=Image.Resampling.BICUBIC, expand=True)
            px = int(round(x + x0 - m + rng.uniform(-1.5, 1.5)))
            py = int(round(y + t - m + rng.uniform(-3, 3)))
            self.img.paste(parche, (px, py), parche)
            bb = parche.getchannel("A").getbbox()
            if bb:
                cajas.append((px + bb[0], py + bb[1], px + bb[2], py + bb[3]))
            x += f.getlength(ch) * (1 + rng.uniform(-0.05, 0.1))
        if not cajas:
            return None
        c = np.array(cajas, dtype=np.float64)
        e = Elemento(
            tipo=tipo,
            pagina=0,
            poligono=rect(c[:, 0].min(), c[:, 1].min(), c[:, 2].max(), c[:, 3].max()),
            valor=valor if valor is not None else texto,
            nivel=nivel,
            capa="raster",
            etiquetas={"fuente": nombre_fuente, "tam_px": tam, "angulo": 0, "irregular": True, **(etiquetas or {})},
        )
        self.elementos.append(e)
        return e

    # -- imágenes ------------------------------------------------------------

    def pegar(
        self,
        img: Image.Image,
        x: float,
        y: float,
        ancho: int | None = None,
        elementos: Iterable[Elemento] = (),
        mascara: Image.Image | None = None,
    ) -> list[Elemento]:
        """Pega ``img`` (opcionalmente reescalada a ``ancho``) con su esquina en (x, y).

        ``elementos`` están en coordenadas de ``img`` y se trasladan/escalan al lienzo.
        """
        sx = sy = 1.0
        if ancho is not None and ancho != img.width:
            nuevo = (ancho, max(1, int(round(img.height * ancho / img.width))))
            sx, sy = nuevo[0] / img.width, nuevo[1] / img.height
            img = img.resize(nuevo, Image.Resampling.LANCZOS)
            if mascara is not None:
                mascara = mascara.resize(nuevo, Image.Resampling.LANCZOS)
        xi, yi = int(round(x)), int(round(y))
        if mascara is None and img.mode == "RGBA":
            mascara = img.getchannel("A")
        self.img.paste(img.convert("RGB"), (xi, yi), mascara)
        m = _traslacion(xi, yi) @ matriz_escala(sx, sy)
        nuevos = transformar_elementos(elementos, m)
        self.elementos.extend(nuevos)
        return nuevos

    # -- transformaciones geométricas (devuelven un lienzo nuevo) -------------

    def rotar(self, grados: float, fondo: tuple[int, int, int] = (255, 255, 255)) -> Lienzo:
        """Gira la imagen completa (antihorario) expandiendo el lienzo. Exacto para múltiplos de 90."""
        g = grados % 360
        w, h = self.ancho, self.alto
        if g in (0, 90, 180, 270):
            transpuesta = {
                0: None,
                90: Image.Transpose.ROTATE_90,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_270,
            }[int(g)]
            img = self.img.copy() if transpuesta is None else self.img.transpose(transpuesta)
            m = {
                0: np.eye(3),
                90: np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64),
                180: np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float64),
                270: np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
            }[int(g)]
        else:
            m, nw, nh = matriz_rotacion(w, h, g)
            arr = cv2.warpAffine(
                np.asarray(self.img),
                _a_convencion_cv(m)[:2],
                (nw, nh),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=fondo,
            )
            img = Image.fromarray(arr)
        nuevos = transformar_elementos(self.elementos, m)
        for e in nuevos:
            e.etiquetas["angulo"] = (e.etiquetas.get("angulo", 0) + grados) % 360
        return Lienzo(img, nuevos)

    def espejar(self, horizontal: bool = True) -> Lienzo:
        w, h = self.ancho, self.alto
        if horizontal:
            img = self.img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            m = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        else:
            img = self.img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            m = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]], dtype=np.float64)
        nuevos = transformar_elementos(self.elementos, m)
        for e in nuevos:
            e.etiquetas["espejo"] = "horizontal" if horizontal else "vertical"
        return Lienzo(img, nuevos)

    def escalar(self, factor: float) -> Lienzo:
        nuevo = (max(1, int(round(self.ancho * factor))), max(1, int(round(self.alto * factor))))
        img = self.img.resize(nuevo, Image.Resampling.LANCZOS)
        m = matriz_escala(nuevo[0] / self.ancho, nuevo[1] / self.alto)
        return Lienzo(img, transformar_elementos(self.elementos, m))

    def inclinar_escaneo(self, grados: float, fondo: tuple[int, int, int] = (250, 250, 247)) -> Lienzo:
        """Giro leve sin expandir (como una hoja mal puesta en el escáner)."""
        m, nw, nh = matriz_rotacion(self.ancho, self.alto, grados, expandir=False)
        arr = cv2.warpAffine(
            np.asarray(self.img),
            _a_convencion_cv(m)[:2],
            (nw, nh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fondo,
        )
        nuevos = transformar_elementos(self.elementos, m)
        for e in nuevos:
            e.etiquetas["angulo"] = (e.etiquetas.get("angulo", 0) + grados) % 360
        return Lienzo(Image.fromarray(arr), nuevos)

    def perspectiva(
        self,
        destino: Poligono,
        fondo: Image.Image,
    ) -> Lienzo:
        """Proyecta la imagen completa sobre el cuadrilátero ``destino`` dentro de ``fondo``.

        ``destino`` son las 4 esquinas (sup-izq, sup-der, inf-der, inf-izq) en coordenadas de ``fondo``.
        """
        w, h = self.ancho, self.alto
        origen = np.array(rect(0, 0, w, h), dtype=np.float32)
        dst = np.array(destino, dtype=np.float32)
        m = cv2.getPerspectiveTransform(origen, dst).astype(np.float64)
        tam = (fondo.width, fondo.height)
        mcv = _a_convencion_cv(m)
        arr = cv2.warpPerspective(
            np.asarray(self.img), mcv, tam, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), mcv, tam, flags=cv2.INTER_LINEAR)
        base = np.asarray(fondo.convert("RGB")).astype(np.float32)
        alfa = (mask.astype(np.float32) / 255.0)[..., None]
        comp = (arr.astype(np.float32) * alfa + base * (1 - alfa)).clip(0, 255).astype(np.uint8)
        nuevos = transformar_elementos(self.elementos, m)
        for e in nuevos:
            e.etiquetas["perspectiva"] = True
        return Lienzo(Image.fromarray(comp), nuevos)

    def copiar(self) -> Lienzo:
        return Lienzo(self.img.copy(), copy.deepcopy(self.elementos))


def _caja_tinta(texto: str, f: ImageFont.FreeTypeFont, x: float, y: float, anchor: str) -> tuple | None:
    """Caja exacta de los píxeles de tinta del texto, dibujándolo en una máscara aparte."""
    x0, t, r, b = f.getbbox(texto, anchor=anchor)
    m = max(4, f.size // 2)
    w, h = int(math.ceil(r - x0)) + 2 * m, int(math.ceil(b - t)) + 2 * m
    if w <= 2 * m or h <= 2 * m:
        return None
    mask = Image.new("L", (w, h), 0)
    ox, oy = m - x0, m - t
    # Se replica el mismo desplazamiento fraccional que usa ImageDraw al dibujar en (x, y).
    ImageDraw.Draw(mask).text(
        (ox + (x - math.floor(x)), oy + (y - math.floor(y))), texto, font=f, fill=255, anchor=anchor
    )
    bb = mask.getbbox()
    if bb is None:
        return None
    bx, by = math.floor(x) - ox, math.floor(y) - oy
    return (bx + bb[0], by + bb[1], bx + bb[2], by + bb[3])


# ---------------------------------------------------------------------------
# Degradaciones (no cambian la geometría)
# ---------------------------------------------------------------------------


def ruido(img: Image.Image, rng: np.random.Generator, sigma: float = 8.0) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr = arr + rng.normal(0, sigma, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def sal_pimienta(img: Image.Image, rng: np.random.Generator, fraccion: float = 0.002) -> Image.Image:
    arr = np.asarray(img).copy()
    n = int(arr.shape[0] * arr.shape[1] * fraccion)
    ys, xs = rng.integers(0, arr.shape[0], n), rng.integers(0, arr.shape[1], n)
    arr[ys[: n // 2], xs[: n // 2]] = 0
    arr[ys[n // 2 :], xs[n // 2 :]] = 255
    return Image.fromarray(arr)


def desenfocar(img: Image.Image, radio: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radio))


def comprimir_jpeg(img: Image.Image, calidad: int = 60) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=calidad)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def iluminacion(img: Image.Image, rng: np.random.Generator, intensidad: float = 0.35) -> Image.Image:
    """Gradiente de luz y viñeta, como una foto tomada con el celular."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5)) * 2
    vin = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2
    factor = 1 - intensidad * (0.5 * (grad + 1) * 0.6 + vin * 0.4)
    arr = np.asarray(img).astype(np.float32) * factor[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def textura_papel(ancho: int, alto: int, rng: np.random.Generator, color=(248, 246, 240)) -> Image.Image:
    base = np.ones((alto, ancho, 3), np.float32) * np.array(color, np.float32)
    ruido_bajo = cv2.resize(rng.normal(0, 1, (alto // 16 + 1, ancho // 16 + 1)).astype(np.float32), (ancho, alto))
    base += ruido_bajo[..., None] * 3 + rng.normal(0, 2, (alto, ancho, 1))
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def textura_mesa(ancho: int, alto: int, rng: np.random.Generator) -> Image.Image:
    """Fondo tipo madera para fotos de documentos sobre una mesa."""
    yy, xx = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    vetas = np.sin(xx / ancho * rng.uniform(20, 40) + np.sin(yy / alto * 6) * 2.5) * 0.5 + 0.5
    ruido_bajo = cv2.resize(rng.normal(0, 1, (alto // 24 + 1, ancho // 24 + 1)).astype(np.float32), (ancho, alto))
    base = np.array([120, 84, 52], np.float32)
    arr = base + (vetas[..., None] * 30) + ruido_bajo[..., None] * 8 + rng.normal(0, 3, (alto, ancho, 1))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Paso de imagen a PDF
# ---------------------------------------------------------------------------


def px_a_pt(elementos: Iterable[Elemento], dpi: float) -> list[Elemento]:
    return transformar_elementos(elementos, matriz_escala(72.0 / dpi))


def mapear_a_rect(
    elementos: Iterable[Elemento], ancho_px: int, alto_px: int, destino: tuple[float, float, float, float]
) -> list[Elemento]:
    """Lleva elementos en píxeles de una imagen a la caja ``destino`` (x0, y0, x1, y1) de una página PDF."""
    x0, y0, x1, y1 = destino
    m = _traslacion(x0, y0) @ matriz_escala((x1 - x0) / ancho_px, (y1 - y0) / alto_px)
    return transformar_elementos(elementos, m)
