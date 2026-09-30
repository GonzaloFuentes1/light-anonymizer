"""Geometría rasterizada: cobertura de un polígono por las zonas censuradas y uniformidad de píxeles.

Convención de coordenadas continuas (la misma de ``banco_pruebas.lienzo``): el píxel (i, j)
ocupa [i, i+1) x [j, j+1) y su centro es (i + 0,5, j + 0,5). Un píxel pertenece a un polígono
si su centro cae dentro.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from banco_pruebas.esquema import Poligono

UMBRAL_COBERTURA = 0.95
UMBRAL_ROSTRO_NUCLEO = 0.95
UMBRAL_ROSTRO_CAJA = 0.80
UMBRAL_ROSTRO_SIN_NUCLEO = 0.90

# Sobremuestreo para la cobertura: muestras por unidad de coordenada.
MUESTRAS_PDF = 8  # por punto
MUESTRAS_IMAGEN = 4  # por píxel
_MAX_MUESTRAS = 6_000_000  # tope de la ventana; si se supera se reduce el sobremuestreo

# Uniformidad de píxeles (comprobaciones P e I)
TOLERANCIA_COLOR = 40
UMBRAL_UNIFORME = 0.95
UMBRAL_UNIFORME_PEQUENO = 0.90
AREA_PEQUENA = 20.0  # px²
# Control de P: una zona que correlaciona así con la entrada no fue censurada (bajo contraste)
UMBRAL_SIN_CAMBIOS = 0.90
_DESVIACION_MINIMA = 1.0  # niveles de gris; bajo esto la zona es plana y no hay correlación que medir

_SHIFT = 4  # bits fraccionarios para cv2.fillPoly


def _a_cv(puntos: np.ndarray) -> np.ndarray:
    """Coordenadas continuas -> enteros con ``_SHIFT`` bits fraccionarios en la convención de OpenCV."""
    return np.round((puntos - 0.5) * (1 << _SHIFT)).astype(np.int32)


def rellenar(mascara: np.ndarray, poligono: Sequence[Sequence[float]], origen=(0.0, 0.0), escala: float = 1.0) -> None:
    """Pinta (valor 1) en ``mascara`` los píxeles cuyo centro cae dentro del polígono."""
    p = (np.asarray(poligono, dtype=np.float64) - np.asarray(origen, dtype=np.float64)) * escala
    cv2.fillPoly(mascara, [_a_cv(p)], 1, lineType=cv2.LINE_8, shift=_SHIFT)


def area(poligono: Sequence[Sequence[float]]) -> float:
    p = np.asarray(poligono, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def caja(poligono: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    p = np.asarray(poligono, dtype=np.float64)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def cajas_se_tocan(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def punto_en_poligono(x: float, y: float, poligono: Sequence[Sequence[float]]) -> bool:
    contorno = np.asarray(poligono, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contorno, (float(x), float(y)), False) >= 0


def puntos_en_poligono(puntos: np.ndarray, poligono: Sequence[Sequence[float]]) -> np.ndarray:
    """Máscara booleana de los puntos (N, 2) que caen dentro del polígono (rayo par-impar)."""
    if len(puntos) == 0:
        return np.zeros(0, dtype=bool)
    poly = np.asarray(poligono, dtype=np.float64)
    x, y = puntos[:, 0], puntos[:, 1]
    dentro = np.zeros(len(puntos), dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        cruza = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi)
        dentro ^= cruza
        xj, yj = xi, yi
    return dentro


def cobertura(poligono: Poligono, zonas: Sequence[Poligono], muestras: float) -> float:
    """Fracción del polígono cubierta por la unión de ``zonas`` (misma página y coordenadas)."""
    x0, y0, x1, y1 = caja(poligono)
    candidatas = [z for z in zonas if cajas_se_tocan((x0, y0, x1, y1), caja(z))]
    if not candidatas:
        return 0.0
    s = float(muestras)
    ancho, alto = (x1 - x0) * s, (y1 - y0) * s
    if ancho * alto > _MAX_MUESTRAS:
        s *= float(np.sqrt(_MAX_MUESTRAS / (ancho * alto)))
    w = int(np.ceil((x1 - x0) * s)) + 2
    h = int(np.ceil((y1 - y0) * s)) + 2
    origen = (x0 - 1 / s, y0 - 1 / s)
    gt = np.zeros((h, w), np.uint8)
    rellenar(gt, poligono, origen, s)
    total = int(gt.sum())
    if total == 0:
        # polígono degenerado: se evalúa su centroide
        cx, cy = np.asarray(poligono, dtype=np.float64).mean(axis=0)
        return 1.0 if any(punto_en_poligono(cx, cy, z) for z in candidatas) else 0.0
    cz = np.zeros((h, w), np.uint8)
    for z in candidatas:
        rellenar(cz, z, origen, s)
    return float((gt & cz).sum()) / total


def se_intersectan(a: Poligono, b: Poligono, muestras: float = 2.0) -> bool:
    """¿Comparten área los dos polígonos? (rasterizado en la ventana común)."""
    ca, cb = caja(a), caja(b)
    if not cajas_se_tocan(ca, cb):
        return False
    x0, y0 = max(ca[0], cb[0]), max(ca[1], cb[1])
    x1, y1 = min(ca[2], cb[2]), min(ca[3], cb[3])
    s = float(muestras)
    if (x1 - x0) * (y1 - y0) * s * s > _MAX_MUESTRAS:
        s = float(np.sqrt(_MAX_MUESTRAS / max(1e-9, (x1 - x0) * (y1 - y0))))
    w, h = int(np.ceil((x1 - x0) * s)) + 2, int(np.ceil((y1 - y0) * s)) + 2
    origen = (x0 - 1 / s, y0 - 1 / s)
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    rellenar(ma, a, origen, s)
    rellenar(mb, b, origen, s)
    return bool((ma & mb).any())


# ---------------------------------------------------------------------------
# Uniformidad de píxeles
# ---------------------------------------------------------------------------


def pixeles_poligono(img: np.ndarray, poligono: Sequence[Sequence[float]]) -> tuple[np.ndarray, float]:
    """Píxeles (N, C) de ``img`` dentro del polígono (recortado a la imagen) y el área del polígono en px².

    Si el polígono es grande se descarta un anillo de 1 px en el borde: ahí el suavizado
    (antialiasing) de la zona rellenada mezcla colores aunque la censura sea perfecta.
    """
    alto, ancho = img.shape[:2]
    x0, y0, x1, y1 = caja(poligono)
    ix0, iy0 = max(0, int(np.floor(x0)) - 1), max(0, int(np.floor(y0)) - 1)
    ix1, iy1 = min(ancho, int(np.ceil(x1)) + 1), min(alto, int(np.ceil(y1)) + 1)
    a = area(poligono)
    if ix1 <= ix0 or iy1 <= iy0:
        return img[:0, :0].reshape(0, img.shape[2] if img.ndim == 3 else 1), a
    m = np.zeros((iy1 - iy0, ix1 - ix0), np.uint8)
    rellenar(m, poligono, (ix0, iy0))
    if m.sum() == 0:
        # polígono más angosto que un píxel: se toma el píxel del centroide
        cx, cy = np.asarray(poligono, dtype=np.float64).mean(axis=0)
        cx, cy = int(np.clip(cx - ix0, 0, m.shape[1] - 1)), int(np.clip(cy - iy0, 0, m.shape[0] - 1))
        m[cy, cx] = 1
    else:
        erosionada = cv2.erode(m, np.ones((3, 3), np.uint8))
        if erosionada.sum() >= 0.5 * m.sum() and erosionada.sum() >= 12:
            m = erosionada
    ventana = img[iy0:iy1, ix0:ix1]
    return ventana[m.astype(bool)], a


def uniformidad(pixeles: np.ndarray) -> float:
    """Fracción de píxeles a distancia (máximo por canal) <= 40 del color modal de la región."""
    if len(pixeles) == 0:
        return 1.0
    p = pixeles.reshape(len(pixeles), -1).astype(np.int16)
    q = (p // 16).astype(np.int32)
    codigo = np.zeros(len(p), np.int32)
    for c in range(q.shape[1]):
        codigo = codigo * 16 + q[:, c]
    modal = np.bincount(codigo).argmax()
    color = np.median(p[codigo == modal], axis=0)
    dif = np.abs(p - color).max(axis=1)
    return float((dif <= TOLERANCIA_COLOR).mean())


def es_uniforme(
    img: np.ndarray, poligono: Sequence[Sequence[float]], umbral: float | None = None
) -> tuple[bool | None, float | None]:
    """Comprobación P/I sobre ``img`` (alto, ancho, canales). ``None`` si el polígono queda fuera.

    Umbral: 95 % de los píxeles con el color modal (90 % si el polígono mide menos de 20 px²),
    salvo que se indique otro (rostros sin núcleo usan el mismo umbral que su cobertura).
    """
    pix, a = pixeles_poligono(img, poligono)
    if len(pix) == 0:
        return None, None
    f = uniformidad(pix)
    if umbral is None:
        umbral = UMBRAL_UNIFORME_PEQUENO if a < AREA_PEQUENA else UMBRAL_UNIFORME
    return f >= umbral, f


def correlacion(salida: np.ndarray, entrada: np.ndarray, poligono: Sequence[Sequence[float]]) -> float | None:
    """Correlación de Pearson (en gris) entre la zona de la salida y la misma zona de la entrada.

    Alta (>= ``UMBRAL_SIN_CAMBIOS``) significa que los píxeles siguen siendo los originales (o
    casi: recomprimidos), aunque la zona parezca uniforme. ``None`` si alguna de las dos es plana
    (una zona rellenada no tiene textura con qué correlacionar).
    """
    alto = min(salida.shape[0], entrada.shape[0])
    ancho = min(salida.shape[1], entrada.shape[1])
    a, _ = pixeles_poligono(salida[:alto, :ancho], poligono)
    b, _ = pixeles_poligono(entrada[:alto, :ancho], poligono)
    if len(a) < 8 or len(a) != len(b):
        return None
    ga = a.reshape(len(a), -1).astype(np.float64).mean(axis=1)
    gb = b.reshape(len(b), -1).astype(np.float64).mean(axis=1)
    if ga.std() < _DESVIACION_MINIMA or gb.std() < _DESVIACION_MINIMA:
        return None
    return float(np.corrcoef(ga, gb)[0, 1])
