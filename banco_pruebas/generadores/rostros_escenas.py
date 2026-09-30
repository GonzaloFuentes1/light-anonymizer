"""Rostros sueltos y en escenas: retratos, giros, grupos compuestos, fotos reales y un afiche fotografiado.

Todos los rostros vienen de ``ctx.rostros`` (``ProveedorRostros``): ``tomar(pose, n)`` para caras
sueltas con su caja y su núcleo, y ``escenas()`` para fotos reales con varias personas anotadas.
El código no supone nada del tamaño ni del aspecto de las imágenes de origen: toda la geometría
se deriva de ``caja`` y ``nucleo`` y se transforma junto con los píxeles.

Convenciones de la verdad de terreno de este módulo:

- ``poligono`` = caja de la cara transformada; ``nucleo`` = núcleo transformado (o ``None`` si
  la escena real no lo trae).
- ``etiquetas["alto_px"]`` = alto de la cara en la imagen de salida, medido en su propio marco
  (largo del borde "vertical" de la caja: no cambia al girar). ``alto_envolvente_px`` es el
  alto de la caja envolvente alineada a los ejes.
- ``etiquetas["angulo"]`` = giro antihorario acumulado, en grados.
"""

from __future__ import annotations

import math
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from banco_pruebas.contexto import Contexto, archivo_imagen, guardar_imagen
from banco_pruebas.esquema import Archivo, Elemento, Poligono
from banco_pruebas.ficticios import FORMATOS_CORREO, FORMATOS_POR_CLASE, FORMATOS_TELEFONO, Ficticios, Persona
from banco_pruebas.lienzo import (
    Lienzo,
    caja_envolvente,
    desenfocar,
    iluminacion,
    matriz_escala,
    matriz_rotacion,
    ruido,
    textura_mesa,
    transformar_elementos,
)
from banco_pruebas.rostros import ProveedorRostros, Rostro

MODULO = "rostros_escenas"
CATEGORIA = "rostros"
CARPETA = "rostros"
PREFIJO = "rost"

LADO_RETRATO = (420, 780)  # lado mayor de los retratos sueltos (dentro de 400-800)
LADO_MAX_ESCENA = 1600
ALTO_MIN_BASE_ESCENA = 24  # bajo esto una cara de escena real es "estres"
ALTO_MIN_BASE_GRUPO = 40  # en los grupos compuestos, 20-40 px es "estres"

PAREDES = [(214, 206, 190), (196, 210, 214), (222, 214, 196), (205, 196, 210), (190, 204, 186)]
ROPA = [(40, 60, 110), (120, 30, 40), (60, 90, 60), (30, 30, 35), (150, 150, 155), (180, 120, 50), (90, 50, 110)]
COLORES_AFICHE = [(22, 92, 125), (150, 40, 50), (40, 110, 70), (90, 60, 130)]
FRASES_AFICHE = [
    "Inscripciones abiertas para vecinos y vecinas de la comuna.",
    "Actividad gratuita, con cupos limitados por orden de llegada.",
    "Lugar: salón multiuso de la junta de vecinos del sector.",
]


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def generar(ctx: Contexto) -> list[Archivo]:
    rng = ctx.rng(MODULO)
    f = ctx.ficticios(MODULO)
    archivos: list[Archivo] = []
    archivos += _retratos(ctx, rng)
    archivos += _rotados(ctx, rng)
    archivos += _grupos(ctx, rng)
    archivos += _escenas_reales(ctx, rng)
    archivos.append(_afiche(ctx, rng, f))
    archivos += _variantes(ctx, rng)
    return archivos


# ---------------------------------------------------------------------------
# Utilidades de geometría y guardado
# ---------------------------------------------------------------------------


def _traslacion(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def _matriz_cv(m: np.ndarray) -> np.ndarray:
    """Matriz en coordenadas continuas -> matriz 2x3 para cv2 (centros de píxel en enteros)."""
    return (_traslacion(-0.5, -0.5) @ m @ _traslacion(0.5, 0.5))[:2]


def _alto_propio(poligono: Poligono) -> float:
    """Alto de la cara en su propio marco: promedio de los bordes izquierdo y derecho de la caja."""
    p = np.asarray(poligono, dtype=np.float64)
    if len(p) != 4:
        x0, y0, x1, y1 = caja_envolvente(poligono)
        return y1 - y0
    return float((np.hypot(*(p[3] - p[0])) + np.hypot(*(p[2] - p[1]))) / 2)


def _alto_cara(e: Elemento) -> float:
    """Alto de la cara: en su propio marco para las generadas; de la caja envolvente en escenas reales.

    Las cajas de escenas reales vienen de anotaciones externas sin orden de vértices garantizado
    (y sin giro), así que ahí se usa el alto de la caja envolvente.
    """
    if "escena_id" in e.etiquetas:
        x0, y0, x1, y1 = caja_envolvente(e.poligono)
        return y1 - y0
    return _alto_propio(e.poligono)


def _completar_rostros(lz: Lienzo) -> None:
    """Anota el alto final de cada cara y normaliza el ángulo."""
    for e in lz.elementos:
        if e.tipo != "rostro" or e.poligono is None:
            continue
        x0, y0, x1, y1 = caja_envolvente(e.poligono)
        e.etiquetas["alto_px"] = round(_alto_cara(e), 1)
        e.etiquetas["alto_envolvente_px"] = round(y1 - y0, 1)
        e.etiquetas["angulo"] = round(float(e.etiquetas.get("angulo", 0)) % 360, 3)


def _guardar_gris(img: Image.Image, destino, calidad: int) -> None:
    """JPEG de un canal, sin metadatos (``guardar_imagen`` siempre convierte a RGB)."""
    gris = img.convert("L")
    Image.frombytes("L", gris.size, gris.tobytes()).save(destino, "JPEG", quality=calidad)


def _guardar(
    ctx: Contexto,
    rng: np.random.Generator,
    lz: Lienzo,
    nombre: str,
    descripcion: str,
    etiquetas: dict[str, Any] | None = None,
    gris: bool = False,
) -> Archivo:
    _completar_rostros(lz)
    ruta = f"{CARPETA}/{nombre}.jpg"
    destino = ctx.ruta(ruta)
    calidad = int(rng.integers(86, 95))
    if gris:
        _guardar_gris(lz.img, destino, calidad)
    else:
        guardar_imagen(lz.img, destino, "jpg", quality=calidad)
    return archivo_imagen(
        id=f"{PREFIJO}_{nombre}",
        ruta=ruta,
        formato="jpg",
        categoria=CATEGORIA,
        descripcion=descripcion,
        lienzo=lz,
        etiquetas={"rostros_sinteticos": ctx.rostros.solo_sinteticos, "calidad_jpeg": calidad, **(etiquetas or {})},
    )


def _retrato(r: Rostro, nivel: str = "base", etiquetas: dict[str, Any] | None = None) -> Lienzo:
    """Lienzo con la foto del rostro tal cual y su elemento (en coordenadas de la foto)."""
    img = r.img.convert("RGB").copy()
    return Lienzo(img, ProveedorRostros.elementos(r, nivel, {"angulo": 0, **(etiquetas or {})}))


def _factor_lado(ancho: float, alto: float, rng: np.random.Generator, lado: tuple[int, int] = LADO_RETRATO) -> float:
    objetivo = float(rng.integers(lado[0], lado[1] + 1))
    return objetivo / max(ancho, alto)


def _ajustar_lado(lz: Lienzo, rng: np.random.Generator, lado: tuple[int, int] = LADO_RETRATO) -> Lienzo:
    return lz.escalar(_factor_lado(lz.ancho, lz.alto, rng, lado))


def _rotar_con_fondo(lz: Lienzo, grados: float, rng: np.random.Generator) -> Lienzo:
    """Gira la foto; si el ángulo no es múltiplo de 90, la deja como una copia impresa sobre una mesa.

    Las esquinas que quedan vacías al girar se rellenan con madera (``textura_mesa``) y una
    sombra suave bajo la foto. Replicar el borde de la propia foto arrastraba el color del cuello
    o la ropa hasta las esquinas y dejaba manchas poco plausibles.
    """
    if grados % 90 == 0:
        return lz.rotar(grados)
    w, h = lz.ancho, lz.alto
    m, nw, nh = matriz_rotacion(w, h, grados)
    mcv = _matriz_cv(m)
    arr = cv2.warpAffine(np.asarray(lz.img), mcv, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    alfa = cv2.warpAffine(
        np.full((h, w), 255, np.uint8), mcv, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    a = alfa.astype(np.float32)[..., None] / 255.0
    mesa = np.asarray(textura_mesa(nw, nh, rng)).astype(np.float32)
    desplazada = cv2.warpAffine(alfa, np.float32([[1, 0, 4], [0, 1, 6]]), (nw, nh)).astype(np.float32)
    sombra = cv2.GaussianBlur(desplazada, (0, 0), 5.0)[..., None] / 255.0
    fondo = mesa * (1 - 0.45 * sombra)
    comp = arr.astype(np.float32) * a + fondo * (1 - a)
    nuevos = transformar_elementos(lz.elementos, m)
    for e in nuevos:
        e.etiquetas["angulo"] = (e.etiquetas.get("angulo", 0) + grados) % 360
        e.etiquetas["relleno_giro"] = "foto_sobre_mesa"
    return Lienzo(Image.fromarray(comp.clip(0, 255).astype(np.uint8)), nuevos)


def _color_piel(img: Image.Image, nucleo: Poligono) -> tuple[int, int, int]:
    """Mediana de color de la franja central del núcleo (mejillas y nariz)."""
    x0, y0, x1, y1 = caja_envolvente(nucleo)
    w, h = x1 - x0, y1 - y0
    caja = (int(x0 + 0.05 * w), int(y0 + 0.4 * h), int(math.ceil(x1 - 0.05 * w)), int(math.ceil(y0 + 0.6 * h)))
    region = np.asarray(img.convert("RGB").crop(caja)).reshape(-1, 3)
    if len(region) == 0:
        return (200, 160, 130)
    med = np.median(region, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


# ---------------------------------------------------------------------------
# 1-2. Retratos sueltos
# ---------------------------------------------------------------------------


def _retratos(ctx: Contexto, rng: np.random.Generator) -> list[Archivo]:
    salida = []
    for i, r in enumerate(ctx.rostros.tomar("frente", 4), 1):
        lz = _ajustar_lado(_retrato(r, "base"), rng)
        salida.append(_guardar(ctx, rng, lz, f"retrato_frente_{i:02d}", "Retrato de frente, una sola persona."))
    for i, r in enumerate(ctx.rostros.tomar("tres_cuartos", 2), 1):
        lz = _ajustar_lado(_retrato(r, "base"), rng)
        salida.append(
            _guardar(ctx, rng, lz, f"retrato_tres_cuartos_{i:02d}", "Retrato en tres cuartos, una sola persona.")
        )
    for i, r in enumerate(ctx.rostros.tomar("perfil", 2), 1):
        lz = _ajustar_lado(_retrato(r, "estres"), rng)
        salida.append(_guardar(ctx, rng, lz, f"perfil_{i:02d}", "Retrato de perfil (difícil para detectores)."))
    return salida


# ---------------------------------------------------------------------------
# 3. Retrato girado
# ---------------------------------------------------------------------------


def _rotados(ctx: Contexto, rng: np.random.Generator) -> list[Archivo]:
    (r,) = ctx.rostros.tomar("frente", 1)
    salida = []
    for g in (90, 180, 270, 15, 45):
        lz = _retrato(r, "base")
        t = math.radians(g)
        rw = abs(lz.ancho * math.cos(t)) + abs(lz.alto * math.sin(t))
        rh = abs(lz.ancho * math.sin(t)) + abs(lz.alto * math.cos(t))
        lz = lz.escalar(_factor_lado(rw, rh, rng))
        lz = _rotar_con_fondo(lz, g, rng)
        salida.append(
            _guardar(
                ctx,
                rng,
                lz,
                f"retrato_rotado_{g}",
                f"Retrato girado {g}° (antihorario); el motor debe encontrar caras en cualquier orientación.",
                {"angulo": g},
            )
        )
    return salida


# ---------------------------------------------------------------------------
# 4. Grupos compuestos
# ---------------------------------------------------------------------------


Elipse = tuple[float, float, float, float]  # cx, cy, rx, ry


def _dentro_elipses(pts: np.ndarray, elipses: list[Elipse]) -> np.ndarray:
    dentro = np.zeros(len(pts), dtype=bool)
    for cx, cy, rx, ry in elipses:
        dentro |= ((pts[:, 0] - cx) / rx) ** 2 + ((pts[:, 1] - cy) / ry) ** 2 <= 1.0
    return dentro


def _malla(caja: tuple[float, float, float, float], n: int = 16, holgura: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = caja
    xs = np.linspace(x0 - holgura, x1 + holgura, n)
    ys = np.linspace(y0 - holgura, y1 + holgura, n)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


def _fondo_sala(w: int, h: int, rng: np.random.Generator) -> Image.Image:
    """Sala simple: pared con degradé de luz, ventana, un cuadro abstracto y piso de madera."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pared = np.array(PAREDES[int(rng.integers(len(PAREDES)))], np.float32)
    luz = 1.0 - 0.16 * (yy / h) - 0.10 * np.abs(xx / w - rng.uniform(0.3, 0.7))
    bajo = cv2.resize(rng.normal(0, 1, (h // 40 + 1, w // 40 + 1)).astype(np.float32), (w, h))
    arr = pared * luz[..., None] + bajo[..., None] * 3
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    piso_y = int(h * rng.uniform(0.74, 0.82))
    img.paste(textura_mesa(w, h - piso_y, rng), (0, piso_y))
    d.rectangle((0, piso_y - 12, w, piso_y), fill=tuple(int(c * 0.7) for c in pared))
    # ventana con cielo
    vw, vh = int(w * rng.uniform(0.18, 0.26)), int(h * rng.uniform(0.3, 0.38))
    vx, vy = int(rng.integers(int(w * 0.05), int(w * 0.7))), int(h * rng.uniform(0.06, 0.12))
    cielo = np.linspace([150, 190, 230], [215, 230, 245], vh).astype(np.uint8)[:, None, :].repeat(vw, 1)
    img.paste(Image.fromarray(cielo), (vx, vy))
    marco = (245, 245, 240)
    d.rectangle((vx - 8, vy - 8, vx + vw + 8, vy + vh + 8), outline=marco, width=10)
    d.line((vx + vw // 2, vy, vx + vw // 2, vy + vh), fill=marco, width=8)
    d.line((vx, vy + vh // 2, vx + vw, vy + vh // 2), fill=marco, width=8)
    # cuadro abstracto en la otra mitad de la pared
    cx0 = int(w * 0.55) if vx < w * 0.4 else int(w * 0.1)
    cw, ch = int(w * 0.16), int(h * 0.2)
    cy0 = int(h * rng.uniform(0.08, 0.16))
    d.rectangle((cx0, cy0, cx0 + cw, cy0 + ch), fill=(60, 45, 35))
    d.rectangle((cx0 + 10, cy0 + 10, cx0 + cw - 10, cy0 + ch - 10), fill=(235, 230, 215))
    for _ in range(4):
        c = tuple(int(v) for v in rng.integers(40, 220, 3))
        a, b = rng.uniform(0.1, 0.6, 2)
        d.rectangle(
            (
                cx0 + 14 + a * cw * 0.6,
                cy0 + 14 + b * ch * 0.6,
                cx0 + 14 + (a + 0.3) * cw * 0.6,
                cy0 + (b + 0.4) * ch * 0.6,
            ),
            fill=c,
        )
    return desenfocar(img, 1.6)


def _preparar_cabeza(r: Rostro) -> tuple[Image.Image, Image.Image, list[Elemento], Elipse]:
    """Recorta la cabeza con una máscara elíptica difuminada que deja el núcleo totalmente opaco.

    Devuelve (recorte, máscara L, elementos en coordenadas del recorte, elipse exterior en el recorte).
    """
    img = r.img.convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = caja_envolvente(r.caja)
    cw, ch = x1 - x0, y1 - y0
    # La caja puede ir de las cejas al mentón: la elipse sube para incluir frente y pelo.
    arriba, abajo = y0 - 0.45 * ch, y1 + 0.08 * ch
    cx, cy = (x0 + x1) / 2, (arriba + abajo) / 2
    rx, ry = 0.62 * cw, (abajo - arriba) / 2
    pluma = 0.05 * min(rx, ry)
    esquinas = np.asarray(r.nucleo, dtype=np.float64)
    for _ in range(30):
        ex, ey = rx - 2.5 * pluma, ry - 2.5 * pluma
        if np.all(((esquinas[:, 0] - cx) / ex) ** 2 + ((esquinas[:, 1] - cy) / ey) ** 2 <= 1.0):
            break
        rx, ry = rx * 1.04, ry * 1.04
    margen = 3 * pluma
    c0 = (max(0, int(cx - rx - margen)), max(0, int(cy - ry - margen)))
    c1 = (min(w, int(math.ceil(cx + rx + margen))), min(h, int(math.ceil(cy + ry + margen))))
    mascara = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mascara).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    mascara = mascara.filter(ImageFilter.GaussianBlur(pluma)).crop((*c0, *c1))
    # rampa en los bordes del recorte, por si la elipse toca el borde de la foto
    mw, mh = mascara.size
    ramp_x = np.minimum(np.arange(mw) + 0.5, mw - np.arange(mw) - 0.5) / max(1.0, pluma)
    ramp_y = np.minimum(np.arange(mh) + 0.5, mh - np.arange(mh) - 0.5) / max(1.0, pluma)
    rampa = np.clip(np.minimum(ramp_y[:, None], ramp_x[None, :]), 0, 1)
    recorte = img.crop((*c0, *c1))
    fondo = _alfa_sin_fondo(img, recorte, c0, r, (x0, y0, x1, y1))
    marr = (np.asarray(mascara).astype(np.float32) * rampa * fondo).astype(np.uint8)
    elementos = transformar_elementos(ProveedorRostros.elementos(r, "base", {"angulo": 0}), _traslacion(-c0[0], -c0[1]))
    elipse = (cx - c0[0], cy - c0[1], rx + 2 * pluma, ry + 2 * pluma)
    return recorte, Image.fromarray(marr, "L"), elementos, elipse


def _alfa_sin_fondo(
    img: Image.Image,
    recorte: Image.Image,
    c0: tuple[int, int],
    r: Rostro,
    caja: tuple[float, float, float, float],
) -> np.ndarray:
    """Alfa (0-1) que quita el fondo liso de estudio alrededor de la cabeza, si lo hay.

    Se miran parches en las esquinas superiores y a media altura de los bordes laterales de la
    foto; los que son lisos (fondo de estudio, sin pelo ni ropa) dan el color de fondo. Los
    píxeles parecidos a ese color se vuelven transparentes, salvo dentro del óvalo de la cara y
    del núcleo, que quedan siempre opacos. Si ningún parche es liso (fondo variado) devuelve unos.
    """
    arr = np.asarray(img).astype(np.float32)
    rec = np.asarray(recorte).astype(np.float32)
    h, w = arr.shape[:2]
    k = max(4, min(h, w) // 24)
    parches = [arr[:k, :k], arr[:k, -k:], arr[h // 2 - k // 2 : h // 2 + k // 2, :k]]
    parches.append(arr[h // 2 - k // 2 : h // 2 + k // 2, -k:])
    lisos = [np.median(p.reshape(-1, 3), axis=0) for p in parches if p.size and p.reshape(-1, 3).std(axis=0).max() < 6]
    if not lisos:
        return np.ones(rec.shape[:2], np.float32)
    # el color de fondo es el que más parches comparten (un parche liso de pelo no lo desplaza)
    cand = np.asarray(lisos)
    cercanos = np.linalg.norm(cand[:, None] - cand[None, :], axis=2) < 20
    mejor = int(np.argmax(cercanos.sum(axis=1) + cand.mean(axis=1) / 1000))
    color = np.median(cand[cercanos[mejor]], axis=0)
    dist = np.linalg.norm(rec - color, axis=2)
    alfa = np.clip((dist - 16) / 26, 0, 1)
    alfa = cv2.morphologyEx(alfa, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # óvalo de la cara y núcleo: siempre opacos
    protegido = Image.new("L", recorte.size, 0)
    dp = ImageDraw.Draw(protegido)
    x0, y0, x1, y1 = (v - d for v, d in zip(caja, (*c0, *c0), strict=True))
    mx, my = 0.1 * (x1 - x0), 0.1 * (y1 - y0)  # óvalo algo menor que la caja: en perfiles la caja incluye fondo
    dp.ellipse((x0 + mx, y0 + my, x1 - mx, y1 - my), fill=255)
    nx0, ny0, nx1, ny1 = caja_envolvente(r.nucleo)
    dp.rectangle((nx0 - c0[0] - 4, ny0 - c0[1] - 4, nx1 - c0[0] + 4, ny1 - c0[1] + 4), fill=255)
    prot = np.asarray(protegido).astype(np.float32) / 255.0
    alfa = np.maximum(alfa, prot)
    return cv2.GaussianBlur(alfa, (0, 0), 1.2)


def _alturas_grupo(rng: np.random.Generator) -> list[tuple[float, str]]:
    n = int(rng.integers(5, 13))
    n_dim = int(rng.integers(2, 4))
    n_gra = int(rng.integers(1, min(4, n - n_dim - 1) + 1))
    n_med = n - n_dim - n_gra
    alturas = [(float(rng.uniform(22, 38)), "diminuto") for _ in range(n_dim)]
    alturas += [(float(rng.uniform(42, 78)), "pequeno") for _ in range(n_med)]
    alturas += [(float(rng.uniform(85, 195)), "grande") for _ in range(n_gra)]
    return sorted(alturas)  # primero las lejanas (chicas): las cercanas tapan a las lejanas


def _grupo(ctx: Contexto, rng: np.random.Generator, ancho: int = 1600, alto: int = 1000) -> Lienzo:
    lz = Lienzo(_fondo_sala(ancho, alto, rng))
    d = ImageDraw.Draw(lz.img)
    alturas = _alturas_grupo(rng)
    forzar = set(int(i) for i in rng.choice(np.arange(1, len(alturas)), size=3, replace=False))
    colocadas: list[dict[str, Any]] = []
    for idx, (alto_obj, clase) in enumerate(alturas):
        pose = ["frente", "frente", "frente", "tres_cuartos", "tres_cuartos", "perfil"][int(rng.integers(6))]
        (r,) = ctx.rostros.tomar(pose, 1)
        recorte, mascara, elems, (ecx, ecy, erx, ery) = _preparar_cabeza(r)
        bx0, by0, bx1, by1 = caja_envolvente(elems[0].poligono)
        s = alto_obj / (by1 - by0)
        nuevo_w = max(1, int(round(recorte.width * s)))
        nuevo_h = max(1, int(round(recorte.height * nuevo_w / recorte.width)))
        sx, sy = nuevo_w / recorte.width, nuevo_h / recorte.height
        fw, fh = (bx1 - bx0) * sx, (by1 - by0) * sy
        colocado = None
        for intento in range(400):
            if idx in forzar and colocadas and intento < 250:
                # junto a una cara ya pegada, con la cabeza nueva metida un poco en su caja
                ref = colocadas[int(rng.integers(len(colocadas)))]
                rx0, ry0, rx1, ry1 = ref["caja"]
                lado = 1 if rng.random() < 0.5 else -1
                semiancho_cabeza = erx * sx
                intrusion = (rx1 - rx0) * rng.uniform(0.04, 0.14)
                fcx = (rx0 + rx1) / 2 + lado * ((rx1 - rx0) / 2 + semiancho_cabeza - intrusion)
                fcy = (ry0 + ry1) / 2 + rng.uniform(-0.1, 0.35) * (ry1 - ry0)
            else:
                banda = 0.25 + 0.45 * min(1.0, alto_obj / 200)  # las caras grandes van más abajo
                fcx = rng.uniform(fw / 2 + 4, ancho - fw / 2 - 4)
                fcy = rng.uniform(alto * (banda - 0.18), alto * (banda + 0.12))
            x = int(round(fcx - (bx0 + bx1) / 2 * sx))
            y = int(round(fcy - (by0 + by1) / 2 * sy))
            caja = (x + bx0 * sx, y + by0 * sy, x + bx1 * sx, y + by1 * sy)
            if caja[0] < 3 or caja[1] < 3 or caja[2] > ancho - 3 or caja[3] > alto - 3:
                continue
            ccx = (caja[0] + caja[2]) / 2
            oclusores = [
                (x + ecx * sx, y + ecy * sy, erx * sx, ery * sy),  # cabeza
                (ccx, caja[3] + 0.15 * fh, 0.2 * fw, 0.3 * fh),  # cuello
                (ccx, caja[3] + 0.95 * fh, 1.15 * fw, 0.85 * fh),  # torso
            ]
            valido, toca = True, False
            for c in colocadas:
                if _dentro_elipses(c["malla_nucleo"], oclusores).any():
                    valido = False
                    break
                frac = float(_dentro_elipses(c["malla_caja"], oclusores).mean())
                if frac > 0.3:
                    valido = False
                    break
                toca |= frac >= 0.03
            if not valido or (idx in forzar and intento < 250 and not toca):
                continue
            colocado = (x, y, caja, oclusores)
            break
        if colocado is None:
            continue
        x, y, caja, oclusores = colocado
        piel = _color_piel(r.img, r.nucleo)
        ropa = ROPA[int(rng.integers(len(ROPA)))]
        for (ox, oy, orx, ory), color in zip(oclusores[2:0:-1], (ropa, tuple(int(c * 0.9) for c in piel)), strict=True):
            d.ellipse((ox - orx, oy - ory, ox + orx, oy + ory), fill=color)
        (e,) = lz.pegar(recorte, x, y, ancho=nuevo_w, elementos=elems, mascara=mascara)
        alto_final = _alto_propio(e.poligono)
        e.nivel = "estres" if alto_final < ALTO_MIN_BASE_GRUPO or pose == "perfil" else "base"
        e.etiquetas.update({"clase_tamano": clase, "orden_capa": idx, "solapamiento_forzado": idx in forzar})
        ex0, ey0, ex1, ey1 = caja_envolvente(e.poligono)
        nx0, ny0, nx1, ny1 = caja_envolvente(e.nucleo)
        colocadas.append(
            {
                "elemento": e,
                "caja": (ex0, ey0, ex1, ey1),
                "malla_caja": _malla((ex0, ey0, ex1, ey1)),
                "malla_nucleo": _malla((nx0, ny0, nx1, ny1), holgura=min(3.0, 0.05 * (ex1 - ex0))),
                "oclusores": oclusores,
            }
        )
    # fracción de cada caja tapada por las personas pegadas después
    for i, c in enumerate(colocadas):
        posteriores = [o for c2 in colocadas[i + 1 :] for o in c2["oclusores"]]
        frac = float(_dentro_elipses(c["malla_caja"], posteriores).mean()) if posteriores else 0.0
        c["elemento"].etiquetas["fraccion_ocluida"] = round(frac, 3)
    return lz


def _grupos(ctx: Contexto, rng: np.random.Generator) -> list[Archivo]:
    salida = []
    for i in (1, 2, 3):
        lz = _grupo(ctx, rng)
        desc = "Grupo compuesto: varias caras de 20 a 200 px pegadas sobre una sala, algunas solapadas."
        if i == 3:
            lz = lz.rotar(90)
            desc += " Imagen girada 90°."
        lz = Lienzo(ruido(lz.img, rng, 2.5), lz.elementos)
        salida.append(_guardar(ctx, rng, lz, f"grupo_compuesto_{i:02d}", desc, {"angulo": 90 if i == 3 else 0}))
    return salida


# ---------------------------------------------------------------------------
# 5. Escenas reales
# ---------------------------------------------------------------------------


def _escenas_reales(ctx: Contexto, rng: np.random.Generator) -> list[Archivo]:
    salida = []
    ids_vistos: set[str] = set()
    for esc in ctx.rostros.escenas():
        img = esc.img.convert("RGB")
        w, h = img.size
        poses = list(esc.etiquetas.get("poses", []))  # opcional: una pose por cara, si la fuente la anota
        elementos = []
        for k, (caja, nucleo) in enumerate(esc.rostros):
            caja_c = [[min(max(float(px), 0.0), w), min(max(float(py), 0.0), h)] for px, py in caja]
            nucleo_c = (
                None
                if nucleo is None
                else [[min(max(float(px), 0.0), w), min(max(float(py), 0.0), h)] for px, py in nucleo]
            )
            elementos.append(
                Elemento(
                    tipo="rostro",
                    pagina=0,
                    poligono=caja_c,
                    nucleo=nucleo_c,
                    valor=None,
                    nivel="base",
                    capa="raster",
                    etiquetas={
                        "pose": poses[k] if k < len(poses) else "sin_anotar",
                        "fuente_rostro": esc.fuente,
                        "escena_id": esc.id,
                        "angulo": 0,
                        "sin_nucleo": nucleo is None,
                    },
                )
            )
        lz = Lienzo(img, elementos)
        if max(w, h) > LADO_MAX_ESCENA:
            lz = lz.escalar(LADO_MAX_ESCENA / max(w, h))
        for e in lz.elementos:
            e.nivel = "base" if _alto_cara(e) >= ALTO_MIN_BASE_ESCENA else "estres"
        nombre_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(esc.id))
        while nombre_id in ids_vistos:
            nombre_id += "_b"
        ids_vistos.add(nombre_id)
        salida.append(
            _guardar(
                ctx,
                rng,
                lz,
                f"escena_real_{nombre_id}",
                f"Foto real con {len(elementos)} caras anotadas (fuente {esc.fuente}).",
                {"escena_id": esc.id, "fuente_escena": esc.fuente},
            )
        )
    return salida


# ---------------------------------------------------------------------------
# 6. Afiche impreso fotografiado
# ---------------------------------------------------------------------------


def _escribir_partes(
    lz: Lienzo,
    x: float,
    y: float,
    partes: list[tuple[str, str, str, dict[str, Any]]],
    **kw: Any,
) -> list[Elemento]:
    """Como ``Lienzo.linea`` pero con nivel y etiquetas propias por segmento: ``(texto, tipo, nivel, etiquetas)``."""
    salida = []
    nombre_fuente, tam = kw.get("nombre_fuente", "sans"), kw.get("tam", 24)
    for texto, tipo, nivel, etq in partes:
        limpio = texto.strip()
        if limpio:
            sangria = lz.ancho_texto(texto[: len(texto) - len(texto.lstrip())], nombre_fuente, tam)
            e = lz.escribir(x + sangria, y, limpio, tipo=tipo, nivel=nivel, etiquetas=etq, **kw)
            if e is not None:
                salida.append(e)
        x += lz.ancho_texto(texto, nombre_fuente, tam)
    return salida


def _nombre(p: Persona) -> tuple[str, str, str, dict[str, Any]]:
    nivel = "base" if p.en_lista else "fuera_de_alcance"
    return (p.nombre_completo, "nombre", nivel, {"en_lista": p.en_lista, "variante": "completo"})


def _afiche(ctx: Contexto, rng: np.random.Generator, f: Ficticios) -> Archivo:
    W, H = 900, 1272
    lz = Lienzo.nuevo(W, H, (250, 248, 242))
    d = ImageDraw.Draw(lz.img)
    color = COLORES_AFICHE[int(rng.integers(len(COLORES_AFICHE)))]
    d.rectangle((0, 0, W, 175), fill=color)
    blanco = (255, 255, 255)
    lz.escribir(50, 88, "TALLER COMUNITARIO", nombre_fuente="sans_negrita", tam=58, color=blanco)
    lz.escribir(50, 145, "Participación ciudadana en el plan comunal", tam=30, color=blanco)
    _escribir_partes(
        lz, 50, 232, [("Fecha: ", "texto", "base", {}), (f.fecha(), "texto", "base", {"senuelo": "fecha"})], tam=30
    )
    # foto de la persona que coordina
    (r,) = ctx.rostros.tomar("frente", 1)
    ancho_foto = int(min(420, 520 * r.img.width / r.img.height))
    alto_foto = int(round(r.img.height * ancho_foto / r.img.width))
    fx, fy = (W - ancho_foto) // 2, 272
    d.rectangle(
        (fx - 10, fy - 10, fx + ancho_foto + 10, fy + alto_foto + 10), fill=(255, 255, 255), outline=(170, 170, 170)
    )
    lz.pegar(
        r.img.convert("RGB"),
        fx,
        fy,
        ancho=ancho_foto,
        elementos=ProveedorRostros.elementos(r, "base", {"angulo": 0, "origen": "afiche"}),
    )
    y = fy + alto_foto + 62
    coordina = f.persona(en_lista=True)
    apoyo = f.persona(en_lista=False)
    _escribir_partes(lz, 50, y, [("Coordina: ", "texto", "base", {}), _nombre(coordina)], tam=30)
    y += 46
    _escribir_partes(lz, 50, y, [("Apoyo: ", "texto", "base", {}), _nombre(apoyo)], tam=26)
    y += 52
    for frase in FRASES_AFICHE:
        lz.escribir(50, y, frase, tam=24, color=(60, 60, 60))
        y += 38
    # línea de contacto: correo + teléfono
    fmt_correo = ["punto", "inicial", "guion_bajo", "con_anio"][int(rng.integers(4))]
    formatos_movil = [x for x in FORMATOS_POR_CLASE["movil"] if FORMATOS_TELEFONO[x] == "base"]
    fmt_tel = formatos_movil[int(rng.integers(len(formatos_movil)))]
    correo, telefono = coordina.correo(fmt_correo), coordina.telefono.formatear(fmt_tel)
    partes = [
        ("Contacto: ", "texto", "base", {}),
        (correo, "correo", FORMATOS_CORREO[fmt_correo], {"formato": fmt_correo}),
        ("  ·  Fono: ", "texto", "base", {}),
        (telefono, "telefono", FORMATOS_TELEFONO[fmt_tel], {"formato": fmt_tel, "clase": "movil"}),
    ]
    tam = 28
    while tam > 18 and lz.ancho_texto("".join(p[0] for p in partes), "sans_negrita", tam) > W - 100:
        tam -= 1
    y += 26
    _escribir_partes(lz, 50, y, partes, nombre_fuente="sans_negrita", tam=tam, color=(20, 20, 20))
    d.rectangle((0, H - 70, W, H), fill=color)
    lz.escribir(50, H - 26, "Programa de Fomento Comunitario", tam=26, color=blanco)

    # foto del afiche sobre una mesa, en perspectiva
    fondo = textura_mesa(1600, 1200, rng)
    j = rng.uniform(-28, 28, (4, 2))
    destino = [
        [430 + j[0, 0], 70 + j[0, 1]],
        [1170 + j[1, 0], 115 + j[1, 1]],
        [1225 + j[2, 0], 1125 + j[2, 1]],
        [385 + j[3, 0], 1085 + j[3, 1]],
    ]
    foto = lz.perspectiva(destino, fondo)
    img = iluminacion(foto.img, rng, 0.3)
    img = ruido(img, rng, 3.0)
    foto = Lienzo(img, foto.elementos)
    return _guardar(
        ctx,
        rng,
        foto,
        "afiche_impreso",
        "Afiche impreso con foto, nombres y línea de contacto, fotografiado en perspectiva sobre una mesa.",
        {"perspectiva": True},
    )


# ---------------------------------------------------------------------------
# 7. Variantes: blanco y negro, baja resolución, oclusión, espejado
# ---------------------------------------------------------------------------


def _baja_resolucion(r: Rostro, rng: np.random.Generator) -> Lienzo:
    lz = _retrato(r, "estres", {"degradacion": "baja_resolucion"})
    x0, y0, x1, y1 = caja_envolvente(r.caja)
    chico = lz.escalar(30.0 / (y1 - y0))
    w, h = chico.ancho, chico.alto
    img = chico.img.resize((w * 4, h * 4), Image.Resampling.BILINEAR)
    img = desenfocar(img, 1.2)
    elementos = transformar_elementos(chico.elementos, matriz_escala(4.0))
    for e in elementos:
        e.etiquetas["alto_origen_px"] = round(_alto_propio(chico.elementos[0].poligono), 1)
    return Lienzo(img, elementos)


def _oclusion(r: Rostro, rng: np.random.Generator) -> Lienzo:
    lz = _ajustar_lado(_retrato(r, "estres", {"oclusion": ["lentes_oscuros", "mano"]}), rng)
    e = lz.elementos[0]
    d = ImageDraw.Draw(lz.img)
    nx0, ny0, nx1, ny1 = caja_envolvente(e.nucleo)
    nw, nh = nx1 - nx0, ny1 - ny0
    s = lz.ancho / r.img.width
    if "ojos" in r.etiquetas and len(r.etiquetas["ojos"]) == 2:
        ojos = [(float(px) * s, float(py) * s) for px, py in r.etiquetas["ojos"]]
    else:
        ojos = [(nx0 + 0.25 * nw, ny0 + 0.14 * nh), (nx0 + 0.75 * nw, ny0 + 0.14 * nh)]
    rx, ry = 0.21 * nw, 0.11 * nh
    for ox, oy in ojos:
        d.ellipse((ox - rx, oy - ry, ox + rx, oy + ry), fill=(18, 18, 22))
        d.ellipse((ox - rx * 0.55, oy - ry * 0.6, ox - rx * 0.1, oy - ry * 0.25), fill=(70, 70, 80))
    (a, b) = ojos
    d.line((a[0] + rx * 0.9, a[1], b[0] - rx * 0.9, b[1]), fill=(18, 18, 22), width=max(2, int(ry * 0.3)))
    # mano: palma y dedos sobre la mejilla y parte de la boca
    cx0, cy0, cx1, cy1 = caja_envolvente(e.poligono)
    cw, ch = cx1 - cx0, cy1 - cy0
    piel = _color_piel(lz.img, e.nucleo)
    tono = tuple(min(255, int(c * 0.92 + 12)) for c in piel)
    sombra = tuple(int(c * 0.75) for c in piel)
    mano = Image.new("L", lz.img.size, 0)
    dm = ImageDraw.Draw(mano)
    px, py = cx0 + 0.86 * cw, cy0 + 0.9 * ch
    dm.ellipse((px - 0.22 * cw, py - 0.17 * ch, px + 0.22 * cw, py + 0.17 * ch), fill=255)
    for k in range(4):
        ang = math.radians(215 + 14 * k + rng.uniform(-3, 3))
        largo = 0.42 * ch * (1.0 - 0.12 * abs(k - 1.5))
        fx, fy = px + math.cos(ang) * largo * 0.5, py + math.sin(ang) * largo
        dm.line((px, py - 0.05 * ch, fx, fy), fill=255, width=max(3, int(0.09 * cw)))
        dm.ellipse((fx - 0.045 * cw, fy - 0.045 * cw, fx + 0.045 * cw, fy + 0.045 * cw), fill=255)
    mano = mano.filter(ImageFilter.GaussianBlur(1.2))
    capa = Image.new("RGB", lz.img.size, tono)
    borde = Image.new("RGB", lz.img.size, sombra)
    lz.img.paste(borde, (0, 0), mano.filter(ImageFilter.MaxFilter(5)))
    lz.img.paste(capa, (0, 0), mano)
    return lz


def _variantes(ctx: Contexto, rng: np.random.Generator) -> list[Archivo]:
    salida = []
    (r,) = ctx.rostros.tomar("frente", 1)
    lz = _ajustar_lado(_retrato(r, "base", {"color": "gris"}), rng)
    lz = Lienzo(lz.img.convert("L").convert("RGB"), lz.elementos)
    salida.append(
        _guardar(ctx, rng, lz, "blanco_y_negro", "Retrato en escala de grises (JPEG de un canal).", gris=True)
    )
    (r,) = ctx.rostros.tomar("frente", 1)
    salida.append(
        _guardar(
            ctx,
            rng,
            _baja_resolucion(r, rng),
            "baja_resolucion",
            "Cara reducida a ~30 px de alto y ampliada 4x (borrosa).",
        )
    )
    (r,) = ctx.rostros.tomar("frente", 1)
    salida.append(
        _guardar(
            ctx, rng, _oclusion(r, rng), "oclusion", "Retrato con lentes oscuros y una mano tapando parte de la cara."
        )
    )
    (r,) = ctx.rostros.tomar("tres_cuartos", 1)
    lz = _ajustar_lado(_retrato(r, "base"), rng).espejar()
    salida.append(_guardar(ctx, rng, lz, "espejado", "Retrato en tres cuartos espejado horizontalmente."))
    return salida
