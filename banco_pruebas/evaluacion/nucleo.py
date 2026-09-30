"""Evaluación archivo por archivo: aplica las comprobaciones C, T, B, P, I, O y V de ``docs/metricas.md``.

Cada elemento del manifiesto recibe un estado:

- ``censurado``: pasaron todas las comprobaciones que le corresponden según su capa.
- ``fuga``: falló al menos una (se listan en ``fallas``).
- ``no_procesado``: el sistema no entregó salida para el archivo (rechazo o resultado faltante).
  Cuenta como no detectado en el recall, pero no como fuga (no hay archivo que publicar).
- ``neutro``: texto sin dato personal (``tipo == "texto"``) fuera del modo "todo el texto";
  solo se usa para las métricas de sobrecensura y conservación del texto.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from banco_pruebas.esquema import Archivo, Censura, Elemento, InformeCensura, Manifiesto, Metadato, ResultadoArchivo
from banco_pruebas.evaluacion import cobertura as cob
from banco_pruebas.evaluacion.buscar import Aguja, Pajar, agujas, agujas_metadato, textos_de_bytes
from banco_pruebas.evaluacion.imagen import SalidaImagen, estructura_tiff, formato_por_firma
from banco_pruebas.evaluacion.pdf import ESCALA, ImagenColocada, SalidaPdf
from banco_pruebas.visualizar import a_visual

# Comprobaciones que corresponden a cada capa (tabla de la sección 3).
COMPROBACIONES_PDF = {
    "texto": ("C", "T", "B", "P"),
    "oculto": ("T", "B"),
    "vector": ("C", "P", "V"),
    "raster": ("C", "P", "I", "O"),
}
COMPROBACIONES_IMAGEN = {
    "texto": ("C", "P"),
    "oculto": ("B",),
    "vector": ("C", "P"),
    "raster": ("C", "P"),
}
# Estructuras no textuales: su sola presencia es fuga aunque el metadato tenga canario.
SIEMPRE_ESTRUCTURAL = ("exif.gps", "exif.miniatura", "pdf.revision_anterior")
TOLERANCIA_PAGINA_PT = 1.0


@dataclass
class EvalElemento:
    id: str
    archivo: str
    categoria: str
    formato: str
    pagina: int
    tipo: str
    nivel: str
    capa: str
    valor: str | None
    etiquetas: dict[str, Any]
    estado: str = "no_procesado"  # censurado | fuga | no_procesado | neutro
    objetivo: bool = True  # cuenta en recall y fugas (el texto neutro solo en modo "todo el texto")
    comprobaciones: dict[str, bool | None] = field(default_factory=dict)  # True = pasa
    fallas: list[str] = field(default_factory=list)
    detectado: bool = False
    cobertura: float | None = None
    cobertura_nucleo: float | None = None
    alto_px: float | None = None  # alto del dato en píxeles de la salida (PDF: a 144 ppp)
    extraible: bool | None = None  # texto neutro de la capa de texto que sigue extraíble
    detalle: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetadato:
    id: str
    archivo: str
    categoria: str
    donde: str
    valor: str | None
    estado: str = "no_procesado"  # eliminado | fuga | no_procesado
    motivo: list[str] = field(default_factory=list)


@dataclass
class EvalArchivo:
    id: str
    ruta: str
    formato: str
    categoria: str
    esperado: str
    # procesado | rechazado | sin_resultado | sin_salida | salida_ilegible (procesables)
    # rechazo_correcto | rechazo_codigo_distinto | no_rechazado | sin_resultado (error esperado)
    estado: str
    error: str | None = None
    tiempo_s: float | None = None
    n_paginas: int = 0
    formato_salida: str | None = None
    modo_todo_el_texto: bool = False
    elementos: list[EvalElemento] = field(default_factory=list)
    metadatos: list[EvalMetadato] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    geometria_distinta: list[str] = field(default_factory=list)
    n_censuras: int = 0
    censuras_sin_dato: int = 0
    etiquetas: dict[str, Any] = field(default_factory=dict)
    fallo_evaluador: str | None = None


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _alto_poligono(poligono: Sequence[Sequence[float]], rostro: bool = False) -> float:
    """Alto de un cuadrilátero, invariante a la rotación.

    Texto: el par de lados opuestos más corto (una línea es más ancha que alta).
    Rostro: el más largo (la caja de una cara es más alta que ancha).
    """
    p = np.asarray(poligono, dtype=np.float64)
    if len(p) == 4:
        lados = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
        pares = ((lados[0] + lados[2]) / 2, (lados[1] + lados[3]) / 2)
        return float(max(pares) if rostro else min(pares))
    return float(p[:, 1].max() - p[:, 1].min())


def _eval_elemento(a: Archivo, e: Elemento) -> EvalElemento:
    return EvalElemento(
        id=e.id,
        archivo=a.id,
        categoria=a.categoria,
        formato=a.formato,
        pagina=e.pagina,
        tipo=e.tipo,
        nivel=e.nivel,
        capa=e.capa,
        valor=e.valor,
        etiquetas=dict(e.etiquetas),
        objetivo=e.tipo != "texto",
        alto_px=None
        if e.poligono is None
        else round(_alto_poligono(e.poligono, e.tipo == "rostro") * (ESCALA if a.formato == "pdf" else 1.0), 2),
    )


def _eval_metadato(a: Archivo, m: Metadato) -> EvalMetadato:
    return EvalMetadato(id=m.id, archivo=a.id, categoria=a.categoria, donde=m.donde, valor=m.valor)


def _codigo_esperado(esperado: str) -> str:
    return esperado.split(":", 1)[1] if ":" in esperado else esperado


def _codigo(error: str | None) -> str:
    return (error or "").split(":", 1)[0].strip().lower()


def _encontrados(pajar: Pajar, agujas_: list[Aguja], origenes: Sequence[str] | None = None) -> list[str]:
    return sorted({f"{origen}:{a.parte}" for a, origen in pajar.buscar(agujas_, origenes)})


# ---------------------------------------------------------------------------
# Estructuras de metadatos
# ---------------------------------------------------------------------------


def estructura_presente(donde: str, est: dict[str, Any]) -> bool | None:
    """¿Sigue presente la estructura que nombra ``donde``? ``None`` si el lugar no se reconoce."""
    d = donde.lower()
    # el contenedor como prefijo ("png.exif.gps", "webp.xmp") no cambia la estructura buscada
    for contenedor in ("png.", "jpeg.", "jpg.", "webp."):
        if d.startswith(tuple(contenedor + x for x in ("exif", "xmp", "iptc", "com"))):
            d = d[len(contenedor) :]
            donde = donde[len(contenedor) :]
            break
    if d.startswith("exif.gps"):
        return bool(est.get("exif.gps"))
    if d.startswith("exif.miniatura"):
        return bool(est.get("exif.miniatura"))
    if d.startswith("exif"):
        return bool(est.get("exif"))
    if d.startswith("xmp"):
        return bool(est.get("xmp"))
    if d.startswith("iptc"):
        return bool(est.get("iptc"))
    if d.startswith(("com", "jpeg.com")):
        return bool(est.get("com"))
    if d.startswith("png.texto"):
        claves = [c.lower() for c in est.get("png.texto", [])]
        partes = donde.split(".", 2)
        return (partes[2].lower() in claves) if len(partes) == 3 else bool(claves)
    if d.startswith("tiff"):
        clave = estructura_tiff(donde)
        return bool(est.get(clave)) if clave else bool(est.get("tiff.etiquetas"))
    if d.startswith("pdf.info"):
        partes = d.split(".", 2)
        return bool(est.get(f"pdf.info.{partes[2]}")) if len(partes) == 3 else bool(est.get("pdf.info"))
    for clave in ("pdf.xmp", "pdf.anotacion", "pdf.adjunto", "pdf.ocg", "pdf.formulario", "pdf.marcador"):
        if d.startswith(clave):
            return bool(est.get(clave))
    if d.startswith("pdf.javascript"):
        return bool(est.get("pdf.javascript"))
    if d.startswith("pdf.revision"):
        return bool(est.get("pdf.revision_anterior"))
    return None


# ---------------------------------------------------------------------------
# Evaluación de un archivo
# ---------------------------------------------------------------------------


def evaluar_archivo(
    a: Archivo, res: ResultadoArchivo | None, dir_salida: Path, dir_entrada: Path | None = None
) -> EvalArchivo:
    """Evalúa un archivo del manifiesto contra su resultado (``None`` si el informe no lo trae).

    ``dir_entrada`` es la raíz del conjunto (donde están los originales). La comprobación O la
    necesita para saber qué imágenes de la entrada contienen datos, y P la usa para reconocer
    píxeles que quedaron iguales al original aunque parezcan uniformes (una cara oscura o un
    texto de bajo contraste). Sin ella esas partes no se evalúan y queda una advertencia.
    """
    ev = EvalArchivo(
        id=a.id,
        ruta=a.ruta,
        formato=a.formato,
        categoria=a.categoria,
        esperado=a.esperado,
        estado="sin_resultado",
        n_paginas=len(a.paginas),
        etiquetas=dict(a.etiquetas),
    )
    if res is not None:
        ev.error, ev.tiempo_s, ev.modo_todo_el_texto = res.error, res.tiempo_s, bool(res.modo_todo_el_texto)
        ev.n_censuras = sum(1 for c in res.censuras if c.estado != "descartado")

    # -- archivos que deben rechazarse ---------------------------------------
    if a.esperado != "procesar":
        if res is None:
            ev.estado = "sin_resultado"
        elif not res.error:
            ev.estado = "no_rechazado"
        elif _codigo(res.error) == _codigo_esperado(a.esperado).lower():
            ev.estado = "rechazo_correcto"
        else:
            ev.estado = "rechazo_codigo_distinto"
        return ev

    ev.elementos = [_eval_elemento(a, e) for e in a.elementos]
    ev.metadatos = [_eval_metadato(a, m) for m in a.metadatos_sensibles]
    for ee in ev.elementos:
        if not ee.objetivo:
            ee.estado = "neutro"
        if ev.modo_todo_el_texto and ee.tipo == "texto":
            ee.objetivo = True
            ee.estado = "no_procesado"
    if res is None:
        return ev
    if res.error:
        ev.estado = "rechazado"
        return ev
    ruta = dir_salida / res.salida if res.salida else None
    if ruta is None or not ruta.is_file():
        ev.estado = "sin_salida"
        return ev
    try:
        _evaluar_salida(a, res, ruta, ev, dir_entrada)
    except Exception:  # noqa: BLE001 - un fallo del evaluador no puede esconder una fuga
        ev.fallo_evaluador = traceback.format_exc(limit=6)
        ev.estado = "salida_ilegible"
        for ee in ev.elementos:
            if ee.objetivo:
                ee.estado, ee.fallas = "fuga", ["error_evaluador"]
        for em in ev.metadatos:
            em.estado, em.motivo = "fuga", ["error_evaluador"]
    return ev


def _evaluar_salida(a: Archivo, res: ResultadoArchivo, ruta: Path, ev: EvalArchivo, dir_entrada: Path | None) -> None:
    with ruta.open("rb") as f:
        datos_inicio = f.read(1024)
    formato_salida = formato_por_firma(datos_inicio + b"\x00" * 16)
    ev.formato_salida = formato_salida
    es_pdf = a.formato == "pdf"
    salida: SalidaPdf | SalidaImagen | None = None
    ilegible = None
    try:
        if formato_salida == "pdf":
            salida = SalidaPdf(ruta)
        else:
            salida = SalidaImagen(ruta)
    except Exception as ex:  # noqa: BLE001
        ilegible = f"{type(ex).__name__}: {ex}"
    ev.estado = "procesado"
    zonas: dict[int, list[list[list[float]]]] = {}
    for c in res.censuras:
        if c.estado != "descartado":
            zonas.setdefault(c.pagina, []).append(c.poligono)
    muestras = cob.MUESTRAS_PDF if es_pdf else cob.MUESTRAS_IMAGEN

    # cobertura (independiente del archivo de salida)
    for e, ee in zip(a.elementos, ev.elementos, strict=True):
        if e.poligono is None or ee.capa == "oculto":
            continue
        z = zonas.get(e.pagina, [])
        ee.cobertura = round(cob.cobertura(e.poligono, z, muestras), 4)
        if e.tipo == "rostro" and e.nucleo:
            ee.cobertura_nucleo = round(cob.cobertura(e.nucleo, z, muestras), 4)
    ev.censuras_sin_dato = _censuras_sin_dato(a, res.censuras, ev)

    if salida is None:
        ev.estado = "salida_ilegible"
        ev.advertencias.append(f"no se pudo abrir la salida: {ilegible}")
        for ee in ev.elementos:
            if ee.objetivo:
                ee.estado, ee.fallas = "fuga", ["salida_ilegible"]
        _evaluar_metadatos_bytes_crudos(a, ruta.read_bytes(), ev)
        return
    entrada = Original(a, ev, dir_entrada)
    try:
        paginas_ok = _geometria(a, salida, es_pdf, formato_salida, ev)
        for e, ee in zip(a.elementos, ev.elementos, strict=True):
            if ee.tipo == "texto" and not ee.objetivo:
                _evaluar_neutro(e, ee, salida, es_pdf)
                continue
            _evaluar_elemento(a, e, ee, salida, es_pdf, paginas_ok, entrada)
        _evaluar_metadatos(a, ev, salida)
        ev.advertencias.extend(salida.advertencias_lectura)
    finally:
        salida.cerrar()
        entrada.cerrar()


class Original:
    """El archivo de entrada, abierto solo si alguna comprobación lo pide (O y el control de P)."""

    def __init__(self, a: Archivo, ev: EvalArchivo, dir_entrada: Path | None) -> None:
        self.ruta = None if dir_entrada is None else Path(dir_entrada) / a.ruta
        self.es_pdf = a.formato == "pdf"
        self.ev = ev
        self._lector: SalidaPdf | SalidaImagen | None = None
        self._intentado = False

    def lector(self) -> SalidaPdf | SalidaImagen | None:
        if not self._intentado:
            self._intentado = True
            if self.ruta is None or not self.ruta.is_file():
                self.ev.advertencias.append(
                    "no se encontró el archivo de entrada: no se evaluaron la comprobación O ni el control "
                    "de píxeles sin cambios de P"
                )
            else:
                try:
                    self._lector = SalidaPdf(self.ruta) if self.es_pdf else SalidaImagen(self.ruta)
                except Exception as ex:  # noqa: BLE001
                    self.ev.advertencias.append(
                        f"no se pudo abrir la entrada ({ex}): O y el control de P no se evaluaron"
                    )
        return self._lector

    def pdf(self) -> SalidaPdf | None:
        lector = self.lector()
        return lector if isinstance(lector, SalidaPdf) else None

    def pagina_visible(self, indice: int) -> np.ndarray | None:
        """Página ``indice`` de la entrada tal como se ve (PDF a 144 ppp; imagen con EXIF aplicado)."""
        lector = self.lector()
        try:
            if isinstance(lector, SalidaPdf):
                return lector.render(indice) if indice < len(lector.doc) else None
            if isinstance(lector, SalidaImagen):
                return lector.cuadros[indice] if indice < len(lector.cuadros) else None
        except Exception:  # noqa: BLE001 - sin entrada legible no hay control
            return None
        return None

    def cerrar(self) -> None:
        if self._lector is not None:
            self._lector.cerrar()


def _geometria(
    a: Archivo, salida: SalidaPdf | SalidaImagen, es_pdf: bool, formato_salida: str, ev: EvalArchivo
) -> list[bool]:
    """Para cada página del manifiesto, si la salida tiene la misma geometría."""
    n = len(a.paginas)
    if es_pdf != isinstance(salida, SalidaPdf):
        ev.geometria_distinta.append(f"la entrada es {a.formato} y la salida es {formato_salida}")
        return [False] * n
    geo = salida.geometria()
    if len(geo) != n:
        ev.geometria_distinta.append(f"la salida tiene {len(geo)} páginas y se esperaban {n}")
        return [False] * n
    ok = []
    for pag, g in zip(a.paginas, geo, strict=True):
        if es_pdf:
            # Se compara la página tal como se ve: una salida que dibuja la página girada en una
            # hoja sin /Rotate (ancho y alto intercambiados) tiene la misma geometría visible.
            w, h, rot = g
            vis_salida = (w, h) if rot % 180 == 0 else (h, w)
            vis_entrada = (pag.ancho, pag.alto) if pag.rotacion % 180 == 0 else (pag.alto, pag.ancho)
            igual = (
                abs(vis_salida[0] - vis_entrada[0]) <= TOLERANCIA_PAGINA_PT
                and abs(vis_salida[1] - vis_entrada[1]) <= TOLERANCIA_PAGINA_PT
            )
            if not igual:
                ev.geometria_distinta.append(
                    f"página {pag.indice}: {w:.1f}x{h:.1f} rot {rot} (se esperaba {pag.ancho:.1f}x{pag.alto:.1f} rot {pag.rotacion})"
                )
        else:
            w, h = g
            igual = abs(w - pag.ancho) < 0.5 and abs(h - pag.alto) < 0.5
            if not igual:
                ev.geometria_distinta.append(f"imagen {pag.indice}: {w}x{h} (se esperaba {pag.ancho:g}x{pag.alto:g})")
        ok.append(igual)
    return ok


def _a_salida(a: Archivo, indice: int, poligono: list[list[float]], pdf: SalidaPdf) -> list[list[float]]:
    """Polígono del manifiesto (página de entrada sin rotar) -> espacio sin rotar de la página de salida.

    Es la identidad si la página de salida tiene la misma rotación y tamaño; si no (misma página
    visible guardada con otro /Rotate), se pasa por la página visible.
    """
    pag = a.paginas[indice]
    w, h, rot = pdf.geometria()[indice]
    if (
        rot % 360 == pag.rotacion % 360
        and abs(w - pag.ancho) <= TOLERANCIA_PAGINA_PT
        and abs(h - pag.alto) <= TOLERANCIA_PAGINA_PT
    ):
        return poligono
    m = pdf.doc[indice].derotation_matrix
    salida = []
    for x, y in a_visual(a, indice, poligono):
        q = pymupdf.Point(x / ESCALA, y / ESCALA) * m
        salida.append([q.x, q.y])
    return salida


def _zona_pixeles(e: Elemento) -> tuple[list[list[float]] | None, float | None]:
    """Polígono y umbral para P/I: en rostros, el núcleo (o la cara completa con el umbral de cobertura)."""
    if e.tipo == "rostro":
        if e.nucleo:
            return e.nucleo, None
        return e.poligono, cob.UMBRAL_ROSTRO_SIN_NUCLEO
    return e.poligono, None


def _detectado(e: Elemento, ee: EvalElemento) -> bool:
    if ee.cobertura is None:
        return False
    if e.tipo == "rostro":
        if e.nucleo:
            return (ee.cobertura_nucleo or 0) >= cob.UMBRAL_ROSTRO_NUCLEO and ee.cobertura >= cob.UMBRAL_ROSTRO_CAJA
        return ee.cobertura >= cob.UMBRAL_ROSTRO_SIN_NUCLEO
    return ee.cobertura >= cob.UMBRAL_COBERTURA


def _evaluar_elemento(
    a: Archivo,
    e: Elemento,
    ee: EvalElemento,
    salida: SalidaPdf | SalidaImagen,
    es_pdf: bool,
    paginas_ok: list[bool],
    entrada: Original | None = None,
) -> None:
    capa = e.capa if e.capa in COMPROBACIONES_PDF else "raster"
    codigos = (COMPROBACIONES_PDF if es_pdf else COMPROBACIONES_IMAGEN)[capa]
    geometria_ok = 0 <= e.pagina < len(paginas_ok) and paginas_ok[e.pagina]
    pdf = salida if isinstance(salida, SalidaPdf) else None
    img = salida if isinstance(salida, SalidaImagen) else None
    ag = agujas(e.tipo, e.valor)
    comp: dict[str, bool | None] = {}
    for codigo in codigos:
        if codigo == "C":
            comp["C"] = _detectado(e, ee)
        elif codigo == "T":
            if not ag or pdf is None:
                comp["T"] = None if pdf is not None else False
                continue
            hallados = _encontrados(pdf.pajar_texto, ag)
            comp["T"] = not hallados
            if hallados:
                ee.detalle["T"] = hallados
        elif codigo == "B":
            if not ag:
                comp["B"] = None
                continue
            hallados = _encontrados(salida.pajar_bytes, ag)
            comp["B"] = not hallados
            if hallados:
                ee.detalle["B"] = hallados
        elif not geometria_ok:
            comp[codigo] = False
        elif codigo == "P":
            poligono, umbral = _zona_pixeles(e)
            if poligono is None:
                comp["P"] = None
                continue
            if pdf is not None:
                lienzo = pdf.render(e.pagina)
                pts = [list(p) for p in a_visual(a, e.pagina, poligono)]
            else:
                assert img is not None
                lienzo = img.cuadros[e.pagina]
                pts = poligono
            ok, f = cob.es_uniforme(lienzo, pts, umbral)
            comp["P"] = True if ok is None else ok
            if comp["P"] and entrada is not None:
                # zona "uniforme" que es la misma del original: dato de bajo contraste sin censurar
                original = entrada.pagina_visible(e.pagina)
                r = None if original is None else cob.correlacion(lienzo, original, pts)
                if r is not None:
                    ee.detalle["correlacion_original"] = round(r, 4)
                    if r >= cob.UMBRAL_SIN_CAMBIOS:
                        comp["P"] = False
                        ee.detalle["P"] = "píxeles iguales a los del original"
            ee.detalle["uniformidad"] = None if f is None else round(f, 4)
        elif codigo == "I":
            comp["I"] = _comprobar_imagenes(a, e, ee, pdf) if pdf is not None else None
        elif codigo == "O":
            original = entrada.pdf() if entrada is not None and pdf is not None else None
            comp["O"] = _comprobar_originales(e, ee, pdf, original) if pdf is not None and original else None
        elif codigo == "V":
            comp["V"] = _comprobar_trazos(a, e, ee, pdf) if pdf is not None else None
    ee.comprobaciones = comp
    ee.fallas = [c for c, v in comp.items() if v is False]
    if not geometria_ok:
        ee.fallas.insert(0, "geometria_distinta")
    if e.capa == "oculto":
        ee.detectado = all(comp.get(c) is not False for c in codigos)
    else:
        ee.detectado = bool(comp.get("C"))
    ee.estado = "fuga" if ee.fallas else "censurado"


def _imagenes_bajo(
    poligono: list[list[float]], imagenes: list[ImagenColocada]
) -> list[tuple[ImagenColocada, np.ndarray | None, float]]:
    """Imágenes colocadas que cubren al menos el 10 % del polígono: (imagen, píxeles bajo el
    polígono o ``None`` si la imagen no se pudo decodificar, área del polígono en px de la imagen)."""
    caja = cob.caja(poligono)
    area_pagina = max(cob.area(poligono), 1e-9)
    salida: list[tuple[ImagenColocada, np.ndarray | None, float]] = []
    for im in imagenes:
        if not cob.cajas_se_tocan(caja, im.bbox):
            continue
        if im.pixeles is None:
            salida.append((im, None, 0.0))
            continue
        en_imagen = im.a_imagen(poligono)
        escala = abs(im.transform.a * im.transform.d - im.transform.b * im.transform.c) / max(1, im.ancho * im.alto)
        pix, area_img = cob.pixeles_poligono(im.pixeles, en_imagen)
        # fracción del polígono que cae sobre esta imagen (área en la página)
        if len(pix) == 0 or len(pix) * escala / area_pagina < 0.10:
            continue
        salida.append((im, pix, area_img))
    return salida


def _comprobar_imagenes(a: Archivo, e: Elemento, ee: EvalElemento, pdf: SalidaPdf) -> bool:
    """I: los píxeles de cada imagen bajo el polígono deben ser uniformes (no basta taparlos)."""
    poligono, umbral = _zona_pixeles(e)
    if poligono is None:
        return True
    poligono = _a_salida(a, e.pagina, poligono, pdf)
    resultado = True
    revisadas = []
    for im, pix, area_img in _imagenes_bajo(poligono, pdf.imagenes(e.pagina)):
        if pix is None:
            revisadas.append({"xref": im.xref, "ilegible": True})
            resultado = False
            continue
        f = cob.uniformidad(pix)
        u = (
            umbral
            if umbral is not None
            else (cob.UMBRAL_UNIFORME_PEQUENO if area_img < cob.AREA_PEQUENA else cob.UMBRAL_UNIFORME)
        )
        revisadas.append({"xref": im.xref, "uniformidad": round(f, 4)})
        if f < u:
            resultado = False
    if revisadas:
        ee.detalle["imagenes"] = revisadas
    return resultado


def _comprobar_originales(e: Elemento, ee: EvalElemento, pdf: SalidaPdf, entrada: SalidaPdf) -> bool:
    """O: ninguna imagen de la entrada que contenga el dato puede seguir intacta en la salida,
    esté dibujada o no (objetos huérfanos, revisiones anteriores)."""
    if e.poligono is None or not 0 <= e.pagina < len(entrada.doc):
        return True
    intactas = []
    for im, _, _ in _imagenes_bajo(e.poligono, entrada.imagenes(e.pagina)):
        if im.pixeles is None or im.xref <= 0:
            continue  # imagen en línea: vive en el flujo de contenido (lo revisan I y P)
        donde = pdf.contiene_imagen(im.pixeles, entrada.flujo_crudo(im.xref))
        if donde:
            intactas.append({"xref_entrada": im.xref, "en_salida": donde})
    if intactas:
        ee.detalle["imagenes_originales"] = intactas
    return not intactas


def _comprobar_trazos(a: Archivo, e: Elemento, ee: EvalElemento, pdf: SalidaPdf) -> bool:
    """V: no deben quedar curvas de Bézier (contornos de glifos) dentro del polígono."""
    if e.poligono is None:
        return True
    puntos = pdf.curvas(e.pagina)
    if len(puntos) == 0:
        return True
    poligono = _a_salida(a, e.pagina, e.poligono, pdf)
    x0, y0, x1, y1 = cob.caja(poligono)
    cerca = puntos[(puntos[:, 0] >= x0) & (puntos[:, 0] <= x1) & (puntos[:, 1] >= y0) & (puntos[:, 1] <= y1)]
    n = int(cob.puntos_en_poligono(cerca, poligono).sum())
    if n:
        ee.detalle["puntos_curva"] = n
    return n == 0


def _evaluar_neutro(e: Elemento, ee: EvalElemento, salida: SalidaPdf | SalidaImagen, es_pdf: bool) -> None:
    """Texto neutro: solo interesa si sigue extraíble (conservación) y si quedó tapado (sobrecensura)."""
    if es_pdf and isinstance(salida, SalidaPdf) and e.capa == "texto" and e.valor:
        ag = [x for x in agujas("texto", e.valor) if x.parte == "valor"]
        ee.extraible = bool(salida.pajar_texto.buscar(ag))


def _censuras_sin_dato(a: Archivo, censuras: list[Censura], ev: EvalArchivo) -> int:
    """Zonas censuradas activas que no tocan ningún dato personal (sobrecensura)."""
    objetivos: dict[int, list[list[list[float]]]] = {}
    for e, ee in zip(a.elementos, ev.elementos, strict=True):
        if ee.objetivo and e.poligono is not None:
            objetivos.setdefault(e.pagina, []).append(e.poligono)
    n = 0
    for c in censuras:
        if c.estado == "descartado":
            continue
        if not any(cob.se_intersectan(c.poligono, p) for p in objetivos.get(c.pagina, [])):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Metadatos
# ---------------------------------------------------------------------------


def _evaluar_metadatos(a: Archivo, ev: EvalArchivo, salida: SalidaPdf | SalidaImagen) -> None:
    textos, est, adv = salida.metadatos[:3]
    ev.advertencias.extend(adv)
    if not ev.metadatos:
        return
    pajar_meta = Pajar()
    for donde, texto in textos:
        pajar_meta.agregar(".".join(donde.split(".")[:2]), texto)
    pajares = [pajar_meta, salida.pajar_bytes]
    if isinstance(salida, SalidaPdf):
        pajares.append(salida.pajar_texto)
    for em, m in zip(ev.metadatos, a.metadatos_sensibles, strict=True):
        motivos = []
        ag = agujas_metadato(m.valor, m.etiquetas.get("tipo"))
        for pajar in pajares:
            motivos.extend(f"canario en {h}" for h in _encontrados(pajar, ag))
        presente = estructura_presente(m.donde, est)
        estructural = m.valor is None or any(m.donde.lower().startswith(s) for s in SIEMPRE_ESTRUCTURAL)
        if estructural:
            if presente is None:
                motivos.append(f"lugar no reconocido por el evaluador: {m.donde}")
            elif presente:
                motivos.append(f"estructura presente: {m.donde}")
        em.motivo = motivos
        em.estado = "fuga" if motivos else "eliminado"


def _evaluar_metadatos_bytes_crudos(a: Archivo, datos: bytes, ev: EvalArchivo) -> None:
    """Salida ilegible: los metadatos solo se pueden buscar en los bytes (sin estructura, se asume fuga)."""
    pajar = Pajar()
    pajar.agregar("bytes", textos_de_bytes(datos), binario=True)
    for em, m in zip(ev.metadatos, a.metadatos_sensibles, strict=True):
        hallados = _encontrados(pajar, agujas_metadato(m.valor, m.etiquetas.get("tipo")))
        em.motivo = [f"canario en {h}" for h in hallados] or ["salida_ilegible"]
        em.estado = "fuga"


# ---------------------------------------------------------------------------
# Todo el conjunto
# ---------------------------------------------------------------------------


def _trabajo(args: tuple[Archivo, ResultadoArchivo | None, Path, Path]) -> EvalArchivo:
    a, res, dir_salida, dir_entrada = args
    return evaluar_archivo(a, res, dir_salida, dir_entrada)


def _clave_ruta(ruta: str) -> str:
    """Ruta relativa comparable: separador "/" y sin "./" inicial."""
    r = ruta.replace("\\", "/")
    while r.startswith("./"):
        r = r[2:]
    return r


def emparejar(manifiesto: Manifiesto, informe: InformeCensura) -> tuple[dict[str, ResultadoArchivo], list[str]]:
    """Resultado de cada archivo del manifiesto (por ruta de entrada) y resultados sin archivo."""
    por_ruta: dict[str, ResultadoArchivo] = {}
    for r in informe.resultados:
        por_ruta[_clave_ruta(r.entrada)] = r
    rutas = {_clave_ruta(a.ruta) for a in manifiesto.archivos}
    huerfanos = sorted(set(por_ruta) - rutas)
    return por_ruta, huerfanos


def evaluar_archivos(
    manifiesto: Manifiesto, informe: InformeCensura, dir_salida: Path, procesos: int | None = None
) -> tuple[list[EvalArchivo], list[str]]:
    """Evalúa todos los archivos (en paralelo si ``procesos`` > 1)."""
    por_ruta, huerfanos = emparejar(manifiesto, informe)
    trabajos = [
        (a, por_ruta.get(_clave_ruta(a.ruta)), Path(dir_salida), Path(manifiesto.raiz)) for a in manifiesto.archivos
    ]
    if procesos is None:
        procesos = min(8, max(1, (os.cpu_count() or 2) // 2)) if len(trabajos) >= 8 else 1
    if procesos <= 1:
        resultados = [_trabajo(t) for t in trabajos]
    else:
        try:
            with ProcessPoolExecutor(max_workers=procesos) as ex:
                resultados = list(ex.map(_trabajo, trabajos, chunksize=1))
        except (BrokenProcessPool, OSError):
            # el sistema no dejó crear o mantener los procesos: se evalúa en este mismo proceso
            resultados = [_trabajo(t) for t in trabajos]
    return resultados, huerfanos


def evaluar(
    manifiesto: Manifiesto, informe: InformeCensura, dir_salida: Path, procesos: int | None = None
) -> dict[str, Any]:
    """Evalúa un informe de censura y devuelve el resultado completo (ver ``agregar.agregar``)."""
    from banco_pruebas.evaluacion.agregar import agregar

    archivos, huerfanos = evaluar_archivos(manifiesto, informe, dir_salida, procesos)
    return agregar(manifiesto, informe, archivos, huerfanos)
