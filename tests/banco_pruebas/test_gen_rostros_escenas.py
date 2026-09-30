"""Invariantes del generador de rostros: archivos, geometría de la verdad de terreno y niveles."""

import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Manifiesto
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.generadores import rostros_escenas
from banco_pruebas.lienzo import Lienzo, caja_envolvente, rect
from banco_pruebas.rostros import Escena, ProveedorRostros

RAIZ_REPO = Path(__file__).resolve().parents[2]

IDS_ESPERADOS = (
    [f"rost_retrato_frente_{i:02d}" for i in range(1, 5)]
    + ["rost_retrato_tres_cuartos_01", "rost_retrato_tres_cuartos_02", "rost_perfil_01", "rost_perfil_02"]
    + [f"rost_retrato_rotado_{g}" for g in (90, 180, 270, 15, 45)]
    + [f"rost_grupo_compuesto_{i:02d}" for i in (1, 2, 3)]
    + ["rost_afiche_impreso", "rost_blanco_y_negro", "rost_baja_resolucion", "rost_oclusion", "rost_espejado"]
)


def _contexto(raiz: Path) -> Contexto:
    return Contexto(
        raiz=raiz,
        semilla=33,
        fict=Ficticios(33),
        rostros=ProveedorRostros(RAIZ_REPO / "datos_prueba" / "cache" / "rostros", 33, permitir_descarga=False),
    )


@pytest.fixture(scope="module")
def generado(tmp_path_factory):
    ctx = _contexto(tmp_path_factory.mktemp("rostros") / "generado")
    inicio = time.perf_counter()
    archivos = rostros_escenas.generar(ctx)
    return ctx, archivos, time.perf_counter() - inicio


def _abrir(ctx: Contexto, archivo) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(ctx.raiz / archivo.ruta))


def _mascara(poligono, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(poligono, np.float64) - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def _por_id(archivos):
    return {a.id: a for a in archivos}


def test_ids_rutas_y_manifiesto(generado):
    ctx, archivos, _ = generado
    ids = [a.id for a in archivos]
    assert len(ids) == len(set(ids))
    assert set(IDS_ESPERADOS) <= set(ids)
    # sin escenas reales (solo rostros dibujados) no debe haber archivos de escena
    if not ctx.rostros.escenas():
        assert not any(i.startswith("rost_escena_real_") for i in ids)
    man = Manifiesto(raiz=str(ctx.raiz), semilla=33)
    for a in archivos:
        assert a.id.startswith("rost_")
        assert a.categoria == "rostros"
        assert a.formato == "jpg" and a.ruta.startswith("rostros/") and a.ruta.endswith(".jpg")
        man.agregar(a)
    assert man.validar() == []


def test_archivos_abren_con_el_tamano_declarado(generado):
    ctx, archivos, _ = generado
    for a in archivos:
        img = _abrir(ctx, a)
        assert img.size == (a.paginas[0].ancho, a.paginas[0].alto), a.id
        assert not img.info.get("exif"), f"{a.id}: no debe llevar EXIF"
    bn = _por_id(archivos)["rost_blanco_y_negro"]
    assert Image.open(ctx.raiz / bn.ruta).mode == "L"


def test_retratos_sueltos_entre_400_y_800(generado):
    ctx, archivos, _ = generado
    for a in archivos:
        if a.id.startswith(("rost_retrato_", "rost_perfil_", "rost_blanco", "rost_oclusion", "rost_espejado")):
            assert 400 <= max(a.paginas[0].ancho, a.paginas[0].alto) <= 800, a.id


def test_poligonos_dentro_de_la_imagen(generado):
    _, archivos, _ = generado
    for a in archivos:
        w, h = a.paginas[0].ancho, a.paginas[0].alto
        for e in a.elementos:
            assert e.capa == "raster" and e.pagina == 0
            for poli in (e.poligono, e.nucleo):
                if poli is None:
                    continue
                x0, y0, x1, y1 = caja_envolvente(poli)
                assert x0 >= -0.5 and y0 >= -0.5 and x1 <= w + 0.5 and y1 <= h + 0.5, (a.id, e.tipo, poli)


def test_rostros_bien_formados_y_nucleo_con_contenido(generado):
    ctx, archivos, _ = generado
    for a in archivos:
        arr = np.asarray(_abrir(ctx, a).convert("L")).astype(np.float32)
        for e in a.elementos:
            if e.tipo != "rostro":
                continue
            assert e.valor is None and e.capa == "raster"
            assert len(e.poligono) == 4
            for clave in ("pose", "fuente_rostro", "alto_px", "angulo"):
                assert clave in e.etiquetas, (a.id, clave)
            if a.id.startswith("rost_escena_real_"):
                # las escenas reales pueden venir sin núcleo anotado
                assert (e.nucleo is None) == e.etiquetas["sin_nucleo"], a.id
            else:
                assert e.nucleo is not None, a.id
            if e.nucleo is not None:
                pix = arr[_mascara(e.nucleo, arr.shape)]
                assert pix.size > 0 and pix.std() > 5, (a.id, e.etiquetas)
            pix_caja = arr[_mascara(e.poligono, arr.shape)]
            assert pix_caja.std() > 5, a.id


def test_texto_con_tinta(generado):
    ctx, archivos, _ = generado
    for a in archivos:
        arr = np.asarray(_abrir(ctx, a).convert("L")).astype(np.float32)
        for e in a.elementos:
            if e.tipo == "rostro":
                continue
            pix = arr[_mascara(e.poligono, arr.shape)]
            assert pix.size > 0 and pix.std() > 12, (a.id, e.valor)


def test_conteos_por_archivo(generado):
    _, archivos, _ = generado
    por_id = _por_id(archivos)
    for i in IDS_ESPERADOS:
        n_rostros = sum(e.tipo == "rostro" for e in por_id[i].elementos)
        if i.startswith("rost_grupo_compuesto_"):
            assert 5 <= n_rostros <= 12, i
        else:
            assert n_rostros == 1, i
    afiche = Counter(e.tipo for e in por_id["rost_afiche_impreso"].elementos)
    assert afiche["correo"] == 1 and afiche["telefono"] == 1 and afiche["nombre"] == 2
    assert afiche["texto"] >= 8
    elementos = por_id["rost_afiche_impreso"].elementos
    assert any(e.etiquetas.get("senuelo") == "fecha" for e in elementos)
    niveles_nombre = sorted(e.nivel for e in elementos if e.tipo == "nombre")
    assert niveles_nombre == ["base", "fuera_de_alcance"]
    for e in elementos:
        if e.tipo == "correo":
            assert "formato" in e.etiquetas
        if e.tipo == "telefono":
            assert {"formato", "clase"} <= set(e.etiquetas)


def test_niveles(generado):
    _, archivos, _ = generado
    por_id = _por_id(archivos)
    estres = {"rost_perfil_01", "rost_perfil_02", "rost_baja_resolucion", "rost_oclusion"}
    for i in IDS_ESPERADOS:
        if i.startswith("rost_grupo_") or i == "rost_afiche_impreso":
            continue
        (e,) = [e for e in por_id[i].elementos if e.tipo == "rostro"]
        assert e.nivel == ("estres" if i in estres else "base"), i
    for i in (1, 2, 3):
        caras = [e for e in por_id[f"rost_grupo_compuesto_{i:02d}"].elementos if e.tipo == "rostro"]
        altos = [e.etiquetas["alto_px"] for e in caras]
        assert min(altos) < 40 and max(altos) >= 80
        assert any(e.etiquetas["fraccion_ocluida"] > 0 for e in caras), "debe haber caras solapadas"
        for e in caras:
            assert 18 <= e.etiquetas["alto_px"] <= 205
            esperado = "estres" if e.etiquetas["alto_px"] < 40 or e.etiquetas["pose"] == "perfil" else "base"
            assert e.nivel == esperado
            assert e.etiquetas["fraccion_ocluida"] <= 0.3


def test_angulos(generado):
    _, archivos, _ = generado
    por_id = _por_id(archivos)
    for g in (90, 180, 270, 15, 45):
        (e,) = por_id[f"rost_retrato_rotado_{g}"].elementos
        assert e.etiquetas["angulo"] == pytest.approx(g)
    assert all(e.etiquetas["angulo"] == 90 for e in por_id["rost_grupo_compuesto_03"].elementos)
    (e,) = por_id["rost_espejado"].elementos
    assert e.etiquetas.get("espejo") == "horizontal"


def test_rotacion_90_coincide_con_pixeles(generado):
    """El giro exacto de 90° debe llevar el recorte de la cara a la misma posición que el GT."""
    ctx, archivos, _ = generado
    por_id = _por_id(archivos)
    girada = np.asarray(_abrir(ctx, por_id["rost_retrato_rotado_90"]).convert("L")).astype(np.float32)
    recta = np.asarray(_abrir(ctx, por_id["rost_retrato_rotado_180"]).convert("L")).astype(np.float32)
    # 180 -> 90: girar 270 más (ambas vienen del mismo rostro con distinta escala)
    atras = np.rot90(recta, k=-1)
    (e90,) = por_id["rost_retrato_rotado_90"].elementos
    (e180,) = por_id["rost_retrato_rotado_180"].elementos
    x0, y0, x1, y1 = (int(round(v)) for v in caja_envolvente(e90.nucleo))
    h180, _ = recta.shape
    # nucleo del 180 llevado al marco girado 270 (horario 90): (x, y) -> (h - y, x)
    pts = np.asarray(e180.nucleo)
    xr0, yr0 = (h180 - pts[:, 1]).min(), pts[:, 0].min()
    xr1, yr1 = (h180 - pts[:, 1]).max(), pts[:, 0].max()
    a = girada[y0:y1, x0:x1]
    b = atras[int(round(yr0)) : int(round(yr1)), int(round(xr0)) : int(round(xr1))]
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(a - b).mean() < 12


def test_escenas_reales(ctx):
    """Con escenas disponibles: se reescalan a <= 1600 px y las cajas se transforman igual."""
    caras = ctx.rostros.tomar("frente", 3)
    lz = Lienzo.nuevo(2400, 1500, (180, 175, 160))
    rostros = []
    for r, (x, ancho) in zip(caras, [(100, 600), (900, 300), (1600, 30)], strict=True):
        (e,) = lz.pegar(r.img, x, 300, ancho=ancho, elementos=ProveedorRostros.elementos(r))
        rostros.append((e.poligono, e.nucleo))
    rostros[1] = (rostros[1][0], None)  # una cara sin núcleo anotado
    rostros.append((rect(2380, 20, 2420, 80), None))  # caja que se sale del borde (se recorta)
    ctx.rostros._escenas = [Escena(id="demo/01", img=lz.img, rostros=rostros, fuente="sintetico_dibujado")]
    archivos = rostros_escenas._escenas_reales(ctx, ctx.rng("prueba"))
    (a,) = archivos
    assert a.id == "rost_escena_real_demo_01" and a.ruta == "rostros/escena_real_demo_01.jpg"
    assert (a.paginas[0].ancho, a.paginas[0].alto) == (1600, 1000)
    img = _abrir(ctx, a)
    assert img.size == (1600, 1000)
    assert len(a.elementos) == 4
    f = 1600 / 2400
    for e, (caja, nucleo) in zip(a.elementos, rostros, strict=True):
        assert e.tipo == "rostro" and e.valor is None and e.capa == "raster"
        cx0, cy0, cx1, cy1 = caja_envolvente(caja)
        ex0, ey0, ex1, ey1 = caja_envolvente(e.poligono)
        assert ex0 == pytest.approx(cx0 * f, abs=0.01) and ey1 == pytest.approx(cy1 * f, abs=0.01)
        assert ex1 <= 1600 + 1e-6
        assert (e.nucleo is None) == (nucleo is None)
        esperado = "base" if (ey1 - ey0) >= 24 else "estres"
        assert e.nivel == esperado
    assert [e.nivel for e in a.elementos][:3] == ["base", "base", "estres"]


def test_tiempo(generado):
    _, _, segundos = generado
    # El tiempo depende de la carga del equipo: se avisa, no se falla (los tiempos se reportan aparte).
    if segundos > 90:
        warnings.warn(f"generación lenta: {segundos:.1f} s", stacklevel=1)


def test_determinista(generado, tmp_path):
    """Dos corridas con la misma semilla producen los mismos bytes y la misma verdad de terreno."""
    ctx, archivos, _ = generado
    ctx2 = _contexto(tmp_path / "otra")
    archivos2 = rostros_escenas.generar(ctx2)
    assert [a.id for a in archivos] == [a.id for a in archivos2]
    for a, b in zip(archivos, archivos2, strict=True):
        assert (ctx.raiz / a.ruta).read_bytes() == (ctx2.raiz / b.ruta).read_bytes(), a.id
        assert [(e.tipo, e.valor, e.nivel, e.poligono, e.nucleo) for e in a.elementos] == [
            (e.tipo, e.valor, e.nivel, e.poligono, e.nucleo) for e in b.elementos
        ], a.id


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _rectificar(img: np.ndarray, poligono, lado: int = 64) -> np.ndarray:
    """Lleva el cuadrilátero (coordenadas continuas) a un cuadrado de ``lado`` px."""
    destino = np.float32([[0, 0], [lado, 0], [lado, lado], [0, lado]])
    m = cv2.getPerspectiveTransform(np.float32(poligono) - 0.5, destino)
    return cv2.warpPerspective(img, m, (lado, lado), flags=cv2.INTER_AREA)


def test_nucleo_coincide_con_la_foto_de_origen(generado):
    """Verificación independiente de la geometría: el núcleo rectificado desde la salida debe
    parecerse al núcleo de la foto de origen (giros, escalas, espejo y perspectiva incluidos),
    y bastante menos si el polígono se corre un 10 % de su ancho."""
    ctx, archivos, _ = generado
    origen = {r.id: r for lista in ctx.rostros._rostros.values() for r in lista}
    revisados = 0
    for a in archivos:
        if a.id in ("rost_oclusion", "rost_baja_resolucion"):
            continue
        arr = np.asarray(_abrir(ctx, a).convert("L")).astype(np.float32)
        for e in a.elementos:
            if e.tipo != "rostro" or e.etiquetas.get("rostro_id") not in origen or e.etiquetas["alto_px"] < 40:
                continue
            if e.etiquetas.get("fraccion_ocluida", 0) > 0:
                continue
            r = origen[e.etiquetas["rostro_id"]]
            plantilla = _rectificar(np.asarray(r.img.convert("L")).astype(np.float32), r.nucleo)
            salida = _rectificar(arr, e.nucleo)
            p = np.asarray(e.nucleo, np.float64)
            corrido = p + 0.1 * (p[1] - p[0])
            assert _ncc(plantilla, salida) > 0.85, (a.id, e.etiquetas)
            assert _ncc(plantilla, salida) > _ncc(plantilla, _rectificar(arr, corrido)) + 0.1, (a.id, e.etiquetas)
            revisados += 1
    assert revisados >= 20
