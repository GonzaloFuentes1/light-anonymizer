"""Agregación de los resultados por elemento en las métricas de ``docs/metricas.md`` (secciones 4 y 5)."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Any

import numpy as np

from banco_pruebas.esquema import NIVELES, TIPOS, TIPOS_CERO_FUGAS, InformeCensura, Manifiesto
from banco_pruebas.evaluacion.nucleo import EvalArchivo, EvalElemento

TIPOS_TEXTO = ("rut", "correo", "telefono", "url", "nombre", "direccion")
TRAMOS_ROSTRO = (
    (0, 24, "<24"),
    (24, 48, "24-48"),
    (48, 96, "48-96"),
    (96, 192, "96-192"),
    (192, float("inf"), ">=192"),
)
TRAMOS_LETRA = (
    (0, 10, "<10"),
    (10, 14, "10-14"),
    (14, 20, "14-20"),
    (20, 28, "20-28"),
    (28, 40, "28-40"),
    (40, float("inf"), ">=40"),
)
DEGRADACIONES = ("ruido", "sal_pimienta", "desenfoque", "desenfocar", "jpeg", "comprimir_jpeg", "perspectiva",
                 "iluminacion", "textura_papel", "textura_mesa", "inclinacion", "escaneo")  # fmt: skip
ESTADOS_NO_PROCESADO = ("rechazado", "sin_resultado", "sin_salida")


def _tramo(valor: float | None, tramos: Iterable[tuple[float, float, str]]) -> str:
    if valor is None:
        return "?"
    for lo, hi, nombre in tramos:
        if lo <= valor < hi:
            return nombre
    return "?"


def _conteo(elementos: list[EvalElemento]) -> dict[str, Any]:
    n = len(elementos)
    detectados = sum(e.detectado for e in elementos)
    fugas = sum(e.estado == "fuga" for e in elementos)
    no_proc = sum(e.estado == "no_procesado" for e in elementos)
    evaluados = n - no_proc
    return {
        "n": n,
        "detectados": detectados,
        "recall": round(detectados / n, 4) if n else None,
        "fugas": fugas,
        "pct_fugas": round(100 * fugas / evaluados, 2) if evaluados else None,
        "no_procesados": no_proc,
    }


def _agrupar(elementos: list[EvalElemento], clave: Callable[[EvalElemento], Any]) -> dict[str, dict[str, Any]]:
    grupos: dict[str, list[EvalElemento]] = defaultdict(list)
    for e in elementos:
        grupos[str(clave(e))].append(e)
    return {k: _conteo(v) for k, v in sorted(grupos.items(), key=lambda kv: _orden(kv[0]))}


def _orden(k: str) -> tuple[int, float, str]:
    for i, tramos in enumerate((TRAMOS_ROSTRO, TRAMOS_LETRA)):
        for j, (_, _, nombre) in enumerate(tramos):
            if k == nombre:
                return (i, j, k)
    try:
        return (5, float(k), k)
    except ValueError:
        return (9, 0, k)


def angulo(e: EvalElemento) -> str:
    if e.etiquetas.get("espejo") or e.etiquetas.get("espejado"):
        return "espejado"
    a = e.etiquetas.get("angulo")
    if a is None:
        return "0"
    try:
        return str(int(round(float(a))) % 360)
    except (TypeError, ValueError):
        return str(a)


def degradacion(e: EvalElemento) -> str:
    d = e.etiquetas.get("degradacion") or e.etiquetas.get("degradaciones")
    if isinstance(d, (list, tuple)):
        return "+".join(str(x) for x in d) or "ninguna"
    if d:
        return str(d)
    presentes = [k for k in DEGRADACIONES if e.etiquetas.get(k)]
    return "+".join(presentes) if presentes else "ninguna"


def alto_letra(e: EvalElemento) -> str:
    tam = e.etiquetas.get("tam_px")
    return _tramo(float(tam) if isinstance(tam, (int, float)) else e.alto_px, TRAMOS_LETRA)


def _estadisticas(valores: list[float]) -> dict[str, Any]:
    if not valores:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    v = np.asarray(valores, dtype=np.float64)
    return {
        "n": len(v),
        "p50": round(float(np.percentile(v, 50)), 4),
        "p95": round(float(np.percentile(v, 95)), 4),
        "max": round(float(v.max()), 4),
    }


def _tabla_tipo_nivel(elementos: list[EvalElemento]) -> dict[str, dict[str, dict[str, Any]]]:
    tabla: dict[str, dict[str, dict[str, Any]]] = {}
    for tipo in TIPOS:
        for nivel in NIVELES:
            sub = [e for e in elementos if e.tipo == tipo and e.nivel == nivel]
            if sub:
                tabla.setdefault(tipo, {})[nivel] = _conteo(sub)
    return tabla


def _fuga_detalle(e: EvalElemento, rutas: dict[str, str]) -> dict[str, Any]:
    return {
        "elemento": e.id,
        "archivo": e.archivo,
        "ruta": rutas.get(e.archivo, ""),
        "categoria": e.categoria,
        "pagina": e.pagina,
        "tipo": e.tipo,
        "nivel": e.nivel,
        "capa": e.capa,
        "valor": e.valor,
        "fallas": e.fallas,
        "cobertura": e.cobertura,
        "cobertura_nucleo": e.cobertura_nucleo,
        "critica": e.tipo in TIPOS_CERO_FUGAS and e.nivel == "base",
        "detalle": e.detalle,
    }


def agregar(
    manifiesto: Manifiesto, informe: InformeCensura, archivos: list[EvalArchivo], huerfanos: list[str]
) -> dict[str, Any]:
    """Métricas, desgloses y veredicto a partir de la evaluación de cada archivo."""
    rutas = {a.id: a.ruta for a in archivos}
    procesables = [a for a in archivos if a.esperado == "procesar"]
    con_error = [a for a in archivos if a.esperado != "procesar"]
    elementos = [e for a in procesables for e in a.elementos]
    objetivos = [e for e in elementos if e.objetivo and e.tipo != "texto"]
    modo_texto = [e for e in elementos if e.objetivo and e.tipo == "texto"]
    neutros = [e for e in elementos if not e.objetivo]
    procesados = [a for a in procesables if a.estado in ("procesado", "salida_ilegible")]
    no_procesados = [a for a in procesables if a.estado in ESTADOS_NO_PROCESADO]

    # -- fugas --------------------------------------------------------------
    fugas = [e for e in objetivos + modo_texto if e.estado == "fuga"]
    fugas.sort(
        key=lambda e: (not (e.tipo in TIPOS_CERO_FUGAS and e.nivel == "base"), NIVELES.index(e.nivel), e.archivo, e.id)
    )
    criticas = [e for e in objetivos if e.estado == "fuga" and e.tipo in TIPOS_CERO_FUGAS and e.nivel == "base"]

    # -- por categoría ------------------------------------------------------
    por_categoria: dict[str, Any] = {}
    for cat in sorted({a.categoria for a in procesables}):
        sub = [e for e in objetivos if e.categoria == cat]
        c = _conteo(sub)
        c["archivos"] = sum(1 for a in procesables if a.categoria == cat)
        c["fugas_criticas"] = sum(1 for e in criticas if e.categoria == cat)
        c["archivos_no_procesados"] = sum(1 for a in no_procesados if a.categoria == cat)
        por_categoria[cat] = c

    # -- desgloses ------------------------------------------------------------
    rostros = [e for e in objetivos if e.tipo == "rostro"]
    textos = [e for e in objetivos if e.tipo in TIPOS_TEXTO]
    textos_raster = [e for e in textos if e.capa == "raster"]
    ruts = [e for e in objetivos if e.tipo == "rut"]
    desgloses = {
        "rostros": {
            "tamano_px": _agrupar(rostros, lambda e: _tramo(e.alto_px, TRAMOS_ROSTRO)),
            "pose": _agrupar(rostros, lambda e: e.etiquetas.get("pose", "?")),
            "angulo": _agrupar(rostros, angulo),
            "origen": _agrupar(rostros, lambda e: e.etiquetas.get("origen", e.categoria)),
            "nivel": _agrupar(rostros, lambda e: e.nivel),
        },
        "texto_en_imagenes": {
            "angulo": _agrupar(textos_raster, angulo),
            "alto_letra_px": _agrupar(textos_raster, alto_letra),
            "fuente": _agrupar(textos_raster, lambda e: e.etiquetas.get("fuente", "?")),
            "degradacion": _agrupar(textos_raster, degradacion),
            "categoria": _agrupar(textos_raster, lambda e: e.categoria),
        },
        "texto": {
            "capa": _agrupar(textos, lambda e: e.capa),
            "formato": _agrupar(textos, lambda e: f"{e.tipo}:{e.etiquetas.get('formato', '?')}"),
            "variante_nombre": _agrupar(
                [e for e in textos if e.tipo == "nombre"], lambda e: e.etiquetas.get("variante", "?")
            ),
        },
        "rut_dv_valido": _agrupar(ruts, lambda e: e.etiquetas.get("dv_valido", "?")),
    }

    # -- metadatos ------------------------------------------------------------
    metas = [m for a in procesables for m in a.metadatos]
    fugas_meta = [m for m in metas if m.estado == "fuga"]
    advertencias = {a.id: a.advertencias for a in procesados if a.advertencias}

    # -- errores esperados ------------------------------------------------------
    errores = [
        {"archivo": a.id, "ruta": a.ruta, "esperado": a.esperado, "estado": a.estado, "error": a.error}
        for a in con_error
    ]
    errores_ok = sum(1 for a in con_error if a.estado == "rechazo_correcto")

    # -- sobrecensura y conservación ------------------------------------------------
    neutros_cob = [e for e in neutros if e.cobertura is not None]
    senuelos = [e for e in neutros_cob if e.etiquetas.get("senuelo")]
    n_zonas = sum(a.n_censuras for a in procesados)
    zonas_sin_dato = sum(a.censuras_sin_dato for a in procesados)
    neutros_texto = [e for e in neutros if e.extraible is not None]

    def _frac(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    sobrecensura = {
        "texto_neutro": {
            "n": len(neutros_cob),
            "cubiertos_50": sum(1 for e in neutros_cob if (e.cobertura or 0) >= 0.5),
            "fraccion": _frac(sum(1 for e in neutros_cob if (e.cobertura or 0) >= 0.5), len(neutros_cob)),
        },
        "senuelos": {
            tipo: {
                "n": len(sub),
                "cubiertos_50": sum(1 for e in sub if (e.cobertura or 0) >= 0.5),
                "fraccion": _frac(sum(1 for e in sub if (e.cobertura or 0) >= 0.5), len(sub)),
            }
            for tipo in sorted({str(e.etiquetas["senuelo"]) for e in senuelos})
            for sub in [[e for e in senuelos if str(e.etiquetas["senuelo"]) == tipo]]
        },
        "zonas": {"n": n_zonas, "sin_dato_personal": zonas_sin_dato, "fraccion": _frac(zonas_sin_dato, n_zonas)},
    }
    conservacion = {
        "n": len(neutros_texto),
        "extraibles": sum(1 for e in neutros_texto if e.extraible),
        "fraccion": _frac(sum(1 for e in neutros_texto if e.extraible), len(neutros_texto)),
    }

    # -- tiempos ------------------------------------------------------------------
    con_tiempo = [a for a in procesados if a.tiempo_s is not None]
    tiempos = {
        "por_archivo": _estadisticas([a.tiempo_s for a in con_tiempo]),
        "por_pagina_pdf": _estadisticas([a.tiempo_s / max(1, a.n_paginas) for a in con_tiempo if a.formato == "pdf"]),
        "por_imagen": _estadisticas([a.tiempo_s for a in con_tiempo if a.formato != "pdf"]),
        "sin_tiempo": [a.id for a in procesados if a.tiempo_s is None],
        "equipo": informe.detalles.get("equipo"),
        "memoria_max_mb": informe.detalles.get("memoria_max_mb"),
    }

    # -- modo "todo el texto" --------------------------------------------------------
    modo = {
        "archivos": [a.id for a in procesables if a.modo_todo_el_texto],
        **_conteo(modo_texto),
    }

    # -- veredicto (sección 5) ---------------------------------------------------------
    criterios = []
    motivos = []
    ok1 = not criticas and not no_procesados
    detalle1 = f"{len(criticas)} fugas de rut/correo/telefono en nivel base"
    if no_procesados:
        detalle1 += f"; {len(no_procesados)} archivos procesables sin salida (no se puede verificar que no filtren)"
    criterios.append(
        {"n": 1, "criterio": "Cero fugas de rut, correo y teléfono en nivel base", "cumple": ok1, "detalle": detalle1}
    )
    if criticas:
        cats = sorted({e.categoria for e in criticas})
        motivos.append(f"{len(criticas)} fugas críticas (rut/correo/teléfono, nivel base) en: {', '.join(cats)}")
    if no_procesados:
        motivos.append(
            f"{len(no_procesados)} archivos procesables rechazados o sin salida: "
            + ", ".join(a.id for a in no_procesados[:10])
            + (" ..." if len(no_procesados) > 10 else "")
        )
    ok2 = not fugas_meta
    criterios.append(
        {
            "n": 2,
            "criterio": "Cero fugas de metadatos sensibles",
            "cumple": ok2,
            "detalle": f"{len(fugas_meta)} de {len(metas)}",
        }
    )
    if fugas_meta:
        motivos.append(f"{len(fugas_meta)} fugas de metadatos sensibles")
    ok3 = errores_ok == len(con_error)
    criterios.append(
        {
            "n": 3,
            "criterio": "Archivos con error esperado rechazados con el código correcto",
            "cumple": ok3,
            "detalle": f"{errores_ok} de {len(con_error)}",
        }
    )
    if not ok3:
        malos = [a.id for a in con_error if a.estado != "rechazo_correcto"]
        motivos.append(f"{len(malos)} archivos con error esperado mal manejados: {', '.join(malos[:10])}")
    criterios.append(
        {
            "n": 4,
            "criterio": "Informe de recall de rostros y OCR con desgloses",
            "cumple": True,
            "detalle": f"{len(rostros)} rostros y {len(textos_raster)} textos en imágenes (informativo, sin umbral)",
        }
    )
    ok5 = bool(procesados) and not tiempos["sin_tiempo"]
    criterios.append(
        {
            "n": 5,
            "criterio": "Tiempos por imagen y por página",
            "cumple": ok5,
            "detalle": f"{len(con_tiempo)} de {len(procesados)} archivos con tiempo",
        }
    )
    if not ok5:
        motivos.append(f"faltan tiempos en {len(tiempos['sin_tiempo'])} archivos procesados")
    aprobado = ok1 and ok2 and ok3 and ok5

    fallos_evaluador = {a.id: a.fallo_evaluador for a in archivos if a.fallo_evaluador}
    resumen = {
        "archivos": len(archivos),
        "procesables": len(procesables),
        "procesados": len(procesados),
        "no_procesados": len(no_procesados),
        "con_error_esperado": len(con_error),
        "elementos_objetivo": len(objetivos),
        "detectados": sum(e.detectado for e in objetivos),
        "recall": _frac(sum(e.detectado for e in objetivos), len(objetivos)),
        "fugas": sum(1 for e in objetivos if e.estado == "fuga"),
        "fugas_criticas": len(criticas),
        "metadatos": len(metas),
        "fugas_metadatos": len(fugas_meta),
        "geometria_distinta": {a.id: a.geometria_distinta for a in procesados if a.geometria_distinta},
        "resultados_sin_archivo": huerfanos,
        "fallos_evaluador": fallos_evaluador,
    }
    return {
        "sistema": informe.sistema,
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "semilla": manifiesto.semilla,
        "veredicto": {"aprobado": aprobado, "motivos": motivos, "criterios": criterios},
        "resumen": resumen,
        "por_tipo_nivel": _tabla_tipo_nivel(objetivos),
        "por_categoria": por_categoria,
        "desgloses": desgloses,
        "metadatos": {
            "n": len(metas),
            "fugas": [asdict(m) | {"ruta": rutas.get(m.archivo, "")} for m in fugas_meta],
            "eliminados": sum(1 for m in metas if m.estado == "eliminado"),
            "no_procesados": sum(1 for m in metas if m.estado == "no_procesado"),
            "advertencias": advertencias,
        },
        "errores_esperados": {"n": len(con_error), "correctos": errores_ok, "archivos": errores},
        "procesables_no_procesados": [
            {"archivo": a.id, "ruta": a.ruta, "estado": a.estado, "error": a.error} for a in no_procesados
        ],
        "sobrecensura": sobrecensura,
        "conservacion_texto": conservacion,
        "tiempos": tiempos,
        "modo_todo_el_texto": modo,
        "fugas": [_fuga_detalle(e, rutas) for e in fugas],
        "archivos": [{k: v for k, v in asdict(a).items() if k not in ("elementos", "metadatos")} for a in archivos],
        "elementos": [asdict(e) for e in elementos],
        "elementos_metadatos": [asdict(m) for m in metas],
    }
