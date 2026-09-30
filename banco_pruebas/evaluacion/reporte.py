"""Escritura del resultado de la evaluación: ``evaluacion.json`` y ``evaluacion.md`` (en español)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from banco_pruebas.esquema import NIVELES


def _pct(x: float | None, decimales: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{decimales}f} %"


def _celda(v: Any) -> str:
    return "—" if v is None else str(v).replace("|", "\\|").replace("\n", " ")


def _tabla(encabezado: list[str], filas: list[list[Any]]) -> list[str]:
    if not filas:
        return ["_(sin datos)_", ""]
    lineas = ["| " + " | ".join(encabezado) + " |", "|" + "|".join("---" for _ in encabezado) + "|"]
    lineas += ["| " + " | ".join(_celda(c) for c in f) + " |" for f in filas]
    return lineas + [""]


def _tabla_conteos(titulo: str, grupos: dict[str, dict[str, Any]]) -> list[str]:
    filas = [
        [
            k,
            g["n"],
            g["detectados"],
            _pct(g["recall"]),
            g["fugas"],
            "—" if g["pct_fugas"] is None else f"{g['pct_fugas']:.1f} %",
        ]
        for k, g in grupos.items()
    ]
    return [f"**{titulo}**", ""] + _tabla(
        [titulo.split(" por ")[-1], "n", "detectados", "recall", "fugas", "% fugas"], filas
    )


def markdown(r: dict[str, Any]) -> str:
    v = r["veredicto"]
    s = r["resumen"]
    L: list[str] = [f"# Evaluación del anonimizador: `{r['sistema']}`", ""]
    L.append(f"Fecha: {r['fecha']} · semilla del conjunto: {r['semilla']}")
    L.append("")
    if v["aprobado"]:
        L += ["## Veredicto: **APROBADO**", ""]
    else:
        L += ["## Veredicto: **NO APROBADO**", ""]
        L += [f"- {m}" for m in v["motivos"]]
        L.append("")
    L += _tabla(
        ["#", "Criterio (sección 5 de docs/metricas.md)", "Cumple", "Detalle"],
        [[c["n"], c["criterio"], "sí" if c["cumple"] else "**NO**", c["detalle"]] for c in v["criterios"]],
    )

    L += ["## Resumen", ""]
    L += _tabla(
        ["Medida", "Valor"],
        [
            [
                "Archivos (procesables / con error esperado)",
                f"{s['archivos']} ({s['procesables']} / {s['con_error_esperado']})",
            ],
            ["Procesados / sin salida", f"{s['procesados']} / {s['no_procesados']}"],
            ["Elementos objetivo", s["elementos_objetivo"]],
            ["Recall global", _pct(s["recall"])],
            ["Fugas (críticas)", f"{s['fugas']} ({s['fugas_criticas']})"],
            ["Metadatos sensibles con fuga", f"{s['fugas_metadatos']} de {s['metadatos']}"],
        ],
    )
    if s["geometria_distinta"]:
        L += ["**Salidas con geometría distinta** (sus elementos cuentan como fuga):", ""]
        L += [f"- `{k}`: {'; '.join(x)}" for k, x in s["geometria_distinta"].items()]
        L.append("")
    if s["fallos_evaluador"]:
        L += ["**Archivos que el evaluador no pudo revisar** (se cuentan como fuga):", ""]
        L += [f"- `{k}`: {x.strip().splitlines()[-1]}" for k, x in s["fallos_evaluador"].items()]
        L.append("")
    if s["resultados_sin_archivo"]:
        L += ["Resultados del informe que no corresponden a ningún archivo del manifiesto: "]
        L += [", ".join(f"`{x}`" for x in s["resultados_sin_archivo"]), ""]

    L += ["## Recall y fugas por tipo y nivel", ""]
    filas = []
    for tipo, niveles in r["por_tipo_nivel"].items():
        for nivel in NIVELES:
            if nivel in niveles:
                g = niveles[nivel]
                filas.append(
                    [
                        tipo,
                        nivel,
                        g["n"],
                        g["detectados"],
                        _pct(g["recall"]),
                        g["fugas"],
                        "—" if g["pct_fugas"] is None else f"{g['pct_fugas']:.1f} %",
                        g["no_procesados"],
                    ]  # fmt: skip
                )
    L += _tabla(["Tipo", "Nivel", "n", "Detectados", "Recall", "Fugas", "% fugas", "No procesados"], filas)
    L += ["Recall = cobertura suficiente (≥ 95 %; rostros: núcleo ≥ 95 % y cara ≥ 80 %). En la capa oculta, "
          "detectado = eliminado (T y B). % fugas sobre los elementos de archivos procesados.", ""]  # fmt: skip

    L += ["## Por categoría de archivo", ""]
    L += _tabla(
        ["Categoría", "Archivos", "Sin salida", "n", "Recall", "Fugas", "Fugas críticas"],
        [
            [k, c["archivos"], c["archivos_no_procesados"], c["n"], _pct(c["recall"]), c["fugas"], c["fugas_criticas"]]
            for k, c in r["por_categoria"].items()
        ],
    )

    m = r["metadatos"]
    L += ["## Metadatos sensibles", ""]
    L.append(f"{m['n']} metadatos sensibles: {m['eliminados']} eliminados, {len(m['fugas'])} con fuga, "
             f"{m['no_procesados']} en archivos sin salida.")  # fmt: skip
    L.append("")
    L += _tabla(
        ["Archivo", "Dónde", "Valor", "Motivo"],
        [[f["archivo"], f["donde"], f["valor"], "; ".join(f["motivo"])] for f in m["fugas"]],
    )

    e = r["errores_esperados"]
    L += ["## Archivos con error esperado", ""]
    L.append(f"{e['correctos']} de {e['n']} rechazados con el código correcto.")
    L.append("")
    L += _tabla(
        ["Archivo", "Esperado", "Estado", "Error informado"],
        [[x["archivo"], x["esperado"], x["estado"], x["error"]] for x in e["archivos"]],
    )
    L += ["## Archivos procesables rechazados o sin salida", ""]
    L += ["Cuentan como no detectados en el recall, no como fugas.", ""]
    L += _tabla(
        ["Archivo", "Estado", "Error"],
        [[x["archivo"], x["estado"], x["error"]] for x in r["procesables_no_procesados"]],
    )

    d = r["desgloses"]
    L += ["## Desgloses de recall", "", "### Rostros", ""]
    L += _tabla_conteos("Rostros por alto de la cara (px en la salida; PDF a 144 ppp)", d["rostros"]["tamano_px"])
    L += _tabla_conteos("Rostros por pose", d["rostros"]["pose"])
    L += _tabla_conteos("Rostros por ángulo", d["rostros"]["angulo"])
    L += _tabla_conteos("Rostros por origen", d["rostros"]["origen"])
    L += _tabla_conteos("Rostros por nivel", d["rostros"]["nivel"])
    L += ["### Texto en imágenes (capa raster)", ""]
    L += _tabla_conteos("Texto por ángulo", d["texto_en_imagenes"]["angulo"])
    L += _tabla_conteos("Texto por alto de letra", d["texto_en_imagenes"]["alto_letra_px"])
    L += _tabla_conteos("Texto por fuente", d["texto_en_imagenes"]["fuente"])
    L += _tabla_conteos("Texto por degradación", d["texto_en_imagenes"]["degradacion"])
    L += _tabla_conteos("Texto por categoría", d["texto_en_imagenes"]["categoria"])
    L += ["### Texto (todas las capas)", ""]
    L += _tabla_conteos("Texto por capa", d["texto"]["capa"])
    L += _tabla_conteos("Texto por formato", d["texto"]["formato"])
    L += _tabla_conteos("Nombres por variante", d["texto"]["variante_nombre"])
    L += _tabla_conteos("RUT por dígito verificador válido", d["rut_dv_valido"])

    sc, cons = r["sobrecensura"], r["conservacion_texto"]
    L += ["## Métricas informativas", ""]
    filas = [
        [
            "Texto neutro cubierto ≥ 50 %",
            f"{sc['texto_neutro']['cubiertos_50']} de {sc['texto_neutro']['n']}",
            _pct(sc["texto_neutro"]["fraccion"]),
        ],  # fmt: skip
    ]
    for tipo, x in sc["senuelos"].items():
        filas.append([f"Señuelos '{tipo}' cubiertos ≥ 50 %", f"{x['cubiertos_50']} de {x['n']}", _pct(x["fraccion"])])
    filas.append(
        [
            "Zonas censuradas que no tocan datos",
            f"{sc['zonas']['sin_dato_personal']} de {sc['zonas']['n']}",
            _pct(sc["zonas"]["fraccion"]),
        ]  # fmt: skip
    )
    filas.append(
        ["Texto neutro del PDF que sigue extraíble", f"{cons['extraibles']} de {cons['n']}", _pct(cons["fraccion"])]
    )
    L += _tabla(["Medida", "Conteo", "Fracción"], filas)

    t = r["tiempos"]
    L += ["### Tiempos (s)", ""]
    L += _tabla(
        ["Medida", "n", "mediana", "p95", "máximo"],
        [
            [k, x["n"], x["p50"], x["p95"], x["max"]]
            for k, x in (
                ("por archivo", t["por_archivo"]),
                ("por página de PDF", t["por_pagina_pdf"]),
                ("por imagen", t["por_imagen"]),
            )
        ],  # fmt: skip
    )
    if t.get("equipo"):
        L += [f"Equipo: {t['equipo']}", ""]
    if t.get("memoria_max_mb"):
        L += [f"Memoria máxima: {t['memoria_max_mb']} MB", ""]
    if t["sin_tiempo"]:
        L += [f"Sin tiempo informado: {len(t['sin_tiempo'])} archivos.", ""]

    mo = r["modo_todo_el_texto"]
    if mo["archivos"]:
        L += ["### Modo «censurar todo el texto»", ""]
        L.append(f"Activado en {len(mo['archivos'])} archivos: {mo['n']} textos, recall {_pct(mo['recall'])}, "
                 f"{mo['fugas']} fugas.")  # fmt: skip
        L.append("")

    L += ["## Detalle de fugas", ""]
    L += [
        "Códigos: C cobertura, T texto extraíble, B bytes, P píxeles (zona no uniforme o igual a la del "
        "original), I imagen tapada, O imagen original intacta en el archivo, V trazos vectoriales, "
        "geometria_distinta: la salida no tiene las páginas o el tamaño esperados.",
        "",
    ]
    L += _tabla(
        ["Archivo", "Pág.", "Tipo", "Nivel", "Capa", "Valor", "Falla", "Cobertura"],
        [
            [
                ("**" if f["critica"] else "") + f["archivo"] + ("**" if f["critica"] else ""),
                f["pagina"],
                f["tipo"],
                f["nivel"],
                f["capa"],
                f["valor"],
                ", ".join(f["fallas"]),
                "—" if f["cobertura"] is None else f"{f['cobertura']:.2f}",
            ]  # fmt: skip
            for f in r["fugas"]
        ],
    )

    if m["advertencias"]:
        L += ["## Advertencias: metadatos residuales (no son fuga)", ""]
        for k, adv in m["advertencias"].items():
            L.append(f"- `{k}`: " + "; ".join(adv))
        L.append("")
    return "\n".join(L)


def escribir(resultado: dict[str, Any], carpeta: Path) -> tuple[Path, Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_json = carpeta / "evaluacion.json"
    ruta_md = carpeta / "evaluacion.md"
    ruta_json.write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ruta_md.write_text(markdown(resultado), encoding="utf-8")
    return ruta_json, ruta_md
