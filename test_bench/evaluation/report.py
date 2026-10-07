"""Writing of the evaluation result: ``evaluation.json`` and ``evaluation.md`` (in Spanish).

The JSON keeps the English codes; the Markdown report is read by humans (Chilean public
officials), so its texts are in Spanish and every code is shown through ``SPANISH_LABELS``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_bench.schema import LEVELS

# Code -> Spanish label shown in the Markdown report (codes not listed are shown as they are).
SPANISH_LABELS = {
    # types, levels, layers
    "email": "correo", "phone": "telefono", "name": "nombre", "address": "direccion", "face": "rostro",
    "signature": "firma", "text": "texto", "stress": "estres", "out_of_scope": "fuera_de_alcance",
    "hidden": "oculto",
    # element, metadata and file statuses; failure reasons
    "redacted": "censurado", "leak": "fuga", "not_processed": "no_procesado", "neutral": "neutro",
    "removed": "eliminado", "processed": "procesado", "rejected": "rechazado", "no_result": "sin_resultado",
    "no_output": "sin_salida", "unreadable_output": "salida_ilegible", "correct_rejection": "rechazo_correcto",
    "wrong_code_rejection": "rechazo_codigo_distinto", "not_rejected": "no_rechazado",
    "evaluator_error": "error_evaluador", "geometry_mismatch": "geometria_distinta",
    # categories
    "pdf_text": "pdf_texto", "pdf_metadata": "pdf_metadatos", "pdf_scanned": "pdf_escaneado",
    "rotated_image": "imagen_rotada", "id_card": "cedula", "screenshot": "pantallazo", "faces": "rostros",
    "errors": "errores",
    # poses
    "front": "frente", "three_quarter": "tres_cuartos", "profile": "perfil",
    # RUT formats
    "dots": "puntos", "no_dots": "sin_puntos", "spaced_hyphen": "espacios_guion", "en_dash": "guion_largo",
    "no_hyphen": "sin_guion", "commas": "comas", "inner_spaces": "espacios_internos",
    # phone formats and kinds
    "mobile_international": "movil_internacional", "mobile_compact": "movil_compacto",
    "mobile_national": "movil_nacional", "mobile_block": "movil_bloque", "mobile_parentheses": "movil_parentesis",
    "mobile_hyphens": "movil_guiones", "santiago_international": "santiago_internacional",
    "santiago_national": "santiago_nacional", "santiago_parentheses": "santiago_parentesis",
    "regional_international": "regional_internacional", "regional_parentheses": "regional_parentesis",
    "regional_compact": "regional_compacto", "regional_hyphen": "regional_guion",
    "old_mobile_09": "antiguo_movil_09", "old_regional_0": "antiguo_regional_0", "old_mobile_8": "antiguo_movil_8",
    "mobile": "movil", "landline": "fijo",
    # email formats
    "dot": "punto", "initial": "inicial", "with_year": "con_anio", "underscore": "guion_bajo",
    "uppercase": "mayusculas", "plus": "mas", "subdomain": "subdominio", "spelled_arroba": "arroba_texto",
    "spaces": "espacios",
    # name variants
    "exact": "exacta", "no_accents": "sin_tildes", "surnames_names": "apellidos_nombres",
    "surnames_first": "apellidos_primero", "partial": "parcial", "first_name_only": "solo_nombre",
    "full": "completo", "given_names": "nombres",
    # angles, systems, decoys, origins
    "mirrored": "espejado", "identity": "identidad", "oracle": "oraculo", "notebook": "cuaderno",
    "prototype": "prototipo", "engine": "motor", "amount": "monto", "date": "fecha", "poster": "afiche",
    # fonts
    "sans_bold": "sans_negrita", "sans_oblique": "sans_oblicua", "serif_bold": "serif_negrita",
    "serif_italic": "serif_cursiva", "mono_bold": "mono_negrita", "stix_italic": "stix_cursiva",
    # degradations
    "noise": "ruido", "light_noise": "ruido_leve", "heavy_noise": "ruido_fuerte", "salt_pepper": "sal_pimienta",
    "blur": "desenfoque", "blurred": "desenfocar", "jpeg_compressed": "comprimir_jpeg",
    "perspective": "perspectiva", "lighting": "iluminacion", "paper_texture": "textura_papel",
    "table_texture": "textura_mesa", "skew": "inclinacion", "scan": "escaneo", "mirror": "espejo",
    "binarized_1bit": "binarizado_1bit", "binarized": "binarizado", "low_resolution": "baja_resolucion",
    "none": "ninguna", "rotation": "giro", "gray": "gris", "binarize": "binarizar", "paper": "papel",
    "quality": "calidad",
    # expected results and error codes
    "process": "procesar", "password": "contrasena", "corrupt": "corrupto", "empty": "vacio",
    "format": "formato", "unsupported": "no_soportado", "no_text": "sin_texto", "exception": "excepcion",
    # redaction statuses
    "redact": "censurar", "dismissed": "descartado",
    # tag values (variants, fields, decoys, kinds, origins)
    "paternal_surname": "apellido_paterno", "maternal_surname": "apellido_materno", "sex": "sexo",
    "birth_date": "fecha_nacimiento", "birth_place": "lugar_nacimiento", "issue_date": "fecha_emision",
    "expiry_date": "fecha_vencimiento", "document_number": "numero_documento", "profession": "profesion",
    "commune": "comuna", "time": "hora", "photo_on_table": "foto_sobre_mesa",
    "application_photo": "foto_postulacion", "credential": "credencial", "card": "tarjeta", "hand": "mano",
    "dark_glasses": "lentes_oscuros", "tiny": "diminuto", "small": "pequeno", "large": "grande",
    "unannotated": "sin_anotar", "lowercase_k": "k_minuscula", "low_contrast": "bajo_contraste",
    "tiny_text": "texto_diminuto", "rotation_90": "giro_90", "rotation_180": "giro_180",
    "rotation_270": "giro_270", "rotated_rut": "rut_girado", "url_with_rut": "url_con_rut",
    "personal_web": "web_personal", "under_image": "bajo_imagen", "white_on_white": "blanco_sobre_blanco",
    "outside_cropbox": "fuera_del_cropbox", "transparent_png": "png_transparente",
    "drawn_rectangle": "rectangulo_dibujado", "unapplied_redact": "redact_sin_aplicar",
    "square_annotation": "anotacion_cuadrada", "freetext_annotation": "anotacion_freetext",
    "ocg_off": "ocg_apagada", "geometry": "geometria", "form": "formulario", "note": "nota",
    "name,email": "nombre,correo", "synthetic_drawn": "sintetico_dibujado", "context": "contexto",
    "full_page": "pagina_completa", "name_list": "lista_nombres",
    # metadata locations
    "exif.thumbnail": "exif.miniatura", "exif.tags": "exif.etiquetas", "png.text": "png.texto",
    "png.text.Author": "png.texto.Author", "png.text.Comment": "png.texto.Comment",
    "png.text.Description": "png.texto.Description", "png.other": "png.otros",
    "png.data_after_iend": "png.datos_tras_iend", "jpeg.data_after_eoi": "jpeg.datos_tras_eoi",
    "webp.other": "webp.otros", "webp.data_after_riff": "webp.datos_tras_riff", "tiff.tags": "tiff.etiquetas",
    "pdf.annotation": "pdf.anotacion", "pdf.attachment": "pdf.adjunto", "pdf.form": "pdf.formulario",
    "pdf.bookmark": "pdf.marcador", "pdf.previous_revision": "pdf.revision_anterior",
    # where a canary was found (haystack origin : needle part)
    "text_pymupdf": "texto_pymupdf", "text_pdfium": "texto_pdfium", "bytes_strings": "bytes_cadenas",
    "objects": "objetos", "object_strings": "objetos_cadenas", "streams": "flujos",
    "streams_strings": "flujos_cadenas", "stream": "flujo", "stream_strings": "flujo_cadenas",
    "text_pymupdf_strings": "texto_pymupdf_cadenas", "text_pdfium_strings": "texto_pdfium_cadenas",
    "value": "valor", "rut_body": "cuerpo_rut", "last7": "ultimos7", "surnames": "apellidos",
    "street_number": "calle_numero", "digits": "digitos", "compact": "compacto",
}  # fmt: skip

# Compound codes whose Spanish label is not the part-by-part translation (gender agreement).
_COMPOUND_LABELS = {"address:full": "direccion:completa"}

# Human reason texts that end with a code (``canario en bytes:value``): only the code is translated.
_REASON_PREFIXES = ("canario en ", "estructura presente: ", "lugar no reconocido por el evaluador: ")

# Names the evaluator gives to the TIFF tags it reads (``evaluation.image.TIFF_TAGS``), as shown
# in canary origins (``tiff.artist``) and in the TIFF warning (``p0.artist``).
_TIFF_NAME_LABELS = {"description": "descripcion", "artist": "artista", "document": "documento", "page": "pagina",
                     "date": "fecha", "host_computer": "equipo"}  # fmt: skip
_TIFF_WARNING = "etiquetas TIFF: "


def spanish_label(v: Any) -> Any:
    """Spanish label of a code; compound codes (``noise+blur``, ``email:dot``, ``error:password``)
    are translated part by part. Values that are not strings are returned as they are."""
    if not isinstance(v, str):
        return v
    if v in _COMPOUND_LABELS:
        return _COMPOUND_LABELS[v]
    if v in SPANISH_LABELS:
        return SPANISH_LABELS[v]
    return "+".join(":".join(SPANISH_LABELS.get(part, part) for part in piece.split(":")) for piece in v.split("+"))


def _reason_label(reason: str) -> str:
    """A reason of a metadata leak, with its trailing code translated."""
    for prefix in _REASON_PREFIXES:
        if reason.startswith(prefix):
            code = reason[len(prefix) :]
            if prefix == "canario en " and code.startswith("tiff."):
                origin, sep, part = code.partition(":")
                name = origin[len("tiff.") :]
                return f"{prefix}tiff.{_TIFF_NAME_LABELS.get(name, name)}{sep}{spanish_label(part)}"
            return prefix + spanish_label(code)
    return SPANISH_LABELS.get(reason, reason)


def _warning_label(warning: str) -> str:
    """A metadata warning; the TIFF tag list (``p0.artist, ...``) is shown with the Spanish tag names."""
    if not warning.startswith(_TIFF_WARNING):
        return warning
    names = []
    for item in warning[len(_TIFF_WARNING) :].split(", "):
        page, sep, name = item.partition(".")
        names.append(f"{page}{sep}{_TIFF_NAME_LABELS.get(name, name)}")
    return _TIFF_WARNING + ", ".join(names)


def _error_label(error: str | None) -> str | None:
    """An error reported by the system (``code: free text``): only the code is translated."""
    if not error:
        return error
    code, sep, rest = error.partition(":")
    return SPANISH_LABELS.get(code.strip(), code) + sep + rest


def _pct(x: float | None, decimals: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{decimals}f} %"


def _cell(v: Any) -> str:
    return "—" if v is None else str(v).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["_(sin datos)_", ""]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in f) + " |" for f in rows]
    return lines + [""]


def _counts_table(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    rows = [
        [
            spanish_label(k),
            g["n"],
            g["detected"],
            _pct(g["recall"]),
            g["leaks"],
            "—" if g["pct_leaks"] is None else f"{g['pct_leaks']:.1f} %",
        ]
        for k, g in groups.items()
    ]
    return [f"**{title}**", ""] + _table(
        [title.split(" por ")[-1], "n", "detectados", "recall", "fugas", "% fugas"], rows
    )


def markdown(r: dict[str, Any]) -> str:
    v = r["verdict"]
    s = r["summary"]
    L: list[str] = [f"# Evaluación del anonimizador: `{spanish_label(r['system'])}`", ""]
    L.append(f"Fecha: {r['date']} · semilla del conjunto: {r['seed']}")
    L.append("")
    if v["passed"]:
        L += ["## Veredicto: **APROBADO**", ""]
    else:
        L += ["## Veredicto: **NO APROBADO**", ""]
        L += [f"- {m}" for m in v["reasons"]]
        L.append("")
    L += _table(
        ["#", "Criterio", "Cumple", "Detalle"],
        [[c["n"], c["criterion"], "sí" if c["met"] else "**NO**", c["detail"]] for c in v["criteria"]],
    )

    L += ["## Resumen", ""]
    L += _table(
        ["Medida", "Valor"],
        [
            [
                "Archivos (procesables / con error esperado)",
                f"{s['files']} ({s['processable']} / {s['with_expected_error']})",
            ],
            ["Procesados / sin salida", f"{s['processed']} / {s['not_processed']}"],
            ["Elementos objetivo", s["target_elements"]],
            ["Recall global", _pct(s["recall"])],
            ["Fugas (críticas)", f"{s['leaks']} ({s['critical_leaks']})"],
            ["Metadatos sensibles con fuga", f"{s['metadata_leaks']} de {s['metadata']}"],
        ],
    )
    if s["geometry_mismatch"]:
        L += ["**Salidas con geometría distinta** (sus elementos cuentan como fuga):", ""]
        L += [f"- `{k}`: {'; '.join(x)}" for k, x in s["geometry_mismatch"].items()]
        L.append("")
    if s["evaluator_failures"]:
        L += ["**Archivos que el evaluador no pudo revisar** (se cuentan como fuga):", ""]
        L += [f"- `{k}`: {x.strip().splitlines()[-1]}" for k, x in s["evaluator_failures"].items()]
        L.append("")
    if s["results_without_file"]:
        L += ["Resultados del informe que no corresponden a ningún archivo del manifiesto: "]
        L += [", ".join(f"`{x}`" for x in s["results_without_file"]), ""]

    L += ["## Recall y fugas por tipo y nivel", ""]
    rows = []
    for type_, levels in r["by_type_level"].items():
        for level in LEVELS:
            if level in levels:
                g = levels[level]
                rows.append(
                    [
                        spanish_label(type_),
                        spanish_label(level),
                        g["n"],
                        g["detected"],
                        _pct(g["recall"]),
                        g["leaks"],
                        "—" if g["pct_leaks"] is None else f"{g['pct_leaks']:.1f} %",
                        g["not_processed"],
                    ]  # fmt: skip
                )
    L += _table(["Tipo", "Nivel", "n", "Detectados", "Recall", "Fugas", "% fugas", "No procesados"], rows)
    L += ["Recall = cobertura suficiente (≥ 95 %; rostros: núcleo ≥ 95 % y cara ≥ 80 %). En la capa oculta, "
          "detectado = eliminado (T y B). % fugas sobre los elementos de archivos procesados.", ""]  # fmt: skip

    L += ["## Por categoría de archivo", ""]
    L += _table(
        ["Categoría", "Archivos", "Sin salida", "n", "Recall", "Fugas", "Fugas críticas"],
        [
            [
                spanish_label(k),
                c["files"],
                c["files_not_processed"],
                c["n"],
                _pct(c["recall"]),
                c["leaks"],
                c["critical_leaks"],
            ]
            for k, c in r["by_category"].items()
        ],
    )

    m = r["metadata"]
    L += ["## Metadatos sensibles", ""]
    L.append(f"{m['n']} metadatos sensibles: {m['removed']} eliminados, {len(m['leaks'])} con fuga, "
             f"{m['not_processed']} en archivos sin salida.")  # fmt: skip
    L.append("")
    L += _table(
        ["Archivo", "Dónde", "Valor", "Motivo"],
        [
            [f["file"], spanish_label(f["location"]), f["value"], "; ".join(_reason_label(x) for x in f["reasons"])]
            for f in m["leaks"]
        ],
    )

    e = r["expected_errors"]
    L += ["## Archivos con error esperado", ""]
    L.append(f"{e['correct']} de {e['n']} rechazados con el código correcto.")
    L.append("")
    L += _table(
        ["Archivo", "Esperado", "Estado", "Error informado"],
        [
            [x["file"], spanish_label(x["expected"]), spanish_label(x["status"]), _error_label(x["error"])]
            for x in e["files"]
        ],
    )
    L += ["## Archivos procesables rechazados o sin salida", ""]
    L += ["Cuentan como no detectados en el recall, no como fugas.", ""]
    L += _table(
        ["Archivo", "Estado", "Error"],
        [[x["file"], spanish_label(x["status"]), _error_label(x["error"])] for x in r["processable_not_processed"]],
    )

    d = r["breakdowns"]
    L += ["## Desgloses de recall", "", "### Rostros", ""]
    L += _counts_table("Rostros por alto de la cara (px en la salida; PDF a 144 ppp)", d["faces"]["size_px"])
    L += _counts_table("Rostros por pose", d["faces"]["pose"])
    L += _counts_table("Rostros por ángulo", d["faces"]["angle"])
    L += _counts_table("Rostros por origen", d["faces"]["origin"])
    L += _counts_table("Rostros por nivel", d["faces"]["level"])
    L += ["### Texto en imágenes (capa raster)", ""]
    L += _counts_table("Texto por ángulo", d["text_in_images"]["angle"])
    L += _counts_table("Texto por alto de letra", d["text_in_images"]["letter_height_px"])
    L += _counts_table("Texto por fuente", d["text_in_images"]["font"])
    L += _counts_table("Texto por degradación", d["text_in_images"]["degradation"])
    L += _counts_table("Texto por categoría", d["text_in_images"]["category"])
    L += ["### Texto (todas las capas)", ""]
    L += _counts_table("Texto por capa", d["text"]["layer"])
    L += _counts_table("Texto por formato", d["text"]["format"])
    L += _counts_table("Nombres por variante", d["text"]["name_variant"])
    L += _counts_table("RUT por dígito verificador válido", d["rut_dv_valid"])

    ov, pres = r["overredaction"], r["text_preservation"]
    L += ["## Métricas informativas", ""]
    rows = [
        [
            "Texto neutro cubierto ≥ 50 %",
            f"{ov['neutral_text']['covered_50']} de {ov['neutral_text']['n']}",
            _pct(ov["neutral_text"]["fraction"]),
        ],  # fmt: skip
    ]
    for kind, x in ov["decoys"].items():
        rows.append(
            [
                f"Señuelos '{spanish_label(kind)}' cubiertos ≥ 50 %",
                f"{x['covered_50']} de {x['n']}",
                _pct(x["fraction"]),
            ]
        )
    rows.append(
        [
            "Zonas censuradas que no tocan datos",
            f"{ov['zones']['without_personal_data']} de {ov['zones']['n']}",
            _pct(ov["zones"]["fraction"]),
        ]  # fmt: skip
    )
    rows.append(
        ["Texto neutro del PDF que sigue extraíble", f"{pres['extractable']} de {pres['n']}", _pct(pres["fraction"])]
    )
    L += _table(["Medida", "Conteo", "Fracción"], rows)

    t = r["times"]
    L += ["### Tiempos (s)", ""]
    L += _table(
        ["Medida", "n", "mediana", "p95", "máximo"],
        [
            [k, x["n"], x["p50"], x["p95"], x["max"]]
            for k, x in (
                ("por archivo", t["per_file"]),
                ("por página de PDF", t["per_pdf_page"]),
                ("por imagen", t["per_image"]),
            )
        ],  # fmt: skip
    )
    if t.get("machine"):
        L += [f"Equipo: {t['machine']}", ""]
    if t.get("max_memory_mb"):
        L += [f"Memoria máxima: {t['max_memory_mb']} MB", ""]
    if t["without_time"]:
        L += [f"Sin tiempo informado: {len(t['without_time'])} archivos.", ""]

    mo = r["all_text_mode"]
    if mo["files"]:
        L += ["### Modo «censurar todo el texto»", ""]
        L.append(f"Activado en {len(mo['files'])} archivos: {mo['n']} textos, recall {_pct(mo['recall'])}, "
                 f"{mo['leaks']} fugas.")  # fmt: skip
        L.append("")

    L += ["## Detalle de fugas", ""]
    L += [
        "Códigos: C cobertura, T texto extraíble, B bytes, P píxeles (zona no uniforme o igual a la del "
        "original), I imagen tapada, O imagen original intacta en el archivo, V trazos vectoriales, "
        "geometria_distinta: la salida no tiene las páginas o el tamaño esperados.",
        "",
    ]
    L += _table(
        ["Archivo", "Pág.", "Tipo", "Nivel", "Capa", "Valor", "Falla", "Cobertura"],
        [
            [
                ("**" if f["critical"] else "") + f["file"] + ("**" if f["critical"] else ""),
                f["page"],
                spanish_label(f["type"]),
                spanish_label(f["level"]),
                spanish_label(f["layer"]),
                f["value"],
                ", ".join(spanish_label(x) for x in f["failures"]),
                "—" if f["coverage"] is None else f"{f['coverage']:.2f}",
            ]  # fmt: skip
            for f in r["leaks"]
        ],
    )

    if m["warnings"]:
        L += ["## Advertencias: metadatos residuales (no son fuga)", ""]
        for k, warn in m["warnings"].items():
            L.append(f"- `{k}`: " + "; ".join(_warning_label(w) for w in warn))
        L.append("")
    return "\n".join(L)


def write_report(result: dict[str, Any], folder: Path) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "evaluation.json"
    md_path = folder / "evaluation.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md_path.write_text(markdown(result), encoding="utf-8")
    return json_path, md_path
