"""Aggregation of the per-element results into the metrics of ``docs/metrics.md`` (sections 4 and 5)."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Any

import numpy as np

from test_bench.evaluation.core import ElementEval, FileEval
from test_bench.evaluation.report import spanish_label
from test_bench.schema import LEVELS, TYPES, ZERO_LEAK_TYPES, Manifest, RedactionReport

TEXT_TYPES = ("rut", "email", "phone", "url", "name", "address")
FACE_BINS = (
    (0, 24, "<24"),
    (24, 48, "24-48"),
    (48, 96, "48-96"),
    (96, 192, "96-192"),
    (192, float("inf"), ">=192"),
)
LETTER_BINS = (
    (0, 10, "<10"),
    (10, 14, "10-14"),
    (14, 20, "14-20"),
    (20, 28, "20-28"),
    (28, 40, "28-40"),
    (40, float("inf"), ">=40"),
)
DEGRADATIONS = ("noise", "salt_pepper", "blur", "blurred", "jpeg", "jpeg_compressed", "perspective",
                "lighting", "paper_texture", "table_texture", "skew", "scan")  # fmt: skip
NOT_PROCESSED_STATUSES = ("rejected", "no_result", "no_output")


def _bin(value: float | None, bins: Iterable[tuple[float, float, str]]) -> str:
    if value is None:
        return "?"
    for lo, hi, name in bins:
        if lo <= value < hi:
            return name
    return "?"


def _count(elements: list[ElementEval]) -> dict[str, Any]:
    n = len(elements)
    detected = sum(e.detected for e in elements)
    leaks = sum(e.status == "leak" for e in elements)
    not_proc = sum(e.status == "not_processed" for e in elements)
    evaluated = n - not_proc
    return {
        "n": n,
        "detected": detected,
        "recall": round(detected / n, 4) if n else None,
        "leaks": leaks,
        "pct_leaks": round(100 * leaks / evaluated, 2) if evaluated else None,
        "not_processed": not_proc,
    }


def _group(elements: list[ElementEval], key: Callable[[ElementEval], Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[ElementEval]] = defaultdict(list)
    for e in elements:
        groups[str(key(e))].append(e)
    return {k: _count(v) for k, v in sorted(groups.items(), key=lambda kv: _order(kv[0]))}


def _order(k: str) -> tuple[int, float, str]:
    for i, bins in enumerate((FACE_BINS, LETTER_BINS)):
        for j, (_, _, name) in enumerate(bins):
            if k == name:
                return (i, j, k)
    try:
        return (5, float(k), k)
    except ValueError:
        # by Spanish label, so groups keep the order they had when the codes were in Spanish
        return (9, 0, spanish_label(k))


def angle(e: ElementEval) -> str:
    if e.tags.get("mirror") or e.tags.get("mirrored"):
        return "mirrored"
    a = e.tags.get("angle")
    if a is None:
        return "0"
    try:
        return str(int(round(float(a))) % 360)
    except (TypeError, ValueError):
        return str(a)


def degradation(e: ElementEval) -> str:
    d = e.tags.get("degradation") or e.tags.get("degradations")
    if isinstance(d, (list, tuple)):
        return "+".join(str(x) for x in d) or "none"
    if d:
        return str(d)
    present = [k for k in DEGRADATIONS if e.tags.get(k)]
    return "+".join(present) if present else "none"


def letter_height(e: ElementEval) -> str:
    size = e.tags.get("size_px")
    return _bin(float(size) if isinstance(size, (int, float)) else e.height_px, LETTER_BINS)


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    v = np.asarray(values, dtype=np.float64)
    return {
        "n": len(v),
        "p50": round(float(np.percentile(v, 50)), 4),
        "p95": round(float(np.percentile(v, 95)), 4),
        "max": round(float(v.max()), 4),
    }


def _type_level_table(elements: list[ElementEval]) -> dict[str, dict[str, dict[str, Any]]]:
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for type_ in TYPES:
        for level in LEVELS:
            sub = [e for e in elements if e.type == type_ and e.level == level]
            if sub:
                table.setdefault(type_, {})[level] = _count(sub)
    return table


def _leak_detail(e: ElementEval, paths: dict[str, str]) -> dict[str, Any]:
    return {
        "element": e.id,
        "file": e.file,
        "path": paths.get(e.file, ""),
        "category": e.category,
        "page": e.page,
        "type": e.type,
        "level": e.level,
        "layer": e.layer,
        "value": e.value,
        "failures": e.failures,
        "coverage": e.coverage,
        "core_coverage": e.core_coverage,
        "critical": e.type in ZERO_LEAK_TYPES and e.level == "base",
        "detail": e.detail,
    }


def aggregate(manifest: Manifest, report: RedactionReport, files: list[FileEval], orphans: list[str]) -> dict[str, Any]:
    """Metrics, breakdowns and verdict from the evaluation of every file."""
    paths = {a.id: a.path for a in files}
    processable = [a for a in files if a.expected == "process"]
    with_error = [a for a in files if a.expected != "process"]
    elements = [e for a in processable for e in a.elements]
    targets = [e for e in elements if e.target and e.type != "text"]
    text_mode = [e for e in elements if e.target and e.type == "text"]
    neutrals = [e for e in elements if not e.target]
    processed = [a for a in processable if a.status in ("processed", "unreadable_output")]
    not_processed = [a for a in processable if a.status in NOT_PROCESSED_STATUSES]

    # -- leaks ----------------------------------------------------------------
    leaks = [e for e in targets + text_mode if e.status == "leak"]
    leaks.sort(
        key=lambda e: (not (e.type in ZERO_LEAK_TYPES and e.level == "base"), LEVELS.index(e.level), e.file, e.id)
    )
    critical = [e for e in targets if e.status == "leak" and e.type in ZERO_LEAK_TYPES and e.level == "base"]

    # -- by category ----------------------------------------------------------
    by_category: dict[str, Any] = {}
    for cat in sorted({a.category for a in processable}, key=spanish_label):
        sub = [e for e in targets if e.category == cat]
        c = _count(sub)
        c["files"] = sum(1 for a in processable if a.category == cat)
        c["critical_leaks"] = sum(1 for e in critical if e.category == cat)
        c["files_not_processed"] = sum(1 for a in not_processed if a.category == cat)
        by_category[cat] = c

    # -- breakdowns -------------------------------------------------------------
    faces = [e for e in targets if e.type == "face"]
    texts = [e for e in targets if e.type in TEXT_TYPES]
    raster_texts = [e for e in texts if e.layer == "raster"]
    ruts = [e for e in targets if e.type == "rut"]
    breakdowns = {
        "faces": {
            "size_px": _group(faces, lambda e: _bin(e.height_px, FACE_BINS)),
            "pose": _group(faces, lambda e: e.tags.get("pose", "?")),
            "angle": _group(faces, angle),
            "origin": _group(faces, lambda e: e.tags.get("origin", e.category)),
            "level": _group(faces, lambda e: e.level),
        },
        "text_in_images": {
            "angle": _group(raster_texts, angle),
            "letter_height_px": _group(raster_texts, letter_height),
            "font": _group(raster_texts, lambda e: e.tags.get("font", "?")),
            "degradation": _group(raster_texts, degradation),
            "category": _group(raster_texts, lambda e: e.category),
        },
        "text": {
            "layer": _group(texts, lambda e: e.layer),
            "format": _group(texts, lambda e: f"{e.type}:{e.tags.get('format', '?')}"),
            "name_variant": _group([e for e in texts if e.type == "name"], lambda e: e.tags.get("variant", "?")),
        },
        "rut_dv_valid": _group(ruts, lambda e: e.tags.get("dv_valid", "?")),
    }

    # -- metadata ---------------------------------------------------------------
    metas = [m for a in processable for m in a.metadata]
    meta_leaks = [m for m in metas if m.status == "leak"]
    warnings = {a.id: a.warnings for a in processed if a.warnings}

    # -- expected errors --------------------------------------------------------
    errors = [
        {"file": a.id, "path": a.path, "expected": a.expected, "status": a.status, "error": a.error} for a in with_error
    ]
    errors_ok = sum(1 for a in with_error if a.status == "correct_rejection")

    # -- over-redaction and preservation ------------------------------------------
    neutrals_cov = [e for e in neutrals if e.coverage is not None]
    decoys = [e for e in neutrals_cov if e.tags.get("decoy")]
    n_zones = sum(a.n_redactions for a in processed)
    zones_without_data = sum(a.redactions_without_data for a in processed)
    neutral_texts = [e for e in neutrals if e.extractable is not None]

    def _frac(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    overredaction = {
        "neutral_text": {
            "n": len(neutrals_cov),
            "covered_50": sum(1 for e in neutrals_cov if (e.coverage or 0) >= 0.5),
            "fraction": _frac(sum(1 for e in neutrals_cov if (e.coverage or 0) >= 0.5), len(neutrals_cov)),
        },
        "decoys": {
            kind: {
                "n": len(sub),
                "covered_50": sum(1 for e in sub if (e.coverage or 0) >= 0.5),
                "fraction": _frac(sum(1 for e in sub if (e.coverage or 0) >= 0.5), len(sub)),
            }
            for kind in sorted({str(e.tags["decoy"]) for e in decoys}, key=spanish_label)
            for sub in [[e for e in decoys if str(e.tags["decoy"]) == kind]]
        },
        "zones": {
            "n": n_zones,
            "without_personal_data": zones_without_data,
            "fraction": _frac(zones_without_data, n_zones),
        },
    }
    preservation = {
        "n": len(neutral_texts),
        "extractable": sum(1 for e in neutral_texts if e.extractable),
        "fraction": _frac(sum(1 for e in neutral_texts if e.extractable), len(neutral_texts)),
    }

    # -- times --------------------------------------------------------------------
    timed = [a for a in processed if a.time_s is not None]
    times = {
        "per_file": _stats([a.time_s for a in timed]),
        "per_pdf_page": _stats([a.time_s / max(1, a.n_pages) for a in timed if a.format == "pdf"]),
        "per_image": _stats([a.time_s for a in timed if a.format != "pdf"]),
        "without_time": [a.id for a in processed if a.time_s is None],
        "machine": report.details.get("machine"),
        "max_memory_mb": report.details.get("max_memory_mb"),
    }

    # -- "all text" mode ------------------------------------------------------------
    mode = {
        "files": [a.id for a in processable if a.all_text_mode],
        **_count(text_mode),
    }

    # -- verdict (section 5) ------------------------------------------------------------
    # criterion/detail/reasons texts are read by humans: they stay in Spanish
    criteria = []
    reasons = []
    ok1 = not critical and not not_processed
    detail1 = f"{len(critical)} fugas de rut/correo/telefono en nivel base"
    if not_processed:
        detail1 += f"; {len(not_processed)} archivos procesables sin salida (no se puede verificar que no filtren)"
    criteria.append(
        {"n": 1, "criterion": "Cero fugas de rut, correo y teléfono en nivel base", "met": ok1, "detail": detail1}
    )
    if critical:
        cats = sorted(spanish_label(c) for c in {e.category for e in critical})
        reasons.append(f"{len(critical)} fugas críticas (rut/correo/teléfono, nivel base) en: {', '.join(cats)}")
    if not_processed:
        reasons.append(
            f"{len(not_processed)} archivos procesables rechazados o sin salida: "
            + ", ".join(a.id for a in not_processed[:10])
            + (" ..." if len(not_processed) > 10 else "")
        )
    ok2 = not meta_leaks
    criteria.append(
        {
            "n": 2,
            "criterion": "Cero fugas de metadatos sensibles",
            "met": ok2,
            "detail": f"{len(meta_leaks)} de {len(metas)}",
        }
    )
    if meta_leaks:
        reasons.append(f"{len(meta_leaks)} fugas de metadatos sensibles")
    ok3 = errors_ok == len(with_error)
    criteria.append(
        {
            "n": 3,
            "criterion": "Archivos con error esperado rechazados con el código correcto",
            "met": ok3,
            "detail": f"{errors_ok} de {len(with_error)}",
        }
    )
    if not ok3:
        bad = [a.id for a in with_error if a.status != "correct_rejection"]
        reasons.append(f"{len(bad)} archivos con error esperado mal manejados: {', '.join(bad[:10])}")
    criteria.append(
        {
            "n": 4,
            "criterion": "Informe de recall de rostros y OCR con desgloses",
            "met": True,
            "detail": f"{len(faces)} rostros y {len(raster_texts)} textos en imágenes (informativo, sin umbral)",
        }
    )
    ok5 = bool(processed) and not times["without_time"]
    criteria.append(
        {
            "n": 5,
            "criterion": "Tiempos por imagen y por página",
            "met": ok5,
            "detail": f"{len(timed)} de {len(processed)} archivos con tiempo",
        }
    )
    if not ok5:
        reasons.append(f"faltan tiempos en {len(times['without_time'])} archivos procesados")
    passed = ok1 and ok2 and ok3 and ok5

    evaluator_failures = {a.id: a.evaluator_failure for a in files if a.evaluator_failure}
    summary = {
        "files": len(files),
        "processable": len(processable),
        "processed": len(processed),
        "not_processed": len(not_processed),
        "with_expected_error": len(with_error),
        "target_elements": len(targets),
        "detected": sum(e.detected for e in targets),
        "recall": _frac(sum(e.detected for e in targets), len(targets)),
        "leaks": sum(1 for e in targets if e.status == "leak"),
        "critical_leaks": len(critical),
        "metadata": len(metas),
        "metadata_leaks": len(meta_leaks),
        "geometry_mismatch": {a.id: a.geometry_mismatch for a in processed if a.geometry_mismatch},
        "results_without_file": orphans,
        "evaluator_failures": evaluator_failures,
    }
    return {
        "system": report.system,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": manifest.seed,
        "verdict": {"passed": passed, "reasons": reasons, "criteria": criteria},
        "summary": summary,
        "by_type_level": _type_level_table(targets),
        "by_category": by_category,
        "breakdowns": breakdowns,
        "metadata": {
            "n": len(metas),
            "leaks": [asdict(m) | {"path": paths.get(m.file, "")} for m in meta_leaks],
            "removed": sum(1 for m in metas if m.status == "removed"),
            "not_processed": sum(1 for m in metas if m.status == "not_processed"),
            "warnings": warnings,
        },
        "expected_errors": {"n": len(with_error), "correct": errors_ok, "files": errors},
        "processable_not_processed": [
            {"file": a.id, "path": a.path, "status": a.status, "error": a.error} for a in not_processed
        ],
        "overredaction": overredaction,
        "text_preservation": preservation,
        "times": times,
        "all_text_mode": mode,
        "leaks": [_leak_detail(e, paths) for e in leaks],
        "files": [{k: v for k, v in asdict(a).items() if k not in ("elements", "metadata")} for a in files],
        "elements": [asdict(e) for e in elements],
        "metadata_elements": [asdict(m) for m in metas],
    }
