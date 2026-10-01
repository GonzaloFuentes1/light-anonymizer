"""Experiment with real public documents: what the notebook's logic achieves today.

For every PDF in ``test_data/real`` it applies the notebook as it is (detect + redact) and
checks the redacted file:

- data the notebook detected that is still recoverable inside the file (in streams that no
  page uses, such as the original content before redaction);
- data still visible in the page text (not detected or not redacted);
- whether the author or other metadata remains;
- pages without a text layer, which the notebook cannot process.

For privacy, the report only shows counts, never the values found.

Usage:
    uv run python -m test_bench.real_docs_experiment [--folder test_data/real] [--output results/real]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pymupdf

from test_bench.baselines import notebook
from test_bench.evaluation.pdf import pdf_strings


def _stream_text(doc: pymupdf.Document, xref: int) -> str:
    try:
        return "\n".join(pdf_strings(doc.xref_stream(xref)))
    except Exception:  # noqa: BLE001 - image or damaged streams
        return ""


def analyze(path: Path, output_dir: Path) -> dict:
    row: dict = {"file": path.as_posix(), "pages": 0}
    doc = pymupdf.open(path)
    row["pages"] = doc.page_count
    row["pages_without_text"] = sum(1 for p in doc if len(p.get_text().strip()) < 20)
    row["input_metadata"] = sorted(
        k for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")
    )
    doc.close()

    if not notebook.has_text(path):
        row["result"] = "el cuaderno lo rechaza: no tiene texto (escaneo)"
        return row

    findings = notebook.detect(path)
    output = output_dir / path.name
    notebook.redact(path, output, findings)
    row["findings"] = len(findings)
    row["by_type"] = {t: sum(1 for h in findings if h[1] == t) for t in notebook.PATTERNS}

    doc = pymupdf.open(output)
    visible = "\n".join(p.get_text() for p in doc)
    used = {x for p in doc for x in p.get_contents()}
    orphans = "\n".join(
        _stream_text(doc, x) for x in range(1, doc.xref_length()) if doc.xref_is_stream(x) and x not in used
    )
    values = {h[2] for h in findings}
    row["detected_still_visible"] = sum(1 for v in values if v in visible)
    row["detected_recoverable_in_file"] = sum(1 for v in values if v in orphans)
    leftovers = {t: len(re.findall(p, visible)) for t, p in notebook.PATTERNS.items()}
    row["patterns_in_visible_text_after_redaction"] = {t: n for t, n in leftovers.items() if n}
    row["output_metadata"] = sorted(
        k for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")
    )
    doc.close()
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Experimento con documentos públicos reales: qué logra hoy la lógica del cuaderno."
    )
    ap.add_argument("--folder", type=Path, default=Path("test_data/real"))
    ap.add_argument("--output", type=Path, default=Path("results/real"))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    seen: set[bytes] = set()
    for path in sorted(args.folder.rglob("*.pdf")):
        content = path.read_bytes()
        if content in seen:
            continue
        seen.add(content)
        try:
            rows.append(analyze(path, args.output))
        except Exception as ex:  # noqa: BLE001 - reported, and the run carries on
            rows.append({"file": path.as_posix(), "result": f"error: {type(ex).__name__}"})
    (args.output / "experiment.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    processed = [r for r in rows if "findings" in r]
    print(
        f"{len(rows)} PDF distintos; {len(processed)} procesados por el cuaderno; {len(rows) - len(processed)} rechazados"
    )
    print(f"{'archivo':<62} {'pág':>3} {'hall':>4} {'recup':>5} {'visib':>5}  metadatos tras censura")
    for r in rows:
        name = Path(r["file"]).name[:60]
        if "findings" not in r:
            print(f"{name:<62} {r.get('pages', 0):>3}  {r.get('result', '')}")
            continue
        print(
            f"{name:<62} {r['pages']:>3} {r['findings']:>4} {r['detected_recoverable_in_file']:>5} "
            f"{r['detected_still_visible']:>5}  {', '.join(r['output_metadata']) or '-'}"
        )
    total = sum(r["findings"] for r in processed)
    recoverable = sum(r["detected_recoverable_in_file"] for r in processed)
    with_meta = sum(1 for r in processed if r["output_metadata"])
    print(f"\nDatos detectados y 'censurados': {total}; recuperables dentro del PDF censurado: {recoverable}")
    print(f"PDF censurados que conservan metadatos: {with_meta} de {len(processed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
