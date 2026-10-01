"""Reproduces the CoP 33 notebook leak: the "redacted" text is still inside the PDF.

Uses the notebook code unchanged (cells 6, 8, 12 and 16; only the identifiers are translated)
and then tries to recover the data from the redacted file by reading its internal objects, as
anyone with a PDF tool would.

Usage (from the repository root):
    uv run python scripts/demo_notebook_leak.py
"""

import re
import tempfile
from pathlib import Path

import pymupdf

# --- Notebook cell 6 (test document) --------------------------------------------------------
test_text = """INFORME DE HONORARIOS - AGOSTO 2026

Nombre: Ana Maria Rojas Pena
RUT: 15.782.334-9
Correo: ana.rojas@ejemplo.cl
Telefono: +56 9 8123 4567
Direccion: Pasaje Los Alerces 442, depto 31

Producto 1: Informe de avance del programa
Monto bruto: $ 1.450.000

Contraparte: Jefatura de la unidad
RUT contraparte: 9.876.543-3
Sitio: https://www.ejemplo.cl/rendiciones
"""

# --- Cell 8 ---------------------------------------------------------------------------------
PATTERNS = {
    "rut": r"\b\d{1,2}\.?\d{3}\.?\d{3}\s*-\s*[\dkK]\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "phone": r"(?:\+?56\s?)?(?:9\s?\d{4}\s?\d{4}|\b2\s?\d{4}\s?\d{4}\b)",
    "url": r"https?://\S+|\bwww\.\S+",
}


# --- Cell 12 --------------------------------------------------------------------------------
def detect(path):
    doc = pymupdf.open(path)
    findings = []
    for n_page, page in enumerate(doc, start=1):
        text = page.get_text()
        for type_, pattern in PATTERNS.items():
            for m in re.finditer(pattern, text):
                findings.append((n_page, type_, m.group(0)))
    doc.close()
    return findings


# --- Cell 16 --------------------------------------------------------------------------------
def redact(source, output, findings):
    doc = pymupdf.open(source)
    for n_page, _type, value in findings:
        page = doc[n_page - 1]
        for rect in page.search_for(value):
            page.add_redact_annot(rect, fill=(0, 0, 0))
    for page in doc:
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
    doc.save(output)
    doc.close()


def hex_strings(stream: bytes) -> str:
    """Text of the hexadecimal strings <...> of a PDF content stream."""
    return " | ".join(bytes.fromhex(h.decode()).decode("latin-1") for h in re.findall(rb"<([0-9A-Fa-f]+)>", stream))


def main() -> None:
    folder = Path(tempfile.mkdtemp())
    source, output = folder / "informe_de_prueba.pdf", folder / "CENSURADO_informe_de_prueba.pdf"

    doc = pymupdf.open()
    doc.new_page().insert_text((60, 70), test_text, fontsize=11, fontname="helv")
    doc.save(source)
    doc.close()

    findings = detect(source)
    redact(source, output, findings)

    doc = pymupdf.open(output)
    visible = doc[0].get_text()
    print("1. Lo que el cuaderno verifica (texto de la página censurada):")
    print("   ¿aparece algún dato?", any(v in visible for _, _, v in findings), "-> el cuaderno informa 'eliminados'\n")

    used = set(doc[0].get_contents())
    print("2. Lo que sigue dentro del archivo censurado:")
    for xref in range(1, doc.xref_length()):
        if doc.xref_is_stream(xref) and xref not in used:
            text = hex_strings(doc.xref_stream(xref))
            if text:
                print(f"   objeto {xref} (no lo usa ninguna página, pero está en el archivo):")
                for line in text.split(" | "):
                    if line.strip():
                        print("     ", line)
    doc.close()

    doc = pymupdf.open(output)
    doc.save(folder / "con_garbage.pdf", garbage=4, deflate=True)
    doc.close()
    clean = pymupdf.open(folder / "con_garbage.pdf")
    leftovers = [
        x
        for x in range(1, clean.xref_length())
        if clean.xref_is_stream(x) and "15.782" in hex_strings(clean.xref_stream(x))
    ]
    print("\n3. Guardando con garbage=4, objetos con el RUT:", leftovers or "ninguno")


if __name__ == "__main__":
    main()
