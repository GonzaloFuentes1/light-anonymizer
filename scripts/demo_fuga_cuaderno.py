"""Reproduce la fuga del cuaderno CoP 33: el texto "censurado" sigue dentro del PDF.

Usa el código del cuaderno sin cambios (celdas 6, 8, 12 y 16) y después intenta recuperar
los datos del archivo censurado leyendo sus objetos internos, como lo haría cualquiera con
una herramienta PDF.

Uso (desde la raíz del repositorio):
    uv run python scripts/demo_fuga_cuaderno.py
"""

import re
import tempfile
from pathlib import Path

import pymupdf

# --- Celda 6 del cuaderno (documento de prueba) ---------------------------------------------
texto_prueba = """INFORME DE HONORARIOS - AGOSTO 2026

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

# --- Celda 8 --------------------------------------------------------------------------------
PATRONES = {
    "rut": r"\b\d{1,2}\.?\d{3}\.?\d{3}\s*-\s*[\dkK]\b",
    "correo": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "telefono": r"(?:\+?56\s?)?(?:9\s?\d{4}\s?\d{4}|\b2\s?\d{4}\s?\d{4}\b)",
    "url": r"https?://\S+|\bwww\.\S+",
}


# --- Celda 12 -------------------------------------------------------------------------------
def detectar(ruta):
    doc = pymupdf.open(ruta)
    hallazgos = []
    for n_pagina, pagina in enumerate(doc, start=1):
        texto = pagina.get_text()
        for tipo, patron in PATRONES.items():
            for m in re.finditer(patron, texto):
                hallazgos.append((n_pagina, tipo, m.group(0)))
    doc.close()
    return hallazgos


# --- Celda 16 -------------------------------------------------------------------------------
def censurar(entrada, salida, hallazgos):
    doc = pymupdf.open(entrada)
    for n_pagina, _tipo, valor in hallazgos:
        pagina = doc[n_pagina - 1]
        for rect in pagina.search_for(valor):
            pagina.add_redact_annot(rect, fill=(0, 0, 0))
    for pagina in doc:
        pagina.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
    doc.save(salida)
    doc.close()


def cadenas_hex(flujo: bytes) -> str:
    """Texto de las cadenas hexadecimales <...> de un flujo de contenido PDF."""
    return " | ".join(bytes.fromhex(h.decode()).decode("latin-1") for h in re.findall(rb"<([0-9A-Fa-f]+)>", flujo))


def main() -> None:
    carpeta = Path(tempfile.mkdtemp())
    entrada, salida = carpeta / "informe_de_prueba.pdf", carpeta / "CENSURADO_informe_de_prueba.pdf"

    doc = pymupdf.open()
    doc.new_page().insert_text((60, 70), texto_prueba, fontsize=11, fontname="helv")
    doc.save(entrada)
    doc.close()

    hallazgos = detectar(entrada)
    censurar(entrada, salida, hallazgos)

    doc = pymupdf.open(salida)
    visible = doc[0].get_text()
    print("1. Lo que el cuaderno verifica (texto de la página censurada):")
    print(
        "   ¿aparece algún dato?", any(v in visible for _, _, v in hallazgos), "-> el cuaderno informa 'eliminados'\n"
    )

    usados = set(doc[0].get_contents())
    print("2. Lo que sigue dentro del archivo censurado:")
    for xref in range(1, doc.xref_length()):
        if doc.xref_is_stream(xref) and xref not in usados:
            texto = cadenas_hex(doc.xref_stream(xref))
            if texto:
                print(f"   objeto {xref} (no lo usa ninguna página, pero está en el archivo):")
                for linea in texto.split(" | "):
                    if linea.strip():
                        print("     ", linea)
    doc.close()

    doc = pymupdf.open(salida)
    doc.save(carpeta / "con_garbage.pdf", garbage=4, deflate=True)
    doc.close()
    limpio = pymupdf.open(carpeta / "con_garbage.pdf")
    restos = [
        x
        for x in range(1, limpio.xref_length())
        if limpio.xref_is_stream(x) and "15.782" in cadenas_hex(limpio.xref_stream(x))
    ]
    print("\n3. Guardando con garbage=4, objetos con el RUT:", restos or "ninguno")


if __name__ == "__main__":
    main()
