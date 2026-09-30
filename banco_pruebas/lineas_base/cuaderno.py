"""Línea base ``cuaderno``: la lógica del cuaderno CoP 33 (boceto original, ya retirado del repositorio) tal cual.

Es una referencia: documenta qué cubre hoy el cuaderno y qué no. Por eso se conserva su
comportamiento aunque sea débil (no limpia metadatos, guarda con ``doc.save`` simple, busca cada
hallazgo con ``search_for`` y no procesa escaneos ni imágenes).

Adaptaciones mínimas para el banco de pruebas:
- ``NOMBRES`` es la ``lista_nombres`` del manifiesto (en el cuaderno es una lista fija).
- Se detecta con ``detectar() + detectar_nombres()`` en una sola pasada (celda 22 del cuaderno).
- El archivo de salida conserva la ruta relativa de la entrada (sin el prefijo ``CENSURADO_``).
- Las funciones devuelven sus resultados en vez de imprimirlos.
- PDF sin texto -> error ``sin_texto`` (el cuaderno lo salta); imagen -> ``no_soportado``;
  cualquier excepción -> ``excepcion:<Tipo>``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf

from banco_pruebas.esquema import Archivo, Censura, Manifiesto, ResultadoArchivo

# Celda 8 del cuaderno, sin cambios.
PATRONES = {
    # 12.345.678-9 o 12345678-9, con o sin puntos, con o sin espacios
    "rut": r"\b\d{1,2}\.?\d{3}\.?\d{3}\s*-\s*[\dkK]\b",
    "correo": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    # celular +56 9 XXXX XXXX y fijo 2 XXXX XXXX
    "telefono": r"(?:\+?56\s?)?(?:9\s?\d{4}\s?\d{4}|\b2\s?\d{4}\s?\d{4}\b)",
    "url": r"https?://\S+|\bwww\.\S+",
}

# Celda 22 del cuaderno; en el banco se reemplaza por la lista del manifiesto.
NOMBRES = [
    "Ana Maria Rojas Pena",
]

Hallazgo = tuple[int, str, str]  # (página desde 1, tipo, texto encontrado)


def tiene_texto(ruta: Path | str, minimo: int = 50) -> bool:
    """Devuelve True si el PDF tiene texto extraible (celda 10)."""
    doc = pymupdf.open(ruta)
    caracteres = sum(len(p.get_text()) for p in doc)
    doc.close()
    return caracteres >= minimo


def detectar(ruta: Path | str) -> list[Hallazgo]:
    """Devuelve una lista de (pagina, tipo, texto encontrado) (celda 12)."""
    doc = pymupdf.open(ruta)
    hallazgos = []
    for n_pagina, pagina in enumerate(doc, start=1):
        texto = pagina.get_text()
        for tipo, patron in PATRONES.items():
            for m in re.finditer(patron, texto):
                hallazgos.append((n_pagina, tipo, m.group(0)))
    doc.close()
    return hallazgos


def detectar_nombres(ruta: Path | str, nombres: list[str] | None = None) -> list[Hallazgo]:
    """Busca cada nombre de la lista con ``search_for`` (celda 22)."""
    nombres = NOMBRES if nombres is None else nombres
    doc = pymupdf.open(ruta)
    h = []
    for n, pagina in enumerate(doc, start=1):
        for nombre in nombres:
            if pagina.search_for(nombre):
                h.append((n, "nombre", nombre))
    doc.close()
    return h


def censurar(entrada: Path | str, salida: Path | str, hallazgos: list[Hallazgo]) -> list[Censura]:
    """Marca cada hallazgo, aplica las censuras y guarda (celda 16). Devuelve las zonas marcadas."""
    doc = pymupdf.open(entrada)
    censuras: list[Censura] = []
    vistas: set[tuple[int, str, tuple[float, ...]]] = set()

    # 1. marcar cada hallazgo
    for n_pagina, tipo, valor in hallazgos:
        pagina = doc[n_pagina - 1]
        for rect in pagina.search_for(valor):
            pagina.add_redact_annot(rect, fill=(0, 0, 0))
            clave = (n_pagina, tipo, tuple(round(v, 3) for v in rect))
            if clave not in vistas:
                vistas.add(clave)
                censuras.append(
                    Censura(
                        pagina=n_pagina - 1,
                        poligono=[[rect.x0, rect.y0], [rect.x1, rect.y0], [rect.x1, rect.y1], [rect.x0, rect.y1]],
                        tipo=tipo,
                        detector="lista_nombres" if tipo == "nombre" else "regex",
                        texto=valor,
                    )
                )

    # 2. aplicar: aca el contenido desaparece del archivo
    for pagina in doc:
        pagina.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)

    doc.save(salida)
    doc.close()
    return censuras


def verificar(salida: Path | str, hallazgos: list[Hallazgo]) -> list[tuple[str, str]]:
    """Datos detectados que siguen en el texto del archivo censurado (celda 18)."""
    doc = pymupdf.open(salida)
    texto = "".join(p.get_text() for p in doc)
    doc.close()
    return [(t, v) for _, t, v in hallazgos if v in texto]


def barrido_final(salida: Path | str) -> dict[str, list[str]]:
    """Vuelve a pasar los patrones sobre el archivo ya censurado (celda 18)."""
    doc = pymupdf.open(salida)
    texto = "".join(p.get_text() for p in doc)
    doc.close()
    restos = {}
    for tipo, patron in PATRONES.items():
        encontrados = re.findall(patron, texto)
        if encontrados:
            restos[tipo] = encontrados
    return restos


# ---------------------------------------------------------------------------


def procesar(archivo: Archivo, manifiesto: Manifiesto, carpeta: Path, detalles: dict[str, Any]) -> ResultadoArchivo:
    entrada = Path(manifiesto.raiz) / archivo.ruta
    destino = carpeta / archivo.ruta
    if entrada.suffix.lower() != ".pdf":
        return ResultadoArchivo(entrada=archivo.ruta, salida=None, error="no_soportado")
    try:
        if not tiene_texto(entrada):
            return ResultadoArchivo(entrada=archivo.ruta, salida=None, error="sin_texto")
        hallazgos = detectar(entrada) + detectar_nombres(entrada, manifiesto.lista_nombres)
        destino.parent.mkdir(parents=True, exist_ok=True)
        censuras = censurar(entrada, destino, hallazgos)
        with pymupdf.open(destino) as doc:
            n_paginas = doc.page_count
    except Exception as err:  # noqa: BLE001 - el informe registra el tipo de excepción
        destino.unlink(missing_ok=True)
        return ResultadoArchivo(entrada=archivo.ruta, salida=None, error=f"excepcion:{type(err).__name__}")

    # Lo que el propio cuaderno informaría (celda 18); solo informativo.
    try:
        verificacion: dict[str, Any] = {
            "detectados": len(hallazgos),
            "fugas_segun_cuaderno": len(verificar(destino, hallazgos)),
            "barrido_final": barrido_final(destino),
        }
    except Exception as err:  # noqa: BLE001
        verificacion = {"detectados": len(hallazgos), "error_verificacion": type(err).__name__}
    detalles.setdefault("verificacion_cuaderno", {})[archivo.ruta] = verificacion
    return ResultadoArchivo(entrada=archivo.ruta, salida=archivo.ruta, censuras=censuras, paginas_procesadas=n_paginas)
