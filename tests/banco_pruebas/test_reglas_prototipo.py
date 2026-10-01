"""Casos de las reglas de contexto del prototipo (todos los textos son inventados)."""

import pytest

from banco_pruebas.lineas_base import reglas
from banco_pruebas.lineas_base.prototipo import buscar

L = reglas.Linea


def _hallados(texto, lista=(), **kw):
    return [(tipo, texto[a:b]) for tipo, a, b in buscar(texto, lista, **kw)]


@pytest.mark.parametrize(
    "texto",
    [
        "La obra se adjudicó por $4.890.290.519 a una empresa constructora.",
        "se aprobó un aporte de $ 2.310.450.000 para el programa",
        "Monto bruto: $1.234.000.-",
        "un presupuesto de 345.120.000 pesos",
    ],
)
def test_montos_no_son_telefonos_ni_rut(texto):
    assert not [h for h in _hallados(texto) if h[0] in ("telefono", "rut")]


def test_telefono_sigue_detectandose():
    assert ("telefono", "+56 9 8123 4567") in _hallados("Fono: +56 9 8123 4567")


URL_LARGA = "https://www.goreficticio.cl/noticias/2026/01/15/titulo-de-una-noticia-muy-\nlarga-de-ejemplo/\n2. Otra"


def test_url_institucional_no_se_censura_y_la_personal_si():
    assert not [h for h in _hallados(URL_LARGA, todas_url=False) if h[0] == "url"]
    assert ("url", "https://teams.microsoft.com/meet/123?p=abc") in _hallados(
        "https://teams.microsoft.com/meet/123?p=abc", todas_url=False
    )
    assert any(t == "url" for t, _ in _hallados("https://ejemplo.cl/ficha?rut=12345678-5", todas_url=False))


def test_url_cortada_en_dos_lineas_se_censura_completa():
    urls = [v for t, v in _hallados(URL_LARGA, todas_url=True) if t == "url"]
    assert urls and urls[0].endswith("larga-de-ejemplo/")


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("PEDRO ANDRES ROJAS TAPIA", "PEDRO ANDRES ROJAS TAPIA"),
        ("NOMBRE PEDRO ROJAS TAPIA", "PEDRO ROJAS TAPIA"),
    ],
)
def test_nombre_de_la_lista_se_amplia_a_los_nombres_de_pila(texto, esperado):
    assert ("nombre", esperado) in _hallados(texto, ("Rojas Tapia",))


@pytest.mark.parametrize("texto", ["ANA LUISA SOTO", "Ana Luisa Soto Fuentes", "De: Paula Andrea Vera Ríos"])
def test_firmantes_y_remitentes_fuera_de_la_lista(texto):
    assert any(t == "nombre" for t, _ in _hallados(texto))


@pytest.mark.parametrize(
    "texto",
    ["FUNCIONARIA", "4.2 Plaza Pública Juan Pérez Soto del sector norte", "Firma Electrónica Avanzada"],
)
def test_no_censura_cargos_ni_nombres_de_lugares_en_titulos(texto):
    assert not [h for h in _hallados(texto) if h[0] == "nombre"]


def test_correo_manuscrito_sin_punto_en_el_dominio():
    assert any(t == "correo" for t, _ in _hallados("rsoto@goreficticiocl", ocr=True))


def test_columnas_de_una_lista_de_asistencia():
    lineas = [
        L("Nombre", 100, 300, 160, 315),
        L("Institución", 260, 300, 340, 315),
        L("Teléfono", 380, 300, 440, 315),
        L("Correo", 500, 300, 560, 315),
        L("Firma", 640, 300, 690, 315),
        L("Rosa Pérez Vidal", 90, 330, 230, 348),
        L("Depto. Finanzas", 250, 330, 360, 348),
        L("rperez@xx", 470, 330, 600, 348),
        L("Juan Soto", 90, 360, 200, 378),
        L("Depto. Finanzas", 250, 360, 360, 378),
    ]
    tramos, rects = reglas.reglas_de_contexto(lineas, 800, 1100)
    censuradas = {i for _, i, _, _ in tramos}
    assert {5, 7, 8} <= censuradas  # nombres y correo
    assert not censuradas & {1, 6, 9}  # la institución no
    assert any(t == "firma" for t, *_ in rects)


def test_celda_de_formulario_nombre_valor():
    lineas = [
        L("NOMBRE", 30, 160, 100, 180),
        L("PEDRO ROJAS", 160, 160, 330, 180),
        L("CARGO", 30, 210, 90, 230),
        L("PROFESIONAL DE APOYO", 160, 210, 450, 230),
    ]
    tramos, _ = reglas.reglas_de_contexto(lineas, 800, 1100)
    assert [(t, i) for t, i, _, _ in tramos] == [("nombre", 1)]
