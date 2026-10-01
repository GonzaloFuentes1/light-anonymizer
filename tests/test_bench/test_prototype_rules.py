"""Cases of the prototype's context rules (all texts are made up)."""

import pytest

from anonymizer.engine import context, names
from anonymizer.engine.text import find_spans
from test_bench.baselines import rules
from test_bench.fake_data import FEMALE_NAMES, MALE_NAMES

L = context.Line


def _found(text, name_list=(), **kw):
    return [(type_, text[a:b]) for type_, a, b in find_spans(text, name_list, **kw)]


@pytest.mark.parametrize(
    "text",
    [
        "La obra se adjudicó por $4.890.290.519 a una empresa constructora.",
        "se aprobó un aporte de $ 2.310.450.000 para el programa",
        "Monto bruto: $1.234.000.-",
        "un presupuesto de 345.120.000 pesos",
    ],
)
def test_amounts_are_not_phones_or_ruts(text):
    assert not [h for h in _found(text) if h[0] in ("phone", "rut")]


def test_phone_still_detected():
    assert ("phone", "+56 9 8123 4567") in _found("Fono: +56 9 8123 4567")


LONG_URL = "https://www.goreficticio.cl/noticias/2026/01/15/titulo-de-una-noticia-muy-\nlarga-de-ejemplo/\n2. Otra"


def test_institutional_url_not_redacted_but_personal_one_is():
    assert not [h for h in _found(LONG_URL, all_urls=False) if h[0] == "url"]
    assert ("url", "https://teams.microsoft.com/meet/123?p=abc") in _found(
        "https://teams.microsoft.com/meet/123?p=abc", all_urls=False
    )
    assert any(t == "url" for t, _ in _found("https://ejemplo.cl/ficha?rut=12345678-5", all_urls=False))


def test_url_split_over_two_lines_redacted_whole():
    urls = [v for t, v in _found(LONG_URL, all_urls=True) if t == "url"]
    assert urls and urls[0].endswith("larga-de-ejemplo/")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("PEDRO ANDRES ROJAS TAPIA", "PEDRO ANDRES ROJAS TAPIA"),
        ("NOMBRE PEDRO ROJAS TAPIA", "PEDRO ROJAS TAPIA"),
    ],
)
def test_listed_name_expands_to_given_names(text, expected):
    assert ("name", expected) in _found(text, ("Rojas Tapia",))


@pytest.mark.parametrize("text", ["ANA LUISA SOTO", "Ana Luisa Soto Fuentes", "De: Paula Andrea Vera Ríos"])
def test_signers_and_senders_outside_the_list(text):
    assert any(t == "name" for t, _ in _found(text))


@pytest.mark.parametrize(
    "text",
    ["FUNCIONARIA", "4.2 Plaza Pública Juan Pérez Soto del sector norte", "Firma Electrónica Avanzada"],
)
def test_positions_and_place_names_in_titles_not_redacted(text):
    assert not [h for h in _found(text) if h[0] == "name"]


def test_handwritten_email_without_dot_in_domain():
    assert any(t == "email" for t, _ in _found("rsoto@goreficticiocl", ocr=True))


def test_attendance_list_columns():
    lines = [
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
    spans, rects = context.context_rules(lines, 800, 1100)
    redacted = {i for _, i, _, _ in spans}
    assert {5, 7, 8} <= redacted  # names and e-mail
    assert not redacted & {1, 6, 9}  # the institution is not
    assert any(t == "signature" for t, *_ in rects)


def test_form_cell_name_value():
    lines = [
        L("NOMBRE", 30, 160, 100, 180),
        L("PEDRO ROJAS", 160, 160, 330, 180),
        L("CARGO", 30, 210, 90, 230),
        L("PROFESIONAL DE APOYO", 160, 210, 450, 230),
    ]
    spans, _ = context.context_rules(lines, 800, 1100)
    assert [(t, i) for t, i, _, _ in spans] == [("name", 1)]


def test_dictionary_covers_the_generator_names():
    generated = {rules._norm(p) for n in [*FEMALE_NAMES, *MALE_NAMES] for p in n.split()}
    assert generated <= names.GIVEN_NAMES


def test_rules_module_still_reexports_the_engine():
    assert rules.context_rules is context.context_rules and rules.Line is context.Line
