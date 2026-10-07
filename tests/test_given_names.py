"""The given-name dictionary (anonymizer/engine/data/given_names.txt) and its ambiguous words (texts made up)."""

from pathlib import Path

import pytest

from anonymizer.engine import names
from anonymizer.engine.text import find_spans

ROOT = Path(__file__).resolve().parents[1]


def _names(text: str) -> list[str]:
    return [text[a:b] for type_, a, b in find_spans(text, ()) if type_ == "name"]


def test_the_dictionary_file_is_sorted_normalized_and_loaded():
    path = ROOT / "anonymizer" / "engine" / "data" / "given_names.txt"
    assert names.GIVEN_NAMES_FILE == path
    entries = path.read_text(encoding="utf-8").splitlines()
    assert entries == sorted(set(entries))
    assert all(e == names.norm(e) and e.isalpha() for e in entries)
    assert len(entries) > 4000 and "alexandra" in entries
    assert set(entries) <= names.GIVEN_NAMES and "yasna" in names.GIVEN_NAMES  # the hand-made list stays


def test_the_executable_bundles_the_dictionary():
    spec = (ROOT / "packaging" / "light_anonymizer.spec").read_text(encoding="utf-8")
    assert '(str(ROOT / "anonymizer" / "engine" / "data"), "anonymizer/engine/data")' in spec


@pytest.mark.parametrize(
    "text",
    [
        "Alexandra Rojas Peña",  # signature line
        "Alexandra Rojas Peña\nAnalista\nDepartamento de Finanzas",  # signature block
        "Alexandra Rojas Peña    Analista    Depto. Finanzas",  # table row
        "Se reunió con doña Alexandra Rojas Peña en la oficina del servicio regional para revisar el informe.",
    ],
)
def test_a_given_name_of_the_dataset_starts_a_name(text):
    assert "Alexandra Rojas Peña" in _names(text)


@pytest.mark.parametrize(
    "text",
    [
        "Paz y orden",
        "Rosa de los vientos",
        "Paz Ciudadana",
        "Luz Verde",
        "Santa María",
        "Santiago Centro",
        "Concepción Talcahuano",
        "Rut Representante Legal",
        "Julio Agosto Septiembre",
    ],
)
def test_ambiguous_words_do_not_start_a_name_by_themselves(text):
    assert _names(text) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("María Paz Rojas", "María Paz Rojas"),  # after another given name
        ("Rosa María Soto", "Rosa María Soto"),  # followed by another given name
        ("Rosa Parra Contreras", "Rosa Parra Contreras"),  # a line that is just a full name
        (
            "Asistió doña Rosa Pérez Vidal a la reunión de coordinación del programa regional de fomento.",
            "Rosa Pérez Vidal",
        ),
        ("Juan Pérez Soto Presidente", "Juan Pérez Soto"),  # positions end the name
    ],
)
def test_ambiguous_words_inside_names(text, expected):
    assert expected in _names(text)
