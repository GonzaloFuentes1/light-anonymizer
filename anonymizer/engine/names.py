"""Names: the user's list (with word order and accent variants) and the given-name dictionary.

The dictionary rule finds full names that are not on the list (signatures, table cells, e-mail
headers): sequences of 2 to 5 capitalized words starting with a common given name.
"""

from __future__ import annotations

import re
from functools import lru_cache

from anonymizer.engine.patterns import norm, normalize_1to1

# Common given names in Chile (lowercase, without accents). It includes every given name the
# test bench's generator uses (``test_bench.fake_data``; a test keeps both in sync).
GIVEN_NAMES = frozenset(
    """
agustin alberto alejandra alejandro alfredo alicia alvaro amanda ana andrea andres angela angelica antonia antonio
arturo barbara beatriz benjamin bernardita blanca bruno camila carla carlos carmen carolina catalina cecilia cesar
claudia claudio constanza cristian cristina cristobal daniel daniela david denisse diego eduardo elena elizabeth
emilia enrique ernesto esteban eugenia eva fabian felipe fernanda fernando francisca francisco gabriel gabriela
gloria gonzalo graciela guillermo gustavo hector hernan hugo ignacia ignacio isabel isidora ivan ivonne jaime javier
javiera jessica joaquin jorge jose josefa juan julia julio karen karina katherine laura leonardo lorena loreto luis
luisa macarena manuel marcela marcelo marco margarita maria mariana mario marta martin matias mauricio maximiliano
miguel monica natalia nelson nicolas olga oscar pablo paola patricia patricio paula paulina pedro rafael ramon raul
rebeca ricardo roberto rocio rodrigo rosa rosario ruben sandra sara sebastian sergio silvia sofia solange sonia
susana tamara teresa tomas valentina valeria vanessa veronica vicente victor victoria viviana ximena yasna yolanda
""".split()
)
_NAME_WORD = re.compile(r"[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+")
# Capitalized words that follow names in documents but are not names (positions, places, headings).
NOT_NAME = frozenset(
    norm(p)
    for p in """
firma electronica avanzada funcionaria funcionario jefe jefa director directora departamento unidad gobierno regional
santiago region chile informe anexo acta fecha nombre institucion correo telefono division seccion gabinete asesor
asesora coordinador coordinadora encargado encargada cargo profesional servicio municipal municipalidad estadio
""".split()
)
_HONORIFIC = re.compile(r"(?i)\b(?:don|doña|sr\.?|sra\.?|srta\.?|señor|señora)\s+")


def is_address(entry: str) -> bool:
    """Entries of the list with a digit are addresses; the rest are people."""
    return any(ch.isdigit() for ch in entry)


@lru_cache(maxsize=64)
def list_patterns(name_list: tuple[str, ...]) -> list[tuple[str, re.Pattern[str]]]:
    """(type, pattern over ``normalize_1to1`` text) for every entry of the list and its variants.

    People with 3 or more words also match as "surnames given names", "surnames, given names",
    "first given name + first surname" and "surnames"; addresses also match up to the number.
    """
    output = []
    for entry in name_list:
        parts = normalize_1to1(entry).split()
        if not parts:
            continue
        variants = {" ".join(parts)}
        address = is_address(entry)
        if not address and len(parts) >= 3:
            surnames, given_names = parts[-2:], parts[:-2]
            variants |= {
                " ".join(surnames + given_names),
                " ".join(surnames) + ", " + " ".join(given_names),
                " ".join([given_names[0], surnames[0]]),
                " ".join(surnames),
            }
        if address:
            m = re.match(r"^(\D*?\d+)", " ".join(parts))
            if m:
                variants.add(m.group(1))
        for v in variants:
            body = r"[\s,]+".join(re.escape(t.strip(",")) for t in v.split())
            output.append(("address" if address else "name", re.compile(rf"(?<!\w){body}(?!\w)")))
    return output


@lru_cache(maxsize=64)
def list_words(name_list: tuple[str, ...]) -> frozenset[str]:
    """Standalone given names and surnames of the people of the list (no addresses)."""
    return frozenset(p for e in name_list if not is_address(e) for p in normalize_1to1(e).split() if len(p) >= 3)


def names_by_dictionary(text: str) -> list[tuple[int, int]]:
    """Sequences of 2 to 5 capitalized words that start with a known given name.

    Applied to short lines (signatures, cells, e-mail headers) and after "don/doña/Sr./Sra.".
    """
    spans = []
    for start, content in _lines_with_offset(text):
        words = list(_NAME_WORD.finditer(content))
        short = len(content.split()) <= 7
        honorifics = [m.end() for m in _HONORIFIC.finditer(content)]
        i = 0
        while i < len(words):
            p = words[i]
            allowed = short or p.start() in honorifics
            if allowed and norm(p.group(0)) in GIVEN_NAMES:
                j = i
                while (
                    j + 1 < len(words)
                    and content[words[j].end() : words[j + 1].start()].strip() == ""
                    and norm(words[j + 1].group(0)) not in NOT_NAME
                    and j + 1 - i < 5
                ):
                    j += 1
                if j > i or p.start() in honorifics:
                    spans.append((start + p.start(), start + words[j].end()))
                i = j + 1
            else:
                i += 1
    return spans


def expand_name(text: str, a: int, b: int) -> tuple[int, int]:
    """Extends a name of the list to the neighbouring capitalized words of the same line (given names)."""
    for _ in range(4):
        m = re.search(r"([A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+)[ \t]+$", text[max(0, a - 40) : a])
        if not m or norm(m.group(1)) in NOT_NAME:
            break
        a = a - (len(text[max(0, a - 40) : a]) - m.start(1))
    for _ in range(4):
        m = re.match(r"[ \t]+([A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+)", text[b : b + 40])
        if not m or norm(m.group(1)) in NOT_NAME:
            break
        b = b + m.end()
    return a, b


def _lines_with_offset(text: str) -> list[tuple[int, str]]:
    output, pos = [], 0
    for line in text.split("\n"):
        output.append((pos, line))
        pos += len(line) + 1
    return output
