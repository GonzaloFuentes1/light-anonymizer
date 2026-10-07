"""Names: the user's list (with word order and accent variants) and the given-name dictionary.

The dictionary rule finds full names that are not on the list (signatures, table cells, e-mail
headers): sequences of 2 to 5 capitalized words starting with a common given name.

The dictionary is ``data/given_names.txt``, the given names registered at least 100 times in Chile
from 1920 to 2021 according to the Servicio de Registro Civil e Identificación (the "guaguas"
dataset, CC0; rebuilt with ``scripts/build_given_names.py``), merged with a short hand-made list.
"""

from __future__ import annotations

import re
from functools import lru_cache

from anonymizer.engine.patterns import norm, normalize_1to1
from anonymizer.paths import resource

GIVEN_NAMES_FILE = resource("anonymizer", "engine", "data", "given_names.txt")
# The first, hand-made list of common given names in Chile (lowercase, without accents). It
# includes every given name the test bench's generator uses (``test_bench.fake_data``; a test keeps
# both in sync).
_HAND_MADE = """
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
"""
GIVEN_NAMES = frozenset(_HAND_MADE.split()) | frozenset(GIVEN_NAMES_FILE.read_text(encoding="utf-8").split())
# Given names that are also ordinary words (virtues, feasts, flowers, colours...) and start
# capitalized lines of public documents ("Paz Ciudadana", "Luz Verde", "Rosa Mística"). They do
# not start a name by themselves: only after don/doña/Sr./Sra. or when another given name follows
# ("Rosa María Pérez"). After a given name they are part of it ("María Paz Pérez").
AMBIGUOUS_GIVEN_NAMES = frozenset(
    norm(p)
    for p in """
paz luz sol mar flor flora rosa gloria victoria mercedes pilar esperanza consuelo dolores rocío belén fe amparo socorro
asunción ascensión encarnación natividad purísima trinidad soledad milagro milagros remedios piedad gracia caridad
constancia prudencia inocencia felicidad libertad justa justo digna digno humilde modesta modesto amable amada amado
perfecto previsto atractiva dilema delicia dulce vital primitiva primitivo perpetua custodia tránsito trancito reina
pastor pastora diosa nirvana olvido segunda segundo máxima máximo bienvenida bienvenido ella cruz reyes rey unidad
alba alma almendra amapola ámbar azucena azul brisa camelia canela celeste cielo coral cristal esmeralda estrella luna
magnolia miel nieve nieves oliva orquídea paloma perla rubí selva violeta aurora
""".split()
)
# Places, dates and labels that are also given names ("Santa María", "Santiago Centro", "Rut
# Representante"): they start a name only after don/doña/Sr./Sra.
PLACE_GIVEN_NAMES = frozenset(
    norm(p)
    for p in """
santiago concepción valdivia serena lautaro galvarino ángeles américa áfrica argentina francia italia grecia irlanda
bélgica venecia galicia roma parís siria libia pacífico santa santo santos abril domingo rut
""".split()
)
_NAME_WORD = re.compile(r"[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+")
# Capitalized words that follow names in documents but are not names (positions, places, headings,
# contact labels, months and days).
NOT_NAME = frozenset(
    norm(p)
    for p in """
firma electronica avanzada funcionaria funcionario jefe jefa director directora departamento unidad gobierno regional
santiago region chile informe anexo acta fecha nombre institucion correo telefono division seccion gabinete asesor
asesora coordinador coordinadora encargado encargada cargo profesional servicio municipal municipalidad estadio
nacional provincial comunal ministerio ministro ministra subsecretaría subsecretario secretaría secretaria secretario
seremi programa proyecto oficina área comité consejo consejero consejera presidente presidenta vicepresidente alcalde
alcaldesa gobernador gobernadora delegado delegada delegación abogado abogada analista ingeniero ingeniera técnico
técnica administrativo administrativa rut run email mail tel fono celular fax dirección domicilio comuna provincia
avenida pasaje población universidad escuela liceo colegio hospital
enero febrero marzo mayo junio agosto septiembre setiembre octubre noviembre diciembre
lunes martes miércoles jueves viernes sábado
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
    Ambiguous given names ("Paz", "Rosa", "Santiago") start one only after those honorifics; the
    word-like ones also when another given name follows or when the line is just a full name.
    """
    spans = []
    for start, content in _lines_with_offset(text):
        words = list(_NAME_WORD.finditer(content))
        short = len(content.split()) <= 7
        honorifics = [m.end() for m in _HONORIFIC.finditer(content)]
        i = 0
        while i < len(words):
            p = words[i]
            honorific = p.start() in honorifics
            if (short or honorific) and norm(p.group(0)) in GIVEN_NAMES:
                j = i
                while (
                    j + 1 < len(words)
                    and content[words[j].end() : words[j + 1].start()].strip() == ""
                    and norm(words[j + 1].group(0)) not in NOT_NAME
                    and j + 1 - i < 5
                ):
                    j += 1
                if honorific or (j > i and _starts_name(content, words, i, j)):
                    spans.append((start + p.start(), start + words[j].end()))
                    i = j + 1
                    continue
            i += 1
    return spans


def _starts_name(content: str, words: list[re.Match[str]], i: int, j: int) -> bool:
    """Whether the capitalized words i..j (i < j) of a line, the first a given name, are a name."""
    first, second = norm(words[i].group(0)), norm(words[i + 1].group(0))
    if first in NOT_NAME or first in PLACE_GIVEN_NAMES:
        return False
    if first not in AMBIGUOUS_GIVEN_NAMES:
        return True
    if second in GIVEN_NAMES and second not in AMBIGUOUS_GIVEN_NAMES | PLACE_GIVEN_NAMES:
        return True  # "Rosa María Pérez"
    # The whole line is a given name and two or more surnames ("Rosa Parra Contreras").
    return j - i >= 2 and not content[: words[i].start()].strip() and not content[words[j].end() :].strip(" \t,.;:")


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
