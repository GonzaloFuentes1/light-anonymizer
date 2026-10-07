"""Personal data in a piece of text: patterns, the name list and the given-name dictionary.

Also the reasons a finding is marked doubtful (shown to the reviewer, in Spanish) and the
"needle" used to look for a finding's text elsewhere (propagation and leak check).
"""

from __future__ import annotations

import re
from functools import lru_cache

from anonymizer.engine import names, patterns
from anonymizer.engine.patterns import (
    EMAIL,
    LOOSE_EMAIL,
    RUT,
    RUT_OCR,
    TYPE_PRIORITY,
    URL,
    normalize_1to1,
    ocr_variant,
    phones,
    rut_is_valid,
    rut_valid_by_shape,
)

# Doubt reasons (Spanish, shown to the reviewer).
DOUBT_RUT = "El dígito verificador no coincide: revisa el original"
DOUBT_OCR = "Texto leído con baja confianza"
DOUBT_FACE = "Rostro pequeño o poco claro"
DOUBT_CONTEXT_NAME = "Nombre detectado por el contexto; no está en la lista"

OCR_MIN_SCORE = 0.75
FACE_MIN_SIDE_PX = 40
FACE_MIN_SCORE = 0.7

Span = tuple[str, int, int, str]  # type, start, end, detector
DETECTOR_PRIORITY = {"regex": 0, "name_list": 1, "context": 2}


def detect_spans(
    text: str,
    name_list: tuple[str, ...],
    ocr: bool = False,
    all_urls: bool = True,
    *,
    personal_urls: bool = True,
    given_names: bool = True,
) -> list[Span]:
    """Spans (type, start, end, detector) with personal data in ``text``.

    Detector: ``regex`` (RUT, e-mail, phone, URL), ``name_list`` (entries of the list and their
    variants, extended to neighbouring given names) or ``context`` (given-name dictionary).
    ``all_urls``: also the URLs that are not personal (``is_personal_url``); ``personal_urls``: the
    personal ones; ``given_names``: the given-name dictionary. RUT, e-mail and phone always run.
    """
    found: list[Span] = []
    variants = [text]
    if ocr:
        variants.append(ocr_variant(text))
    for t in variants:
        found += [("rut", m.start(), m.end(), "regex") for m in RUT.finditer(t) if rut_valid_by_shape(m)]
        if ocr:
            found += [("rut", m.start(), m.end(), "regex") for m in RUT_OCR.finditer(t)]
        found += [("phone", a, b, "regex") for a, b in phones(t)]
    found = [s for s in found if not patterns.is_amount(text, s[1], s[2])]
    found += [("email", m.start(), m.end(), "regex") for m in EMAIL.finditer(text)]
    found += [("email", m.start(), m.end(), "regex") for m in LOOSE_EMAIL.finditer(text)]
    for m in URL.finditer(text):
        a, b = patterns.complete_url(text, m.start(), m.end())
        personal = is_personal_url(text[a:b], name_list)
        if (personal and personal_urls) or (not personal and all_urls):
            found.append(("url", a, b, "regex"))
    normalized = normalize_1to1(text)
    for type_, pattern in names.list_patterns(name_list):
        for m in pattern.finditer(normalized):
            a, b = names.expand_name(text, m.start(), m.end()) if type_ == "name" else (m.start(), m.end())
            found.append((type_, a, b, "name_list"))
    if given_names:
        found += [("name", a, b, "context") for a, b in names.names_by_dictionary(text)]
    return found


def dedup_spans(spans: list[Span]) -> list[Span]:
    """Drops repeated spans and spans inside another one (their zone is inside the other's zone).

    For the same characters, the most specific detector stays: patterns, then the list, then context.
    """
    ordered = sorted(
        set(spans),
        key=lambda s: (s[1], -s[2], DETECTOR_PRIORITY.get(s[3], 9), TYPE_PRIORITY.index(s[0])
                       if s[0] in TYPE_PRIORITY else 99),
    )  # fmt: skip
    kept: list[Span] = []
    for s in ordered:
        if s[2] <= s[1]:
            continue
        if any(k[1] <= s[1] and s[2] <= k[2] for k in kept):
            continue
        kept.append(s)
    return kept


def find_spans(
    text: str, name_list: tuple[str, ...], ocr: bool = False, all_urls: bool = True
) -> list[tuple[str, int, int]]:
    """Spans (type, start, end) with personal data in ``text`` (the prototype's interface)."""
    return [(type_, a, b) for type_, a, b, _ in detect_spans(text, name_list, ocr=ocr, all_urls=all_urls)]


def is_personal_url(url: str, name_list: tuple[str, ...]) -> bool:
    """D12: social networks, meetings, shared files, identifiers in the query, or a piece of personal
    data inside. The other URLs (institutional sites, public documents) are optional findings."""
    return bool(patterns.PERSONAL_URL.search(url)) or has_data(url, name_list)


def has_data(url: str, name_list: tuple[str, ...]) -> bool:
    """The URL contains a piece of personal data (RUT, e-mail, phone or a name of the list)."""
    no_dashes = normalize_1to1(url.replace("-", " ").replace("_", " "))
    return bool(
        RUT.search(url)
        or LOOSE_EMAIL.search(url)
        or phones(url)
        or any(p.search(no_dashes) for _, p in names.list_patterns(name_list))
    )


def in_list(text: str, name_list: tuple[str, ...]) -> bool:
    """``text`` contains a person or address of the list."""
    if not name_list:
        return False
    normalized = normalize_1to1(text)
    return any(p.search(normalized) for _, p in names.list_patterns(name_list))


# Decided 2026-10-06: a RUT whose check digit does not match, written as a bare run of digits (no
# dots, no dash) and with no RUT label right before it, is often not a RUT (a folio, a code): it is
# an optional finding that starts unapplied, like the other URLs (D12).
OPTIONAL_RUT = "rut"
RUT_LABEL = re.compile(r"(?<![a-zñ])(?:r\.?\s?u\.?\s?[tn]\.?(?![a-zñ])|rol\s+[uú]nico)", re.IGNORECASE)
RUT_LABEL_REACH = 30  # characters before the number, on the same line, where a label counts
_BARE = re.compile(r"\d{6,9}[\dkKxX]")


def rut_suggested(text: str, a: int, b: int, value: str | None = None) -> bool:
    """The RUT at ``text[a:b]`` is only suggested (``OPTIONAL_RUT``): its check digit does not match,
    it is a bare run of digits, it is not also a phone number, and no "RUT", "RUN", "R.U.T." or
    "Rol Único" comes within ``RUT_LABEL_REACH`` characters before it on its line. ``value``: the number as read (an OCR
    variant of ``text[a:b]``, same length), if not ``text[a:b]``."""
    value = (text[a:b] if value is None else value).strip()
    if not _BARE.fullmatch(value) or rut_is_valid(value):
        return False
    if any(pa < b and a < pb for pa, pb in phones(text)):
        return False  # also a phone number (a mobile written without +56): personal data anyway
    start = max(text.rfind(chr(10), 0, a) + 1, a - RUT_LABEL_REACH)
    return not RUT_LABEL.search(text[start:a])


def has_rut_number(text: str, ocr: bool = False) -> bool:
    """``text`` holds a number shaped like a RUT (what a finding of type "rut" must contain)."""
    for t in (text, ocr_variant(text)) if ocr else (text,):
        if any(rut_valid_by_shape(m) for m in RUT.finditer(t)) or (ocr and RUT_OCR.search(t)):
            return True
    return False


def context_type(type_: str, value: str, name_list: tuple[str, ...], ocr: bool = False) -> str | None:
    """The type of a value a context rule found under ``type_`` ("RUT: ..." or a RUT column). A
    "rut" must hold a RUT-shaped number: otherwise the value keeps the type of another datum it
    holds (an e-mail, a phone, a name...), or is no finding (None: an empty value, a dash)."""
    if type_ != "rut" or has_rut_number(value, ocr):
        return type_
    others = [s[0] for s in dedup_spans(detect_spans(value, name_list, ocr=ocr)) if s[0] != "rut"]
    return min(others, key=TYPE_PRIORITY.index) if others else None


def rut_doubt(text: str | None) -> str | None:
    """``DOUBT_RUT`` when ``text`` has RUTs and none of them has a valid check digit."""
    if not text:
        return None
    candidates = [m.group(0) for t in (text, ocr_variant(text)) for m in (*RUT.finditer(t), *RUT_OCR.finditer(t))]
    if not candidates or any(rut_is_valid(c) for c in candidates):
        return None
    return DOUBT_RUT


def face_doubt(min_side_px: float, score: float) -> str | None:
    return DOUBT_FACE if min_side_px < FACE_MIN_SIDE_PX or score < FACE_MIN_SCORE else None


def ocr_doubt(score: float | None) -> str | None:
    return DOUBT_OCR if score is not None and score < OCR_MIN_SCORE else None


# ---------------------------------------------------------------------------
# Needles: the same text somewhere else
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def needle(text: str) -> re.Pattern[str] | None:
    """Pattern that finds ``text`` again in ``normalize_1to1`` text, whole words only.

    Separators between words may differ (spaces, line breaks, punctuation): the text extracted
    from a PDF changes its spacing when parts of the page are removed.
    """
    tokens = re.findall(r"\w+", normalize_1to1(text))
    if not tokens:
        return None
    return re.compile(r"(?<!\w)" + r"\W*".join(re.escape(t) for t in tokens) + r"(?!\w)")


def strong_needle(text: str) -> bool:
    """Texts specific enough to look for in the whole document (not just their page)."""
    tokens = re.findall(r"\w+", normalize_1to1(text))
    alnum = sum(len(t) for t in tokens)
    return alnum >= 6 and (len(tokens) >= 2 or any(c.isdigit() for c in text) or "@" in text)
