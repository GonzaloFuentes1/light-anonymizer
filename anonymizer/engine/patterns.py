"""Patterns for RUT, e-mail, phone and URL, with the variants OCR produces.

Moved from the prototype (``test_bench/baselines/prototype.py``) without changing behaviour:
the test bench measures these exact expressions.
"""

from __future__ import annotations

import re
import unicodedata

_DASH = r"[-‐‑‒–—−]"
RUT = re.compile(rf"(?<![\d.,])\d{{1,3}}(?:[.,·\s]?\d{{3}}){{2}}\s?{_DASH}?\s?[\dkK](?![\w@])")
EMAIL = re.compile(
    r"[\w.+'-]+\s?(?:@|＠|©|\[at\]|\(at\)|\[arroba\]|\(arroba\)|\sarroba\s)\s?[\w-]+(?:\s?[.,]\s?[\w-]+)*\s?[.,]\s?[a-z]{2,4}\b",
    re.IGNORECASE,
)
URL = re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE)
# Anything with an at sign: OCR often loses the dot of the domain ("...@goreficticiocl").
LOOSE_EMAIL = re.compile(r"[\w.+'-]+[ \t]?[@＠][ \t]?[\w.,-]{2,}")
# In OCR the check-digit dash is sometimes read as a dot, comma or space, and the K as an X.
RUT_OCR = re.compile(r"(?<![\d])\d{1,3}(?:[.,·\s]?\d{3}){2}\s?[-‐‑‒–—−.,·]\s?[\dkKxX](?![\w@])")
# Digits with separators on a single line; the dot only between digits (not the full stop of a sentence).
_RUN = re.compile(r"[+(]?\d(?:[\d \t()+‐‑–—-]|\.(?=\d)){5,24}\d")
_PHONE_LABEL = re.compile(r"(?i)(fono|tel[eé]?f?|cel|m[oó]vil|whats|wsp|fax|contacto)[^\n]{0,20}$")
# Letters OCR confuses with digits, applied only to "words" that are mostly digits.
CONFUSIONS = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8", "Z": "2"}
)

# When an OCR line has several types, the first one in this order wins (kept from the prototype
# so that the reported types do not change).
TYPE_PRIORITY = ("email", "address", "name", "rut", "phone", "url")

# Personal URLs (social networks, meetings, shared files, identifiers in the query).
PERSONAL_URL = re.compile(
    r"(?i)(facebook|instagram|linkedin|twitter\.|//x\.com|tiktok|wa\.me|whatsapp|teams\.microsoft|zoom\.us|meet\.google"
    r"|drive\.google|docs\.google|dropbox|onedrive|1drv\.ms|sharepoint|wetransfer|forms\.gle|calendly)"
    r"|[?&](rut|run|id|token|key|pwd|p|user|usuario|email|mail|correo)="
)
_URL_CONTINUATION = re.compile(r"\n([\w\-./%?=&#~+:]+)(?=[ \t]*(?:\n|$))")


def rut_valid_by_shape(m: re.Match[str]) -> bool:
    """Avoids taking amounts or dates without a dash as a RUT: without a dash it requires 8-10 chars in a row."""
    text = m.group(0)
    if re.search(_DASH, text):
        return True
    return bool(re.fullmatch(r"\d{7,9}[\dkK]", re.sub(r"\s", "", text))) and "." not in text


def rut_check_digit(body: str) -> str:
    total, factor = 0, 2
    for digit in reversed(body):
        total += int(digit) * factor
        factor = 2 if factor == 7 else factor + 1
    rest = 11 - total % 11
    return {11: "0", 10: "K"}.get(rest, str(rest))


def rut_is_valid(text: str) -> bool:
    """The check digit of a RUT matches (the OCR's X is read as K)."""
    clean = re.sub(r"[^\dkKxX]", "", text).upper().replace("X", "K")
    if len(clean) < 2 or not clean[:-1].isdigit():
        return False
    return rut_check_digit(clean[:-1]) == clean[-1]


def ocr_variant(text: str) -> str:
    """``text`` with the letters OCR confuses with digits replaced, in words that are mostly digits."""
    return re.sub(
        r"\S+",
        lambda m: (
            m.group(0).translate(CONFUSIONS)
            if sum(c.isdigit() for c in m.group(0)) >= len(m.group(0)) / 2
            else m.group(0)
        ),
        text,
    )


def phones(text: str) -> list[tuple[int, int]]:
    """Chilean phone numbers (9 digits after the country code, or 8 after a phone label)."""
    output = []
    for m in _RUN.finditer(text):
        d = re.sub(r"\D", "", m.group(0))
        if d.startswith("0056"):
            d = d[4:]
        elif d.startswith("56") and len(d) >= 10:
            d = d[2:]
        if len(d) == 10 and d.startswith("0"):
            d = d[1:]
        ok = len(d) == 9 and d[0] in "23456789"
        if not ok and len(d) == 8 and _PHONE_LABEL.search(text[max(0, m.start() - 25) : m.start()]):
            ok = True
        if ok:
            output.append((m.start(), m.end()))
    return output


def complete_url(text: str, a: int, b: int) -> tuple[int, int]:
    """If the URL continues on the next line (PDF line break), includes that piece."""
    m = _URL_CONTINUATION.match(text, b)
    if m and ("/" in m.group(1) or "-" in m.group(1)) and " " not in m.group(1):
        return a, m.end(1)
    return a, b


def is_amount(text: str, a: int, b: int) -> bool:
    """Money figures or numbers with thousands grouping: they are neither phones nor RUTs."""
    before = text[max(0, a - 4) : a]
    after = text[b : b + 12].lower()
    value = text[a:b].strip()
    if "$" in before or "us$" in before.lower() or "clp" in before.lower():
        return True
    if re.match(r"\s*(millones|mil\b|pesos|uf\b|%|usd)", after):
        return True
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?", value))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def norm(t: str) -> str:
    """Lowercase without accents, trimmed (lengths may change)."""
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c)).casefold().strip()


def _strip_accents(c: str) -> str:
    base = unicodedata.normalize("NFKD", c)
    base = "".join(x for x in base if not unicodedata.combining(x))
    # [:1] again after casefold: "ß" folds to "ss", and the result must stay one character long.
    return (base[:1] or c).casefold()[:1] or c


def normalize_1to1(text: str) -> str:
    """Normalizes character by character (same length), so positions map back to the original."""
    return "".join(_strip_accents(c) if c.strip() else " " for c in text)
