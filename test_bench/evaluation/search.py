"""Search for values (canaries) and their critical fragments in extracted text and in bytes.

Every manifest value becomes one or more *needles*: the normalized form of the full value and
the critical fragments defined in ``docs/metrics.md``:

- RUT: body without the check digit (if it has at least 7 digits).
- Phone: last 7 digits.
- Email: local part plus ``@`` (or, in formats without ``@``, everything before the domain).
- Name: joined surnames (what comes before the comma, or the last two words).
- Address: street and number (up to the first number).

Every needle belongs to a *kind* that says how the text being searched (the *haystack*) is
normalized:

- ``digits``: uppercase and without separators between digits (``12.345.678-5`` -> ``123456785``).
  Only separators *between* digits are removed, so numbers of an arbitrary binary are not joined.
- ``compact``: without spaces and lowercase (emails and URLs).
- ``general``: ``schema.normalize`` (no accents, lowercase, collapsed spaces).

Bytes are turned into text with four encodings (UTF-8, Latin-1, UTF-16 BE and UTF-16 LE)
extracting only the printable runs (like ``strings``), so that searching a file of several MB
does not trigger random matches inside compressed data.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

from test_bench.schema import normalize

KINDS = ("digits", "compact", "general")

# Separators that may appear between the digits of a RUT or phone. Includes Unicode dashes and
# the C1 control characters (0x80-0x9f), which is how the WinAnsi and PDFDocEncoding en dash
# looks when decoded as Latin-1.
_DASHES = "".join(chr(c) for c in range(0x2010, 0x2016)) + chr(0x2212)  # Unicode dashes and minus sign
_DIGIT_SEPARATORS = r"[\s.,\-" + _DASHES + r"_()+/\x80-\x9f]"
_JOIN_DIGITS = re.compile(rf"(?<=[0-9K]){_DIGIT_SEPARATORS}{{1,3}}(?=[0-9K])")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Needle:
    """A normalized string that must not appear in the output."""

    form: str
    kind: str  # digits | compact | general
    part: str  # "value" or the name of the critical fragment
    # Alternative regular expression for values with non-ASCII letters: each one is accepted as
    # 0 to 2 arbitrary characters, because some writers replace them with "?", drop them or
    # leave them as UTF-8 mojibake read as Latin-1 ("MarÃ + soft hyphen + a").
    pattern: str | None = None
    # Forms as they were written (uppercase), for digit needles searched in binary sources,
    # where joining digits separated by spaces would give random matches.
    literals: tuple[str, ...] = ()

    def found_in(self, haystack: str) -> bool:
        if self.form in haystack:
            return True
        return self.pattern is not None and _regex(self.pattern).search(haystack) is not None


# Combining marks (accents, diaereses...): the ones ``schema.normalize`` removes in Latin text.
_COMBINING_RANGES = ((0x0300, 0x036F), (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF), (0xFE20, 0xFE2F))
_COMBINING = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _COMBINING_RANGES) + "]")


def normalize_general(text: str) -> str:
    """Same as ``schema.normalize`` for names and text, but without a Python loop (MB-sized haystacks)."""
    t = _COMBINING.sub("", unicodedata.normalize("NFKD", text)).casefold()
    return " ".join(t.split())


def normalize_haystack(text: str, kind: str) -> str:
    """Normalizes a text to be searched (haystack) according to the needle kind."""
    if kind == "digits":
        return _JOIN_DIGITS.sub("", text.upper())
    if kind == "compact":
        return _SPACES.sub("", text).casefold()
    return normalize_general(text)


_WITHOUT_DV = re.compile(r"[\s\-" + _DASHES + r"]*[0-9K]\s*$")


def _last_digits(text: str, n: int) -> str:
    """Suffix of ``text`` (as written) that holds its last ``n`` digits."""
    seen = 0
    for i in range(len(text) - 1, -1, -1):
        if text[i].isdigit():
            seen += 1
            if seen == n:
                return text[i:]
    return text


def _digits(value: str) -> str:
    return "".join(c for c in value.upper() if c.isdigit() or c == "K")


@lru_cache(maxsize=4096)
def _regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _general_needle(text: str, part: str) -> Needle:
    form = normalize("text", text)
    pattern = None
    if not text.isascii():
        clean = " ".join(unicodedata.normalize("NFC", text).split()).casefold()
        pattern = "".join(re.escape(c) if c.isascii() else ".{0,2}" for c in clean)
    return Needle(form, "general", part, pattern)


def _is_initial(word: str) -> bool:
    clean = word.strip(".,;")
    return len(clean) <= 2 or word.endswith(".")


def surnames(value: str) -> str | None:
    """Surnames of a name written in any of the usual variants."""
    if "," in value:
        candidate = value.split(",")[0]
    else:
        words = value.split()
        if len(words) < 3:
            return None
        candidate = " ".join(words[-2:])
    words = candidate.split()
    if len(words) < 2 or any(_is_initial(p) for p in words):
        return None
    return candidate


def street_number(value: str) -> str | None:
    """Street and number of an address (up to and including the first number)."""
    m = re.match(r"^(\D*?\S*\d+)", value)
    if not m or not re.search(r"[^\W\d]", m.group(1)):
        return None
    return m.group(1)


def needles(type: str, value: str | None) -> list[Needle]:
    """Needles (full value and critical fragments) of a manifest element."""
    if not value or type in ("face", "signature"):
        return []
    out: list[Needle] = []
    if type == "rut":
        d = _digits(normalize("rut", value))
        written = value.upper().strip()
        if d:
            out.append(Needle(d, "digits", "value", literals=(written,)))
        if len(d) - 1 >= 7:
            body = _WITHOUT_DV.sub("", written)
            out.append(Needle(d[:-1], "digits", "rut_body", literals=(body,)))
    elif type == "phone":
        d = "".join(c for c in value if c.isdigit())
        written = value.upper().strip()
        if d:
            out.append(Needle(d, "digits", "value", literals=(written,)))
        if len(d) > 7:
            out.append(Needle(d[-7:], "digits", "last7", literals=(_last_digits(written, 7),)))
    elif type in ("email", "url", "qr"):
        c = normalize_haystack(value, "compact")
        out.append(Needle(c, "compact", "value"))
        if type == "email":
            if "@" in c:
                out.append(Needle(c.split("@")[0] + "@", "compact", "local@"))
            else:
                m = re.match(r"^(.+?)[a-z0-9-]+(?:\.[a-z0-9-]+)+$", c)
                if m:
                    out.append(Needle(m.group(1), "compact", "local@"))
    else:
        out.append(_general_needle(value, "value"))
        fragment = surnames(value) if type == "name" else street_number(value) if type == "address" else None
        if fragment:
            part = "surnames" if type == "name" else "street_number"
            out.append(_general_needle(fragment, part))
    # no empty or repeated needles
    seen: set[tuple[str, str]] = set()
    unique = []
    for a in out:
        if a.form and (a.form, a.kind) not in seen:
            seen.add((a.form, a.kind))
            unique.append(a)
    return unique


def metadata_needles(value: str | None, type: str | None = None) -> list[Needle]:
    """Needles of a metadata canary: the full value in the three normalizations."""
    if not value:
        return []
    if type and type not in ("text",):
        base = needles(type, value)
        if base:
            return base
    out = [
        _general_needle(value, "value"),
        Needle(normalize_haystack(value, "compact"), "compact", "value"),
    ]
    d = _digits(value)
    if len(d) >= 7 and len(d) >= 0.6 * len(value.replace(" ", "")):
        out.append(Needle(d, "digits", "value", literals=(value.upper().strip(),)))
    return [a for a in out if a.form]


# ---------------------------------------------------------------------------
# Haystack: texts to search, grouped by origin, normalized only once per kind.
# ---------------------------------------------------------------------------


@dataclass
class Haystack:
    """Collection of texts of a file, grouped by origin (``text_pymupdf``, ``bytes``...)."""

    groups: dict[str, list[str]] = field(default_factory=dict)
    binaries: set[str] = field(default_factory=set)
    _cache: dict[tuple[str, str], str] = field(default_factory=dict)

    def add(self, origin: str, text: str | Iterable[str], binary: bool = False) -> None:
        """Adds texts to an origin. ``binary``: runs taken from raw bytes (content streams,
        image data...), full of numbers separated by spaces; there digits are not joined and
        digit needles are searched as they were written."""
        texts = [text] if isinstance(text, str) else list(text)
        texts = [t for t in texts if t]
        if binary:
            self.binaries.add(origin)
        if texts:
            self.groups.setdefault(origin, []).extend(texts)
            for kind in KINDS:
                self._cache.pop((origin, kind), None)

    def _normalized(self, origin: str, kind: str) -> str:
        key = (origin, kind)
        if key not in self._cache:
            # \x00 separates distinct texts: it is neither a digit separator nor a space
            if kind == "digits" and origin in self.binaries:
                self._cache[key] = "\n\x00\n".join(t.upper() for t in self.groups[origin])
            else:
                self._cache[key] = "\n\x00\n".join(normalize_haystack(t, kind) for t in self.groups[origin])
        return self._cache[key]

    def search(self, needles_: Iterable[Needle], origins: Iterable[str] | None = None) -> list[tuple[Needle, str]]:
        """Returns the (needle, origin) pairs found."""
        found = []
        items = list(needles_)
        for origin in origins if origins is not None else list(self.groups):
            if origin not in self.groups:
                continue
            binary = origin in self.binaries
            for a in items:
                haystack = self._normalized(origin, a.kind)
                if binary and a.kind == "digits":
                    hit = any(x and x in haystack for x in a.literals)
                else:
                    hit = a.found_in(haystack)
                if hit:
                    found.append((a, origin))
        return found

    def contains(self, needles_: Iterable[Needle], origins: Iterable[str] | None = None) -> bool:
        return bool(self.search(needles_, origins))


# ---------------------------------------------------------------------------
# Bytes -> texts in several encodings
# ---------------------------------------------------------------------------

_RUN_LATIN1 = re.compile(rb"[\t\n\r\x20-\x7e\x80-\xff]{4,}")
_RUN_UTF8 = re.compile(rb"(?:[\t\n\r\x20-\x7e]|[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}){4,}")
# UTF-16: printable Latin-1 plus the dashes U+2010..U+2015 (en dash of the RUT)
_RUN_UTF16LE = re.compile(rb"(?:[\t\n\r\x20-\x7e\xa0-\xff]\x00|[\x10-\x15]\x20){4,}")
_RUN_UTF16BE = re.compile(rb"(?:\x00[\t\n\r\x20-\x7e\xa0-\xff]|\x20[\x10-\x15]){4,}")


def texts_from_bytes(data: bytes) -> list[str]:
    """Printable runs of ``data`` decoded as UTF-8, Latin-1, UTF-16 BE and UTF-16 LE."""
    if not data:
        return []
    out = []
    latin = [m.group().decode("latin-1") for m in _RUN_LATIN1.finditer(data)]
    if latin:
        out.append("\n".join(latin))
    if re.search(rb"[\x80-\xff]", data):
        utf8 = [m.group().decode("utf-8", "ignore") for m in _RUN_UTF8.finditer(data)]
        if utf8:
            out.append("\n".join(utf8))
    if b"\x00" in data:
        le = [m.group().decode("utf-16-le", "ignore") for m in _RUN_UTF16LE.finditer(data)]
        be = [m.group().decode("utf-16-be", "ignore") for m in _RUN_UTF16BE.finditer(data)]
        out.extend("\n".join(x) for x in (le, be) if x)
    return out


def decode_string(data: bytes) -> list[str]:
    """Decodes a PDF string or a metadata field: UTF-16/UTF-8 BOM, then UTF-8, then Latin-1."""
    if data.startswith(b"\xfe\xff"):
        return [data[2:].decode("utf-16-be", "ignore")]
    if data.startswith(b"\xff\xfe"):
        return [data[2:].decode("utf-16-le", "ignore")]
    if data.startswith(b"\xef\xbb\xbf"):
        return [data[3:].decode("utf-8", "ignore")]
    out = []
    try:
        out.append(data.decode("utf-8"))
    except UnicodeDecodeError:
        out.append(data.decode("latin-1"))
    if b"\x00" in data and len(data) >= 4:
        out.append(data.decode("utf-16-be", "ignore"))
        out.append(data.decode("utf-16-le", "ignore"))
    return out
