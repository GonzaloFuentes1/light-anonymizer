"""The user's exceptions list (D10): RUTs of institutions, phones and 600/800 numbers.

They are still found and still shown: recall over precision, so a validation never discards a
finding. A finding whose every datum is on the list becomes an *optional* finding that starts
"suggested" (shown, not applied), like the other URLs of D12, with ``optional_reason`` set to
``"exception"``; the reviewer censors it or leaves it visible, and the audit report records it.

One value per line. Values are compared normalized: a RUT by its digits and check digit (no dots
or dash, the check digit in uppercase), a phone by its digits without the country code (+56 or
0056) or the old trunk 0. An entry written as a RUT (with a dash before the check digit, a K, or
the dots of a RUT) is only a RUT; one written as a phone (with a + or parentheses, spaces or
dashes between digit groups) is only a phone; bare digits can be either. A RUT found in a
document is compared only with the RUTs of the list, and a phone only with its phones.

A zone is left unapplied only when nothing but listed values, URLs that are not personal and one
label phrase right before each value ("RUT:", "Fono", "Mesa central:"...) would stay visible: an
OCR zone is a whole line, and a context value can hold more than the pattern found, so any other
word, initial or digit (a name, a direct line no pattern takes) keeps the zone applied. Label
words that are also names (Rut, Mesa) count only before a colon, or as "RUT" in capitals.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from anonymizer.engine.common import now_iso
from anonymizer.engine.model import Finding, HistoryEntry
from anonymizer.engine.patterns import norm
from anonymizer.engine.text import dedup_spans, detect_spans, is_personal_url

REASON = "exception"
NOTE = "En tu lista de excepciones"  # Spanish, shown next to the finding
# Only these types can be on the list; a zone that also covers a name, an e-mail or an address is
# never an exception.
TYPES = ("rut", "phone")

_DASH = r"\-‐‑‒–—−"  # for character classes: the hyphen is escaped
_RUT_DV = re.compile(rf"[{_DASH}]\s*[\dkK]\s*$")  # a dash right before a one-character check digit
_DOTTED = re.compile(rf"^\s*\d{{1,2}}\.\d{{3}}\.\d{{3}}\s*[{_DASH}]?\s*[\dkK]\s*$")  # 12.345.678-5
_RUT_CHARS = re.compile(rf"[\d\s.{_DASH}kK]+")
_PHONE_CHARS = re.compile(rf"[\d\s.{_DASH}+()]+")
# A finding whose text is only a value (digits and separators), with no label or words around it.
_VALUE_ONLY = re.compile(rf"[\d\s.,·+(){_DASH}kKxX]+")
# Label phrases that may stay visible right before a listed value (words in lowercase, without
# accents). Any other text keeps the zone applied: recall comes first.
LABEL_PHRASES = frozenset(
    tuple(phrase.split())
    for phrase in """
rut|rut n|n rut|r u t|run|rut del servicio|rut de la institucion|rol unico tributario
fono|fonos|telefono|telefonos|tel|telf|cel|celular|movil|whatsapp|fax|contacto|fono contacto|telefono contacto
mesa central|fono mesa central|telefono mesa central|mesa de ayuda|linea gratuita|call center|central telefonica
atencion ciudadana|informaciones|oficina de partes|numero|nro|n|web|sitio web|pagina web
""".replace("\n", "|").split("|")
    if phrase.strip()
)
# Words of those phrases that are also names: they count as a label only before a colon, or as the
# uppercase "RUT" ("Rut Mesa" is a person; "Mesa central:" and "RUT N°" are labels).
_NAME_LIKE = frozenset({"rut", "mesa"})


def rut_key(text: str) -> str | None:
    """``"72.123.456-8"`` -> ``"721234568"``; None when it cannot be a RUT (OCR's X is read as K)."""
    key = re.sub(r"[^\dkKxX]", "", text).upper().replace("X", "K")
    return key if re.fullmatch(r"\d{6,9}[\dK]", key) else None


def phone_key(text: str) -> str | None:
    """``"+56 600 123 4567"`` -> ``"6001234567"``; None when it cannot be a Chilean phone."""
    digits = re.sub(r"\D", "", text)
    if digits.startswith("0056"):
        digits = digits[4:]
    elif digits.startswith("56") and len(digits) >= 10:  # the same rules as ``patterns.phones``
        digits = digits[2:]
    if len(digits) == 10 and digits.startswith("0"):
        digits = digits[1:]  # old trunk prefix: 0 2 2345 6789
    return digits if 8 <= len(digits) <= 10 else None


def parse(entry: str) -> tuple[str | None, str | None]:
    """(RUT key, phone key) of one line of the list; (None, None) when it is neither."""
    text = entry.strip()
    if not text or not any(c.isdigit() for c in text):
        return None, None
    as_rut = "k" in text.lower() or _RUT_DV.search(text) is not None or _DOTTED.search(text) is not None
    if as_rut:
        return (rut_key(text) if _RUT_CHARS.fullmatch(text) else None), None
    if not _PHONE_CHARS.fullmatch(text):
        return None, None
    bare = text.isdigit()
    return (rut_key(text) if bare else None), phone_key(text)


def clean(entries: Iterable[str]) -> tuple[list[str], list[str]]:
    """(entries to keep, one per value, as the user wrote them; entries that are not a RUT or a phone)."""
    kept, invalid, seen = [], [], set()
    for entry in entries:
        value = re.sub(r"\s+", " ", str(entry)).strip()[:60]
        if not value:
            continue
        key = parse(value)
        if key == (None, None):
            invalid.append(value)
        elif key not in seen:
            seen.add(key)
            kept.append(value)
    return kept, invalid


@dataclass(frozen=True)
class Exceptions:
    ruts: frozenset[str]
    phones: frozenset[str]

    @classmethod
    def from_entries(cls, entries: Iterable[str]) -> Exceptions:
        keys = [parse(e) for e in entries]
        return cls(frozenset(r for r, _ in keys if r), frozenset(p for _, p in keys if p))

    def __bool__(self) -> bool:
        return bool(self.ruts or self.phones)

    def matches(self, value: str, type_: str) -> bool:
        """``value``, a RUT or a phone (``type_``) in any format, is on the list as that type."""
        if type_ == "rut":
            return rut_key(value) in self.ruts
        if type_ == "phone":
            return phone_key(value) in self.phones
        return False


def is_label(gap: str) -> bool:
    """``gap``, the text right before a listed value, is nothing, punctuation or one label phrase."""
    if any(c.isdigit() for c in gap):
        return False
    words = tuple(re.findall(r"[^\W\d_]+", norm(gap)))
    if not words:
        return True
    if words not in LABEL_PHRASES:
        return False
    if _NAME_LIKE.isdisjoint(words):
        return True
    return re.search(r":\s*$", gap) is not None or re.search(r"\bRUT\b", gap) is not None


def excepted(finding: Finding, exc: Exceptions, name_list: tuple[str, ...]) -> bool:
    """Everything ``finding`` would leave visible is on the list, a URL that is not personal (D12)
    or a label word.

    The text of a finding is its value (text layer, also the value of a context rule) or a whole
    OCR line. The data are found again in it with the same detectors: each must be a listed value
    or a URL that is not personal, with nothing but one label phrase before it (``is_label``) and
    nothing but punctuation after the last one. A name, an initial, an e-mail, a RUT that is not
    listed or digits no pattern takes keep the zone applied.
    """
    if finding.optional or finding.type not in TYPES or not finding.text:
        return False
    text = finding.text
    spans = dedup_spans(detect_spans(text, name_list, ocr=True))
    if not spans:  # a value the patterns do not take (a context rule found it): only digits count
        return _VALUE_ONLY.fullmatch(text) is not None and exc.matches(text, finding.type)
    listed = False
    end = 0
    for type_, a, b, _ in sorted(spans, key=lambda s: s[1]):
        value = text[a:b]
        if type_ in TYPES and exc.matches(value, type_):
            listed = True
        elif not (type_ == "url" and not is_personal_url(value, name_list)):
            return False
        if a >= end and not is_label(text[end:a]):
            return False
        end = max(end, b)
    return listed and re.search(r"[^\W_]", text[end:]) is None  # after the last value: punctuation only


def apply(findings: list[Finding], entries: Iterable[str], name_list: Iterable[str] = ()) -> int:
    """Turns the findings covered by the list into optional suggestions. Returns how many changed."""
    exc = Exceptions.from_entries(entries)
    if not exc:
        return 0
    names = tuple(name_list)
    changed = 0
    for f in findings:
        if not excepted(f, exc, names):
            continue
        f.optional = True
        f.optional_reason = REASON
        if f.status == "proposed":
            f.status = "suggested"
            f.history = [HistoryEntry(at=f.history[0].at if f.history else now_iso(), action="suggested")]
        changed += 1
    return changed
