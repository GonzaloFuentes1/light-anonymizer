"""Builds the given-name dictionary (``anonymizer/engine/data/given_names.txt``) from an open dataset.

Usage (from the repository root):
    uv run python scripts/build_given_names.py [--threshold 100]

Source: the "guaguas" dataset by Riva Quiroga, first names registered in Chile from 1920 to 2021
according to the Servicio de Registro Civil e Identificación (obtained through the Transparency
Portal), released under CC0 1.0 (https://github.com/rivaquiroga/guaguas, CRAN package ``guaguas``).
The plain CSV is downloaded from a fixed commit and its SHA-256 is checked, so the list can be
rebuilt byte for byte.

The registrations of every year and both sexes are added up per given name; a registration with
several words ("MARIA JOSE") counts for each of them; names are normalized like the engine does
(lowercase, without accents: ``anonymizer.engine.patterns.norm``), and the ones registered at least
``--threshold`` times in total are written, one per line, sorted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anonymizer.engine.patterns import norm  # noqa: E402

COMMIT = "2e4d4efc02eb1a346bb3370f9b64a44465e0faf4"  # rivaquiroga/guaguas, main on 2022-03-09 (v0.3.0)
URL = f"https://raw.githubusercontent.com/rivaquiroga/guaguas/{COMMIT}/data-raw/1920-2021.csv"
SHA256 = "ff253ffebbf3f3255bfac46a4f8dbf2f0ecb1d04bb1ae1ebaf93c469b23805e0"
DESTINATION = ROOT / "anonymizer" / "engine" / "data" / "given_names.txt"
# Chosen by measurement (October 2026): 100 registrations in a century keeps 4 353 names that
# cover 96.9 % of all registrations; lower thresholds add mostly rare spellings and stray words.
DEFAULT_THRESHOLD = 100
_SEPARATOR = re.compile(r"[\s,.\-]+")
_WORD = re.compile(r"[a-zñ]{2,}(?:'[a-zñ]+)?")


def download() -> bytes:
    request = urllib.request.Request(URL, headers={"User-Agent": "anonymizer-dev/0.1"})
    with urllib.request.urlopen(request, timeout=120) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"SHA-256 does not match: expected {SHA256}, got {digest}")
    return data


def count_names(data: bytes) -> Counter[str]:
    """Total registrations per normalized given name (all years, both sexes)."""
    totals: Counter[str] = Counter()
    for row in csv.DictReader(io.StringIO(data.decode("utf-8"))):
        for token in _SEPARATOR.split(row["nombre"]):
            name = norm(token)
            if _WORD.fullmatch(name):
                totals[name] += int(row["n"])
    return totals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the given-name dictionary from the guaguas dataset.")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="minimum total registrations")
    args = ap.parse_args(argv)
    totals = count_names(download())
    names = sorted(n for n, c in totals.items() if c >= args.threshold)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text("".join(f"{n}\n" for n in names), encoding="utf-8", newline="\n")
    covered = sum(totals[n] for n in names) / sum(totals.values())
    print(f"{len(names)} given names (>= {args.threshold} registrations, {covered:.1%} of all) -> {DESTINATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
