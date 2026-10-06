"""Generates long invented documents to exercise the scrolling review (not versioned).

Usage (from the repository root):
    uv run python scripts/generate_long_docs.py [--output test_data/long] [--seed 13]

Writes three files, all with invented data only (names, RUTs, emails and phones from
``test_bench.fake_data``) and the same bytes on every run for a given seed:

- ``texto_120_paginas.pdf``: 120 A4 text pages (a text layer, no images);
- ``escaneo_60_paginas.pdf``: 60 A4 pages, each a 300 dpi JPEG page image (no text layer), with an
  invented RUT, email and phone on most pages;
- ``tiff_30_paginas.tif``: 30 frames of a multi-page TIFF, at 150 and 200 dpi mixed.
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_bench.canvas import font, font_path  # noqa: E402
from test_bench.fake_data import FakeData, Person  # noqa: E402

A4_PT = (595.0, 842.0)
A4_INCHES = (8.27, 11.69)

FILLER = [
    "Se deja constancia de la revisión de los antecedentes presentados por la unidad requirente.",
    "El presente documento es ficticio y fue generado para probar la herramienta de anonimización.",
    "La comisión evaluadora sesionó en la fecha indicada y revisó cada uno de los puntos de la tabla.",
    "Los montos señalados corresponden a valores referenciales y no constituyen compromiso alguno.",
    "Se solicita a la jefatura correspondiente tomar conocimiento y dar curso a lo resuelto.",
    "Las observaciones formuladas deberán ser respondidas dentro del plazo de cinco días hábiles.",
    "El expediente queda disponible para consulta en la oficina de partes durante el horario habitual.",
    "Sin otro particular, se cierra la presente acta con la firma de los asistentes.",
]


def _contact_lines(p: Person, rng: random.Random) -> list[str]:
    """The personal data of a page: name, RUT, email and phone in varied formats."""
    return [
        f"Nombre: {p.full_name}",
        f"RUT: {p.rut(rng.choice(['dots', 'no_dots']))}",
        f"Correo electrónico: {p.email(rng.choice(['dot', 'initial', 'underscore']))}",
        f"Teléfono: {p.phone.format(rng.choice(['mobile_international', 'mobile_national']))}",
    ]


def _page_lines(n: int, total: int, title: str, fake: FakeData, rng: random.Random, with_data: bool) -> list[str]:
    lines = [f"{title}, página {n} de {total}", "", f"Folio {fake.folio()} · {fake.date()}", ""]
    body = [rng.choice(FILLER) for _ in range(rng.randint(10, 16))]
    if with_data:
        at = rng.randint(2, len(body) - 2)
        body[at:at] = ["", *_contact_lines(fake.person(), rng), ""]
    body.append(f"Monto referencial: {fake.amount()}")
    return lines + body


def text_pdf(fake: FakeData, pages: int = 120) -> bytes:
    """A text PDF: every page has a text layer; most pages carry one invented person's data."""
    rng = random.Random(f"{fake.seed}:text_pdf")
    doc = pymupdf.open()
    sans = str(font_path("sans"))
    for n in range(1, pages + 1):
        page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
        page.insert_font(fontname="dejavu", fontfile=sans)
        page.draw_rect(pymupdf.Rect(36, 36, A4_PT[0] - 36, A4_PT[1] - 36), color=(0.25, 0.3, 0.5), width=1)
        lines = _page_lines(n, pages, "Acta de sesión ficticia", fake, rng, with_data=n % 5 != 0)
        y = 80.0
        for i, line in enumerate(lines):
            size = 15 if i == 0 else 10
            page.insert_text((60, y), line, fontname="dejavu", fontsize=size)
            y += 22 if i == 0 else 17
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return data


def page_image(lines: list[str], size: tuple[int, int], dpi: int) -> Image.Image:
    """A white page in grayscale with the lines written as a scanner would see them."""
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    px = dpi / 72  # pixels per point
    margin = int(54 * px)
    draw.rectangle((margin // 2, margin // 2, size[0] - margin // 2, size[1] - margin // 2), outline=90, width=max(2, dpi // 100))
    y = margin
    for i, line in enumerate(lines):
        pt = 15 if i == 0 else 10
        draw.text((margin, y), line, fill=20, font=font("sans_bold" if i == 0 else "sans", int(pt * px)))
        y += int((24 if i == 0 else 18) * px)
    return img


def scanned_pdf(fake: FakeData, pages: int = 60, dpi: int = 300) -> bytes:
    """A scanned PDF: each page is only a 300 dpi JPEG; most pages show an invented RUT, email and phone."""
    rng = random.Random(f"{fake.seed}:scanned_pdf")
    size = (round(A4_INCHES[0] * dpi), round(A4_INCHES[1] * dpi))
    doc = pymupdf.open()
    for n in range(1, pages + 1):
        lines = _page_lines(n, pages, "Informe escaneado ficticio", fake, rng, with_data=n % 6 != 0)
        buf = io.BytesIO()
        page_image(lines, size, dpi).save(buf, "JPEG", quality=80)
        page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
        page.insert_image(page.rect, stream=buf.getvalue(), keep_proportion=False)
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return data


def tiff(fake: FakeData, frames: int = 30) -> bytes:
    """A multi-page TIFF: frames at 150 and 200 dpi mixed (A4 scans at two resolutions)."""
    rng = random.Random(f"{fake.seed}:tiff")
    images = []
    for n in range(1, frames + 1):
        dpi = 200 if n % 3 == 0 else 150
        size = (round(A4_INCHES[0] * dpi), round(A4_INCHES[1] * dpi))
        lines = _page_lines(n, frames, "Expediente digitalizado ficticio", fake, rng, with_data=n % 4 != 0)
        images.append(page_image(lines, size, dpi))
    buf = io.BytesIO()
    images[0].save(buf, "TIFF", save_all=True, append_images=images[1:], compression="tiff_deflate")
    return _zero_strip_padding(buf.getvalue())


def _zero_strip_padding(data: bytes) -> bytes:
    """libtiff pads a frame's image data to an even offset with one byte it never sets: zero it,
    so the same seed always gives the same file (the pixels are the same either way)."""
    out = bytearray(data)
    with Image.open(io.BytesIO(data)) as im:
        for frame in range(im.n_frames):
            im.seek(frame)
            end = max(o + c for o, c in zip(im.tag_v2[273], im.tag_v2[279]))
            if end % 2 and end < len(out):
                out[end] = 0
    return bytes(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("test_data/long"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    fake = FakeData(args.seed)
    outputs = {
        "texto_120_paginas.pdf": lambda: text_pdf(fake.derive("text_pdf")),
        "escaneo_60_paginas.pdf": lambda: scanned_pdf(fake.derive("scanned_pdf")),
        "tiff_30_paginas.tif": lambda: tiff(fake.derive("tiff")),
    }
    for name, make in outputs.items():
        path = args.output / name
        path.write_bytes(make())
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
