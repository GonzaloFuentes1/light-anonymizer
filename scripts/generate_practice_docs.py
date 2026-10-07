"""Generates the invented practice documents of the user manual and the usability test (not versioned).

Usage (from the repository root):
    uv run python scripts/generate_practice_docs.py [--output test_data/practice] [--seed 21]

The fictitious test set (scripts/generate_test_data.py) has no page that the app exports as an
image, no doubtful RUT written as bare digits and no signature that the app misses. These two
files have one of each, with invented data only (``test_bench.fake_data``), and the same bytes
on every run for a given seed:

- ``acta_con_timbre.pdf``: a text page with a round stamp drawn over the signer's name (the stamp's
  ring crosses that name's zone, so the page is exported as an image), an inventory number that
  looks like a RUT without dots or dash and whose check digit does not match (a doubtful RUT, not
  censored by default) and an institutional link (another link, not censored by default);
- ``oficio_fotografiado.jpg``: a letter photographed under uneven light (so not a plain sheet of
  paper) with a handwritten signature that has no "Firma" label and no signature line next to it,
  which the app does not find: it has to be covered with "Dibujar zona".
"""

from __future__ import annotations

import argparse
import io
import math
import random
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anonymizer.engine.text import rut_suggested  # noqa: E402
from test_bench.canvas import font, font_path  # noqa: E402
from test_bench.fake_data import FakeData, check_digit  # noqa: E402

A4_PT = (595.0, 842.0)
A4_INCHES = (8.27, 11.69)
LINK = "https://www.goreficticio.cl/inventario/equipos-2026"


def inventory_number(rng: random.Random) -> str:
    """Eight bare digits shaped like a RUT whose check digit does not match, that the app suggests."""
    while True:
        body = rng.randint(3_000_000, 8_999_999)
        dv = rng.choice([d for d in "0123456789" if d != check_digit(body)])
        value = f"{body}{dv}"
        line = f"N° de inventario del equipo: {value}"
        if rut_suggested(line, len(line) - len(value), len(line)):
            return value


def acta_con_timbre(fake: FakeData) -> bytes:
    rng = random.Random(f"{fake.seed}:acta_con_timbre")
    receiver, signer = fake.person(), fake.person()
    doc = pymupdf.open()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    page.insert_font(fontname="dejavu", fontfile=str(font_path("sans")))
    page.insert_font(fontname="dejavub", fontfile=str(font_path("sans_bold")))
    page.draw_rect(pymupdf.Rect(36, 36, A4_PT[0] - 36, A4_PT[1] - 36), color=(0.25, 0.3, 0.5), width=1)
    lines = [
        ("Acta de entrega de equipo computacional (documento ficticio)", "dejavub", 13),
        ("", "dejavu", 10),
        (f"Folio {fake.folio()} · {fake.date()}", "dejavu", 10),
        ("", "dejavu", 10),
        ("La Unidad de Informática entrega en préstamo el equipo que se indica a la persona", "dejavu", 10),
        ("individualizada a continuación, para el desempeño de sus funciones.", "dejavu", 10),
        ("", "dejavu", 10),
        (f"Nombre: {receiver.full_name}", "dejavu", 10),
        (f"RUT: {receiver.rut('dots')}", "dejavu", 10),
        (f"Correo electrónico: {receiver.email('dot')}", "dejavu", 10),
        (f"Teléfono: {receiver.phone.format('mobile_international')}", "dejavu", 10),
        ("", "dejavu", 10),
        ("Equipo: computador portátil de 14 pulgadas, con cargador y bolso.", "dejavu", 10),
        (f"N° de inventario del equipo: {inventory_number(rng)}", "dejavu", 10),
        ("El equipo se devuelve al término del préstamo en las mismas condiciones.", "dejavu", 10),
        (f"Procedimiento de préstamo: {LINK}", "dejavu", 10),
        ("", "dejavu", 10),
        ("Entrega conforme,", "dejavu", 10),
    ]
    y = 80.0
    for text, fontname, size in lines:
        if text:
            page.insert_text((60, y), text, fontname=fontname, fontsize=size)
        y += 22 if size > 10 else 17
    # The signer's block, and a round stamp drawn over the name: the stamp's ring crosses the name's zone.
    y += 40
    name_at = pymupdf.Point(60, y)
    page.insert_text(name_at, signer.full_name, fontname="dejavu", fontsize=10)
    page.insert_text((60, y + 15), "Jefatura Unidad de Informática", fontname="dejavu", fontsize=10)
    center = pymupdf.Point(60 + 0.55 * pymupdf.get_text_length(signer.full_name, fontsize=10), y - 4)
    blue = (0.15, 0.25, 0.65)
    page.draw_circle(center, 38, color=blue, width=2)
    page.draw_circle(center, 31, color=blue, width=1)
    for i, word in enumerate(("GOBIERNO", "REGIONAL", "FICTICIO")):
        page.insert_text((center.x - 22, center.y - 8 + i * 9), word, fontname="dejavub", fontsize=6.5, color=blue)
    doc.subset_fonts()
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return data


def _signature(draw: ImageDraw.ImageDraw, x: float, y: float, scale: float, rng: random.Random) -> None:
    """A handwritten-looking stroke: uneven loops that drift up and to the right, then a short tail."""
    points = []
    for i in range(320):
        t = i / 319
        angle = t * 9 * math.pi
        amplitude = 20 + 9 * math.sin(t * 5.3) + rng.uniform(-0.8, 0.8)
        points.append(
            (
                x + scale * (t * 230 + 14 * math.cos(angle * 1.1)),
                y + scale * (amplitude * math.sin(angle) - 26 * t + 7 * math.sin(t * 13)),
            )
        )
    last = points[-1]
    points += [(last[0] + scale * 12 * k, last[1] + scale * 9 * k) for k in (1, 2, 3)]
    draw.line(points, fill=25, width=max(3, round(3 * scale)), joint="curve")


def oficio_fotografiado(fake: FakeData, size: tuple[int, int] = (1240, 1754)) -> bytes:
    """A letter photographed with a phone: uneven light, so it is not a plain sheet of paper."""
    rng = random.Random(f"{fake.seed}:oficio_fotografiado")
    person, chief = fake.person(), fake.person()
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    px = size[0] / A4_PT[0]
    margin = int(60 * px)
    lines = [
        ("OFICIO ORDINARIO (documento ficticio)", True),
        (f"Folio {fake.folio()} · {fake.date()}", False),
        ("", False),
        (f"A: {person.full_name}", False),
        ("", False),
        ("Junto con saludar, informo a usted que su solicitud fue recibida y será", False),
        ("revisada por la comisión en su próxima sesión. Para consultas, puede", False),
        (f"escribir a {person.email('dot')}.", False),
        ("", False),
        ("Sin otro particular, le saluda atentamente,", False),
    ]
    y = margin
    for text, bold in lines:
        if text:
            draw.text((margin, y), text, fill=20, font=font("sans_bold" if bold else "sans", int(11 * px)))
        y += int(20 * px)
    _signature(draw, margin + 30 * px, y + 40 * px, px * 0.75, rng)
    y += int(95 * px)
    draw.text((margin, y), chief.full_name.upper(), fill=20, font=font("sans_bold", int(10 * px)))
    draw.text((margin, y + int(15 * px)), "Jefatura de Gabinete", fill=20, font=font("sans", int(10 * px)))
    # The light of a desk lamp: bright in one corner, a shadow towards the other, and sensor noise.
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float32)
    light = 1.0 - 0.55 * np.clip((xx / size[0] + yy / size[1]) / 2, 0, 1) ** 1.3
    noise = np.random.default_rng(fake.seed).normal(0, 4, (size[1], size[0]))
    pixels = np.clip(np.asarray(img, np.float32) * light + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).convert("RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("test_data/practice"))
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    fake = FakeData(args.seed)
    outputs = {
        "acta_con_timbre.pdf": lambda: acta_con_timbre(fake.derive("acta_con_timbre")),
        "oficio_fotografiado.jpg": lambda: oficio_fotografiado(fake.derive("oficio_fotografiado")),
    }
    for name, make in outputs.items():
        path = args.output / name
        path.write_bytes(make())
        print(f"{path}  {path.stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
