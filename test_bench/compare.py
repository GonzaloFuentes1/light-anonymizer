"""Before-and-after sheets: the original file next to the output of each system.

Usage:
    uv run python -m test_bench.compare [--systems prototype notebook] [--output results/examples] [--id ID ...]

Without ``--id``, it takes one file per category. Each sheet shows the first page (or the page
given with ``ID:page``) of the original and of the output of each system in ``results/details/<system>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from test_bench.canvas import font
from test_bench.schema import Manifest
from test_bench.visualize import visible_pages

TITLES = {
    "original": "Original",
    "oracle": "Oráculo (con la respuesta)",
    "notebook": "Cuaderno actual",
    "identity": "Identidad (sin cambios)",
    "prototype": "Prototipo (detección real)",
}
DEFAULT_IDS = [
    "pdft_cuaderno",
    "pdft_honorarios_01",
    "pdfm_redaccion_falsa",
    "pdfe_informe_300dpi",
    "img_rot_nota_45",
    "img_exif_orientacion_6",
    "ced_frente_foto_perspectiva",
    "pant_chat_movil",
    "rost_grupo_compuesto_01",
]
HEIGHT = 900


def _page(path: Path, format: str, index: int) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        pages = visible_pages(path, format)
    except Exception:  # noqa: BLE001 - the output may fail to open
        return None
    return pages[min(index, len(pages) - 1)] if pages else None


def _notice(text: str, width: int) -> Image.Image:
    img = Image.new("RGB", (width, HEIGHT), (236, 236, 236))
    ImageDraw.Draw(img).multiline_text((24, HEIGHT // 2 - 40), text, font=font("sans", 22), fill=(90, 90, 90))
    return img


def sheet(man: Manifest, file_id: str, page: int, systems: list[str], results_root: Path) -> Image.Image:
    f = next(x for x in man.files if x.id == file_id)
    original = _page(Path(man.root) / f.path, f.format, page)
    assert original is not None, f"no se pudo abrir {f.path}"
    scale = HEIGHT / original.height
    width = max(1, int(original.width * scale))
    columns = [("original", original.resize((width, HEIGHT)))]
    for s in systems:
        img = _page(results_root / s / "files" / f.path, f.format, page)
        if img is None:
            columns.append((s, _notice("Sin salida:\neste sistema no\nprocesa el archivo", width)))
        else:
            columns.append((s, img.resize((width, HEIGHT))))
    margin, header = 16, 56
    canvas = Image.new("RGB", (len(columns) * (width + margin) + margin, HEIGHT + header + margin), "white")
    d = ImageDraw.Draw(canvas)
    for i, (name, img) in enumerate(columns):
        x = margin + i * (width + margin)
        d.text((x, 14), TITLES.get(name, name), font=font("sans_bold", 20), fill=(20, 20, 20))
        canvas.paste(img, (x, header))
        d.rectangle((x - 1, header - 1, x + width, header + HEIGHT), outline=(180, 180, 180))
    return canvas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Láminas de antes y después.")
    ap.add_argument("--manifest", type=Path, default=Path("test_data/generated/manifest.json"))
    ap.add_argument("--results", type=Path, default=Path("results/details"))
    ap.add_argument("--systems", nargs="+", default=["prototype", "notebook"])
    ap.add_argument("--output", type=Path, default=Path("results/examples"))
    ap.add_argument("--id", nargs="*", help="ID o ID:pagina (sin esto, uno por categoría)")
    args = ap.parse_args(argv)
    man = Manifest.load(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    for request in args.id or DEFAULT_IDS:
        file_id, _, page = request.partition(":")
        img = sheet(man, file_id, int(page or 0), args.systems, args.results)
        dest = args.output / f"{file_id}{'_p' + page if page else ''}.jpg"
        img.save(dest, quality=88)
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
