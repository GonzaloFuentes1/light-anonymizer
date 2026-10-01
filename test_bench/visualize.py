"""Overlays the ground truth on every page, to check by eye that the polygons fit.

Usage:
    uv run python -m test_bench.visualize test_data/generated/manifest.json --output overlays/ [--id ID ...]

Colors: red = personal data, blue = neutral text, green = whole face, yellow = face core,
magenta = hidden layer (drawn with a dashed line even though it is not visible on the page).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageOps, ImageSequence

from test_bench.schema import FileEntry, Manifest

PDF_SCALE = 2.0  # 144 dpi


def visible_pages(path: Path, format: str) -> list[Image.Image]:
    """Every page as it is seen (images with EXIF applied; PDF in its visible orientation)."""
    if format == "pdf":
        doc = pymupdf.open(path)
        out = []
        for p in doc:
            pix = p.get_pixmap(matrix=pymupdf.Matrix(PDF_SCALE, PDF_SCALE), alpha=False)
            out.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        doc.close()
        return out
    img = Image.open(path)
    if getattr(img, "n_frames", 1) > 1:
        return [ImageOps.exif_transpose(f.copy()).convert("RGB") for f in ImageSequence.Iterator(img)]
    return [ImageOps.exif_transpose(img).convert("RGB")]


def to_visual(file_entry: FileEntry, index: int, polygon: list[list[float]]) -> list[tuple[float, float]]:
    """Maps a polygon from (unrotated) page coordinates to pixels of the visible page."""
    page = file_entry.pages[index]
    if file_entry.format != "pdf":
        return [(x, y) for x, y in polygon]
    w, h, r = page.width, page.height, page.rotation % 360
    points = []
    for x, y in polygon:
        if r == 90:
            x, y = h - y, x
        elif r == 180:
            x, y = w - x, h - y
        elif r == 270:
            x, y = y, w - x
        points.append((x * PDF_SCALE, y * PDF_SCALE))
    return points


def _dashed(d: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color, step: float = 8) -> None:
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True):
        length = max(1.0, float(np.hypot(x1 - x0, y1 - y0)))
        for t in np.arange(0, length, step * 2):
            a, b = t / length, min(1.0, (t + step) / length)
            d.line(
                [(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                fill=color,
                width=2,
            )


def overlay(file_entry: FileEntry, root: Path) -> list[Image.Image]:
    pages = visible_pages(root / file_entry.path, file_entry.format)
    for i, img in enumerate(pages):
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for e in file_entry.elements:
            if e.page != i or e.polygon is None:
                continue
            pts = to_visual(file_entry, i, e.polygon)
            if e.layer == "hidden":
                _dashed(d, pts, (220, 0, 220, 255))
                continue
            if e.type == "face":
                d.polygon(pts, outline=(0, 170, 0, 255), width=3)
                if e.core:
                    d.polygon(to_visual(file_entry, i, e.core), outline=(240, 200, 0, 255), width=2)
            elif e.type == "text":
                d.polygon(pts, outline=(40, 90, 255, 200), width=1)
            else:
                d.polygon(pts, fill=(255, 0, 0, 50), outline=(255, 0, 0, 255), width=2)
        pages[i] = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return pages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dibuja la verdad de terreno sobre los archivos del conjunto.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--id", nargs="*", help="solo estos archivos")
    ap.add_argument("--max-side", type=int, default=1600)
    args = ap.parse_args(argv)
    man = Manifest.load(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    for f in man.files:
        if args.id and f.id not in args.id:
            continue
        if f.expected != "process":
            continue
        for i, img in enumerate(overlay(f, Path(man.root))):
            img.thumbnail((args.max_side, args.max_side))
            img.save(args.output / f"{f.id}_p{i}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
