"""Context shared by every generator of the test dataset."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from test_bench.canvas import Canvas
from test_bench.faces import FaceProvider
from test_bench.fake_data import FakeData
from test_bench.schema import Element, FileEntry, MetadataEntry, Page


@dataclass
class Context:
    root: Path
    seed: int
    fake: FakeData
    faces: FaceProvider
    options: dict[str, Any] = field(default_factory=dict)

    def fake_data(self, name: str) -> FakeData:
        """Data factory for a module: deterministic by name, never repeating other modules' values."""
        return self.fake.derive(name)

    def rng(self, name: str) -> np.random.Generator:
        return np.random.default_rng([self.seed, zlib.crc32(name.encode())])

    def path(self, relative: str) -> Path:
        dest = self.root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest


def image_file_entry(
    *,
    id: str,
    path: str,
    format: str,
    category: str,
    description: str,
    canvas: Canvas | None = None,
    size: tuple[int, int] | None = None,
    elements: list[Element] | None = None,
    metadata: list[MetadataEntry] | None = None,
    tags: dict[str, Any] | None = None,
) -> FileEntry:
    """Builds the manifest entry for a single-page image.

    The size and the elements must be in the *displayed* geometry (after applying EXIF).
    """
    if canvas is not None:
        size = size or (canvas.width, canvas.height)
        elements = elements if elements is not None else canvas.elements
    assert size is not None and elements is not None
    for e in elements:
        e.page = 0
    return FileEntry(
        id=id,
        path=path,
        format=format,
        category=category,
        description=description,
        pages=[Page(index=0, width=size[0], height=size[1], unit="px")],
        elements=list(elements),
        sensitive_metadata=list(metadata or []),
        tags=dict(tags or {}),
    )


def save_image(img: Image.Image, dest: Path, format: str, **options: Any) -> None:
    """Saves without carrying metadata over by accident (metadata is added explicitly)."""
    clean = Image.frombytes(img.mode, img.size, img.tobytes())
    if img.mode == "P":
        clean.putpalette(img.getpalette())
    pil_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "tiff": "TIFF"}[format]
    if pil_format == "JPEG":
        options.setdefault("quality", 90)
        clean = clean.convert("RGB")
    clean.save(dest, pil_format, **options)
