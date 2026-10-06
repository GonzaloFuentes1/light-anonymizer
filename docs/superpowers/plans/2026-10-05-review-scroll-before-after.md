# Review screen: continuous scroll and real before/after — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-page viewer of Revisar with a continuous vertical scroll of pages, each shown as the original with its zones (before) next to the real redacted result rendered by the export code (after).

**Architecture:** The engine's per-page redaction is split out of the export functions (`pdf.redact_page`, `pdf.redaction_rects`, `image.frame`, `image.redact_frame`) and reused by a new `render_result` engine method, served by `GET /api/files/{id}/pages/{n}.png?redacted=true`. The UI's pure logic (geometry, versions, request queue, memory budget, anchors) lives in a new classic script `anonymizer/ui/review-core.js`, tested with `node --test`; `app.js` builds one row per page, loads images lazily and keeps the after up to date.

**Tech Stack:** Python 3.12, PyMuPDF 1.28.2, Pillow, OpenCV, NumPy, FastAPI (tests with `tests/live_client.py`), vanilla JS (no build step), Node ≥ 20 for `node --test`, pytest via `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-10-02-review-scroll-before-after-design.md` (approved by the user on 2026-10-05, including its section 2.2). Read it before any task; section numbers below refer to it.

## Global Constraints

- Work in the worktree `C:/Users/gfuen/dev/la-wt/review-scroll`, branch `review-scroll`. Run Python with `uv run ...` from there.
- Invented data only. `test_data/real/` (main checkout) is private: never open, use or quote it.
- Code, comments, docs, commit messages, file and folder names in English; every text the end user sees in Spanish (Chile).
- Commits under the configured git user. Never add `Co-Authored-By` lines or any mention of Claude/AI. Do not push, merge or touch other branches.
- Match the surrounding code style and comment density (`app.js`: `h()` helper, `$`/`$$`, `keepFocus`, Spanish strings inline; Python: type hints, short docstrings).
- The UI loads nothing from the network (CSP `script-src 'self'`); new scripts are local files referenced relatively.
- Export behaviour must not change: every existing export, leak-check and API test keeps passing.
- Zoom limits as today: effective scale 0.05–8; fit caps 1.6 (PDF points) and 1 (image pixels); request zoom capped so the longest side ≤ 9000 px, rounded to 0.01, minimum 0.02.
- Constants from the spec: stacked layout when a column < 420 px; observer margin = one viewport height; images kept within 3 viewport heights and ≤ 150 megapixels total; at most 3 image requests in flight; programmatic scroll instant when > 2 viewport heights; no requests until 150 ms without scroll events; select change debounced 300 ms; resize debounced 150 ms; zoom re-request pause 120 ms; selection scroll margin 48 px.
- Long jobs (full test suite, app runs) start in the background (Bash `run_in_background: true`) with a log; never run a job longer than ~3 minutes in the foreground.

## Review Focus

1. **A PDF with a hidden optional-content layer**: the export reveals hidden layers and redacts them; the after must show the same (layer visible, its data redacted), never the layer hidden. Pinned by a parity case in Task 3.
2. **An RGBA or palette PNG and a 3-page TIFF**: before (`render_page`, keeps RGBA/L) and after (RGB) must have the same size so rows align, and the after must equal the export. Pinned in Tasks 1 and 3.
3. **A PDF whose pages have different sizes (A4 portrait + landscape) and a TIFF with frames of different resolutions**: one fit scale for the PDF (widest page), per-frame fit for images; reserved row sizes match the images. Pinned in Task 6 (`scales`).
4. **A finding partly or wholly outside its page** (off-page text layer): scrolling to it stays inside its own row. Pinned in Task 6 (`scrollTarget`).
5. **The open file is processed again while its afters are loading**: `redacted=true` answers 409 `not_ready` during processing (Task 5) and the UI drops late responses silently by key/generation (Task 6 `acceptResponse`, Task 10).

---

## File map

| File | Responsibility | Tasks |
|---|---|---|
| `anonymizer/engine/image.py` | `view_frame`, `frame`, `redact_frame` (shared by export and after) | 1 |
| `anonymizer/engine/pdf.py` | `redaction_rects`, `redact_page` (shared by export and after) | 2 |
| `anonymizer/engine/real.py` | `render_result`, PNG encoding outside `PDF_LOCK`, `_export_pdf` uses `redaction_rects` | 2, 3 |
| `anonymizer/engine/fake.py` | `_redact_page`, `_fill`, `render_result` | 4 |
| `anonymizer/engine/__init__.py` | contract docstring and `Engine.render_result` | 4 |
| `anonymizer/api/server.py` | `?redacted=true` on the page route | 5 |
| `anonymizer/ui/review-core.js` (new) | pure functions: geometry, versions, keys, queue, budget, anchors | 6 |
| `tests/ui/review-core.test.js` (new) + `tests/test_ui_core.py` (new) | `node --test` suite and its pytest wrapper | 6 |
| `anonymizer/ui/index.html`, `app.css` | new Revisar markup and styles | 7 |
| `anonymizer/ui/app.js` | file bar, rows, layout, loading, after updates, interaction | 8–12 |
| `scripts/generate_long_docs.py` (new) | long invented documents in `test_data/long/` | 13 |
| `tests/test_engine_real.py`, `tests/test_api.py`, `tests/test_ui_static.py` | tests | 1–7 |
| `PLAN.md`, `README.md`, `README.es.md`, the spec's status line | docs | 14 |

---

### Task 1: Image helpers shared by export and after

**Files:**
- Modify: `anonymizer/engine/image.py` (functions `frames`, `redact`; add `view_frame`, `frame`, `redact_frame`)
- Test: `tests/test_engine_real.py` (new section at the end "after: image helpers")

**Interfaces:**
- Produces: `image.view_frame(frame: Image.Image) -> Image.Image` (RGB, EXIF orientation applied); `image.frame(path: str, n: int) -> Image.Image` (raises `IndexError` for a missing frame, `FileError("corrupt")` for a damaged file); `image.redact_frame(arr: np.ndarray, polygons: list[list[list[float]]]) -> np.ndarray` (fills in place and returns `arr`).

- [ ] **Step 1: Write the failing tests**

```python
# ---------------------------------------------------------------------------
# after: image helpers (spec 4.1)
# ---------------------------------------------------------------------------


def _tiff(path: Path, sizes) -> Path:
    frames = []
    for i, (w, h) in enumerate(sizes):
        img = Image.new("RGB", (w, h), (255, 255, 255))
        ImageDraw.Draw(img).rectangle((10, 10, 60 + 10 * i, 40), fill=(200, 30 * i, 0))
        frames.append(img)
    frames[0].save(path, "TIFF", save_all=True, append_images=frames[1:], compression="tiff_deflate")
    return path


def test_image_frame_equals_the_frames_generator(tmp_path):
    import numpy as np

    from anonymizer.engine import image

    path = _tiff(tmp_path / "multi.tif", [(300, 200), (240, 320), (500, 260)])
    every = list(image.frames(str(path)))
    for n in range(3):
        assert np.array_equal(np.array(image.frame(str(path), n)), np.array(every[n]))
    with pytest.raises(IndexError):
        image.frame(str(path), 3)


def test_image_frame_applies_exif_orientation(tmp_path):
    from anonymizer.engine import image

    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (300, 200), "white").save(tmp_path / "o6.png", exif=exif.tobytes())
    assert image.frame(str(tmp_path / "o6.png"), 0).size == (200, 300)


def test_redact_frame_fills_the_grown_polygon_only():
    import numpy as np

    from anonymizer.engine import image

    arr = np.full((100, 200, 3), 255, np.uint8)
    out = image.redact_frame(arr, [[[50, 40], [150, 40], [150, 60], [50, 60]]])
    assert out is arr
    assert arr[45:56, 55:146].max() == 0
    assert arr[5:20, 5:40].min() == 255
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_real.py -k "image_frame or redact_frame" -v`
Expected: FAIL with `AttributeError: module 'anonymizer.engine.image' has no attribute 'frame'` (and `redact_frame`).

- [ ] **Step 3: Implement**

In `anonymizer/engine/image.py`, replace `frames` and the fill loop of `redact`:

```python
def view_frame(frame: Image.Image) -> Image.Image:
    """A frame as the user sees it: EXIF orientation applied, RGB."""
    return ImageOps.exif_transpose(frame.copy()).convert("RGB")


def frames(path: str) -> Iterator[Image.Image]:
    """RGB frames in view space (EXIF orientation applied). Raises ``FileError`` if damaged."""
    with _open(path) as img:
        try:
            for frame in _frames(img):
                yield view_frame(frame)
        except (OSError, SyntaxError, ValueError) as exc:
            raise FileError("corrupt", repr(exc)) from exc


def frame(path: str, n: int) -> Image.Image:
    """Frame ``n`` in view space: the pixels ``frames`` yields for it, without decoding the frames
    before it. Raises ``IndexError`` when there is no such frame (an MPO has only frame 0)."""
    with _open(path) as img:
        if n < 0 or (img.format == "MPO" and n > 0):
            raise IndexError(n)
        try:
            if n:
                img.seek(n)
            return view_frame(img)
        except EOFError as exc:
            raise IndexError(n) from exc
        except (OSError, SyntaxError, ValueError) as exc:
            raise FileError("corrupt", repr(exc)) from exc


def redact_frame(arr: np.ndarray, polygons) -> np.ndarray:
    """Fills each polygon, enlarged by ``FILL_GROWTH`` around its center, with black (in place)."""
    for polygon in polygons:
        pol = np.asarray(polygon, np.float64)
        center = pol.mean(axis=0)
        grown = (pol - center) * FILL_GROWTH + center
        cv2.fillPoly(arr, [np.round(grown).astype(np.int32)], (0, 0, 0))
    return arr
```

and in `redact`, the loop body becomes:

```python
    for n, frame in enumerate(frames(source)):
        arr = redact_frame(np.array(frame), polygons_by_page.get(n, []))
        # A new image from raw pixels: no metadata is carried over.
        outputs.append(Image.fromarray(arr))
```

- [ ] **Step 4: Run the new tests and the image export tests**

Run: `uv run pytest tests/test_engine_real.py -k "image_frame or redact_frame or exif_orientation or image" -v`
Expected: PASS (including the existing `test_zone_drawn_on_an_image_with_exif_orientation`).

- [ ] **Step 5: Commit**

```bash
git add anonymizer/engine/image.py tests/test_engine_real.py
git commit -m "Split the image frame and fill helpers out of the image export"
```

---

### Task 2: PDF helpers shared by export and after

**Files:**
- Modify: `anonymizer/engine/pdf.py:422-458` (`redact`; add `redaction_rects`, `redact_page`)
- Modify: `anonymizer/engine/real.py` (`_export_pdf`, the static method at the end)
- Test: `tests/test_engine_real.py` (new section "after: PDF helpers")

**Interfaces:**
- Consumes: `common.bbox_of(polygon) -> (x0, y0, x1, y1)`.
- Produces: `pdf.redaction_rects(doc: pymupdf.Document, polygons_by_page: dict[int, list]) -> dict[int, list[pymupdf.Rect]]` (caller holds `PDF_LOCK`; pages out of range are ignored); `pdf.redact_page(doc: pymupdf.Document, n: int, rects: list[pymupdf.Rect]) -> None` (caller holds `PDF_LOCK`).

- [ ] **Step 1: Write the failing tests**

```python
# ---------------------------------------------------------------------------
# after: PDF helpers (spec 4.1)
# ---------------------------------------------------------------------------


def test_redaction_rects_move_view_boxes_to_the_unrotated_page(tmp_path):
    from anonymizer.engine import pdf
    from anonymizer.engine.locks import PDF_LOCK

    path = _boxed_pdf(tmp_path / "r.pdf", 90, "cropbox")
    with PDF_LOCK, pymupdf.open(path) as doc:
        rects = pdf.redaction_rects(doc, {0: [common.rect_polygon(10, 20, 110, 60)], 5: [common.rect_polygon(0, 0, 1, 1)]})
        assert list(rects) == [0]
        expected = (pymupdf.Rect(10, 20, 110, 60) * pymupdf.Matrix(doc[0].derotation_matrix)).normalize()
        assert rects[0] == [expected]


def test_redact_page_removes_text_and_annotations_of_that_page_only(tmp_path):
    from anonymizer.engine import pdf
    from anonymizer.engine.locks import PDF_LOCK

    path = make_pdf(tmp_path / "a.pdf")
    with pymupdf.open(path) as doc:
        doc[0].add_freetext_annot(pymupdf.Rect(300, 600, 500, 640), "Nota de Persona Inventada")
        doc.save(tmp_path / "b.pdf")
    with PDF_LOCK, pymupdf.open(tmp_path / "b.pdf") as doc:
        hit = doc[0].search_for(EMAIL)[0]
        pdf.redact_page(doc, 0, [hit])
        assert EMAIL not in doc[0].get_text()
        assert VALID_RUT in doc[0].get_text()
        assert not list(doc[0].annots() or [])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_real.py -k "redaction_rects or redact_page" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'redaction_rects'`.

- [ ] **Step 3: Implement**

In `anonymizer/engine/pdf.py` add `bbox_of` to the import from `anonymizer.engine.common`, then:

```python
def redaction_rects(doc: pymupdf.Document, polygons_by_page: dict[int, list]) -> dict[int, list[pymupdf.Rect]]:
    """The bounding box of each polygon (view space) in the unrotated space of its page, where
    ``redact_page`` places the zones. Pages out of range are skipped. The caller holds ``PDF_LOCK``."""
    rects: dict[int, list[pymupdf.Rect]] = {}
    for n, polygons in polygons_by_page.items():
        if not 0 <= n < doc.page_count:
            continue
        to_page = pymupdf.Matrix(doc[n].derotation_matrix)
        rects[n] = [(pymupdf.Rect(*bbox_of(p)) * to_page).normalize() for p in polygons]
    return rects


def redact_page(doc: pymupdf.Document, n: int, rects: list[pymupdf.Rect]) -> None:
    """Removes ``rects`` from page ``n`` for real (text, vector paths and image pixels) and cleans
    the page (annotations, form fields, page-level actions and metadata). The caller holds ``PDF_LOCK``."""
    page = doc[n]
    # MuPDF misplaces redaction zones on a rotated page whose CropBox or MediaBox does not
    # start at (0, 0): the zone moved or fell off the page and left the data visible.
    # Unrotated, the page space is exactly the space of the zones; the rotation is put back.
    rotation = page.rotation
    if rotation:
        page.set_rotation(0)
    for r in rects:
        page.add_redact_annot(r, fill=(0, 0, 0))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
    if rotation:
        page.set_rotation(rotation)
    for annot in list(page.annots() or []):
        page.delete_annot(annot)
    for widget in list(page.widgets() or []):
        page.delete_widget(widget)
    for key in _PAGE_KEYS:
        doc.xref_set_key(page.xref, key, "null")
```

`redact` keeps its signature and docstring; its page loop becomes:

```python
            reveal_layers(doc)
            for n in range(doc.page_count):
                redact_page(doc, n, rects_by_page.get(n, []))
```

In `anonymizer/engine/real.py`, `_export_pdf` becomes:

```python
    @staticmethod
    def _export_pdf(file: AnalyzedFile, active: list[Finding], staged: Path) -> None:
        import pymupdf

        from anonymizer.engine import pdf

        polygons: dict[int, list] = {}
        for f in active:
            polygons.setdefault(f.page, []).append(f.polygon)
        with PDF_LOCK:
            with pymupdf.open(file.path, filetype="pdf") as doc:
                rects = pdf.redaction_rects(doc, polygons)
        pdf.redact(file.path, str(staged), rects)
```

- [ ] **Step 4: Run the new tests and every export test**

Run: `uv run pytest tests/test_engine_real.py tests/test_api.py -v` (background if slow; log to a file)
Expected: PASS, including the 8 cases of `test_zone_drawn_in_view_space_lands_there_on_rotated_cropped_pages`.

- [ ] **Step 5: Commit**

```bash
git add anonymizer/engine/pdf.py anonymizer/engine/real.py tests/test_engine_real.py
git commit -m "Split the per-page PDF redaction out of the export"
```

---

### Task 3: `RealEngine.render_result` and PNG encoding outside the lock

**Files:**
- Modify: `anonymizer/engine/real.py` (`render_page`, add `render_result` and a module function `_png`)
- Test: `tests/test_engine_real.py` (new section "after: parity with the export")

**Interfaces:**
- Consumes: Task 1 (`image.frame`, `image.redact_frame`), Task 2 (`pdf.redaction_rects`, `pdf.redact_page`).
- Produces: `RealEngine.render_result(file: AnalyzedFile, page: int, zoom: float, findings: list[Finding]) -> bytes` (PNG). Keeps only `f.page == page and f.active`. Writes nothing to disk.

- [ ] **Step 1: Write the failing parity tests**

```python
# ---------------------------------------------------------------------------
# after: parity with the export (spec 9.1)
# ---------------------------------------------------------------------------


def _pixels(png: bytes):
    import numpy as np

    with Image.open(io.BytesIO(png)) as img:
        return np.array(img.convert("RGB"))


def _exported_render(engine, file: AnalyzedFile, page: int, zoom: float, tmp_path: Path):
    result = engine.export(file, str(tmp_path / f"out{page}{zoom}"))
    assert result.exported, [leak.message for leak in result.leaks]
    out = AnalyzedFile(id="o", name=Path(result.output_path).name, path=result.output_path, kind=file.kind)
    return _pixels(engine.render_page(out, page, zoom))


def _manual(page: int, x0, y0, x1, y1, fid="m", status="added") -> Finding:
    return Finding(id=f"{fid}{page}", file_id="f1", page=page, type="manual",
                   polygon=common.rect_polygon(x0, y0, x1, y1), detector="reviewer", status=status)  # fmt: skip


def _assert_parity(engine, file, page, zoom, tmp_path, lossy=False):
    import numpy as np

    after = _pixels(engine.render_result(file, page, zoom, list(file.findings)))
    exported = _exported_render(engine, file, page, zoom, tmp_path)
    assert after.shape == exported.shape
    if lossy:
        assert np.abs(after.astype(int) - exported.astype(int)).mean() < 2
    else:
        assert np.array_equal(after, exported)


def _file(path: Path, kind: str, findings) -> AnalyzedFile:
    return AnalyzedFile(id="f1", name=path.name, path=str(path), kind=kind, status="ready", findings=findings)


def test_after_equals_export_text_page(tmp_path):
    file = analyzed(make_pdf(tmp_path / "a.pdf"))
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)


def test_after_equals_export_page_with_annotation_and_no_findings(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        page.insert_text((40, 60), "Texto neutro de relleno.", fontsize=11)
        page.add_freetext_annot(pymupdf.Rect(40, 100, 300, 140), "Nota de Persona Inventada")
        doc.save(tmp_path / "n.pdf")
    _assert_parity(RealEngine(), _file(tmp_path / "n.pdf", "pdf", []), 0, 1.0, tmp_path)


@pytest.mark.parametrize("zoom", [1.0, 1.9])
def test_after_equals_export_scanned_page(tmp_path, zoom):
    scan = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(scan)
    draw.text((100, 200), f"RUT {VALID_RUT}", fill=(0, 0, 0), font=font(40))
    buf = io.BytesIO()
    scan.save(buf, "JPEG", quality=85)
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=buf.getvalue())
        doc.save(tmp_path / "s.pdf")
    file = _file(tmp_path / "s.pdf", "pdf", [_manual(0, 40, 85, 320, 125)])
    _assert_parity(RealEngine(), file, 0, zoom, tmp_path)


def test_after_equals_export_rotated_cropped_page(tmp_path):
    path = _boxed_pdf(tmp_path / "b.pdf", 90, "cropbox")
    file = analyzed(path)
    file.findings.append(_manual(0, 30, 30, 200, 80))
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)


def test_after_hides_an_annotation_with_a_name(tmp_path):
    path = make_pdf(tmp_path / "a.pdf")
    with pymupdf.open(path) as doc:
        doc[0].add_freetext_annot(pymupdf.Rect(300, 600, 560, 640), "Persona Inventada Rojas")
        doc.save(tmp_path / "an.pdf")
    file = analyzed(tmp_path / "an.pdf")
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)


def _shared_resource_pdf(path: Path, kind: str) -> Path:
    """Two pages that draw the same Form XObject (letterhead) or the same image XObject."""
    with pymupdf.open() as src:
        head = src.new_page(width=400, height=80)
        head.insert_text((20, 40), f"Membrete con RUT {VALID_RUT}", fontsize=12)
        src.save(path.with_suffix(".head.pdf"))
    img = Image.new("RGB", (400, 80), "white")
    ImageDraw.Draw(img).text((20, 30), f"RUT {VALID_RUT}", fill=(0, 0, 0), font=font(20))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    with pymupdf.open() as doc, pymupdf.open(path.with_suffix(".head.pdf")) as head_doc:
        for _ in range(2):
            page = doc.new_page(width=400, height=300)
            if kind == "form":
                page.show_pdf_page(pymupdf.Rect(0, 0, 400, 80), head_doc, 0)
            else:
                page.insert_image(pymupdf.Rect(0, 0, 400, 80), stream=buf.getvalue())
        doc.save(path)
    return path


@pytest.mark.parametrize("kind", ["form", "image"])
def test_after_equals_export_with_a_resource_shared_by_two_pages(tmp_path, kind):
    path = _shared_resource_pdf(tmp_path / "shared.pdf", kind)
    file = _file(path, "pdf", [_manual(1, 10, 10, 390, 70)])
    for page in (0, 1):
        _assert_parity(RealEngine(), file, page, 1.0, tmp_path)


def test_after_reveals_and_redacts_a_hidden_layer(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        xref = doc.add_ocg("Capa oculta", on=False)
        page.insert_text((40, 60), f"RUT {VALID_RUT}", fontsize=12, oc=xref)
        page.insert_text((40, 200), "Texto neutro visible.", fontsize=12)
        doc.save(tmp_path / "ocg.pdf")
    file = analyzed(tmp_path / "ocg.pdf")
    assert any(f.type == "rut" for f in file.findings)
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)


@pytest.mark.parametrize("mode", ["exif6", "rgba", "palette"])
def test_after_equals_export_png(tmp_path, mode):
    img = Image.new("RGBA" if mode == "rgba" else "RGB", (300, 200), (255, 255, 255, 255) if mode == "rgba" else "white")
    ImageDraw.Draw(img).rectangle((20, 20, 120, 80), fill=(10, 10, 10) if mode != "rgba" else (10, 10, 10, 128))
    kwargs = {}
    if mode == "exif6":
        exif = Image.Exif()
        exif[0x0112] = 6
        kwargs["exif"] = exif.tobytes()
    if mode == "palette":
        img = img.convert("P")
    img.save(tmp_path / "p.png", **kwargs)
    file = _file(tmp_path / "p.png", "image", [_manual(0, 15, 15, 130, 90)])
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path)
    _assert_parity(RealEngine(), file, 0, 0.62, tmp_path)


def test_after_equals_export_tiff_page_2(tmp_path):
    path = _tiff(tmp_path / "t.tif", [(300, 200), (240, 320), (500, 260)])
    file = _file(path, "image", [_manual(1, 10, 10, 100, 50), _manual(2, 5, 5, 50, 50)])
    _assert_parity(RealEngine(), file, 1, 1.0, tmp_path)


@pytest.mark.parametrize("fmt,ext", [("JPEG", "jpg"), ("WEBP", "webp")])
def test_after_close_to_export_lossy(tmp_path, fmt, ext):
    img = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(img).text((20, 80), f"RUT {VALID_RUT}", fill=(0, 0, 0), font=font(18))
    img.save(tmp_path / f"f.{ext}", fmt, quality=92)
    file = _file(tmp_path / f"f.{ext}", "image", [_manual(0, 15, 70, 250, 110)])
    _assert_parity(RealEngine(), file, 0, 1.0, tmp_path, lossy=True)


def test_after_ignores_removed_suggested_and_other_pages(tmp_path):
    import numpy as np

    path = _tiff(tmp_path / "t.tif", [(300, 200), (300, 200)])
    engine = RealEngine()
    file = _file(path, "image", [])
    plain = _pixels(engine.render_result(file, 0, 1.0, []))
    kept = [
        _manual(0, 10, 10, 100, 50, fid="r", status="removed"),
        _manual(0, 10, 60, 100, 90, fid="s", status="suggested"),
        _manual(1, 10, 10, 100, 50, fid="o"),
    ]
    assert np.array_equal(_pixels(engine.render_result(file, 0, 1.0, kept)), plain)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_real.py -k after_ -v`
Expected: FAIL with `AttributeError: 'RealEngine' object has no attribute 'render_result'`.

- [ ] **Step 3: Implement**

In `anonymizer/engine/real.py` add at module level (after the imports):

```python
def _png(mode: str, size: tuple[int, int], samples: bytes) -> bytes:
    """PNG of raw pixels, encoded outside ``PDF_LOCK`` (lossless; level 1 is fast)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.frombytes(mode, size, samples).save(buf, "PNG", compress_level=1)
    return buf.getvalue()
```

`render_page`'s PDF branch copies the pixels inside the lock and encodes after it:

```python
            with PDF_LOCK:
                with pymupdf.open(file.path, filetype="pdf") as doc:
                    pdf.reveal_layers(doc)
                    pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                    size, samples = (pix.width, pix.height), bytes(pix.samples)
            return _png("RGB", size, samples)
```

and add:

```python
    def render_result(self, file: AnalyzedFile, page: int, zoom: float, findings: list[Finding]) -> bytes:
        """PNG of page ``page`` as it will be exported: the active findings of that page applied
        with the export's own redaction, on an in-memory copy. Nothing is written to disk."""
        zoom = max(0.05, min(8.0, float(zoom)))
        polygons = [f.polygon for f in findings if f.page == page and f.active]
        kind = file.kind or sniff(file.path)
        if kind == "pdf":
            import pymupdf

            from anonymizer.engine import pdf

            with PDF_LOCK:
                with pymupdf.open(file.path, filetype="pdf") as doc:
                    pdf.reveal_layers(doc)
                    rects = pdf.redaction_rects(doc, {page: polygons})
                    pdf.redact_page(doc, page, rects.get(page, []))
                    pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                    size, samples = (pix.width, pix.height), bytes(pix.samples)
            return _png("RGB", size, samples)
        import numpy as np
        from PIL import Image

        from anonymizer.engine import image

        arr = image.redact_frame(np.array(image.frame(file.path, page)), polygons)
        out = Image.fromarray(arr)
        if zoom != 1.0:
            out = out.resize((max(1, round(out.width * zoom)), max(1, round(out.height * zoom))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        out.save(buf, "PNG", compress_level=1)
        return buf.getvalue()
```

If a parity case fails, do not loosen the test: find the difference (order of operations, layer reveal, rotation) and fix `render_result` or the shared helper. If a PDF case is genuinely impossible to make equal, stop and report it with the measured difference.

- [ ] **Step 4: Run the parity tests and the render tests**

Run: `uv run pytest tests/test_engine_real.py -k "after_ or render" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add anonymizer/engine/real.py tests/test_engine_real.py
git commit -m "Render a page as it will be exported, with the export's redaction"
```

---

### Task 4: Development engine and the engine contract

**Files:**
- Modify: `anonymizer/engine/fake.py` (`_export_pdf`, `_export_image`, `render_page`; add `_redact_page`, `_fill`, `render_result`)
- Modify: `anonymizer/engine/__init__.py` (contract docstring, `Engine` protocol)
- Test: `tests/test_api.py` (new tests at the end, using `FakeEngine` directly)

**Interfaces:**
- Produces: `FakeEngine.render_result(file, page, zoom, findings) -> bytes`, same semantics as Task 3; `Engine.render_result` in the protocol.

- [ ] **Step 1: Write the failing tests**

```python
def _png_pixels(png: bytes):
    import numpy as np

    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))


def test_fake_engine_after_equals_its_export(tmp_path):
    import numpy as np

    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(rotation=90))
    engine = FakeEngine()
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(path))
    engine.analyze(file, [])
    after = _png_pixels(engine.render_result(file, 0, 1.0, list(file.findings)))
    result = engine.export(file, str(tmp_path / "out"))
    out = AnalyzedFile(id="o", name="a.pdf", path=result.output_path, kind="pdf")
    assert np.array_equal(after, _png_pixels(engine.render_page(out, 0, 1.0)))


def test_fake_engine_after_of_an_image(tmp_path):
    img = Image.new("RGB", (200, 100), "white")
    img.save(tmp_path / "f.png")
    file = AnalyzedFile(id="f1", name="f.png", path=str(tmp_path / "f.png"), kind="image", status="ready")
    finding = Finding(id="m", file_id="f1", page=0, type="manual",
                      polygon=[[10, 10], [60, 10], [60, 40], [10, 40]], detector="reviewer", status="added")  # fmt: skip
    pixels = _png_pixels(FakeEngine().render_result(file, 0, 1.0, [finding]))
    assert pixels[20:30, 20:50].max() == 0 and pixels[70:90, 100:190].min() == 255
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_api.py -k fake_engine_after -v`
Expected: FAIL with `AttributeError: 'FakeEngine' object has no attribute 'render_result'`.

- [ ] **Step 3: Implement**

In `fake.py`, move the body of the `_export_pdf` page loop into:

```python
    @staticmethod
    def _redact_page(doc: pymupdf.Document, n: int, findings: list[Finding]) -> None:
        """Real redaction of ``findings`` on page ``n`` and its annotations and form fields removed."""
        page = doc[n]
        to_page = pymupdf.Matrix(page.derotation_matrix)
        # Zones are placed on the unrotated page (see ``pdf.redact``): MuPDF misplaces them
        # on rotated pages whose CropBox or MediaBox does not start at (0, 0).
        rotation = page.rotation
        if rotation:
            page.set_rotation(0)
        for f in findings:
            x0, y0, x1, y1 = bbox_of(f.polygon)
            page.add_redact_annot((pymupdf.Rect(x0, y0, x1, y1) * to_page).normalize(), fill=(0, 0, 0))
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
        if rotation:
            page.set_rotation(rotation)
        for annot in list(page.annots() or []):
            page.delete_annot(annot)
        for widget in list(page.widgets() or []):
            page.delete_widget(widget)

    @staticmethod
    def _fill(frame: Image.Image, findings: list[Finding]) -> Image.Image:
        """Frame ``frame`` (view space, RGB) with each finding's polygon filled black."""
        draw = ImageDraw.Draw(frame)
        for f in findings:
            draw.polygon([(x, y) for x, y in f.polygon], fill=(0, 0, 0))
        return frame
```

`_export_pdf`'s loop becomes `for n in range(doc.page_count): self._redact_page(doc, n, by_page.get(n, []))`; `_export_image`'s loop body becomes `out = self._fill(ImageOps.exif_transpose(frame.copy()).convert("RGB"), [f for f in active if f.page == n])`. Then:

```python
    def render_result(self, file: AnalyzedFile, page: int, zoom: float, findings: list[Finding]) -> bytes:
        zoom = max(0.05, min(8.0, float(zoom)))
        mine = [f for f in findings if f.page == page and f.active]
        kind = file.kind or sniff(file.path)
        buf = io.BytesIO()
        if kind == "pdf":
            with PDF_LOCK:
                with pymupdf.open(file.path, filetype="pdf") as doc:
                    self._redact_page(doc, page, mine)
                    pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                    size, samples = (pix.width, pix.height), bytes(pix.samples)
            Image.frombytes("RGB", size, samples).save(buf, "PNG", compress_level=1)
            return buf.getvalue()
        with Image.open(file.path) as img:
            img.seek(page)
            out = self._fill(ImageOps.exif_transpose(img.copy()).convert("RGB"), mine)
        if zoom != 1.0:
            out = out.resize((max(1, round(out.width * zoom)), max(1, round(out.height * zoom))), Image.Resampling.LANCZOS)
        out.save(buf, "PNG", compress_level=1)
        return buf.getvalue()
```

In `anonymizer/engine/__init__.py`, add to the contract docstring after `render_page`:

```
``render_result(file, page, zoom, findings) -> bytes``
    PNG of one page as it will be exported (the "after" of the review): the engine keeps the
    findings of ``findings`` on that page that are active (neither "removed" nor "suggested"),
    the same selection ``export`` makes, and applies them with the export's own redaction to an
    in-memory copy. Nothing is written to disk. Same view space and zoom as ``render_page``.
```

and to `Engine`: `def render_result(self, file: AnalyzedFile, page: int, zoom: float, findings: list[Finding]) -> bytes: ...` (import `Finding` from the model).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_api.py -v` (background, log)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add anonymizer/engine/fake.py anonymizer/engine/__init__.py tests/test_api.py
git commit -m "Give the development engine the after render and document it in the contract"
```

---

### Task 5: `?redacted=true` on the page route

**Files:**
- Modify: `anonymizer/api/server.py` (`page_png` at about line 842; the module docstring's route list at about line 22)
- Test: `tests/test_api.py`, `tests/test_engine_real.py`

**Interfaces:**
- Consumes: `engine.render_result` (Tasks 3–4).
- Produces: `GET /api/files/{id}/pages/{n}.png?zoom=Z&redacted=true` → `image/png`; errors 404 `not_found`, 409 `not_viewable`, 409 `not_ready`, 409 `render`, 422 `invalid`. Extra query parameters (`v`) are ignored.

- [ ] **Step 1: Write the failing tests**

In `tests/test_api.py`:

```python
def test_redacted_page(client, tmp_path):
    file_id = upload(client, "a.pdf", make_pdf())
    assert client.get(f"/api/files/{file_id}/pages/0.png?redacted=true").json()["error"] == "not_ready"
    client.post("/api/process", json={"file_ids": [file_id]})
    assert wait_status(client, file_id)["status"] == "ready"
    url = f"/api/files/{file_id}/pages/0.png?zoom=0.5&redacted=true&v=abc"
    r = client.get(url)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (298, 421)
    assert client.get(f"/api/files/{file_id}/pages/3.png?redacted=true").status_code == 404
    assert client.get(f"/api/files/{file_id}/pages/0.png?redacted=true&zoom=0").status_code == 422
    plain = client.get(f"/api/files/{file_id}/pages/0.png?zoom=0.5").content
    assert _png_pixels(r.content).tolist() != _png_pixels(plain).tolist()  # the after has black zones
    file = client.get(f"/api/files/{file_id}").json()
    for f in file["findings"]:
        client.patch(f"/api/files/{file_id}/findings/{f['id']}", json={"action": "remove", "reason": "No es un dato personal"})
    nothing = client.get(url).content
    assert _png_pixels(nothing).tolist() == _png_pixels(plain).tolist()
```

In `tests/test_engine_real.py` (review state through the route, and nothing on disk):

```python
def test_redacted_route_equals_export_and_writes_nothing(tmp_path, monkeypatch):
    import hashlib
    import tempfile

    import numpy as np

    from anonymizer.api.server import create_app
    from tests.live_client import LiveClient
    from tests.test_api import TOKEN, upload, wait_status

    app = create_app(RealEngine(), TOKEN)
    with LiveClient(app) as client:
        client.headers["X-Session-Token"] = TOKEN
        file_id = upload(client, "informe.pdf", make_pdf(tmp_path / "a.pdf").read_bytes())
        client.post("/api/process", json={"file_ids": [file_id]})
        assert wait_status(client, file_id)["status"] == "ready"
        file = client.get(f"/api/files/{file_id}").json()
        signer = next(f for f in file["findings"] if f["text"] == SIGNER)
        client.patch(f"/api/files/{file_id}/findings/{signer['id']}", json={"action": "remove", "reason": "Otro motivo"})
        session = app.state.session
        working = Path(session.get(file_id).path)
        digest = hashlib.sha256(working.read_bytes()).hexdigest()
        listing = sorted((p.name, p.stat().st_mtime_ns) for p in Path(session.dir).iterdir())
        empty = tmp_path / "tmp"
        empty.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(empty))
        monkeypatch.setenv("TMP", str(empty))
        monkeypatch.setenv("TEMP", str(empty))
        after = client.get(f"/api/files/{file_id}/pages/0.png?redacted=true").content
        assert list(empty.iterdir()) == []
        assert sorted((p.name, p.stat().st_mtime_ns) for p in Path(session.dir).iterdir()) == listing
        assert hashlib.sha256(working.read_bytes()).hexdigest() == digest
        monkeypatch.undo()
        client.post(f"/api/files/{file_id}/confirm")
        dest = tmp_path / "salida"
        client.post("/api/export", json={"dest_dir": str(dest), "audit_pdf": False, "audit_json": False})
        out = AnalyzedFile(id="o", name="informe.pdf", path=str(dest / "informe.pdf"), kind="pdf")
        assert np.array_equal(_pixels(after), _pixels(RealEngine().render_page(out, 0, 1.0)))
```

(`app.state.session` is how `tests/test_api.py:344` reaches the session; check the attribute name there and use the same.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_api.py::test_redacted_page tests/test_engine_real.py::test_redacted_route_equals_export_and_writes_nothing -v`
Expected: FAIL (`redacted` is ignored, so the first assertion gets a 200 PNG instead of `not_ready`).

- [ ] **Step 3: Implement**

At the top of `server.py`, `from dataclasses import replace` (if not imported). Replace `page_png`:

```python
    @app.get("/api/files/{file_id}/pages/{page}.png")
    def page_png(file_id: str, page: int, zoom: float = 1.0, redacted: bool = False):
        if redacted:
            # One critical section: a re-process that starts in between empties the findings under
            # this lock, so the after can never be rendered with nothing applied.
            with session.lock:
                file = session.get(file_id)
                check_render(file, page, zoom)
                if file.status not in ("ready", "confirmed", "exported"):
                    raise ApiError(409, "not_ready", "Este archivo todavía no está listo para revisar.")
                snapshot = [
                    replace(f, text=None, history=[], polygon=[list(p) for p in f.polygon])
                    for f in file.findings
                    if f.page == page
                ]
            render = lambda: session.engine.render_result(file, page, min(max(zoom, 0.05), 8.0), snapshot)  # noqa: E731
        else:
            file = session.get(file_id)
            check_render(file, page, zoom)
            render = lambda: session.engine.render_page(file, page, min(max(zoom, 0.05), 8.0))  # noqa: E731
        try:
            png = render()
        except Exception:
            log.exception("render of %s page %d failed (redacted=%s)", file_id, page, redacted)
            raise ApiError(409, "render", "No se pudo mostrar esta página.") from None
        return Response(png, media_type="image/png")

    def check_render(file: AnalyzedFile, page: int, zoom: float) -> None:
        if not math.isfinite(zoom) or zoom <= 0:
            raise ApiError(422, "invalid", "El zoom no es válido.")
        if file.kind is None or file.status == "error":
            raise ApiError(409, "not_viewable", "Este archivo no se puede mostrar.")
        if page < 0 or (file.pages and page >= len(file.pages)):
            raise ApiError(404, "not_found", "Esa página no existe.")
```

Note the order with `redacted=true`: the spec lists zoom, not-viewable, not-ready, page range. A file still `queued` has `kind` set and no pages: `check_render` passes, then `not_ready` fires — what the first test expects. Add the route to the module docstring: `GET /api/files/{id}/pages/{n}.png?zoom=1.5&redacted=true  image/png (the page as it will be exported)`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_api.py tests/test_engine_real.py -v` (background, log)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add anonymizer/api/server.py tests/test_api.py tests/test_engine_real.py
git commit -m "Serve the redacted render of a page for the review's after column"
```

---

### Task 6: `review-core.js`, the pure logic of the new viewer

**Files:**
- Create: `anonymizer/ui/review-core.js`
- Create: `tests/ui/review-core.test.js`
- Create: `tests/test_ui_core.py`

**Interfaces:**
- Produces (`globalThis.ReviewCore`, and `module.exports` under Node):
  - `isActive(f) -> boolean` — status neither `removed` nor `suggested`.
  - `pageVersions(findings, pageCount) -> string[]` — per page, an 8-hex-digit FNV-1a hash of its active findings sorted by id (`id|status|polygon rounded to 0.01`).
  - `columns({areaWidth, gap, after, minColumn = 420}) -> {cols, stacked, colWidth}`.
  - `fitScales(pages, {colWidth, rot}) -> number[]` — `pages` items `{width, height, unit}`; PDF (`unit !== "px"`): one scale for all = `min(1.6, colWidth / widest)`; images: each `min(1, colWidth / its width)`; widths rotated when `rot % 180`.
  - `effectiveScale(fit, zoom) -> number` — `clamp(fit * zoom, 0.05, 8)`.
  - `displaySize(page, scale, rot) -> {w, h}`.
  - `zoneRect(bbox, scale, rot, page) -> {x, y, w, h}` — a view-space bbox `{x, y, w, h}` to display pixels inside the row's cell (rotation-aware, see code).
  - `pointToPage(ox, oy, scale, rot, page) -> {x, y}` — inverse, clamped to the page, in view-space units.
  - `requestZoom(scale, dpr, page) -> number`.
  - `imageKey({file, gen, page, zoom, side, version}) -> string`; `acceptResponse(wanted, got) -> boolean`.
  - `planQueue(jobs, {inView, inMargin, center}) -> jobs` — `jobs` items `{row, side, edited}`; drops jobs whose row is not in `inMargin` (a `Set`); sorts.
  - `releasePlan(loaded, {keep, center, budgetMp = 150}) -> number[]` — `loaded` items `{row, mp}`; rows to release.
  - `currentRow(rows, viewTop, viewHeight) -> number` — `rows` items `{top, height}` in scroll coordinates; `viewTop`/`viewHeight` describe the visible area below the sticky header.
  - `anchorOf(rows, viewTop, viewHeight) -> {row, fy}`; `scrollTopFor(anchor, rows, viewHeight, headerHeight) -> number`.
  - `scrollTarget(zone, row, view, margin = 48) -> number | null` — `zone` `{y, h}` relative to the row's top, `row` `{top, height}`, `view` `{scrollTop, headerHeight, height}`; returns the new `scrollTop` or `null` when the zone is already well inside; the zone is clamped to the row first (off-page findings).
  - `isLongScroll(from, to, viewHeight) -> boolean` — `Math.abs(to - from) > 2 * viewHeight`.

- [ ] **Step 1: Write the failing Node tests**

`tests/ui/review-core.test.js`:

```js
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const C = require("../../anonymizer/ui/review-core.js");

const box = (x, y, w, h) => [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
const f = (id, page, status = "proposed", polygon = box(1, 2, 3, 4)) => ({ id, page, status, polygon });

test("active statuses", () => {
  assert.equal(C.isActive({ status: "proposed" }), true);
  assert.equal(C.isActive({ status: "added" }), true);
  assert.equal(C.isActive({ status: "removed" }), false);
  assert.equal(C.isActive({ status: "suggested" }), false);
});

test("page versions change only for the page whose active findings changed", () => {
  const a = [f("a", 0), f("b", 1), f("c", 1, "suggested")];
  const v1 = C.pageVersions(a, 3);
  assert.equal(v1.length, 3);
  assert.match(v1[0], /^[0-9a-f]{8}$/);
  const v2 = C.pageVersions([f("a", 0), f("b", 1, "removed"), f("c", 1, "suggested")], 3);
  assert.equal(v2[0], v1[0]);
  assert.notEqual(v2[1], v1[1]);
  assert.equal(v2[2], v1[2]);
  const v3 = C.pageVersions([f("c", 1, "suggested"), f("b", 1), f("a", 0)], 3);
  assert.deepEqual(v3, v1); // order of the list does not matter
  const v4 = C.pageVersions([f("a", 0, "proposed", box(1, 2, 3, 5)), f("b", 1)], 3);
  assert.notEqual(v4[0], v1[0]); // a moved polygon changes the version
});

test("columns stack under 420 px per column", () => {
  assert.deepEqual(C.columns({ areaWidth: 1000, gap: 24, after: true }), { cols: 2, stacked: false, colWidth: 488 });
  assert.equal(C.columns({ areaWidth: 800, gap: 24, after: true }).stacked, true);
  assert.equal(C.columns({ areaWidth: 800, gap: 24, after: true }).colWidth, 800);
  assert.deepEqual(C.columns({ areaWidth: 600, gap: 24, after: false }), { cols: 1, stacked: false, colWidth: 600 });
});

test("fit: one scale for a PDF (widest page), per frame for images", () => {
  const pdf = [{ width: 595, height: 842, unit: "pt" }, { width: 842, height: 595, unit: "pt" }];
  assert.deepEqual(C.fitScales(pdf, { colWidth: 421, rot: 0 }), [0.5, 0.5]);
  assert.deepEqual(C.fitScales(pdf, { colWidth: 2000, rot: 0 }), [1.6, 1.6]);
  assert.deepEqual(C.fitScales(pdf, { colWidth: 421, rot: 90 }), [0.5, 0.5]);
  const tif = [{ width: 2480, height: 3508, unit: "px" }, { width: 1240, height: 1754, unit: "px" }, { width: 300, height: 200, unit: "px" }];
  assert.deepEqual(C.fitScales(tif, { colWidth: 620, rot: 0 }), [0.25, 0.5, 1]);
  assert.equal(C.effectiveScale(0.5, 100), 8);
  assert.equal(C.effectiveScale(0.01, 1), 0.05);
});

test("display size and zone rectangles follow the rotation", () => {
  const page = { width: 200, height: 100 };
  assert.deepEqual(C.displaySize(page, 2, 0), { w: 400, h: 200 });
  assert.deepEqual(C.displaySize(page, 2, 90), { w: 200, h: 400 });
  const b = { x: 10, y: 20, w: 30, h: 5 };
  assert.deepEqual(C.zoneRect(b, 1, 0, page), { x: 10, y: 20, w: 30, h: 5 });
  assert.deepEqual(C.zoneRect(b, 1, 90, page), { x: 75, y: 10, w: 5, h: 30 });
  assert.deepEqual(C.zoneRect(b, 1, 180, page), { x: 160, y: 75, w: 30, h: 5 });
  assert.deepEqual(C.zoneRect(b, 1, 270, page), { x: 20, y: 160, w: 5, h: 30 });
  for (const rot of [0, 90, 180, 270]) {
    const r = C.zoneRect(b, 2, rot, page);
    const p = C.pointToPage(r.x + 0.001, r.y + 0.001, 2, rot, page);
    const q = C.pointToPage(r.x + r.w - 0.001, r.y + r.h - 0.001, 2, rot, page);
    const xs = [p.x, q.x].sort((m, n) => m - n), ys = [p.y, q.y].sort((m, n) => m - n);
    assert.ok(Math.abs(xs[0] - 10) < 0.01 && Math.abs(xs[1] - 40) < 0.01, `rot ${rot} x`);
    assert.ok(Math.abs(ys[0] - 20) < 0.01 && Math.abs(ys[1] - 25) < 0.01, `rot ${rot} y`);
  }
  assert.deepEqual(C.pointToPage(-50, 9999, 1, 0, page), { x: 0, y: 100 }); // clamped
});

test("request zoom: dpr, 9000 px cap, rounding", () => {
  const a4 = { width: 595, height: 842 };
  assert.equal(C.requestZoom(1, 1.25, a4), 1.25);
  assert.equal(C.requestZoom(8, 2, a4), Math.round((9000 / 842) * 100) / 100);
  assert.equal(C.requestZoom(0.001, 1, a4), 0.02);
});

test("image keys: a response for another key is dropped", () => {
  const want = C.imageKey({ file: "f", gen: 2, page: 3, zoom: 1.25, side: "after", version: "abc" });
  assert.equal(C.acceptResponse(want, want), true);
  for (const other of [
    { file: "g", gen: 2, page: 3, zoom: 1.25, side: "after", version: "abc" },
    { file: "f", gen: 1, page: 3, zoom: 1.25, side: "after", version: "abc" },
    { file: "f", gen: 2, page: 3, zoom: 1.5, side: "after", version: "abc" },
    { file: "f", gen: 2, page: 3, zoom: 1.25, side: "after", version: "abd" },
    { file: "f", gen: 2, page: 3, zoom: 1.25, side: "before", version: "" },
  ]) assert.equal(C.acceptResponse(want, C.imageKey(other)), false);
  assert.equal(C.acceptResponse(null, want), false);
});

test("queue: drops rows outside the margin and orders the rest", () => {
  const jobs = [
    { row: 9, side: "after" }, { row: 9, side: "before" },
    { row: 5, side: "after" }, { row: 5, side: "before" },
    { row: 6, side: "after", edited: true },
    { row: 7, side: "before" }, { row: 30, side: "before" },
  ];
  const out = C.planQueue(jobs, { inView: new Set([5, 6]), inMargin: new Set([4, 5, 6, 7, 8, 9]), center: 5 });
  assert.deepEqual(out.map((j) => `${j.row}${j.side[0]}`), ["5b", "6a", "5a", "7b", "9b", "9a"]);
});

test("release: far rows first, then the farthest until under budget", () => {
  const loaded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((row) => ({ row, mp: 20 }));
  const keep = new Set([3, 4, 5, 6, 7, 8, 9, 10]);
  const out = C.releasePlan(loaded, { keep, center: 6, budgetMp: 150 });
  assert.deepEqual(out.slice(0, 3).sort((a, b) => a - b), [0, 1, 2]);
  assert.ok(out.includes(10) || out.includes(3));
  const kept = loaded.filter((l) => !out.includes(l.row)).reduce((s, l) => s + l.mp, 0);
  assert.ok(kept <= 150);
  assert.ok(!out.includes(6)); // never the row at the center
});

test("current row, anchors and scroll targets", () => {
  const rows = [{ top: 0, height: 500 }, { top: 520, height: 500 }, { top: 1040, height: 300 }];
  assert.equal(C.currentRow(rows, 0, 400), 0);
  assert.equal(C.currentRow(rows, 400, 400), 1);
  const anchor = C.anchorOf(rows, 600, 400);
  assert.equal(anchor.row, 1);
  assert.ok(Math.abs(C.scrollTopFor(anchor, rows, 400, 0) - 600) < 0.5);
  const view = { scrollTop: 0, headerHeight: 40, height: 440 };
  assert.equal(C.scrollTarget({ y: 100, h: 20 }, rows[0], view), null); // well inside
  const t = C.scrollTarget({ y: 450, h: 20 }, rows[1], view); // below the view
  assert.ok(Math.abs(t - (520 + 460 - 40 - 200)) < 0.5);
  const off = C.scrollTarget({ y: 900, h: 50 }, rows[0], view); // off-page: clamped to its row
  assert.ok(off <= 500, "stays in its own row");
  assert.equal(C.isLongScroll(0, 900, 400), true);
  assert.equal(C.isLongScroll(0, 700, 400), false);
});
```

`tests/test_ui_core.py`:

```python
"""The pure logic of the review viewer (anonymizer/ui/review-core.js), tested with node --test."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_review_core():
    result = subprocess.run(
        ["node", "--test", str(ROOT / "tests" / "ui" / "review-core.test.js")],
        capture_output=True, text=True, timeout=120, check=False, cwd=ROOT,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 2: Run them to verify they fail**

Run: `node --test tests/ui/review-core.test.js`
Expected: FAIL with `Cannot find module '../../anonymizer/ui/review-core.js'`.

- [ ] **Step 3: Implement `anonymizer/ui/review-core.js`**

```js
/* Pure logic of the review viewer: geometry, page versions, image keys, the request queue,
 * the memory budget and scroll anchors. No DOM access: app.js calls these, and
 * tests/ui/review-core.test.js tests them with node --test. Loaded before app.js as a classic
 * script (it sets globalThis.ReviewCore; under Node it is a CommonJS module).
 */
"use strict";

(function (root) {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const round2 = (v) => Math.round(v * 100) / 100;

  const isActive = (f) => f.status !== "removed" && f.status !== "suggested";

  /** 32-bit FNV-1a of a string, as 8 hex digits. */
  function fnv(text) {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i += 1) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
  }

  /** One version per page: changes when an active finding of that page appears, goes or moves. */
  function pageVersions(findings, pageCount) {
    const parts = Array.from({ length: pageCount }, () => []);
    for (const f of findings || []) {
      if (!isActive(f) || !(f.page >= 0 && f.page < pageCount)) continue;
      const poly = (f.polygon || []).map(([x, y]) => `${round2(x)},${round2(y)}`).join(";");
      parts[f.page].push(`${f.id}|${f.status}|${poly}`);
    }
    return parts.map((p) => fnv(p.sort().join("\n")));
  }

  function columns({ areaWidth, gap, after, minColumn = 420 }) {
    if (!after) return { cols: 1, stacked: false, colWidth: areaWidth };
    const half = (areaWidth - gap) / 2;
    if (half < minColumn) return { cols: 1, stacked: true, colWidth: areaWidth };
    return { cols: 2, stacked: false, colWidth: half };
  }

  const turned = (rot) => rot % 180 !== 0;
  const shownWidth = (p, rot) => (turned(rot) ? p.height : p.width);

  /** Fit-to-column scale per page: one for a PDF (its widest page), one per frame for images. */
  function fitScales(pages, { colWidth, rot }) {
    if (!pages.length) return [];
    if (pages.some((p) => p.unit !== "px")) {
      const widest = Math.max(...pages.map((p) => shownWidth(p, rot) || 1));
      const s = clamp(colWidth / widest, 0.05, 1.6);
      return pages.map(() => s);
    }
    return pages.map((p) => clamp(colWidth / (shownWidth(p, rot) || 1), 0.05, 1));
  }

  const effectiveScale = (fit, zoom) => clamp(fit * zoom, 0.05, 8);

  function displaySize(page, scale, rot) {
    const w = page.width * scale, h = page.height * scale;
    return turned(rot) ? { w: h, h: w } : { w, h };
  }

  /** A view-space box to display pixels in its cell, with the cell's rotation transform
   *  (90: translate(H, 0) rotate(90deg); 180: translate(W, H) rotate(180deg); 270: translate(0, W) rotate(270deg)). */
  function zoneRect(b, scale, rot, page) {
    const W = page.width * scale, H = page.height * scale;
    const x = b.x * scale, y = b.y * scale, w = b.w * scale, h = b.h * scale;
    switch (rot) {
      case 90: return { x: H - y - h, y: x, w: h, h: w };
      case 180: return { x: W - x - w, y: H - y - h, w, h };
      case 270: return { x: y, y: W - x - w, w: h, h: w };
      default: return { x, y, w, h };
    }
  }

  /** A point of the cell (display pixels) back to view-space units, clamped to the page. */
  function pointToPage(ox, oy, scale, rot, page) {
    const W = page.width * scale, H = page.height * scale;
    let x, y;
    switch (rot) {
      case 90: x = oy; y = H - ox; break;
      case 180: x = W - ox; y = H - oy; break;
      case 270: x = W - oy; y = ox; break;
      default: x = ox; y = oy;
    }
    return { x: clamp(x, 0, W) / scale, y: clamp(y, 0, H) / scale };
  }

  function requestZoom(scale, dpr, page) {
    let zoom = scale * (dpr || 1);
    zoom = Math.min(zoom, 9000 / Math.max(page.width, page.height, 1));
    return Math.max(0.02, round2(zoom));
  }

  const imageKey = ({ file, gen, page, zoom, side, version }) =>
    [file, gen, page, zoom, side, side === "after" ? version || "" : ""].join("|");
  const acceptResponse = (wanted, got) => !!wanted && wanted === got;

  /** Jobs whose row left the margin are dropped; then: befores in view, edited afters, afters in
   *  view, then the margin by distance to the center (before first). */
  function planQueue(jobs, { inView, inMargin, center }) {
    const rank = (j) => {
      if (inView.has(j.row)) {
        if (j.side === "before") return 0;
        return j.edited ? 1 : 2;
      }
      if (j.side === "after" && j.edited) return 1;
      return 3;
    };
    return jobs
      .filter((j) => inMargin.has(j.row))
      .map((j, i) => ({ j, i, r: rank(j) }))
      .sort((a, b) => a.r - b.r
        || (a.r === 3 ? Math.abs(a.j.row - center) - Math.abs(b.j.row - center) : 0)
        || (a.j.side === b.j.side ? 0 : a.j.side === "before" ? -1 : 1)
        || a.i - b.i)
      .map((x) => x.j);
  }

  /** Rows whose images are released: every loaded row outside ``keep``, then the farthest from
   *  the center until the total is within the budget. The center row is never released. */
  function releasePlan(loaded, { keep, center, budgetMp = 150 }) {
    const out = loaded.filter((l) => !keep.has(l.row)).map((l) => l.row);
    let rest = loaded.filter((l) => keep.has(l.row));
    let total = rest.reduce((s, l) => s + l.mp, 0);
    rest = rest.sort((a, b) => Math.abs(b.row - center) - Math.abs(a.row - center));
    for (const l of rest) {
      if (total <= budgetMp) break;
      if (l.row === center) continue;
      out.push(l.row);
      total -= l.mp;
    }
    return out;
  }

  /** The row crossing the vertical center of the visible area (or the nearest one). */
  function currentRow(rows, viewTop, viewHeight) {
    const mid = viewTop + viewHeight / 2;
    let best = 0, dist = Infinity;
    rows.forEach((r, i) => {
      const d = mid < r.top ? r.top - mid : mid > r.top + r.height ? mid - r.top - r.height : 0;
      if (d < dist) { dist = d; best = i; }
    });
    return best;
  }

  function anchorOf(rows, viewTop, viewHeight) {
    const row = currentRow(rows, viewTop, viewHeight);
    const r = rows[row];
    const mid = viewTop + viewHeight / 2;
    return { row, fy: r && r.height ? clamp((mid - r.top) / r.height, 0, 1) : 0 };
  }

  /** The scrollTop that puts the anchor back at the center of the visible area. */
  function scrollTopFor(anchor, rows, viewHeight, headerHeight) {
    const r = rows[clamp(anchor.row, 0, rows.length - 1)];
    if (!r) return 0;
    return Math.max(0, r.top + anchor.fy * r.height - viewHeight / 2 - headerHeight);
  }

  /** New scrollTop when the zone is closer than ``margin`` to the visible area's edges (below the
   *  sticky header); the zone is first clamped to its own row. null when no scroll is needed. */
  function scrollTarget(zone, row, view, margin = 48) {
    const y0 = clamp(zone.y, 0, row.height), y1 = clamp(zone.y + zone.h, 0, row.height);
    const top = row.top + y0, bottom = row.top + Math.max(y1, y0);
    const visTop = view.scrollTop + view.headerHeight;
    const visBottom = view.scrollTop + view.height;
    if (top >= visTop + margin && bottom <= visBottom - margin) return null;
    const visHeight = view.height - view.headerHeight;
    return Math.max(0, (top + bottom) / 2 - view.headerHeight - visHeight / 2);
  }

  const isLongScroll = (from, to, viewHeight) => Math.abs(to - from) > 2 * viewHeight;

  const ReviewCore = {
    isActive, pageVersions, columns, fitScales, effectiveScale, displaySize, zoneRect, pointToPage,
    requestZoom, imageKey, acceptResponse, planQueue, releasePlan, currentRow, anchorOf, scrollTopFor,
    scrollTarget, isLongScroll,
  };
  if (typeof module === "object" && module.exports) module.exports = ReviewCore;
  else root.ReviewCore = ReviewCore;
})(typeof globalThis !== "undefined" ? globalThis : this);
```

Check the expected numbers in the tests against this code by running them; when a test and the code disagree, decide which one matches the spec and fix that one (the tests above encode the spec's rules; the arithmetic of `scrollTarget`'s expectation is `(top + bottom) / 2 - header - visHeight / 2` with `top = 970`, `bottom = 990`: `980 - 40 - 200 = 740`).

- [ ] **Step 4: Run the tests**

Run: `node --test tests/ui/review-core.test.js` then `uv run pytest tests/test_ui_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add anonymizer/ui/review-core.js tests/ui/review-core.test.js tests/test_ui_core.py
git commit -m "Add the pure logic of the scrolling before/after viewer"
```

---

### Task 7: Markup and styles of the new Revisar

**Files:**
- Modify: `anonymizer/ui/index.html:91-147` (screen 3), `:13` (scripts)
- Modify: `anonymizer/ui/app.css` (review section, tokens at `:root` and both dark blocks, media queries)
- Test: `tests/test_ui_static.py`

**Interfaces:**
- Produces (element ids `app.js` uses from Task 8 on): `#rv-bar`, `#rv-file` (select), `#rv-chip`, `#rv-prev`, `#rv-next`, `#rv-page` (number input), `#rv-pages` (the "de N" text), `#rv-status` (status line), `#b-zout`, `#b-zfit`, `#b-zin`, `#b-rot`, `#b-draw` (with `.lbl` child), `#b-view`, `#viewport`, `#vp-head`, `#rows`, `#hpan` (shared horizontal scrollbar) with `#hpan-inner`, `#review-loading`; kept: `#review-empty`, `#review`, `.col-find` children, `.keys`, `.verify`.

- [ ] **Step 1: Update the static tests first**

In `tests/test_ui_static.py`: add `"review-core.js"` to the parametrized remote-URL test list and to the syntax test (run `node --check` on both scripts); in `test_new_spanish_strings_are_present` add for `index.html`: `"Mostrar el después"`, `"Páginas del documento"`, `"Antes"`, `"Después"`, `"Dibujar zona"`; for `app.js`: `"Dibujando · Esc para salir"`, `"Cargando…"`, `"Actualizando…"`, `"No se pudo mostrar el resultado de esta página"`, `"Reintentar"`, `"Ver página"`, `"Cargando el archivo…"`; for `app.css`: `"--draw:"`, `"--draw-ink:"`, `"--draw-edge:"`, `".prow"`, `".vp-head"`; and remove `".result .zone.suggested"` from the CSS list. Add:

```python
def test_review_core_loads_before_app():
    html = read("index.html")
    assert html.index('src="review-core.js"') < html.index('src="app.js"')


def test_old_viewer_is_gone():
    html, js = read("index.html"), read("app.js")
    for gone in ('id="thumbs"', 'id="rfiles"', 'id="pagebox"', "Ver como quedará"):
        assert gone not in html
    for gone in ("renderThumbs", "thumbCache", "setPage(", "rv.result"):
        assert gone not in js
```

(The strings for `app.js` and `test_old_viewer_is_gone` will pass only after Tasks 8–12; run only the `index.html`/`app.css` checks in this task with `-k "index or css or review_core_loads"` and leave the rest failing until Task 12 — note this in the commit message.)

- [ ] **Step 2: Replace the markup of screen 3**

`index.html` line 13 becomes two lines: `<script src="review-core.js" defer></script>` then `<script src="app.js" defer></script>`. The `#review` block becomes:

```html
      <div class="review" id="review">
        <section class="col col-doc" aria-label="Documento">
          <div class="rv-bar" id="rv-bar">
            <label class="fname" for="rv-file">Archivo</label>
            <select id="rv-file"></select>
            <span id="rv-chip"></span>
            <button type="button" class="tool filestep" id="rv-prev" aria-label="Archivo anterior" title="Archivo anterior">‹</button>
            <button type="button" class="tool filestep" id="rv-next" aria-label="Archivo siguiente" title="Archivo siguiente">›</button>
            <label class="pageind" for="rv-page">pág. <input type="number" class="pagefield mono" id="rv-page" min="1" value="1" inputmode="numeric"> <span id="rv-pages">de 1</span></label>
          </div>
          <p class="rv-status" id="rv-status" role="status" hidden></p>
          <div class="tools" role="toolbar" aria-label="Herramientas del visor">
            <button type="button" class="tool" id="b-zout" aria-label="Alejar (−)">−</button>
            <button type="button" class="tool zoomval mono" id="b-zfit" title="Ajustar al ancho">100 %</button>
            <button type="button" class="tool" id="b-zin" aria-label="Acercar (+)">+</button>
            <button type="button" class="tool" id="b-rot">Girar vista <kbd>R</kbd></button>
            <span class="sep" aria-hidden="true"></span>
            <button type="button" class="tool draw" id="b-draw" aria-pressed="false"><span aria-hidden="true">✎</span> <span class="lbl">Dibujar zona</span> <kbd>D</kbd></button>
            <button type="button" class="tool" id="b-view" aria-pressed="true">Mostrar el después <kbd>V</kbd></button>
          </div>
          <div class="viewport" id="viewport" tabindex="0" role="region" aria-label="Páginas del documento">
            <div class="vp-head" id="vp-head" aria-hidden="true"><span class="h-before">Antes</span><span class="h-after">Después</span></div>
            <div class="rows" id="rows"></div>
            <p class="loading fmeta" id="review-loading" hidden>Cargando el archivo…</p>
          </div>
          <div class="hpan" id="hpan" hidden><div id="hpan-inner"></div></div>
          <div>
            <div class="keys" aria-label="Atajos de teclado"><span><kbd>J</kbd> <kbd>K</kbd> siguiente y anterior</span><span><kbd>Supr</kbd> quitar censura</span><span><kbd>D</kbd> dibujar zona</span><span><kbd>+</kbd> <kbd>−</kbd> zoom</span><span><kbd>R</kbd> girar vista</span><span><kbd>V</kbd> mostrar u ocultar el después</span></div>
            <div class="verify">
              <div class="msg"><span class="dot" id="vdot" aria-hidden="true"></span><div><strong id="vtitle">Revisión lista para confirmar</strong><div class="fmeta" id="vsum"></div></div></div>
              <div class="row" id="vactions"><button type="button" class="btn primary" id="b-confirm">Confirmar este archivo</button></div>
            </div>
          </div>
        </section>
        <!-- the findings column (aside.col-find) stays exactly as it is -->
```

- [ ] **Step 3: Styles**

In `app.css`: add to `:root` `--draw: #FFD23F; --draw-ink: #1F1A00; --draw-edge: #8A6100;` and to both dark-theme blocks `--draw-edge: #FFD23F;`. Remove the `.col-files`, `.fbtn`, `.thumbs`, `.thumb*`, `.pagebox`, `.pageinner`, `.pageimg`, `.result .zone*` rules. `.review` gets two columns: `grid-template-columns: minmax(0, 1fr) 340px` (1700 px: `minmax(0, 1fr) 400px`; 1100 px: `minmax(0, 1fr) 300px`). Add:

```css
.rv-bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 8px 12px; background: var(--surface); border-bottom: 1px solid var(--line); }
.rv-bar select { min-width: 0; max-width: min(520px, 100%); text-overflow: ellipsis; border: 1px solid var(--line); background: var(--surface); border-radius: var(--r-s); padding: 5px 8px; font-size: 14px; }
.rv-bar .filestep { font-size: 17px; line-height: 1; padding: 3px 11px 5px; }
.pageind { margin-left: auto; font-size: 13px; color: var(--muted); white-space: nowrap; display: inline-flex; align-items: center; gap: 6px; }
.pagefield { width: 4.6em; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: var(--r-s); padding: 3px 6px; font-size: 13px; text-align: right; }
.rv-status { margin: 0; padding: 6px 12px; font-size: 13px; background: var(--bad-soft); color: var(--ink); border-bottom: 1px solid var(--line); }
.tool.draw { background: var(--draw); border-color: var(--draw-edge); color: var(--draw-ink); font-weight: 700; }
.tool.draw:hover:not(:disabled) { background: var(--draw); filter: brightness(.95); }
.tool.draw kbd { background: rgba(255, 255, 255, .6); color: var(--draw-ink); border-color: rgba(0, 0, 0, .3); }
.tool.draw[aria-pressed="true"] { background: var(--draw-ink); color: var(--draw); border-color: var(--draw-edge); }
.tool.draw[aria-pressed="true"] kbd { background: transparent; color: inherit; border-color: currentColor; }

/* One vertical scroll for every page. --cw: the width of a column; --gap between columns. */
.viewport { --cw: 600px; --gap: 24px; overflow-y: auto; overflow-x: hidden; scrollbar-gutter: stable; position: relative; min-height: 0; padding: 0; overscroll-behavior: contain; }
.viewport:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
.vp-head { position: sticky; top: 0; z-index: 6; display: grid; grid-template-columns: var(--cw) var(--cw); column-gap: var(--gap); justify-content: center; padding: 8px 24px; background: var(--surface-2); border-bottom: 1px solid var(--line); }
.vp-head span, .cell .cap { font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); text-align: center; }
.rows { padding: 16px 24px 24px; }
.prow { display: grid; grid-template-columns: var(--cw) var(--cw); column-gap: var(--gap); row-gap: 8px; justify-content: center; margin-bottom: 24px; }
.plabel { grid-column: 1 / -1; text-align: center; font-size: 12px; color: var(--muted); }
.cell { position: relative; overflow: hidden; display: grid; justify-items: center; gap: 6px; }
.cell .cap { display: none; }
.pbox { position: relative; background: var(--page); box-shadow: 0 2px 10px rgba(0, 0, 0, .18); }
.pinner { position: absolute; left: 0; top: 0; transform-origin: 0 0; }
.pinner img { display: block; max-width: none; user-select: none; -webkit-user-drag: none; }
.cell .zones { position: absolute; inset: 0; overflow: hidden; }
.cell .state { position: absolute; left: 50%; top: 12px; transform: translateX(-50%); z-index: 4; }
.cell.after.updating img { opacity: .45; }
.cell .fail { position: absolute; inset: 0; display: grid; place-content: center; gap: 8px; padding: 16px; text-align: center; background: var(--surface); color: var(--ink); }
.viewport.drawing .cell.before .pbox { cursor: crosshair; touch-action: none; outline: 3px dashed var(--draw-edge); outline-offset: 3px; }
.viewport.drawing .zone { pointer-events: none; }
.viewport.no-after .cell.after, .viewport.no-after .vp-head .h-after { display: none; }
.viewport.no-after .prow, .viewport.no-after .vp-head { grid-template-columns: var(--cw); }
.viewport.stacked .vp-head { display: none; }
.viewport.stacked .prow { grid-template-columns: var(--cw); }
.viewport.stacked .cell .cap { display: block; }
.hpan { overflow-x: auto; overflow-y: hidden; height: 14px; background: var(--surface-2); border-top: 1px solid var(--line); }
#hpan-inner { height: 1px; }
```

`.col-doc` gets `grid-template-rows: auto auto auto minmax(0, 1fr) auto auto`. The 860 px block loses `.col-files` and `.thumbs`, keeps `.viewport { max-height: 75vh; }`.

- [ ] **Step 4: Run the static tests for HTML and CSS**

Run: `uv run pytest tests/test_ui_static.py -k "index or css or remote or review_core_loads or local_assets" -v`
Expected: PASS for those; the app.js string checks and `test_old_viewer_is_gone` still fail (they belong to Tasks 8–12).

- [ ] **Step 5: Commit**

```bash
git add anonymizer/ui/index.html anonymizer/ui/app.css tests/test_ui_static.py
git commit -m "Lay out the scrolling before/after review (markup and styles)"
```

---

### Task 8: File bar and the file lifecycle in `app.js`

**Files:**
- Modify: `anonymizer/ui/app.js` — state `S.rv` (line ~248), `refreshState` (~368-379), `renderAll`, `go`, `openReview`, `closeReviewFile`, `loadReviewFile`, `renderReview`, `renderReviewFiles` (replaced), `renderVerify` ("Siguiente archivo por revisar"), the Procesar "Revisar" buttons and `#b-review-ready`, `init`.

**Interfaces:**
- Consumes: Task 7 ids.
- Produces: `S.rv` = `{ id, file, gen, sel, hidden, zoom (factor, 1 = fit), rot, draw, after (true), seen, loadingId, rows: [], versions: [], pan: 0 }`; functions `openFile(id, { keepView = false } = {})`, `teardownRows()`, `dropReviewFile()`, `renderFileBar()`, `buildRows()` (stub in this task: creates nothing; Task 9 fills it), `focusViewport()`.

- [ ] **Step 1:** Remove `renderReviewFiles` and the `#rfiles` code. Add `statusText(f)` returning the text `statusChip` shows (refactor `statusChip` to use it), with `"Procesando"` for `processing`.
- [ ] **Step 2:** Add `renderFileBar()`: builds the `<select id="rv-file">` options once (on first call, or when the set of file ids changes — then rebuilt with the same value selected); otherwise updates in place only the options whose text (`${name} · ${statusText(f)}`) or `disabled` (`!REVIEWABLE.has(f.status)`) changed; sets `select.title` to the open file's full name; renders the open file's `statusChip` into `#rv-chip`; enables `#rv-prev`/`#rv-next` only when a reviewable file exists before/after the open one in `S.files` order (no wrap). Call it from `renderAll()` on every poll (any screen), never rebuilding the element.
- [ ] **Step 3:** Wire the select: on `change`, clear a pending timer and set one of 300 ms that calls `openFile(select.value)`; ‹ › call `openFile` on the previous/next reviewable file and keep focus on themselves unless they become disabled (then `focusViewport()`).
- [ ] **Step 4:** Implement the lifecycle exactly as spec 6.3:

```js
  function teardownRows() {
    S.rv.gen += 1;
    stopObserver();          // Task 10 defines it; until then a no-op function
    clearQueue();            // Task 10
    for (const row of S.rv.rows) releaseRow(row, { all: true }); // Task 10; until then revoke blob URLs here
    S.rv.rows = [];
    $("#rows").replaceChildren();
  }

  /** The open file can no longer be reviewed (processed again, removed, failed): forget it. */
  function dropReviewFile() {
    const id = S.rv.id;
    teardownRows();
    if (id) S.rv.seen.delete(id);
    S.rv.id = null; S.rv.file = null; S.rv.sel = null; S.rv.versions = [];
  }

  async function openFile(id, { keepView = false } = {}) {
    const sum = fileById(id);
    if (!sum || !REVIEWABLE.has(sum.status)) return;
    if (!keepView || id !== S.rv.id) {
      teardownRows();
      Object.assign(S.rv, { id, file: null, sel: null, zoom: 1, rot: 0, pan: 0 });
      S.rv.hidden.clear();
      setDraw(false);
      $("#review-loading").hidden = false;
    }
    renderFileBar();
    await loadReviewFile(id, { keepView });
  }
```

`loadReviewFile` keeps its fetch and the stale-id guard; after storing `rv.file` it: hides `#review-loading`; compares the page list (count and each width/height) with the rows' pages — when `keepView` and unchanged, keeps rows and calls `refreshVersions()` (Task 11) and `renderZonesAll()` (Task 12); otherwise `teardownRows()` (if rows exist) then `buildRows()` and, for a fresh open, selects the first of `orderedVisible()` (adding it to the seen set) and scrolls to it instantly (Task 12 `revealFinding(id, { instant: true })`), else scrolls to the top; then `focusViewport()` when the open came from the select.
- [ ] **Step 5:** In `refreshState`, replace the "keep the open review file in sync" block:

```js
    const rv = S.rv;
    if (rv.id) {
      const sum = fileById(rv.id);
      if (!sum || !REVIEWABLE.has(sum.status)) {
        const why = sum && sum.status === "error" ? sum.error_message || "Este archivo no se pudo procesar." : "";
        dropReviewFile();
        showReviewStatus(why);
        if (S.screen === 3) renderReview();
      } else if (rv.file && sum.status !== rv.file.status) {
        openFile(rv.id, { keepView: true });
      }
    }
```

`showReviewStatus(text)` sets `#rv-status` text and `hidden = !text`; it is cleared by the next `openFile`.
- [ ] **Step 6:** `renderReview()` picks a file when none is open (`ready` first, then any reviewable — as today) through `openFile`; shows `#review-empty` / `#review` as today. Every other entry point uses `openFile`: Procesar's "Revisar" (`jobRow`), `#b-review-ready`, `renderVerify`'s "Siguiente archivo por revisar" (then `go(3)` if needed). `go(3)` no longer reloads anything when a file is open: it calls `relayout()` (Task 9) to restore the anchor.
- [ ] **Step 7:** Check by hand: `ANONYMIZER_ENGINE=fake uv run python -m anonymizer.app --browser` (background) with two invented PDFs from `tests/test_api.py::make_pdf` saved to the scratch folder; switch files with the select (keyboard arrows: only the last one loads), ‹ ›, and from Procesar. Then `node --check anonymizer/ui/app.js`.
- [ ] **Step 8: Commit**

```bash
git add anonymizer/ui/app.js
git commit -m "Choose the reviewed file from a bar above the document"
```

---

### Task 9: Rows, geometry, layout and horizontal panning

**Files:**
- Modify: `anonymizer/ui/app.js` — remove `currentPage`, `fitScale`, `scaleNow`, `layoutPage`, `loadPageImage`, `setPage`, thumbnails (`thumbCache` … `renderThumbs`), the old `zoomBy`/`zoomFit`/`rotate`; add the functions below.

**Interfaces:**
- Consumes: `ReviewCore.columns`, `fitScales`, `effectiveScale`, `displaySize`, `currentRow`, `anchorOf`, `scrollTopFor`.
- Produces: row objects `{ i, page, el, scale, w, h, before: { cell, box, inner, img, zones, state, key, url, mp }, after: { cell, box, inner, img, state, key, url, mp, version, failed } }`; `buildRows()`, `relayout({ keepAnchor = true } = {})`, `rowMetrics() -> [{top, height}]`, `currentPageIndex()`, `scaleOf(row)`, `zoomBy(factor)`, `zoomFit()`, `rotate()`, `setAfter(on)`, `setPan(fraction)`, `updatePageField()`.

- [ ] **Step 1:** `buildRows()` creates, for each page `p` of `S.rv.file.pages`, a `.prow` (`role="group"`, `aria-label="Página n de N"`) with: a before `.cell.before` containing `.cap` "Antes", a `.pbox` with `.pinner` (an `<img alt="Página n de N, original">` and a `.zones` layer) and a `.state` "Cargando…"; an after `.cell.after` with `.cap` "Después: como quedará", `.pbox`/`.pinner`/`<img alt="Página n de N, como quedará">` and its `.state` "Cargando…"; and a `.plabel` (text from Task 12's `rowLabel(i)`). Zone clicks are delegated on `#rows` (Task 12). Then `relayout({ keepAnchor: false })`.
- [ ] **Step 2:** `relayout()`:

```js
  function relayout({ keepAnchor = true } = {}) {
    const rv = S.rv, vp = $("#viewport");
    if (!rv.file || !rv.rows.length || !vp.clientWidth) return;
    const anchor = keepAnchor ? anchorNow() : null;
    const rowsBox = $("#rows");
    const pad = parseFloat(getComputedStyle(rowsBox).paddingLeft) + parseFloat(getComputedStyle(rowsBox).paddingRight);
    const gap = 24;
    const layout = ReviewCore.columns({ areaWidth: vp.clientWidth - pad, gap, after: rv.after });
    vp.classList.toggle("no-after", !rv.after);
    vp.classList.toggle("stacked", layout.stacked);
    const fits = ReviewCore.fitScales(rv.file.pages, { colWidth: layout.colWidth, rot: rv.rot });
    let widest = 0;
    rv.rows.forEach((row, i) => {
      row.scale = ReviewCore.effectiveScale(fits[i], rv.zoom);
      const { w, h } = ReviewCore.displaySize(row.page, row.scale, rv.rot);
      row.w = w; row.h = h; widest = Math.max(widest, w);
      for (const side of [row.before, row.after]) sizeCell(side, row);
    });
    const col = Math.min(layout.colWidth, widest);
    vp.style.setProperty("--cw", `${Math.max(col, 1)}px`);
    vp.style.setProperty("--gap", `${gap}px`);
    updatePan(widest, layout.colWidth);
    if (anchor) vp.scrollTop = ReviewCore.scrollTopFor(anchor, rowMetrics(), visibleHeight(), headerHeight());
    updateZoomButton();
    updatePageField();
    renderZonesAll();       // Task 12
    scheduleLoads();        // Task 10
  }
```

`sizeCell(side, row)` sets `.pbox` to `row.w × row.h` and `.pinner` to the unrotated size `page.width × scale` by `page.height × scale` with today's rotation transforms (90: `translate(${H}px, 0) rotate(90deg)` with H the unrotated height, etc.) and the image to the unrotated size; it then applies the shared pan: when `widest > colWidth` the `.pbox` is shifted with `transform: translateX(${-pan * (row.w - colWidth)}px)` inside the clipping `.cell`.
`headerHeight()` is `#vp-head`'s `offsetHeight` when visible, else 0; `visibleHeight()` is `vp.clientHeight - headerHeight()`; `rowMetrics()` reads each `.prow`'s `offsetTop` (relative to `#rows`, plus `#rows.offsetTop`) and `offsetHeight`; `anchorNow()` = `ReviewCore.anchorOf(rowMetrics(), vp.scrollTop + headerHeight(), visibleHeight())`.
- [ ] **Step 3:** Panning: `#hpan` is shown when `widest > colWidth`; `#hpan-inner`'s width is `vp.clientWidth * widest / colWidth`; its `scroll` event sets `rv.pan = scrollLeft / (scrollWidth - clientWidth)` and re-applies the shift to every row (no image reload). A `wheel` listener on the viewport with `deltaX` (or `shiftKey` with `deltaY`) moves `#hpan.scrollLeft` and calls `preventDefault()` for that part only.
- [ ] **Step 4:** Zoom/rotation/V: `zoomBy(k)` sets `rv.zoom = clamp(rv.zoom * k, …)` so the effective scale stays in 0.05–8 (use the current page's fit), then `relayout()` (the anchor keeps the center), then `scheduleImageRefresh(120)` (Task 10); `zoomFit()` sets `rv.zoom = 1`; `rotate()` turns `rv.rot` by 90 and toasts as today; `setAfter(on)` sets `rv.after`, `aria-pressed` on `#b-view`, `relayout()`. The zoom button shows `Math.round(scale of the current row * 100) %`.
- [ ] **Step 5:** Resize: a `ResizeObserver` on `#viewport` debounced 150 ms calls `relayout()` (fit and manual zoom alike); `document.fonts.ready` calls `renderZonesAll()`. `go(3)` calls `relayout()` after the screen is shown (`requestAnimationFrame`).
- [ ] **Step 6:** Page field: on scroll (rAF-throttled) `updatePageField()` writes `currentPageIndex() + 1` into `#rv-page` unless it has focus, and `#rv-pages` = `de N`, `#rv-page.max = N`. Enter in `#rv-page` scrolls that row's top to the top of the visible area (instant if long, Task 12 `scrollToY`) and calls `focusViewport()`.
- [ ] **Step 7:** Check by hand with the fake engine: a 3-page invented PDF, window resized across the stacked threshold, zoom in until the pan bar appears and pan, rotate, V. `node --check`.
- [ ] **Step 8: Commit**

```bash
git add anonymizer/ui/app.js
git commit -m "Show every page in one scroll, before and after side by side"
```

---

### Task 10: Lazy loading, the request queue and the memory budget

**Files:**
- Modify: `anonymizer/ui/app.js`

**Interfaces:**
- Consumes: `ReviewCore.requestZoom`, `imageKey`, `acceptResponse`, `planQueue`, `releasePlan`, `isLongScroll`; `apiBlobUrl` (existing).
- Produces: `scheduleLoads()`, `scheduleImageRefresh(ms)`, `clearQueue()`, `stopObserver()`, `releaseRow(row, { all })`, `requestAfter(row, { edited })`, `S.load = { observer, inView: Set, inMargin: Set, queue: [], inFlight: 0, scrolling: false, settleTimer }`.

- [ ] **Step 1:** Observer: in `buildRows()`, create an `IntersectionObserver` with `root: #viewport`, `rootMargin: "${vp.clientHeight}px 0px"`, observing every `.prow`; its callback updates `S.load.inMargin`, and a second observer with `rootMargin: "0px"` updates `S.load.inView`; both call `scheduleLoads()`. `stopObserver()` disconnects both.
- [ ] **Step 2:** `scheduleLoads()`: for each row in `inMargin`, if the before's wanted key (`imageKey({ file, gen, page: i, zoom: requestZoom(row.scale, dpr, row.page), side: "before" })`) differs from its shown key and no job for it is queued, enqueue `{ row: i, side: "before" }`; the same for the after with `version: S.rv.versions[i]` (only when `S.rv.after`). Then `pump()`.
- [ ] **Step 3:** `pump()`: while `inFlight < 3` and not `S.load.scrolling`: `S.load.queue = ReviewCore.planQueue(queue, { inView, inMargin, center: currentPageIndex() })`, shift one job, compute its key and path (`/api/files/${enc(id)}/pages/${i}.png?zoom=${zoom}` plus `&redacted=true&v=${version}` for the after), fetch with `apiBlobUrl`, and on response: if `!ReviewCore.acceptResponse(wantedKeyNow(row, side), key)` revoke and drop silently; else set the `<img>` src, remember `url`, `key` and `mp = width*height/1e6` of the request, hide the cell's `.state`, remove `.updating`, revoke the replaced URL. On error: if the key is still wanted — before: show "No se pudo mostrar esta página" in its `.state` and toast once per file (as today); after: show `.fail` with "No se pudo mostrar el resultado de esta página" and a "Reintentar" button (calls `requestAfter(row, { edited: true })`), remove the image `src` (never show the original). Decrement `inFlight`, `pump()` again.
- [ ] **Step 4:** Scroll settle: the viewport `scroll` listener sets `S.load.scrolling = true` and restarts a 150 ms timer that sets it false, calls `releaseFar()` and `scheduleLoads()`.
- [ ] **Step 5:** `releaseFar()`: rows within 3 viewport heights of the view are `keep`; `ReviewCore.releasePlan(loaded, { keep, center, budgetMp: 150 })` (loaded rows' `mp` summed over both sides); `releaseRow(row)` removes both images' `src`, revokes their URLs, clears keys, shows "Cargando…" again and empties the zone layer (Task 12 redraws when it comes back).
- [ ] **Step 6:** `scheduleImageRefresh(ms)` (zoom): after `ms`, rows outside `inMargin` are released; rows inside get new wanted keys through `scheduleLoads()`.
- [ ] **Step 7:** Check by hand with a 120-page text PDF from Task 13's generator (run it early, it is quick): scrolling fast loads only the rows where it stops; the devtools Network panel (or a log counter) shows no request for rows passed by; memory stays bounded. `node --check`.
- [ ] **Step 8: Commit**

```bash
git add anonymizer/ui/app.js
git commit -m "Load page images lazily with a bounded queue and memory budget"
```

---

### Task 11: Keeping the after up to date

**Files:**
- Modify: `anonymizer/ui/app.js` — `replaceFinding`, `applyAllOptional`, drawing's POST success, `loadReviewFile` (keepView)

**Interfaces:**
- Consumes: `ReviewCore.pageVersions`; Task 10's `requestAfter`.
- Produces: `refreshVersions()` — recomputes `S.rv.versions`; for every loaded row whose version changed: adds `.updating` to its after cell, shows its `.state` "Actualizando…" and calls `requestAfter(row, { edited: true })`; for unloaded rows only the version changes (they load the new one when they come near).

- [ ] **Step 1:** Call `refreshVersions()` from `replaceFinding`, `applyAllOptional` (after its loop), the drawing success path and the keepView reload; drop the old `renderThumbs()` calls there. Update row labels (Task 12 `renderRowLabels()`).
- [ ] **Step 2:** The first image of an after cell shows "Cargando…"; a replacement shows "Actualizando…" over the dimmed previous image (spec 6.5); `.fail` is removed when a retry starts.
- [ ] **Step 3:** Check by hand: remove a finding on page 2 of a 3-page invented PDF and watch only page 2's after update; restore; draw a zone; "Censurar todos los otros enlaces" on a PDF with an invented institutional URL (`make_url_pdf` in `tests/test_api.py`). Stop the server mid-update to see the error cell and "Reintentar". `node --check`.
- [ ] **Step 4: Commit**

```bash
git add anonymizer/ui/app.js
git commit -m "Update the after of a page right after each change to its findings"
```

---

### Task 12: Interaction: selection, zones, drawing, keyboard, leaks

**Files:**
- Modify: `anonymizer/ui/app.js` — `placeLabels`, `renderZones` (→ `renderZonesAll`/`renderRowZones`), `scrollToSelected` (→ `revealFinding`), `select`, `move`, `renderFindings` (leaks box), drawing (`pointToInner`, `initDrawing`), `setDraw`, `setView` (→ `setAfter`), `initKeyboard`, `init`
- Modify: `tests/test_ui_static.py` (now every check passes)

**Interfaces:**
- Consumes: `ReviewCore.zoneRect`, `pointToPage`, `scrollTarget`, `isLongScroll`.
- Produces: `renderRowZones(row)`, `renderZonesAll()`, `revealFinding(id, { instant })`, `scrollToY(y)`, `rowLabel(i)`, `renderRowLabels()`, `focusViewport()`.

- [ ] **Step 1: Zones per row.** `renderRowZones(row)` draws the zones of page `row.i` (filtered by the type chips, `rv.hidden`) into `row.before.zones` only when the before image of that row is loaded or loading (not released), using `ReviewCore.zoneRect(bbox, row.scale, rv.rot, row.page)` and `placeLabels` over that page's items only; a label whose "top" placement would leave the page (`r.y - LABEL_H - 2 < 0`) is placed "below" (or "none" if that is taken). `renderZonesAll()` calls it for every loaded row. Zone clicks: one delegated `click` listener on `#rows` reads `data-id` and calls `select(id)` unless drawing.
- [ ] **Step 2: Selection and scrolling.** `select(id)` sets `rv.sel`, marks seen (unless `auto`), redraws the zones of the previous and the new selected rows, `renderFindings()`, `renderVerify()`, then `revealFinding(id)`:

```js
  function revealFinding(id, { instant = false } = {}) {
    const f = findingById(id), rv = S.rv;
    if (!f || !rv.rows[f.page]) return;
    const row = rv.rows[f.page], m = rowMetrics()[f.page], vp = $("#viewport");
    const b = bbox(f.polygon);
    const z = ReviewCore.zoneRect(b, row.scale, rv.rot, row.page);
    const y = ReviewCore.scrollTarget({ y: z.y, h: z.h }, m, { scrollTop: vp.scrollTop, headerHeight: headerHeight(), height: vp.clientHeight });
    if (y != null) scrollToY(y, { instant });
    panToShow(z, row);               // moves the shared pan when the zone is outside the visible columns
    scrollListToSelected();          // the findings list part of today's scrollToSelected
  }

  function scrollToY(y, { instant = false } = {}) {
    const vp = $("#viewport");
    const jump = instant || reducedMotion() || ReviewCore.isLongScroll(vp.scrollTop, y, vp.clientHeight);
    vp.scrollTo({ top: y, behavior: jump ? "auto" : "smooth" });
  }
```

`move(delta)` is unchanged apart from calling the new `select`.
- [ ] **Step 3: Row labels.** `rowLabel(i)` = `Página ${i + 1} de ${N} · ${plural(k, "zona", "zonas")}` with `k` = active findings of page `i`; `renderRowLabels()` after every change.
- [ ] **Step 4: Leaks.** In `renderFindings()`'s leaks box, a leak with `finding_id` keeps "Ver"; a leak with `page != null` and no finding gets a "Ver página" button that calls `scrollToY(rowMetrics()[l.page].top - headerHeight())` and `focusViewport()`.
- [ ] **Step 5: Drawing.** `pointerdown` on a `.cell.before .pbox` (delegated on `#rows`) while `rv.draw`: remember `row`, capture the pointer on that `.pbox`, create the ghost in that row's `.zones`; `pointermove`: convert with that row's live `getBoundingClientRect()` and `ReviewCore.pointToPage(ox, oy, row.scale, rv.rot, row.page)` (clamped to that page); `pointerup`: POST `{ page: row.i, polygon }` (rounded to 0.01) as today, then `setDraw(false)`, `replaceFinding`, `select`. The row is marked `row.busy = true` during the drag so `releaseFar()` skips it. `setDraw(on)` sets `aria-pressed`, the label `.lbl` ("Dibujando · Esc para salir" / "Dibujar zona"), toggles `.drawing` on `#viewport` and never touches `rv.after`.
- [ ] **Step 6: Keyboard.** In `initKeyboard`: `v` → `setAfter(!rv.after)`; Escape → cancel a drag, else leave draw mode and `focusViewport()`, else `handled = false` (never hides the after); PageUp/PageDown/Home/End/Space/arrows are not handled (native scroll of the focused viewport). `focusViewport()` = `$("#viewport").focus({ preventScroll: true })`.
- [ ] **Step 7:** Remove every remaining reference to the old viewer (`rv.page`, `rv.result`, `#pagebox`, `#pageimg`, `#thumbs`, `lastImageKey`, `setView`). Run `uv run pytest tests/test_ui_static.py tests/test_ui_core.py -v` → all PASS (including `test_old_viewer_is_gone` and the new app.js strings).
- [ ] **Step 8:** Check by hand with the fake engine: J/K across pages, a click on a zone, the findings list, a drag that leaves the cell and crosses into the next row (stops at the edge), Esc in both states, V, the leaks box (force a leak with `tests/test_api.py::test_leak_blocks_export`'s technique or by drawing nothing over a name on an exported file).
- [ ] **Step 9: Commit**

```bash
git add anonymizer/ui/app.js tests/test_ui_static.py
git commit -m "Select, draw and navigate across the pages of the scrolling review"
```

---

### Task 13: Long invented documents and the in-app verification

**Files:**
- Create: `scripts/generate_long_docs.py`
- Modify: `.gitignore` (add `test_data/long/`)

**Interfaces:**
- Produces: `uv run python scripts/generate_long_docs.py [--output test_data/long]` → `texto_120_paginas.pdf`, `escaneo_60_paginas.pdf` (300 dpi JPEG page images with an invented RUT, email and phone on most pages), `tiff_30_paginas.tif` (30 frames, two resolutions mixed). Invented data only (reuse `test_bench.fake_data` for names, RUTs, emails, phones, and `test_bench.canvas.font`).

- [ ] **Step 1:** Write the generator (deterministic seed, `--output`), run it, check the three files open with PyMuPDF/Pillow and have the page counts above.
- [ ] **Step 2:** Run the app with the real engine in the background: `uv run python -m anonymizer.app --browser --no-open --url-file <scratch>/url.json` (see README: the file holds the URL and the session token; deleting it stops the app). Drive it from a script in the scratch folder that starts headless Edge (`"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --remote-debugging-port=9333 --window-size=1366,900 <url>`) and talks to it over the Chrome DevTools Protocol with `uv run --with websocket-client python <script>` (nothing is added to the project): upload files through the API with the token, click and press keys with `Input.dispatch*`, read state with `Runtime.evaluate`, take screenshots with `Page.captureScreenshot`.
- [ ] **Step 3:** Exercise and screenshot (spec 9.5): open each long document and a regular one from `test_data/generated`; scroll; zoom in and pan; rotate; V; narrow window (stacked); J/K across pages and a jump > 30 pages; the page field; draw on page 2 and across a row edge; remove/restore and watch the after update; a page leak's "Ver página". Record: time until both sides of the visible row appear after opening, after the long jump and after one zoom step; the number of rows holding images (count `img[src]` in `#rows` via CDP). Save screenshots under `results/review-scroll/` (git-ignored).
- [ ] **Step 4:** Fix what the verification finds (each fix with its own test where the logic is in `review-core.js` or Python); re-run the affected checks.
- [ ] **Step 5: Commit**

```bash
git add scripts/generate_long_docs.py .gitignore
git commit -m "Generate long invented documents to exercise the scrolling review"
```

---

### Task 14: Documentation and the full suite

**Files:**
- Modify: `PLAN.md` (the review screen description; `GET /api/files/{id}/pages/{n}.png?redacted=true`; spec section 7's figures with the timings of Task 13), `README.md` and `README.es.md` (one line on the before/after review if they describe the review), `docs/superpowers/specs/2026-10-02-review-scroll-before-after-design.md` (status line: "Approved by the user on 2026-10-05 and implemented on branch `review-scroll`").

- [ ] **Step 1:** Write the docs changes.
- [ ] **Step 2:** Run the full suite in the background: `uv run pytest -q > results/review-scroll/pytest.log 2>&1` and `uv run ruff check .` if ruff is configured (`pyproject.toml`); expected: all pass (report any failure with its output; do not skip tests).
- [ ] **Step 3: Commit**

```bash
git add PLAN.md README.md README.es.md docs/superpowers/specs/2026-10-02-review-scroll-before-after-design.md
git commit -m "Document the scrolling before/after review"
```
