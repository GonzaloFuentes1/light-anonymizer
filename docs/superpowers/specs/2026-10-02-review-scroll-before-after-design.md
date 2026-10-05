# Review screen: continuous scroll and a real before/after

Date: 2026-10-02. Status: design approved in conversation, then checked against the code by
three reviews (engine parity, UI behaviour, tests/accessibility/performance); this written spec
awaits the user's review. Mockup: `docs/mockup.html` (Revisar screen).

## 1. Goal

Make the review step simpler to use. Today the reviewer sees one page at a time and switches
pages with the thumbnails on the left; "Ver como quedará" (V) flips that page between the
original and an approximation of the result. The user asked for:

- the pages of the document one below the other, moved through with the scroll wheel, without
  buttons to change pages;
- the original and the anonymized version side by side;
- the anonymized side to be the real result, not an approximation.

Success: a reviewer opens a file in Revisar, scrolls down through every page and, for each one,
sees the original with its colored zones on the left and exactly what will be published on the
right, updated a moment after every change they make.

## 2. Decisions

### 2.1 Taken with the user

| # | Decision |
|---|---|
| 1 | "Changing pages with buttons" meant the pages of the document in Revisar. The four steps (Elegir archivos, Procesar, Revisar, Exportar) stay as they are. |
| 2 | Before and after side by side, page by page, in one vertical scroll. |
| 3 | The after side is rendered by the engine with the same code that exports, applied to an in-memory copy of the page. Nothing is written to disk. |
| 4 | The left column (file list and page thumbnails) goes away. The files move to a selector bar above the document, so before and after get the whole width. |
| 5 | V shows or hides the after column (the original then takes the whole width). In a narrow window each page's after goes under its before. |
| 6 | No ↑K/↓J buttons in the toolbar. The J and K keys keep working and stay in the keyboard hints; a click in the findings list does the same. |
| 7 | "Dibujar zona" stands out in its own color, marker yellow, since it is the one tool that adds a censorship by hand. While drawing, the button turns dark, says "Dibujando · Esc para salir", and the before pages get a dashed edge. |

### 2.2 Added after the code review (for the user to confirm)

| # | Addition | Why |
|---|---|---|
| A | The page indicator is a small number field: "pág. [3] de 12". Typing a number and Enter scrolls to that page. It is not a button per page. | Without thumbnails, a 200-page PDF could only be crossed with the wheel, and a keyboard user could reach only pages that have findings. Pages without findings are where a missed name hides. |
| B | A leak that names a page but no finding gets a "Ver página" button. | Today the reviewer goes to that page through its thumbnail. |
| C | Each row's label says how many zones the page has: "Página 3 de 12 · 4 zonas". | The thumbnails showed that count. |
| D | Side by side switches to stacked (after under before) when each column would be narrower than 420 px, not at a fixed window width. On a 1366 px laptop it stays side by side; a window of about 1100 px or less stacks. | The findings column keeps 300 px, so a fixed 860 px window threshold would leave pages of about 300 px. |
| E | When zoomed past the column width, before and after pan sideways together (one shared horizontal scroll under the columns), so both show the same region of the page. | With one plain scroll container, zooming in would push the same spot of the after out of view, exactly when checking small text. |

## 3. Out of scope

- Steps 1, 2 and 4, the findings column, the dialogs, the audit report and the findings API do
  not change (apart from the strings and the leak button named below).
- Clicking the after side does nothing: it only shows the result. Selecting, removing and drawing
  happen on the before side and in the findings column, as today.
- No DOM virtualization: a row per page is cheap. Images and zone overlays are loaded and
  released lazily (section 6.4).

## 4. Engine

### 4.1 One redaction code path

Today the per-page redaction of a PDF lives inside `pdf.redact`, the function that writes the
exported file, and the rectangles are computed in `RealEngine._export_pdf`. The image fill lives
inside `image.redact`, and the frame conversion inside `image.frames`. A preview that copied that
code could drift from the export, and the difference would be invisible until a document is
published. So the code is split, and both paths call the same functions:

- `pdf.redaction_rects(doc, polygons_by_page) -> dict[int, list[Rect]]`: the bounding box of
  each polygon, moved to unrotated page space with that page's derotation matrix (today's body
  of `_export_pdf`). It reads only the pages it is given, from a document that is already open.
- `pdf.redact_page(doc, n, rects)`: everything `pdf.redact` does to one page today, in the same
  order: unrotate if rotated, add the redaction annotations, `apply_redactions(images=PDF_REDACT_IMAGE_PIXELS)`,
  restore the rotation, delete annotations and widgets, null the `_PAGE_KEYS`.
- `pdf.redact(source, dest, rects_by_page)` keeps its signature: open, `reveal_layers`,
  `redact_page` for every page, then the document-level cleanup and the save, exactly as today.
  `_export_pdf` opens the original, calls `redaction_rects`, closes it and calls `pdf.redact`.
- `image.view_frame(frame) -> Image`: `ImageOps.exif_transpose(frame.copy()).convert("RGB")`.
  `image.frames` uses it for every frame.
- `image.frame(path, n) -> Image`: open, `seek(n)` (an MPO only has frame 0, as `_frames` rules),
  `view_frame`, close. It returns the same pixels `image.frames` yields for frame n, without
  decoding the frames before it.
- `image.redact_frame(arr, polygons)`: fills each polygon grown by `FILL_GROWTH` with black
  (today's loop in `image.redact`). `image.redact` calls it for every frame.

Behaviour of the export does not change. The existing export and leak tests guard that.

### 4.2 Rendering the after side

A new engine method, separate from `render_page` so the original can never be returned by
accident for the after side:

`render_result(file, page, zoom, findings) -> bytes` (PNG). `findings` is required. The engine
keeps those with `f.page == page and f.active`, the same selection `export` makes, so the
review rule lives in one place:

- **PDF.** Open the original, `reveal_layers`, `redaction_rects` for that page,
  `redact_page(doc, page, rects)`, render the pixmap at `zoom`, close without saving.
- **Image.** `image.frame(path, page)`, `redact_frame` with the polygons, then resize like
  today's render. The after is exactly the export's pixels before the output encoding. A JPEG or
  WEBP output (and an MPO, which is exported as JPEG) adds its own lossy compression on top. A
  transparent PNG shows its transparency gone, as the export does. The before keeps using
  `render_page`. Both apply the same EXIF orientation, so their sizes match and the rows align.

`render_page` and `render_result` hold `PDF_LOCK` only while MuPDF works: they copy the pixmap's
samples, width and height inside the lock and encode the PNG with Pillow after releasing it
(lossless, so the parity tests are unaffected). Today the PNG encoding runs inside the lock, and
on a noisy scan it alone takes 0.3–1.2 s.

`FakeEngine` (the development engine, the fallback when running from source without the models,
never used by the packaged app) mirrors 4.1 inside `fake.py`: its PDF export loop body becomes
`_redact_page(doc, n, findings)` and its image fill becomes `_fill(frame, findings)`.
`render_result` calls them on an in-memory copy, without saving (under `PDF_LOCK` for PDFs).

The `Engine` protocol and the contract docstring in `anonymizer/engine/__init__.py` get
`render_result`: what it receives (the findings of the file, possibly more than one page), that
the engine applies `active` and the page itself, and that it writes nothing to disk.

## 5. API

`GET /api/files/{id}/pages/{n}.png?zoom=Z&redacted=true`

- Without `redacted` (or with `redacted=false`) nothing changes: `render_page`, no lock.
- With `redacted=true`, in one `with session.lock:` block, status read first: the zoom check,
  the "not viewable" check, 409 `not_ready` when the status is not `ready`, `confirmed` or
  `exported`, the page-range check, and a snapshot of the findings of page `n` as copies without
  text or history (`replace(f, text=None, history=[])`). Then, outside the lock,
  `render_result(file, n, zoom, snapshot)`. Doing the checks and the snapshot in one block means
  a re-process that starts in between (it empties the findings under the lock) cannot produce an
  after with nothing applied.
- Render failure: 409 `render`, "No se pudo mostrar esta página.", as today. The log line
  carries ids and the page number, never finding text.
- The client may add `v=<version>`; the server ignores it. It only makes requests easy to tell
  apart in logs and tests. Stale responses are handled by the client (6.5).

## 6. User interface

### 6.1 Layout of Revisar

```
┌ 1 Elegir · 2 Procesar · 3 Revisar · 4 Exportar ─────────────────────────────────┐
│ Archivo [ informe.pdf · 3 por revisar ▾ ] (chip)  ‹ ›           pág. [3] de 12  │
│ [− 100 % +]  [Girar R]  [✎ Dibujar zona D] (yellow)  [Mostrar el después V]     │
├─────────────────────────────────────────────────────────────┬───────────────────┤
│          ANTES                        DESPUÉS     (sticky)  │ findings column   │
│  ┌───────────────────────┐   ┌───────────────────────┐      │ (unchanged)       │
│  │ original + zones      │   │ real result           │      │                   │
│  └───────────────────────┘   └───────────────────────┘      │                   │
│              Página 1 de 12 · 4 zonas                       │                   │
│  ┌───────────────────────┐   ┌───────────────────────┐   ↓  │                   │
│  ════════ shared horizontal scrollbar (only when zoomed past the column) ═════  │
├─────────────────────────────────────────────────────────────┴───────────────────┤
│ keyboard hints · verification bar (unchanged)                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

- **Removed:** `.col-files` with `#rfiles` and `#thumbs`, all thumbnail code (`thumbCache`,
  `pumpThumbs`, `requestThumb`, `renderThumbs`, `clearThumbCache`, `setPage`), the single-page
  viewer state (`rv.page`, `#pagebox`, `#pageinner`, `#pageimg`, `lastImageKey`), and the
  `rv.result` overlay (`.result .zone` rules) that "Ver como quedará" used. `rv.result` becomes
  `rv.after`, with its new meaning, so no old code path is reused by mistake.
- **File bar** (`.rv-bar`), above the toolbar:
  - a `<select>` labelled "Archivo" with one option per file: "nombre · estado". The state text
    comes from the same function as the status chips (`statusChip`'s texts, including "N por
    revisar"), except that a file being processed says "Procesando" without a percentage. Files
    that cannot be reviewed are disabled. The closed select cuts a long name with an ellipsis
    and its `title` shows the full name;
  - next to it, the open file's colored status chip;
  - "Archivo anterior" and "Archivo siguiente" (‹ ›), skipping files that cannot be reviewed,
    disabled at the first and last reviewable file (no wrap-around);
  - on the right, the page field "pág. [n] de N" (2.2 A): a number input that follows the
    current page while scrolling; Enter scrolls that page's row to the top of the viewport and
    moves focus to the viewport. It is not a live region (it changes while scrolling and would
    be noisy).
- **Toolbar:** zoom (−, fit, +), Girar vista (R), Dibujar zona (D) and **"Mostrar el después"
  (V)**, which replaces "Ver como quedará": a toggle with `aria-pressed`, on by default. The ↑K
  and ↓J buttons go away (decision 6); their keys stay.
- **"Dibujar zona"** (decision 7) uses new tokens: `--draw: #FFD23F` (fill, both themes),
  `--draw-ink: #1F1A00` (text on it, about 13:1), `--draw-edge` (`#8A6100` light, `#FFD23F` dark).
  It shows a pencil (✎, `aria-hidden`) before its text. Pressed, it is filled with `--draw-ink`,
  its text is `--draw` and reads "Dibujando · Esc para salir", and every before cell gets a
  3 px dashed outline in `--draw-edge`. The state is also in `aria-pressed`, not only in color.
- **Viewport** (`#viewport`): one vertical scroll container, `tabindex="0"`, `role="region"`,
  `aria-label="Páginas del documento"`, with a visible focus ring. Its first child is the header
  row "Antes" | "Después", `position: sticky; top: 0`, opaque, outside the padded area (the
  padding goes on the rows wrapper, otherwise Chromium sticks the header below the padding and
  page content shows above it). With V off the header shows only "Antes"; when stacked (6.2) the
  header is hidden and each cell carries its own caption ("Antes" / "Después: como quedará").
- **Rows:** one per page (`.prow`), a group labelled "Página n de N". It holds the before cell
  (image, zone layer, drawing), the after cell (image and its status text) and the label
  "Página n de N · k zonas" (k = zones that will be censored on that page).
- **Below the toolbar's keyboard hints**, the hint for V reads "V mostrar u ocultar el después".
- **The 860 px rules that exist today stay** (the body scrolls, the review becomes one column,
  the viewport is capped at 75vh).

### 6.2 Geometry

- **Columns.** With the after shown, the area for pages is split into two equal columns with a
  gap. They are stacked instead (after under before) when a column would be narrower than
  420 px. With V off there is one column.
- **Scale.** Zoom is a factor over "fit to the column width" (1 = fit), clamped so the effective
  scale stays between 0.05 and 8 as today, with today's fit caps (1.6 for PDF points, 1 for image
  pixels). For a PDF, fit is one scale for the whole document, the one that fits its widest page
  (with the current rotation): PDF pages are in points, so equal sheets show equal. For an image
  file, each frame is fitted on its own: frames are in pixels, and two A4 scans at different
  resolutions would otherwise show at different sizes. The zoom button shows the effective scale
  of the current page in %; clicking it fits again.
- **Reserved size.** Each row reserves its final size (page size × scale, rotated) before its
  images arrive, so the scroll height is right from the start and nothing jumps. A page narrower
  than its column is centered in it; its after is centered the same way.
- **Horizontal panning** (2.2 E). The viewport never scrolls sideways. Each cell clips its
  content. When the widest page at the current scale is wider than a column, one horizontal
  scrollbar appears at the bottom of the page area (sticky), and the same horizontal offset is
  applied to the before and after cells of every row, so both columns show the same region.
  Shift+wheel and horizontal trackpad gestures over the viewport move that offset too.
- **Rotation** (R) applies to both cells of every row, with today's transform per cell.
- **Current page** is the row that crosses the vertical center of the visible area (below the
  sticky header). It drives the page field. `rv.page` does not exist any more; drawing uses the
  row's own page.
- **Anchor and relayout.** One relayout routine saves the anchor (current page, relative y within
  that row, and the horizontal offset as a fraction of the page width), recomputes the columns,
  scales and reserved sizes, and restores the anchor. It runs on window resize (debounced
  150 ms, in fit and in manual zoom), when the stacked/side-by-side layout changes, on V, on
  rotation, on zoom (the anchor is then the point at the center of the before column) and when
  Revisar is shown again. `document.fonts.ready` redraws the zones of every loaded row.
- **Showing Revisar again** (after visiting another step) keeps the open file, the selection and
  the anchor, and loads nothing again.

### 6.3 Opening and closing a file

- **One entry point.** Every way of opening a file goes through one `openFile(id)`: the select,
  ‹ ›, "Siguiente archivo por revisar", "Revisar" on a row of Procesar, "Revisar los listos",
  and the automatic choice when the open file closes (`ready` first, as today).
- `openFile` updates the select, tears the old file down (below), resets zoom, rotation, type
  chips and draw mode as today, **keeps V**, and once the file is loaded selects its first
  finding (doubtful ones first, as today) and scrolls to it instantly, or to the top when there
  are none.
- **The select** switches files on `change` after a 300 ms pause, so browsing it with the arrow
  keys (Chromium fires `change` on each arrow press of a closed select) only loads the file the
  reviewer stops on. When that file has loaded, focus moves to the viewport, so J/K and the other
  shortcuts work at once (the key handler ignores keys while a SELECT or INPUT has focus, and the
  select's type-ahead would take the letters). ‹ › keep focus on themselves; if one becomes
  disabled, focus moves to the viewport.
- **The select is built once.** On each state poll (every 700 ms while files are processed) only
  an option whose text or disabled state changed is updated, in place; the element and its
  options are never replaced, so an open dropdown stays open.
- **Teardown.** Switching or closing the file, or reloading it with a different page list,
  empties the viewport (showing "Cargando el archivo…"), disconnects the observer, clears the
  request queue, revokes every blob URL and increases a load generation. Every image request
  carries the file id and the generation; a response from an older generation is revoked and
  dropped silently, with no toast and no error cell. A drag cannot start on an empty viewport.
- **Reload keeping the view** (`keepView`, today's reload when the open file goes between
  reviewable statuses: the first edit of a confirmed or exported file, an export): when the page
  list (count and sizes) is unchanged, the rows, the anchor, the scale, rotation, chips, draw mode
  and the loaded before images stay; the page versions are recomputed (6.5) and the zones of
  loaded rows redrawn. If the page list changed, the rows are rebuilt and the anchor page is
  restored, clamped.
- **Processing the open file again.** When a poll sees the open file in a status that cannot be
  reviewed (`queued`, `processing`, `error`, `cancelled`), the review drops it on any screen, not
  only on Revisar: teardown, `rv.file = null`, its seen set cleared. It is loaded fresh, without
  keepView, the next time it is opened or becomes reviewable. Today re-processing a ready file
  ends `ready` again and the old findings stay on screen; with the after column the two sides
  would disagree and edits would go to ids that no longer exist.
- **A file that needs help** (`error`) shows its `error_message` in a status line under the file
  bar when it is the reason the review closed (today a toast showed it on a click on the file).

### 6.4 Lazy loading, the request queue and memory

- **Observer.** An `IntersectionObserver` on the rows (root: the viewport, margin: one viewport
  height) marks which rows are near the view.
- **Queue.** Jobs are images: (row, side). When a request slot frees up the queue is re-sorted
  and any job whose row is no longer within the margin is dropped. Order: before images of rows
  in view, then their after images, then rows in the margin, nearest first. An after that changed
  because of an edit goes ahead of everything except the before images in view. At most three
  requests in flight. Requests already sent are never aborted: the server cannot stop a render
  once started.
- **Programmatic scrolls.** A scroll caused by the app (J/K, a click in the findings list,
  opening a file, the page field) that is longer than two viewport heights is instant; shorter
  ones are smooth unless `prefers-reduced-motion`. While it runs, and until no scroll event has
  arrived for 150 ms, no new request starts, so rows passed on the way are not requested.
- **Zoom.** After a zoom change (and the 120 ms pause of today), only rows inside the margin
  request new images. Other loaded rows drop their images and load them again when they come
  near.
- **Memory budget.** Images are kept only for rows within three viewport heights of the view,
  and at most about 150 megapixels in total across both sides; the farthest rows are released
  first. A released row removes its `src`, revokes its blob URL and drops its zone overlays, and
  keeps its reserved size.
- **Image size.** Each row's request zoom = effective scale × `devicePixelRatio`, capped so the
  longest side is at most 9000 px, rounded to 0.01, the same for both cells of the row (as
  today's `loadPageImage`).
- **Zones** are drawn only in the before cells of loaded rows. Scrolling to a finding uses its
  geometry (row offset plus bounding box × scale, rotation-aware), so it works for a row whose
  zones are not drawn yet.

### 6.5 Keeping the after up to date

- Each page has a version: a short hash of the id, polygon and status of its active findings.
- After any change to the findings (remove, restore, add a zone, censor or uncensor an optional
  URL, "Censurar todos los otros enlaces", a keepView reload) the versions are recomputed. Each
  loaded page whose version changed requests its after again.
- Each cell keeps the key of the image it wants: (file, generation, page, zoom, version for the
  after). A response for any other key is revoked and dropped. When the wanted image is shown,
  the blob URL of the one it replaced is revoked.
- Until its first image arrives, each cell shows "Cargando…" over its reserved space. The after
  cell never shows a blank page without that text, which could look like a clean page.
- While a new after is on its way, the current one is dimmed and shows "Actualizando…"; when it
  arrives it replaces the old one.
- If the after fails, the cell shows no image and the text "No se pudo mostrar el resultado de
  esta página" with a "Reintentar" button. It never shows the original in its place, which could
  be mistaken for a redacted page.
- The type chips of the findings column only hide zone overlays on the before side. The after
  side always shows the real result with every active finding.

### 6.6 Interaction

- **Selecting** a finding (list, J/K, click on a zone) scrolls the viewport only when its zone is
  less than 48 px from the edge of the visible area (which excludes the sticky header), and then
  centers it in that area; the horizontal offset moves the same way when the zone is outside the
  columns' visible region. Both sides share the row and the offset, so they stay aligned. The
  findings list keeps scrolling the selected item into view, as today.
- **Off-page findings.** Text outside the page box is extracted and redacted, so a finding can lie
  partly or wholly outside its page. Zone layers are clipped to their page box, and scrolling to
  such a finding stops at the nearest edge of its own row, never in a neighbouring row.
- **Labels** are placed per row (`placeLabels` over that page's zones only) and must stay inside
  the page box: a label that would leave the page at the top goes below its zone (the existing
  "label-below"). A selection change redraws the zones of the old and the new selected rows.
- **Seen findings.** Scrolling does not mark findings as seen; only selecting them does (list,
  J/K, click) and being the first finding of a newly opened file, as today, so "sin abrir" and
  the warning before confirming keep their meaning.
- **Drawing** (D). The page is fixed at `pointerdown` on a before cell. That cell captures the
  pointer; every move is converted with that row's live rectangle, its own page size and the
  rotation (and the horizontal offset), then clamped to that page. Leaving the cell or crossing
  into another row stops at the page edge, and scrolling during the drag works. The ghost lives
  in that row's zone layer, and the row is not released or redrawn while the drag lasts. The
  crosshair cursor shows only on before cells.
- **V, D and Esc are independent.** V toggles the after column and the layout is recomputed (6.2);
  the choice lasts while the app is open, across files. D does not change V. Esc cancels a drag
  in progress, otherwise leaves draw mode (focus then returns to the viewport), and does nothing
  else: it never hides the after column.
- **Keyboard.** J, K, Supr, D, +, −, R, V and Esc work as today. PageUp, PageDown, Home, End,
  Space and the arrow keys are left to the browser, so they scroll the focused viewport.
- **Leaks.** A leak with a finding keeps its "Ver" button. A leak that names a page but no finding
  gets "Ver página", which scrolls that row to the top of the viewport (2.2 B).

### 6.7 Accessibility

- The viewport is focusable and labelled (6.1); rows are groups labelled "Página n de N".
- The before image's alt is "Página n de N, original"; the after's is "Página n de N, como
  quedará".
- The after's states ("Cargando…", "Actualizando…", the error) are text, not only color or
  opacity.
- The file select and the page field have visible labels.
- `prefers-reduced-motion` makes every programmatic scroll instant.

## 7. Performance

Measured on the development machine (i7-13620H) with invented documents and PyMuPDF 1.28.2:

| Page | Before | After |
|---|---|---|
| Text PDF page | about 0.04 s | about 0.04 s |
| Scanned A4 page, 300 dpi, zoom 1.0 | 0.09 s | 0.45 s (the pixel redaction alone 0.28 s) |
| Scanned A4 page, 300 dpi, zoom 1.9 | 0.36 s | 0.69 s |

- Every PDF render holds the one process-wide `PDF_LOCK`, shared with analysis and export. PDF
  requests therefore run one at a time even with three in flight, and a render waits between
  the pages of an analysis running on another file. Moving the PNG encoding out of the lock
  (4.2) shortens what it holds.
- So scrolling a scanned PDF costs about 0.6–1 s of serialized work per row, which is why the
  queue drops rows that are passed by (6.4).
- Target, on the development machine with nothing else processing: on a 60-page scanned PDF,
  after scrolling stops or after a J/K jump of more than 30 pages, the visible row shows both
  sides within about 2 s. A text PDF shows both sides as fast as the wheel turns.

## 8. Error handling

| Case | Behaviour |
|---|---|
| Before image fails | That cell shows "No se pudo mostrar esta página", and one toast per file, as today. |
| After image fails | That after cell shows the error text and "Reintentar"; no image (6.5). |
| A response from an older file or load generation | Revoked and dropped silently (6.3). |
| The open file is processed again, removed, or fails | The review drops it on any screen (6.3); the select shows its new state. |
| `redacted=true` before the file is ready | 409 `not_ready` (the UI never asks for it then; a late response is dropped as above). |

## 9. Testing

### 9.1 Engine parity (`tests/test_engine_real.py`)

Fictitious documents built inside the tests, findings created by hand (as the existing tests do
for scanned pages with `page.insert_image`), so no OCR model is needed. For each case, the after
PNG must equal the render of the same page of the exported file at the same zoom, compared on the
decoded pixels (`numpy.array_equal`), not the PNG bytes:

- a text PDF page;
- a page without findings but with an annotation;
- a scanned page (image only), at zoom 1 and 1.9;
- a page with `/Rotate 90` whose CropBox does not start at (0, 0) (the existing `_boxed_pdf`
  helper);
- a page with an annotation containing an invented name (the export deletes it, so the after
  must not show it);
- a Form XObject letterhead shown on two pages with a finding only on the second, comparing both
  pages;
- one image XObject drawn on two pages with pixels redacted on one, comparing both pages.

Every PDF case requires exact equality; if a future PyMuPDF breaks it, the test fails and this
spec is revisited. Images:

- a PNG with EXIF orientation 6, a palette PNG and page 2 of a 3-page TIFF: exact equality;
- a JPEG and a WEBP: mean absolute difference under 2 levels (their output is lossy);
- `image.frame(p, n)` equals `list(image.frames(p))[n]` pixel for pixel.

The development engine gets the same parity check against its own export.

### 9.2 Review state, privacy and the API

- **Review state** (engine): removed findings, suggested URLs and findings of other pages passed
  to `render_result` are not applied. Through the route as well, with `RealEngine` and
  `LiveClient` (as `test_engine_real.py` does): one finding removed, one URL suggested,
  `?redacted=true` compared pixel for pixel with the render of the exported file.
- **Nothing on disk:** `tempfile.tempdir`, `TMP` and `TEMP` point to an empty folder under
  `tmp_path`; after rendering afters of a PDF and an image it is still empty, the session folder's
  listing and modification times are unchanged, and the working copy's SHA-256 is unchanged.
- **API** (`tests/test_api.py`, development engine): `redacted=true` returns a PNG; 409
  `not_ready` before the file is ready; 404 for a page out of range; 422 for a bad zoom; the image
  changes after a `PATCH` that removes a finding; the plain route is unchanged.

### 9.3 Client logic

The new logic that is easiest to get wrong is kept as small pure functions in a separate classic
script, `anonymizer/ui/review-core.js`, loaded before `app.js` (it exposes `globalThis.ReviewCore`,
and `module.exports` when loaded by Node): the page versions, the queue order and pruning, the
key check for responses, the memory-budget release plan and the anchor math. They are tested with
`node --test` from a pytest wrapper that is skipped when Node is missing, like
`test_app_js_syntax`.

### 9.4 Static checks (`tests/test_ui_static.py`)

The new strings ("Antes", "Después", "Mostrar el después", "Dibujando · Esc para salir",
"Cargando…", "Actualizando…", "No se pudo mostrar el resultado de esta página", "Reintentar",
"Ver página", "Páginas del documento"); the `--draw` tokens in `app.css`;
`.result .zone.suggested` leaves the list of required CSS rules; `node --check` on both scripts;
the local-assets test covers `review-core.js`.

### 9.5 In the app

The fictitious test set has at most 3 pages per document, too short to exercise the queue and the
memory budget. Long invented documents are generated into `test_data/long/` (not versioned): a
120-page text PDF, a 60-page 300 dpi scanned PDF with findings on most pages, and a 30-page TIFF.
With them and with the regular set, in the app:

- scroll through each; zoom in and pan sideways; rotate; V on and off; a narrow window (stacked);
- J/K across pages and a jump of more than 30 pages; the page field;
- draw a zone on page 2 and across a row edge; remove and restore a finding and watch its after
  update; a page leak's "Ver página";
- record the time until both sides of the visible row appear after opening, after the long J/K
  jump and after one zoom step, and check that no more than about 20 rows hold images.

Screenshots of each, and the timings, are reported.

## 10. Documentation

- `docs/mockup.html`: the Revisar screen redrawn with this layout, with the after side simulated
  (the mockup has no engine). Its comments moved to English, and its design notes moved under
  the app window so the window gets the whole width, as the app does. Done with this spec.
- `PLAN.md`: the description of the review screen, the new route parameter and section 7's
  figures.
- `anonymizer/engine/__init__.py`: the engine contract (4.2).
