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

  /** What a page's after does when the versions are recomputed. ``shown``: the version of the image
   *  it shows (null without one); ``busy``: an image is on its way or failed; ``was``/``now``: the
   *  page's version before and after. "update": the image shown is out of date (dim it, load again);
   *  "current": it is up to date (also an edit undone before its update arrived); "load": no image,
   *  and the one on its way or failed was for another version; null: nothing to do. */
  function afterChange({ shown, busy, was, now }) {
    if (shown != null) return shown === now ? "current" : "update";
    return busy && was !== now ? "load" : null;
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
  /** The zoom factor times ``k``, kept so the effective scale of a page with this ``fit`` stays in 0.05–8. */
  const nextZoom = (zoom, k, fit) => clamp(zoom * k, 0.05 / fit, 8 / fit);

  function displaySize(page, scale, rot) {
    const w = page.width * scale, h = page.height * scale;
    return turned(rot) ? { w: h, h: w } : { w, h };
  }

  /** The CSS transform of a cell's page layer (unrotated, W × H display pixels, origin at its top-left
   *  corner) for a view rotation. zoneRect and pointToPage follow the same transforms. */
  function innerTransform(rot, W, H) {
    switch (rot) {
      case 90: return `translate(${H}px, 0) rotate(90deg)`;
      case 180: return `translate(${W}px, ${H}px) rotate(180deg)`;
      case 270: return `translate(0, ${W}px) rotate(270deg)`;
      default: return "none";
    }
  }

  /** How much wider than its column a row's page is (0 when it fits, ignoring rounding noise). */
  const overflow = (rowWidth, colWidth) => (rowWidth - colWidth > 0.5 ? rowWidth - colWidth : 0);
  /** The horizontal shift of a row's page in its column. ``pan`` is shared by every row: 0 shows the
   *  left edge of each page, 1 its right edge. A page that fits its column is not shifted. */
  const panShift = (pan, rowWidth, colWidth) => -clamp(pan, 0, 1) * overflow(rowWidth, colWidth) || 0;
  /** The center of the visible part of a row, as a fraction of its page width (0.5 when it fits). */
  function panCenter(pan, rowWidth, colWidth) {
    const extra = overflow(rowWidth, colWidth);
    return extra ? (clamp(pan, 0, 1) * extra + colWidth / 2) / rowWidth : 0.5;
  }
  /** The pan that puts ``fx`` (a fraction of the row's page width) at the center of its column;
   *  ``fallback`` when the row fits, since any pan then shows all of it. */
  function panFor(fx, rowWidth, colWidth, fallback = 0) {
    const extra = overflow(rowWidth, colWidth);
    return extra ? clamp((fx * rowWidth - colWidth / 2) / extra, 0, 1) : fallback;
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

  /** Jobs whose row is not in ``inMargin`` are dropped (``inMargin`` must include the rows in view).
   *  Ranks: befores in view, edited afters, afters in view, then the rest of the margin. Within a
   *  rank: rows in view first, then by distance to ``center``, then before before after, then
   *  input order. */
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
        || (inView.has(b.j.row) - inView.has(a.j.row))
        || Math.abs(a.j.row - center) - Math.abs(b.j.row - center)
        || (a.j.side === b.j.side ? 0 : a.j.side === "before" ? -1 : 1)
        || a.i - b.i)
      .map((x) => x.j);
  }

  /** Rows whose images are released: every loaded row outside ``keep``, then the farthest from
   *  the center until the total is within the budget. Never released: the ``center`` row and the
   *  ``pinned`` rows (rows in view, the row being dragged), even over budget. */
  function releasePlan(loaded, { keep, center, budgetMp = 150, pinned = new Set() }) {
    const safe = (row) => row === center || pinned.has(row);
    const out = loaded.filter((l) => !keep.has(l.row) && !safe(l.row)).map((l) => l.row);
    let rest = loaded.filter((l) => keep.has(l.row) || safe(l.row));
    let total = rest.reduce((s, l) => s + l.mp, 0);
    rest = rest.sort((a, b) => Math.abs(b.row - center) - Math.abs(a.row - center));
    for (const l of rest) {
      if (total <= budgetMp) break;
      if (safe(l.row)) continue;
      out.push(l.row);
      total -= l.mp;
    }
    return out;
  }

  /** Indices of the rows (``{ top, height }``, scroll coordinates) that overlap the view from
   *  ``top`` to ``bottom`` grown by ``reach`` on both sides: the rows whose images are kept. */
  function rowsWithin(rows, top, bottom, reach) {
    const out = new Set();
    rows.forEach((r, i) => {
      if (r.top + r.height >= top - reach && r.top <= bottom + reach) out.add(i);
    });
    return out;
  }

  /** Whether an image job may start: one for a row in view always does; any other only while the
   *  images held and on their way (``totalMp``), with ``newMp`` replacing ``oldMp``, stay within the
   *  budget. Otherwise rows in the margin would load, be released at the next scroll and load again. */
  const admits = ({ totalMp, oldMp, newMp, budgetMp = 150, inView }) =>
    inView || totalMp - oldMp + newMp <= budgetMp;

  /** The row crossing the vertical center of the visible area (or the nearest one). ``viewTop`` and
   *  ``viewHeight`` describe the area BELOW the sticky header: viewTop = scrollTop + headerHeight,
   *  viewHeight = viewport height - headerHeight. Same for anchorOf and scrollTopFor. */
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

  /** The scrollTop that puts the anchor back at the center of the visible area (``viewHeight`` is
   *  the viewport height minus the header, as above; the header is subtracted from the result). */
  function scrollTopFor(anchor, rows, viewHeight, headerHeight) {
    const r = rows[clamp(anchor.row, 0, rows.length - 1)];
    if (!r) return 0;
    return Math.max(0, r.top + anchor.fy * r.height - viewHeight / 2 - headerHeight);
  }

  /** New scrollTop when the zone is closer than ``margin`` to the visible area's edges (below the
   *  sticky header); the zone is first clamped to its own row. null when no scroll is needed.
   *  Unlike the anchor functions, ``view.height`` is the FULL viewport height; the header is
   *  subtracted here. */
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

  /** The pan that brings a zone (``{ x, w }``, display pixels of its page) into the visible part of
   *  the column when it is closer than ``margin`` to its sides or outside them: the zone, clamped to
   *  its page, is centered. null when the page fits its column or nothing would move. */
  function panTarget(zone, rowWidth, colWidth, pan, margin = 48) {
    const extra = overflow(rowWidth, colWidth);
    if (!extra) return null;
    const x0 = clamp(zone.x, 0, rowWidth), x1 = Math.max(x0, clamp(zone.x + zone.w, 0, rowWidth));
    const left = clamp(pan, 0, 1) * extra;
    if (x0 >= left + margin && x1 <= left + colWidth - margin) return null;
    const next = panFor((x0 + x1) / 2 / rowWidth, rowWidth, colWidth, pan);
    return Math.abs(next - pan) < 1e-6 ? null : next;
  }

  /** Label placement of the zones of one page (``items``: ``{ id, r, text }``, display pixels):
   *  "top", "below", or "none" when the label would cover another zone or label (it then shows on
   *  hover). A label never leaves the page: one that would go above its top goes below. The selected
   *  zone is placed first and always shows its label. */
  function placeLabels(items, { sel, pageHeight, labelH = 16, labelWidth = (t) => 10 + t.length * 6.2 }) {
    const placed = [];
    const out = new Map();
    const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    const order = [...items].sort((a, b) => (b.id === sel) - (a.id === sel) || a.r.y - b.r.y || a.r.x - b.r.x);
    for (const it of order) {
      const w = labelWidth(it.text);
      const above = { x: it.r.x - 2, y: it.r.y - labelH - 2, w, h: labelH };
      const below = { x: it.r.x - 2, y: it.r.y + it.r.h + 2, w, h: labelH };
      const inside = { top: above.y >= 0, below: below.y + labelH <= pageHeight };
      const free = (cand) => !placed.some((p) => overlaps(cand, p)) && !items.some((o) => o !== it && overlaps(cand, o.r));
      let where = "none";
      if (it.id === sel) where = inside.top || !inside.below ? "top" : "below";
      else if (inside.top && free(above)) where = "top";
      else if (inside.below && free(below)) where = "below";
      if (where !== "none") placed.push(where === "top" ? above : below);
      out.set(it.id, where);
    }
    return out;
  }

  /** The rectangle dragged between two points (page units) as the polygon sent to the API. */
  function rectPolygon(a, b) {
    const x0 = Math.min(a.x, b.x), y0 = Math.min(a.y, b.y), x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => [round2(x), round2(y)]);
  }

  /** Per page, the zones that will be censored (its row label says "k zonas"). */
  function zoneCounts(findings, pageCount) {
    const out = new Array(pageCount).fill(0);
    for (const f of findings || []) if (isActive(f) && f.page >= 0 && f.page < pageCount) out[f.page] += 1;
    return out;
  }

  const ReviewCore = {
    isActive, pageVersions, afterChange, columns, fitScales, effectiveScale, nextZoom, displaySize, innerTransform,
    overflow, panShift, panCenter, panFor, zoneRect, pointToPage, requestZoom, imageKey, acceptResponse,
    planQueue, releasePlan, rowsWithin, admits, currentRow, anchorOf, scrollTopFor, scrollTarget, isLongScroll,
    panTarget, placeLabels, rectPolygon, zoneCounts,
  };
  if (typeof module === "object" && module.exports) module.exports = ReviewCore;
  else root.ReviewCore = ReviewCore;
})(typeof globalThis !== "undefined" ? globalThis : this);
