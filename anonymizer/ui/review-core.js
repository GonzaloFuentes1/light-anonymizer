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

  const ReviewCore = {
    isActive, pageVersions, columns, fitScales, effectiveScale, displaySize, zoneRect, pointToPage,
    requestZoom, imageKey, acceptResponse, planQueue, releasePlan, currentRow, anchorOf, scrollTopFor,
    scrollTarget, isLongScroll,
  };
  if (typeof module === "object" && module.exports) module.exports = ReviewCore;
  else root.ReviewCore = ReviewCore;
})(typeof globalThis !== "undefined" ? globalThis : this);
