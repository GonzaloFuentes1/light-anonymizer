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

test("the page layer's rotation transform agrees with zoneRect", () => {
  const page = { width: 200, height: 100 }, s = 2, W = 400, H = 200;
  const b = { x: 10, y: 20, w: 30, h: 40 };
  assert.equal(C.innerTransform(0, W, H), "none");
  for (const rot of [0, 90, 180, 270]) {
    // CSS applies "translate(tx, ty) rotate(deg)" right to left, around the layer's top-left corner.
    const m = /^translate\(([\d.]+)(?:px)?, ([\d.]+)(?:px)?\) rotate\((\d+)deg\)$/.exec(C.innerTransform(rot, W, H));
    const [tx, ty, deg] = m ? m.slice(1).map(Number) : [0, 0, 0];
    const cos = Math.round(Math.cos((deg * Math.PI) / 180)), sin = Math.round(Math.sin((deg * Math.PI) / 180));
    const map = ([x, y]) => [cos * x - sin * y + tx, sin * x + cos * y + ty];
    const [p, q] = [[b.x * s, b.y * s], [(b.x + b.w) * s, (b.y + b.h) * s]].map(map);
    const shown = { x: Math.min(p[0], q[0]), y: Math.min(p[1], q[1]), w: Math.abs(p[0] - q[0]), h: Math.abs(p[1] - q[1]) };
    assert.deepEqual(shown, C.zoneRect(b, s, rot, page), `rot ${rot}`);
  }
});

test("pan: one shared fraction, a shift per row, the center kept across a zoom", () => {
  assert.equal(C.overflow(1000, 400), 600);
  assert.equal(C.overflow(400.2, 400), 0); // rounding noise at fit is not an overflow
  assert.equal(C.panShift(0, 1000, 400), 0);
  assert.equal(C.panShift(1, 1000, 400), -600);
  assert.equal(C.panShift(0.5, 300, 400), 0); // fits its column: not shifted
  assert.equal(C.panCenter(0.7, 300, 400), 0.5);
  assert.equal(C.panCenter(0, 1000, 400), 0.2);
  const fx = C.panCenter(0.25, 800, 400); // (0.25 * 400 + 200) / 800
  assert.equal(fx, 0.375);
  const pan = C.panFor(fx, 1600, 400); // the same point after zooming 2x
  assert.ok(Math.abs(pan * 1200 + 200 - fx * 1600) < 1e-9);
  assert.equal(C.panFor(0.01, 1600, 400), 0); // clamped to the edges
  assert.equal(C.panFor(0.99, 1600, 400), 1);
  assert.equal(C.panFor(0.5, 300, 400, 0.7), 0.7); // a row that fits keeps the current pan
});

test("zoom keeps the effective scale within 0.05-8", () => {
  assert.ok(Math.abs(C.nextZoom(1, 1.2, 0.5) - 1.2) < 1e-12);
  assert.equal(C.nextZoom(15, 1.2, 0.5), 16);
  assert.equal(C.nextZoom(0.11, 1 / 1.2, 0.5), 0.1);
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
  assert.deepEqual(out, [0, 1, 2, 10]);
  const kept = loaded.filter((l) => !out.includes(l.row)).reduce((s, l) => s + l.mp, 0);
  assert.ok(kept <= 150);
  assert.ok(!out.includes(6)); // never the row at the center
});

test("queue: edited afters and befores are ordered by view, then distance", () => {
  const jobs = [
    { row: 9, side: "after", edited: true }, { row: 4, side: "after", edited: true },
    { row: 6, side: "after", edited: true }, { row: 5, side: "after", edited: true },
    { row: 8, side: "before" }, { row: 6, side: "before" }, { row: 5, side: "before" },
  ];
  const out = C.planQueue(jobs, { inView: new Set([5, 6]), inMargin: new Set([4, 5, 6, 7, 8, 9]), center: 5 });
  assert.deepEqual(out.map((j) => `${j.row}${j.side[0]}`), ["5b", "6b", "5a", "6a", "4a", "9a", "8b"]);
});

test("release: pinned rows and an over-budget center survive", () => {
  assert.deepEqual(C.releasePlan([{ row: 3, mp: 162 }, { row: 4, mp: 10 }], { keep: new Set([3, 4]), center: 3 }), [4]);
  const loaded = [0, 1, 2, 3].map((row) => ({ row, mp: 100 }));
  const out = C.releasePlan(loaded, { keep: new Set([0, 1, 2, 3]), center: 1, pinned: new Set([0, 2]) });
  assert.deepEqual(out, [3]);
  const dropped = C.releasePlan(loaded, { keep: new Set([1]), center: 1, pinned: new Set([0]) });
  assert.deepEqual(dropped.sort(), [2, 3]);
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
