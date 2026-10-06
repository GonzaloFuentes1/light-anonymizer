/* Anonymizer desktop UI (vanilla JS, no build step, no external requests).
 *
 * Talks only to the local API under /api (see the HTTP API contract). Every request carries the
 * session token read from <meta name="session-token">. Finding coordinates are in view space
 * (points for PDFs, pixels for images); the viewer scales them by the current display scale.
 * All text shown to the user is Spanish (Chile).
 */
"use strict";

(() => {
  // ---------------------------------------------------------------------------------------------
  // Constants and user-facing strings
  // ---------------------------------------------------------------------------------------------
  const TYPE_LABELS = {
    rut: "RUT",
    email: "Correo",
    phone: "Teléfono",
    url: "Enlace",
    name: "Nombre",
    address: "Dirección",
    face: "Rostro",
    signature: "Firma",
    qr: "Código QR",
    text: "Texto en imagen",
    manual: "Agregada",
  };
  const TYPE_ORDER = Object.keys(TYPE_LABELS);
  // Shown in the list when a finding has no text (faces, signatures, drawn zones).
  const TYPE_PLACEHOLDERS = {
    face: "Rostro",
    signature: "Firma",
    qr: "Código QR",
    text: "Texto en imagen",
    manual: "Zona dibujada",
  };
  const DETECTOR_LABELS = {
    regex: "patrones",
    name_list: "lista de nombres",
    context: "contexto",
    ocr: "OCR",
    faces: "rostros",
    qr: "código QR",
    reviewer: "agregada por ti",
  };
  // Stages of an analysis (summary "timings"), shown when a file is done.
  const STAGE_LABELS = { text: "texto", ocr: "texto en imágenes", faces: "rostros", qr: "QR" };
  const SUPPORTED_EXT = ["pdf", "jpg", "jpeg", "png", "webp", "tif", "tiff"];
  const REVIEWABLE = new Set(["ready", "confirmed", "exported"]);
  const MSG = {
    network: "No se pudo conectar con el motor de la aplicación. Cierra el programa y vuelve a abrirlo.",
    generic: "Ocurrió un problema. Inténtalo de nuevo.",
    noToken: "No se pudo iniciar la sesión de trabajo. Cierra el programa y vuelve a abrirlo.",
  };
  const POLL_MS = 700;
  const THEME_KEY = "anonymizer.theme";
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const tokenMeta = document.querySelector('meta[name="session-token"]');
  const TOKEN = tokenMeta && tokenMeta.content !== "__SESSION_TOKEN__" ? tokenMeta.content : "";

  // ---------------------------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------------------------
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  /** Build an element. Boolean false / null props are skipped, so pass ARIA values as strings. */
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "style") Object.assign(el.style, v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, String(v));
    }
    for (const c of kids.flat(Infinity)) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : String(c));
    }
    return el;
  }

  /** Re-render a container keeping keyboard focus on the element with the same data-fk. */
  function keepFocus(container, render) {
    const active = document.activeElement;
    const key = active && container.contains(active) ? active.dataset.fk : null;
    render();
    if (key) {
      const again = container.querySelector(`[data-fk="${CSS.escape(key)}"]`);
      if (again) again.focus({ preventScroll: true });
    }
  }

  /** File name with line-break opportunities after "_", "-" and "." (long names wrap cleanly). */
  function nameNode(name) {
    const frag = document.createDocumentFragment();
    for (const part of String(name).split(/(?<=[_.-])/)) {
      frag.append(part);
      frag.append(document.createElement("wbr"));
    }
    return frag;
  }

  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  /** Spanish list: "a", "a y b", "a, b y c". */
  const joinEs = (items) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} y ${items[items.length - 1]}`);

  /** Whole seconds as "45 s", "1 min 20 s", "2 h 5 min". */
  function fmtDuration(total) {
    if (total < 60) return `${total} s`;
    const hours = Math.floor(total / 3600), min = Math.floor((total % 3600) / 60), sec = total % 60;
    if (hours) return min ? `${hours} h ${min} min` : `${hours} h`;
    return sec ? `${min} min ${sec} s` : `${min} min`;
  }
  /** An estimate, rounded the way people say it: "menos de 1 s", "≈ 40 s", "≈ 1 min 20 s". */
  function fmtEstimate(seconds) {
    if (!(seconds >= 1)) return "menos de 1 s";
    const step = seconds < 10 ? 1 : seconds < 60 ? 5 : seconds < 3600 ? 10 : 60;
    return `≈ ${fmtDuration(Math.max(1, Math.round(seconds / step) * step))}`;
  }
  /** A measured time: "0,4 s" under a second, then "12 s", "1 min 20 s". */
  function fmtMeasured(seconds) {
    if (seconds < 0.05) return "menos de 0,1 s";
    if (seconds < 1) return `${seconds.toLocaleString("es-CL", { maximumFractionDigits: 1 })} s`;
    return fmtDuration(Math.round(seconds));
  }
  const fmtNum = (n) => Number(n).toLocaleString("es-CL");
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const ext = (name) => (String(name).split(".").pop() || "").toLowerCase();
  const isSupported = (name) => SUPPORTED_EXT.includes(ext(name));

  function fmtSize(bytes) {
    if (bytes == null) return null;
    if (bytes < 1024 * 1024) return `${fmtNum(Math.max(1, Math.round(bytes / 1024)))} KB`;
    return `${(bytes / (1024 * 1024)).toLocaleString("es-CL", { maximumFractionDigits: 1 })} MB`;
  }

  function fileBadge(f) {
    const e = ext(f.name);
    if (e === "jpeg") return "JPG";
    if (e === "tif") return "TIFF";
    return (e || "?").toUpperCase().slice(0, 4);
  }

  function isImage(f) {
    if (f.kind) return f.kind === "image";
    return ["jpg", "jpeg", "png", "webp", "tif", "tiff"].includes(ext(f.name));
  }

  function bbox(polygon) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of polygon || []) {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    if (!Number.isFinite(x0)) return { x: 0, y: 0, w: 0, h: 0 };
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  // ---------------------------------------------------------------------------------------------
  // Toast and fatal banner
  // ---------------------------------------------------------------------------------------------
  let toastTimer = null;
  function toast(msg, { bad = false } = {}) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.toggle("bad", bad);
    t.classList.add("on");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("on"), bad ? 6000 : 3200);
  }
  function showError(err) {
    toast(err && err.message ? err.message : MSG.generic, { bad: true });
  }
  function setFatal(msg) {
    const el = $("#fatal");
    el.textContent = msg || "";
    el.hidden = !msg;
  }

  // ---------------------------------------------------------------------------------------------
  // API
  // ---------------------------------------------------------------------------------------------
  class ApiError extends Error {
    constructor(code, message, status) {
      super(message);
      this.code = code;
      this.status = status;
    }
  }

  async function request(path, { method = "GET", json, form } = {}) {
    const headers = { "X-Session-Token": TOKEN };
    let body;
    if (json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(json);
    } else if (form) {
      body = form;
    }
    let res;
    try {
      res = await fetch(path, { method, headers, body, cache: "no-store", credentials: "same-origin" });
    } catch {
      throw new ApiError("network", MSG.network, 0);
    }
    if (!res.ok) {
      let data = null;
      try { data = await res.json(); } catch { /* not JSON */ }
      throw new ApiError((data && data.error) || `http_${res.status}`, (data && data.message) || MSG.generic, res.status);
    }
    return res;
  }
  async function api(path, opts) {
    const res = await request(path, opts);
    if (res.status === 204) return null;
    return res.json();
  }
  async function apiBlobUrl(path) {
    const res = await request(path);
    return URL.createObjectURL(await res.blob());
  }
  const enc = encodeURIComponent;

  // pywebview bridge (desktop shell). Optional: the UI also works in a plain browser.
  const bridge = () => (window.pywebview && window.pywebview.api) || null;
  const hasBridge = (fn) => !!(bridge() && typeof bridge()[fn] === "function");

  // ---------------------------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------------------------
  const S = {
    screen: 1,
    files: [], // summaries from GET /api/state
    namesCount: 0,
    engine: "",
    groups: [], // detection groups from GET /api/options (texts in Spanish, current state)
    estimate: null, // GET /api/estimate for the files that would be processed now
    estimateKey: null, // the ids that estimate is for
    loaded: false,
    requested: new Map(), // id -> time it was sent to /api/process by this window
    dirty: new Set(), // processed files whose options changed (need processing again)
    sizes: new Map(), // id -> bytes, known for files added from this window
    pollTimer: null,
    rv: {
      id: null,
      file: null, // full AnalyzedFile
      gen: 0, // load generation: increased by every teardown, carried by every image request
      sel: null,
      hidden: new Set(), // finding types hidden by the filter chips
      zoom: 1, // factor over "fit to the column width"
      rot: 0,
      draw: false,
      after: true, // the after column is shown (V); kept across files
      seen: new Map(), // file id -> Set of finding ids the reviewer opened
      loadingId: null,
      rows: [], // one per page of the open file (see buildRows)
      versions: [], // per page: hash of its active findings (ReviewCore.pageVersions)
      pan: 0, // shared horizontal offset, as a fraction of the overflow
    },
    exp: { dest: "", results: new Map(), busy: false, last: null },
  };
  const fileById = (id) => S.files.find((f) => f.id === id) || null;
  // Applied on export: not kept visible by the reviewer and not a suggestion left unapplied (D12).
  const isApplied = (f) => f.status !== "removed" && f.status !== "suggested";
  const REQUEST_GRACE_MS = 4000; // the worker may take a moment to mark a file as queued
  const isActive = (f) => f.status === "processing" || (f.status === "queued" && S.requested.has(f.id));
  const justRequested = () => [...S.requested.values()].some((t) => Date.now() - t < REQUEST_GRACE_MS);
  const isPending = (f) =>
    !S.requested.has(f.id) && ((f.status === "queued") || f.status === "cancelled" || S.dirty.has(f.id));
  const seenSet = (id) => {
    if (!S.rv.seen.has(id)) S.rv.seen.set(id, new Set());
    return S.rv.seen.get(id);
  };

  // ---------------------------------------------------------------------------------------------
  // Theme (Claro / Sistema / Oscuro), remembered per computer
  // ---------------------------------------------------------------------------------------------
  function applyTheme(choice, save) {
    const root = document.documentElement;
    if (choice === "light" || choice === "dark") root.setAttribute("data-theme", choice);
    else root.removeAttribute("data-theme");
    $$("[data-theme-choice]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeChoice === choice)));
    if (save) {
      try { localStorage.setItem(THEME_KEY, choice); } catch { /* storage unavailable */ }
    }
  }
  function initTheme() {
    let saved = "system";
    try { saved = localStorage.getItem(THEME_KEY) || "system"; } catch { /* storage unavailable */ }
    applyTheme(saved, false);
    $$("[data-theme-choice]").forEach((b) => b.addEventListener("click", () => applyTheme(b.dataset.themeChoice, true)));
  }

  // ---------------------------------------------------------------------------------------------
  // Navigation between the four steps
  // ---------------------------------------------------------------------------------------------
  function go(n, { focus = true } = {}) {
    S.screen = n;
    $$(".screen").forEach((s) => s.classList.toggle("on", s.dataset.screen === String(n)));
    if (n !== 3) { setDraw(false); }
    renderSteps();
    if (n === 3) {
      renderReview(); // loads nothing when a file is open: the rows and the anchor stay
      relayout();
    }
    if (n === 4) enterExport();
    schedulePoll();
    if (focus) {
      const title = $(`[data-screen="${n}"] [tabindex="-1"]`);
      if (title) title.focus({ preventScroll: true });
    }
  }

  function renderSteps() {
    const files = S.files;
    const any = files.length > 0;
    const reviewable = files.filter((f) => REVIEWABLE.has(f.status));
    const done = {
      1: any,
      2: any && !files.some(isActive) && !files.some(isPending) && reviewable.length > 0,
      3: reviewable.length > 0 && reviewable.every((f) => f.status !== "ready"),
      4: files.some((f) => f.status === "exported"),
    };
    $$(".step").forEach((b) => {
      const k = Number(b.dataset.go);
      if (k === S.screen) b.setAttribute("aria-current", "step");
      else b.removeAttribute("aria-current");
      b.classList.toggle("done", !!done[k]);
      let sr = $(".sr-only", b);
      if (!sr) { sr = h("span", { class: "sr-only" }); b.append(sr); }
      sr.textContent = done[k] ? " (completado)" : "";
    });
  }

  // ---------------------------------------------------------------------------------------------
  // State polling
  // ---------------------------------------------------------------------------------------------
  async function refreshState() {
    clearTimeout(S.pollTimer);
    let data;
    try {
      data = await api("/api/state");
    } catch (err) {
      if (!S.loaded) setFatal(err.message || MSG.network);
      S.pollTimer = setTimeout(refreshState, 2500);
      return;
    }
    S.loaded = true;
    setFatal("");
    S.files = Array.isArray(data.files) ? data.files : [];
    S.namesCount = data.names_count || 0;
    S.engine = data.engine || "";
    for (const f of S.files) {
      // A file that finished, failed or was cancelled no longer needs the "requested" mark.
      const t = S.requested.get(f.id);
      if (!["queued", "processing"].includes(f.status) && t != null && Date.now() - t >= REQUEST_GRACE_MS) {
        S.requested.delete(f.id);
      }
      // Seen queued or running: from now on the mark goes away as soon as it finishes.
      if (f.status === "processing" || (f.status === "queued" && t != null)) S.requested.set(f.id, 0);
    }
    const ids = new Set(S.files.map((f) => f.id));
    for (const id of [...S.requested]) if (!ids.has(id)) S.requested.delete(id);
    for (const id of [...S.dirty]) if (!ids.has(id)) S.dirty.delete(id);

    $("#engine-chip").hidden = S.engine !== "fake";
    renderAll();

    // Keep the open review file in sync, on any screen. Processed again, removed or failed: the
    // review forgets it. Between reviewable statuses (first edit after confirming, an export):
    // reloaded keeping the view.
    const rv = S.rv;
    if (rv.id) {
      const sum = fileById(rv.id);
      if (!sum || !REVIEWABLE.has(sum.status)) {
        const why = sum && sum.status === "error"
          ? `${sum.name} necesita tu ayuda. ${sum.error_message || "Este archivo no se pudo procesar."}`
          : "";
        dropReviewFile();
        showReviewStatus(why);
        renderFileBar();
        if (S.screen === 3) renderReview();
      } else if (rv.file && sum.status !== rv.file.status && rv.loadingId !== rv.id) {
        openFile(rv.id, { keepView: true }); // not while one is on its way: a slow one would never land
      }
    } else if (S.screen === 3 && S.files.some((f) => REVIEWABLE.has(f.status))) {
      renderReview(); // Revisar showed no file and one became reviewable: open it
    }
    schedulePoll();
  }

  function schedulePoll() {
    clearTimeout(S.pollTimer);
    const busy = S.files.some(isActive) || justRequested();
    if (busy) S.pollTimer = setTimeout(refreshState, POLL_MS);
  }

  function renderAll() {
    renderSteps();
    renderHome();
    renderJobs();
    renderFileBar();
    if (S.screen === 4) renderExport();
  }

  // ---------------------------------------------------------------------------------------------
  // Screen 1: choose files
  // ---------------------------------------------------------------------------------------------
  /** The text of a file's status chip, except "Procesando" instead of the percentage. */
  function statusText(f) {
    const c = f.counts || {};
    switch (f.status) {
      case "processing": return "Procesando";
      case "queued": return S.requested.has(f.id) ? "En espera" : "Sin procesar";
      case "ready": return c.doubtful ? `${c.doubtful} por revisar` : "Listo para revisar";
      case "confirmed": return "Confirmado";
      case "exported": return "Exportado";
      case "error": return "Necesita tu ayuda";
      case "cancelled": return "Cancelado";
      default: return f.status || "";
    }
  }
  const STATUS_TONE = { processing: "run", confirmed: "ok", exported: "ok", error: "bad" };
  function statusChip(f) {
    const tone = f.status === "ready" ? ((f.counts || {}).doubtful ? "warn" : "ok") : STATUS_TONE[f.status] || "wait";
    const text = f.status === "processing" ? `${Math.round((f.progress || 0) * 100)} %` : statusText(f);
    return h("span", { class: `state ${tone}`, text });
  }

  function fileMeta(f) {
    const parts = [];
    parts.push(isImage(f) ? "Imagen" : "PDF");
    if (f.pages) parts.push(plural(f.pages, "página", "páginas"));
    const size = fmtSize(f.size || S.sizes.get(f.id));
    if (size) parts.push(size);
    return parts.join(" · ");
  }

  function renderHome() {
    const files = S.files;
    const panel = $("#files-panel");
    panel.hidden = files.length === 0;
    const pending = files.filter(isPending);
    $("#drop").classList.toggle("compact", files.length > 0);
    $("#files-title").textContent =
      pending.length === files.length
        ? `${plural(files.length, "archivo", "archivos")} ${files.length === 1 ? "listo" : "listos"} para procesar`
        : `${plural(files.length, "archivo", "archivos")} en esta sesión`;

    const list = $("#filelist");
    keepFocus(list, () => {
      list.replaceChildren(
        ...files.map((f) => {
          const meta = h("div", { class: "fmeta" }, fileMeta(f));
          const info = h("div", null, h("div", { class: "fname" }, nameNode(f.name)), meta);
          if (!isPending(f)) info.append(h("div", { class: "row" }, statusChip(f)));
          if (isImage(f)) {
            const cb = h("input", {
              type: "checkbox",
              dataset: { fk: `all:${f.id}` },
              disabled: f.status === "processing" || undefined,
              onchange: (e) => setAllText(f, e.target.checked),
            });
            cb.checked = !!f.all_text;
            info.append(h("label", { class: "toggle" }, cb, " Censurar todo el texto de esta imagen"));
            if (S.dirty.has(f.id)) {
              info.append(h("div", { class: "hint", text: "Cambiaste esta opción: vuelve a procesar el archivo para aplicarla." }));
            }
          }
          return h(
            "li",
            null,
            h("span", { class: "ficon", "aria-hidden": "true", text: fileBadge(f) }),
            info,
            h("button", {
              type: "button",
              class: "btn ghost small",
              dataset: { fk: `rm:${f.id}` },
              "aria-label": `Quitar ${f.name}`,
              text: "Quitar",
              onclick: () => removeFile(f),
            }),
          );
        }),
      );
    });

    $("#names-meta").textContent =
      S.namesCount > 0
        ? `${plural(S.namesCount, "entrada", "entradas")}. Se censuran donde aparezcan, aunque estén sin tildes o en otro orden.`
        : "La lista está vacía. Agrega los nombres y direcciones que siempre deben censurarse, aunque estén sin tildes o en otro orden.";

    renderDetects();
    renderEstimate();
    refreshEstimate();

    const btn = $("#b-process");
    const active = files.some(isActive);
    $("#process-hint").textContent = pending.length
      ? "Se procesan en este computador. Podrás revisar cada archivo antes de exportarlo."
      : "";
    if (pending.length) {
      btn.disabled = false;
      btn.textContent = `Procesar ${plural(pending.length, "archivo", "archivos")}`;
      btn.dataset.mode = "process";
    } else if (active) {
      btn.disabled = false;
      btn.textContent = "Ver el procesamiento";
      btn.dataset.mode = "view";
    } else {
      btn.disabled = true;
      btn.textContent = files.length ? "No hay archivos nuevos para procesar" : "Agrega archivos para empezar";
      btn.dataset.mode = "";
    }
  }

  async function setAllText(f, value) {
    try {
      const sum = await api(`/api/files/${enc(f.id)}/options`, { method: "POST", json: { all_text: value } });
      Object.assign(f, sum || { all_text: value });
      if (["ready", "confirmed", "exported", "error"].includes(f.status)) S.dirty.add(f.id);
      renderHome();
      toast(value ? "Se censurará todo el texto de esta imagen." : "Se censurará solo el texto con datos personales.");
    } catch (err) {
      showError(err);
      renderHome();
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Screen 1: what to search for (detection groups) and the time estimate
  // ---------------------------------------------------------------------------------------------
  /** Detection groups that were off for a file analyzed with ``options`` (switches that only
   *  decide whether something starts applied, like "the other URLs", are not detections). */
  const groupsOff = (options) =>
    options ? S.groups.filter((g) => g.detection && !g.locked && options[g.key] === false) : [];

  async function loadGroups() {
    try {
      const res = await api("/api/options");
      S.groups = (res && res.groups) || [];
    } catch (err) {
      showError(err);
    }
    renderDetects();
    renderEstimate();
    if (S.screen === 3 && S.rv.file) renderSkipped(); // the banner needs the names of the groups
  }

  function estimateFor(g) {
    const est = S.estimate;
    if (!est || !(est.files || []).length || !g.detection) return "";
    const seconds = (est.groups || {})[g.key] || 0;
    if (!g.enabled) return seconds >= 1 ? `ahorras ${fmtEstimate(seconds)}` : "";
    return fmtEstimate(seconds);
  }

  function renderDetects() {
    const panel = $("#detect-panel");
    panel.hidden = !S.groups.length;
    if (!S.groups.length) return;
    const hasFiles = !!(S.estimate && (S.estimate.files || []).length);
    $("#detect-hint").textContent = hasFiles ? "Tiempo estimado para los archivos sin procesar" : "";
    const list = $("#detects");
    keepFocus(list, () => {
      list.replaceChildren(...S.groups.map((g) => {
        const id = `det-${g.key}`;
        const est = estimateFor(g);
        const described = [`${id}-d`];
        if (g.locked) described.push(`${id}-r`);
        if (est) described.push(`${id}-t`);
        const warn = !g.enabled && g.warning;
        if (warn) described.push(`${id}-w`);
        const cb = h("input", {
          type: "checkbox", id, dataset: { fk: `det:${g.key}` },
          disabled: g.locked || undefined,
          "aria-describedby": described.join(" "),
          onchange: (e) => setGroup(g, e.target.checked),
        });
        cb.checked = !!g.enabled;
        const text = h("div", null,
          h("label", { for: id, text: g.label }),
          h("div", { class: "fmeta", id: `${id}-d`, text: g.description }),
          g.locked ? h("div", { class: "fmeta", id: `${id}-r`, text: g.locked_reason }) : null);
        const row = h("li", { class: `detect${g.locked ? " locked" : ""}${g.detection ? "" : " sub"}` },
          cb, text, h("span", { class: "est", id: `${id}-t`, text: est }));
        if (warn) row.append(h("p", { class: "warnline", id: `${id}-w` }, h("span", { "aria-hidden": "true", text: "⚠ " }), g.warning));
        return row;
      }));
    });
  }

  function renderEstimate() {
    const est = S.estimate;
    const line = $("#process-estimate");
    const show = !!(est && (est.files || []).length && S.files.some(isPending));
    line.textContent = show ? `Tiempo estimado: ${fmtEstimate(est.total)} · depende del computador` : "";
    line.title = show && !est.calibrated ? "Se ajusta con el tiempo real de cada archivo que proceses." : "";
    line.hidden = !show;
    // Groups the user turned off (on by default); "censurar también los otros enlaces" is not a detection.
    const off = S.groups.filter((g) => g.detection && !g.locked && g.default && !g.enabled);
    const warn = $("#process-off");
    warn.textContent = off.length
      ? `Apagaste ${plural(off.length, "detección", "detecciones")}: ${joinEs(off.map((g) => g.short))}.`
      : "";
    warn.hidden = !off.length;
  }

  /** Asks for the estimate when the files that would be processed change (or ``force``). */
  let estimateSeq = 0;
  async function refreshEstimate(force = false) {
    const key = S.files.filter(isPending).map((f) => f.id).sort().join(",");
    if (!force && key === S.estimateKey) return;
    S.estimateKey = key;
    const seq = ++estimateSeq;
    if (!key) {
      S.estimate = null;
    } else {
      try {
        const res = await api(`/api/estimate?ids=${enc(key)}`);
        if (seq !== estimateSeq) return;
        S.estimate = res;
      } catch {
        if (seq !== estimateSeq) return;
        S.estimate = null; // the estimate is only a guide: no error for it
        S.estimateKey = null;
      }
    }
    renderDetects();
    renderEstimate();
  }

  async function setGroup(g, value) {
    try {
      const res = await api("/api/options", { method: "PUT", json: { groups: { [g.key]: value } } });
      S.groups = (res && res.groups) || S.groups;
      let msg;
      if (g.key === "urls_other") {
        msg = value
          ? "Los otros enlaces se censurarán desde el inicio. Podrás dejar visibles los que quieras."
          : "Los otros enlaces quedarán sin censurar. En la revisión podrás censurar los que quieras.";
      } else {
        msg = value ? `Se buscará: ${g.short}.` : `No se buscará: ${g.short}.`;
      }
      if (S.files.some((f) => REVIEWABLE.has(f.status))) msg += " Se aplica a los archivos que proceses desde ahora.";
      toast(msg);
    } catch (err) {
      showError(err);
    }
    renderDetects();
    renderEstimate();
    refreshEstimate(true);
  }

  async function removeFile(f) {
    try {
      await api(`/api/files/${enc(f.id)}`, { method: "DELETE" });
      S.requested.delete(f.id);
      S.dirty.delete(f.id);
      S.sizes.delete(f.id);
      S.exp.results.delete(f.id);
      toast(`Se quitó ${f.name} de la lista. El original no cambia.`);
    } catch (err) {
      showError(err);
    }
    await refreshState();
  }

  async function removeAll() {
    const files = [...S.files];
    if (!files.length) return;
    const ok = await ask({
      title: "¿Quitar todos los archivos de la lista?",
      text: "Se descarta la revisión de cada archivo. Los originales no cambian.",
      yes: "Quitar todos",
      danger: true,
    });
    if (!ok) return;
    let failed = 0;
    for (const f of files) {
      try { await api(`/api/files/${enc(f.id)}`, { method: "DELETE" }); } catch { failed += 1; }
    }
    S.requested.clear(); S.dirty.clear(); S.exp.results.clear();
    await refreshState();
    toast(failed ? `No se pudieron quitar ${plural(failed, "archivo", "archivos")}.` : "Se quitaron todos los archivos de la lista.", { bad: failed > 0 });
  }

  async function uploadFiles(fileArray) {
    const all = [...fileArray].filter((f) => f && f.name);
    const files = all.filter((f) => isSupported(f.name));
    const skipped = all.length - files.length;
    if (!files.length) {
      toast(
        all.length ? "Ninguno de esos archivos es PDF, JPG, PNG, WEBP o TIFF." : "No se encontraron archivos para agregar.",
        { bad: all.length > 0 },
      );
      return;
    }
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    toast(`Agregando ${plural(files.length, "archivo", "archivos")}…`);
    try {
      const res = await api("/api/files", { method: "POST", form });
      const added = (res && res.files) || [];
      added.forEach((a, i) => {
        const src = files[i];
        if (src && src.name === a.name) S.sizes.set(a.id, src.size);
      });
      toast(addedMessage(added.length, skipped));
    } catch (err) {
      showError(err);
    }
    await refreshState();
    if (S.screen !== 1) go(1);
  }

  function addedMessage(added, skipped) {
    let msg = added
      ? `Se ${added === 1 ? "agregó" : "agregaron"} ${plural(added, "archivo", "archivos")}.`
      : "No se agregó ningún archivo.";
    if (skipped) msg += ` Se ${skipped === 1 ? "omitió" : "omitieron"} ${plural(skipped, "archivo", "archivos")} que no son PDF ni imágenes.`;
    return msg;
  }

  /** Add files by path (native dialog of the desktop shell): the server copies them. */
  async function addPaths(paths) {
    const list = (Array.isArray(paths) ? paths : [paths]).filter((x) => typeof x === "string" && x);
    if (!list.length) return; // the dialog was cancelled
    toast("Agregando archivos…");
    try {
      const res = await api("/api/files/from-paths", { method: "POST", json: { paths: list } });
      const added = (res && res.files) || [];
      const skipped = (res && res.skipped) || [];
      const details = skipped.map((x) => [x.name, x.message].filter(Boolean).join(": ")).filter(Boolean);
      const msg = addedMessage(added.length, 0) + (details.length ? ` ${details.slice(0, 3).join(" ")}` : "");
      toast(msg, { bad: !added.length && details.length > 0 });
    } catch (err) {
      showError(err);
    }
    await refreshState();
    if (S.screen !== 1) go(1);
  }

  async function chooseFiles() {
    if (hasBridge("choose_files")) {
      try {
        await addPaths(await bridge().choose_files());
        return;
      } catch {
        /* fall back to the browser picker */
      }
    }
    $("#in-files").click();
  }

  async function chooseFolderToAdd() {
    if (hasBridge("choose_folder")) {
      try {
        const res = await bridge().choose_folder();
        if (res) await addPaths(res);
        return;
      } catch {
        /* fall back to the browser picker */
      }
    }
    $("#in-dir").click();
  }

  // Drag and drop of files and whole folders anywhere on the window.
  async function filesFromDrop(dt) {
    const items = [...(dt.items || [])].filter((i) => i.kind === "file");
    if (!items.length) return [...(dt.files || [])];
    // Entries must be taken synchronously, before any await.
    const entries = items.map((i) => (typeof i.webkitGetAsEntry === "function" ? i.webkitGetAsEntry() : null));
    const loose = items.map((i, k) => (entries[k] ? null : i.getAsFile())).filter(Boolean);
    const out = [...loose];
    const readFile = (entry) => new Promise((resolve) => entry.file(resolve, () => resolve(null)));
    const readBatch = (reader) => new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
    async function walk(entry, depth) {
      if (!entry || depth > 20) return;
      if (entry.isFile) {
        const f = await readFile(entry);
        if (f) out.push(f);
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        for (;;) {
          const batch = await readBatch(reader);
          if (!batch.length) break;
          for (const e of batch) await walk(e, depth + 1);
        }
      }
    }
    for (const e of entries) await walk(e, 0);
    return out;
  }

  function initDragDrop() {
    const veil = $("#dropveil");
    let depth = 0;
    const hasFiles = (e) => e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files");
    window.addEventListener("dragenter", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth += 1;
      veil.hidden = false;
    });
    window.addEventListener("dragover", (e) => {
      // Always prevent the default, or the window would navigate to the dropped file.
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = hasFiles(e) ? "copy" : "none";
    });
    window.addEventListener("dragleave", (e) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (!depth) veil.hidden = true;
    });
    window.addEventListener("drop", async (e) => {
      e.preventDefault();
      depth = 0;
      veil.hidden = true;
      if (!hasFiles(e)) return;
      const files = await filesFromDrop(e.dataTransfer);
      uploadFiles(files);
    });
  }

  async function startProcessing(ids) {
    if (!ids.length) return;
    try {
      const res = await api("/api/process", { method: "POST", json: { file_ids: ids } });
      const started = (res && res.started) || ids;
      const now = Date.now();
      for (const id of new Set([...ids, ...started])) S.requested.set(id, now);
      for (const id of ids) S.dirty.delete(id);
    } catch (err) {
      showError(err);
    }
    if (S.screen !== 2) go(2);
    await refreshState();
  }

  // ---------------------------------------------------------------------------------------------
  // Name list dialog
  // ---------------------------------------------------------------------------------------------
  const parseNames = (text) => text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  function updateNamesCount() {
    const n = parseNames($("#names-text").value).length;
    $("#names-count").textContent = n ? `${plural(n, "entrada", "entradas")} en la lista.` : "La lista está vacía.";
  }
  async function openNames() {
    let entries = [];
    try {
      const res = await api("/api/names");
      entries = (res && res.entries) || [];
    } catch (err) {
      showError(err);
      return;
    }
    $("#names-text").value = entries.join("\n");
    updateNamesCount();
    openDialog($("#dlg-names"));
    $("#names-text").focus();
  }
  async function saveNames(e) {
    e.preventDefault();
    const entries = parseNames($("#names-text").value);
    const btn = $("#dlg-names-yes");
    btn.disabled = true;
    try {
      const res = await api("/api/names", { method: "PUT", json: { entries } });
      const saved = (res && res.entries) || entries;
      $("#dlg-names").close();
      const processed = S.files.some((f) => REVIEWABLE.has(f.status));
      toast(
        `Lista guardada: ${plural(saved.length, "entrada", "entradas")}.` +
          (processed ? " Se aplica a los archivos que proceses desde ahora." : ""),
      );
      await refreshState();
    } catch (err) {
      showError(err);
    } finally {
      btn.disabled = false;
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Dialog helpers
  // ---------------------------------------------------------------------------------------------
  function openDialog(dlg) {
    if (typeof dlg.showModal === "function") dlg.showModal();
    else dlg.setAttribute("open", "");
  }
  const anyDialogOpen = () => $$("dialog").some((d) => d.open);

  let askResolve = null;
  function ask({ title, text, yes = "Aceptar", danger = false }) {
    $("#dlg-ask-t").textContent = title;
    $("#dlg-ask-p").textContent = text;
    const y = $("#dlg-ask-yes");
    y.textContent = yes;
    y.className = `btn ${danger ? "danger" : "primary"}`;
    openDialog($("#dlg-ask"));
    $("#dlg-ask-no").focus();
    return new Promise((resolve) => { askResolve = resolve; });
  }
  function initAskDialog() {
    const dlg = $("#dlg-ask");
    $("#dlg-ask-form").addEventListener("submit", (e) => {
      e.preventDefault();
      dlg.close("yes");
    });
    $("#dlg-ask-no").addEventListener("click", () => dlg.close("no"));
    dlg.addEventListener("close", () => {
      const r = askResolve;
      askResolve = null;
      if (r) r(dlg.returnValue === "yes");
      dlg.returnValue = "";
    });
  }

  // ---------------------------------------------------------------------------------------------
  // Screen 2: processing
  // ---------------------------------------------------------------------------------------------
  function jobMeta(f) {
    const c = f.counts || {};
    switch (f.status) {
      case "processing": return f.step || "Procesando…";
      case "queued": return S.requested.has(f.id) ? f.step || "En espera" : "Todavía no se procesa";
      case "ready":
      case "confirmed":
      case "exported": {
        const parts = [`${plural(c.total || 0, "zona propuesta", "zonas propuestas")}`];
        if (c.doubtful) parts.push(`${plural(c.doubtful, "dudosa", "dudosas")} para revisar primero`);
        if (c.suggested) parts.push(`${plural(c.suggested, "otro enlace", "otros enlaces")} sin censurar`);
        return parts.join(" · ");
      }
      case "error": return "No se pudo procesar";
      case "cancelled": return "Se canceló el procesamiento";
      default: return "";
    }
  }

  /** "Listo en 12 s · texto en imágenes 9 s · rostros 2 s · QR 0,1 s" from the summary timings. */
  function timingText(t) {
    if (!t || t.analyze == null) return "";
    const parts = [`Listo en ${fmtMeasured(t.analyze)}`];
    for (const [stage, label] of Object.entries(STAGE_LABELS)) {
      if (t[stage] >= 0.05) parts.push(`${label} ${fmtMeasured(t[stage])}`);
    }
    return parts.join(" · ");
  }

  function jobRow(f) {
    const pct = Math.round(clamp(f.progress || 0, 0, 1) * 100);
    const actions = h("div", { class: "jright" }, statusChip(f));
    if (isActive(f)) {
      actions.append(h("button", {
        type: "button", class: "btn ghost small", dataset: { fk: `jc:${f.id}` },
        "aria-label": `Cancelar ${f.name}`, text: "Cancelar", onclick: () => cancel(f.id),
      }));
    } else if (REVIEWABLE.has(f.status)) {
      actions.append(h("button", {
        type: "button", class: "btn ghost small", dataset: { fk: `jr:${f.id}` },
        "aria-label": `Revisar ${f.name}`, text: "Revisar", onclick: () => openReview(f.id),
      }));
    } else if (f.status === "cancelled" || (f.status === "queued" && !S.requested.has(f.id))) {
      actions.append(h("button", {
        type: "button", class: "btn ghost small", dataset: { fk: `jp:${f.id}` },
        "aria-label": `Procesar ${f.name}`, text: "Procesar", onclick: () => startProcessing([f.id]),
      }));
    }
    const barClass = f.status === "error" ? "bar bad" : REVIEWABLE.has(f.status) ? "bar ok" : "bar";
    const shown = REVIEWABLE.has(f.status) ? 100 : f.status === "error" || f.status === "cancelled" ? 0 : pct;
    const fill = h("i", { style: { width: `${shown}%` } });
    const row = h(
      "div",
      { class: "job", dataset: { id: f.id } },
      h("span", { class: "ficon", "aria-hidden": "true", text: fileBadge(f) }),
      h("div", null,
        h("div", { class: "fname" }, nameNode(f.name)),
        h("div", { class: "fmeta", text: jobMeta(f) }),
        REVIEWABLE.has(f.status) && timingText(f.timings) ? h("div", { class: "fmeta", text: timingText(f.timings) }) : null),
      actions,
      h("div", {
        class: barClass, role: "progressbar", "aria-label": `Avance de ${f.name}`,
        "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(shown),
      }, fill),
    );
    if (f.status === "error") {
      const box = h("div", { class: "errbox" },
        h("p", null, h("b", { text: f.error_message || "Este archivo no se pudo procesar." })));
      const btns = h("div", { class: "row" });
      if (f.error === "internal") {
        btns.append(h("button", {
          type: "button", class: "btn small", dataset: { fk: `je:${f.id}` },
          text: "Intentar de nuevo", onclick: () => startProcessing([f.id]),
        }));
      }
      btns.append(h("button", {
        type: "button", class: "btn ghost small", dataset: { fk: `jx:${f.id}` },
        "aria-label": `Omitir ${f.name}`, text: "Omitir este archivo", onclick: () => removeFile(f),
      }));
      box.append(btns);
      row.append(box);
    }
    return row;
  }

  function renderJobs() {
    const files = S.files;
    const active = files.filter(isActive);
    $("#s2-title").textContent = active.length
      ? `Procesando ${plural(active.length, "archivo", "archivos")}`
      : files.some((f) => f.status !== "queued")
        ? "Procesamiento terminado"
        : "No hay archivos en proceso";
    $("#s2-sub").textContent = active.length
      ? "Puedes empezar a revisar los archivos listos mientras el resto termina."
      : "Revisa cada archivo antes de exportarlo: tú decides qué se censura.";
    $("#b-cancel-all").disabled = active.length === 0;
    $("#b-review-ready").disabled = !files.some((f) => REVIEWABLE.has(f.status));
    const box = $("#jobs");
    keepFocus(box, () => {
      if (!files.length) {
        box.replaceChildren(h("div", { class: "empty" }, "Todavía no agregas archivos. Vuelve a ", h("button", {
          type: "button", class: "btn ghost small", text: "Elegir archivos", onclick: () => go(1),
        }), "."));
        return;
      }
      box.replaceChildren(...files.map(jobRow));
    });
  }

  async function cancel(fileId) {
    try {
      await api("/api/cancel", { method: "POST", json: fileId ? { file_id: fileId } : {} });
      toast(fileId ? "Se canceló el procesamiento del archivo." : "Se canceló el procesamiento.");
    } catch (err) {
      showError(err);
    }
    await refreshState();
  }

  // ---------------------------------------------------------------------------------------------
  // Screen 3: review
  // ---------------------------------------------------------------------------------------------
  const rvFindings = () => (S.rv.file && S.rv.file.findings) || [];
  const findingById = (id) => rvFindings().find((f) => f.id === id) || null;
  const typeClass = (t) => `t-${TYPE_LABELS[t] ? t : "other"}`;
  const typeLabel = (t) => TYPE_LABELS[t] || "Otro dato";
  const findingValue = (f) => (f.text && f.text.trim()) || TYPE_PLACEHOLDERS[f.type] || typeLabel(f.type);
  function removedReason(f) {
    const hist = f.history || [];
    for (let i = hist.length - 1; i >= 0; i -= 1) if (hist[i].action === "removed") return hist[i].reason || "";
    return "";
  }
  const sortKey = (f) => {
    const b = bbox(f.polygon);
    return [f.page, Math.round(b.y), Math.round(b.x)];
  };
  const byPosition = (a, b) => {
    const ka = sortKey(a), kb = sortKey(b);
    return ka[0] - kb[0] || ka[1] - kb[1] || ka[2] - kb[2];
  };
  /** The order of the list and of J/K: doubtful first, then the rest, then the other URLs (D12). */
  function orderedVisible() {
    const vis = rvFindings().filter((f) => !S.rv.hidden.has(f.type));
    const main = vis.filter((f) => !f.optional);
    return [
      ...main.filter((f) => f.doubtful).sort(byPosition),
      ...main.filter((f) => !f.doubtful).sort(byPosition),
      ...vis.filter((f) => f.optional).sort(byPosition),
    ];
  }

  // --- opening and closing a file: every way in goes through openFile ---
  /** Procesar's "Revisar" and "Revisar los listos". The open file only shows Revisar again. */
  function openReview(id) {
    if (id && id !== S.rv.id) openFile(id);
    go(3);
  }

  // Stubs of the scrolling viewer, each replaced by the task that builds that part.
  function buildRows() {} // replaced in Task 9
  function relayout() {} // replaced in Task 9
  function anchorNow() { return null; } // replaced in Task 9
  function restoreAnchor() {} // replaced in Task 9
  function stopObserver() {} // replaced in Task 10
  function clearQueue() {} // replaced in Task 10
  function releaseRow() {} // replaced in Task 10
  function refreshVersions() {} // replaced in Task 11
  function renderZonesAll() {} // replaced in Task 12
  function revealFinding() {} // replaced in Task 12

  function focusViewport() { $("#viewport").focus({ preventScroll: true }); }

  /** Empty the viewport. The new generation makes every image response still on its way stale. */
  function teardownRows() {
    S.rv.gen += 1;
    stopObserver();
    clearQueue();
    for (const row of S.rv.rows) releaseRow(row, { all: true });
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

  /** Under the file bar: why the review closed a file ("" hides the line). */
  function showReviewStatus(text) {
    const el = $("#rv-status");
    el.hidden = !text;
    el.textContent = text || "";
  }

  /** Opens a file for review: fit zoom, no rotation, every type shown, draw mode off; V is kept.
   *  ``keepView`` reloads the open file in place. ``focus``: the viewport takes the focus once the
   *  file is loaded (an open from the select). ``auto``: the automatic choice, which keeps the
   *  status line and a choice still pending in the select. */
  async function openFile(id, { keepView = false, focus = false, auto = false } = {}) {
    try {
      const sum = fileById(id);
      if (!sum || !REVIEWABLE.has(sum.status)) return;
      const fresh = !keepView || id !== S.rv.id;
      if (fresh) {
        teardownRows();
        Object.assign(S.rv, { id, file: null, sel: null, zoom: 1, rot: 0, pan: 0, versions: [] });
        S.rv.hidden.clear();
        setDraw(false);
        $("#review-loading").hidden = false;
        if (!auto) {
          clearTimeout(fileSwitchTimer);
          fileSwitchTimer = null;
          showReviewStatus("");
        }
      }
      renderFileBar();
      const load = loadReviewFile(id, { keepView, focus }).catch(reportError); // sets rv.loadingId at once
      // The findings column and the verify bar leave the previous file now, not when this one arrives.
      if (fresh && S.screen === 3) renderReview();
      await load;
    } catch (err) {
      reportError(err);
    }
  }

  /** A failure in a flow started without await (polls, clicks): one message, no unhandled rejection. */
  function reportError(err) {
    console.error(err);
    showError(err instanceof ApiError ? err : null);
  }

  let loadSeq = 0; // only the latest load is applied
  async function loadReviewFile(id, { keepView = false, focus = false } = {}) {
    const rv = S.rv;
    const seq = ++loadSeq;
    rv.loadingId = id;
    let file;
    try {
      file = await api(`/api/files/${enc(id)}`);
    } catch (err) {
      if (seq !== loadSeq) return;
      rv.loadingId = null;
      if (rv.id !== id) return;
      $("#review-loading").hidden = true;
      if (!rv.file && S.screen === 3) {
        // Not renderReview(): it would try again at once, and again on every failure.
        renderReviewPlaceholder("No se pudo cargar el archivo.");
        $("#findlist").append(h("button", { type: "button", class: "btn small", text: "Reintentar", onclick: () => openFile(id) }));
      }
      showError(err);
      return;
    }
    if (seq !== loadSeq) return;
    rv.loadingId = null;
    if (rv.id !== id) return; // the user moved on to another file
    const keep = keepView && !!rv.file && rv.file.id === id;
    const pages = file.pages || [];
    const samePages = keep && rv.rows.length === pages.length
      && rv.rows.every((row, i) => row.page.width === pages[i].width && row.page.height === pages[i].height);
    rv.file = file;
    $("#review-loading").hidden = true;
    if (keep) {
      if (rv.sel && !findingById(rv.sel)) rv.sel = null;
    } else {
      // The first finding (doubtful ones come first) starts selected and highlighted.
      const first = orderedVisible()[0];
      rv.sel = first ? first.id : null;
      if (first) seenSet(id).add(first.id);
    }
    let anchor = null;
    if (samePages) {
      // Same pages: the rows, the anchor and the loaded befores stay; only the afters may change.
      rv.rows.forEach((row, i) => { row.page = pages[i]; });
      refreshVersions();
      renderZonesAll();
    } else {
      if (keep && rv.rows.length) anchor = anchorNow();
      if (rv.rows.length) teardownRows();
      rv.versions = ReviewCore.pageVersions(file.findings, pages.length);
      buildRows();
    }
    if (S.screen === 3) renderReview();
    if (anchor) restoreAnchor(anchor); // the page list changed: back to the same page, clamped
    else if (!keep) {
      if (rv.sel) revealFinding(rv.sel, { instant: true });
      else $("#viewport").scrollTop = 0;
    }
    // Unless the reviewer moved the focus elsewhere meanwhile.
    const active = document.activeElement;
    if (focus && S.screen === 3 && (!active || active === document.body || active.id === "rv-file")) focusViewport();
  }

  function renderReview() {
    const rv = S.rv;
    if (!rv.id) {
      // Automatic choice: the first file ready for review, else any reviewable one.
      const reviewable = S.files.filter((f) => REVIEWABLE.has(f.status));
      const next = reviewable.find((f) => f.status === "ready") || reviewable[0];
      if (next) openFile(next.id, { auto: true });
    } else if (!rv.file && rv.loadingId !== rv.id) {
      openFile(rv.id, { auto: true }); // its last load failed: try again
    }
    $("#review-empty").hidden = !!rv.id || S.files.some((f) => REVIEWABLE.has(f.status));
    $("#review").hidden = !rv.id;
    if (!rv.id) return;
    renderFileBar();
    if (!rv.file) {
      renderReviewPlaceholder("Cargando el archivo…");
      return;
    }
    renderSkipped();
    renderFindings();
    renderVerify();
  }

  /** The findings column and the verify bar with no file in them (loading, or the load failed). */
  function renderReviewPlaceholder(text) {
    $("#fcount").textContent = "";
    $("#chips").replaceChildren();
    for (const id of ["#skipped", "#leaks"]) { $(id).hidden = true; $(id).replaceChildren(); }
    $("#findlist").replaceChildren(h("p", { class: "nofind", text }));
    $("#vdot").className = "dot";
    $("#vtitle").textContent = "";
    $("#vsum").textContent = "";
    $("#vactions").replaceChildren();
  }

  // --- the file bar ---
  let fileSwitchTimer = null; // a choice in the select, waiting for its pause
  /** The reviewable file before (-1) or after (1) the open one, in list order, without wrapping. */
  function neighbourFile(delta) {
    const i = S.files.findIndex((f) => f.id === S.rv.id);
    if (i < 0) return null;
    for (let k = i + delta; k >= 0 && k < S.files.length; k += delta) {
      if (REVIEWABLE.has(S.files[k].status)) return S.files[k];
    }
    return null;
  }

  /** Runs on every poll, on any screen. The select and its options are never replaced, only
   *  updated where they changed, so an open dropdown stays open. */
  function renderFileBar() {
    const rv = S.rv;
    const select = $("#rv-file");
    const byId = new Map([...select.options].map((o) => [o.value, o]));
    S.files.forEach((f, i) => {
      let opt = byId.get(f.id);
      if (opt) byId.delete(f.id);
      else opt = h("option", { value: f.id });
      if (select.options[i] !== opt) select.insertBefore(opt, select.options[i] || null);
      const text = `${f.name} · ${statusText(f)}`;
      if (opt.textContent !== text) opt.textContent = text;
      const off = !REVIEWABLE.has(f.status);
      if (opt.disabled !== off) opt.disabled = off;
    });
    for (const gone of byId.values()) gone.remove();
    if (!fileSwitchTimer && select.value !== (rv.id || "")) select.value = rv.id || "";
    const sum = rv.id ? fileById(rv.id) : null;
    select.title = sum ? sum.name : "";
    const box = $("#rv-chip");
    const chip = sum ? statusChip(sum) : null;
    const old = box.firstElementChild;
    if (!chip) box.replaceChildren();
    else if (!old || old.className !== chip.className || old.textContent !== chip.textContent) box.replaceChildren(chip);
    for (const [delta, btn] of [[-1, $("#rv-prev")], [1, $("#rv-next")]]) {
      const off = !neighbourFile(delta);
      if (btn.disabled === off) continue;
      const focused = document.activeElement === btn;
      btn.disabled = off;
      if (off && focused) focusViewport();
    }
  }

  /** Banner when the file was analyzed with detection groups off: what was not searched. */
  function renderSkipped() {
    const file = S.rv.file;
    const box = $("#skipped");
    const off = file ? groupsOff(file.options) : [];
    box.hidden = !off.length;
    if (!off.length) { box.replaceChildren(); return; }
    const parts = [h("p", null,
      h("b", { text: `En este archivo no se buscaron: ${joinEs(off.map((g) => g.short))}.` }),
      " Revisa esas partes a mano.")];
    if (file.options && file.options.ocr === false) {
      const scanned = (file.pages || []).filter((p) => p.scanned).length;
      if (file.kind === "image") {
        parts.push(h("p", { text: "Es una imagen y su texto no se leyó: no se buscó ningún dato escrito dentro de ella (RUT, correos, teléfonos, nombres ni enlaces)." }));
      } else if (scanned) {
        parts.push(h("p", { text: `Tiene ${plural(scanned, "página escaneada", "páginas escaneadas")} y su texto no se leyó: no se buscó ningún dato dentro de ${scanned === 1 ? "ella" : "ellas"}.` }));
      }
    }
    box.replaceChildren(...parts);
  }

  // --- thumbnails (real page renders, loaded lazily, two at a time) ---
  const thumbCache = new Map(); // "fileId:page" -> object URL
  const thumbQueue = [];
  let thumbRunning = 0;
  let thumbObserver = null;
  function clearThumbCache(fileId) {
    for (const [k, url] of thumbCache) {
      if (k.startsWith(`${fileId}:`)) { URL.revokeObjectURL(url); thumbCache.delete(k); }
    }
  }
  function pumpThumbs() {
    while (thumbRunning < 2 && thumbQueue.length) {
      const job = thumbQueue.shift();
      thumbRunning += 1;
      apiBlobUrl(job.path)
        .then((url) => {
          thumbCache.set(job.key, url);
          const img = document.querySelector(`img[data-thumb="${CSS.escape(job.key)}"]`);
          if (img) img.src = url;
        })
        .catch(() => { /* the main viewer reports page errors */ })
        .finally(() => { thumbRunning -= 1; pumpThumbs(); });
    }
  }
  function requestThumb(img) {
    const key = img.dataset.thumb;
    if (thumbCache.has(key)) { img.src = thumbCache.get(key); return; }
    if (thumbQueue.some((j) => j.key === key)) return;
    thumbQueue.push({ key, path: img.dataset.path });
    pumpThumbs();
  }

  function renderThumbs() {
    const rv = S.rv;
    const file = rv.file;
    const box = $("#thumbs");
    if (thumbObserver) thumbObserver.disconnect();
    thumbQueue.length = 0;
    const width = Math.max(80, box.clientWidth - 28 || 170);
    const dpr = window.devicePixelRatio || 1;
    const counts = new Map();
    for (const f of rvFindings()) if (isApplied(f)) counts.set(f.page, (counts.get(f.page) || 0) + 1);
    thumbObserver = "IntersectionObserver" in window
      ? new IntersectionObserver((entries) => {
        for (const e of entries) if (e.isIntersecting) { requestThumb(e.target); thumbObserver.unobserve(e.target); }
      }, { root: $(".col-files"), rootMargin: "300px" })
      : null;
    keepFocus(box, () => {
      box.replaceChildren(
        ...(file.pages || []).map((p, i) => {
          const zoom = Math.max(0.02, Math.round(((width * dpr) / (p.width || 1)) * 100) / 100);
          const key = `${file.id}:${i}`;
          const n = counts.get(i) || 0;
          const img = h("img", {
            alt: "",
            dataset: { thumb: key, path: `/api/files/${enc(file.id)}/pages/${i}.png?zoom=${zoom}` },
            style: { aspectRatio: `${p.width} / ${p.height}` },
          });
          const btn = h("button", {
            type: "button",
            class: "thumb",
            dataset: { fk: `th:${i}` },
            "aria-current": i === rv.page ? "true" : null,
            "aria-label": `Página ${i + 1}, ${plural(n, "zona", "zonas")}`,
            onclick: () => setPage(i),
          }, img, h("span", { class: "pn", "aria-hidden": "true", text: String(i + 1) }), h("b", { "aria-hidden": "true", text: String(n) }));
          if (thumbCache.has(key)) img.src = thumbCache.get(key);
          else if (thumbObserver) thumbObserver.observe(img);
          else requestThumb(img);
          return btn;
        }),
      );
    });
  }

  function setPage(i) {
    const rv = S.rv;
    if (!rv.file) return;
    const n = (rv.file.pages || []).length;
    if (!n) return;
    i = clamp(i, 0, n - 1);
    if (i === rv.page) return;
    rv.page = i;
    $("#viewport").scrollTo({ top: 0, left: 0 });
    renderThumbs();
    layoutPage();
    renderFindings();
  }

  // --- viewer geometry ---
  function currentPage() {
    const rv = S.rv;
    return rv.file && rv.file.pages ? rv.file.pages[rv.page] || null : null;
  }
  function fitScale() {
    const p = currentPage();
    if (!p) return 1;
    const vp = $("#viewport");
    const avail = Math.max(200, vp.clientWidth - 56);
    const w = S.rv.rot % 180 ? p.height : p.width;
    const cap = p.unit === "pt" ? 1.6 : 1;
    return clamp(avail / (w || 1), 0.05, cap);
  }
  const scaleNow = () => S.rv.scale ?? fitScale();

  function layoutPage() {
    const rv = S.rv;
    const p = currentPage();
    const box = $("#pagebox"), inner = $("#pageinner"), img = $("#pageimg");
    if (!p) {
      box.hidden = true;
      $("#pageno").textContent = "";
      return;
    }
    box.hidden = false;
    const s = scaleNow();
    const W = p.width * s, H = p.height * s;
    Object.assign(inner.style, { width: `${W}px`, height: `${H}px` });
    Object.assign(img.style, { width: `${W}px`, height: `${H}px` });
    const rotated = rv.rot % 180 !== 0;
    Object.assign(box.style, { width: `${rotated ? H : W}px`, height: `${rotated ? W : H}px` });
    const t = {
      0: "none",
      90: `translate(${H}px, 0) rotate(90deg)`,
      180: `translate(${W}px, ${H}px) rotate(180deg)`,
      270: `translate(0, ${W}px) rotate(270deg)`,
    }[rv.rot];
    inner.style.transform = t;
    inner.classList.toggle("drawing", rv.draw);
    inner.classList.toggle("result", rv.result);
    img.alt = `Página ${rv.page + 1} de ${rv.file.name}`;
    $("#b-zfit").textContent = `${Math.round(s * 100)} %`;
    $("#b-zfit").setAttribute("aria-label", `Zoom ${Math.round(s * 100)} %. Ajustar al ancho`);
    $("#pageno").textContent = `· ${rv.page + 1} de ${rv.file.pages.length}`;
    $("#b-draw").setAttribute("aria-pressed", String(rv.draw));
    $("#b-view").setAttribute("aria-pressed", String(rv.result));
    renderZones();
    loadPageImage();
  }

  let lastImageKey = "";
  let imageTimer = null;
  let imageSeq = 0;
  let imageUrl = null;
  function loadPageImage() {
    const rv = S.rv;
    const p = currentPage();
    if (!p) return;
    const dpr = window.devicePixelRatio || 1;
    let zoom = scaleNow() * dpr;
    const maxSide = 9000;
    zoom = Math.min(zoom, maxSide / Math.max(p.width, p.height, 1));
    zoom = Math.max(0.02, Math.round(zoom * 100) / 100);
    const key = `${rv.file.id}:${rv.page}:${zoom}`;
    if (key === lastImageKey) return;
    if (!lastImageKey.startsWith(`${rv.file.id}:${rv.page}:`)) $("#pageimg").removeAttribute("src");
    lastImageKey = key;
    clearTimeout(imageTimer);
    const seq = ++imageSeq;
    const path = `/api/files/${enc(rv.file.id)}/pages/${rv.page}.png?zoom=${zoom}`;
    imageTimer = setTimeout(async () => {
      const loading = $("#page-loading");
      loading.hidden = false;
      try {
        const url = await apiBlobUrl(path);
        if (seq !== imageSeq) { URL.revokeObjectURL(url); return; }
        const img = $("#pageimg");
        const old = imageUrl;
        imageUrl = url;
        img.src = url;
        if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
      } catch (err) {
        if (seq === imageSeq) { lastImageKey = ""; showError(err); }
      } finally {
        if (seq === imageSeq) loading.hidden = true;
      }
    }, 120);
  }

  // Approximate size of a zone label (10px bold UI font), used to keep labels off other zones.
  const LABEL_H = 16;
  const labelWidth = (text) => 10 + text.length * 6.2;
  const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

  /** Label placement per zone: "top" (as in the design), "below", or "none" when the label
   *  would cover another zone or label (it then shows on hover). The selected zone always
   *  shows its label. */
  function placeLabels(items) {
    const placed = [];
    const out = new Map();
    const sel = S.rv.sel;
    const order = [...items].sort((a, b) => (b.f.id === sel) - (a.f.id === sel) || a.r.y - b.r.y || a.r.x - b.r.x);
    for (const it of order) {
      const suffix = { removed: " · quitada", suggested: " · sin censurar" }[it.f.status] || "";
      const text = typeLabel(it.f.type) + suffix;
      const w = labelWidth(text);
      const above = { x: it.r.x - 2, y: it.r.y - LABEL_H - 2, w, h: LABEL_H };
      const below = { x: it.r.x - 2, y: it.r.y + it.r.h + 2, w, h: LABEL_H };
      const free = (cand) => !placed.some((p) => overlaps(cand, p)) && !items.some((o) => o !== it && overlaps(cand, o.r));
      let where = "none";
      if (it.f.id === sel || free(above)) where = "top";
      else if (free(below)) where = "below";
      if (where !== "none") placed.push(where === "top" ? above : below);
      out.set(it.f.id, where);
    }
    return out;
  }

  function renderZones() {
    const rv = S.rv;
    const layer = $("#zones");
    const s = scaleNow();
    const items = rvFindings()
      .filter((f) => f.page === rv.page && !rv.hidden.has(f.type))
      .map((f) => {
        const b = bbox(f.polygon);
        return { f, r: { x: b.x * s, y: b.y * s, w: Math.max(4, b.w * s), h: Math.max(4, b.h * s) } };
      });
    const labels = placeLabels(items);
    const zones = items
      .map(({ f, r }) => {
        const cls = ["zone", typeClass(f.type)];
        const where = labels.get(f.id);
        if (where === "below") cls.push("label-below");
        if (where === "none") cls.push("nolabel");
        if (f.doubtful) cls.push("doubt");
        if (f.status === "removed") cls.push("removed");
        if (f.status === "suggested") cls.push("suggested");
        if (f.id === rv.sel) cls.push("sel");
        return h("div", {
          class: cls.join(" "),
          dataset: { id: f.id, label: typeLabel(f.type) },
          title: `${typeLabel(f.type)}: ${findingValue(f)}${f.status === "suggested" ? " (sin censurar)" : ""}`,
          style: { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` },
          onclick: (e) => { e.stopPropagation(); if (!rv.draw) select(f.id); },
        });
      });
    layer.replaceChildren(...zones);
  }

  function scrollToSelected() {
    const rv = S.rv;
    const behavior = reducedMotion() ? "auto" : "smooth";
    const zone = rv.sel ? $(`#zones [data-id="${CSS.escape(rv.sel)}"]`) : null;
    if (zone) {
      const vp = $("#viewport");
      const vr = vp.getBoundingClientRect(), zr = zone.getBoundingClientRect();
      const m = 48;
      let top = vp.scrollTop, left = vp.scrollLeft;
      if (zr.top < vr.top + m || zr.bottom > vr.bottom - m) top += zr.top + zr.height / 2 - (vr.top + vr.height / 2);
      if (zr.left < vr.left + m || zr.right > vr.right - m) left += zr.left + zr.width / 2 - (vr.left + vr.width / 2);
      vp.scrollTo({ top, left, behavior });
    }
    const item = rv.sel ? $(`#findlist .item[data-id="${CSS.escape(rv.sel)}"]`) : null;
    if (item) {
      const list = $("#findlist");
      const lr = list.getBoundingClientRect(), ir = item.getBoundingClientRect();
      if (ir.top < lr.top) list.scrollTo({ top: list.scrollTop + ir.top - lr.top - 40, behavior });
      else if (ir.bottom > lr.bottom) list.scrollTo({ top: list.scrollTop + ir.bottom - lr.bottom + 12, behavior });
    }
  }

  function select(id, { auto = false } = {}) {
    const rv = S.rv;
    const f = findingById(id);
    if (!f) return;
    rv.sel = id;
    if (!auto) seenSet(rv.id).add(id);
    if (f.page !== rv.page) {
      rv.page = f.page;
      renderThumbs();
      layoutPage();
    } else {
      renderZones();
    }
    renderFindings();
    renderVerify();
    requestAnimationFrame(scrollToSelected);
  }

  function move(delta) {
    const list = orderedVisible();
    if (!list.length) { toast("No hay hallazgos visibles con el filtro actual."); return; }
    let i = list.findIndex((f) => f.id === S.rv.sel);
    i = i < 0 ? (delta > 0 ? 0 : list.length - 1) : (i + delta + list.length) % list.length;
    select(list[i].id);
  }

  // --- findings column ---
  function findingItem(f) {
    const rv = S.rv;
    const removed = f.status === "removed";
    const suggested = f.status === "suggested";
    const why = [typeLabel(f.type), `pág. ${f.page + 1}`, DETECTOR_LABELS[f.detector] || f.detector || ""].filter(Boolean).join(" · ");
    const whyEl = h("span", { class: "why" }, why);
    if (f.doubtful) whyEl.append(" · ", h("b", { text: f.doubt_reason || "Hallazgo dudoso" }));
    if (f.doubtful && !seenSet(rv.id).has(f.id)) whyEl.append(" · ", h("span", { class: "new", text: "sin abrir" }));
    if (removed) {
      const r = removedReason(f);
      whyEl.append(` · quitada${r ? `: ${r.toLowerCase()}` : ""}`);
    }
    if (f.optional) whyEl.append(suggested ? " · sin censurar" : " · se censura");
    let action;
    if (f.optional) {
      // D12: an optional URL is censored or left visible with one click, without a reason.
      action = h("button", {
        type: "button", class: "btn ghost small", dataset: { fk: `fa:${f.id}` },
        "aria-label": suggested ? `Censurar el enlace ${findingValue(f)}` : `No censurar el enlace ${findingValue(f)}`,
        text: suggested ? "Censurar" : "No censurar",
        onclick: () => toggleOptional(f.id),
      });
    } else if (removed) {
      action = h("button", {
        type: "button", class: "btn ghost small", dataset: { fk: `fa:${f.id}` },
        "aria-label": `Restaurar la censura de ${findingValue(f)}`, text: "Restaurar",
        onclick: () => restoreFinding(f.id),
      });
    } else {
      action = h("button", {
        type: "button", class: "btn ghost small", dataset: { fk: `fa:${f.id}` },
        "aria-label": `Quitar la censura de ${findingValue(f)}`, text: "Quitar…",
        onclick: () => askRemove(f.id),
      });
    }
    return h("li", {
      class: `item ${typeClass(f.type)}${f.id === rv.sel ? " sel" : ""}${removed ? " removed" : ""}${suggested ? " suggested" : ""}`,
      dataset: { id: f.id },
    },
    h("button", {
      type: "button", class: "imain", dataset: { fk: `fm:${f.id}` },
      "aria-current": f.id === rv.sel ? "true" : null,
      onclick: () => select(f.id),
    },
    h("span", { class: "sw", "aria-hidden": "true" }),
    h("span", { class: "val", text: findingValue(f) }),
    whyEl),
    h("span", { class: "acts" }, action));
  }

  function renderFindings() {
    const rv = S.rv;
    const all = rvFindings();
    const active = all.filter(isApplied).length;
    $("#fcount").textContent = `${plural(active, "zona", "zonas")} en este archivo`;

    // Type chips double as the color legend: color dot + text label + count.
    const chips = $("#chips");
    const present = TYPE_ORDER.filter((t) => all.some((f) => f.type === t));
    for (const f of all) if (!TYPE_LABELS[f.type] && !present.includes(f.type)) present.push(f.type);
    keepFocus(chips, () => {
      chips.replaceChildren(
        ...present.map((t) => {
          const n = all.filter((f) => f.type === t).length;
          const on = !rv.hidden.has(t);
          return h("button", {
            type: "button", class: `chip ${typeClass(t)}`, dataset: { fk: `ch:${t}` },
            "aria-pressed": String(on),
            title: on ? `Ocultar ${typeLabel(t)}` : `Mostrar ${typeLabel(t)}`,
            onclick: () => {
              if (rv.hidden.has(t)) rv.hidden.delete(t); else rv.hidden.add(t);
              renderZones();
              renderFindings();
            },
          }, h("i", { "aria-hidden": "true" }), `${typeLabel(t)} ${n}`);
        }),
      );
    });

    // Leaks found by the last leak check (after an export attempt).
    const leaks = (rv.file && rv.file.leaks) || [];
    const lbox = $("#leaks");
    lbox.hidden = leaks.length === 0;
    if (leaks.length) {
      lbox.replaceChildren(
        h("b", { text: `${plural(leaks.length, "fuga sin resolver", "fugas sin resolver")}` }),
        h("ul", null, leaks.map((l) => h("li", null,
          (l.page != null ? `Pág. ${l.page + 1}: ` : "") + (l.message || ""),
          l.finding_id && findingById(l.finding_id)
            ? [" ", h("button", { type: "button", class: "btn ghost small", text: "Ver", onclick: () => select(l.finding_id) })]
            : null))),
      );
    }

    const vis = orderedVisible();
    const doubt = vis.filter((f) => f.doubtful && !f.optional);
    const rest = vis.filter((f) => !f.doubtful && !f.optional);
    const other = vis.filter((f) => f.optional);
    const anyOther = all.some((f) => f.optional);
    const unapplied = all.filter((f) => f.status === "suggested").length;
    const list = $("#findlist");
    keepFocus(list, () => {
      const parts = [];
      if (!all.length) {
        parts.push(h("p", { class: "nofind" },
          "No se detectaron datos personales en este archivo. Revisa igual cada página y agrega zonas con «Dibujar zona» si hace falta."));
      } else {
        parts.push(h("h3", { class: "group doubt", id: "g-doubt" }, h("span", { text: "Para revisar primero" }), h("span", { text: String(doubt.length) })));
        parts.push(doubt.length
          ? h("ul", { class: "items", "aria-labelledby": "g-doubt" }, doubt.map(findingItem))
          : h("p", { class: "nofind", text: "No hay hallazgos dudosos con el filtro actual." }));
        parts.push(h("h3", { class: "group", id: "g-found" }, h("span", { text: "Detectados" }), h("span", { text: String(rest.length) })));
        parts.push(rest.length
          ? h("ul", { class: "items", "aria-labelledby": "g-found" }, rest.map(findingItem))
          : h("p", { class: "nofind", text: "No hay hallazgos con el filtro actual." }));
      }
      if (anyOther) {
        // With "Censurar también los otros enlaces" on, this file's other URLs started censored.
        const startedApplied = !!(rv.file && rv.file.options && rv.file.options.urls_other);
        parts.push(h("h3", { class: "group", id: "g-other" },
          h("span", { text: startedApplied ? "Otros enlaces" : "Otros enlaces (sin censurar)" }), h("span", { text: String(other.length) })));
        parts.push(h("p", { class: "group-note", text: startedApplied
          ? "Sitios institucionales o documentos públicos. Se censuran porque así lo elegiste antes de procesar: deja visibles los que quieras."
          : "Sitios institucionales o documentos públicos. No se censuran a menos que tú lo decidas." }));
        parts.push(h("div", { class: "group-acts" }, h("button", {
          type: "button", class: "btn small", dataset: { fk: "apply-other" },
          disabled: !unapplied || undefined,
          text: "Censurar todos los otros enlaces",
          onclick: applyAllOptional,
        })));
        parts.push(other.length
          ? h("ul", { class: "items", "aria-labelledby": "g-other" }, other.map(findingItem))
          : h("p", { class: "nofind", text: "No hay otros enlaces con el filtro actual." }));
      }
      list.replaceChildren(...parts);
    });
  }

  // --- verification footer ---
  function renderVerify() {
    const rv = S.rv;
    const file = rv.file;
    if (!file) return;
    const all = rvFindings();
    const active = all.filter(isApplied).length;
    const removed = all.filter((f) => f.status === "removed").length;
    const suggested = all.filter((f) => f.status === "suggested").length;
    const added = all.filter((f) => f.type === "manual" || f.detector === "reviewer").length;
    const seen = seenSet(file.id);
    const unseen = all.filter((f) => f.doubtful && !seen.has(f.id)).length;
    const leaks = (file.leaks || []).length;
    const parts = [`${plural(active, "zona", "zonas")} a censurar`];
    if (removed) parts.push(`${plural(removed, "quitada", "quitadas")} por ti`);
    if (added) parts.push(`${plural(added, "agregada", "agregadas")} por ti`);
    if (suggested) parts.push(`${plural(suggested, "otro enlace", "otros enlaces")} sin censurar`);
    if (unseen) parts.push(`${plural(unseen, "hallazgo dudoso", "hallazgos dudosos")} sin abrir`);
    if (leaks) parts.push(`${plural(leaks, "fuga sin resolver", "fugas sin resolver")}`);
    else parts.push("la verificación de fugas se hace al exportar");
    $("#vsum").textContent = parts.join(" · ");

    const dot = $("#vdot");
    dot.className = `dot${leaks ? " bad" : unseen ? " warn" : ""}`;
    const title = $("#vtitle");
    const actions = $("#vactions");
    const nextReady = S.files.find((f) => f.status === "ready" && f.id !== file.id);
    const btns = [];
    if (file.status === "ready") {
      title.textContent = "Revisión lista para confirmar";
      btns.push(h("button", { type: "button", class: "btn primary", id: "b-confirm", text: "Confirmar este archivo", onclick: confirmFile }));
    } else {
      title.textContent = file.status === "exported" ? "Archivo exportado" : "Revisión confirmada";
      if (nextReady) {
        btns.push(h("button", {
          type: "button", class: "btn", text: "Siguiente archivo por revisar",
          onclick: () => { openFile(nextReady.id); if (S.screen !== 3) go(3); },
        }));
      }
      btns.push(h("button", { type: "button", class: "btn primary", text: "Ir a Exportar", onclick: () => go(4) }));
    }
    keepFocus(actions, () => actions.replaceChildren(...btns));
  }

  async function confirmFile() {
    const rv = S.rv;
    const file = rv.file;
    if (!file || file.status !== "ready") return;
    const seen = seenSet(file.id);
    const unseen = rvFindings().filter((f) => f.doubtful && !seen.has(f.id)).length;
    if (unseen) {
      const ok = await ask({
        title: `${unseen === 1 ? "Queda" : "Quedan"} ${plural(unseen, "hallazgo dudoso", "hallazgos dudosos")} sin abrir`,
        text: "Te recomendamos revisarlos uno por uno (con J y K) antes de confirmar. ¿Quieres confirmar igual?",
        yes: "Confirmar igual",
      });
      if (!ok) return;
    }
    try {
      const sum = await api(`/api/files/${enc(file.id)}/confirm`, { method: "POST" });
      file.status = (sum && sum.status) || "confirmed";
      toast("Archivo confirmado. Puedes seguir con el siguiente o exportar.");
    } catch (err) {
      showError(err);
    }
    await refreshState();
    renderVerify();
  }

  // --- remove / restore ---
  let pendingRemove = null;
  function askRemove(id) {
    const f = findingById(id || S.rv.sel);
    if (!f) { toast("Primero selecciona un hallazgo."); return; }
    if (f.optional) {
      // D12: an optional URL is left visible with "No censurar" (no reason needed).
      if (f.status === "suggested") toast("Este enlace no está censurado. Usa «Censurar» si quieres censurarlo.");
      else toggleOptional(f.id);
      return;
    }
    if (f.status === "removed") { toast("Esta censura ya está quitada. Usa «Restaurar» si quieres volver a censurarla."); return; }
    pendingRemove = f.id;
    if (S.rv.sel !== f.id) select(f.id);
    $("#dlg-remove-val").textContent = `${typeLabel(f.type)}: ${findingValue(f)}`;
    $("#dlg-reason").selectedIndex = 0;
    $("#dlg-note").value = "";
    openDialog($("#dlg-remove"));
    $("#dlg-remove-no").focus();
  }
  async function doRemove(e) {
    e.preventDefault();
    const id = pendingRemove;
    if (!id) return;
    const reason = $("#dlg-reason").value;
    const note = $("#dlg-note").value.trim();
    const btn = $("#dlg-remove-yes");
    btn.disabled = true;
    try {
      const body = { action: "remove", reason };
      if (note) body.note = note;
      const updated = await api(`/api/files/${enc(S.rv.id)}/findings/${enc(id)}`, { method: "PATCH", json: body });
      replaceFinding(updated);
      $("#dlg-remove").close();
      toast("Censura quitada. Queda registrada en el informe de auditoría.");
      focusFinding(id);
      refreshState();
    } catch (err) {
      showError(err);
    } finally {
      btn.disabled = false;
      pendingRemove = null;
    }
  }
  async function restoreFinding(id) {
    try {
      const updated = await api(`/api/files/${enc(S.rv.id)}/findings/${enc(id)}`, { method: "PATCH", json: { action: "restore" } });
      replaceFinding(updated);
      toast("Censura restaurada.");
      focusFinding(id);
      refreshState();
    } catch (err) {
      showError(err);
    }
  }
  // --- other URLs (D12): censor or leave visible ---
  async function toggleOptional(id) {
    const f = findingById(id);
    if (!f || !f.optional) return;
    const action = f.status === "suggested" ? "apply" : "skip";
    try {
      const updated = await api(`/api/files/${enc(S.rv.id)}/findings/${enc(id)}`, { method: "PATCH", json: { action } });
      replaceFinding(updated);
      toast(action === "apply"
        ? "Este enlace se censurará."
        : "Este enlace quedará visible. Queda registrado en el informe de auditoría.");
      focusFinding(id);
      refreshState();
    } catch (err) {
      showError(err);
    }
  }
  async function applyAllOptional() {
    try {
      const res = await api(`/api/files/${enc(S.rv.id)}/findings/apply-optional`, { method: "POST" });
      const applied = (res && res.applied) || [];
      if (S.rv.file) {
        const list = S.rv.file.findings;
        for (const u of applied) {
          const i = list.findIndex((f) => f.id === u.id);
          if (i >= 0) list[i] = u;
        }
      }
      renderZones();
      renderFindings();
      renderThumbs();
      renderVerify();
      toast(applied.length
        ? `Se ${applied.length === 1 ? "censurará" : "censurarán"} ${plural(applied.length, "enlace más", "enlaces más")}.`
        : "No quedaban otros enlaces sin censurar.");
      const first = applied[0] && $(`#findlist [data-fk="fa:${CSS.escape(applied[0].id)}"]`);
      if (first) first.focus({ preventScroll: true });
      refreshState();
    } catch (err) {
      showError(err);
    }
  }

  function replaceFinding(updated) {
    if (!updated || !S.rv.file) return;
    const list = S.rv.file.findings;
    const i = list.findIndex((f) => f.id === updated.id);
    if (i >= 0) list[i] = updated; else list.push(updated);
    renderZones();
    renderFindings();
    renderThumbs();
    renderVerify();
  }
  function focusFinding(id) {
    const el = $(`#findlist [data-fk="fa:${CSS.escape(id)}"]`) || $(`#findlist [data-fk="fm:${CSS.escape(id)}"]`);
    if (el) el.focus({ preventScroll: true });
  }

  // --- tools ---
  function zoomBy(factor) {
    const rv = S.rv;
    if (!currentPage()) return;
    const vp = $("#viewport");
    const cx = (vp.scrollLeft + vp.clientWidth / 2) / Math.max(1, vp.scrollWidth);
    const cy = (vp.scrollTop + vp.clientHeight / 2) / Math.max(1, vp.scrollHeight);
    rv.scale = clamp(scaleNow() * factor, 0.05, 8);
    layoutPage();
    vp.scrollLeft = cx * vp.scrollWidth - vp.clientWidth / 2;
    vp.scrollTop = cy * vp.scrollHeight - vp.clientHeight / 2;
  }
  function zoomFit() {
    S.rv.scale = null;
    layoutPage();
  }
  function rotate() {
    S.rv.rot = (S.rv.rot + 90) % 360;
    layoutPage();
    toast(S.rv.rot ? `Vista girada ${S.rv.rot}°. El archivo exportado conserva su orientación.` : "Vista sin girar.");
  }
  function setDraw(on) {
    const rv = S.rv;
    if (rv.draw === on) return;
    rv.draw = on;
    $("#b-draw").setAttribute("aria-pressed", String(on));
    $("#pageinner").classList.toggle("drawing", on);
    if (!on) cancelGhost();
    if (on) toast("Arrastra sobre el documento para agregar una zona. Esc para salir.");
  }
  function setView(on) { // replaced by setAfter in Task 9
    S.rv.after = on;
    $("#b-view").setAttribute("aria-pressed", String(on));
    relayout();
  }

  // --- draw a manual zone (screen coordinates -> view space) ---
  let drag = null;
  function pointToInner(clientX, clientY) {
    const p = currentPage();
    const s = scaleNow();
    const W = p.width * s, H = p.height * s;
    const r = $("#pagebox").getBoundingClientRect();
    const ox = clientX - r.left, oy = clientY - r.top;
    let x, y;
    switch (S.rv.rot) {
      case 90: x = oy; y = H - ox; break;
      case 180: x = W - ox; y = H - oy; break;
      case 270: x = W - oy; y = ox; break;
      default: x = ox; y = oy;
    }
    return { x: clamp(x, 0, W), y: clamp(y, 0, H) };
  }
  function cancelGhost() {
    if (drag && drag.ghost) drag.ghost.remove();
    drag = null;
  }
  function initDrawing() {
    const inner = $("#pageinner");
    inner.addEventListener("pointerdown", (e) => {
      if (!S.rv.draw || !currentPage() || e.button !== 0) return;
      e.preventDefault();
      const start = pointToInner(e.clientX, e.clientY);
      const ghost = h("div", { class: "zone t-manual ghost", dataset: { label: TYPE_LABELS.manual } });
      $("#zones").append(ghost);
      drag = { start, ghost, rect: { x: start.x, y: start.y, w: 0, h: 0 } };
      inner.setPointerCapture(e.pointerId);
    });
    inner.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const p = pointToInner(e.clientX, e.clientY);
      const r = {
        x: Math.min(p.x, drag.start.x), y: Math.min(p.y, drag.start.y),
        w: Math.abs(p.x - drag.start.x), h: Math.abs(p.y - drag.start.y),
      };
      drag.rect = r;
      Object.assign(drag.ghost.style, { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` });
    });
    inner.addEventListener("pointercancel", cancelGhost);
    inner.addEventListener("pointerup", async () => {
      if (!drag) return;
      const { rect } = drag;
      cancelGhost();
      if (rect.w < 6 || rect.h < 6) { toast("La zona es muy pequeña. Arrastra para dibujar un rectángulo."); return; }
      const s = scaleNow();
      const x0 = rect.x / s, y0 = rect.y / s, x1 = (rect.x + rect.w) / s, y1 = (rect.y + rect.h) / s;
      const r2 = (v) => Math.round(v * 100) / 100;
      const polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => [r2(x), r2(y)]);
      try {
        const f = await api(`/api/files/${enc(S.rv.id)}/findings`, { method: "POST", json: { page: S.rv.page, polygon } });
        setDraw(false);
        replaceFinding(f);
        select(f.id);
        toast("Zona agregada.");
        refreshState();
      } catch (err) {
        showError(err);
      }
    });
  }

  // --- keyboard ---
  function initKeyboard() {
    document.addEventListener("keydown", (e) => {
      if (S.screen !== 3 || anyDialogOpen() || !S.rv.file) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(tag)) return;
      const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      let handled = true;
      if (k === "j") move(1);
      else if (k === "k") move(-1);
      else if (k === "Delete" || k === "Backspace") { // an optional URL toggles (see askRemove)
        const item = document.activeElement && document.activeElement.closest ? document.activeElement.closest("#findlist .item") : null;
        askRemove((item && item.dataset.id) || S.rv.sel);
      }
      else if (k === "d") setDraw(!S.rv.draw);
      else if (k === "v") setView(!S.rv.after);
      else if (k === "r") rotate();
      else if (k === "+" || k === "=" || k === "Add") zoomBy(1.2);
      else if (k === "-" || k === "Subtract" || k === "−") zoomBy(1 / 1.2);
      else if (k === "Escape") {
        if (drag) cancelGhost();
        else if (S.rv.draw) setDraw(false);
        else handled = false; // Esc never hides the after column
      } else handled = false;
      if (handled) e.preventDefault();
    });
  }

  // ---------------------------------------------------------------------------------------------
  // Screen 4: export
  // ---------------------------------------------------------------------------------------------
  async function enterExport() {
    if (!S.exp.dest) {
      try {
        const res = await api("/api/default-export-dir");
        if (!S.exp.dest) S.exp.dest = (res && res.path) || "";
        $("#dest-path").value = S.exp.dest;
      } catch (err) {
        showError(err);
      }
    }
    renderExport();
  }

  function exportRow(f) {
    const c = f.counts || {};
    const saved = S.exp.results.get(f.id);
    const res = saved && saved._status === f.status ? saved : null;
    const processed = REVIEWABLE.has(f.status);
    const verif = h("td");
    const state = h("td");
    if (res) {
      const leaks = res.leaks || [];
      if (leaks.length) {
        verif.append(h("span", { class: "badtxt", text: plural(leaks.length, "fuga sin resolver", "fugas sin resolver") }));
        verif.append(h("ul", { class: "fmeta" }, leaks.map((l) => h("li", { text: (l.page != null ? `Pág. ${l.page + 1}: ` : "") + l.message }))));
      } else if (res.exported) {
        verif.append("Sin fugas");
      } else {
        verif.append("—");
      }
    } else if (f.status === "confirmed") {
      verif.append("Se verifica al exportar");
    } else {
      verif.append("—");
    }
    // The leak check only covers what was searched: say what was not (detection groups off).
    const off = processed ? groupsOff(f.options) : [];
    if (off.length) verif.append(h("div", { class: "warntxt", text: `No se buscaron: ${joinEs(off.map((g) => g.short))}. Revisa esas partes a mano.` }));
    if (res && res.exported) {
      state.append(h("span", { class: "state ok", text: "Exportado" }));
      if (res.output_path) {
        const base = res.output_path.split(/[\/]/).pop();
        state.append(h("div", { class: "fmeta", title: res.output_path }, "Guardado como ", h("span", { class: "mono", text: base })));
      }
    } else if (res && !res.exported) {
      state.append(h("span", { class: "state bad", text: "No se exporta" }));
      const why = res.message || "No se pudo exportar.";
      state.append(h("div", { class: "fmeta", text: /vuelve a revisar/i.test(why) ? why : `${why} Vuelve a Revisar.` }));
    } else if (f.status === "confirmed") {
      state.append(h("span", { class: "state ok", text: "Confirmado" }));
    } else if (f.status === "exported") {
      state.append(h("span", { class: "state ok", text: "Exportado" }));
      state.append(h("div", { class: "fmeta", text: "Si necesitas corregirlo, haz el cambio en Revisar y vuelve a confirmarlo."}));
    } else if (f.status === "ready") {
      state.append(h("span", { class: "state warn", text: "No se exporta" }));
      state.append(h("div", { class: "fmeta", text: "Falta confirmar la revisión. Vuelve a Revisar." }));
    } else if (f.status === "error") {
      state.append(h("span", { class: "state wait", text: "Omitido" }));
      state.append(h("div", { class: "fmeta", text: f.error_message || "No se pudo procesar." }));
    } else if (f.status === "cancelled") {
      state.append(h("span", { class: "state wait", text: "Omitido" }));
      state.append(h("div", { class: "fmeta", text: "Se canceló el procesamiento." }));
    } else {
      state.append(h("span", { class: "state wait", text: "No se exporta" }));
      state.append(h("div", { class: "fmeta", text: isActive(f) ? "Todavía se está procesando." : "Todavía no se procesa." }));
    }
    // counts.total is the number of active zones (removed ones are counted apart).
    const censored = processed ? res ? res.redactions_applied : c.total || 0 : "—";
    const removed = processed ? res ? res.removed_by_reviewer : c.removed || 0 : "—";
    return h("tr", null,
      h("td", { class: "fname" }, nameNode(f.name)),
      h("td", { class: "num", text: String(censored) }),
      h("td", { class: "num", text: String(removed) }),
      verif, state);
  }

  function renderExport() {
    const rows = $("#exp-rows");
    rows.replaceChildren(...(S.files.length
      ? S.files.map(exportRow)
      : [h("tr", null, h("td", { colspan: "5", class: "fmeta", text: "Todavía no hay archivos. Agrégalos en «Elegir archivos»." }))]));
    const confirmed = S.files.filter((f) => f.status === "confirmed");
    const btn = $("#b-export");
    btn.textContent = S.exp.busy
      ? "Exportando…"
      : confirmed.length ? `Exportar ${plural(confirmed.length, "archivo", "archivos")}` : "Exportar";
    btn.disabled = S.exp.busy || !confirmed.length || !S.exp.dest.trim();
    const pick = $("#b-dest");
    pick.hidden = !hasBridge("choose_folder");
    const input = $("#dest-path");
    if (document.activeElement !== input && input.value !== S.exp.dest) input.value = S.exp.dest;
  }

  async function chooseFolder() {
    if (!hasBridge("choose_folder")) { $("#dest-path").focus(); return; }
    try {
      let res = await bridge().choose_folder();
      if (Array.isArray(res)) res = res[0];
      if (res && typeof res === "object" && res.path) res = res.path;
      if (typeof res === "string" && res) {
        S.exp.dest = res;
        $("#dest-path").value = res;
        renderExport();
      }
    } catch {
      toast("No se pudo abrir el selector de carpetas. Escribe la ruta en el campo.", { bad: true });
      $("#dest-path").focus();
    }
  }

  async function doExport() {
    const ids = S.files.filter((f) => f.status === "confirmed").map((f) => f.id);
    const dest = S.exp.dest.trim();
    if (!ids.length || !dest) return;
    S.exp.busy = true;
    renderExport();
    const out = $("#exp-result");
    try {
      const res = await api("/api/export", {
        method: "POST",
        json: { dest_dir: dest, file_ids: ids, audit_pdf: $("#audit-pdf").checked, audit_json: $("#audit-json").checked },
      });
      const results = (res && res.results) || [];
      for (const r of results) S.exp.results.set(r.file_id, r);
      const ok = results.filter((r) => r.exported);
      const blocked = results.filter((r) => !r.exported);
      const nameOf = (id) => (fileById(id) || {}).name || id;
      const parts = [];
      parts.push(h("strong", {
        text: ok.length
          ? `Se ${ok.length === 1 ? "exportó" : "exportaron"} ${plural(ok.length, "archivo", "archivos")} en la carpeta de destino.`
          : "No se exportó ningún archivo.",
      }));
      if (blocked.length) {
        parts.push(h("p", {
          text: blocked.every((r) => (r.leaks || []).length)
            ? "Estos archivos no se exportaron porque la verificación encontró datos que siguen visibles:"
            : "Estos archivos no se exportaron:",
        }));
        parts.push(h("ul", null, blocked.map((r) => h("li", null, h("b", { text: nameOf(r.file_id) }), `: ${r.message || "tiene fugas sin resolver."}`))));
      }
      // Exported, but analyzed with detection groups off: the leak check did not cover those parts.
      const partial = ok.filter((r) => groupsOff((fileById(r.file_id) || {}).options).length);
      if (partial.length) {
        parts.push(h("p", { text: "En estos archivos no se buscó todo. Revisa esas partes a mano antes de publicarlos:" }));
        parts.push(h("ul", null, partial.map((r) => h("li", null, h("b", { text: nameOf(r.file_id) }),
          `: no se buscaron ${joinEs(groupsOff(fileById(r.file_id).options).map((g) => g.short))}.`))));
      }
      const audit = (res && res.audit) || {};
      if (audit.message) parts.push(h("p", null, h("b", { text: audit.message })));
      if (audit.pdf_path || audit.json_path) {
        parts.push(h("p", { text: "Informe de auditoría:" }));
        if (audit.pdf_path) parts.push(h("div", { class: "path", text: audit.pdf_path }));
        if (audit.json_path) parts.push(h("div", { class: "path", text: audit.json_path }));
      }
      parts.push(h("p", { class: "fmeta", text: "Revisa los archivos exportados antes de publicarlos: la decisión de publicar es tuya." }));
      out.className = `result${blocked.length ? " bad" : ""}`;
      out.replaceChildren(...parts);
      out.hidden = false;
    } catch (err) {
      out.className = "result bad";
      out.replaceChildren(h("strong", { text: err.message || MSG.generic }));
      out.hidden = false;
    } finally {
      S.exp.busy = false;
    }
    await refreshState();
    for (const r of S.exp.results.values()) {
      if (r._status === undefined) r._status = (fileById(r.file_id) || {}).status;
    }
    renderExport();
  }

  // ---------------------------------------------------------------------------------------------
  // "Acerca de": name, version, license and source (the legal notices of the AGPL)
  // ---------------------------------------------------------------------------------------------
  let aboutLoaded = false;
  async function openAbout() {
    if (!aboutLoaded) {
      let info;
      try {
        info = await api("/api/about");
      } catch (err) {
        showError(err);
        return;
      }
      $("#dlg-about-t").textContent = info.name;
      $("#about-version").textContent = `Versión ${info.version}`;
      $("#about-copy").textContent = `${info.copyright}. Licencia: ${info.license}.`;
      $("#about-url").value = info.source_url || "";
      $("#about-comps").replaceChildren(...(info.components || []).map((c) => h("tr", null,
        h("td", { text: c.name }), h("td", { class: "mono", text: c.license }), h("td", { text: c.use }))));
      const text = info.license_text || "";
      $("#about-license").textContent = text;
      $("#b-license").hidden = !text;
      const missing = $("#about-nolicense");
      missing.hidden = !!text;
      missing.textContent = text ? "" : "El texto completo de la licencia no viene en esta instalación: está en el código fuente.";
      aboutLoaded = true;
    }
    $("#about-copied").textContent = "";
    openDialog($("#dlg-about"));
    // The title takes the focus: the dialog opens at its top and screen readers start there.
    $("#dlg-about-t").focus();
  }
  function toggleLicense() {
    const box = $("#about-license");
    const show = box.hidden;
    box.hidden = !show;
    const btn = $("#b-license");
    btn.setAttribute("aria-expanded", String(show));
    btn.textContent = show ? "Ocultar licencia" : "Ver licencia completa";
    if (show) box.focus();
  }
  async function copySourceUrl() {
    const input = $("#about-url");
    const done = $("#about-copied");
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error("no clipboard");
      await navigator.clipboard.writeText(input.value);
      done.textContent = "Enlace copiado.";
    } catch {
      // Without clipboard access the link is left selected, ready for Ctrl+C.
      input.focus();
      input.select();
      done.textContent = "El enlace quedó seleccionado: cópialo con Ctrl+C.";
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------------------------------
  function init() {
    initTheme();
    initAskDialog();
    initDragDrop();
    initDrawing();
    initKeyboard();
    if (!TOKEN) setFatal(MSG.noToken);

    $$("[data-go]").forEach((b) => b.addEventListener("click", () => go(Number(b.dataset.go))));
    $("#b-about").addEventListener("click", openAbout);
    $("#dlg-about-close").addEventListener("click", () => $("#dlg-about").close());
    $("#b-copy-url").addEventListener("click", copySourceUrl);
    $("#b-license").addEventListener("click", toggleLicense);

    // Screen 1
    $("#b-choose").addEventListener("click", chooseFiles);
    $("#b-choose-dir").addEventListener("click", chooseFolderToAdd);
    for (const id of ["#in-files", "#in-dir"]) {
      $(id).addEventListener("change", (e) => {
        const files = [...e.target.files];
        e.target.value = "";
        uploadFiles(files);
      });
    }
    $("#b-remove-all").addEventListener("click", removeAll);
    $("#b-names").addEventListener("click", openNames);
    $("#names-text").addEventListener("input", updateNamesCount);
    $("#dlg-names-form").addEventListener("submit", saveNames);
    $("#dlg-names-no").addEventListener("click", () => $("#dlg-names").close());
    $("#b-process").addEventListener("click", () => {
      const mode = $("#b-process").dataset.mode;
      if (mode === "process") startProcessing(S.files.filter(isPending).map((f) => f.id));
      else if (mode === "view") go(2);
    });

    // Screen 2
    $("#b-cancel-all").addEventListener("click", () => cancel(null));
    $("#b-review-ready").addEventListener("click", () => {
      const first = S.files.find((f) => f.status === "ready") || S.files.find((f) => REVIEWABLE.has(f.status));
      openReview(first && first.id);
    });

    // Screen 3
    $("#rv-file").addEventListener("change", () => {
      // A closed select fires "change" on every arrow key: open only the file the reviewer stops on.
      clearTimeout(fileSwitchTimer);
      fileSwitchTimer = setTimeout(() => {
        fileSwitchTimer = null;
        const id = $("#rv-file").value;
        if (id === S.rv.id && (S.rv.file || S.rv.loadingId === id)) return; // already open
        openFile(id, { focus: true });
        if (S.rv.id !== id) renderFileBar(); // it could not be opened: show the open file again
      }, 300);
    });
    $("#rv-prev").addEventListener("click", () => { const f = neighbourFile(-1); if (f) openFile(f.id); });
    $("#rv-next").addEventListener("click", () => { const f = neighbourFile(1); if (f) openFile(f.id); });
    $("#b-prev").addEventListener("click", () => move(-1));
    $("#b-next").addEventListener("click", () => move(1));
    $("#b-zin").addEventListener("click", () => zoomBy(1.2));
    $("#b-zout").addEventListener("click", () => zoomBy(1 / 1.2));
    $("#b-zfit").addEventListener("click", zoomFit);
    $("#b-rot").addEventListener("click", rotate);
    $("#b-draw").addEventListener("click", () => setDraw(!S.rv.draw));
    $("#b-view").addEventListener("click", () => setView(!S.rv.after));
    $("#dlg-remove-form").addEventListener("submit", doRemove);
    $("#dlg-remove-no").addEventListener("click", () => $("#dlg-remove").close());
    $("#dlg-remove").addEventListener("close", () => { pendingRemove = null; });

    // Screen 4
    $("#dest-path").addEventListener("input", (e) => { S.exp.dest = e.target.value; renderExport(); });
    $("#b-dest").addEventListener("click", chooseFolder);
    $("#b-export").addEventListener("click", doExport);
    window.addEventListener("pywebviewready", () => { if (S.screen === 4) renderExport(); });

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { if (S.screen === 3 && S.rv.file) relayout(); }, 150);
    });
    // Zone labels use the UI font: re-layout once the bundled fonts are ready.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { if (S.screen === 3 && S.rv.file) renderZonesAll(); });

    go(1, { focus: false });
    loadGroups();
    refreshState().then(() => {
      if (!S.files.length) return;
      if (S.files.some(isActive)) go(2, { focus: false });
      else if (S.files.some((f) => REVIEWABLE.has(f.status))) go(3, { focus: false });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
