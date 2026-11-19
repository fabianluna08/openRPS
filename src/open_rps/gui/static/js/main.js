// OpenRPS - interfaz gráfica: arranque y coordinación de vistas, paneles, menús y simulación.
import { api, decodeResult } from "./api.js";
import { renderLegend, renderProbe, renderStats } from "./charts.js";
import { METRICS, fmt } from "./colors.js";
import { currentLayer, invalidateLayer, metricOptions, pixelAt } from "./coverage.js";
import { centroid, round } from "./geometry.js";
import { MODE_LABELS, deleteSelection, duplicateSelection, el, renderProperties, renderTree } from "./panels.js";
import { Store, nextTxId } from "./state.js";
import { View2D } from "./view2d.js";
import { View3D } from "./view3d.js";

const $ = (id) => document.getElementById(id);
const store = new Store();
const SVG_METRICS = new Set(["rx_power", "best_server", "delay_spread", "sinr", "interference"]);
const TOOL_HINTS = {
  select: "Clic: seleccionar · arrastrar: mover · arrastra un vértice para editarlo · Supr: eliminar",
  pan: "Arrastra para desplazar la vista · rueda: zoom",
  tx: "Clic para colocar un transmisor",
  rect: "Arrastra para dibujar un edificio rectangular",
  poly: "Clic en cada vértice · doble clic, Intro o clic en el primer vértice para cerrar · Esc: cancelar",
  wall: "Clic en el inicio y en el final del muro · Esc: cancelar",
  probe: "Clic (o arrastra) para colocar el receptor de prueba: rayos y respuesta al impulso",
  measure: "Arrastra para medir una distancia y su azimut",
};

let view2d;
let view3d;
let running = false;
let probeSeq = 0;
let probeTimer = null;
let rebuild3dPending = false;

const app = {
  setTool(tool) {
    if (view2d) view2d.cancelDraft();
    store.setView({ tool });
    for (const b of document.querySelectorAll("[data-tool]")) b.classList.toggle("active", b.dataset.tool === tool);
    app.updateHint();
  },

  updateHint() {
    const tool = store.view.tool;
    let text = TOOL_HINTS[tool] || "";
    const d = view2d && view2d.draft;
    if (d && d.kind === "poly") text = `${d.points.length} vértice(s) · ${TOOL_HINTS.poly}`;
    if (d && d.kind === "wall") text = "Clic en el final del muro · Esc: cancelar";
    $("tool-hint").textContent = text;
  },

  setViewMode(mode) {
    store.setView({ mode });
    const main = $("viewports");
    main.className = `view-${mode}`;
    for (const b of document.querySelectorAll("[data-view]")) b.classList.toggle("active", b.dataset.view === mode);
    requestAnimationFrame(() => {
      view2d.resize();
      view2d.fit();
      view3d.setVisible(mode !== "2d");
    });
  },

  notify(level, text, { modal = false } = {}) {
    store.addLog(level, text);
    if (modal) showModal(level === "error" ? "No se pudo completar" : "Aviso", [el("p", { text })]);
    else if (level === "error" || level === "warn") {
      const hint = $("tool-hint");
      hint.textContent = text;
      clearTimeout(app.hintTimer);
      app.hintTimer = setTimeout(() => app.updateHint(), 4500);
    }
  },

  showCursor(w) {
    if (!w) { $("status-cursor").textContent = "x, y: —"; $("status-value").textContent = ""; return; }
    $("status-cursor").textContent = `x = ${fmt(w[0], 1)} m, y = ${fmt(w[1], 1)} m`;
    const layer = currentLayer(store);
    if (!layer) { $("status-value").textContent = ""; return; }
    const k = pixelAt(layer.grid, w[0], w[1]);
    if (k < 0) { $("status-value").textContent = ""; return; }
    const v = layer.values[k];
    $("status-value").textContent = layer.def.kind === "cat" ? `${layer.label}: ${v >= 0 ? store.txName(v) : "ninguno"}`
      : `${METRICS[layer.metric].short}: ${isFinite(v) ? `${fmt(v, layer.def.digits)} ${layer.def.unit}` : "sin datos"}`;
  },

  async run() {
    if (running || !store.project) return;
    running = true;
    $("run-button").disabled = true;
    $("busy").hidden = false;
    $("busy-text").textContent = "Simulando…";
    const revision = store.revision;
    const snapshot = JSON.parse(store.snapshot());
    try {
      const payload = await api.simulate(snapshot);
      const result = decodeResult(payload);
      result.revision = revision;
      result.projectSnapshot = snapshot;
      store.result = result;
      invalidateLayer();
      store.emit("result");
      const rep = result.report;
      store.addLog("info", `Predicción lista en ${fmt(rep.elapsed_s, 2)} s: ${MODE_LABELS[rep.mode]}, ${result.grid.nx} × ${result.grid.ny} píxeles, `
        + `${rep.transmitters} transmisor(es), ${rep.reflectors} muro(s) en el motor (${rep.obstructing_reflectors} obstruyen).`);
      for (const w of rep.warnings || []) store.addLog("warn", w);
      if (rep.warnings && rep.warnings.length) app.notify("warn", `${rep.warnings.length} aviso(s) del modelo: mira la pestaña Mensajes.`);
    } catch (error) {
      app.notify("error", error.message, { modal: true });
    } finally {
      running = false;
      $("run-button").disabled = false;
      $("busy").hidden = true;
    }
  },

  setProbe(point) {
    const z = store.project.area.rx_height_m;
    store.probe = { point: [point[0], point[1], z], data: store.probe && store.probe.data, loading: true };
    store.emit("probe");
    app.refreshProbe(0);
  },

  refreshProbe(delay = 120) {
    if (!store.probe) return;
    clearTimeout(probeTimer);
    probeTimer = setTimeout(async () => {
      const seq = ++probeSeq;
      const point = store.probe.point.slice(0, 2);
      try {
        const data = await api.probe(JSON.parse(store.snapshot()), point);
        if (seq !== probeSeq || !store.probe) return;
        store.probe = { point: data.point, data, loading: false };
      } catch (error) {
        if (seq !== probeSeq || !store.probe) return;
        store.probe = { point: store.probe.point, data: null, error: error.message };
      }
      store.view.hoverRay = null;
      store.emit("probe");
    }, delay);
  },

  hoverRay(hover) {
    store.view.hoverRay = hover;
    view2d.request();
    view3d.updateRays();
  },

  focusProperties() {
    const first = $("props").querySelector("input, select, textarea");
    if (first) first.focus();
  },

  refreshProperties() {
    renderProperties($("props"), $("props-title"), store, app);
  },

  zoomTo(sel) {
    if (sel) store.select(sel);
    const obj = store.selectedObject();
    if (!obj) return;
    let c = null;
    if (sel.type === "tx") c = [obj.x, obj.y];
    else if (sel.type === "building") c = centroid(obj.footprint);
    else if (sel.type === "wall") c = [(obj.start[0] + obj.end[0]) / 2, (obj.start[1] + obj.end[1]) / 2];
    if (!c) { view2d.fit(); return; }
    view2d.ox = view2d.width / 2 - c[0] * view2d.scale;
    view2d.oy = view2d.height / 2 + c[1] * view2d.scale;
    view2d.request();
  },

  addTransmitterCentered() {
    const a = store.project.area;
    view2d.addTransmitter([round((a.x_min + a.x_max) / 2, 1), round((a.y_min + a.y_max) / 2, 1)]);
  },

  fitAreaToObjects() {
    const p = store.project;
    const xs = [];
    const ys = [];
    for (const t of p.transmitters) { xs.push(t.x); ys.push(t.y); }
    for (const b of p.buildings) for (const [x, y] of b.footprint) { xs.push(x); ys.push(y); }
    for (const w of p.walls) { xs.push(w.start[0], w.end[0]); ys.push(w.start[1], w.end[1]); }
    if (!xs.length) return;
    const margin = Math.max(5, 0.05 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)));
    const r = p.area.resolution_m;
    store.change((proj) => {
      proj.area.x_min = Math.floor((Math.min(...xs) - margin) / r) * r;
      proj.area.x_max = Math.ceil((Math.max(...xs) + margin) / r) * r;
      proj.area.y_min = Math.floor((Math.min(...ys) - margin) / r) * r;
      proj.area.y_max = Math.ceil((Math.max(...ys) + margin) / r) * r;
    });
    view2d.fit();
  },
};

// ------------------------------------------------------------------ render de paneles
function renderAll() {
  renderTree($("tree"), store, app);
  renderProperties($("props"), $("props-title"), store, app);
  updateTitle();
  updateStatus();
  updateOutdated();
}

function updateTitle() {
  const p = store.project;
  $("project-title").textContent = p ? `${p.name}${store.modified ? " •" : ""}` : "";
  document.title = p ? `${p.name} — OpenRPS` : "OpenRPS";
}

function updateStatus() {
  const p = store.project;
  if (!p) return;
  const a = p.area;
  const nx = Math.floor((a.x_max - a.x_min) / a.resolution_m + 1e-9) + 1;
  const ny = Math.floor((a.y_max - a.y_min) / a.resolution_m + 1e-9) + 1;
  $("status-grid").textContent = `${MODE_LABELS[p.settings.mode]} · ${nx} × ${ny} píxeles de ${fmt(a.resolution_m, 2)} m · Rx a ${fmt(a.rx_height_m, 1)} m`;
  const r = store.result;
  $("status-run").textContent = r ? `Última predicción: ${fmt(r.report.elapsed_s, 2)} s` : "Sin predicción";
}

function updateOutdated() {
  $("outdated-badge").hidden = !store.outdated;
}

function updateMetricControls() {
  const select = $("metric-select");
  const options = metricOptions(store);
  select.replaceChildren(...options.map(([id, label]) => el("option", { value: id, text: label })));
  if (!options.some(([id]) => id === store.view.metric)) store.view.metric = "rx_power";
  select.value = store.view.metric;
  const filter = $("tx-filter");
  const r = store.result;
  const perTx = r && r.perTx && (store.view.metric === "rx_power" || store.view.metric === "path_loss");
  filter.replaceChildren(el("option", { value: "best", text: "Mejor servidor" }),
    ...(perTx ? r.tx_ids.map((id) => el("option", { value: String(id), text: store.txName(id) })) : []));
  if (!perTx || !r.tx_ids.includes(Number(store.view.txFilter))) store.view.txFilter = "best";
  filter.value = String(store.view.txFilter);
  filter.disabled = !perTx;
}

function refreshCoverage() {
  renderLegend($("legend2d"), store);
  renderLegend($("legend3d"), store);
  view2d.request();
  view3d.updateGround();
  renderStats($("stats-body"), store, app);
  if (!store.view.rangeAuto) return;
  const layer = currentLayer(store);
  if (layer && layer.range) { $("range-min").value = layer.range[0]; $("range-max").value = layer.range[1]; }
}

function schedule3dRebuild() {
  if (rebuild3dPending) return;
  rebuild3dPending = true;
  requestAnimationFrame(() => {
    rebuild3dPending = false;
    view3d.rebuildScene();
  });
}

function renderLog() {
  const body = $("log-body");
  body.replaceChildren(...store.log.slice().reverse().map((entry) => el("div", { class: `log-line ${entry.level}` },
    el("time", { text: entry.time.toLocaleTimeString("es-MX") }), document.createTextNode(entry.text))));
  const problems = store.log.filter((e) => e.level !== "info").length;
  $("log-count").hidden = problems === 0;
  $("log-count").textContent = String(problems);
}

// ------------------------------------------------------------------ modal, archivos
function showModal(title, children) {
  $("modal-title").textContent = title;
  $("modal-body").replaceChildren(...children);
  $("modal").hidden = false;
  $("modal-close").focus();
}

function download(name, href) {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
}

function saveProject() {
  const text = JSON.stringify(store.project, null, 2);
  const slug = (store.project.name || "proyecto").normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^\w-]+/g, "_").replace(/^_+|_+$/g, "") || "proyecto";
  const url = URL.createObjectURL(new Blob([text + "\n"], { type: "application/json" }));
  download(`${slug}.orps.json`, url);
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  store.savedRevision = store.revision;
  updateTitle();
  store.addLog("info", `Proyecto guardado como ${slug}.orps.json (en tu carpeta de descargas).`);
}

async function loadProject(project, message) {
  try {
    const { project: clean } = await api.validate(project);
    store.setProject(clean);
    view2d.needsFit = true;
    view2d.fit();
    view3d.needsFit = true;
    view3d.fitCamera();
    if (message) store.addLog("info", message);
    await app.run();
  } catch (error) {
    app.notify("error", `No se pudo abrir el proyecto: ${error.message}`, { modal: true });
  }
}

function confirmDiscard() {
  return !store.modified || window.confirm("Hay cambios sin guardar en el proyecto actual. ¿Descartarlos?");
}

async function newProject() {
  if (!confirmDiscard()) return;
  await loadProject({ name: "Proyecto nuevo", area: { x_min: 0, x_max: 300, y_min: 0, y_max: 200, resolution_m: 2, rx_height_m: 1.5 },
    transmitters: [{ tx_id: 0, name: "Tx 0", x: 60, y: 100, z: 20, tx_power_dbm: 40, frequency_hz: 2.4e9, antenna_gain_dbi: 10 }] },
  "Proyecto nuevo.");
}

function exportResult(what) {
  const r = store.result;
  if (!r) { app.notify("warn", "Primero ejecuta la predicción (F5)."); return; }
  if (store.outdated) app.notify("warn", "Se exporta la última predicción: el proyecto cambió después.");
  let target = what;
  if (what === "svg") {
    const metric = SVG_METRICS.has(store.view.metric) ? store.view.metric : "rx_power";
    if (metric !== store.view.metric) app.notify("warn", `El SVG no tiene "${METRICS[store.view.metric].label}": se exporta la potencia recibida.`);
    target = `svg-${metric}`;
  }
  download("", api.exportUrl(r.result_id, target));
}

function exportPng() {
  const use3d = store.view.mode === "3d";
  const url = use3d ? view3d.toPng() : view2d.toPng();
  if (url) download(`openrps_${use3d ? "3d" : "2d"}.png`, url);
}

function helpModal() {
  const rows = [["V", "Seleccionar / mover"], ["H o espacio", "Desplazar la vista"], ["T", "Transmisor"], ["R", "Edificio rectangular"],
    ["P", "Edificio poligonal (doble clic o Intro para cerrar)"], ["W", "Muro"], ["S", "Receptor de prueba"], ["M", "Medir"],
    ["F5 o Ctrl+Intro", "Ejecutar la predicción"], ["1 / 2 / 3", "Vista 2D / 3D / ambas"], ["F", "Ajustar la vista al área"],
    ["Supr", "Eliminar la selección"], ["Ctrl+D", "Duplicar la selección"], ["Ctrl+Z / Ctrl+Y", "Deshacer / rehacer"],
    ["Ctrl+S / Ctrl+O", "Guardar / abrir proyecto"], ["Alt al arrastrar", "Sin ajuste a la cuadrícula"], ["Esc", "Cancelar el dibujo"]];
  const table = el("table", {}, ...rows.map(([k, v]) => el("tr", {}, el("td", {}, el("kbd", { text: k })), el("td", { text: v }))));
  showModal("Atajos y uso", [
    el("p", { text: "Flujo de trabajo: define el área de predicción y la configuración, coloca transmisores, dibuja edificios (2.5D: planta poligonal y altura) y muros con su material, ejecuta la predicción y explora los mapas. Con la herramienta Receptor ves los rayos en 2D y 3D y la respuesta al impulso del canal en cualquier punto." }),
    table,
    el("p", { class: "note", text: "Los proyectos se guardan como archivos .orps.json (el mismo formato que open_rps.project.Project en Python). El último proyecto se guarda también en el navegador." }),
  ]);
}

function aboutModal() {
  const info = store.info || {};
  showModal("Acerca de OpenRPS", [
    el("p", { text: `OpenRPS ${info.version || ""}: simulador de propagación de ondas de radio de código abierto, para uso docente. Licencia GPL-2.0.` }),
    el("p", { text: "Motor: ray tracing 3D (rayo directo, reflexión en el suelo y un rebote especular por muro, con obstrucción por muros y techos) y modelos empíricos (FSPL, Two-Ray, Log-distancia, Okumura-Hata, COST-231 Hata)." }),
    el("p", { class: "note", text: "Limitaciones: no hay difracción ni rebotes múltiples, el terreno es plano y no hay validación experimental. Consulta la sección «Limitaciones físicas» del README antes de usar los resultados." }),
    el("p", { class: "note", text: "La vista 3D usa three.js r187 (licencia MIT), incluido en el paquete." }),
  ]);
}

// ------------------------------------------------------------------ acciones y teclado
const actions = {
  new: newProject,
  open: () => { if (confirmDiscard()) $("file-input").click(); },
  save: saveProject,
  undo: () => store.undo(),
  redo: () => store.redo(),
  duplicate: () => duplicateSelection(store, app),
  delete: () => { if (!deleteSelection(store) && store.selection && store.selection.type === "material") app.notify("error", "Ese material está en uso."); },
  "view-2d": () => app.setViewMode("2d"),
  "view-3d": () => app.setViewMode("3d"),
  "view-split": () => app.setViewMode("split"),
  fit: () => { view2d.fit(); view3d.fitCamera(); },
  run: () => app.run(),
  "select-area": () => store.select({ type: "area" }),
  "select-settings": () => store.select({ type: "settings" }),
  "export-csv": () => exportResult("csv"),
  "export-json": () => exportResult("json"),
  "export-svg": () => exportResult("svg"),
  "export-png": exportPng,
  help: helpModal,
  about: aboutModal,
  "cam-top": () => view3d.topView(),
  "cam-persp": () => view3d.fitCamera(),
  "cam-probe": () => (store.probe ? view3d.focusProbe() : app.notify("warn", "Coloca primero un receptor (S).")),
};

function bindUi() {
  document.addEventListener("click", (e) => {
    const button = e.target.closest("[data-action]");
    if (button) {
      const action = actions[button.dataset.action];
      if (action) action();
      for (const m of document.querySelectorAll(".menu.open")) m.classList.remove("open");
      return;
    }
    const menuButton = e.target.closest(".menu-button");
    const menus = document.querySelectorAll(".menu");
    if (menuButton) {
      const menu = menuButton.parentElement;
      const open = menu.classList.contains("open");
      for (const m of menus) m.classList.remove("open");
      if (!open) menu.classList.add("open");
    } else if (!e.target.closest(".menu-items")) {
      for (const m of menus) m.classList.remove("open");
    }
  });
  for (const button of document.querySelectorAll(".menu-button")) {
    button.addEventListener("mouseenter", () => {
      if (!document.querySelector(".menu.open")) return;
      for (const m of document.querySelectorAll(".menu")) m.classList.remove("open");
      button.parentElement.classList.add("open");
    });
  }
  for (const b of document.querySelectorAll("[data-tool]")) b.addEventListener("click", () => app.setTool(b.dataset.tool));

  $("metric-select").addEventListener("change", (e) => { store.setView({ metric: e.target.value }); updateMetricControls(); refreshCoverage(); });
  $("tx-filter").addEventListener("change", (e) => { store.setView({ txFilter: e.target.value }); refreshCoverage(); });
  $("opacity").addEventListener("input", (e) => { store.setView({ opacity: Number(e.target.value) }); view2d.request(); view3d.updateGround(); });
  $("range-auto").addEventListener("change", (e) => {
    const auto = e.target.checked;
    $("range-min").disabled = auto;
    $("range-max").disabled = auto;
    store.setView({ rangeAuto: auto, rangeMin: Number($("range-min").value), rangeMax: Number($("range-max").value) });
    refreshCoverage();
  });
  for (const id of ["range-min", "range-max"]) {
    $(id).addEventListener("change", () => {
      const lo = Number($("range-min").value);
      const hi = Number($("range-max").value);
      if (!(hi > lo)) { app.notify("error", "El máximo de la escala debe ser mayor que el mínimo."); return; }
      store.setView({ rangeMin: lo, rangeMax: hi });
      refreshCoverage();
    });
  }
  const options = { "opt-grid": "grid", "opt-snap": "snap", "opt-smooth": "smooth", "opt-labels": "labels", "opt-relief": "relief",
    "opt-transparent": "transparent", "opt-blocked": "showBlocked" };
  for (const [id, key] of Object.entries(options)) {
    $(id).checked = !!store.view[key];
    $(id).addEventListener("change", (e) => {
      store.setView({ [key]: e.target.checked });
      view2d.request();
      if (key === "relief") view3d.rebuildScene();
      else if (key === "smooth") view3d.updateGround(true);
      else if (key === "transparent" || key === "labels") view3d.rebuildScene();
      else if (key === "showBlocked") view3d.updateRays();
    });
  }
  $("file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    try {
      const project = JSON.parse(await file.text());
      await loadProject(project, `Proyecto abierto: ${file.name}`);
    } catch (error) {
      app.notify("error", `No se pudo leer ${file.name}: ${error.message}`, { modal: true });
    }
  });
  $("modal-close").addEventListener("click", () => { $("modal").hidden = true; });
  $("modal").addEventListener("click", (e) => { if (e.target === $("modal")) $("modal").hidden = true; });

  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t === tab);
      for (const body of document.querySelectorAll(".tab-body")) body.hidden = body.dataset.body !== tab.dataset.tab;
      $("app").classList.remove("bottom-collapsed");
      $("bottom-toggle").textContent = "▾";
      onLayoutChange();
    });
  }
  $("bottom-toggle").addEventListener("click", () => {
    const collapsed = $("app").classList.toggle("bottom-collapsed");
    $("bottom-toggle").textContent = collapsed ? "▴" : "▾";
    onLayoutChange();
  });

  const handle = $("bottom-resize");
  handle.addEventListener("pointerdown", (e) => {
    handle.setPointerCapture(e.pointerId);
    const startY = e.clientY;
    const startH = $("bottom-panel").getBoundingClientRect().height;
    const move = (ev) => {
      const h = Math.max(120, Math.min(window.innerHeight * 0.7, startH + (startY - ev.clientY)));
      document.documentElement.style.setProperty("--bottom-h", `${Math.round(h)}px`);
      $("app").classList.remove("bottom-collapsed");
      onLayoutChange();
    };
    const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });
  window.addEventListener("keydown", onKey);
  window.addEventListener("beforeunload", (e) => { if (store.modified) { e.preventDefault(); e.returnValue = ""; } });
}

function onLayoutChange() {
  requestAnimationFrame(() => { view2d.resize(); view3d.resize(); });
}

function onKey(e) {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName);
  const ctrl = e.ctrlKey || e.metaKey;
  if (e.key === "Escape") {
    if (!$("modal").hidden) { $("modal").hidden = true; return; }
    if (typing) { e.target.blur(); return; }
    view2d.cancelDraft();
    if (store.view.tool !== "select") app.setTool("select");
    else store.select(null);
    return;
  }
  if (e.key === "F5" || (ctrl && e.key === "Enter")) { e.preventDefault(); app.run(); return; }
  if (ctrl && e.key.toLowerCase() === "s") { e.preventDefault(); saveProject(); return; }
  if (ctrl && e.key.toLowerCase() === "o") { e.preventDefault(); actions.open(); return; }
  if (typing) return;
  if (ctrl && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? store.redo() : store.undo(); return; }
  if (ctrl && e.key.toLowerCase() === "y") { e.preventDefault(); store.redo(); return; }
  if (ctrl && e.key.toLowerCase() === "d") { e.preventDefault(); duplicateSelection(store, app); return; }
  if (ctrl || e.altKey) return;
  if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); actions.delete(); return; }
  if (e.key === "Enter" && view2d.draft && view2d.draft.kind === "poly") { view2d.finishPolygon(); return; }
  const tools = { v: "select", h: "pan", t: "tx", r: "rect", p: "poly", w: "wall", s: "probe", m: "measure" };
  const key = e.key.toLowerCase();
  if (tools[key]) { app.setTool(tools[key]); return; }
  if (key === "f") { actions.fit(); return; }
  if (key === "1") app.setViewMode("2d");
  if (key === "2") app.setViewMode("3d");
  if (key === "3") app.setViewMode("split");
}

// ------------------------------------------------------------------ eventos del estado
function bindStore() {
  store.on("project", ({ live, reset } = {}) => {
    renderTree($("tree"), store, app);
    if (!live) renderProperties($("props"), $("props-title"), store, app);
    else updateLiveFields();
    updateTitle();
    updateStatus();
    updateOutdated();
    view2d.request();
    if (reset) { view3d.rebuildScene(); } else schedule3dRebuild();
    if (store.result) { renderLegend($("legend2d"), store); renderLegend($("legend3d"), store); view3d.updateGround(); }
    if (store.probe && !reset) app.refreshProbe(live ? 60 : 0);
  });
  store.on("selection", () => {
    renderTree($("tree"), store, app);
    renderProperties($("props"), $("props-title"), store, app);
    view2d.request();
    schedule3dRebuild();
  });
  store.on("result", () => {
    updateMetricControls();
    refreshCoverage();
    updateStatus();
    updateOutdated();
  });
  store.on("probe", () => {
    renderProbe($("probe-body"), store, app);
    view2d.request();
    view3d.updateRays();
  });
  store.on("log", renderLog);
}

// Durante un arrastre solo se actualizan los números del formulario, sin reconstruirlo (conserva el foco)
function updateLiveFields() {
  const s = store.selection;
  const obj = store.selectedObject();
  if (!s || !obj) return;
  if (s.type === "tx") {
    const inputs = $("props").querySelectorAll("input[type=number]");
    if (inputs[0]) inputs[0].value = obj.x;
    if (inputs[1]) inputs[1].value = obj.y;
  } else renderProperties($("props"), $("props-title"), store, app);
}

// ------------------------------------------------------------------ arranque
async function start() {
  view2d = new View2D($("canvas2d"), store, app);
  view3d = new View3D($("canvas3d"), store, app);
  Object.assign(window.openrps, { view2d, view3d });
  bindUi();
  bindStore();
  app.setTool("select");
  app.setViewMode("2d");
  try {
    store.info = await api.info();
  } catch (error) {
    showModal("Sin conexión", [el("p", { text: error.message })]);
    return;
  }
  $("version").textContent = store.info.version;
  $("examples-menu").replaceChildren(...store.info.examples.map((ex) => el("button", { role: "menuitem", text: ex.name,
    onclick: async () => {
      for (const m of document.querySelectorAll(".menu.open")) m.classList.remove("open");
      if (!confirmDiscard()) return;
      const { project } = await api.example(ex.id);
      await loadProject(project, `Ejemplo cargado: ${ex.name}`);
    } })));
  const initial = await api.initialProject();
  const restored = initial.path ? null : Store.restore();
  if (restored) {
    try {
      const { project } = await api.validate(restored);
      store.setProject(project);
      store.addLog("info", "Se recuperó el último proyecto guardado en este navegador (Archivo → Ejemplos para empezar de nuevo).");
    } catch (error) {
      store.setProject(initial.project);
    }
  } else {
    store.setProject(initial.project);
    store.addLog("info", initial.path ? `Proyecto abierto: ${initial.path}` : "Ejemplo cargado: microcelda urbana.");
  }
  renderAll();
  renderProbe($("probe-body"), store, app);
  renderStats($("stats-body"), store, app);
  updateMetricControls();
  view2d.fit();
  view3d.rebuildScene();
  await app.run();
}

window.openrps = { store, app };          // para depurar desde la consola del navegador
start();
