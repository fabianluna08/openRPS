// OpenRPS - interfaz gráfica: arranque y coordinación de vistas, paneles, menús y simulación.
import { api, decodeResult } from "./api.js";
import { renderLegend, renderMeasurements, renderProbe, renderStats } from "./charts.js";
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
  sample: "Clic donde mediste: escribe la potencia que leíste (dBm) · Esc: terminar",
  "plan-scale": "Clic en dos puntos del plano cuya distancia real conozcas · Esc: cancelar",
};
const MAX_PLAN_SIDE_PX = 4096;           // imágenes más grandes se reducen al cargarlas
const MAX_PLAN_CHARS = 30e6;             // el mismo límite que open_rps.project (data URL)

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
    if (tool === "plan-scale" && view2d && view2d.planScale) text = "Clic en el segundo punto · Esc: cancelar";
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
    const s = store.project.settings;
    const slow = s.mode !== "raytracing" ? [] : [
      s.max_reflections === 2 && "reflexiones dobles",
      s.diffraction && s.max_diffractions === 2 && "difracción doble",
      s.diffraction && s.mixed_interactions && "reflexión + difracción",
    ].filter(Boolean);
    $("busy-text").textContent = slow.length ? `Simulando (con ${slow.join(", ")}: puede tardar)…` : "Simulando…";
    const revision = store.revision;
    const simKey = store.simulationKey();
    const snapshot = store.apiProject();
    try {
      const payload = await api.simulate(snapshot);
      const result = decodeResult(payload);
      result.revision = revision;
      result.simKey = simKey;
      result.projectSnapshot = snapshot;
      store.result = result;
      invalidateLayer();
      store.emit("result");
      const rep = result.report;
      store.addLog("info", `Predicción lista en ${fmt(rep.elapsed_s, 2)} s: ${MODE_LABELS[rep.mode]}, ${result.grid.nx} × ${result.grid.ny} píxeles, `
        + `${rep.transmitters} transmisor(es), ${rep.reflectors} muro(s) en el motor (${rep.obstructing_reflectors} obstruyen)`
        + (rep.mode === "raytracing" ? `, ${rep.edges} arista(s) que difractan, ${rep.max_reflections === 2 ? "hasta dos rebotes" : "un rebote"}`
          + (rep.edges ? `, ${rep.max_diffractions === 2 ? "hasta dos difracciones" : "una difracción"} por rayo` : "")
          + (rep.mixed_interactions ? ", rayos de reflexión + difracción" : "")
          + ` (${rep.candidate_rays} rayos candidatos por punto`
          + (rep.active_rays_per_tx && rep.active_rays_per_tx.length
            ? `; tras descartar los imposibles, ${rep.active_rays_per_tx.join(" / ")} por transmisor` : "")
          + ")." : "."));
      for (const w of rep.warnings || []) store.addLog("warn", w);
      if (rep.warnings && rep.warnings.length) app.notify("warn", `${rep.warnings.length} aviso(s) del modelo: mira la pestaña Mensajes.`);
      if (store.project.measurements.length) await app.compareMeasurements({ quiet: true });
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
        const data = await api.probe(store.apiProject(), point);
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
    else if (sel.type === "measurement") c = [obj.x, obj.y];
    else if (sel.type === "floorplan") c = [obj.x_m + obj.width_m / 2, obj.y_m + obj.width_m * obj.pixel_height / obj.pixel_width / 2];
    if (!c) { view2d.fit(); return; }
    view2d.ox = view2d.width / 2 - c[0] * view2d.scale;
    view2d.oy = view2d.height / 2 + c[1] * view2d.scale;
    view2d.request();
  },

  // ----- plano de fondo
  pickFloorPlan() { $("image-input").click(); },

  async loadFloorPlanFile(file) {
    try {
      let dataUrl = await readAsDataUrl(file);
      let img = await loadImage(dataUrl);
      let note = "";
      const side = Math.max(img.naturalWidth, img.naturalHeight);
      if (side > MAX_PLAN_SIDE_PX) {
        const k = MAX_PLAN_SIDE_PX / side;
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(img.naturalWidth * k);
        canvas.height = Math.round(img.naturalHeight * k);
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        dataUrl = canvas.toDataURL(file.type === "image/jpeg" ? "image/jpeg" : "image/png", 0.92);
        img = await loadImage(dataUrl);
        note = ` (reducida de ${side} a ${MAX_PLAN_SIDE_PX} px en su lado mayor)`;
      }
      if (dataUrl.length > MAX_PLAN_CHARS) throw new Error(`la imagen ocupa ${(dataUrl.length / 1e6).toFixed(1)} MB; el máximo es ${MAX_PLAN_CHARS / 1e6} MB`);
      const token = store.internImage(dataUrl);
      const a = store.project.area;
      const old = store.project.floor_plan;
      store.change((p) => {
        p.floor_plan = old
          ? { ...old, image: token, pixel_width: img.naturalWidth, pixel_height: img.naturalHeight, name: file.name }
          : { image: token, pixel_width: img.naturalWidth, pixel_height: img.naturalHeight, x_m: a.x_min, y_m: a.y_min,
            width_m: round(a.x_max - a.x_min, 3), opacity: 0.7, visible: true, name: file.name };
      });
      store.select({ type: "floorplan" });
      store.addLog("info", `Plano de fondo: ${file.name}, ${img.naturalWidth} × ${img.naturalHeight} px${note}.`
        + (old ? "" : " Ahora escálalo con dos puntos de distancia conocida."));
      if (!old) app.startPlanScale();
    } catch (error) {
      app.notify("error", `No se pudo cargar la imagen ${file.name}: ${error.message}`, { modal: true });
    }
  },

  startPlanScale() {
    if (!store.project.floor_plan) { app.notify("warn", "Primero carga la imagen del plano."); return; }
    app.setTool("plan-scale");
  },

  finishPlanScale(a, b) {
    const mapDist = Math.hypot(b[0] - a[0], b[1] - a[1]);
    app.setTool("select");
    store.select({ type: "floorplan" });
    if (!(mapDist > 0)) { app.notify("warn", "Los dos puntos coinciden: vuelve a intentarlo."); return; }
    const input = el("input", { type: "number", min: "0", step: "any", value: +mapDist.toPrecision(4) });
    const ax = el("input", { type: "number", step: "any", value: round(a[0], 3) });
    const ay = el("input", { type: "number", step: "any", value: round(a[1], 3) });
    const apply = () => {
      const real = Number(input.value);
      if (!(real > 0)) { input.setCustomValidity("Escribe una distancia mayor que 0"); input.reportValidity(); return; }
      const tx = Number(ax.value);
      const ty = Number(ay.value);
      if (ax.value.trim() === "" || ay.value.trim() === "" || !isFinite(tx) || !isFinite(ty)) {
        ax.setCustomValidity("Escribe las coordenadas del primer punto"); ax.reportValidity(); return;
      }
      const k = real / mapDist;
      // Escala alrededor del primer punto y después lo lleva a las coordenadas indicadas
      store.change((p) => {
        const plan = p.floor_plan;
        plan.width_m = round(plan.width_m * k, 6);
        plan.x_m = round(a[0] - (a[0] - plan.x_m) * k + (tx - a[0]), 4);
        plan.y_m = round(a[1] - (a[1] - plan.y_m) * k + (ty - a[1]), 4);
      });
      $("modal").hidden = true;
      const plan = store.project.floor_plan;
      store.addLog("info", `Plano escalado: ${fmt(100 * plan.width_m / plan.pixel_width, 2)} cm por píxel, ${fmt(plan.width_m, 2)} m de ancho.`);
    };
    for (const field of [input, ax, ay]) field.addEventListener("keydown", (e) => { if (e.key === "Enter") apply(); });
    const unit = (field, text) => el("span", { class: "with-unit" }, field, el("span", { class: "unit", text }));
    showModal("Escala y posición del plano", [
      el("p", { text: `Los dos puntos están a ${fmt(mapDist, 2)} m en el mapa. ¿Cuál es su distancia real?` }),
      el("label", { class: "field" }, el("span", { text: "Distancia real" }), unit(input, "m")),
      el("p", { class: "note", text: "Coordenadas del primer punto en el proyecto (déjalas así para no moverlo; por ejemplo, 0 y 0 si marcaste la esquina que usas de origen):" }),
      el("label", { class: "field" }, el("span", { text: "X del primer punto" }), unit(ax, "m")),
      el("label", { class: "field" }, el("span", { text: "Y del primer punto" }), unit(ay, "m")),
      el("div", { class: "props-actions" }, el("button", { class: "btn primary", text: "Aplicar", onclick: apply })),
    ]);
    input.focus();
    input.select();
  },

  fitAreaToPlan() {
    const plan = store.project.floor_plan;
    if (!plan) return;
    const h = plan.width_m * plan.pixel_height / plan.pixel_width;
    const r = store.project.area.resolution_m;
    store.change((p) => {
      p.area.x_min = Math.floor(plan.x_m / r) * r;
      p.area.x_max = Math.ceil((plan.x_m + plan.width_m) / r) * r;
      p.area.y_min = Math.floor(plan.y_m / r) * r;
      p.area.y_max = Math.ceil((plan.y_m + h) / r) * r;
    });
    view2d.fit();
  },

  // ----- mediciones
  pickMeasurementsCsv() { $("csv-input").click(); },

  async importMeasurementsFile(file) {
    try {
      const { measurements } = await api.parseMeasurements(store.apiProject(), await file.text());
      const current = store.project.measurements.length;
      const put = (replace) => {
        store.change((p) => { p.measurements = replace ? measurements : p.measurements.concat(measurements); });
        $("modal").hidden = true;
        store.select({ type: "measurements" });
        store.addLog("info", `${measurements.length} medición(es) importadas de ${file.name}${replace || !current ? "" : ` (ahora hay ${store.project.measurements.length})`}.`);
        app.showMeasurements();
        if (store.result) app.compareMeasurements({ quiet: true });
      };
      if (!current) { put(true); return; }
      showModal("Importar mediciones", [
        el("p", { text: `${file.name} tiene ${measurements.length} medición(es) y el proyecto ya tiene ${current}.` }),
        el("div", { class: "props-actions" },
          el("button", { class: "btn primary", text: "Reemplazar", onclick: () => put(true) }),
          el("button", { class: "btn", text: "Añadir", onclick: () => put(false) })),
      ]);
    } catch (error) {
      app.notify("error", `No se pudieron leer las mediciones de ${file.name}: ${error.message}`, { modal: true });
    }
  },

  addMeasurementAt(point) {
    const p = store.project;
    const power = el("input", { type: "number", step: "any", placeholder: "−67", value: app.lastSample ? app.lastSample.rssi : "" });
    const tx = el("select", {}, el("option", { value: "", text: "El más fuerte (mejor servidor)" }),
      ...p.transmitters.map((t) => el("option", { value: String(t.tx_id), text: t.name || `Tx ${t.tx_id}` })));
    tx.value = app.lastSample ? app.lastSample.tx : "";
    const label = el("input", { type: "text", placeholder: "opcional" });
    const add = () => {
      const value = Number(power.value);
      if (power.value.trim() === "" || !isFinite(value)) { power.setCustomValidity("Escribe la potencia medida en dBm"); power.reportValidity(); return; }
      app.lastSample = { rssi: value, tx: tx.value };
      store.change((proj) => proj.measurements.push({ x: point[0], y: point[1], rssi_dbm: value, z: null,
        tx_id: tx.value === "" ? null : Number(tx.value), label: label.value.trim() }));
      $("modal").hidden = true;
      store.addLog("info", `Medición ${store.project.measurements.length}: ${fmt(value, 1)} dBm en x = ${fmt(point[0], 1)} m, y = ${fmt(point[1], 1)} m.`);
    };
    for (const input of [power, label]) input.addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });
    showModal("Nueva medición", [
      el("p", { class: "note", text: `En x = ${fmt(point[0], 2)} m, y = ${fmt(point[1], 2)} m, a la altura del receptor del área (${fmt(p.area.rx_height_m, 2)} m).` }),
      el("label", { class: "field" }, el("span", { text: "Potencia medida" }), el("span", { class: "with-unit" }, power, el("span", { class: "unit", text: "dBm" }))),
      el("label", { class: "field" }, el("span", { text: "Transmisor" }), tx),
      el("label", { class: "field" }, el("span", { text: "Etiqueta" }), label),
      el("div", { class: "props-actions" }, el("button", { class: "btn primary", text: "Agregar", onclick: add })),
    ]);
    power.focus();
    power.select();
  },

  showMeasurements() {
    const tab = document.querySelector('.tab[data-tab="measurements"]');
    if (tab) tab.click();
  },

  async compareMeasurements({ quiet = false } = {}) {
    if (!store.project.measurements.length) { if (!quiet) app.notify("warn", "No hay mediciones: importa un CSV o anota puntos con la herramienta Medición (N)."); return; }
    const revision = store.measurementRevision();
    try {
      const data = await api.compare(store.apiProject());
      if (revision !== store.measurementRevision()) return;
      store.comparison = { revision, data };
      store.emit("comparison");
      const st = data.stats;
      if (st.n) store.addLog("info", `Comparación con ${st.n} medición(es): sesgo ${st.bias_db >= 0 ? "+" : "−"}${fmt(Math.abs(st.bias_db), 1)} dB, desviación ${fmt(st.std_db, 1)} dB, RMSE ${fmt(st.rmse_db, 1)} dB.`);
      else store.addLog("warn", "Ninguna medición se pudo comparar (sin señal predicha, demasiado cerca de una antena o con un transmisor desactivado).");
    } catch (error) {
      app.notify("error", `No se pudo comparar con las mediciones: ${error.message}`, { modal: !quiet });
    }
  },

  async calibrate(options) {
    if (running) return;
    running = true;
    $("busy").hidden = false;
    $("busy-text").textContent = "Calibrando: cada paso vuelve a trazar los rayos en los puntos medidos…";
    const revision = store.measurementRevision();
    try {
      const data = await api.calibrate(store.apiProject(), options);
      store.calibration = { revision, data };
      app.scrollToCalibration = true;
      store.emit("comparison");
      store.addLog("info", `Calibración: RMSE ${fmt(data.before.stats.rmse_db, 1)} → ${fmt(data.after.stats.rmse_db, 1)} dB con ${data.after.stats.n} punto(s), ${data.iterations} iteraciones y ${data.evaluations} evaluaciones del modelo.`);
      for (const note of data.notes) store.addLog("warn", `Calibración: ${note}`);
    } catch (error) {
      app.notify("error", `No se pudo calibrar: ${error.message}`, { modal: true });
    } finally {
      running = false;
      $("busy").hidden = true;
    }
  },

  async applyCalibration() {
    const c = store.calibration;
    if (!c || c.revision !== store.measurementRevision()) { app.notify("warn", "El proyecto cambió después de calibrar: vuelve a calibrar."); return; }
    const ch = c.data.changes;
    store.change((p) => {
      for (const m of ch.materials) {
        const k = p.materials.findIndex((x) => x.id === m.id);
        if (k >= 0) p.materials[k] = m;
      }
      for (const t of ch.transmitters) {
        const tx = p.transmitters.find((x) => x.tx_id === t.tx_id);
        if (tx) tx.tx_power_dbm = t.tx_power_dbm;
      }
      Object.assign(p.settings, ch.settings);
    });
    store.calibration = null;
    store.addLog("info", `Calibración aplicada: ${ch.materials.length} material(es), ${ch.transmitters.length} transmisor(es)`
      + `${Object.keys(ch.settings).length ? " y el exponente" : ""}. Ctrl+Z la deshace.`);
    await app.run();
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

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("no se pudo leer el archivo"));
    reader.readAsDataURL(file);
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("el navegador no reconoce la imagen (usa PNG, JPEG, WebP o GIF)"));
    img.src = src;
  });
}

function saveProject() {
  const text = JSON.stringify(store.fullProject(), null, 2);
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
    ["N", "Anotar una medición"],
    ["F5 o Ctrl+Intro", "Ejecutar la predicción"], ["1 / 2 / 3", "Vista 2D / 3D / ambas"], ["F", "Ajustar la vista al área"],
    ["Supr", "Eliminar la selección"], ["Ctrl+D", "Duplicar la selección"], ["Ctrl+Z / Ctrl+Y", "Deshacer / rehacer"],
    ["Ctrl+S / Ctrl+O", "Guardar / abrir proyecto"], ["Alt al arrastrar", "Sin ajuste a la cuadrícula"], ["Esc", "Cancelar el dibujo"]];
  const table = el("table", {}, ...rows.map(([k, v]) => el("tr", {}, el("td", {}, el("kbd", { text: k })), el("td", { text: v }))));
  showModal("Atajos y uso", [
    el("p", { text: "Flujo de trabajo: define el área de predicción y la configuración, coloca transmisores, dibuja edificios (2.5D: planta poligonal y altura) y muros con su material, ejecuta la predicción y explora los mapas. Con la herramienta Receptor ves los rayos en 2D y 3D y la respuesta al impulso del canal en cualquier punto." }),
    table,
    el("p", { text: "Con mediciones: Archivo → Plano de fondo para cargar el plano y escalarlo con dos puntos; Archivo → Importar mediciones (CSV con x, y y la potencia en dBm) o la herramienta Medición (N); después, la pestaña Mediciones compara y calibra." }),
    el("p", { class: "note", text: "Los proyectos se guardan como archivos .orps.json (el mismo formato que open_rps.project.Project en Python), con el plano y las mediciones dentro. El último proyecto se guarda también en el navegador." }),
  ]);
}

function aboutModal() {
  const info = store.info || {};
  showModal("Acerca de OpenRPS", [
    el("p", { text: `OpenRPS ${info.version || ""}: simulador de propagación de ondas de radio de código abierto, para uso docente. Licencia GPL-2.0.` }),
    el("p", { text: "Motor: ray tracing 3D (rayo directo, reflexión en el suelo, uno o dos rebotes especulares en muros y suelo, difracción UTD en una o dos aristas por rayo, rayos que combinan un rebote con una difracción, y obstrucción por muros y techos) y modelos empíricos (FSPL, Two-Ray, Log-distancia, Okumura-Hata, COST-231 Hata)." }),
    el("p", { text: "Desde la 1.7.0 puedes cargar el plano de fondo, importar o anotar mediciones, compararlas con la predicción y calibrar las pérdidas de los materiales y la potencia de las antenas (pestaña Mediciones)." }),
    el("p", { class: "note", text: "Limitaciones: a lo más dos interacciones por rayo (dos rebotes, dos difracciones o un rebote y una difracción) y terreno plano. Los modelos no están validados contra campañas de medición: calíbralos con las tuyas. Consulta la sección «Limitaciones físicas» del README antes de usar los resultados." }),
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
  floorplan: () => app.pickFloorPlan(),
  "import-measurements": () => app.pickMeasurementsCsv(),
  compare: () => { app.showMeasurements(); app.compareMeasurements(); },
  "show-measurements": () => app.showMeasurements(),
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
  $("image-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (file) app.loadFloorPlanFile(file);
  });
  $("csv-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (file) app.importMeasurementsFile(file);
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
  const tools = { v: "select", h: "pan", t: "tx", r: "rect", p: "poly", w: "wall", s: "probe", m: "measure", n: "sample" };
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
    if (!live) renderMeasurementsTab();
  });
  store.on("selection", () => {
    renderTree($("tree"), store, app);
    renderProperties($("props"), $("props-title"), store, app);
    view2d.request();
    schedule3dRebuild();
  });
  store.on("result", () => {
    updateMetricControls();
    view2d.request();
    if (store.project && store.project.measurements.length) schedule3dRebuild();     // color de los marcadores
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
  store.on("image", () => { view2d.request(); view3d.updateGround(true); });
  store.on("comparison", () => { renderMeasurementsTab(); view2d.request(); renderProperties($("props"), $("props-title"), store, app); });
}

function renderMeasurementsTab() {
  renderMeasurements($("measurements-body"), store, app);
  const n = store.project ? store.project.measurements.length : 0;
  $("meas-count").hidden = n === 0;
  $("meas-count").textContent = String(n);
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
  renderMeasurementsTab();
  updateMetricControls();
  view2d.fit();
  view3d.rebuildScene();
  await app.run();
}

window.openrps = { store, app };          // para depurar desde la consola del navegador
start();
