// Leyenda del mapa, panel del receptor (perfil potencia-retardo y tabla de rayos) y estadísticas.
import { AXIS, INK2, MUTED, SURFACE, fmt, niceTicks, num, paletteRamp, txColor } from "./colors.js";
import { currentLayer } from "./coverage.js";
import { el } from "./panels.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

// ------------------------------------------------------------------ leyenda
export function renderLegend(container, store) {
  container.replaceChildren();
  const layer = currentLayer(store);
  if (!layer) return;
  container.append(el("div", { class: "legend-title", text: layer.def.unit ? `${layer.label} (${layer.def.unit})` : layer.label }));
  if (layer.def.kind === "cat") {
    const ids = store.result.tx_ids;
    ids.forEach((id) => {
      container.append(el("div", { class: "legend-row" }, el("span", { class: "legend-swatch", style: `background:${txColor(store.txSlot(id))}` }),
        el("span", { text: store.txName(id) })));
    });
    container.append(el("div", { class: "legend-row" }, el("span", { class: "legend-swatch", style: "background:#e1e0d9" }),
      el("span", { class: "muted", text: "Sin servidor / sin datos" })));
  } else {
    const [lo, hi] = layer.range;
    const width = 200;
    const height = 40;
    const pad = 16;
    const bar = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `Escala de ${fmt(lo, 1)} a ${fmt(hi, 1)}` });
    const defs = svg("defs");
    const gradId = `grad-${container.id || "legend"}`;
    const grad = svg("linearGradient", { id: gradId, x1: "0", x2: "1", y1: "0", y2: "0" });
    const ramp = paletteRamp(store.view);
    if (store.view.banded) {
      ramp.forEach((c, k) => {
        grad.append(svg("stop", { offset: `${(k / ramp.length) * 100}%`, "stop-color": c }),
          svg("stop", { offset: `${((k + 1) / ramp.length) * 100}%`, "stop-color": c }));
      });
    } else {
      ramp.forEach((c, k) => grad.append(svg("stop", { offset: `${(k / (ramp.length - 1)) * 100}%`, "stop-color": c })));
    }
    defs.append(grad);
    bar.append(defs, svg("rect", { x: pad, y: 4, width: width - 2 * pad, height: 12, rx: 2, fill: `url(#${gradId})` }));
    const integer = Boolean(layer.def.integer);
    let ticks = niceTicks(lo, hi, 4);
    if (integer) {
      ticks = ticks.filter((t) => Number.isInteger(t));
      if (ticks.length < 2) ticks = [...new Set([Math.ceil(lo), Math.floor(hi)])].filter((t) => t >= lo && t <= hi);
    }
    for (const t of ticks) {
      const x = pad + ((t - lo) / (hi - lo || 1)) * (width - 2 * pad);
      bar.append(svg("line", { x1: x, x2: x, y1: 16, y2: 20, stroke: INK2, "stroke-width": 1 }));
      const label = svg("text", { x, y: 32, "text-anchor": "middle", "font-size": 11, fill: INK2 });
      label.textContent = fmt(t, integer || Math.abs(hi - lo) >= 5 ? 0 : 1);
      bar.append(label);
    }
    container.append(bar);
    if (layer.hideBelow !== null) {
      const unit = layer.def.unit ? ` ${layer.def.unit}` : "";
      container.append(el("div", { class: "legend-row" }, el("span", { class: "legend-swatch hollow" }),
        el("span", { class: "muted", text: `Sin pintar: por debajo de ${fmt(layer.hideBelow, layer.def.digits)}${unit} o sin señal` })));
    } else {
      container.append(el("div", { class: "legend-row" }, el("span", { class: "legend-swatch", style: "background:#e1e0d9" }),
        el("span", { class: "muted", text: layer.def.noDataLegend || "Sin datos (sin señal o interior bloqueado)" })));
    }
  }
  if (layer.mask) container.append(el("div", { class: "legend-row muted", text: "Solo dentro de los muros" }));
  if (layer.maskNote) container.append(el("div", { class: "legend-row warn-note", text: layer.maskNote }));
  if (store.outdated) container.append(el("div", { class: "muted", text: "El proyecto cambió: vuelve a ejecutar (F5)." }));
}

// ------------------------------------------------------------------ receptor
export function renderProbe(container, store, app) {
  container.replaceChildren();
  const probe = store.probe;
  if (!probe) {
    container.append(el("div", { class: "empty-state", text: "Elige la herramienta Receptor (S) y haz clic en el mapa (2D o 3D) para ver los rayos, la potencia y la respuesta al impulso del canal en ese punto. Puedes arrastrarlo: se recalcula al vuelo." }));
    return;
  }
  if (probe.error) {
    container.append(el("div", { class: "empty-state", text: `No se pudo calcular el receptor: ${probe.error}` }));
    return;
  }
  if (!probe.data) {
    container.append(el("div", { class: "empty-state", text: "Calculando…" }));
    return;
  }
  const data = probe.data;
  const [x, y, z] = data.point;
  const layout = el("div", { class: "probe-layout" });
  const left = el("div");
  const right = el("div");
  layout.append(left, right);
  container.append(layout);

  left.append(el("div", { class: "note", text: `Receptor en x = ${fmt(x, 1)} m, y = ${fmt(y, 1)} m, z = ${fmt(z, 2)} m · sin sombra log-normal` }));
  const summary = el("table", { class: "data" });
  summary.append(el("tr", {}, ...["Transmisor", "P_r (dBm)", "PL (dB)", "τ̄ exceso (ns)", "σ_τ (ns)", "σ AoD (°)", "Rayos"].map((h, k) =>
    el("th", { class: k ? "num" : "", text: h }))));
  let best = null;
  data.transmitters.forEach((t) => {
    const p = num(t.rx_power_dbm);
    if (isFinite(p) && (best === null || p > best.p)) best = { p, id: t.tx_id };
  });
  for (const t of data.transmitters) {
    const live = t.rays.filter((r) => !r.blocked).length;
    const name = el("td", {}, el("span", { class: "key-line", style: `background:${txColor(store.txSlot(t.tx_id))}` }),
      document.createTextNode(store.txName(t.tx_id) + (best && best.id === t.tx_id ? " ★" : "")));
    summary.append(el("tr", {}, name,
      el("td", { class: "num", text: t.too_close ? "cerca" : fmt(num(t.rx_power_dbm), 1) }),
      el("td", { class: "num", text: fmt(num(t.path_loss_db), 1) }),
      el("td", { class: "num", text: fmt(num(t.mean_excess_delay_ns), 1) }),
      el("td", { class: "num", text: fmt(num(t.rms_delay_spread_ns), 1) }),
      el("td", { class: "num", text: fmt(num(t.aod_spread_deg), 1) }),
      el("td", { class: "num", text: data.mode === "raytracing" ? `${live}/${t.rays.length}` : "—" })));
  }
  left.append(summary);
  if (best) left.append(el("div", { class: "note", text: `★ Mejor servidor: ${store.txName(best.id)}. τ̄ exceso: retardo medio respecto al primer rayo. Las métricas de retardo y ángulo ponderan cada rayo por su potencia.` }));
  if (data.wifi && data.wifi.serving_tx_id !== null && data.wifi.serving_tx_id !== undefined) left.append(probeWifi(store, data.wifi));
  if (data.mode !== "raytracing") {
    if (data.mode === "multiwall") {
      const rows = data.transmitters.filter((t) => t.walls).map((t) => el("tr", {},
        el("td", { text: store.txName(t.tx_id) }),
        el("td", { class: "num", text: String(t.walls.length) }),
        el("td", { class: "num", text: fmt(t.wall_loss_db, 1) }),
        el("td", { text: t.walls.map((w) => `${w.name} (${fmt(w.loss_db, 1)} dB)`).join(" · ") || "línea de vista" })));
      left.append(el("table", { class: "data" },
        el("tr", {}, ...["Transmisor", "Muros", "Pérdida (dB)", "Muros que cruza la línea directa"].map((h) => el("th", { text: h }))), ...rows));
    }
    left.append(el("p", { class: "note", text: "En los modelos empíricos no hay rayos ni respuesta al impulso: solo la potencia que predice el modelo. Cambia a ray tracing en la Configuración para verlos." }));
    return;
  }

  // Tabla de rayos (también es la vista en tabla del gráfico)
  const rays = [];
  data.transmitters.forEach((t, ti) => t.rays.forEach((r, ri) => rays.push({ t, ti, r, ri })));
  rays.sort((a, b) => a.r.tau_ns - b.r.tau_ns);
  const table = el("table", { class: "data" });
  table.append(el("tr", {}, ...["Rayo", "Tx", "P (dBm)", "τ (ns)", "AoD φ/θ (°)", "AoA φ/θ (°)", "Muros (dB)"].map((h, k) =>
    el("th", { class: k > 1 ? "num" : "", text: h }))));
  const rows = [];
  for (const item of rays) {
    const { t, ti, r, ri } = item;
    const tr = el("tr", { class: r.blocked ? "blocked" : "" },
      el("td", { text: r.label + (r.blocked ? " (bloqueado)" : "") }),
      el("td", {}, el("span", { class: "key-line", style: `background:${txColor(store.txSlot(t.tx_id))}` }), document.createTextNode(store.txName(t.tx_id))),
      el("td", { class: "num", text: fmt(num(r.power_dbm), 1) }),
      el("td", { class: "num", text: fmt(r.tau_ns, 1) }),
      el("td", { class: "num", text: `${fmt(r.aod_phi_deg, 0)} / ${fmt(r.aod_theta_deg, 0)}` }),
      el("td", { class: "num", text: `${fmt(r.aoa_phi_deg, 0)} / ${fmt(r.aoa_theta_deg, 0)}` }),
      el("td", { class: "num", text: fmt(num(r.obstruction_loss_db), 1) }));
    tr.addEventListener("mouseenter", () => app.hoverRay({ tx: ti, ray: ri }));
    tr.addEventListener("mouseleave", () => app.hoverRay(null));
    item.row = tr;
    rows.push(tr);
    table.append(tr);
  }
  left.append(el("div", { class: "props-section", text: "Rayos (ordenados por retardo)" }), table);
  right.append(powerDelayProfile(store, app, data, rays));
}

function powerDelayProfile(store, app, data, rays) {
  const box = el("div", { class: "chart-box" });
  const live = rays.filter((x) => !x.r.blocked && isFinite(num(x.r.power_dbm)));
  box.append(el("div", { class: "props-section", text: "Perfil potencia–retardo (respuesta al impulso)" }));
  if (!live.length) {
    box.append(el("div", { class: "empty-state", text: "Ningún rayo llega a este punto (todos están bloqueados)." }));
    return box;
  }
  const W = 560;
  const H = 180;
  const m = { l: 46, r: 14, t: 10, b: 34 };
  const taus = live.map((x) => x.r.tau_ns);
  const pows = live.map((x) => num(x.r.power_dbm));
  let t0 = Math.min(...taus);
  let t1 = Math.max(...taus);
  const pad = Math.max(1, (t1 - t0) * 0.06);
  t0 -= pad;
  t1 += pad;
  const pTop = Math.ceil(Math.max(...pows) / 10) * 10;
  const pBottom = Math.min(Math.floor(Math.min(...pows) / 10) * 10 - 10, pTop - 20);
  const sx = (t) => m.l + ((t - t0) / (t1 - t0)) * (W - m.l - m.r);
  const sy = (p) => m.t + ((pTop - p) / (pTop - pBottom)) * (H - m.t - m.b);
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Perfil potencia-retardo: un trazo por rayo" });
  for (const p of niceTicks(pBottom, pTop, 4)) {
    chart.append(svg("line", { x1: m.l, x2: W - m.r, y1: sy(p), y2: sy(p), stroke: AXIS, "stroke-width": 1, "stroke-opacity": 0.6 }));
    const label = svg("text", { x: m.l - 6, y: sy(p) + 4, "text-anchor": "end", "font-size": 11, fill: MUTED });
    label.textContent = fmt(p, 0);
    chart.append(label);
  }
  for (const t of niceTicks(t0, t1, 6)) {
    const label = svg("text", { x: sx(t), y: H - m.b + 16, "text-anchor": "middle", "font-size": 11, fill: MUTED });
    label.textContent = fmt(t, 0);
    chart.append(label);
  }
  chart.append(svg("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: AXIS, "stroke-width": 1 }));
  const xLabel = svg("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", "font-size": 11, fill: INK2 });
  xLabel.textContent = "Retardo τ (ns)";
  const yLabel = svg("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "font-size": 11, fill: INK2,
    transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` });
  yLabel.textContent = "Potencia (dBm)";
  chart.append(xLabel, yLabel);

  // Leyenda (encima del gráfico): siempre que haya 2 o más transmisores
  const ids = [...new Set(live.map((x) => x.t.tx_id))];
  if (data.transmitters.length >= 2) {
    const legend = el("div", { class: "note" });
    for (const id of data.transmitters.map((t) => t.tx_id)) {
      legend.append(el("span", { class: "key-line", style: `background:${txColor(store.txSlot(id))}` }),
        document.createTextNode(`${store.txName(id)}${ids.includes(id) ? "" : " (sin rayos)"}   `));
    }
    box.append(legend);
  }
  const tooltip = el("div", { class: "tooltip", hidden: true });
  for (const item of live) {
    const { t, ti, r, ri } = item;
    const color = txColor(store.txSlot(t.tx_id));
    const x = sx(r.tau_ns);
    const y = sy(num(r.power_dbm));
    const g = svg("g", { tabindex: 0 });
    g.append(svg("line", { x1: x, x2: x, y1: H - m.b, y2: y, stroke: color, "stroke-width": 2 }));
    g.append(svg("circle", { cx: x, cy: y, r: 4, fill: color, stroke: SURFACE, "stroke-width": 2 }));
    const hit = svg("rect", { x: x - 12, y: Math.min(y - 12, H - m.b - 24), width: 24, height: Math.max(24, H - m.b - y + 12), fill: "transparent" });
    g.append(hit);
    const show = () => {
      tooltip.replaceChildren(el("div", { class: "tt-value", text: `${fmt(num(r.power_dbm), 1)} dBm` }),
        el("div", { class: "tt-row" }, el("span", { class: "tt-key", style: `background:${color}` }), el("span", { class: "tt-label", text: r.label })),
        el("div", { class: "tt-label", text: `${store.txName(t.tx_id)} · τ = ${fmt(r.tau_ns, 1)} ns` }));
      tooltip.hidden = false;
      const bw = box.clientWidth;
      const px = (x / W) * bw;
      tooltip.style.left = `${Math.min(px + 12, bw - 200)}px`;
      tooltip.style.top = `${(y / W) * bw + 10}px`;
      g.firstChild.setAttribute("stroke-width", 3);
      app.hoverRay({ tx: ti, ray: ri });
      if (item.row) item.row.classList.add("hover");
    };
    const hide = () => {
      tooltip.hidden = true;
      g.firstChild.setAttribute("stroke-width", 2);
      app.hoverRay(null);
      if (item.row) item.row.classList.remove("hover");
    };
    g.addEventListener("pointerenter", show);
    g.addEventListener("pointerleave", hide);
    g.addEventListener("focus", show);
    g.addEventListener("blur", hide);
    chart.append(g);
  }
  box.append(chart, tooltip);
  return box;
}

// ------------------------------------------------------------------ estadísticas
export function renderStats(container, store, app) {
  container.replaceChildren();
  const r = store.result;
  if (!r) {
    container.append(el("div", { class: "empty-state", text: "Ejecuta la predicción (F5) para ver las estadísticas del área." }));
    return;
  }
  const grid = el("div", { class: "stats-grid" });
  container.append(grid);
  const total = r.grid.nx * r.grid.ny;

  // 1. Cobertura
  const cov = el("div");
  cov.append(el("h3", { text: "Cobertura del área" }));
  const power = r.metrics.rx_power;
  const thrInput = el("input", { type: "number", class: "narrow", step: "1", value: store.view.coverageThreshold });
  thrInput.addEventListener("change", () => { store.setView({ coverageThreshold: Number(thrInput.value) }); renderStats(container, store, app); });
  cov.append(el("div", { class: "coverage-input" }, el("span", { text: "P_r ≥" }), thrInput, el("span", { text: "dBm" })));
  let covered = 0;
  let withData = 0;
  for (let k = 0; k < total; k++) {
    if (isFinite(power[k])) { withData += 1; if (power[k] >= store.view.coverageThreshold) covered += 1; }
  }
  const rows = [["Píxeles con señal", `${fmt((100 * withData) / total, 1)} %`],
    ["Área cubierta (P_r ≥ umbral)", `${fmt((100 * covered) / total, 1)} %`],
    ["Cubiertos, de los píxeles con señal", `${withData ? fmt((100 * covered) / withData, 1) : "—"} %`]];
  if (r.metrics.sinr) {
    const sinr = r.metrics.sinr;
    let ok = 0;
    for (let k = 0; k < total; k++) if (isFinite(sinr[k]) && sinr[k] >= store.view.sinrThreshold) ok += 1;
    rows.push([`Área con SINR ≥ ${fmt(store.view.sinrThreshold, 0)} dB`, `${fmt((100 * ok) / total, 1)} %`]);
  }
  cov.append(simpleTable(rows));
  grid.append(cov);

  // 2. Distribución de la métrica actual
  const layer = currentLayer(store);
  if (layer && layer.def.kind === "seq") {
    const dist = el("div");
    dist.append(el("h3", { text: `${layer.label} (${layer.def.unit})` }));
    const finite = [];
    for (let k = 0; k < layer.values.length; k++) if (isFinite(layer.values[k])) finite.push(layer.values[k]);
    finite.sort((a, b) => a - b);
    if (finite.length) {
      const q = (f) => finite[Math.min(finite.length - 1, Math.max(0, Math.round(f * (finite.length - 1))))];
      const mean = finite.reduce((s, v) => s + v, 0) / finite.length;
      dist.append(simpleTable([["Mínimo", fmt(finite[0], 1)], ["Percentil 10", fmt(q(0.1), 1)], ["Mediana", fmt(q(0.5), 1)],
        ["Media", fmt(mean, 1)], ["Percentil 90", fmt(q(0.9), 1)], ["Máximo", fmt(finite[finite.length - 1], 1)]]));
      dist.append(histogram(finite, layer));
    } else dist.append(el("div", { class: "empty-state", text: "Sin datos." }));
    grid.append(dist);
  }

  // 3. Reparto del área entre servidores
  const share = el("div");
  share.append(el("h3", { text: "Área servida por cada transmisor" }));
  const counts = new Map(r.tx_ids.map((id) => [id, 0]));
  let none = 0;
  const bs = r.metrics.best_server;
  for (let k = 0; k < total; k++) {
    if (counts.has(bs[k])) counts.set(bs[k], counts.get(bs[k]) + 1);
    else none += 1;
  }
  const table = el("table", { class: "data" });
  table.append(el("tr", {}, el("th", { text: "Transmisor" }), el("th", { class: "num", text: "Área" }), el("th", { text: "" })));
  for (const [id, n] of counts) {
    const pct = (100 * n) / total;
    const bar = el("div", { style: `height:8px;border-radius:0 4px 4px 0;background:${txColor(store.txSlot(id))};width:${Math.max(1, pct)}%` });
    table.append(el("tr", {}, el("td", {}, el("span", { class: "key-line", style: `background:${txColor(store.txSlot(id))}` }),
      document.createTextNode(store.txName(id))), el("td", { class: "num", text: `${fmt(pct, 1)} %` }), el("td", { style: "width:45%" }, bar)));
  }
  table.append(el("tr", {}, el("td", { class: "muted", text: "Sin servidor" }), el("td", { class: "num", text: `${fmt((100 * none) / total, 1)} %` }), el("td")));
  share.append(table);
  const rep = r.report;
  share.append(el("p", { class: "note", text: `${r.grid.nx} × ${r.grid.ny} píxeles · ${rep.transmitters} transmisores · ${rep.reflectors} muros en el motor (${rep.obstructing_reflectors} obstruyen) · ${rep.edges || 0} aristas · ${fmt(rep.elapsed_s, 2)} s` }));
  grid.append(share);
}

function simpleTable(rows) {
  const table = el("table", { class: "data" });
  for (const [k, v] of rows) table.append(el("tr", {}, el("td", { text: k }), el("td", { class: "num", text: v })));
  return table;
}

function histogram(sorted, layer) {
  const W = 320;
  const H = 120;
  const m = { l: 8, r: 8, t: 8, b: 22 };
  const [lo, hi] = layer.range;
  const bins = 24;
  const counts = new Array(bins).fill(0);
  for (const v of sorted) {
    const b = Math.max(0, Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins)));
    counts[b] += 1;
  }
  const max = Math.max(...counts, 1);
  const box = el("div", { class: "chart-box" });
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Histograma de la métrica sobre el área" });
  const bw = (W - m.l - m.r) / bins;
  const tooltip = el("div", { class: "tooltip", hidden: true });
  counts.forEach((c, b) => {
    const h = ((H - m.t - m.b) * c) / max;
    const x = m.l + b * bw;
    const rect = svg("rect", { x: x + 1, y: H - m.b - h, width: Math.max(1, bw - 2), height: h, fill: "#2a78d6", rx: h > 4 ? 2 : 0 });
    const hit = svg("rect", { x, y: m.t, width: bw, height: H - m.t - m.b, fill: "transparent" });
    hit.addEventListener("pointerenter", () => {
      tooltip.replaceChildren(el("div", { class: "tt-value", text: `${fmt((100 * c) / sorted.length, 1)} %` }),
        el("div", { class: "tt-label", text: `${fmt(lo + (b / bins) * (hi - lo), 1)} a ${fmt(lo + ((b + 1) / bins) * (hi - lo), 1)} ${layer.def.unit}` }));
      tooltip.hidden = false;
      const scale = box.clientWidth / W;
      tooltip.style.left = `${Math.min(x * scale + 10, box.clientWidth - 150)}px`;
      tooltip.style.top = `${10}px`;
      rect.setAttribute("stroke", INK2);
    });
    hit.addEventListener("pointerleave", () => { tooltip.hidden = true; rect.removeAttribute("stroke"); });
    chart.append(rect, hit);
  });
  chart.append(svg("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: AXIS }));
  for (const t of niceTicks(lo, hi, 4)) {
    const label = svg("text", { x: m.l + ((t - lo) / (hi - lo)) * (W - m.l - m.r), y: H - 6, "text-anchor": "middle", "font-size": 11, fill: MUTED });
    label.textContent = fmt(t, 0);
    chart.append(label);
  }
  box.append(chart, tooltip);
  return box;
}

// ------------------------------------------------------------------ mediciones: comparación y calibración
const signed = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${v >= 0 ? "+" : "−"}${fmt(Math.abs(v), d)}`);

function statsRows(st, counts) {
  const rows = [["Puntos comparados", String(st.n)]];
  const skipped = Object.entries(counts || {}).filter(([k, v]) => k !== "ok" && v);
  if (skipped.length) rows.push(["Sin comparar", skipped.map(([k, v]) => `${v} ${k}`).join(", ")]);
  if (st.n) {
    rows.push(["Sesgo (predicho − medido)", `${signed(st.bias_db)} dB`], ["Desviación estándar", `${fmt(st.std_db, 1)} dB`],
      ["RMSE", `${fmt(st.rmse_db, 1)} dB`], ["Error máximo", `${fmt(st.max_abs_db, 1)} dB`],
      ["Correlación r", st.r === null ? "—" : fmt(st.r, 3)]);
  }
  return rows;
}

export function renderMeasurements(container, store, app) {
  container.replaceChildren();
  const p = store.project;
  if (!p) return;
  const ms = p.measurements || [];
  if (!ms.length) {
    container.append(el("div", { class: "empty-state" },
      el("p", { text: "No hay mediciones. Con ellas puedes ver cuánto se aleja la predicción de la realidad y ajustar el modelo." }),
      el("p", { text: "Importa un CSV (Archivo → Importar mediciones) con columnas x, y y la potencia en dBm (rssi, dbm, potencia o señal); opcionales: z, tx (id o nombre del transmisor) y una etiqueta. O usa la herramienta Medición (N): clic donde mediste y escribe la potencia que leíste." }),
      el("div", { class: "props-actions" },
        el("button", { class: "btn", text: "Importar CSV…", onclick: () => app.pickMeasurementsCsv() }),
        el("button", { class: "btn", text: "Anotar en el mapa (N)", onclick: () => app.setTool("sample") }))));
    return;
  }
  const grid = el("div", { class: "stats-grid" });
  container.append(grid);
  const rev = store.measurementRevision();
  const cmp = store.comparison && store.comparison.revision === rev ? store.comparison.data : null;

  // 1. Comparación
  const box = el("div");
  box.append(el("h3", { text: "Predicción frente a mediciones" }));
  if (!cmp) {
    box.append(el("p", { class: "note", text: store.comparison ? "El proyecto cambió desde la última comparación." : "Aún no se compara." }),
      el("div", { class: "props-actions" }, el("button", { class: "btn primary", text: "Comparar", onclick: () => app.compareMeasurements() })));
  } else {
    box.append(simpleTable(statsRows(cmp.stats, cmp.counts)));
    box.append(el("p", { class: "note", text: "Error = predicho − medido: positivo si el modelo es optimista. Sin sombra aleatoria: se compara la parte determinista de la predicción." }));
  }
  grid.append(box);

  // 2. Dispersión predicho vs. medido
  if (cmp && cmp.stats.n) {
    const sc = el("div");
    sc.append(el("h3", { text: "Predicho vs. medido (dBm)" }));
    sc.append(scatter(store, app, cmp.points));
    grid.append(sc);
  }

  // 3. Calibración
  grid.append(calibrationBox(store, app, rev));

  // 4. Puntos con más error
  if (cmp && cmp.stats.n) {
    const worst = cmp.points.filter((q) => q.status === "ok").sort((a, b) => Math.abs(b.error_db) - Math.abs(a.error_db)).slice(0, 8);
    const w = el("div");
    w.append(el("h3", { text: "Puntos con más error" }));
    const table = el("table", { class: "data clickable" },
      el("tr", {}, el("th", { text: "#" }), el("th", { text: "Medido" }), el("th", { text: "Predicho" }), el("th", { text: "Error" }), el("th", { text: "Tx" })));
    for (const q of worst) {
      const row = el("tr", { title: "Clic para seleccionarla en el mapa", onclick: () => { app.setTool("select"); app.zoomTo({ type: "measurement", id: q.index }); } },
        el("td", { text: `${q.index + 1}${q.label ? ` · ${q.label}` : ""}` }), el("td", { class: "num", text: fmt(q.measured_dbm, 1) }),
        el("td", { class: "num", text: fmt(q.predicted_dbm, 1) }), el("td", { class: "num", text: signed(q.error_db) }),
        el("td", { text: q.tx_id === null ? "—" : store.txName(q.tx_id) }));
      table.append(row);
    }
    w.append(table);
    grid.append(w);
  }
}

function scatter(store, app, points) {
  const W = 320;
  const H = 230;
  const m = { l: 40, r: 10, t: 10, b: 30 };
  const ok = points.filter((q) => q.status === "ok");
  const values = ok.flatMap((q) => [q.measured_dbm, q.predicted_dbm]);
  const ticks0 = niceTicks(Math.min(...values), Math.max(...values), 5);
  const lo = Math.min(ticks0[0], Math.min(...values));
  const hi = Math.max(ticks0[ticks0.length - 1], Math.max(...values));
  const span = hi - lo || 1;
  const X = (v) => m.l + ((v - lo) / span) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - lo) / span) * (H - m.t - m.b);
  const box = el("div", { class: "chart-box" });
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Potencia predicha frente a la medida en cada punto" });
  for (const t of niceTicks(lo, hi, 5)) {
    chart.append(svg("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: AXIS, "stroke-opacity": 0.45 }),
      svg("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: AXIS, "stroke-opacity": 0.45 }));
    const lx = svg("text", { x: X(t), y: H - m.b + 13, "text-anchor": "middle", "font-size": 10.5, fill: MUTED });
    lx.textContent = fmt(t, 0);
    const ly = svg("text", { x: m.l - 5, y: Y(t) + 3.5, "text-anchor": "end", "font-size": 10.5, fill: MUTED });
    ly.textContent = fmt(t, 0);
    chart.append(lx, ly);
  }
  // Línea 1:1 (predicción perfecta)
  chart.append(svg("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), stroke: INK2, "stroke-width": 1.2, "stroke-dasharray": "4 3" }));
  const xl = svg("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", "font-size": 11, fill: INK2 });
  xl.textContent = "Medido";
  const yl = svg("text", { x: 11, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "font-size": 11, fill: INK2, transform: `rotate(-90 11 ${(m.t + H - m.b) / 2})` });
  yl.textContent = "Predicho";
  chart.append(xl, yl);
  const tooltip = el("div", { class: "tooltip", hidden: true });
  for (const q of ok) {
    const color = q.tx_id === null ? INK2 : txColor(store.txSlot(q.tx_id));
    const dot = svg("circle", { cx: X(q.measured_dbm), cy: Y(q.predicted_dbm), r: 4, fill: color, "fill-opacity": 0.85, stroke: SURFACE, "stroke-width": 1.5 });
    dot.style.cursor = "pointer";
    dot.addEventListener("pointerenter", () => {
      tooltip.replaceChildren(el("div", { class: "tt-value", text: `${signed(q.error_db)} dB` }),
        el("div", { class: "tt-label", text: `Medición ${q.index + 1}${q.label ? ` · ${q.label}` : ""}: medido ${fmt(q.measured_dbm, 1)}, predicho ${fmt(q.predicted_dbm, 1)} dBm` }),
        el("div", { class: "tt-label", text: q.tx_id === null ? "" : store.txName(q.tx_id) }));
      tooltip.hidden = false;
      const scale = box.clientWidth / W;
      tooltip.style.left = `${Math.min(X(q.measured_dbm) * scale + 10, box.clientWidth - 190)}px`;
      tooltip.style.top = `${Math.max(0, Y(q.predicted_dbm) * scale - 50)}px`;
      dot.setAttribute("stroke", INK2);
    });
    dot.addEventListener("pointerleave", () => { tooltip.hidden = true; dot.setAttribute("stroke", SURFACE); });
    dot.addEventListener("click", () => { app.setTool("select"); app.zoomTo({ type: "measurement", id: q.index }); });
    chart.append(dot);
  }
  box.append(chart, tooltip);
  const txs = [...new Set(ok.map((q) => q.tx_id))].filter((id) => id !== null);
  if (txs.length > 1) {
    box.append(el("div", { class: "legend-inline" }, ...txs.map((id) => el("span", { class: "legend-item" },
      el("span", { class: "dot", style: `background:${txColor(store.txSlot(id))}` }), el("span", { text: store.txName(id) })))));
  }
  box.append(el("p", { class: "note", text: "Sobre la diagonal, la predicción coincide con la medición; por encima, el modelo es optimista." }));
  return box;
}

function calibrationBox(store, app, rev) {
  const p = store.project;
  const box = el("div");
  box.append(el("h3", { text: "Calibración" }));
  const opts = app.calibOptions ||= { excluded: new Set(), tx_power: "auto", exponent: true };
  const f = (() => {
    if (p.settings.reference_frequency_hz) return p.settings.reference_frequency_hz;
    const tx = p.transmitters.find((t) => t.enabled);
    return tx ? tx.frequency_hz : 2.4e9;
  })();
  const used = new Set([...p.buildings.map((b) => b.material), ...p.walls.map((w) => w.material)]);
  const usesWalls = p.settings.mode === "raytracing" || p.settings.mode === "multiwall";
  const candidates = usesWalls
    ? p.materials.filter((m) => used.has(m.id) && (m.penetration === "fixed" || m.penetration === "frequency")) : [];
  const form = el("div", { class: "calib-form" });
  if (candidates.length) {
    form.append(el("div", { class: "note", text: "Pérdida por cruce de los materiales:" }));
    for (const m of candidates) {
      const loss = m.penetration === "fixed" ? m.loss_db : Math.max(0, m.loss_a_db + m.loss_b_db_per_ghz * f / 1e9);
      const check = el("input", { type: "checkbox" });
      check.checked = !opts.excluded.has(m.id);
      check.addEventListener("change", () => { if (check.checked) opts.excluded.delete(m.id); else opts.excluded.add(m.id); });
      form.append(el("label", { class: "inline-check" }, check, el("span", { class: "swatch", style: `background:${m.color}` }),
        el("span", { text: `${m.name || m.id} (${fmt(loss, 1)} dB)` })));
    }
  } else if (usesWalls) {
    form.append(el("p", { class: "note", text: "Ningún material usado tiene una pérdida finita que ajustar." }));
  } else {
    form.append(el("p", { class: "note", text: "Los modelos empíricos no usan los materiales: se ajustan la potencia y, en log-distancia, el exponente." }));
  }
  const power = el("select", {}, ...[["auto", "Automático"], ["per_tx", "Una por transmisor"], ["common", "Una común a todos"], ["none", "No ajustar"]]
    .map(([v, t]) => el("option", { value: v, text: t })));
  power.value = opts.tx_power;
  power.addEventListener("change", () => { opts.tx_power = power.value; });
  form.append(el("label", { class: "field", title: "Automático: una por transmisor si todas las mediciones dicen de qué transmisor son; si no, una común (por ejemplo, la ganancia del equipo de medición)" },
    el("span", { text: "Potencia de los Tx" }), power));
  if (p.settings.mode === "logdistance" || p.settings.mode === "multiwall") {
    const exp = el("input", { type: "checkbox" });
    exp.checked = opts.exponent;
    exp.addEventListener("change", () => { opts.exponent = exp.checked; });
    form.append(el("label", { class: "inline-check" }, exp, el("span", { text: `Exponente n (ahora ${fmt(p.settings.log_distance_exponent, 2)})` })));
  }
  form.append(el("div", { class: "props-actions" }, el("button", { class: "btn primary", text: "Calibrar", onclick: () => {
    const materials = candidates.filter((m) => !opts.excluded.has(m.id)).map((m) => m.id);
    app.calibrate({ materials: usesWalls ? materials : false, tx_power: opts.tx_power, exponent: opts.exponent });
  } })));
  box.append(form);

  const c = store.calibration;
  if (c && c.revision === rev) {
    const d = c.data;
    const table = el("table", { class: "data calib-table" },
      el("tr", {}, el("th", { text: "Parámetro" }), el("th", { text: "Antes" }), el("th", { text: "Después" }), el("th", { text: "± 1σ" }), el("th", { text: "Puntos" })));
    for (const q of d.parameters) {
      const unit = q.unit ? ` ${q.unit}` : "";
      const isOffset = q.kind === "tx_power";
      table.append(el("tr", {},
        el("td", { text: q.name }),
        el("td", { class: "num", text: isOffset ? "—" : `${fmt(q.before, 2)}${unit}` }),
        el("td", { class: "num", text: isOffset ? `${signed(q.after, 2)} dB` : `${fmt(q.after, 2)}${unit}` }),
        el("td", { class: "num", text: q.uncertainty === null ? "—" : fmt(q.uncertainty, 2) }),
        el("td", { class: "num", text: String(q.points) })));
    }
    const heading = el("div", { class: "props-section", text: "Resultado" });
    box.append(heading, table);
    if (app.scrollToCalibration) {
      app.scrollToCalibration = false;
      requestAnimationFrame(() => heading.scrollIntoView({ block: "start", behavior: "smooth" }));
    }
    box.append(simpleTable([["RMSE", `${fmt(d.before.stats.rmse_db, 2)} → ${fmt(d.after.stats.rmse_db, 2)} dB`],
      ["Sesgo", `${signed(d.before.stats.bias_db, 2)} → ${signed(d.after.stats.bias_db, 2)} dB`],
      ["Desviación", `${fmt(d.before.stats.std_db, 2)} → ${fmt(d.after.stats.std_db, 2)} dB`]]));
    for (const note of d.notes) box.append(el("p", { class: "note warn-note", text: note }));
    box.append(el("p", { class: "note", text: "± 1σ: incertidumbre de cada valor según la dispersión que queda y cuántos puntos dependen de él. Con pocos puntos, o si una pared casi no se cruza, el valor es poco fiable." }));
    box.append(el("div", { class: "props-actions" },
      el("button", { class: "btn primary", text: "Aplicar al proyecto", onclick: () => app.applyCalibration() }),
      el("button", { class: "btn", text: "Descartar", onclick: () => { store.calibration = null; store.emit("comparison"); } })));
  } else if (c) {
    box.append(el("p", { class: "note", text: "El proyecto cambió después de calibrar: vuelve a calibrar." }));
  }
  return box;
}

// Wi-Fi en el punto del receptor: SINR con lo que mete cada AP según cuánto se enciman los canales, y la MCS resultante
function probeWifi(store, w) {
  const box = el("div", { class: "probe-wifi" });
  const serving = store.tx(w.serving_tx_id);
  const ch = serving && serving.wifi_channel !== null && serving.wifi_channel !== undefined
    ? `canal ${serving.wifi_channel}, ${serving.wifi_width_mhz} MHz` : "sin canal Wi-Fi";
  box.append(el("div", { class: "props-section", text: "Wi-Fi en este punto" }));
  const rows = [["Servidor", `${store.txName(w.serving_tx_id)} (${ch})`],
    ["SINR", `${fmt(num(w.sinr_db), 1)} dB`],
    ["Interferencia + ruido", `${fmt(num(w.interference_plus_noise_dbm), 1)} dBm (ruido ${fmt(num(w.noise_dbm), 1)} dBm)`]];
  if (w.standard) {
    const mcs = num(w.mcs);
    rows.push(["MCS", mcs < 0 ? "ninguna: el SINR no alcanza ni MCS 0" : `MCS ${mcs} · ${w.streams} flujo(s) · ${w.standard === "ax" ? "802.11ax" : w.standard === "ac" ? "802.11ac" : "802.11n"}`],
      ["Tasa PHY", `${fmt(num(w.rate_mbps), 1)} Mbit/s`],
      ["AP que comparten el canal", `${w.contention} (por arriba del umbral CCA)`]);
  }
  box.append(simpleTable(rows));
  if (w.interferers && w.interferers.length) {
    const table = el("table", { class: "data" },
      el("tr", {}, ...["Interfiere", "Canal", "P_r (dBm)", "Encimado (dB)", "Aporta (dBm)"].map((h, k) => el("th", { class: k > 1 ? "num" : "", text: h }))));
    for (const i of w.interferers) {
      const t = store.tx(i.tx_id);
      table.append(el("tr", {},
        el("td", {}, el("span", { class: "key-line", style: `background:${txColor(store.txSlot(i.tx_id))}` }), document.createTextNode(store.txName(i.tx_id))),
        el("td", { text: t && t.wifi_channel !== null && t.wifi_channel !== undefined ? `${t.wifi_channel} (${t.wifi_width_mhz} MHz)` : "—" }),
        el("td", { class: "num", text: fmt(num(i.rx_power_dbm), 1) }),
        el("td", { class: "num", text: fmt(num(i.overlap_db), 1) }),
        el("td", { class: "num", text: fmt(num(i.interference_dbm), 1) })));
    }
    box.append(table);
    box.append(el("p", { class: "note", text: "Encimado: fracción de la potencia del otro AP que cae dentro del canal del servidor, según la máscara espectral de 802.11 (0 dB = mismo canal). Los AP en canales que no se enciman no aparecen." }));
  } else box.append(el("p", { class: "note", text: "Ningún otro AP transmite en un canal que se encime con el del servidor." }));
  return box;
}
