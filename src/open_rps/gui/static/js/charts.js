// Leyenda del mapa, panel del receptor (perfil potencia-retardo y tabla de rayos) y estadísticas.
import { AXIS, BLUE_RAMP, INK2, MUTED, SURFACE, fmt, niceTicks, num, txColor } from "./colors.js";
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
    BLUE_RAMP.forEach((c, k) => grad.append(svg("stop", { offset: `${(k / (BLUE_RAMP.length - 1)) * 100}%`, "stop-color": c })));
    defs.append(grad);
    bar.append(defs, svg("rect", { x: pad, y: 4, width: width - 2 * pad, height: 12, rx: 2, fill: `url(#${gradId})` }));
    for (const t of niceTicks(lo, hi, 4)) {
      const x = pad + ((t - lo) / (hi - lo)) * (width - 2 * pad);
      bar.append(svg("line", { x1: x, x2: x, y1: 16, y2: 20, stroke: INK2, "stroke-width": 1 }));
      const label = svg("text", { x, y: 32, "text-anchor": "middle", "font-size": 11, fill: INK2 });
      label.textContent = fmt(t, Math.abs(hi - lo) < 5 ? 1 : 0);
      bar.append(label);
    }
    container.append(bar);
    container.append(el("div", { class: "legend-row" }, el("span", { class: "legend-swatch", style: "background:#e1e0d9" }),
      el("span", { class: "muted", text: "Sin datos (sin señal o interior bloqueado)" })));
  }
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
  if (data.mode !== "raytracing") {
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
