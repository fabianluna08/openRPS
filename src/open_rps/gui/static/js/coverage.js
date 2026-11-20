// Capa de cobertura actual (métrica + transmisor + escala) y su imagen, compartida por las vistas 2D y 3D.
import { METRICS, NODATA, autoRange, hexToRgb, rampLUT, txColor } from "./colors.js";

const LUT = rampLUT();
let cache = { key: null, layer: null };

export function metricOptions(store) {
  const r = store.result;
  const out = [];
  for (const [id, def] of Object.entries(METRICS)) {
    if (r && !r.metrics[id]) continue;          // p. ej. SINR desactivado, o sin resultado
    out.push([id, def.label]);
  }
  return out;
}

// Valores [ny * nx] de la capa actual, o null si no hay resultado
export function currentLayer(store) {
  const r = store.result;
  if (!r) return null;
  const v = store.view;
  const metric = r.metrics[v.metric] ? v.metric : "rx_power";
  const def = METRICS[metric];
  const perTxAllowed = (metric === "rx_power" || metric === "path_loss") && r.perTx && v.txFilter !== "best";
  const txPos = perTxAllowed ? r.tx_ids.indexOf(Number(v.txFilter)) : -1;
  const key = [r.result_id, metric, txPos, v.rangeAuto, v.rangeMin, v.rangeMax].join("|");
  if (cache.key === key) return cache.layer;

  let values = r.metrics[metric];
  let label = def.label;
  if (txPos >= 0) {
    const txId = r.tx_ids[txPos];
    const power = r.perTx[txPos];
    if (metric === "rx_power") values = power;
    else {
      const tx = r.projectSnapshot.transmitters.find((t) => t.tx_id === txId);
      values = new Float32Array(power.length);
      for (let k = 0; k < power.length; k++) values[k] = isFinite(power[k]) ? tx.tx_power_dbm - power[k] : NaN;
    }
    label = `${def.label} · ${store.txName(txId)}`;
  }
  if (def.negativeIsNone) values = values.map((x) => (x < 0 ? NaN : x));   // MCS -1: no connection
  let range = null;
  if (def.kind === "seq") {
    range = v.rangeAuto || !(v.rangeMax > v.rangeMin) ? autoRange(values, def.zeroBased) : [v.rangeMin, v.rangeMax];
  }
  const layer = { metric, def, label, values, range, grid: r.grid, txPos };
  layer.image = buildImage(store, layer);
  cache = { key, layer };
  return layer;
}

export function invalidateLayer() {
  cache = { key: null, layer: null };
}

function buildImage(store, layer) {
  const { nx, ny } = layer.grid;
  const canvas = document.createElement("canvas");
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(nx, ny);
  const data = img.data;
  const nodata = hexToRgb(NODATA);
  const values = layer.values;
  const categorical = layer.def.kind === "cat";
  const slotColors = new Map();
  if (categorical) {
    for (const id of store.result.tx_ids) slotColors.set(id, hexToRgb(txColor(store.txSlot(id))));
  }
  const [lo, hi] = layer.range || [0, 1];
  const span = hi - lo || 1;
  for (let j = 0; j < ny; j++) {
    const row = ny - 1 - j;                    // la fila 0 de la imagen es la y máxima
    for (let i = 0; i < nx; i++) {
      const value = values[j * nx + i];
      const o = 4 * (row * nx + i);
      let rgb = nodata;
      if (categorical) {
        if (value >= 0 && slotColors.has(value)) rgb = slotColors.get(value);
      } else if (isFinite(value)) {
        const k = Math.max(0, Math.min(255, Math.round(((value - lo) / span) * 255)));
        data[o] = LUT[3 * k];
        data[o + 1] = LUT[3 * k + 1];
        data[o + 2] = LUT[3 * k + 2];
        data[o + 3] = 255;
        continue;
      }
      data[o] = rgb[0];
      data[o + 1] = rgb[1];
      data[o + 2] = rgb[2];
      data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// Índice del píxel de la rejilla más cercano a (x, y), o -1 fuera del área
export function pixelAt(grid, x, y) {
  const i = grid.dx > 0 ? Math.round((x - grid.x0) / grid.dx) : 0;
  const j = grid.dy > 0 ? Math.round((y - grid.y0) / grid.dy) : 0;
  if (i < 0 || j < 0 || i >= grid.nx || j >= grid.ny) return -1;
  return j * grid.nx + i;
}

// Extensión de la imagen en metros: las celdas están centradas en las coordenadas de la rejilla
export function imageExtent(grid) {
  const hx = grid.dx / 2 || 0.5;
  const hy = grid.dy / 2 || 0.5;
  return {
    x0: grid.x0 - hx, x1: grid.x0 + (grid.nx - 1) * grid.dx + hx,
    y0: grid.y0 - hy, y1: grid.y0 + (grid.ny - 1) * grid.dy + hy,
  };
}
