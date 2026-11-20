// Vista 2D: plano del escenario, editor (transmisores, edificios, muros) y mapa de cobertura.
import { INK, INK2, MUTED, SURFACE, fmt, niceStep, rampColor, txColor } from "./colors.js";
import { currentLayer, imageExtent, pixelAt } from "./coverage.js";
import { centroid, distToSegment, isSimplePolygon, pointInPolygon, round, snap } from "./geometry.js";
import { nextId, nextTxId } from "./state.js";

const HANDLE_PX = 7;
const TX_HIT_PX = 11;
const WALL_HIT_PX = 7;
const SAMPLE_HIT_PX = 8;
const SNAP_M = 1;
const SELECT = "#1c5cab";

export class View2D {
  constructor(canvas, store, app) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.store = store;
    this.app = app;
    this.scale = 1;          // píxeles por metro
    this.ox = 0;
    this.oy = 0;             // pantalla = (x * scale + ox, oy - y * scale): el eje y apunta hacia arriba
    this.width = 1;
    this.height = 1;
    this.needsFit = true;
    this.drag = null;
    this.draft = null;       // polígono en construcción, o inicio de un muro
    this.measure = null;
    this.planScale = null;   // primer punto al escalar el plano de fondo
    this.hover = null;       // [x, y] del cursor en metros
    this.spaceDown = false;
    this.pending = false;
    this.tooltip = document.getElementById("tooltip2d");

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", (e) => this.onUp(e));
    canvas.addEventListener("pointerleave", () => this.onLeave());
    canvas.addEventListener("dblclick", (e) => this.onDoubleClick(e));
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    window.addEventListener("keydown", (e) => { if (e.code === "Space" && !isTyping(e)) this.spaceDown = true; });
    window.addEventListener("keyup", (e) => { if (e.code === "Space") this.spaceDown = false; });
  }

  // ----- transformaciones
  w2s(x, y) { return [x * this.scale + this.ox, this.oy - y * this.scale]; }
  s2w(px, py) { return [(px - this.ox) / this.scale, (this.oy - py) / this.scale]; }

  eventPoint(e) {
    const rect = this.canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  snapPoint(p, e) {
    if (!this.store.view.snap || (e && e.altKey)) return [round(p[0], 2), round(p[1], 2)];
    return [snap(p[0], SNAP_M), snap(p[1], SNAP_M)];
  }

  resize() {
    const parent = this.canvas.parentElement;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.dpr = dpr;
    if (this.needsFit || this.width <= 1) this.fit();
    else {
      // mantener el centro al cambiar el tamaño
      const [cx, cy] = this.s2w(this.width / 2, this.height / 2);
      this.width = w;
      this.height = h;
      this.ox = w / 2 - cx * this.scale;
      this.oy = h / 2 + cy * this.scale;
    }
    this.width = w;
    this.height = h;
    this.request();
  }

  fit() {
    const parent = this.canvas.parentElement;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    const p = this.store.project;
    if (!p || !w || !h) { this.needsFit = true; return; }
    this.width = w;
    this.height = h;
    const a = p.area;
    const spanX = Math.max(1e-6, a.x_max - a.x_min);
    const spanY = Math.max(1e-6, a.y_max - a.y_min);
    this.scale = Math.min((w - 70) / spanX, (h - 60) / spanY);
    this.ox = w / 2 - ((a.x_min + a.x_max) / 2) * this.scale;
    this.oy = h / 2 + ((a.y_min + a.y_max) / 2) * this.scale;
    this.needsFit = false;
    this.request();
  }

  request() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.draw();
    });
  }

  // ----- dibujo
  draw() {
    const ctx = this.ctx;
    const p = this.store.project;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = "#ecebe6";
    ctx.fillRect(0, 0, this.width, this.height);
    if (!p) return;
    const v = this.store.view;
    const a = p.area;
    const [ax0, ay1] = this.w2s(a.x_min, a.y_max);
    const [ax1, ay0] = this.w2s(a.x_max, a.y_min);
    ctx.fillStyle = SURFACE;
    ctx.fillRect(ax0, ay1, ax1 - ax0, ay0 - ay1);

    const layer = currentLayer(this.store);
    if (layer) {
      const ext = imageExtent(layer.grid);
      const [ix0, iy1] = this.w2s(ext.x0, ext.y1);
      const [ix1, iy0] = this.w2s(ext.x1, ext.y0);
      ctx.save();
      ctx.globalAlpha = v.opacity * (this.store.outdated ? 0.55 : 1);
      ctx.imageSmoothingEnabled = v.smooth;
      ctx.drawImage(layer.image, ix0, iy1, ix1 - ix0, iy0 - iy1);
      ctx.restore();
    }
    this.drawFloorPlan(p);
    if (v.grid) this.drawGrid(ax0, ax1, ay1, ay0);
    ctx.strokeStyle = INK2;
    ctx.lineWidth = 1;
    ctx.strokeRect(ax0 + 0.5, ay1 + 0.5, ax1 - ax0, ay0 - ay1);

    this.drawBuildings(p, !!layer);
    this.drawWalls(p);
    this.drawMeasurements(p);
    this.drawRays();
    this.drawTransmitters(p);
    this.drawProbe();
    this.drawDraft();
    this.drawMeasure();
    this.drawPlanScale();
    this.drawScaleBar();
  }

  // Plano de fondo: en modo "multiply", lo blanco del plano deja ver el mapa y las líneas quedan oscuras
  planRect(plan) {
    const h = plan.width_m * plan.pixel_height / plan.pixel_width;
    const [x0, y0] = this.w2s(plan.x_m, plan.y_m + h);
    const [x1, y1] = this.w2s(plan.x_m + plan.width_m, plan.y_m);
    return [x0, y0, x1 - x0, y1 - y0];
  }

  drawFloorPlan(p) {
    const plan = p.floor_plan;
    if (!plan) return;
    const selected = this.store.isSelected("floorplan", undefined);
    const img = plan.visible ? this.store.floorPlanImage() : null;
    const [x, y, w, h] = this.planRect(plan);
    const ctx = this.ctx;
    if (img) {
      ctx.save();
      ctx.globalAlpha = plan.opacity;
      ctx.globalCompositeOperation = "multiply";
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, x, y, w, h);
      ctx.restore();
    }
    if (selected) {
      ctx.save();
      ctx.strokeStyle = SELECT;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }
  }

  // Puntos medidos: rombos rellenos con su valor en la escala del mapa (si el mapa es de potencia)
  drawMeasurements(p) {
    const ms = p.measurements || [];
    if (!ms.length) return;
    const ctx = this.ctx;
    const layer = currentLayer(this.store);
    const range = layer && layer.metric === "rx_power" && layer.range ? layer.range : null;
    const c = this.store.comparison;
    const points = c && c.revision === this.store.measurementRevision() ? c.data.points : null;
    ctx.save();
    ctx.font = "600 10.5px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ms.forEach((m, k) => {
      const [sx, sy] = this.w2s(m.x, m.y);
      if (sx < -20 || sy < -20 || sx > this.width + 20 || sy > this.height + 20) return;
      const selected = this.store.isSelected("measurement", k);
      const r = selected ? 7.5 : 6;
      const diamond = (q) => {
        ctx.beginPath();
        ctx.moveTo(sx, sy - q);
        ctx.lineTo(sx + q, sy);
        ctx.lineTo(sx, sy + q);
        ctx.lineTo(sx - q, sy);
        ctx.closePath();
      };
      diamond(r + 2);                      // halo claro para separarlo del mapa
      ctx.fillStyle = SURFACE;
      ctx.fill();
      diamond(r);
      ctx.fillStyle = range ? rampColor((m.rssi_dbm - range[0]) / (range[1] - range[0] || 1)) : SURFACE;
      ctx.fill();
      ctx.lineWidth = selected ? 2 : 1.2;
      ctx.strokeStyle = selected ? SELECT : INK;
      ctx.stroke();
      const pt = points && points[k];
      if (this.store.view.labels && pt && pt.status === "ok" && this.scale * 1 > 0) {
        const e = pt.error_db;
        haloText(ctx, `${e >= 0 ? "+" : "−"}${fmt(Math.abs(e), 1)}`, sx + r + 3, sy, INK2);
      }
    });
    ctx.restore();
  }

  drawPlanScale() {
    const d = this.planScale;
    if (!d || !this.hover) return;
    const b = [round(this.hover[0], 3), round(this.hover[1], 3)];
    const [x0, y0] = this.w2s(...d.a);
    const [x1, y1] = this.w2s(...b);
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = SELECT;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    this.drawHandle(...d.a);
    ctx.font = "600 12px system-ui, sans-serif";
    haloText(ctx, `${fmt(Math.hypot(b[0] - d.a[0], b[1] - d.a[1]), 2)} m en el mapa`, (x0 + x1) / 2 + 8, (y0 + y1) / 2 - 8, INK);
    ctx.restore();
  }

  sampleAt(sx, sy) {
    const ms = this.store.project.measurements || [];
    for (let k = ms.length - 1; k >= 0; k--) {
      const [mx, my] = this.w2s(ms[k].x, ms[k].y);
      if (Math.hypot(mx - sx, my - sy) <= SAMPLE_HIT_PX) return k;
    }
    return -1;
  }

  drawGrid(ax0, ax1, ay1, ay0) {
    const ctx = this.ctx;
    const step = niceStep(80 / this.scale, 1);
    const [wx0, wy1] = this.s2w(0, 0);
    const [wx1, wy0] = this.s2w(this.width, this.height);
    ctx.save();
    ctx.strokeStyle = "rgba(195, 194, 183, 0.55)";
    ctx.lineWidth = 1;
    ctx.fillStyle = MUTED;
    ctx.font = "11px system-ui, sans-serif";
    ctx.beginPath();
    for (let x = Math.ceil(wx0 / step) * step; x <= wx1; x += step) {
      const [sx] = this.w2s(x, 0);
      ctx.moveTo(Math.round(sx) + 0.5, 0);
      ctx.lineTo(Math.round(sx) + 0.5, this.height);
    }
    for (let y = Math.ceil(wy0 / step) * step; y <= wy1; y += step) {
      const [, sy] = this.w2s(0, y);
      ctx.moveTo(0, Math.round(sy) + 0.5);
      ctx.lineTo(this.width, Math.round(sy) + 0.5);
    }
    ctx.stroke();
    // rótulos en metros en los bordes de la vista
    ctx.textBaseline = "top";
    for (let x = Math.ceil(wx0 / step) * step; x <= wx1; x += step) {
      const [sx] = this.w2s(x, 0);
      ctx.fillText(`${+x.toFixed(6)}`, sx + 3, 3);
    }
    ctx.textBaseline = "bottom";
    for (let y = Math.ceil(wy0 / step) * step; y <= wy1; y += step) {
      const [, sy] = this.w2s(0, y);
      ctx.fillText(`${+y.toFixed(6)}`, 3, sy - 2);
    }
    ctx.restore();
  }

  tracePolygon(poly) {
    const ctx = this.ctx;
    ctx.beginPath();
    poly.forEach(([x, y], k) => {
      const [sx, sy] = this.w2s(x, y);
      if (k === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
    ctx.closePath();
  }

  drawBuildings(p, withCoverage) {
    const ctx = this.ctx;
    for (const b of p.buildings) {
      const mat = this.store.material(b.material);
      const selected = this.store.isSelected("building", b.id);
      this.tracePolygon(b.footprint);
      ctx.save();
      ctx.globalAlpha = withCoverage ? 0.92 : 0.8;
      ctx.fillStyle = mat ? mat.color : "#9a9a9a";
      ctx.fill();
      ctx.restore();
      ctx.lineWidth = selected ? 2 : 1;
      ctx.strokeStyle = selected ? SELECT : "rgba(11, 11, 11, 0.55)";
      ctx.stroke();
      if (this.store.view.labels) this.labelBuilding(b);
      if (selected) for (const [x, y] of b.footprint) this.drawHandle(x, y);
    }
  }

  labelBuilding(b) {
    const xs = b.footprint.map((q) => q[0]);
    const ys = b.footprint.map((q) => q[1]);
    const wpx = (Math.max(...xs) - Math.min(...xs)) * this.scale;
    const hpx = (Math.max(...ys) - Math.min(...ys)) * this.scale;
    if (wpx < 46 || hpx < 26) return;
    const [cx, cy] = centroid(b.footprint);
    const [sx, sy] = this.w2s(cx, cy);
    const ctx = this.ctx;
    ctx.save();
    ctx.textAlign = "center";
    ctx.font = "600 11px system-ui, sans-serif";
    haloText(ctx, b.name || b.id, sx, sy - 2, INK);
    ctx.font = "11px system-ui, sans-serif";
    haloText(ctx, `${fmt(b.height_m, 0)} m`, sx, sy + 11, INK2);
    ctx.restore();
  }

  drawWalls(p) {
    const ctx = this.ctx;
    for (const w of p.walls) {
      const mat = this.store.material(w.material);
      const selected = this.store.isSelected("wall", w.id);
      const [x0, y0] = this.w2s(...w.start);
      const [x1, y1] = this.w2s(...w.end);
      ctx.save();
      ctx.lineCap = "round";
      ctx.strokeStyle = selected ? SELECT : "rgba(11, 11, 11, 0.7)";
      ctx.lineWidth = selected ? 7 : 5;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      ctx.strokeStyle = mat ? mat.color : "#9a9a9a";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.restore();
      if (selected) { this.drawHandle(...w.start); this.drawHandle(...w.end); }
    }
  }

  drawTransmitters(p) {
    const ctx = this.ctx;
    p.transmitters.forEach((tx, slot) => {
      const [sx, sy] = this.w2s(tx.x, tx.y);
      const color = txColor(slot);
      const selected = this.store.isSelected("tx", tx.tx_id);
      if (tx.pattern === "sector") {
        const half = Math.min(179, tx.beamwidth_az_deg / 2) * Math.PI / 180;
        const az = -tx.azimuth_deg * Math.PI / 180;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.arc(sx, sy, 30, az - half, az + half);
        ctx.closePath();
        ctx.globalAlpha = tx.enabled ? 0.28 : 0.12;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = tx.enabled ? 0.9 : 0.4;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
      }
      ctx.save();
      ctx.beginPath();
      ctx.arc(sx, sy, selected ? 8 : 6.5, 0, 2 * Math.PI);
      ctx.fillStyle = tx.enabled ? color : SURFACE;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = SURFACE;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(sx, sy, (selected ? 8 : 6.5) + 1.5, 0, 2 * Math.PI);
      ctx.lineWidth = selected ? 2 : 1;
      ctx.strokeStyle = selected ? SELECT : tx.enabled ? INK : MUTED;
      ctx.stroke();
      if (this.store.view.labels) {
        ctx.font = "600 12px system-ui, sans-serif";
        ctx.textBaseline = "middle";
        haloText(ctx, tx.name || `Tx ${tx.tx_id}`, sx + 12, sy, tx.enabled ? INK : MUTED);
        ctx.font = "11px system-ui, sans-serif";
        haloText(ctx, `${fmt(tx.z, 0)} m · ${fmt(tx.tx_power_dbm, 0)} dBm`, sx + 12, sy + 13, INK2);
      }
      ctx.restore();
    });
  }

  drawHandle(x, y) {
    const [sx, sy] = this.w2s(x, y);
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = SELECT;
    ctx.lineWidth = 1.5;
    ctx.fillRect(sx - HANDLE_PX / 2, sy - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX);
    ctx.strokeRect(sx - HANDLE_PX / 2, sy - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX);
    ctx.restore();
  }

  drawRays() {
    const probe = this.store.probe;
    if (!probe || !probe.data) return;
    const ctx = this.ctx;
    const hover = this.store.view.hoverRay;
    const powers = [];
    for (const t of probe.data.transmitters) for (const r of t.rays) if (!r.blocked) powers.push(numberOf(r.power_dbm));
    const pmax = powers.length ? Math.max(...powers) : 0;
    ctx.save();
    ctx.lineCap = "round";
    probe.data.transmitters.forEach((t, ti) => {
      const color = txColor(this.store.txSlot(t.tx_id));
      t.rays.forEach((r, ri) => {
        if (!r.path) return;
        const isHover = hover && hover.tx === ti && hover.ray === ri;
        if (r.blocked && !this.store.view.showBlocked && !isHover) return;
        ctx.beginPath();
        r.path.forEach(([x, y], k) => {
          const [sx, sy] = this.w2s(x, y);
          if (k === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        });
        if (r.blocked) {
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = MUTED;
          ctx.globalAlpha = 0.8;
          ctx.lineWidth = isHover ? 2.5 : 1;
        } else {
          const rel = Math.max(0, Math.min(1, (numberOf(r.power_dbm) - (pmax - 40)) / 40));
          ctx.setLineDash([]);
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.35 + 0.65 * rel;
          ctx.lineWidth = isHover ? 4 : 1 + 2 * rel;
        }
        if (isHover) {
          ctx.save();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
          ctx.strokeStyle = SURFACE;
          ctx.lineWidth = 7;
          ctx.stroke();
          ctx.restore();
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = r.blocked ? MUTED : color;
        for (let k = 1; k + 1 < r.path.length; k++) {
          const [bx, by] = this.w2s(r.path[k][0], r.path[k][1]);
          const size = isHover ? 4 : 2.5;
          ctx.beginPath();
          const atEdge = r.interactions && r.interactions[k - 1] && r.interactions[k - 1][0] === "edge";
          if (atEdge) ctx.rect(bx - size - 0.5, by - size - 0.5, 2 * size + 1, 2 * size + 1);
          else ctx.arc(bx, by, size, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    });
    ctx.restore();
  }

  drawProbe() {
    const probe = this.store.probe;
    if (!probe) return;
    const [sx, sy] = this.w2s(probe.point[0], probe.point[1]);
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = SURFACE;
    ctx.lineWidth = 4;
    crosshair(ctx, sx, sy);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    crosshair(ctx, sx, sy);
    ctx.font = "600 11px system-ui, sans-serif";
    haloText(ctx, "Rx", sx + 9, sy - 9, INK);
    ctx.restore();
  }

  drawDraft() {
    const d = this.draft;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = SELECT;
    ctx.fillStyle = "rgba(28, 92, 171, 0.12)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    if (this.drag && this.drag.kind === "rect") {
      const { a, b } = this.drag;
      const [x0, y0] = this.w2s(Math.min(a[0], b[0]), Math.max(a[1], b[1]));
      const [x1, y1] = this.w2s(Math.max(a[0], b[0]), Math.min(a[1], b[1]));
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
      ctx.font = "11px system-ui, sans-serif";
      haloText(ctx, `${fmt(Math.abs(b[0] - a[0]), 1)} × ${fmt(Math.abs(b[1] - a[1]), 1)} m`, x1 + 6, y1 + 4, INK);
    }
    if (d && d.kind === "poly") {
      const pts = this.hover ? [...d.points, this.snapPoint(this.hover)] : d.points;
      ctx.beginPath();
      pts.forEach(([x, y], k) => {
        const [sx, sy] = this.w2s(x, y);
        if (k === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      if (pts.length > 2) { ctx.closePath(); ctx.fill(); }
      ctx.stroke();
      ctx.setLineDash([]);
      for (const [x, y] of d.points) this.drawHandle(x, y);
    }
    if (d && d.kind === "wall" && this.hover) {
      const [x0, y0] = this.w2s(...d.start);
      const end = this.snapPoint(this.hover);
      const [x1, y1] = this.w2s(...end);
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "11px system-ui, sans-serif";
      haloText(ctx, `${fmt(Math.hypot(end[0] - d.start[0], end[1] - d.start[1]), 1)} m`, x1 + 8, y1, INK);
    }
    ctx.restore();
  }

  drawMeasure() {
    const m = this.measure;
    if (!m) return;
    const ctx = this.ctx;
    const [x0, y0] = this.w2s(...m.a);
    const [x1, y1] = this.w2s(...m.b);
    ctx.save();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.setLineDash([]);
    const d = Math.hypot(m.b[0] - m.a[0], m.b[1] - m.a[1]);
    const az = Math.atan2(m.b[1] - m.a[1], m.b[0] - m.a[0]) * 180 / Math.PI;
    ctx.font = "600 12px system-ui, sans-serif";
    haloText(ctx, `${fmt(d, 1)} m · ${fmt(az, 0)}°`, (x0 + x1) / 2 + 8, (y0 + y1) / 2 - 8, INK);
    ctx.restore();
  }

  drawScaleBar() {
    const ctx = this.ctx;
    const step = niceStep(120 / this.scale, 1);
    const px = step * this.scale;
    const x = this.width - px - 22;
    const y = this.height - 16;
    ctx.save();
    ctx.fillStyle = "rgba(252, 252, 251, 0.85)";
    ctx.fillRect(x - 6, y - 16, px + 12, 26);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + px, y);
    ctx.moveTo(x, y - 4);
    ctx.lineTo(x, y + 4);
    ctx.moveTo(x + px, y - 4);
    ctx.lineTo(x + px, y + 4);
    ctx.stroke();
    ctx.fillStyle = INK;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${+step.toFixed(6)} m`, x + px / 2, y - 6);
    ctx.restore();
  }

  // ----- selección
  hitTest(sx, sy) {
    const p = this.store.project;
    const [x, y] = this.s2w(sx, sy);
    for (let k = p.transmitters.length - 1; k >= 0; k--) {
      const tx = p.transmitters[k];
      const [tx2, ty2] = this.w2s(tx.x, tx.y);
      if (Math.hypot(tx2 - sx, ty2 - sy) <= TX_HIT_PX) return { type: "tx", id: tx.tx_id };
    }
    const sample = this.sampleAt(sx, sy);
    if (sample >= 0) return { type: "measurement", id: sample };
    for (let k = p.walls.length - 1; k >= 0; k--) {
      const w = p.walls[k];
      if (distToSegment(x, y, w.start, w.end) * this.scale <= WALL_HIT_PX) return { type: "wall", id: w.id };
    }
    for (let k = p.buildings.length - 1; k >= 0; k--) {
      const b = p.buildings[k];
      if (pointInPolygon(x, y, b.footprint)) return { type: "building", id: b.id };
      const n = b.footprint.length;
      for (let i = 0; i < n; i++) {
        if (distToSegment(x, y, b.footprint[i], b.footprint[(i + 1) % n]) * this.scale <= 4) return { type: "building", id: b.id };
      }
    }
    // El plano solo se agarra cuando ya está seleccionado (si no, cualquier clic en el mapa lo tomaría)
    const plan = p.floor_plan;
    if (plan && this.store.isSelected("floorplan", undefined)) {
      const [rx, ry, rw, rh] = this.planRect(plan);
      if (sx >= rx && sx <= rx + rw && sy >= ry && sy <= ry + rh) return { type: "floorplan" };
    }
    return null;
  }

  handleAt(sx, sy) {
    const s = this.store.selection;
    if (!s) return null;
    const near = ([x, y]) => {
      const [hx, hy] = this.w2s(x, y);
      return Math.abs(hx - sx) <= HANDLE_PX && Math.abs(hy - sy) <= HANDLE_PX;
    };
    if (s.type === "building") {
      const b = this.store.building(s.id);
      const k = b ? b.footprint.findIndex(near) : -1;
      if (k >= 0) return { type: "building", id: s.id, vertex: k };
    }
    if (s.type === "wall") {
      const w = this.store.wall(s.id);
      if (w && near(w.start)) return { type: "wall", id: s.id, end: "start" };
      if (w && near(w.end)) return { type: "wall", id: s.id, end: "end" };
    }
    return null;
  }

  // ----- eventos
  onDown(e) {
    const s = this.eventPoint(e);
    const w = this.s2w(...s);
    this.canvas.setPointerCapture(e.pointerId);
    const tool = this.store.view.tool;
    if (e.button === 1 || e.button === 2 || tool === "pan" || this.spaceDown) {
      this.drag = { kind: "pan", last: s };
      this.canvas.style.cursor = "grabbing";
      return;
    }
    if (e.button !== 0) return;
    if (tool === "select") {
      const handle = this.handleAt(...s);
      if (handle) {
        this.drag = { kind: "vertex", handle, moved: false };
        return;
      }
      const hit = this.hitTest(...s);
      if (hit) {
        if (!this.store.isSelected(hit.type, hit.id)) this.store.select(hit);
        this.drag = { kind: "move", hit, start: w, moved: false, orig: JSON.parse(JSON.stringify(this.store.selectedObject())) };
      } else {
        this.store.select(null);
        this.drag = { kind: "pan", last: s };
      }
      return;
    }
    const p = this.snapPoint(w, e);
    if (tool === "tx") this.addTransmitter(p);
    else if (tool === "rect") this.drag = { kind: "rect", a: p, b: p };
    else if (tool === "poly") this.addPolygonVertex(p, s);
    else if (tool === "wall") this.addWallPoint(p);
    else if (tool === "probe") { this.drag = { kind: "probe" }; this.app.setProbe(p); }
    else if (tool === "measure") { this.measure = { a: p, b: p }; this.drag = { kind: "measure" }; }
    else if (tool === "sample") this.app.addMeasurementAt(p);
    else if (tool === "plan-scale") {
      const q = [round(w[0], 3), round(w[1], 3)];          // sin ajuste a la cuadrícula: hace falta precisión
      if (!this.planScale) this.planScale = { a: q };
      else {
        const a = this.planScale.a;
        this.planScale = null;
        this.app.finishPlanScale(a, q);
      }
    }
    this.request();
  }

  onMove(e) {
    const s = this.eventPoint(e);
    const w = this.s2w(...s);
    this.hover = w;
    this.app.showCursor(w);
    const d = this.drag;
    if (!d) {
      this.updateCursor(s);
      this.updateTooltip(s, w);
      if (this.draft) this.request();
      return;
    }
    this.tooltip.hidden = true;
    if (d.kind === "pan") {
      this.ox += s[0] - d.last[0];
      this.oy += s[1] - d.last[1];
      d.last = s;
      this.app.view3dSync?.();
    } else if (d.kind === "move") {
      const p0 = this.snapPoint(d.start, e);
      const p1 = this.snapPoint(w, e);
      const dx = p1[0] - p0[0];
      const dy = p1[1] - p0[1];
      if (!d.moved && dx === 0 && dy === 0) return;
      const first = !d.moved;
      d.moved = true;
      this.store.change((proj) => moveObject(proj, d.hit, d.orig, dx, dy), { checkpoint: first, live: true });
    } else if (d.kind === "vertex") {
      const p = this.snapPoint(w, e);
      const first = !d.moved;
      d.moved = true;
      this.store.change((proj) => {
        const h = d.handle;
        if (h.type === "building") proj.buildings.find((b) => b.id === h.id).footprint[h.vertex] = p;
        else proj.walls.find((x) => x.id === h.id)[h.end] = p;
      }, { checkpoint: first, live: true });
    } else if (d.kind === "rect") {
      d.b = this.snapPoint(w, e);
    } else if (d.kind === "measure") {
      this.measure.b = this.snapPoint(w, e);
    } else if (d.kind === "probe") {
      this.app.setProbe(this.snapPoint(w, e));
    }
    this.request();
  }

  onUp(e) {
    const d = this.drag;
    this.drag = null;
    this.canvas.style.cursor = "";
    if (!d) return;
    if ((d.kind === "move" || d.kind === "vertex") && d.moved) {
      if (d.kind === "vertex" && d.handle.type === "building") {
        const b = this.store.building(d.handle.id);
        if (b && !isSimplePolygon(b.footprint)) {
          this.store.undo();
          this.store.redoStack = [];
          this.app.notify("error", "Ese vértice haría que el polígono se cruzara consigo mismo: se deshizo el cambio.");
          return;
        }
      }
      this.store.endLive();
    }
    if (d.kind === "rect") {
      const { a, b } = d;
      if (Math.abs(b[0] - a[0]) >= 0.5 && Math.abs(b[1] - a[1]) >= 0.5) {
        const x0 = Math.min(a[0], b[0]);
        const x1 = Math.max(a[0], b[0]);
        const y0 = Math.min(a[1], b[1]);
        const y1 = Math.max(a[1], b[1]);
        this.addBuilding([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
      }
    }
    this.request();
  }

  onLeave() {
    this.hover = null;
    this.tooltip.hidden = true;
    this.app.showCursor(null);
    if (this.draft) this.request();
  }

  onDoubleClick(e) {
    if (this.store.view.tool === "poly") this.finishPolygon();
    else if (this.store.view.tool === "select") {
      const hit = this.hitTest(...this.eventPoint(e));
      if (hit) this.app.focusProperties();
    }
  }

  onWheel(e) {
    e.preventDefault();
    const [sx, sy] = this.eventPoint(e);
    const [wx, wy] = this.s2w(sx, sy);
    const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
    this.scale = Math.max(1e-3, Math.min(5000, this.scale * factor));
    this.ox = sx - wx * this.scale;
    this.oy = sy + wy * this.scale;
    this.request();
  }

  updateCursor(s) {
    const tool = this.store.view.tool;
    let cursor = "default";
    if (tool === "pan") cursor = "grab";
    else if (tool === "select") cursor = this.handleAt(...s) ? "move" : this.hitTest(...s) ? "pointer" : "default";
    else cursor = "crosshair";
    this.canvas.style.cursor = cursor;
  }

  updateTooltip(s, w) {
    const layer = currentLayer(this.store);
    const tip = this.tooltip;
    const tool = this.store.view.tool;
    const sample = tool === "select" || tool === "probe" || tool === "pan" || tool === "sample" ? this.sampleAt(...s) : -1;
    if (sample >= 0) { this.sampleTooltip(s, sample); return; }
    if (!layer || this.store.view.tool !== "select" && this.store.view.tool !== "probe" && this.store.view.tool !== "pan") {
      tip.hidden = true;
      return;
    }
    const k = pixelAt(layer.grid, w[0], w[1]);
    if (k < 0) { tip.hidden = true; return; }
    tip.replaceChildren();
    const value = layer.values[k];
    const strong = document.createElement("div");
    strong.className = "tt-value";
    if (layer.def.kind === "cat") strong.textContent = value >= 0 ? this.store.txName(value) : "Sin servidor";
    else strong.textContent = isFinite(value) ? `${fmt(value, layer.def.digits)} ${layer.def.unit}` : "Sin datos";
    const label = document.createElement("div");
    label.className = "tt-label";
    label.textContent = layer.label;
    tip.append(strong, label);
    const r = this.store.result;
    if (r.perTx && r.tx_ids.length > 1 && (layer.metric === "rx_power" || layer.metric === "best_server")) {
      r.tx_ids.forEach((id, pos) => {
        const row = document.createElement("div");
        row.className = "tt-row";
        const key = document.createElement("span");
        key.className = "tt-key";
        key.style.background = txColor(this.store.txSlot(id));
        const text = document.createElement("span");
        const pv = r.perTx[pos][k];
        text.textContent = `${this.store.txName(id)}: ${isFinite(pv) ? fmt(pv, 1) + " dBm" : "—"}`;
        row.append(key, text);
        tip.append(row);
      });
    }
    const coords = document.createElement("div");
    coords.className = "tt-label";
    coords.textContent = `x ${fmt(w[0], 1)} m · y ${fmt(w[1], 1)} m`;
    tip.append(coords);
    tip.hidden = false;
    const box = this.canvas.parentElement.getBoundingClientRect();
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    tip.style.left = `${Math.min(s[0] + 16, box.width - tw - 6)}px`;
    tip.style.top = `${Math.min(s[1] + 16, box.height - th - 6)}px`;
  }

  sampleTooltip(s, k) {
    const tip = this.tooltip;
    const m = this.store.project.measurements[k];
    const c = this.store.comparison;
    const pt = c && c.revision === this.store.measurementRevision() ? c.data.points[k] : null;
    const line = (cls, text) => { const d = document.createElement("div"); d.className = cls; d.textContent = text; return d; };
    tip.replaceChildren(line("tt-value", `${fmt(m.rssi_dbm, 1)} dBm medidos`),
      line("tt-label", `Medición ${k + 1}${m.label ? ` · ${m.label}` : ""} · ${m.tx_id === null || m.tx_id === undefined ? "el Tx más fuerte" : this.store.txName(m.tx_id)}`));
    if (pt && pt.status === "ok") {
      tip.append(line("tt-label", `Predicho ${fmt(pt.predicted_dbm, 1)} dBm · error ${pt.error_db >= 0 ? "+" : "−"}${fmt(Math.abs(pt.error_db), 1)} dB`));
    } else if (pt) tip.append(line("tt-label", `Sin comparación: ${pt.status}`));
    else tip.append(line("tt-label", "Ejecuta la predicción (F5) para compararla"));
    tip.hidden = false;
    const box = this.canvas.parentElement.getBoundingClientRect();
    tip.style.left = `${Math.min(s[0] + 16, box.width - tip.offsetWidth - 6)}px`;
    tip.style.top = `${Math.min(s[1] + 16, box.height - tip.offsetHeight - 6)}px`;
  }

  // ----- creación de objetos
  addTransmitter(p) {
    const proj = this.store.project;
    const last = proj.transmitters[proj.transmitters.length - 1];
    const id = nextTxId(proj.transmitters);
    const tx = last ? { ...JSON.parse(JSON.stringify(last)), tx_id: id, name: `Tx ${id}`, x: p[0], y: p[1], enabled: true }
      : { tx_id: id, name: `Tx ${id}`, x: p[0], y: p[1], z: 25, tx_power_dbm: 43, frequency_hz: 2.4e9, antenna_gain_dbi: 14,
        pattern: "isotropic", azimuth_deg: 0, beamwidth_az_deg: 65, beamwidth_el_deg: 65, downtilt_deg: 0, enabled: true };
    this.store.change((pr) => pr.transmitters.push(tx));
    this.store.select({ type: "tx", id });
    this.app.setTool("select");
  }

  addBuilding(footprint) {
    if (!isSimplePolygon(footprint)) {
      this.app.notify("error", "El polígono no es válido (se cruza consigo mismo o no tiene área).");
      return;
    }
    const proj = this.store.project;
    const last = proj.buildings[proj.buildings.length - 1];
    const id = nextId("b", proj.buildings);
    const material = last ? last.material : (proj.materials[0] || {}).id;
    const height = last ? last.height_m : 15;
    this.store.change((pr) => pr.buildings.push({ id, name: `Edificio ${pr.buildings.length + 1}`, footprint, height_m: height,
      material, roof: true }));
    this.store.select({ type: "building", id });
  }

  addPolygonVertex(p, s) {
    if (!this.draft || this.draft.kind !== "poly") this.draft = { kind: "poly", points: [] };
    const pts = this.draft.points;
    if (pts.length >= 3) {
      const [fx, fy] = this.w2s(...pts[0]);
      if (Math.hypot(fx - s[0], fy - s[1]) <= HANDLE_PX + 2) { this.finishPolygon(); return; }
    }
    const last = pts[pts.length - 1];
    if (!last || last[0] !== p[0] || last[1] !== p[1]) pts.push(p);
    this.app.updateHint();
  }

  finishPolygon() {
    const d = this.draft;
    if (!d || d.kind !== "poly") return;
    // un doble clic añade el último vértice dos veces
    const pts = d.points.filter((q, k) => k === 0 || q[0] !== d.points[k - 1][0] || q[1] !== d.points[k - 1][1]);
    this.draft = null;
    if (pts.length >= 3) this.addBuilding(pts);
    else this.app.notify("warn", "Un edificio necesita al menos 3 vértices.");
    this.app.updateHint();
    this.request();
  }

  addWallPoint(p) {
    if (!this.draft || this.draft.kind !== "wall") {
      this.draft = { kind: "wall", start: p };
      this.app.updateHint();
      return;
    }
    const start = this.draft.start;
    this.draft = null;
    if (start[0] === p[0] && start[1] === p[1]) return;
    const proj = this.store.project;
    const last = proj.walls[proj.walls.length - 1];
    const id = nextId("w", proj.walls);
    const material = last ? last.material : (proj.materials.find((m) => m.id === "ladrillo") || proj.materials[0] || {}).id;
    this.store.change((pr) => pr.walls.push({ id, name: `Muro ${pr.walls.length + 1}`, start, end: p,
      z_min_m: last ? last.z_min_m : 0, z_max_m: last ? last.z_max_m : 3, material }));
    this.store.select({ type: "wall", id });
    this.app.updateHint();
  }

  cancelDraft() {
    this.draft = null;
    this.measure = null;
    this.planScale = null;
    this.drag = null;
    this.app.updateHint();
    this.request();
  }

  toPng() {
    return this.canvas.toDataURL("image/png");
  }
}

function moveObject(proj, hit, orig, dx, dy) {
  if (hit.type === "tx") {
    const tx = proj.transmitters.find((t) => t.tx_id === hit.id);
    tx.x = round(orig.x + dx, 3);
    tx.y = round(orig.y + dy, 3);
  } else if (hit.type === "building") {
    const b = proj.buildings.find((x) => x.id === hit.id);
    b.footprint = orig.footprint.map(([x, y]) => [round(x + dx, 3), round(y + dy, 3)]);
  } else if (hit.type === "wall") {
    const w = proj.walls.find((x) => x.id === hit.id);
    w.start = [round(orig.start[0] + dx, 3), round(orig.start[1] + dy, 3)];
    w.end = [round(orig.end[0] + dx, 3), round(orig.end[1] + dy, 3)];
  } else if (hit.type === "measurement") {
    const m = proj.measurements[hit.id];
    m.x = round(orig.x + dx, 3);
    m.y = round(orig.y + dy, 3);
  } else if (hit.type === "floorplan") {
    proj.floor_plan.x_m = round(orig.x_m + dx, 4);
    proj.floor_plan.y_m = round(orig.y_m + dy, 4);
  }
}

function haloText(ctx, text, x, y, color) {
  ctx.save();
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(252, 252, 251, 0.92)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function crosshair(ctx, x, y) {
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, 2 * Math.PI);
  ctx.moveTo(x - 11, y);
  ctx.lineTo(x - 3, y);
  ctx.moveTo(x + 3, y);
  ctx.lineTo(x + 11, y);
  ctx.moveTo(x, y - 11);
  ctx.lineTo(x, y - 3);
  ctx.moveTo(x, y + 3);
  ctx.lineTo(x, y + 11);
  ctx.stroke();
}

export function numberOf(value) {
  if (value === "-inf") return -Infinity;
  if (value === "inf") return Infinity;
  return value === null ? NaN : Number(value);
}

function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}
