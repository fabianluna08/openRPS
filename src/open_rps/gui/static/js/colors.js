// Colores, escalas y formato.  Las rampas y la paleta son las mismas que usan los SVG de OpenRPS
// (open_rps/visualizer.py), para que un mapa se vea igual en la interfaz y en la exportación.

// Rampa secuencial de un solo tono (claro -> oscuro): magnitudes
export const BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

// Okabe-Ito: un color fijo por transmisor (orden en que se añadieron), nunca reciclado.
// A partir del noveno, todos comparten el gris "Otros" y se distinguen por su rótulo en el mapa.
export const OKABE_ITO = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#56b4e9", "#cc79a7", "#f0e442", "#000000"];
export const OTHER = "#898781";
export const NODATA = "#e1e0d9";
export const INK = "#0b0b0b";
export const INK2 = "#52514e";
export const MUTED = "#898781";
export const AXIS = "#c3c2b7";
export const SURFACE = "#fcfcfb";

export function txColor(index) {
  return index >= 0 && index < OKABE_ITO.length ? OKABE_ITO[index] : OTHER;
}

export function hexToRgb(hex) {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

// Tabla de 256 colores interpolados linealmente entre los pasos de la rampa (igual que _ramp_color)
export function rampLUT(ramp = BLUE_RAMP, n = 256) {
  const stops = ramp.map(hexToRgb);
  const lut = new Uint8ClampedArray(n * 3);
  for (let k = 0; k < n; k++) {
    const pos = (k / (n - 1)) * (stops.length - 1);
    const lo = Math.min(Math.floor(pos), stops.length - 2);
    const frac = pos - lo;
    for (let c = 0; c < 3; c++) lut[3 * k + c] = Math.round(stops[lo][c] + (stops[lo + 1][c] - stops[lo][c]) * frac);
  }
  return lut;
}

// Paleta de los mapas de calor Wi-Fi (1.9.0), de débil a fuerte: azul marino, azul, cian, verde, amarillo,
// naranja y rojo oscuro.  Es la convención de las herramientas de site survey (rojo = señal fuerte), útil para
// comparar con ellas; no es perceptualmente uniforme, así que la rampa azul sigue siendo la de por defecto.
export const WIFI_HEAT_RAMP = ["#10163f", "#1b2c8f", "#1f4fd1", "#2a8be6", "#3cc6e8", "#2fb59a", "#1f8f4a", "#3fae3a",
  "#8cc63f", "#e3e12c", "#f7b52a", "#f07d22", "#e2453a", "#c8102e", "#8b0a1a"];

export const PALETTES = {
  blue: { label: "Azul (OpenRPS)", ramp: BLUE_RAMP },
  heat: { label: "Mapa de calor Wi-Fi (rojo = fuerte)", ramp: WIFI_HEAT_RAMP },
};

export function paletteRamp(view) {
  return (PALETTES[view && view.palette] || PALETTES.blue).ramp;
}

// 256 colores de la paleta: interpolados, o en bandas sólidas (un color por paso de la rampa)
export function paletteLUT(ramp, banded) {
  if (!banded) return rampLUT(ramp);
  const stops = ramp.map(hexToRgb);
  const lut = new Uint8ClampedArray(256 * 3);
  for (let k = 0; k < 256; k++) {
    const band = Math.min(stops.length - 1, Math.floor((k / 256) * stops.length));
    for (let c = 0; c < 3; c++) lut[3 * k + c] = stops[band][c];
  }
  return lut;
}

export function rampColor(norm, ramp = BLUE_RAMP) {
  const stops = ramp.map(hexToRgb);
  const pos = Math.min(1, Math.max(0, norm)) * (stops.length - 1);
  const lo = Math.min(Math.floor(pos), stops.length - 2);
  const frac = pos - lo;
  const rgb = stops[lo].map((v, c) => Math.round(v + (stops[lo + 1][c] - v) * frac));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

// Métricas que se pueden dibujar.  kind: "seq" (magnitud, rampa) o "cat" (identidad del transmisor)
export const METRICS = {
  rx_power: { label: "Potencia recibida", short: "P_r", unit: "dBm", kind: "seq", digits: 1 },
  path_loss: { label: "Pérdida de trayecto", short: "PL", unit: "dB", kind: "seq", digits: 1 },
  best_server: { label: "Mejor servidor", short: "Servidor", unit: "", kind: "cat" },
  sinr: { label: "SINR", short: "SINR", unit: "dB", kind: "seq", digits: 1 },
  interference: { label: "Interferencia + ruido", short: "I + N", unit: "dBm", kind: "seq", digits: 1 },
  delay_spread: { label: "Dispersión de retardo RMS", short: "σ_τ", unit: "ns", kind: "seq", digits: 1, zeroBased: true },
  angular_spread: { label: "Dispersión angular (salida)", short: "σ_φ", unit: "°", kind: "seq", digits: 1, zeroBased: true },
  wifi_rate: { label: "Wi-Fi: tasa de datos (PHY)", short: "Tasa", unit: "Mbit/s", kind: "seq", digits: 0, zeroBased: true },
  wifi_mcs: { label: "Wi-Fi: MCS", short: "MCS", unit: "", kind: "seq", digits: 0, zeroBased: true, integer: true,
    negativeIsNone: true, noData: "sin conexión o sin datos", noDataLegend: "Sin conexión (SINR bajo el de MCS 0), sin señal o interior bloqueado" },
  wifi_contention: { label: "Wi-Fi: AP que comparten el canal", short: "AP", unit: "", kind: "seq", digits: 0, zeroBased: true, integer: true },
};

export function niceStep(span, target = 6) {
  if (!(span > 0) || !isFinite(span)) return 1;
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  let best = mag;
  for (const m of [1, 2, 5, 10]) if (Math.abs(span / (m * mag) - target) < Math.abs(span / best - target)) best = m * mag;
  return best;
}

export function niceTicks(lo, hi, target = 6) {
  const span = hi - lo;
  if (!(span > 0) || !isFinite(span)) return [lo];
  const step = niceStep(span, target);
  const ticks = [];
  for (let n = Math.ceil(lo / step - 1e-9); n * step <= hi + 1e-9 * span; n++) ticks.push(+(n * step).toFixed(10));
  return ticks;
}

// Rango automático: percentiles 1 y 99 de los valores finitos, redondeados a pasos "bonitos"
export function autoRange(values, zeroBased = false) {
  const finite = [];
  for (let k = 0; k < values.length; k++) if (isFinite(values[k])) finite.push(values[k]);
  if (!finite.length) return [0, 1];
  finite.sort((a, b) => a - b);
  let lo = finite[Math.floor(0.01 * (finite.length - 1))];
  let hi = finite[Math.ceil(0.99 * (finite.length - 1))];
  if (zeroBased) lo = 0;
  if (hi - lo < 1e-9) {
    if (zeroBased) hi = lo + 1;
    else { lo -= 1; hi += 1; }
  }
  const step = niceStep(hi - lo, 8);
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

export function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value === Infinity || value === "inf") return "∞";
  if (value === -Infinity || value === "-inf") return "−∞";
  const text = Number(value).toFixed(digits);
  return (text === "-0" || /^-0\.0*$/.test(text) ? text.slice(1) : text).replace("-", "−");
}

export function fmtFreq(hz) {
  if (!(hz > 0)) return "—";
  return hz >= 1e9 ? `${+(hz / 1e9).toFixed(3)} GHz` : `${+(hz / 1e6).toFixed(1)} MHz`;
}

// Valores que el servidor manda como texto ("inf", "-inf") o null
export function num(value) {
  if (value === "inf") return Infinity;
  if (value === "-inf") return -Infinity;
  if (value === null || value === undefined) return NaN;
  return Number(value);
}
