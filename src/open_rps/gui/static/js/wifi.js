// Canales Wi-Fi: las mismas tablas que open_rps.wifi (el servidor valida y calcula; aquí solo se arman los menús)
export const WIFI_BANDS = [["2.4", "2.4 GHz"], ["5", "5 GHz"], ["6", "6 GHz"]];
export const WIFI_STANDARDS = [["n", "802.11n (Wi-Fi 4)"], ["ac", "802.11ac (Wi-Fi 5)"], ["ax", "802.11ax (Wi-Fi 6/6E)"]];
export const GUARD_INTERVALS = { n: [0.8, 0.4], ac: [0.8, 0.4], ax: [0.8, 1.6, 3.2] };
const BASE_MHZ = { "2.4": 2407, 5: 5000, 6: 5950 };
const CH5 = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
  149, 153, 157, 161, 165, 169, 173, 177];
const BLOCKS5 = {
  40: [[36, 40, 38], [44, 48, 46], [52, 56, 54], [60, 64, 62], [100, 104, 102], [108, 112, 110], [116, 120, 118],
    [124, 128, 126], [132, 136, 134], [140, 144, 142], [149, 153, 151], [157, 161, 159], [165, 169, 167], [173, 177, 175]],
  80: [[36, 48, 42], [52, 64, 58], [100, 112, 106], [116, 128, 122], [132, 144, 138], [149, 161, 155], [165, 177, 171]],
  160: [[36, 64, 50], [100, 128, 114], [149, 177, 163]],
};
const range4 = (a, b) => { const out = []; for (let c = a; c <= b; c += 4) out.push(c); return out; };
export const isDfs = (band, c) => String(band) === "5" && c >= 52 && c <= 144;

export function widthsFor(band, standard) {
  if (String(band) === "2.4") return [20];
  return standard === "n" ? [20, 40] : [20, 40, 80, 160];
}

// Bloques de canales que ocupa cada ancho: [{channels, centre}]
export function blocks(band, width) {
  band = String(band);
  if (width === 20) {
    const list = band === "2.4" ? Array.from({ length: 13 }, (_, k) => k + 1) : band === "5" ? CH5 : range4(1, 233);
    return list.map((c) => ({ channels: [c], centre: c }));
  }
  if (band === "5") return (BLOCKS5[width] || []).map(([lo, hi, centre]) => ({ channels: range4(lo, hi), centre }));
  if (band === "6") {
    const n = width / 20;
    const out = [];
    for (let first = 1; first <= 233; first += 4 * n) {
      const last = first + 4 * (n - 1);
      if (last <= 233) out.push({ channels: range4(first, last), centre: first + 2 * (n - 1) });
    }
    return out;
  }
  return [];
}

export function centreMHz(band, channel, width) {
  const b = blocks(band, width).find((x) => x.channels.includes(channel));
  return b ? BASE_MHZ[String(band)] + 5 * b.centre : null;
}

export function channelOptions(band, width) {
  const out = [];
  for (const b of blocks(band, width)) {
    const f = BASE_MHZ[String(band)] + 5 * b.centre;
    for (const c of b.channels) {
      const span = b.channels.length > 1 ? ` · bloque ${b.channels[0]}–${b.channels[b.channels.length - 1]}` : "";
      out.push([c, `${c}${span} · ${f} MHz${isDfs(band, c) ? " · DFS" : ""}`]);
    }
  }
  return out;
}

// Canal por omisión de una banda: el primero de los habituales
export function defaultChannel(band, width) {
  const preferred = { "2.4": [1, 6, 11], 5: [36, 149, 44], 6: [5, 37, 1] }[String(band)] || [];
  const valid = blocks(band, width).flatMap((b) => b.channels);
  return preferred.find((c) => valid.includes(c)) ?? valid[0] ?? null;
}

// Deja el canal de un transmisor consistente con su banda y su ancho, y su frecuencia en el centro del canal
export function normalizeWifi(t, standard) {
  if (!t.wifi_band) { t.wifi_band = null; t.wifi_channel = null; return t; }
  const widths = widthsFor(t.wifi_band, standard);
  if (!widths.includes(t.wifi_width_mhz)) t.wifi_width_mhz = widths.includes(20) ? 20 : widths[0];
  const valid = blocks(t.wifi_band, t.wifi_width_mhz).flatMap((b) => b.channels);
  // Si el canal sigue existiendo con el nuevo ancho se conserva (ocupa el bloque que lo contiene); si no, el habitual
  if (!valid.includes(t.wifi_channel)) t.wifi_channel = defaultChannel(t.wifi_band, t.wifi_width_mhz);
  if (!t.wifi_streams) t.wifi_streams = 2;
  t.frequency_hz = centreMHz(t.wifi_band, t.wifi_channel, t.wifi_width_mhz) * 1e6;
  return t;
}

export function channelLabel(t) {
  if (!t || t.wifi_channel === null || t.wifi_channel === undefined) return "";
  return `canal ${t.wifi_channel}${t.wifi_width_mhz > 20 ? ` (${t.wifi_width_mhz} MHz)` : ""}`;
}
