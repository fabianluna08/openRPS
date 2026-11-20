// Cálculos pequeños que la interfaz muestra al editar (1.9.0): pérdida de los muros, potencia en mW y
// modelos de antena.  Son los mismos que open_rps.project (slab_loss_db, Wall.penetration_loss_db,
// ANTENNA_PRESETS); el motor de Python es el que simula.

const EPS0 = 8.8541878128e-12;
const C0 = 299792458;

// Modelos de antena: diagrama, ganancia y haces a −3 dB (igual que open_rps.project.ANTENNA_PRESETS)
export const ANTENNA_PRESETS = {
  omni_2_2: { label: "Omnidireccional 2.2 dBi", pattern: "isotropic", antenna_gain_dbi: 2.2 },
  omni_5: { label: "Omnidireccional 5 dBi", pattern: "isotropic", antenna_gain_dbi: 5.0 },
  yagi_13: { label: "Yagi 13 dBi (40° × 36°)", pattern: "sector", antenna_gain_dbi: 13.0, beamwidth_az_deg: 40.0, beamwidth_el_deg: 36.0 },
  sector_6: { label: "Sector 6 dBi (120° × 60°)", pattern: "sector", antenna_gain_dbi: 6.0, beamwidth_az_deg: 120.0, beamwidth_el_deg: 60.0 },
  panel_8: { label: "Panel 8 dBi (65° × 65°)", pattern: "sector", antenna_gain_dbi: 8.0, beamwidth_az_deg: 65.0, beamwidth_el_deg: 65.0 },
};

// Modelo de antena de un transmisor, o "custom" si sus valores no son los de ninguno
export function antennaPresetOf(tx) {
  for (const [id, preset] of Object.entries(ANTENNA_PRESETS)) {
    if (tx.pattern !== preset.pattern || Math.abs(tx.antenna_gain_dbi - preset.antenna_gain_dbi) > 1e-9) continue;
    if (preset.pattern === "sector" && (Math.abs(tx.beamwidth_az_deg - preset.beamwidth_az_deg) > 1e-9
      || Math.abs(tx.beamwidth_el_deg - preset.beamwidth_el_deg) > 1e-9)) continue;
    return id;
  }
  return "custom";
}

export function applyAntennaPreset(tx, id) {
  const preset = ANTENNA_PRESETS[id];
  if (!preset) return;
  for (const [k, v] of Object.entries(preset)) if (k !== "label") tx[k] = v;
}

export const dbmToMw = (dbm) => 10 ** (dbm / 10);
export const mwToDbm = (mw) => 10 * Math.log10(mw);

// Frecuencia a la que se evalúan los materiales: la de referencia del proyecto o la del primer Tx activo
export function referenceFrequency(project) {
  const ref = project.settings && project.settings.reference_frequency_hz;
  if (ref) return ref;
  const tx = project.transmitters.find((t) => t.enabled);
  return tx ? tx.frequency_hz : 2.4e9;
}

export function conductivity(material, f) {
  return material.sigma_c * (f / 1e9) ** material.sigma_d;
}

// Pérdida de una losa de espesor d a incidencia normal (ITU-R P.2040, una capa, con las reflexiones internas)
export function slabLossDb(epsR, sigma, f, d) {
  if (!(d > 0)) return 0;
  const im = -sigma / (2 * Math.PI * f * EPS0);
  const mod = Math.hypot(epsR, im);
  const nr = Math.sqrt((mod + epsR) / 2);
  const ni = -Math.sqrt(Math.max(0, (mod - epsR) / 2));          // Im n <= 0: la onda se atenúa
  const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const div = (a, b) => { const den = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den]; };
  const r = div([1 - nr, -ni], [1 + nr, ni]);
  const r2 = mul(r, r);
  const k = (2 * Math.PI * f * d) / C0;
  const mag = Math.exp(k * ni);
  const e = [mag * Math.cos(k * nr), -mag * Math.sin(k * nr)];   // exp(-j q)
  const num = mul([1 - r2[0], -r2[1]], e);
  const r2e2 = mul(r2, mul(e, e));
  const t = div(num, [1 - r2e2[0], -r2e2[1]]);
  return Math.max(0, -20 * Math.log10(Math.hypot(t[0], t[1])));
}

// Pérdida por cruce de un material: null si no tapa, Infinity si es opaco
export function materialLossDb(material, f) {
  if (material.penetration === "none") return null;
  if (material.penetration === "opaque") return Infinity;
  if (material.penetration === "fixed") return material.loss_db;
  return Math.max(0, material.loss_a_db + (material.loss_b_db_per_ghz * f) / 1e9);
}

// Pérdida por cruce de un muro: con espesor, la de la losa (si el material tiene una pérdida finita)
export function wallLossDb(wall, material, f) {
  const thickness = wall.thickness_m;
  if (thickness === null || thickness === undefined || !["fixed", "frequency"].includes(material.penetration)) {
    return materialLossDb(material, f);
  }
  return slabLossDb(material.eps_r, conductivity(material, f), f, thickness);
}
