// Panel del proyecto (árbol) y panel de propiedades (formularios de cada elemento).
import { fmt, fmtFreq, txColor } from "./colors.js";
import { GUARD_INTERVALS, WIFI_BANDS, WIFI_STANDARDS, centreMHz, channelOptions, normalizeWifi, widthsFor } from "./wifi.js";
import { centroid, isSimplePolygon, round } from "./geometry.js";
import { nextId, nextTxId } from "./state.js";
import { ANTENNA_PRESETS, antennaPresetOf, applyAntennaPreset, dbmToMw, mwToDbm, referenceFrequency, wallLossDb } from "./physics.js";

const MODE_LABELS = {
  raytracing: "Ray tracing (multitrayecto)",
  fspl: "Espacio libre (FSPL)",
  tworay: "Two-Ray (reflexión en el suelo)",
  logdistance: "Log-distancia",
  hata: "Okumura-Hata / COST-231",
  multiwall: "Interiores: log-distancia + muros (multimuro)",
};
const HATA_LABELS = { urban_large: "Urbano (ciudad grande)", urban_medium: "Urbano (ciudad mediana)", suburban: "Suburbano", open: "Rural abierto" };
const PENETRATION_LABELS = { none: "No obstruye (solo refleja)", fixed: "Pérdida fija", frequency: "Pérdida lineal con f (3GPP)", opaque: "Opaco" };
const REFLECTION_LABELS = { fresnel: "Fresnel (εr, σ)", constant: "Coeficiente constante" };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

// ------------------------------------------------------------------ árbol
export function renderTree(container, store, app) {
  const p = store.project;
  container.replaceChildren();
  if (!p) return;
  const head = (label, sel, count, onAdd, addTitle) => {
    const row = el("div", { class: "tree-head" + (sel && store.isSelected(sel.type, sel.id) ? " selected" : "") },
      el("span", { text: label }), count === null ? null : el("span", { class: "count", text: `(${count})` }));
    if (sel) row.addEventListener("click", () => store.select(sel));
    if (onAdd) {
      const add = el("button", { class: "add", title: addTitle, text: "+", onclick: (e) => { e.stopPropagation(); onAdd(); } });
      row.append(add);
    }
    return row;
  };
  const item = (sel, name, meta, mark, disabled = false) => {
    const row = el("div", { class: "tree-item" + (store.isSelected(sel.type, sel.id) ? " selected" : "") + (disabled ? " disabled" : ""),
      onclick: () => store.select(sel), ondblclick: () => app.zoomTo(sel) }, mark, el("span", { class: "name", text: name }),
    el("span", { class: "meta", text: meta }));
    return row;
  };
  const section = (...rows) => el("div", { class: "tree-section" }, ...rows);

  container.append(section(
    head(p.name || "Proyecto", { type: "project" }, null),
    item({ type: "area" }, "Área de predicción", `${fmt(p.area.resolution_m, 2).replace(/\.?0+$/, "")} m`, el("span", { class: "swatch", style: "background:#f4f3ef" })),
    item({ type: "settings" }, "Configuración", p.settings.mode === "multiwall" ? "Multimuro" : MODE_LABELS[p.settings.mode].split(" (")[0], el("span", { class: "swatch", style: "background:#1c5cab" })),
  ));
  container.append(section(
    head("Transmisores", null, p.transmitters.length, () => app.addTransmitterCentered(), "Añadir transmisor (o usa la herramienta Transmisor)"),
    ...p.transmitters.map((tx, slot) => item({ type: "tx", id: tx.tx_id }, tx.name || `Tx ${tx.tx_id}`,
      `${fmtFreq(tx.frequency_hz)}`, el("span", { class: "dot", style: `background:${txColor(slot)}` }), !tx.enabled)),
  ));
  container.append(section(
    head("Edificios", null, p.buildings.length, () => app.setTool("rect"), "Dibujar un edificio rectangular"),
    ...p.buildings.map((b) => item({ type: "building", id: b.id }, b.name || b.id, `${fmt(b.height_m, 0)} m`,
      el("span", { class: "swatch", style: `background:${(store.material(b.material) || {}).color || "#999"}` }))),
  ));
  container.append(section(
    head("Muros", null, p.walls.length, () => app.setTool("wall"), "Dibujar un muro"),
    ...p.walls.map((w) => item({ type: "wall", id: w.id }, w.name || w.id,
      `${fmt(Math.hypot(w.end[0] - w.start[0], w.end[1] - w.start[1]), 1)} m`,
      el("span", { class: "swatch", style: `background:${(store.material(w.material) || {}).color || "#999"}` }))),
  ));
  const plan = p.floor_plan;
  container.append(section(
    head("Plano de fondo", null, null, () => app.pickFloorPlan(), plan ? "Cambiar la imagen del plano" : "Cargar la imagen de un plano (PNG, JPEG…)"),
    plan ? item({ type: "floorplan" }, plan.name || "Plano", `${fmt(plan.width_m, 1)} × ${fmt(plan.width_m * plan.pixel_height / plan.pixel_width, 1)} m`,
      el("span", { class: "swatch", style: "background:repeating-linear-gradient(45deg,#fcfcfb 0 2px,#898781 2px 3px)" }), !plan.visible)
      : el("div", { class: "tree-empty", text: "Ninguno: + para cargar una imagen" }),
  ));
  const nMeas = (p.measurements || []).length;
  container.append(section(
    head("Mediciones", { type: "measurements" }, nMeas, () => app.pickMeasurementsCsv(), "Importar mediciones desde un CSV"),
    nMeas ? null : el("div", { class: "tree-empty", text: "Ninguna: importa un CSV o usa la herramienta Medición (N)" }),
  ));
  container.append(section(
    head("Materiales", null, p.materials.length, () => addMaterial(store), "Añadir material"),
    ...p.materials.map((m) => item({ type: "material", id: m.id }, m.name || m.id, penetrationSummary(m, refFrequency(p)),
      el("span", { class: "swatch", style: `background:${m.color}` }))),
  ));
}

function addMaterial(store) {
  const id = nextId("mat", store.project.materials);
  store.change((p) => p.materials.push({ id, name: "Material nuevo", color: "#a0a0a0", reflection: "fresnel", reflectivity: 0.5,
    eps_r: 5.0, sigma_c: 0.01, sigma_d: 0.0, penetration: "fixed", loss_db: 10.0, loss_a_db: 0.0, loss_b_db_per_ghz: 0.0, note: "" }));
  store.select({ type: "material", id });
}

export function refFrequency(p) {
  if (p.settings.reference_frequency_hz) return p.settings.reference_frequency_hz;
  const tx = p.transmitters.find((t) => t.enabled);
  return tx ? tx.frequency_hz : 2.4e9;
}

export function penetrationLoss(m, f) {
  if (m.penetration === "none") return null;
  if (m.penetration === "opaque") return Infinity;
  if (m.penetration === "fixed") return m.loss_db;
  return Math.max(0, m.loss_a_db + m.loss_b_db_per_ghz * f / 1e9);
}

function penetrationSummary(m, f) {
  const loss = penetrationLoss(m, f);
  if (loss === null) return "no tapa";
  if (loss === Infinity) return "opaco";
  return `${fmt(loss, 1)} dB`;
}

// ------------------------------------------------------------------ propiedades
// Campo: {key, label, type, unit, scale, min, step, options, show, nullable, help}
function fieldRow(obj, spec, onCommit) {
  const value = spec.get ? spec.get(obj) : obj[spec.key];
  let input;
  if (spec.type === "checkbox") {
    input = el("input", { type: "checkbox" });
    input.checked = !!value;
    input.addEventListener("change", () => onCommit(spec.key, input.checked));
    return el("label", { class: "field check" }, el("span", { text: spec.label }), input);
  }
  if (spec.type === "select") {
    input = el("select");
    for (const [v, label] of typeof spec.options === "function" ? spec.options(obj) : spec.options) {
      const option = el("option", { value: v, text: label });
      if (String(v) === String(value)) option.selected = true;
      input.append(option);
    }
    input.addEventListener("change", () => onCommit(spec.key, spec.integer ? Number(input.value) : input.value));
  } else if (spec.type === "color") {
    input = el("input", { type: "color", value });
    input.addEventListener("change", () => onCommit(spec.key, input.value));
  } else if (spec.type === "textarea") {
    input = el("textarea", { rows: 3 });
    input.value = value || "";
    input.addEventListener("change", () => onCommit(spec.key, input.value));
  } else if (spec.type === "text") {
    input = el("input", { type: "text" });
    input.value = value ?? "";
    input.addEventListener("change", () => onCommit(spec.key, input.value));
  } else {
    const scale = spec.scale || 1;
    const shown = (x) => (x === null || x === undefined ? "" : +(x / scale).toPrecision(spec.precision || 10));
    input = el("input", { type: "number", step: spec.step ?? "any", min: spec.min, max: spec.max });
    input.value = shown(value);
    if (spec.nullable) input.placeholder = spec.placeholder || "auto";
    input.addEventListener("change", () => {
      const text = input.value.trim();
      if (text === "" && spec.nullable) { onCommit(spec.key, null); return; }
      const number = Number(text);
      if (text === "" || !isFinite(number) || (spec.min !== undefined && number < spec.min)
          || (spec.max !== undefined && number > spec.max)) {
        input.value = shown(value);
        input.setCustomValidity("Valor no válido");
        input.reportValidity();
        setTimeout(() => input.setCustomValidity(""), 1500);
        return;
      }
      onCommit(spec.key, spec.integer ? Math.round(number) : spec.fromView ? spec.fromView(number) : number * scale);
    });
  }
  const control = spec.unit ? el("span", { class: "with-unit" }, input, el("span", { class: "unit", text: spec.unit })) : input;
  const row = el("label", { class: "field", title: spec.help }, el("span", { text: spec.label }), control);
  return row;
}

function buildForm(container, obj, specs, onCommit) {
  for (const spec of specs) {
    if (spec.section) {
      if (!spec.show || spec.show(obj)) container.append(el("div", { class: "props-section", text: spec.section }));
      continue;
    }
    if (spec.show && !spec.show(obj)) continue;
    container.append(fieldRow(obj, spec, onCommit));
  }
}

const TX_FIELDS = (standard) => [
  { key: "name", label: "Nombre", type: "text" },
  { key: "enabled", label: "Activo", type: "checkbox" },
  { section: "Posición" },
  { key: "x", label: "X", unit: "m" },
  { key: "y", label: "Y", unit: "m" },
  { key: "z", label: "Altura", unit: "m", min: 0 },
  { section: "Radio" },
  { key: "tx_power_dbm", label: "Potencia", unit: "dBm", precision: 6 },
  { key: "tx_power_dbm", label: "Potencia", unit: "mW", get: (t) => dbmToMw(t.tx_power_dbm), fromView: mwToDbm,
    min: 1e-9, precision: 4, help: "La misma potencia en milivatios (9 mW = 9.5 dBm, 100 mW = 20 dBm)" },
  { key: "frequency_hz", label: "Frecuencia", unit: "MHz", scale: 1e6, min: 0, show: (t) => !t.wifi_band },
  { key: "antenna_gain_dbi", label: "Ganancia", unit: "dBi" },
  { section: "Wi-Fi" },
  { key: "wifi_band", label: "Banda", type: "select", options: [["", "— (sin Wi-Fi)"], ...WIFI_BANDS],
    help: "Con un canal Wi-Fi la frecuencia es la del centro del canal, y el SINR cuenta lo que se enciman los canales" },
  { key: "wifi_width_mhz", label: "Ancho", type: "select", integer: true, show: (t) => !!t.wifi_band,
    options: (t) => widthsFor(t.wifi_band, standard).map((w) => [w, `${w} MHz`]) },
  { key: "wifi_channel", label: "Canal", type: "select", integer: true, show: (t) => !!t.wifi_band,
    options: (t) => channelOptions(t.wifi_band, t.wifi_width_mhz || 20),
    help: "El canal primario; con 40, 80 o 160 MHz el AP ocupa todo el bloque" },
  { key: "wifi_streams", label: "Flujos espaciales", type: "select", integer: true, show: (t) => !!t.wifi_band,
    options: [1, 2, 3, 4, 5, 6, 7, 8].map((n) => [n, `${n} (${n}×${n} MIMO)`]), help: "Se usa el menor entre el AP y el cliente" },
  { section: "Antena" },
  { key: "_antenna", label: "Modelo", type: "select", get: antennaPresetOf,
    options: [...Object.entries(ANTENNA_PRESETS).map(([id, a]) => [id, a.label]), ["custom", "Personalizada"]],
    help: "Pone el diagrama, la ganancia y los haces de un modelo genérico; luego puedes ajustar cada valor" },
  { key: "pattern", label: "Diagrama", type: "select", options: [["isotropic", "Isotrópica"], ["sector", "Sector 3GPP"]] },
  { key: "azimuth_deg", label: "Azimut", unit: "°", show: (t) => t.pattern === "sector", help: "0° = eje +x, 90° = eje +y" },
  { key: "downtilt_deg", label: "Inclinación", unit: "°", show: (t) => t.pattern === "sector", help: "Positiva hacia el suelo" },
  { key: "beamwidth_az_deg", label: "Haz horizontal", unit: "°", min: 0.1, show: (t) => t.pattern === "sector" },
  { key: "beamwidth_el_deg", label: "Haz vertical", unit: "°", min: 0.1, show: (t) => t.pattern === "sector" },
];

const AREA_FIELDS = [
  { key: "x_min", label: "X mínima", unit: "m" },
  { key: "x_max", label: "X máxima", unit: "m" },
  { key: "y_min", label: "Y mínima", unit: "m" },
  { key: "y_max", label: "Y máxima", unit: "m" },
  { key: "resolution_m", label: "Resolución", unit: "m", min: 0.01, help: "Paso de la rejilla de predicción" },
  { key: "rx_height_m", label: "Altura del Rx", unit: "m", min: 0 },
];

const SETTINGS_FIELDS = [
  { key: "mode", label: "Modelo", type: "select", options: Object.entries(MODE_LABELS) },
  { key: "diffraction", label: "Difracción (UTD)", type: "checkbox", show: (s) => s.mode === "raytracing",
    help: "Rayos difractados en las esquinas y bordes de techo de los edificios y en los bordes de los muros" },
  { key: "max_diffractions", label: "Difracciones por rayo", type: "select", integer: true,
    show: (s) => s.mode === "raytracing" && s.diffraction,
    options: [[1, "Una arista"], [2, "Hasta dos aristas (más lento)"]],
    help: "Con dos, el rayo puede pasar por dos aristas seguidas: sobre un edificio alto y ancho, o rodeando dos esquinas" },
  { key: "mixed_interactions", label: "Reflexión + difracción", type: "checkbox",
    show: (s) => s.mode === "raytracing" && s.diffraction,
    help: "Rayos que rebotan en un muro y luego se difractan en una arista, o al revés (más lento)" },
  { key: "max_reflections", label: "Reflexiones", type: "select", integer: true, show: (s) => s.mode === "raytracing",
    options: [[1, "Un rebote"], [2, "Hasta dos rebotes (más lento)"]],
    help: "Con dos rebotes se suman las reflexiones muro–muro y muro–suelo" },
  { key: "coherent", label: "Suma coherente", type: "checkbox", show: (s) => s.mode === "raytracing",
    help: "Suma las amplitudes complejas de los rayos: aparece el desvanecimiento rápido" },
  { key: "hata_environment", label: "Entorno", type: "select", options: Object.entries(HATA_LABELS), show: (s) => s.mode === "hata" },
  { key: "log_distance_exponent", label: "Exponente n", min: 0.01, show: (s) => s.mode === "logdistance" || s.mode === "multiwall",
    help: "2 = espacio libre; en interiores, típicamente 2.5 a 3.5 (muebles, gente)" },
  { key: "log_distance_d0_m", label: "Distancia d0", unit: "m", min: 0.01, show: (s) => s.mode === "logdistance" || s.mode === "multiwall" },
  { key: "min_distance_m", label: "Distancia mínima", unit: "m", min: 0, help: "Píxeles más cerca de una antena quedan sin dato" },
  { section: "Sombra log-normal" },
  { key: "shadow_sigma_db", label: "σ", unit: "dB", min: 0 },
  { key: "seed", label: "Semilla", nullable: true, integer: true, min: 0, placeholder: "aleatoria" },
  { section: "SINR" },
  { key: "sinr", label: "Calcular SINR", type: "checkbox" },
  { key: "noise_figure_db", label: "Figura de ruido", unit: "dB", show: (s) => s.sinr },
  { key: "bandwidth_hz", label: "Ancho de banda", unit: "MHz", scale: 1e6, min: 0, show: (s) => s.sinr,
    help: "Para los transmisores sin canal Wi-Fi (los Wi-Fi usan el ancho de su canal)" },
  { section: "Wi-Fi", show: (s) => s._wifi },
  { key: "wifi_standard", label: "Estándar", type: "select", options: WIFI_STANDARDS, show: (s) => s._wifi,
    help: "Generación de los AP y de los clientes. Wi-Fi 5 en 2.4 GHz trabaja como Wi-Fi 4; 6 GHz requiere Wi-Fi 6E" },
  { key: "wifi_guard_interval_us", label: "Intervalo de guarda", type: "select", integer: true, show: (s) => s._wifi,
    options: (s) => (GUARD_INTERVALS[s.wifi_standard] || [0.8]).map((g) => [g, `${g} µs`]) },
  { key: "wifi_client_streams", label: "Flujos del cliente", type: "select", integer: true, show: (s) => s._wifi,
    options: [1, 2, 3, 4, 5, 6, 7, 8].map((n) => [n, String(n)]), help: "Un teléfono suele tener 2; una laptop, 2 o 3" },
  { key: "wifi_cca_dbm", label: "Umbral CCA", unit: "dBm", show: (s) => s._wifi,
    help: "Un AP del mismo canal que llega por arriba de este nivel comparte el tiempo de aire (mapa de contención)" },
  { section: "Suelo y materiales" },
  { key: "ground_eps_r", label: "εr del suelo", min: 1 },
  { key: "ground_sigma", label: "σ del suelo", unit: "S/m", min: 0 },
  { key: "reference_frequency_hz", label: "f de referencia", unit: "MHz", scale: 1e6, nullable: true, min: 0,
    placeholder: "la del 1.er Tx", help: "Frecuencia a la que se evalúan σ y la penetración de los materiales" },
];

const BUILDING_FIELDS = (materials) => [
  { key: "name", label: "Nombre", type: "text" },
  { key: "height_m", label: "Altura", unit: "m", min: 0.01 },
  { key: "material", label: "Material", type: "select", options: materials.map((m) => [m.id, m.name || m.id]) },
  { key: "roof", label: "Techo", type: "checkbox", help: "Cierra el edificio por arriba (refleja y obstruye como las fachadas)" },
];

const WALL_FIELDS = (materials) => [
  { key: "name", label: "Nombre", type: "text" },
  { key: "material", label: "Material", type: "select", options: materials.map((m) => [m.id, m.name || m.id]) },
  { key: "thickness_m", label: "Espesor", unit: "cm", scale: 0.01, min: 0.001, max: 500, nullable: true,
    placeholder: "del material",
    help: "Vacío: la pérdida fija del material. Con espesor: la de una losa de ese grosor con la εr y la σ del material (ITU-R P.2040)" },
  { key: "z_min_m", label: "Base", unit: "m" },
  { key: "z_max_m", label: "Altura superior", unit: "m" },
];

const MATERIAL_FIELDS = [
  { key: "name", label: "Nombre", type: "text" },
  { key: "color", label: "Color", type: "color" },
  { section: "Reflexión" },
  { key: "reflection", label: "Modelo", type: "select", options: Object.entries(REFLECTION_LABELS) },
  { key: "reflectivity", label: "|Γ|", min: 0, show: (m) => m.reflection === "constant", help: "Entre 0 y 1" },
  { key: "eps_r", label: "εr", min: 1, show: (m) => m.reflection === "fresnel" },
  { key: "sigma_c", label: "σ: c", unit: "S/m", min: 0, show: (m) => m.reflection === "fresnel", help: "σ = c · f_GHz^d (ITU-R P.2040)" },
  { key: "sigma_d", label: "σ: d", show: (m) => m.reflection === "fresnel" },
  { section: "Penetración" },
  { key: "penetration", label: "Modelo", type: "select", options: Object.entries(PENETRATION_LABELS) },
  { key: "loss_db", label: "Pérdida", unit: "dB", min: 0, show: (m) => m.penetration === "fixed" },
  { key: "loss_a_db", label: "a", unit: "dB", show: (m) => m.penetration === "frequency", help: "L = a + b · f_GHz" },
  { key: "loss_b_db_per_ghz", label: "b", unit: "dB/GHz", show: (m) => m.penetration === "frequency" },
  { key: "note", label: "Nota", type: "textarea" },
];


const PLAN_FIELDS = [
  { key: "name", label: "Nombre", type: "text" },
  { key: "visible", label: "Visible", type: "checkbox" },
  { key: "opacity", label: "Opacidad", min: 0, max: 1, step: 0.05, help: "0 = invisible, 1 = opaco. Las zonas blancas del plano dejan ver el mapa." },
  { section: "Posición y escala" },
  { key: "x_m", label: "X (esquina inf. izq.)", unit: "m" },
  { key: "y_m", label: "Y (esquina inf. izq.)", unit: "m" },
  { key: "width_m", label: "Ancho", unit: "m", min: 0.001, help: "El alto sale de las proporciones de la imagen" },
];

const MEASUREMENT_FIELDS = (transmitters) => [
  { key: "rssi_dbm", label: "Potencia medida", unit: "dBm" },
  { key: "tx_id", label: "Transmisor", type: "select",
    options: [["", "El más fuerte (mejor servidor)"], ...transmitters.map((t) => [t.tx_id, t.name || `Tx ${t.tx_id}`])] },
  { key: "label", label: "Etiqueta", type: "text" },
  { section: "Posición" },
  { key: "x", label: "X", unit: "m" },
  { key: "y", label: "Y", unit: "m" },
  { key: "z", label: "Altura", unit: "m", min: 0, nullable: true, placeholder: "la del área" },
];

const PROJECT_FIELDS = [
  { key: "name", label: "Nombre", type: "text" },
  { key: "description", label: "Descripción", type: "textarea" },
];

export function renderProperties(container, titleEl, store, app) {
  container.replaceChildren();
  const s = store.selection;
  const p = store.project;
  if (!p) return;
  const obj = store.selectedObject();
  if (!s || !obj) {
    titleEl.textContent = "Propiedades";
    container.append(el("p", { class: "note", text: "Selecciona un elemento en el mapa o en el árbol del proyecto para editarlo." }),
      el("p", { class: "note", text: "Herramientas: V seleccionar · T transmisor · R edificio rectangular · P edificio poligonal · W muro · S receptor · M medir. Rueda: zoom; botón derecho o espacio: desplazar." }));
    return;
  }
  const commit = (mutate) => (key, value) => store.change(() => mutate(key, value));
  if (s.type === "tx") {
    titleEl.textContent = "Transmisor";
    const standard = p.settings.wifi_standard || "ax";
    const view = { ...obj, wifi_band: obj.wifi_band || "", wifi_width_mhz: obj.wifi_width_mhz || 20, wifi_streams: obj.wifi_streams || 2 };
    buildForm(container, view, TX_FIELDS(standard), commit((k, v) => {
      const t = store.tx(s.id);
      if (k === "wifi_band" || k === "wifi_width_mhz" || k === "wifi_channel") {
        t[k] = k === "wifi_band" ? (v || null) : v;
        normalizeWifi(t, standard);
      } else if (k === "_antenna") {
        applyAntennaPreset(t, v);
      } else t[k] = v;
    }));
    const slot = store.txSlot(obj.tx_id);
    const wifi = obj.wifi_band ? ` · canal ${obj.wifi_channel} en ${fmt(centreMHz(obj.wifi_band, obj.wifi_channel, obj.wifi_width_mhz || 20), 0)} MHz` : "";
    container.append(el("div", { class: "readout" }, el("span", { class: "key-line", style: `background:${txColor(slot)}` }),
      el("span", { text: `tx_id ${obj.tx_id} · EIRP ${fmt(obj.tx_power_dbm + obj.antenna_gain_dbi, 1)} dBm · λ = ${fmt(299792458 / obj.frequency_hz * 100, 1)} cm${wifi}` })));
    container.append(actions(store, app, s));
  } else if (s.type === "building") {
    titleEl.textContent = "Edificio";
    buildForm(container, obj, BUILDING_FIELDS(p.materials), commit((k, v) => { store.building(s.id)[k] = v; }));
    const area = Math.abs(polygonAreaOf(obj.footprint));
    container.append(el("div", { class: "readout", text: `${obj.footprint.length} vértices · área ${fmt(area, 0)} m² · ${obj.footprint.length + (obj.roof ? 1 : 0)} muros en el motor` }));
    container.append(vertexTable(store, app, obj));
    container.append(actions(store, app, s));
  } else if (s.type === "wall") {
    titleEl.textContent = "Muro";
    buildForm(container, obj, WALL_FIELDS(p.materials), commit((k, v) => { store.wall(s.id)[k] = v; }));
    const pointSpecs = [
      { key: 0, label: "Inicio X", unit: "m" }, { key: 1, label: "Inicio Y", unit: "m" },
    ];
    buildForm(container, obj.start, pointSpecs, commit((k, v) => { store.wall(s.id).start[k] = v; }));
    buildForm(container, obj.end, [{ key: 0, label: "Fin X", unit: "m" }, { key: 1, label: "Fin Y", unit: "m" }],
      commit((k, v) => { store.wall(s.id).end[k] = v; }));
    const material = p.materials.find((m) => m.id === obj.material);
    const f = referenceFrequency(p);
    const loss = material ? wallLossDb(obj, material, f) : null;
    const lossText = loss === null ? "no atenúa" : loss === Infinity ? "opaco" : `${fmt(loss, 1)} dB`;
    const how = obj.thickness_m && material && ["fixed", "frequency"].includes(material.penetration)
      ? `losa de ${fmt(obj.thickness_m * 100, obj.thickness_m < 0.01 ? 2 : 1)} cm` : "ley del material";
    container.append(el("div", { class: "readout",
      text: `Longitud ${fmt(Math.hypot(obj.end[0] - obj.start[0], obj.end[1] - obj.start[1]), 2)} m · pérdida por cruce a ${fmt(f / 1e6, 0)} MHz: ${lossText} (${how})` }));
    container.append(actions(store, app, s));
  } else if (s.type === "material") {
    titleEl.textContent = "Material";
    buildForm(container, obj, MATERIAL_FIELDS, commit((k, v) => { store.material(s.id)[k] = v; }));
    const f = refFrequency(p);
    const loss = penetrationLoss(obj, f);
    const sigma = obj.sigma_c * (f / 1e9) ** obj.sigma_d;
    const parts = [`A ${fmtFreq(f)}:`];
    if (obj.reflection === "fresnel") parts.push(`σ = ${sigma.toPrecision(3)} S/m`);
    parts.push(loss === null ? "no obstruye" : loss === Infinity ? "opaco" : `penetración ${fmt(loss, 1)} dB`);
    const uses = p.buildings.filter((b) => b.material === obj.id).length + p.walls.filter((w) => w.material === obj.id).length;
    container.append(el("div", { class: "readout", text: `${parts.join(" ")} · usado por ${uses} elemento(s)` }));
    container.append(actions(store, app, s, uses > 0 ? "Este material está en uso: cambia antes el material de esos elementos." : null));
  } else if (s.type === "area") {
    titleEl.textContent = "Área de predicción";
    buildForm(container, obj, AREA_FIELDS, commit((k, v) => { store.project.area[k] = v; }));
    const a = obj;
    const nx = Math.floor((a.x_max - a.x_min) / a.resolution_m + 1e-9) + 1;
    const ny = Math.floor((a.y_max - a.y_min) / a.resolution_m + 1e-9) + 1;
    container.append(el("div", { class: "readout", text: `${nx} × ${ny} = ${(nx * ny).toLocaleString("es-MX")} píxeles` }));
    container.append(el("div", { class: "props-actions" }, el("button", { class: "btn", text: "Ajustar a los objetos", onclick: () => app.fitAreaToObjects() })));
  } else if (s.type === "settings") {
    titleEl.textContent = "Configuración de la simulación";
    const anyWifi = p.transmitters.some((t) => t.wifi_band);
    const view = { wifi_standard: "ax", wifi_guard_interval_us: 0.8, wifi_client_streams: 2, wifi_cca_dbm: -82, ...obj, _wifi: anyWifi };
    buildForm(container, view, SETTINGS_FIELDS, commit((k, v) => {
      const st = store.project.settings;
      st[k] = v;
      if (k === "wifi_standard") {
        // Guarda y anchos que el nuevo estándar admite
        const gis = GUARD_INTERVALS[v];
        if (!gis.includes(st.wifi_guard_interval_us)) st.wifi_guard_interval_us = gis[0];
        for (const t of store.project.transmitters) if (t.wifi_band) normalizeWifi(t, v);
      }
    }));
    if (!anyWifi) container.append(el("p", { class: "note", text: "Para los mapas de Wi-Fi (tasa de datos, MCS y contención), asigna una banda y un canal Wi-Fi a los transmisores." }));
    if (obj.mode === "multiwall") {
      container.append(el("p", { class: "note", text: "Multimuro: pérdida log-distancia (FSPL hasta d0 y 10·n·log10(d/d0) después) más la pérdida de cada muro que cruza la línea directa del AP al punto, como en las herramientas de planeación Wi-Fi. Los muros no reflejan ni difractan aquí: para eso, ray tracing." }));
    } else if (obj.mode !== "raytracing") {
      container.append(el("p", { class: "note", text: "Los modelos empíricos ignoran los edificios y muros: ni reflejan ni obstruyen. Usa ray tracing o el modelo multimuro para verlos en el mapa." }));
    } else {
      container.append(el("p", { class: "note", text: "Ray tracing de OpenRPS: rayo directo, reflexión en el suelo y rebotes especulares en los muros (uno, o hasta dos), difracción UTD en las aristas (una, o hasta dos por rayo), rayos que combinan un rebote con una difracción y obstrucción por los muros que tapan. Cada rayo tiene a lo más dos interacciones." }));
    }
  } else if (s.type === "floorplan") {
    titleEl.textContent = "Plano de fondo";
    buildForm(container, obj, PLAN_FIELDS, commit((k, v) => { store.project.floor_plan[k] = v; }));
    const h = obj.width_m * obj.pixel_height / obj.pixel_width;
    container.append(el("div", { class: "readout", text: `${obj.pixel_width} × ${obj.pixel_height} px · ${fmt(obj.width_m, 2)} × ${fmt(h, 2)} m · ${fmt(100 * obj.width_m / obj.pixel_width, 2)} cm por píxel` }));
    container.append(el("p", { class: "note", text: "Para escalarlo: marca dos puntos del plano cuya distancia real conozcas (una pared, una cota) y escribe esa distancia. Con el plano seleccionado, arrástralo con la herramienta Seleccionar para moverlo." }));
    container.append(el("div", { class: "props-actions" },
      el("button", { class: "btn", text: "Escalar con dos puntos", onclick: () => app.startPlanScale() }),
      el("button", { class: "btn", text: "Área de predicción = plano", onclick: () => app.fitAreaToPlan() }),
      el("button", { class: "btn", text: "Cambiar imagen…", onclick: () => app.pickFloorPlan() }),
      el("button", { class: "btn danger", text: "Quitar", onclick: () => deleteSelection(store) })));
  } else if (s.type === "measurements") {
    titleEl.textContent = "Mediciones";
    const n = obj.length;
    const named = obj.filter((m) => m.tx_id !== null && m.tx_id !== undefined).length;
    container.append(el("div", { class: "readout", text: n ? `${n} punto(s) · ${named} con transmisor indicado, ${n - named} comparados con el más fuerte` : "Sin mediciones" }));
    container.append(el("p", { class: "note", text: "CSV con columnas x, y y la potencia (rssi, dbm, potencia o señal); opcionales: z o altura, tx (id o nombre del transmisor) y una etiqueta. Separador coma, punto y coma o tabulador." }));
    container.append(el("div", { class: "props-actions" },
      el("button", { class: "btn", text: "Importar CSV…", onclick: () => app.pickMeasurementsCsv() }),
      el("button", { class: "btn", text: "Anotar en el mapa (N)", onclick: () => app.setTool("sample") }),
      n ? el("button", { class: "btn", text: "Comparar y calibrar", onclick: () => app.showMeasurements() }) : null,
      n ? el("button", { class: "btn danger", text: "Borrar todas", onclick: () => deleteSelection(store) }) : null));
  } else if (s.type === "measurement") {
    titleEl.textContent = `Medición ${s.id + 1}`;
    const view = { ...obj, tx_id: obj.tx_id ?? "" };
    buildForm(container, view, MEASUREMENT_FIELDS(p.transmitters), commit((k, v) => {
      store.project.measurements[s.id][k] = k === "tx_id" ? (v === "" ? null : Number(v)) : v;
    }));
    const point = measurementResult(store, s.id);
    if (point) {
      container.append(el("div", { class: "readout", text: point.status === "ok"
        ? `Predicho ${fmt(point.predicted_dbm, 1)} dBm · error ${point.error_db >= 0 ? "+" : "−"}${fmt(Math.abs(point.error_db), 1)} dB (predicho − medido)`
        : `Sin comparación: ${point.status}` }));
    }
    container.append(el("div", { class: "props-actions" },
      el("button", { class: "btn", text: "Centrar en el mapa", onclick: () => app.zoomTo(s) }),
      el("button", { class: "btn danger", text: "Eliminar", onclick: () => deleteSelection(store) })));
  } else if (s.type === "project") {
    titleEl.textContent = "Proyecto";
    buildForm(container, obj, PROJECT_FIELDS, commit((k, v) => { store.project[k] = v; }));
    container.append(el("div", { class: "readout", text: `${p.transmitters.length} transmisores · ${p.buildings.length} edificios · ${p.walls.length} muros · ${p.materials.length} materiales` }));
  }
}

function polygonAreaOf(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return a / 2;
}

function vertexTable(store, app, building) {
  const wrap = el("div");
  wrap.append(el("div", { class: "props-section", text: "Vértices (m)" }));
  const table = el("table", { class: "vertex-table" });
  const update = (k, axis, value) => {
    const candidate = building.footprint.map((q) => q.slice());
    candidate[k][axis] = value;
    if (!isSimplePolygon(candidate)) { app.notify("error", "Con ese valor el polígono se cruzaría consigo mismo."); app.refreshProperties(); return; }
    store.change(() => { store.building(building.id).footprint[k][axis] = value; });
  };
  building.footprint.forEach(([x, y], k) => {
    const ix = el("input", { type: "number", step: "any" });
    ix.value = x;
    ix.addEventListener("change", () => isFinite(Number(ix.value)) && ix.value !== "" && update(k, 0, Number(ix.value)));
    const iy = el("input", { type: "number", step: "any" });
    iy.value = y;
    iy.addEventListener("change", () => isFinite(Number(iy.value)) && iy.value !== "" && update(k, 1, Number(iy.value)));
    const del = el("button", { class: "btn", title: "Quitar vértice", text: "−", disabled: building.footprint.length <= 3 });
    del.addEventListener("click", () => {
      const candidate = building.footprint.filter((_, i) => i !== k);
      if (!isSimplePolygon(candidate)) { app.notify("error", "Sin ese vértice el polígono no sería válido."); return; }
      store.change(() => { store.building(building.id).footprint.splice(k, 1); });
    });
    table.append(el("tr", {}, el("td", { class: "idx", text: String(k + 1) }), el("td", {}, ix), el("td", {}, iy), el("td", {}, del)));
  });
  wrap.append(table);
  wrap.append(el("div", { class: "props-actions" }, el("button", { class: "btn", text: "Añadir vértice", onclick: () => {
    store.change(() => {
      const fp = store.building(building.id).footprint;
      const a = fp[fp.length - 1];
      const b = fp[0];
      fp.push([round((a[0] + b[0]) / 2, 3), round((a[1] + b[1]) / 2, 3)]);
    });
  } })));
  return wrap;
}

function actions(store, app, s, deleteBlocked = null) {
  const row = el("div", { class: "props-actions" });
  if (s.type !== "material") row.append(el("button", { class: "btn", text: "Centrar en el mapa", onclick: () => app.zoomTo(s) }));
  row.append(el("button", { class: "btn", text: "Duplicar", onclick: () => duplicateSelection(store, app) }));
  const del = el("button", { class: "btn danger", text: "Eliminar", onclick: () => {
    if (deleteBlocked) { app.notify("error", deleteBlocked); return; }
    deleteSelection(store);
  } });
  row.append(del);
  return row;
}

// Resultado de la última comparación para la medición k (si sigue vigente)
export function measurementResult(store, k) {
  const c = store.comparison;
  if (!c || c.revision !== store.measurementRevision()) return null;
  return c.data.points[k] || null;
}

export function deleteSelection(store) {
  const s = store.selection;
  if (!s) return false;
  if (s.type === "floorplan") {
    store.change((p) => { p.floor_plan = null; });
    store.select(null);
    return true;
  }
  if (s.type === "measurements") {
    if (!store.project.measurements.length) return false;
    if (!window.confirm(`¿Borrar las ${store.project.measurements.length} mediciones?`)) return false;
    store.change((p) => { p.measurements = []; });
    store.select(null);
    return true;
  }
  if (s.type === "measurement") {
    store.change((p) => { p.measurements.splice(s.id, 1); });
    store.select(null);
    return true;
  }
  if (s.type === "material") {
    const used = store.project.buildings.some((b) => b.material === s.id) || store.project.walls.some((w) => w.material === s.id);
    if (used) return false;
  }
  const lists = { tx: ["transmitters", "tx_id"], building: ["buildings", "id"], wall: ["walls", "id"], material: ["materials", "id"] };
  const entry = lists[s.type];
  if (!entry) return false;
  store.change((p) => { p[entry[0]] = p[entry[0]].filter((item) => item[entry[1]] !== s.id); });
  store.select(null);
  return true;
}

export function duplicateSelection(store, app) {
  const s = store.selection;
  const obj = store.selectedObject();
  if (!s || !obj) return;
  const copy = JSON.parse(JSON.stringify(obj));
  const shift = Math.max(2, store.project.area.resolution_m * 3);
  if (s.type === "tx") {
    copy.tx_id = nextTxId(store.project.transmitters);
    copy.name = `${obj.name || "Tx"} (copia)`;
    copy.x = round(copy.x + shift, 3);
    copy.y = round(copy.y - shift, 3);
    store.change((p) => p.transmitters.push(copy));
    store.select({ type: "tx", id: copy.tx_id });
  } else if (s.type === "building") {
    copy.id = nextId("b", store.project.buildings);
    copy.name = `${obj.name || "Edificio"} (copia)`;
    copy.footprint = copy.footprint.map(([x, y]) => [round(x + shift, 3), round(y - shift, 3)]);
    store.change((p) => p.buildings.push(copy));
    store.select({ type: "building", id: copy.id });
  } else if (s.type === "wall") {
    copy.id = nextId("w", store.project.walls);
    copy.name = `${obj.name || "Muro"} (copia)`;
    copy.start = [round(copy.start[0] + shift, 3), round(copy.start[1] - shift, 3)];
    copy.end = [round(copy.end[0] + shift, 3), round(copy.end[1] - shift, 3)];
    store.change((p) => p.walls.push(copy));
    store.select({ type: "wall", id: copy.id });
  } else if (s.type === "material") {
    copy.id = nextId("mat", store.project.materials);
    copy.name = `${obj.name || "Material"} (copia)`;
    store.change((p) => p.materials.push(copy));
    store.select({ type: "material", id: copy.id });
  }
}

export function objectCenter(store, s) {
  const obj = store.selectedObject();
  if (!obj || !s) return null;
  if (s.type === "tx") return [obj.x, obj.y];
  if (s.type === "building") return centroid(obj.footprint);
  if (s.type === "wall") return [(obj.start[0] + obj.end[0]) / 2, (obj.start[1] + obj.end[1]) / 2];
  if (s.type === "measurement") return [obj.x, obj.y];
  return null;
}

export { MODE_LABELS, el };
