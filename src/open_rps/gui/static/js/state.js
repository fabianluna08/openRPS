// Estado de la aplicación: proyecto (el mismo JSON que open_rps.project.Project.to_dict), selección,
// resultado, receptor de prueba y opciones de la vista.  Historial para deshacer/rehacer.

const AUTOSAVE_KEY = "openrps.project.v1";
const MAX_HISTORY = 80;

export class Store {
  constructor() {
    this.project = null;
    this.info = null;
    this.selection = null;          // {type: "tx" | "building" | "wall" | "material" | "area" | "settings" | "project", id}
    this.result = null;             // resultado decodificado (api.decodeResult) + revision + projectSnapshot
    this.probe = null;              // {point: [x, y, z], data, loading, error}
    this.revision = 0;              // cambia con cada modificación del proyecto
    this.savedRevision = 0;
    this.view = {
      mode: "2d", tool: "select", metric: "rx_power", txFilter: "best", opacity: 0.85,
      rangeAuto: true, rangeMin: null, rangeMax: null, grid: true, snap: true, smooth: false, labels: true,
      relief: false, transparent: false, showBlocked: false, hoverRay: null,
      coverageThreshold: -85, sinrThreshold: 0,
    };
    this.undoStack = [];
    this.redoStack = [];
    this.listeners = {};
    this.log = [];
  }

  on(event, fn) {
    (this.listeners[event] ||= []).push(fn);
  }

  emit(event, payload) {
    for (const fn of this.listeners[event] || []) fn(payload);
  }

  // ----- proyecto
  setProject(project, { keepHistory = false } = {}) {
    if (keepHistory) this.checkpoint();
    else { this.undoStack = []; this.redoStack = []; }
    this.project = project;
    this.revision += 1;
    this.savedRevision = this.revision;
    this.selection = null;
    this.result = null;
    this.probe = null;
    this.emit("project", { reset: true });
    this.emit("selection");
    this.emit("result");
    this.emit("probe");
    this.autosave();
  }

  snapshot() {
    return JSON.stringify(this.project);
  }

  // Guarda el estado actual en el historial (antes de modificarlo)
  checkpoint() {
    const snap = this.snapshot();
    if (this.undoStack[this.undoStack.length - 1] !== snap) {
      this.undoStack.push(snap);
      if (this.undoStack.length > MAX_HISTORY) this.undoStack.shift();
    }
    this.redoStack = [];
  }

  // Modifica el proyecto con fn(project).  checkpoint=false para los pasos intermedios de un arrastre.
  change(fn, { checkpoint = true, live = false } = {}) {
    if (checkpoint) this.checkpoint();
    fn(this.project);
    this.revision += 1;
    this.emit("project", { live });
    if (!live) this.autosave();
  }

  // Fin de un arrastre: ya se guardó el checkpoint al empezar
  endLive() {
    this.emit("project", { live: false });
    this.autosave();
  }

  undo() {
    if (!this.undoStack.length) return false;
    this.redoStack.push(this.snapshot());
    this.project = JSON.parse(this.undoStack.pop());
    this.afterHistory();
    return true;
  }

  redo() {
    if (!this.redoStack.length) return false;
    this.undoStack.push(this.snapshot());
    this.project = JSON.parse(this.redoStack.pop());
    this.afterHistory();
    return true;
  }

  afterHistory() {
    this.revision += 1;
    if (this.selection && !this.selectedObject()) this.selection = null;
    this.emit("project", {});
    this.emit("selection");
    this.autosave();
  }

  autosave() {
    try {
      localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(this.project));
    } catch (error) { /* almacenamiento no disponible: sin autoguardado */ }
  }

  static restore() {
    try {
      const text = localStorage.getItem(AUTOSAVE_KEY);
      return text ? JSON.parse(text) : null;
    } catch (error) {
      return null;
    }
  }

  get outdated() {
    return !!this.result && this.result.revision !== this.revision;
  }

  get modified() {
    return this.revision !== this.savedRevision;
  }

  // ----- selección
  select(selection) {
    this.selection = selection;
    this.emit("selection");
  }

  isSelected(type, id) {
    return !!this.selection && this.selection.type === type && this.selection.id === id;
  }

  selectedObject() {
    const s = this.selection;
    if (!s || !this.project) return null;
    if (s.type === "tx") return this.tx(s.id);
    if (s.type === "building") return this.building(s.id);
    if (s.type === "wall") return this.wall(s.id);
    if (s.type === "material") return this.material(s.id);
    if (s.type === "area") return this.project.area;
    if (s.type === "settings") return this.project.settings;
    if (s.type === "project") return this.project;
    return null;
  }

  tx(id) { return this.project.transmitters.find((t) => t.tx_id === id) || null; }
  building(id) { return this.project.buildings.find((b) => b.id === id) || null; }
  wall(id) { return this.project.walls.find((w) => w.id === id) || null; }
  material(id) { return this.project.materials.find((m) => m.id === id) || null; }

  // Posición del transmisor en la lista del proyecto: fija su color (no cambia al desactivar otros)
  txSlot(txId) {
    return this.project.transmitters.findIndex((t) => t.tx_id === txId);
  }

  txName(txId) {
    const tx = this.tx(txId);
    return tx ? (tx.name || `Tx ${tx.tx_id}`) : `Tx ${txId}`;
  }

  // ----- vista
  setView(patch) {
    Object.assign(this.view, patch);
    this.emit("view", patch);
  }

  // ----- mensajes
  addLog(level, text) {
    this.log.push({ level, text, time: new Date() });
    if (this.log.length > 300) this.log.shift();
    this.emit("log");
  }
}

export function nextId(prefix, items) {
  let n = items.length + 1;
  const used = new Set(items.map((item) => item.id));
  while (used.has(`${prefix}${n}`)) n += 1;
  return `${prefix}${n}`;
}

export function nextTxId(transmitters) {
  return transmitters.reduce((m, t) => Math.max(m, t.tx_id), -1) + 1;
}
