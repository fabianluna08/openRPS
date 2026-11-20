// Vista 3D (three.js): edificios extruidos, muros, antenas, mapa de cobertura (opcionalmente en relieve)
// y los rayos hasta el receptor de prueba.  El eje z apunta hacia arriba, como en el motor.
import * as THREE from "../vendor/three.bundle.min.js";
import { MUTED, rampColor, txColor } from "./colors.js";
import { currentLayer, imageExtent } from "./coverage.js";
import { numberOf } from "./view2d.js";

const SELECT = 0x1c5cab;

export class View3D {
  constructor(canvas, store, app) {
    this.canvas = canvas;
    this.store = store;
    this.app = app;
    this.visible = false;
    this.ok = false;
    this.layerKey = null;
    this.layerRef = null;
    try {
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    } catch (error) {
      document.getElementById("webgl-error").hidden = false;
      return;
    }
    this.ok = true;
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xe9ecef);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 50000);
    this.camera.up.set(0, 0, 1);
    this.controls = new THREE.OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = false;
    this.controls.maxPolarAngle = Math.PI * 0.495;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8d8a83, 2.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-0.5, -0.8, 1.2);
    this.scene.add(sun);

    this.groundGroup = new THREE.Group();
    this.sceneGroup = new THREE.Group();
    this.rayGroup = new THREE.Group();
    this.scene.add(this.groundGroup, this.sceneGroup, this.rayGroup);

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    let down = null;
    canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
    canvas.addEventListener("pointerup", (e) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4 && e.button === 0) this.onClick(e);
      down = null;
    });
    this.controls.addEventListener("change", () => this.requestRender());
    this.needsFit = true;
  }

  setVisible(visible) {
    this.visible = visible;
    if (!this.ok) return;
    if (visible) {
      this.resize();
      if (this.needsFit) this.fitCamera();
      this.requestRender();
    }
  }

  resize() {
    if (!this.ok) return;
    const parent = this.canvas.parentElement;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  requestRender() {
    if (!this.ok || !this.visible || this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      const moving = this.controls.update();
      this.renderer.render(this.scene, this.camera);
      if (moving) this.requestRender();
    });
  }

  areaInfo() {
    const a = this.store.project.area;
    const cx = (a.x_min + a.x_max) / 2;
    const cy = (a.y_min + a.y_max) / 2;
    const size = Math.max(a.x_max - a.x_min, a.y_max - a.y_min);
    return { a, cx, cy, size };
  }

  fitCamera() {
    if (!this.ok || !this.store.project) return;
    const { cx, cy, size } = this.areaInfo();
    this.camera.position.set(cx - 0.42 * size, cy - 0.68 * size, 0.58 * size);
    this.camera.near = Math.max(0.01, size / 5000);
    this.camera.far = size * 40;
    this.camera.updateProjectionMatrix();
    this.controls.target.set(cx, cy, 0);
    this.controls.update();
    this.needsFit = false;
    this.requestRender();
  }

  topView() {
    if (!this.ok) return;
    const { cx, cy, size } = this.areaInfo();
    this.camera.position.set(cx, cy - size * 0.001, size * 1.35);
    this.controls.target.set(cx, cy, 0);
    this.controls.update();
    this.requestRender();
  }

  focusProbe() {
    const probe = this.store.probe;
    if (!this.ok || !probe) return;
    const { size } = this.areaInfo();
    const [x, y, z] = probe.point;
    const d = size * 0.25;
    this.controls.target.set(x, y, z);
    this.camera.position.set(x - d * 0.6, y - d * 0.9, z + d * 0.6);
    this.controls.update();
    this.requestRender();
  }

  // ----- contenido
  rebuildScene() {
    if (!this.ok || !this.store.project) return;
    disposeGroup(this.sceneGroup);
    const p = this.store.project;
    const v = this.store.view;
    const { size } = this.areaInfo();
    const seeThrough = v.transparent || v.relief;
    const opacity = seeThrough ? 0.32 : 0.94;
    for (const b of p.buildings) {
      const mat = this.store.material(b.material);
      const shape = new THREE.Shape(b.footprint.map(([x, y]) => new THREE.Vector2(x, y)));
      const geom = new THREE.ExtrudeGeometry(shape, { depth: b.height_m, bevelEnabled: false });
      const mesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({
        color: mat ? mat.color : "#9a9a9a", transparent: true, opacity, depthWrite: !seeThrough,
      }));
      mesh.userData = { type: "building", id: b.id };
      const selected = this.store.isSelected("building", b.id);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom, 25), new THREE.LineBasicMaterial({
        color: selected ? SELECT : 0x2c2c2a, transparent: true, opacity: selected ? 1 : 0.55,
      }));
      this.sceneGroup.add(mesh, edges);
    }
    for (const w of p.walls) {
      const mat = this.store.material(w.material);
      const geom = new THREE.BufferGeometry();
      const [x0, y0] = w.start;
      const [x1, y1] = w.end;
      geom.setAttribute("position", new THREE.Float32BufferAttribute([
        x0, y0, w.z_min_m, x1, y1, w.z_min_m, x1, y1, w.z_max_m, x0, y0, w.z_max_m], 3));
      geom.setIndex([0, 1, 2, 0, 2, 3]);
      geom.computeVertexNormals();
      const mesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({
        color: mat ? mat.color : "#9a9a9a", side: THREE.DoubleSide, transparent: true, opacity: v.transparent ? 0.45 : 0.95,
      }));
      const selected = this.store.isSelected("wall", w.id);
      const outline = new THREE.LineLoop(new THREE.BufferGeometry().setAttribute("position",
        new THREE.Float32BufferAttribute([x0, y0, w.z_min_m, x1, y1, w.z_min_m, x1, y1, w.z_max_m, x0, y0, w.z_max_m], 3)),
      new THREE.LineBasicMaterial({ color: selected ? SELECT : 0x2c2c2a }));
      this.sceneGroup.add(mesh, outline);
    }
    const r = size / 160;
    p.transmitters.forEach((tx, slot) => {
      const color = new THREE.Color(txColor(slot));
      const mast = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(tx.x, tx.y, 0), new THREE.Vector3(tx.x, tx.y, tx.z)]),
      new THREE.LineBasicMaterial({ color: 0x52514e }));
      const head = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), new THREE.MeshLambertMaterial({
        color: tx.enabled ? color : new THREE.Color(MUTED), emissive: tx.enabled ? color.clone().multiplyScalar(0.25) : 0x000000,
      }));
      head.position.set(tx.x, tx.y, tx.z);
      this.sceneGroup.add(mast, head);
      if (tx.pattern === "sector") {
        const length = size / 14;
        const half = Math.min(75, tx.beamwidth_az_deg / 2) * Math.PI / 180;
        const cone = new THREE.ConeGeometry(length * Math.tan(half), length, 32, 1, true);
        cone.translate(0, -length / 2, 0);            // vértice en el origen, abriéndose hacia -y
        const mesh = new THREE.Mesh(cone, new THREE.MeshBasicMaterial({
          color, transparent: true, opacity: tx.enabled ? 0.22 : 0.08, side: THREE.DoubleSide, depthWrite: false,
        }));
        const az = tx.azimuth_deg * Math.PI / 180;
        const el = -tx.downtilt_deg * Math.PI / 180;
        const dir = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir);
        mesh.position.set(tx.x, tx.y, tx.z);
        this.sceneGroup.add(mesh);
      }
      if (v.labels) {
        const label = textSprite(tx.name || `Tx ${tx.tx_id}`, size / 38);
        label.position.set(tx.x, tx.y, tx.z + r * 3.2);
        this.sceneGroup.add(label);
      }
    });
    // Mediciones: un rombo (octaedro) a la altura medida, con su valor en la escala del mapa si es de potencia
    const ms = p.measurements || [];
    if (ms.length) {
      const layer = currentLayer(this.store);
      const range = layer && layer.metric === "rx_power" && layer.range ? layer.range : null;
      const geom = new THREE.OctahedronGeometry(r * 0.7);
      ms.forEach((m, k) => {
        const z = m.z === null || m.z === undefined ? p.area.rx_height_m : m.z;
        const color = range ? rampColor((m.rssi_dbm - range[0]) / (range[1] - range[0] || 1)) : "#fcfcfb";
        const mesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ color: new THREE.Color(color) }));
        mesh.position.set(m.x, m.y, z + r * 0.7);
        const outline = new THREE.LineSegments(new THREE.EdgesGeometry(geom), new THREE.LineBasicMaterial({
          color: this.store.isSelected("measurement", k) ? SELECT : 0x0b0b0b }));
        outline.position.copy(mesh.position);
        this.sceneGroup.add(mesh, outline);
      });
    }
    this.updateGround(true);
    this.updateRays();
    this.requestRender();
  }

  updateGround(force = false) {
    if (!this.ok || !this.store.project) return;
    const layer = currentLayer(this.store);
    const v = this.store.view;
    // The layer is compared by identity: currentLayer() makes a new one whenever the result, the metric, the
    // transmitter or the scale change (a canvas in a joined string would always read "[object ...]")
    const plan = this.store.project.floor_plan;
    const planImage = plan && plan.visible ? this.store.floorPlanImage() : null;
    const key = [v.relief, v.opacity, this.store.outdated, v.smooth, JSON.stringify(this.store.project.area),
      planImage ? [plan.image, plan.x_m, plan.y_m, plan.width_m, plan.opacity].join(",") : ""].join("|");
    if (!force && layer === this.layerRef && key === this.layerKey && this.groundGroup.children.length) return;
    this.layerKey = key;
    this.layerRef = layer;
    disposeGroup(this.groundGroup);
    const { a, cx, cy, size } = this.areaInfo();
    // suelo neutro alrededor del área
    const base = new THREE.Mesh(new THREE.PlaneGeometry(size * 6, size * 6), new THREE.MeshLambertMaterial({ color: 0xd7d6cf }));
    base.position.set(cx, cy, -0.05);
    this.groundGroup.add(base);
    const area = new THREE.Mesh(new THREE.PlaneGeometry(a.x_max - a.x_min, a.y_max - a.y_min),
      new THREE.MeshLambertMaterial({ color: 0xf4f3ef }));
    area.position.set(cx, cy, -0.02);
    this.groundGroup.add(area);
    if (planImage) {
      // Plano de fondo sobre el suelo (z = 0), bajo el mapa de cobertura (que está a la altura del receptor)
      const h = plan.width_m * plan.pixel_height / plan.pixel_width;
      const texture = new THREE.Texture(planImage);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.needsUpdate = true;
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(plan.width_m, h), new THREE.MeshBasicMaterial({
        map: texture, transparent: true, opacity: plan.opacity, depthWrite: false }));
      mesh.position.set(plan.x_m + plan.width_m / 2, plan.y_m + h / 2, -0.005);
      this.groundGroup.add(mesh);
    }
    const z = a.rx_height_m;
    if (layer) {
      const texture = new THREE.CanvasTexture(layer.image);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.magFilter = v.smooth ? THREE.LinearFilter : THREE.NearestFilter;
      texture.minFilter = THREE.LinearFilter;
      texture.generateMipmaps = false;
      const material = new THREE.MeshBasicMaterial({
        map: texture, transparent: true, opacity: v.opacity * (this.store.outdated ? 0.55 : 1), side: THREE.DoubleSide,
        depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1,
      });
      const mesh = v.relief && layer.def.kind === "seq" ? reliefMesh(layer, z, size, material) : flatMesh(layer, z, material);
      this.groundGroup.add(mesh);
    }
    const grid = new THREE.GridHelper(size * 2, 20, 0xb9b8ae, 0xcfcec6);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(cx, cy, -0.01);
    this.groundGroup.add(grid);
    this.requestRender();
  }

  updateRays() {
    if (!this.ok) return;
    disposeGroup(this.rayGroup);
    const probe = this.store.probe;
    if (!probe) { this.requestRender(); return; }
    const { size } = this.areaInfo();
    const [px, py, pz] = probe.point;
    const marker = new THREE.Mesh(new THREE.SphereGeometry(size / 220, 16, 12), new THREE.MeshBasicMaterial({ color: 0x0b0b0b }));
    marker.position.set(px, py, pz);
    const stick = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(px, py, 0), new THREE.Vector3(px, py, pz)]), new THREE.LineBasicMaterial({ color: 0x0b0b0b }));
    this.rayGroup.add(marker, stick);
    if (probe.data) {
      const hover = this.store.view.hoverRay;
      const powers = [];
      for (const t of probe.data.transmitters) for (const r of t.rays) if (!r.blocked) powers.push(numberOf(r.power_dbm));
      const pmax = powers.length ? Math.max(...powers) : 0;
      probe.data.transmitters.forEach((t, ti) => {
        const color = new THREE.Color(txColor(this.store.txSlot(t.tx_id)));
        t.rays.forEach((r, ri) => {
          if (!r.path) return;
          const isHover = hover && hover.tx === ti && hover.ray === ri;
          if (r.blocked && !this.store.view.showBlocked && !isHover) return;
          const points = r.path.map(([x, y, zz]) => new THREE.Vector3(x, y, zz));
          if (r.blocked) {
            const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
              new THREE.LineDashedMaterial({ color: isHover ? 0x0b0b0b : 0x898781, dashSize: size / 150, gapSize: size / 200 }));
            line.computeLineDistances();
            this.rayGroup.add(line);
            return;
          }
          const rel = Math.max(0, Math.min(1, (numberOf(r.power_dbm) - (pmax - 40)) / 40));
          const radius = (size / 700) * (0.5 + 1.5 * rel) * (isHover ? 1.8 : 1);
          const material = new THREE.MeshBasicMaterial({ color: isHover ? new THREE.Color(0x0b0b0b) : color,
            transparent: true, opacity: 0.35 + 0.65 * rel });
          for (let k = 0; k + 1 < points.length; k++) this.rayGroup.add(segment(points[k], points[k + 1], radius, material));
          for (let k = 1; k + 1 < points.length; k++) {
            const atEdge = r.interactions && r.interactions[k - 1] && r.interactions[k - 1][0] === "edge";
            const geometry = atEdge ? new THREE.BoxGeometry(radius * 4, radius * 4, radius * 4)
              : new THREE.SphereGeometry(radius * 2.2, 10, 8);
            const hit = new THREE.Mesh(geometry, material);
            hit.position.copy(points[k]);
            this.rayGroup.add(hit);
          }
        });
      });
    }
    this.requestRender();
  }

  onClick(e) {
    if (this.store.view.tool !== "probe") return;
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -this.store.project.area.rx_height_m);
    const point = new THREE.Vector3();
    if (ray.ray.intersectPlane(plane, point)) this.app.setProbe([Math.round(point.x * 10) / 10, Math.round(point.y * 10) / 10]);
  }

  toPng() {
    if (!this.ok) return null;
    this.renderer.render(this.scene, this.camera);
    return this.canvas.toDataURL("image/png");
  }
}

function flatMesh(layer, z, material) {
  const ext = imageExtent(layer.grid);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(ext.x1 - ext.x0, ext.y1 - ext.y0), material);
  mesh.position.set((ext.x0 + ext.x1) / 2, (ext.y0 + ext.y1) / 2, z);
  return mesh;
}

// Superficie con un vértice por píxel (en su centro) y altura proporcional al valor
function reliefMesh(layer, z, size, material) {
  const { nx, ny, x0, y0, dx, dy } = layer.grid;
  const step = Math.max(1, Math.ceil(Math.sqrt((nx * ny) / 250000)));       // limita el número de vértices
  const cols = Math.floor((nx - 1) / step) + 1;
  const rows = Math.floor((ny - 1) / step) + 1;
  const [lo, hi] = layer.range;
  const span = hi - lo || 1;
  const height = size * 0.05;
  const pos = new Float32Array(cols * rows * 3);
  const uv = new Float32Array(cols * rows * 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = c * step;
      const j = r * step;
      const value = layer.values[j * nx + i];
      const k = r * cols + c;
      pos[3 * k] = x0 + i * dx;
      pos[3 * k + 1] = y0 + j * dy;
      pos[3 * k + 2] = z + (isFinite(value) ? Math.max(0, Math.min(1, (value - lo) / span)) * height : 0);
      uv[2 * k] = (i + 0.5) / nx;
      uv[2 * k + 1] = (j + 0.5) / ny;
    }
  }
  const index = [];
  for (let r = 0; r + 1 < rows; r++) {
    for (let c = 0; c + 1 < cols; c++) {
      const k = r * cols + c;
      index.push(k, k + 1, k + cols + 1, k, k + cols + 1, k + cols);
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geom.setIndex(index);
  return new THREE.Mesh(geom, material);
}

function segment(a, b, radius, material) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const length = dir.length();
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 8, 1), material);
  mesh.position.copy(a).addScaledVector(dir, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return mesh;
}

function textSprite(text, worldHeight) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const font = "600 44px system-ui, sans-serif";
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 24;
  canvas.width = w;
  canvas.height = 64;
  ctx.font = font;
  ctx.fillStyle = "rgba(252, 252, 251, 0.9)";
  ctx.beginPath();
  ctx.roundRect(0, 4, w, 56, 12);
  ctx.fill();
  ctx.fillStyle = "#0b0b0b";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 12, 33);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
  sprite.scale.set(worldHeight * (w / 64), worldHeight, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function disposeGroup(group) {
  for (const child of [...group.children]) {
    group.remove(child);
    child.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of materials) {
          if (m.map && m.map.dispose && !(m.map.image && m.map.image.__shared)) m.map.dispose();
          m.dispose();
        }
      }
    });
  }
}
