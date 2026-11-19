"""
OpenRPS - Project files (scenarios)

A project is a plain JSON document that describes a complete scenario, in the
spirit of an RPS project: the prediction area, the transmitters, 2.5D buildings
(polygon footprint extruded to a height), free-standing walls, a library of
materials and the simulation settings.  It is the file format of the graphical
interface (``open-rps-gui``), and it can also be used from scripts::

    from open_rps.project import Project
    project = Project.load("examples/proyecto_urbano.orps.json")
    result = project.run()
    cir = project.probe((120.0, 80.0))

``Project.build_simulator()`` turns the scenario into a ``PropagationSimulator``:
every edge of a building footprint becomes a finite vertical wall, the roof a
horizontal wall bounded by the footprint polygon, and every material sets the
reflection (constant or Fresnel) and the penetration loss of its walls.

Materials (``Material``) follow ITU-R P.2040 for the electrical properties
(eps_r constant, sigma = c * f_GHz^d S/m) and can use a penetration loss that is
fixed, linear in frequency (L = a + b * f_GHz, the form of 3GPP TR 38.901,
Table 7.4.3-1) or opaque.  Since one wall is shared by all the transmitters, the
frequency-dependent values are evaluated at a single *reference frequency*: the
one of the settings, or the frequency of the first enabled transmitter.
"""

import copy
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import Cost231Hata, FreeSpacePathLoss, LogDistancePathLoss, OkumuraHata, TwoRayGroundReflection
from .multipath import ChannelImpulseResponse
from .raytracer import PlanarReflector, Transmitter
from .simulation import DEFAULT_MIN_DISTANCE_M, PropagationSimulator, SimulationResult

PROJECT_FORMAT = "openrps-project"
PROJECT_VERSION = 1
SIMULATION_MODES = ("raytracing", "fspl", "tworay", "logdistance", "hata")
REFLECTION_MODELS = ("fresnel", "constant")
PENETRATION_MODELS = ("none", "fixed", "frequency", "opaque")
HATA_ENVIRONMENTS = ("urban_large", "urban_medium", "suburban", "open")
# Largest grid accepted by Project.validate (same bound as the CLI)
MAX_GRID_CELLS = 4_000_000


class ProjectError(ValueError):
    """Invalid project: the message says which element and which field is wrong."""


def _number(value, what: str, minimum: Optional[float] = None, strict: bool = False,
            allow_inf: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ProjectError(f"{what}: se esperaba un número, se recibió {value!r}") from None
    if math.isnan(number) or (math.isinf(number) and not allow_inf):
        raise ProjectError(f"{what}: debe ser un número finito, se recibió {value!r}")
    if minimum is not None and (number <= minimum if strict else number < minimum):
        relation = ">" if strict else ">="
        raise ProjectError(f"{what}: debe ser {relation} {minimum:g}, se recibió {number:g}")
    return number


def _choice(value, options: Sequence[str], what: str) -> str:
    if value not in options:
        raise ProjectError(f"{what}: debe ser uno de {', '.join(options)}; se recibió {value!r}")
    return value


def _from_dict(cls, data: Dict[str, Any], what: str):
    """Dataclass ``cls`` from a dict, rejecting unknown keys (typos would be silently ignored)."""
    if not isinstance(data, dict):
        raise ProjectError(f"{what}: se esperaba un objeto JSON, se recibió {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ProjectError(f"{what}: campos desconocidos {', '.join(unknown)}")
    try:
        return cls(**data)
    except TypeError as error:
        raise ProjectError(f"{what}: {error}") from None


# --------------------------------------------------------------------------- materials

@dataclass
class Material:
    """
    Electrical properties of the walls made of this material.

    reflection: "fresnel" (default) uses the angle-dependent Fresnel coefficient of a
        smooth dielectric with ``eps_r`` and sigma = ``sigma_c`` * f_GHz ^ ``sigma_d``
        (ITU-R P.2040, Table 3); "constant" uses ``reflectivity`` (|Gamma|, 0..1,
        fixed 180 deg phase).
    penetration: what happens to a ray that crosses a wall of this material:
        "none" (it passes unattenuated), "fixed" (``loss_db`` dB per crossing),
        "frequency" (``loss_a_db`` + ``loss_b_db_per_ghz`` * f_GHz dB, the linear
        form of 3GPP TR 38.901 Table 7.4.3-1) or "opaque" (it is blocked).
    """
    id: str
    name: str = ""
    color: str = "#9a9a9a"
    reflection: str = "fresnel"
    reflectivity: float = 0.5
    eps_r: float = 5.24
    sigma_c: float = 0.0462
    sigma_d: float = 0.7822
    penetration: str = "fixed"
    loss_db: float = 10.0
    loss_a_db: float = 0.0
    loss_b_db_per_ghz: float = 0.0
    note: str = ""

    def validate(self, what: str = "material"):
        what = f"{what} {self.id!r}"
        if not isinstance(self.id, str) or not self.id:
            raise ProjectError("material: el id debe ser un texto no vacío")
        _choice(self.reflection, REFLECTION_MODELS, f"{what}.reflection")
        _choice(self.penetration, PENETRATION_MODELS, f"{what}.penetration")
        reflectivity = _number(self.reflectivity, f"{what}.reflectivity", 0.0)
        if reflectivity > 1.0:
            raise ProjectError(f"{what}.reflectivity: debe estar entre 0 y 1, se recibió {reflectivity:g}")
        _number(self.eps_r, f"{what}.eps_r", 1.0)
        _number(self.sigma_c, f"{what}.sigma_c", 0.0)
        _number(self.sigma_d, f"{what}.sigma_d")
        _number(self.loss_db, f"{what}.loss_db", 0.0)
        _number(self.loss_a_db, f"{what}.loss_a_db")
        _number(self.loss_b_db_per_ghz, f"{what}.loss_b_db_per_ghz")

    def conductivity(self, frequency_hz: float) -> float:
        """sigma in S/m at ``frequency_hz`` (ITU-R P.2040: c * f_GHz^d)."""
        return float(self.sigma_c) * (frequency_hz / 1e9) ** float(self.sigma_d)

    def penetration_loss_db(self, frequency_hz: float) -> Optional[float]:
        """Loss per crossing in dB at ``frequency_hz``: None (no obstruction), a value >= 0, or inf."""
        if self.penetration == "none":
            return None
        if self.penetration == "opaque":
            return float("inf")
        if self.penetration == "fixed":
            return float(self.loss_db)
        return max(0.0, float(self.loss_a_db) + float(self.loss_b_db_per_ghz) * frequency_hz / 1e9)

    def reflector_kwargs(self, frequency_hz: float) -> Dict[str, Any]:
        """Keyword arguments of ``PlanarReflector`` for a wall of this material."""
        kwargs: Dict[str, Any] = {"transmission_loss_db": self.penetration_loss_db(frequency_hz)}
        if self.reflection == "constant":
            kwargs["reflectivity"] = float(self.reflectivity)
        else:
            kwargs["eps_r"] = float(self.eps_r)
            kwargs["sigma"] = self.conductivity(frequency_hz)
        return kwargs


def default_materials() -> List[Material]:
    """
    Materials library.  eps_r and sigma: ITU-R P.2040-3, Table 3 (valid roughly
    1-100 GHz; brick 1-10 GHz).  Penetration: 3GPP TR 38.901 Table 7.4.3-1 where it
    gives a formula (concrete, glass, IRR glass, wood); typical fixed values otherwise.
    """
    return [
        Material("concreto", "Concreto", "#8d8a83", "fresnel", 0.5, 5.24, 0.0462, 0.7822,
                 "frequency", 15.0, 5.0, 4.0, "P.2040; penetración 3GPP: 5 + 4·f dB"),
        Material("ladrillo", "Ladrillo", "#b0603c", "fresnel", 0.5, 3.91, 0.0238, 0.16,
                 "fixed", 10.0, 0.0, 0.0, "P.2040 (1–10 GHz); penetración típica 10 dB"),
        Material("vidrio", "Vidrio", "#7fb7d9", "fresnel", 0.5, 6.31, 0.0036, 1.3394,
                 "frequency", 2.5, 2.0, 0.2, "P.2040; penetración 3GPP: 2 + 0.2·f dB"),
        Material("vidrio_irr", "Vidrio IRR (bajo emisivo)", "#4f8fb3", "fresnel", 0.5, 6.31, 0.0036, 1.3394,
                 "frequency", 23.7, 23.0, 0.3, "Penetración 3GPP: 23 + 0.3·f dB"),
        Material("madera", "Madera", "#a47844", "fresnel", 0.5, 1.99, 0.0047, 1.0718,
                 "frequency", 5.1, 4.85, 0.12, "P.2040; penetración 3GPP: 4.85 + 0.12·f dB"),
        Material("tablaroca", "Tablaroca", "#d8d3c6", "fresnel", 0.5, 2.73, 0.0085, 0.9395,
                 "fixed", 3.0, 0.0, 0.0, "P.2040; penetración típica 3 dB"),
        Material("metal", "Metal", "#5c6670", "constant", 1.0, 1.0, 1e7, 0.0,
                 "opaque", 0.0, 0.0, 0.0, "Reflector perfecto y opaco"),
        Material("ideal", "Muro ideal (solo refleja)", "#b9b4a8", "constant", 0.5, 5.0, 0.0, 0.0,
                 "none", 0.0, 0.0, 0.0, "Comportamiento de OpenRPS 1.2: refleja y no tapa"),
    ]


# --------------------------------------------------------------------------- geometry

def _clean_polygon(points, what: str) -> List[List[float]]:
    """Vertices as [[x, y], ...]: finite, consecutive duplicates and closing vertex removed, >= 3."""
    if not isinstance(points, (list, tuple)):
        raise ProjectError(f"{what}: se esperaba una lista de vértices [x, y]")
    polygon: List[List[float]] = []
    for k, vertex in enumerate(points):
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            raise ProjectError(f"{what}: el vértice {k} debe ser [x, y]")
        point = [_number(vertex[0], f"{what}[{k}].x"), _number(vertex[1], f"{what}[{k}].y")]
        if not polygon or point != polygon[-1]:
            polygon.append(point)
    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon.pop()
    if len(polygon) < 3:
        raise ProjectError(f"{what}: un polígono necesita al menos 3 vértices distintos")
    if abs(polygon_area(polygon)) == 0.0:
        raise ProjectError(f"{what}: el polígono tiene área nula")
    if _self_intersects(polygon):
        raise ProjectError(f"{what}: el polígono se cruza consigo mismo")
    return polygon


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """Signed area (shoelace formula): > 0 counter-clockwise."""
    n = len(polygon)
    return 0.5 * sum(polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]
                     for i in range(n))


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if value == 0 else (1 if value > 0 else -1)

    def on_segment(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on_segment(p1, p2, q1)) or (o2 == 0 and on_segment(p1, p2, q2))
            or (o3 == 0 and on_segment(q1, q2, p1)) or (o4 == 0 and on_segment(q1, q2, p2)))


def _self_intersects(polygon) -> bool:
    n = len(polygon)
    edges = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue                                   # adjacent edges share a vertex
            if _segments_intersect(*edges[i], *edges[j]):
                return True
    return False


def vertical_wall(start: Sequence[float], end: Sequence[float], z_min: float, z_max: float,
                  name: str, **material_kwargs) -> PlanarReflector:
    """
    Finite vertical wall over the segment start -> end (x, y), from z_min to z_max.
    The axis-aligned box of the segment limits the patch exactly, also for oblique
    walls: the vertical plane of the segment meets that box only along the segment.
    """
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0.0:
        raise ProjectError(f"{name}: el muro tiene longitud nula")
    return PlanarReflector(p0=[x0, y0, 0.0], normal=[dy / length, -dx / length, 0.0], name=name,
                           bounds=(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), z_min, z_max),
                           **material_kwargs)


@dataclass
class Building:
    """2.5D building: polygon footprint (x, y in meters) extruded from the ground to ``height_m``."""
    id: str
    name: str = ""
    footprint: List[List[float]] = field(default_factory=list)
    height_m: float = 10.0
    material: str = "concreto"
    roof: bool = True

    def validate(self, materials: Dict[str, Material], what: str = "edificio"):
        what = f"{what} {self.name or self.id!r}"
        self.footprint = _clean_polygon(self.footprint, f"{what}.footprint")
        self.height_m = _number(self.height_m, f"{what}.height_m", 0.0, strict=True)
        if self.material not in materials:
            raise ProjectError(f"{what}: material desconocido {self.material!r}")
        self.roof = bool(self.roof)

    def reflectors(self, material: Material, frequency_hz: float) -> List[PlanarReflector]:
        """One finite vertical wall per edge of the footprint, plus the roof if ``roof``."""
        kwargs = material.reflector_kwargs(frequency_hz)
        label = self.name or self.id
        walls = []
        n = len(self.footprint)
        for k in range(n):
            start, end = self.footprint[k], self.footprint[(k + 1) % n]
            walls.append(vertical_wall(start, end, 0.0, self.height_m, f"{label} · fachada {k + 1}", **kwargs))
        if self.roof:
            x0, y0 = self.footprint[0]
            walls.append(PlanarReflector(p0=[x0, y0, self.height_m], normal=[0.0, 0.0, 1.0],
                                         name=f"{label} · techo", footprint=self.footprint, **kwargs))
        return walls

    def contains_xy(self, x, y) -> np.ndarray:
        """True where the points (x, y) are inside the footprint (edges included)."""
        from .raytracer import _in_polygon
        return _in_polygon(x, y, np.asarray(self.footprint, dtype=float))


@dataclass
class Wall:
    """Free-standing vertical wall over the segment ``start`` -> ``end``, from ``z_min_m`` to ``z_max_m``."""
    id: str
    name: str = ""
    start: List[float] = field(default_factory=lambda: [0.0, 0.0])
    end: List[float] = field(default_factory=lambda: [10.0, 0.0])
    z_min_m: float = 0.0
    z_max_m: float = 3.0
    material: str = "ladrillo"

    def validate(self, materials: Dict[str, Material], what: str = "muro"):
        what = f"{what} {self.name or self.id!r}"
        for attr in ("start", "end"):
            point = getattr(self, attr)
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ProjectError(f"{what}.{attr}: debe ser [x, y]")
            setattr(self, attr, [_number(point[0], f"{what}.{attr}.x"), _number(point[1], f"{what}.{attr}.y")])
        if self.start == self.end:
            raise ProjectError(f"{what}: el muro tiene longitud nula")
        self.z_min_m = _number(self.z_min_m, f"{what}.z_min_m")
        self.z_max_m = _number(self.z_max_m, f"{what}.z_max_m")
        if self.z_max_m <= self.z_min_m:
            raise ProjectError(f"{what}: z_max_m debe ser mayor que z_min_m")
        if self.material not in materials:
            raise ProjectError(f"{what}: material desconocido {self.material!r}")

    def reflector(self, material: Material, frequency_hz: float) -> PlanarReflector:
        return vertical_wall(self.start, self.end, self.z_min_m, self.z_max_m, self.name or self.id,
                             **material.reflector_kwargs(frequency_hz))


# --------------------------------------------------------------------------- transmitters, area, settings

@dataclass
class TransmitterSpec:
    """A ``Transmitter`` plus the fields of the interface (name, enabled)."""
    tx_id: int
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 25.0
    tx_power_dbm: float = 43.0
    frequency_hz: float = 2.4e9
    antenna_gain_dbi: float = 14.0
    pattern: str = "isotropic"
    azimuth_deg: float = 0.0
    beamwidth_az_deg: float = 65.0
    beamwidth_el_deg: float = 65.0
    downtilt_deg: float = 0.0
    enabled: bool = True

    def validate(self, what: str = "transmisor"):
        what = f"{what} {self.name or self.tx_id!r}"
        if isinstance(self.tx_id, bool) or not isinstance(self.tx_id, int) or self.tx_id < 0:
            raise ProjectError(f"{what}.tx_id: debe ser un entero >= 0")
        for name in ("x", "y", "tx_power_dbm", "antenna_gain_dbi", "azimuth_deg", "downtilt_deg"):
            setattr(self, name, _number(getattr(self, name), f"{what}.{name}"))
        self.z = _number(self.z, f"{what}.z", 0.0)
        self.frequency_hz = _number(self.frequency_hz, f"{what}.frequency_hz", 0.0, strict=True)
        for name in ("beamwidth_az_deg", "beamwidth_el_deg"):
            setattr(self, name, _number(getattr(self, name), f"{what}.{name}", 0.0, strict=True))
        _choice(self.pattern, ("isotropic", "sector"), f"{what}.pattern")
        self.enabled = bool(self.enabled)

    def to_transmitter(self) -> Transmitter:
        return Transmitter(tx_id=self.tx_id, x=self.x, y=self.y, z=self.z, tx_power_dbm=self.tx_power_dbm,
                           frequency_hz=self.frequency_hz, antenna_gain_dbi=self.antenna_gain_dbi,
                           azimuth_deg=self.azimuth_deg, beamwidth_az_deg=self.beamwidth_az_deg,
                           beamwidth_el_deg=self.beamwidth_el_deg, downtilt_deg=self.downtilt_deg,
                           pattern=self.pattern)


@dataclass
class Area:
    """Prediction area: rectangle, grid step and receiver height (all in meters)."""
    x_min: float = 0.0
    x_max: float = 500.0
    y_min: float = 0.0
    y_max: float = 500.0
    resolution_m: float = 5.0
    rx_height_m: float = 1.5

    def validate(self):
        for name in ("x_min", "x_max", "y_min", "y_max"):
            setattr(self, name, _number(getattr(self, name), f"area.{name}"))
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ProjectError("area: x_max y y_max deben ser mayores que x_min y y_min")
        self.resolution_m = _number(self.resolution_m, "area.resolution_m", 0.0, strict=True)
        self.rx_height_m = _number(self.rx_height_m, "area.rx_height_m", 0.0)
        cells = self.cells()
        if cells > MAX_GRID_CELLS:
            raise ProjectError(f"area: la grilla tiene {cells} píxeles (máximo {MAX_GRID_CELLS}); "
                               "aumenta la resolución (resolution_m) o reduce el área")

    def cells(self) -> int:
        nx = int(math.floor((self.x_max - self.x_min) / self.resolution_m + 1e-9)) + 1
        ny = int(math.floor((self.y_max - self.y_min) / self.resolution_m + 1e-9)) + 1
        return nx * ny


@dataclass
class Settings:
    """Simulation settings (same meaning as the arguments of ``run_raytracing_simulation``)."""
    mode: str = "raytracing"
    coherent: bool = False
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M
    shadow_sigma_db: float = 0.0
    seed: Optional[int] = None
    hata_environment: str = "urban_medium"
    log_distance_exponent: float = 3.0
    log_distance_d0_m: float = 1.0
    sinr: bool = True
    noise_figure_db: float = 7.0
    bandwidth_hz: float = 20e6
    ground_eps_r: float = 15.0
    ground_sigma: float = 0.005
    reference_frequency_hz: Optional[float] = None

    def validate(self):
        _choice(self.mode, SIMULATION_MODES, "settings.mode")
        _choice(self.hata_environment, HATA_ENVIRONMENTS, "settings.hata_environment")
        self.coherent = bool(self.coherent)
        self.sinr = bool(self.sinr)
        self.min_distance_m = _number(self.min_distance_m, "settings.min_distance_m", 0.0)
        self.shadow_sigma_db = _number(self.shadow_sigma_db, "settings.shadow_sigma_db", 0.0)
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0):
            raise ProjectError("settings.seed: debe ser null o un entero >= 0")
        self.log_distance_exponent = _number(self.log_distance_exponent, "settings.log_distance_exponent", 0.0, True)
        self.log_distance_d0_m = _number(self.log_distance_d0_m, "settings.log_distance_d0_m", 0.0, True)
        self.noise_figure_db = _number(self.noise_figure_db, "settings.noise_figure_db")
        self.bandwidth_hz = _number(self.bandwidth_hz, "settings.bandwidth_hz", 0.0, True)
        self.ground_eps_r = _number(self.ground_eps_r, "settings.ground_eps_r", 1.0)
        self.ground_sigma = _number(self.ground_sigma, "settings.ground_sigma", 0.0)
        if self.reference_frequency_hz is not None:
            self.reference_frequency_hz = _number(self.reference_frequency_hz,
                                                  "settings.reference_frequency_hz", 0.0, True)


# --------------------------------------------------------------------------- project

@dataclass
class ProbeResult:
    """Rays and channel metrics at one receiver point (see ``Project.probe``)."""
    point: Tuple[float, float, float]
    mode: str
    per_transmitter: List[Dict[str, Any]]
    reflector_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({"point": list(self.point), "mode": self.mode,
                           "transmitters": self.per_transmitter})


def _json_safe(value):
    """Replace NaN / +-inf by None (or "inf" / "-inf" for infinite losses) so that the output is strict JSON."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass
class Project:
    """A complete scenario.  ``Project.from_dict`` / ``to_dict`` map it to and from JSON."""
    name: str = "Proyecto OpenRPS"
    description: str = ""
    area: Area = field(default_factory=Area)
    settings: Settings = field(default_factory=Settings)
    materials: List[Material] = field(default_factory=default_materials)
    transmitters: List[TransmitterSpec] = field(default_factory=list)
    buildings: List[Building] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)

    # ----- (de)serialization
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        if not isinstance(data, dict):
            raise ProjectError("proyecto: se esperaba un objeto JSON")
        fmt = data.get("format", PROJECT_FORMAT)
        if fmt != PROJECT_FORMAT:
            raise ProjectError(f"proyecto: formato {fmt!r} desconocido (se esperaba {PROJECT_FORMAT!r})")
        version = data.get("version", PROJECT_VERSION)
        if not isinstance(version, int) or version > PROJECT_VERSION:
            raise ProjectError(f"proyecto: versión {version!r} no soportada (esta versión lee hasta la {PROJECT_VERSION})")
        known = {"format", "version", "name", "description", "area", "settings", "materials",
                 "transmitters", "buildings", "walls"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ProjectError(f"proyecto: campos desconocidos {', '.join(unknown)}")
        materials = data.get("materials")
        project = cls(
            name=str(data.get("name", "Proyecto OpenRPS")),
            description=str(data.get("description", "")),
            area=_from_dict(Area, data.get("area", {}), "area"),
            settings=_from_dict(Settings, data.get("settings", {}), "settings"),
            materials=default_materials() if materials is None
            else [_from_dict(Material, m, "material") for m in materials],
            transmitters=[_from_dict(TransmitterSpec, t, "transmisor") for t in data.get("transmitters", [])],
            buildings=[_from_dict(Building, b, "edificio") for b in data.get("buildings", [])],
            walls=[_from_dict(Wall, w, "muro") for w in data.get("walls", [])],
        )
        project.validate()
        return project

    def to_dict(self) -> Dict[str, Any]:
        data = {"format": PROJECT_FORMAT, "version": PROJECT_VERSION, "name": self.name,
                "description": self.description, "area": asdict(self.area), "settings": asdict(self.settings),
                "materials": [asdict(m) for m in self.materials],
                "transmitters": [asdict(t) for t in self.transmitters],
                "buildings": [asdict(b) for b in self.buildings],
                "walls": [asdict(w) for w in self.walls]}
        return copy.deepcopy(data)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")

    # ----- validation
    def validate(self) -> "Project":
        """Check every element (raises ``ProjectError``) and normalize the values in place."""
        self.area.validate()
        self.settings.validate()
        ids = [m.id for m in self.materials]
        if len(set(ids)) != len(ids):
            raise ProjectError("materiales: hay ids repetidos")
        for material in self.materials:
            material.validate()
        materials = self.material_map()
        tx_ids = [t.tx_id for t in self.transmitters]
        if len(set(tx_ids)) != len(tx_ids):
            raise ProjectError("transmisores: hay tx_id repetidos (cada transmisor necesita uno distinto)")
        for tx in self.transmitters:
            tx.validate()
        for building in self.buildings:
            building.validate(materials)
        for wall in self.walls:
            wall.validate(materials)
        return self

    def material_map(self) -> Dict[str, Material]:
        return {m.id: m for m in self.materials}

    def enabled_transmitters(self) -> List[TransmitterSpec]:
        return [t for t in self.transmitters if t.enabled]

    def reference_frequency_hz(self) -> float:
        """Frequency at which the frequency-dependent material properties are evaluated."""
        if self.settings.reference_frequency_hz is not None:
            return float(self.settings.reference_frequency_hz)
        enabled = self.enabled_transmitters()
        return float(enabled[0].frequency_hz) if enabled else 2.4e9

    # ----- simulation
    def build_simulator(self) -> PropagationSimulator:
        """``PropagationSimulator`` with the enabled transmitters and every wall of the scenario."""
        self.validate()
        sim = PropagationSimulator()
        sim.ray_tracer.ground_eps_r = self.settings.ground_eps_r
        sim.ray_tracer.ground_sigma = self.settings.ground_sigma
        for tx in self.enabled_transmitters():
            sim.add_transmitter(tx.to_transmitter())
        materials = self.material_map()
        frequency = self.reference_frequency_hz()
        for building in self.buildings:
            for wall in building.reflectors(materials[building.material], frequency):
                sim.add_reflector(wall)
        for wall in self.walls:
            sim.add_reflector(wall.reflector(materials[wall.material], frequency))
        return sim

    def empirical_model(self):
        """Propagation model of the empirical modes (None in ray-tracing mode)."""
        s = self.settings
        if s.mode == "fspl":
            return FreeSpacePathLoss()
        if s.mode == "tworay":
            return TwoRayGroundReflection(eps_r=s.ground_eps_r, sigma=s.ground_sigma)
        if s.mode == "logdistance":
            return LogDistancePathLoss(path_loss_exponent=s.log_distance_exponent, d0_m=s.log_distance_d0_m)
        if s.mode == "hata":
            # Okumura-Hata up to 1500 MHz and COST-231 Hata above, as in the CLI
            if self.reference_frequency_hz() <= 1500e6:
                return OkumuraHata(s.hata_environment)
            # COST-231 has no "open" (rural) environment: the closest one is suburban
            env = s.hata_environment if s.hata_environment in Cost231Hata.ENVIRONMENTS else "suburban"
            return Cost231Hata(env)
        return None

    def run(self, sim: Optional[PropagationSimulator] = None) -> SimulationResult:
        """Run the simulation of the project over its area (warnings of the models are emitted normally)."""
        sim = sim or self.build_simulator()
        a, s = self.area, self.settings
        area = ((a.x_min, a.x_max), (a.y_min, a.y_max))
        common = dict(resolution_m=a.resolution_m, rx_height_m=a.rx_height_m,
                      min_distance_m=s.min_distance_m, shadow_fading_sigma_db=s.shadow_sigma_db, seed=s.seed)
        model = self.empirical_model()
        if model is None:
            result = sim.run_raytracing_simulation(*area, coherent=s.coherent, **common)
        else:
            result = sim.run_empirical_simulation(model, *area, **common)
        if s.sinr and sim.transmitters:
            result.compute_sinr(noise_figure_db=s.noise_figure_db, bandwidth_hz=s.bandwidth_hz)
        return result

    def run_with_report(self) -> Tuple[SimulationResult, Dict[str, Any]]:
        """``run()`` plus a report: elapsed time, warnings of the models, counts of the scene."""
        start = time.perf_counter()
        sim = self.build_simulator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.run(sim)
        report = {
            "elapsed_s": time.perf_counter() - start,
            "warnings": list(dict.fromkeys(str(w.message) for w in caught)),
            "transmitters": len(sim.transmitters),
            "reflectors": len(sim.ray_tracer.reflectors),
            "obstructing_reflectors": sum(1 for r in sim.ray_tracer.reflectors if r.obstructs),
            "reference_frequency_hz": self.reference_frequency_hz(),
            "mode": self.settings.mode,
        }
        return result, report

    def probe(self, point: Sequence[float], sim: Optional[PropagationSimulator] = None) -> ProbeResult:
        """
        Rays and channel metrics at one receiver point (x, y) or (x, y, z); z defaults
        to the receiver height of the area.  No shadow fading is applied: the values
        are the deterministic part of the prediction.

        Ray tracing: every ray of every enabled transmitter, with its geometry
        (``path``), power, delay, angles and obstruction loss, and the power-delay
        metrics of the point.  Empirical modes: the power predicted by the model.
        """
        if len(point) not in (2, 3):
            raise ProjectError("probe: el punto debe ser [x, y] o [x, y, z]")
        x, y = _number(point[0], "probe.x"), _number(point[1], "probe.y")
        z = _number(point[2], "probe.z", 0.0) if len(point) == 3 else float(self.area.rx_height_m)
        sim = sim or self.build_simulator()
        names = [r.name for r in sim.ray_tracer.reflectors]
        model = self.empirical_model()
        entries = []
        for tx in sim.transmitters:
            distance = math.dist((tx.x, tx.y, tx.z), (x, y, z))
            entry: Dict[str, Any] = {"tx_id": tx.tx_id, "distance_m": distance,
                                     "too_close": bool(self.settings.min_distance_m > 0
                                                       and distance < self.settings.min_distance_m)}
            if model is None:
                cir = sim.ray_tracer.trace_paths(tx, (x, y, z))
                entry.update(_cir_metrics(cir, tx, self.settings.coherent))
                entry["rays"] = [_ray_dict(ray, names) for ray in cir.rays]
            else:
                entry.update(_empirical_point(model, tx, x, y, z))
                entry["rays"] = []
            entries.append(entry)
        return ProbeResult(point=(x, y, z), mode=self.settings.mode, per_transmitter=entries, reflector_names=names)


def _cir_metrics(cir: ChannelImpulseResponse, tx: Transmitter, coherent: bool) -> Dict[str, Any]:
    p_incoh = cir.total_received_power_dbm()
    p_coh = cir.total_received_power_dbm(coherent=True)
    power = p_coh if coherent else p_incoh
    mean_bs, spread_bs = cir.angular_spread(plane="phi", station="bs")
    mean_ms, spread_ms = cir.angular_spread(plane="phi", station="ms")
    return {
        "rx_power_dbm": power,
        "rx_power_incoherent_dbm": p_incoh,
        "rx_power_coherent_dbm": p_coh,
        "path_loss_db": tx.tx_power_dbm - power if power > float("-inf") else float("nan"),
        "mean_delay_ns": cir.mean_excess_delay(),
        "mean_excess_delay_ns": cir.mean_excess_delay(relative_to_first=True),
        "rms_delay_spread_ns": cir.rms_delay_spread(),
        "aod_mean_deg": mean_bs, "aod_spread_deg": spread_bs,
        "aoa_mean_deg": mean_ms, "aoa_spread_deg": spread_ms,
    }


def _ray_dict(ray, names: List[str]) -> Dict[str, Any]:
    if ray.kind == "direct":
        label = "Directo (LoS)"
    elif ray.kind == "ground":
        label = "Reflexión en el suelo"
    elif 0 <= ray.reflector_index < len(names):
        label = f"Reflexión: {names[ray.reflector_index]}"
    else:
        label = "Reflexión"
    power = ray.power_linear
    return {
        "kind": ray.kind, "label": label, "reflector_index": ray.reflector_index,
        "path": [list(p) for p in ray.path] if ray.path else None,
        "power_dbm": 10.0 * math.log10(power) if power > 0 else float("-inf"),
        "phase_deg": math.degrees(math.atan2(ray.magnitude.imag, ray.magnitude.real)),
        "tau_ns": ray.tau_ns,
        "aod_phi_deg": ray.tx_phi_deg, "aod_theta_deg": ray.tx_theta_deg,
        "aoa_phi_deg": ray.rx_phi_deg, "aoa_theta_deg": ray.rx_theta_deg,
        "obstruction_loss_db": ray.obstruction_loss_db,
        "blocked": power == 0.0,
    }


def _empirical_point(model, tx: Transmitter, x: float, y: float, z: float) -> Dict[str, Any]:
    """Same formula as ``run_empirical_simulation`` for one point (without shadow fading)."""
    ground = math.hypot(x - tx.x, y - tx.y)
    distance = ground if getattr(model, "uses_ground_distance", False) else math.sqrt(ground ** 2 + (z - tx.z) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl = float(model.path_loss(distance_m=distance, frequency_hz=tx.frequency_hz, ht_m=tx.z, hr_m=max(z, 1e-3),
                                   tx_gain_dbi=tx.antenna_gain_dbi))
    if not tx.is_isotropic:
        phi = math.degrees(math.atan2(y - tx.y, x - tx.x))
        theta = math.degrees(math.atan2(z - tx.z, max(1e-6, ground)))
        pl -= tx.pattern_gain_db(phi, theta)
    return {"rx_power_dbm": tx.tx_power_dbm - pl, "path_loss_db": pl}


# --------------------------------------------------------------------------- examples

def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def example_urban() -> Project:
    """Urban microcell: a few blocks of buildings and two sector base stations at 2.4 GHz."""
    return Project(
        name="Microcelda urbana",
        description="Manzanas de edificios con fachadas de concreto y vidrio y dos estaciones base "
                    "sectoriales a 2.4 GHz por encima de los techos más bajos.",
        area=Area(0.0, 600.0, 0.0, 400.0, 4.0, 1.5),
        settings=Settings(mode="raytracing"),
        transmitters=[
            TransmitterSpec(0, "BS Norte", 150.0, 300.0, 30.0, 40.0, 2.4e9, 15.0, "sector", -45.0, 65.0, 10.0, 6.0),
            TransmitterSpec(1, "BS Sur", 470.0, 95.0, 25.0, 40.0, 2.4e9, 15.0, "sector", 135.0, 65.0, 10.0, 6.0),
        ],
        buildings=[
            Building("b1", "Torre A", _rect(60, 60, 150, 140), 35.0, "concreto"),
            Building("b2", "Bloque B", _rect(200, 50, 300, 150), 18.0, "ladrillo"),
            Building("b3", "Edificio L", [[340, 40], [430, 40], [430, 80], [380, 80], [380, 160], [340, 160]],
                     22.0, "concreto"),
            Building("b4", "Oficinas", _rect(60, 200, 130, 270), 28.0, "vidrio"),
            Building("b5", "Plaza comercial", [[220, 210], [320, 200], [340, 280], [240, 300]], 12.0, "concreto"),
            Building("b6", "Torre C", _rect(400, 230, 520, 330), 40.0, "vidrio"),
            Building("b7", "Bodega", _rect(520, 40, 580, 170), 9.0, "metal"),
        ],
        walls=[Wall("w1", "Barda del parque", [170.0, 340.0], [360.0, 380.0], 0.0, 3.0, "ladrillo")],
    )


def example_office() -> Project:
    """Indoor: one floor of offices with plasterboard partitions and a Wi-Fi access point at 5.2 GHz."""
    walls = []

    def add(name, start, end, material="tablaroca"):
        walls.append(Wall(f"w{len(walls) + 1}", name, list(start), list(end), 0.0, 3.0, material))

    # Outer walls (concrete) and partitions (plasterboard), 40 m x 20 m floor
    add("Fachada sur", (0, 0), (40, 0), "concreto")
    add("Fachada este", (40, 0), (40, 20), "concreto")
    add("Fachada norte", (40, 20), (0, 20), "vidrio")
    add("Fachada oeste", (0, 20), (0, 0), "concreto")
    add("Pasillo sur", (0, 8), (17, 8))
    add("Pasillo sur 2", (19, 8), (40, 8))
    add("Pasillo norte", (0, 12), (25, 12))
    add("Pasillo norte 2", (27, 12), (40, 12))
    for k, x in enumerate((10, 20, 30)):
        add(f"División sur {k + 1}", (x, 0), (x, 8))
    for k, x in enumerate((13, 26)):
        add(f"División norte {k + 1}", (x, 12), (x, 20))
    add("Cubo de elevador", (33, 12), (33, 20), "metal")
    return Project(
        name="Oficina (interior)",
        description="Una planta de oficinas de 40 m × 20 m con divisiones de tablaroca y un punto de acceso Wi-Fi "
                    "a 5.2 GHz en el pasillo. Los muros llegan al techo (3 m).",
        area=Area(-2.0, 42.0, -2.0, 22.0, 0.25, 1.0),
        settings=Settings(mode="raytracing", min_distance_m=0.5, bandwidth_hz=20e6),
        transmitters=[TransmitterSpec(0, "AP pasillo", 12.0, 10.0, 2.8, 20.0, 5.2e9, 3.0)],
        walls=walls,
    )


EXAMPLES = {"urbano": example_urban, "oficina": example_office}
