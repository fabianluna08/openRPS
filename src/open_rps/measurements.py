"""
Measured received power compared with the prediction, and calibration of the
scenario to fit it (new in 1.7.0).

A measurement is a point (x, y[, z]) where a receiver read ``rssi_dbm`` from a
transmitter (``tx_id``) or, if no transmitter is given, from the strongest one.
``compare`` evaluates the prediction of a project exactly at those points (no grid
interpolation, no random shadow fading) and gives the error of each point,
predicted - measured, and its statistics.  ``calibrate`` adjusts the parameters
that are usually unknown in a real deployment so that the prediction fits the
measurements in the least-squares sense:

* the penetration loss of the materials (ray tracing),
* the power of the transmitters, one offset each or a common one,
* the path-loss exponent (log-distance mode).

Measurements are read from CSV files with ``read_csv``.
"""

import copy
import csv
import io
import math
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Measurement", "MeasurementError", "read_csv", "predict_points", "compare", "calibrate",
           "Comparison", "PointComparison", "CalibrationResult", "CalibrationParameter"]

# Search space of the parameters
MATERIAL_LOSS_BOUNDS_DB = (0.0, 60.0)
TX_OFFSET_BOUNDS_DB = (-40.0, 40.0)
EXPONENT_BOUNDS = (1.0, 8.0)
MIN_POINTS_PER_PARAMETER = 2     # at least this many usable points per parameter, plus one
MIN_DEPENDENT_POINTS = 3         # a parameter that fewer measured points depend on is not fitted


class MeasurementError(ValueError):
    """Invalid measurement data or calibration request (message in Spanish, for the user)."""


@dataclass
class Measurement:
    """One measured point: received power ``rssi_dbm`` at (x, y, z) from transmitter ``tx_id``."""
    x: float
    y: float
    rssi_dbm: float
    z: Optional[float] = None       # None: the receiver height of the project area
    tx_id: Optional[int] = None     # None: compared with the strongest transmitter (best server)
    label: str = ""

    def validate(self, what: str = "medición"):
        for name in ("x", "y", "rssi_dbm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise MeasurementError(f"{what}.{name}: debe ser un número finito, se recibió {value!r}")
            setattr(self, name, float(value))
        if self.z is not None:
            if isinstance(self.z, bool) or not isinstance(self.z, (int, float)) or not math.isfinite(self.z) \
                    or self.z < 0:
                raise MeasurementError(f"{what}.z: debe ser null o un número >= 0, se recibió {self.z!r}")
            self.z = float(self.z)
        if self.tx_id is not None and (isinstance(self.tx_id, bool) or not isinstance(self.tx_id, int)
                                       or self.tx_id < 0):
            raise MeasurementError(f"{what}.tx_id: debe ser null o un entero >= 0, se recibió {self.tx_id!r}")
        self.label = str(self.label or "")
        return self


# --------------------------------------------------------------------------- CSV

_COLUMNS = {
    "x": ("x", "x_m", "este", "east"),
    "y": ("y", "y_m", "norte", "north"),
    "z": ("z", "z_m", "altura", "height", "h"),
    "rssi_dbm": ("rssi", "rssi_dbm", "dbm", "potencia", "potencia_dbm", "power", "power_dbm", "senal",
                 "senal_dbm", "signal", "signal_dbm", "nivel", "level", "medido", "measured"),
    "tx": ("tx", "tx_id", "transmisor", "transmitter", "ap", "bs", "antena", "celda", "cell"),
    "label": ("etiqueta", "label", "nota", "note", "nombre", "name", "punto", "point", "id"),
}


def _plain(text: str) -> str:
    """Lower case, no accents, '_' for spaces and units in brackets removed: 'Señal (dBm)' -> 'senal_dbm'."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().strip().lower()
    for ch in "()[]{}":
        text = text.replace(ch, " ")
    return "_".join(text.replace("-", " ").replace(".", " ").split())


def _to_float(text: str, decimal_comma: bool, what: str) -> float:
    s = text.strip().replace("−", "-")
    if decimal_comma:
        s = s.replace(".", "").replace(",", ".") if s.count(",") == 1 else s
    try:
        value = float(s)
    except ValueError:
        raise MeasurementError(f"{what}: {text!r} no es un número") from None
    if not math.isfinite(value):
        raise MeasurementError(f"{what}: {text!r} no es un número finito")
    return value


def read_csv(source, transmitters: Sequence[Any] = ()) -> List[Measurement]:
    """
    Read measurements from a CSV file (path, file object or text).

    The first row names the columns; names are matched without accents or case, and
    a unit in brackets is ignored ("Señal (dBm)").  Required: x, y and the power
    (rssi, dbm, potencia, señal, signal...).  Optional: z or altura (receiver
    height, default the one of the project), tx (transmitter: its tx_id or its name
    in ``transmitters``, a list of objects with ``tx_id`` and ``name``) and a label.
    The separator (",", ";" or tab) is detected; with ";" or tab a decimal comma is
    accepted too ("-67,5").
    """
    if hasattr(source, "read"):
        text = source.read()
    elif isinstance(source, (bytes, bytearray)) or (isinstance(source, str) and "\n" in source):
        text = source
    else:
        with open(source, encoding="utf-8-sig", newline="") as f:
            text = f.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    text = text.lstrip("﻿")
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) < 2:
        raise MeasurementError("CSV: hace falta una fila de encabezados y al menos una medición")
    header_line = lines[0]
    delimiter = max((";", "\t", ","), key=lambda d: header_line.count(d))
    if header_line.count(delimiter) == 0:
        raise MeasurementError("CSV: no se encontró el separador (coma, punto y coma o tabulador) en los encabezados")
    decimal_comma = delimiter != ","
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
    header = [_plain(h) for h in rows[0]]
    index: Dict[str, int] = {}
    for key, names in _COLUMNS.items():
        for i, h in enumerate(header):
            if h in names and key not in index:
                index[key] = i
    missing = [name for name in ("x", "y", "rssi_dbm") if name not in index]
    if missing:
        nice = {"x": "x", "y": "y", "rssi_dbm": "potencia (rssi, dbm, potencia, señal...)"}
        raise MeasurementError("CSV: faltan las columnas " + ", ".join(nice[m] for m in missing)
                               + f" (encabezados leídos: {', '.join(rows[0])})")
    by_name = {_plain(getattr(t, "name", "") or ""): t.tx_id for t in transmitters if getattr(t, "name", "")}
    known_ids = {t.tx_id for t in transmitters}
    result = []
    for line_no, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        what = f"CSV, fila {line_no}"

        def cell(key):
            i = index.get(key)
            return row[i].strip() if i is not None and i < len(row) else ""

        for key in ("x", "y", "rssi_dbm"):
            if not cell(key):
                raise MeasurementError(f"{what}: falta el valor de la columna {rows[0][index[key]]!r}")
        m = Measurement(x=_to_float(cell("x"), decimal_comma, f"{what}, x"),
                        y=_to_float(cell("y"), decimal_comma, f"{what}, y"),
                        rssi_dbm=_to_float(cell("rssi_dbm"), decimal_comma, f"{what}, potencia"),
                        label=cell("label"))
        if cell("z"):
            m.z = _to_float(cell("z"), decimal_comma, f"{what}, z")
        tx = cell("tx")
        if tx:
            if _plain(tx) in by_name:
                m.tx_id = by_name[_plain(tx)]
            else:
                try:
                    m.tx_id = int(_to_float(tx, decimal_comma, f"{what}, tx"))
                except MeasurementError:
                    raise MeasurementError(f"{what}: no hay ningún transmisor llamado {tx!r}") from None
                if known_ids and m.tx_id not in known_ids:
                    raise MeasurementError(f"{what}: no hay ningún transmisor con tx_id {m.tx_id}")
        result.append(m.validate(what))
    if not result:
        raise MeasurementError("CSV: no hay mediciones")
    return result


# --------------------------------------------------------------------------- prediction at points

def _points(project, measurements: Sequence[Measurement]) -> np.ndarray:
    z0 = float(project.area.rx_height_m)
    return np.array([[m.x, m.y, z0 if m.z is None else m.z] for m in measurements], dtype=float).reshape(-1, 3)


def predict_points(project, points, sim=None) -> np.ndarray:
    """
    Received power in dBm of every enabled transmitter at each point ([num_tx, N]):
    the same computation as the maps of ``Project.run`` but at exact positions and
    without shadow fading.  -inf where no ray arrives, NaN closer to the antenna than
    ``settings.min_distance_m``.
    """
    from .simulation import _MAX_CHUNK_POINTS, _MAX_RAY_POINTS
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    sim = sim or project.build_simulator()
    s = project.settings
    out = np.full((len(sim.transmitters), len(pts)), float("-inf"))
    model = project.empirical_model()
    tracer = sim.ray_tracer
    for t, tx in enumerate(sim.transmitters):
        if model is None:
            n_candidates = 2 + len(tracer.reflectors) + len(tracer.edges) + len(tracer._active_higher_order(tx))
            chunk = min(_MAX_CHUNK_POINTS, max(1, _MAX_RAY_POINTS // n_candidates))
            for i0 in range(0, len(pts), chunk):
                rays = tracer.trace_points(tx, pts[i0:i0 + chunk], compact=True)
                total = rays.total_power_linear(coherent=True) if s.coherent else rays.power_linear.sum(axis=0)
                with np.errstate(divide="ignore"):
                    out[t, i0:i0 + chunk] = 10.0 * np.log10(total)
        else:
            ground = np.hypot(pts[:, 0] - tx.x, pts[:, 1] - tx.y)
            distance = ground if getattr(model, "uses_ground_distance", False) \
                else np.sqrt(ground ** 2 + (pts[:, 2] - tx.z) ** 2)
            pl = np.empty(len(pts))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # One call per receiver height (the empirical models take hr_m as a scalar)
                for z in np.unique(pts[:, 2]):
                    sel = pts[:, 2] == z
                    kwargs = dict(frequency_hz=tx.frequency_hz, ht_m=tx.z, hr_m=max(float(z), 1e-3),
                                  tx_gain_dbi=tx.antenna_gain_dbi)
                    if getattr(model, "supports_arrays", False):
                        pl[sel] = np.asarray(model.path_loss(distance_m=distance[sel], **kwargs), dtype=float)
                    else:
                        pl[sel] = [model.path_loss(distance_m=float(d), **kwargs) for d in distance[sel]]
            if not tx.is_isotropic:
                phi = np.degrees(np.arctan2(pts[:, 1] - tx.y, pts[:, 0] - tx.x))
                theta = np.degrees(np.arctan2(pts[:, 2] - tx.z, np.maximum(1e-6, ground)))
                pl = pl - tx.pattern_gain_db(phi, theta)
            out[t] = tx.tx_power_dbm - pl
        if s.min_distance_m > 0:
            near = np.linalg.norm(pts - np.array([tx.x, tx.y, tx.z]), axis=1) < s.min_distance_m
            out[t, near] = np.nan
    return out


# --------------------------------------------------------------------------- comparison

@dataclass
class PointComparison:
    index: int
    x: float
    y: float
    z: float
    measured_dbm: float
    predicted_dbm: float          # nan when not available
    error_db: float               # predicted - measured (nan when not available)
    tx_id: Optional[int]          # transmitter compared (the best server if the measurement names none)
    status: str                   # "ok", "sin señal", "muy cerca", "transmisor ausente"
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in self.__dict__.items()}


def _stats(errors: np.ndarray, measured: np.ndarray) -> Dict[str, Any]:
    n = int(len(errors))
    if n == 0:
        return {"n": 0, "bias_db": None, "std_db": None, "rmse_db": None, "mae_db": None, "max_abs_db": None,
                "r": None}
    predicted = measured + errors
    r = None
    if n >= 3 and np.std(predicted) > 0 and np.std(measured) > 0:
        r = float(np.corrcoef(predicted, measured)[0, 1])
    return {"n": n, "bias_db": float(np.mean(errors)),
            "std_db": float(np.std(errors, ddof=1)) if n > 1 else 0.0,
            "rmse_db": float(np.sqrt(np.mean(errors ** 2))), "mae_db": float(np.mean(np.abs(errors))),
            "max_abs_db": float(np.max(np.abs(errors))), "r": r}


@dataclass
class Comparison:
    """Prediction vs measurements: one ``PointComparison`` per measurement and the statistics of the usable ones."""
    points: List[PointComparison]
    stats: Dict[str, Any]
    counts: Dict[str, int]

    def errors(self) -> np.ndarray:
        return np.array([p.error_db for p in self.points if p.status == "ok"])

    def to_dict(self) -> Dict[str, Any]:
        return {"points": [p.to_dict() for p in self.points], "stats": self.stats, "counts": self.counts}


def _attribute(pred: np.ndarray, tx_index: np.ndarray, offsets: Optional[np.ndarray] = None):
    """
    Predicted power of each point (dBm) and the row of ``pred`` used: the named
    transmitter (``tx_index`` >= 0) or the strongest one (-1).  ``offsets`` (dB per
    transmitter) are added first.  -2 in ``tx_index`` = transmitter not enabled.
    """
    p = pred if offsets is None else pred + offsets[:, None]
    n = p.shape[1]
    rows = np.where(tx_index >= 0, tx_index, 0)
    if p.shape[0]:
        with np.errstate(invalid="ignore"):
            best = np.argmax(np.where(np.isnan(p), -np.inf, p), axis=0)
        rows = np.where(tx_index == -1, best, rows)
        value = p[rows, np.arange(n)]
    else:
        value = np.full(n, np.nan)
    value = np.where(tx_index == -2, np.nan, value)
    return value, rows


def _comparison(project, measurements, pred, tx_ids, offsets=None) -> Comparison:
    tx_index = _tx_index(measurements, tx_ids)
    value, rows = _attribute(pred, tx_index, offsets)
    pts = _points(project, measurements)
    out = []
    for i, m in enumerate(measurements):
        if tx_index[i] == -2:
            status, v = "transmisor ausente", float("nan")
        elif math.isnan(value[i]):
            status, v = "muy cerca", float("nan")
        elif math.isinf(value[i]):
            status, v = "sin señal", float("nan")
        else:
            status, v = "ok", float(value[i])
        tx = tx_ids[rows[i]] if status in ("ok", "sin señal", "muy cerca") and len(tx_ids) else m.tx_id
        out.append(PointComparison(index=i, x=float(pts[i, 0]), y=float(pts[i, 1]), z=float(pts[i, 2]),
                                   measured_dbm=m.rssi_dbm, predicted_dbm=v, error_db=v - m.rssi_dbm,
                                   tx_id=None if tx is None else int(tx), status=status, label=m.label))
    ok = np.array([p.status == "ok" for p in out], dtype=bool)
    errors = np.array([p.error_db for p in out])[ok] if out else np.zeros(0)
    measured = np.array([p.measured_dbm for p in out])[ok] if out else np.zeros(0)
    counts = {s: sum(1 for p in out if p.status == s) for s in ("ok", "sin señal", "muy cerca", "transmisor ausente")}
    return Comparison(points=out, stats=_stats(errors, measured), counts=counts)


def _tx_index(measurements, tx_ids) -> np.ndarray:
    """Row of each measurement in the prediction: >= 0 its transmitter, -1 best server, -2 not enabled."""
    pos = {tx: i for i, tx in enumerate(tx_ids)}
    return np.array([-1 if m.tx_id is None else pos.get(m.tx_id, -2) for m in measurements], dtype=int)


def _measurements(project, measurements) -> List[Measurement]:
    ms = list(project.measurements if measurements is None else measurements)
    if not ms:
        raise MeasurementError("no hay mediciones: importa un CSV o agrega puntos medidos")
    return [copy.copy(m).validate(f"medición {i + 1}") for i, m in enumerate(ms)]


def compare(project, measurements: Optional[Sequence[Measurement]] = None, sim=None) -> Comparison:
    """
    Compare the prediction of ``project`` with ``measurements`` (default: the ones
    of the project).  The error of each point is predicted - measured, in dB: a
    positive bias means that the model is optimistic.  Points where no ray arrives,
    closer to an antenna than the minimum distance, or whose transmitter is not
    enabled are reported but left out of the statistics.
    """
    ms = _measurements(project, measurements)
    sim = sim or project.build_simulator()
    pred = predict_points(project, _points(project, ms), sim)
    return _comparison(project, ms, pred, [tx.tx_id for tx in sim.transmitters])


# --------------------------------------------------------------------------- calibration

@dataclass
class CalibrationParameter:
    kind: str                     # "material", "tx_power" or "exponent"
    key: str                      # material id, tx_id as text ("*" = every transmitter) or "exponent"
    name: str                     # for the user, in Spanish
    before: float
    after: float = float("nan")
    uncertainty: Optional[float] = None    # 1-sigma from the curvature at the solution
    points: int = 0               # measured points that depend on it
    unit: str = "dB"

    def to_dict(self) -> Dict[str, Any]:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in self.__dict__.items()}


@dataclass
class CalibrationResult:
    parameters: List[CalibrationParameter]
    before: Comparison
    after: Comparison
    project: Any                  # calibrated copy of the project
    iterations: int
    evaluations: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"parameters": [p.to_dict() for p in self.parameters], "before": self.before.to_dict(),
                "after": self.after.to_dict(), "iterations": self.iterations, "evaluations": self.evaluations,
                "notes": list(self.notes), "project": self.project.to_dict()}


def _set_material_loss(material, frequency_hz: float, value: float):
    """Penetration loss of ``material`` at ``frequency_hz`` set to ``value`` dB (a linear law keeps its slope)."""
    if material.penetration == "fixed":
        material.loss_db = float(value)
    elif material.penetration == "frequency":
        material.loss_a_db = float(value) - float(material.loss_b_db_per_ghz) * frequency_hz / 1e9


def _used_materials(project) -> List[str]:
    used = {b.material for b in project.buildings} | {w.material for w in project.walls}
    return [m.id for m in project.materials if m.id in used]


def calibrate(project, measurements: Optional[Sequence[Measurement]] = None, materials="auto",
              tx_power: str = "auto", exponent: bool = True, max_iterations: int = 25,
              progress=None) -> CalibrationResult:
    """
    Fit the scenario to the measurements (least squares on the error in dB,
    Levenberg-Marquardt with bounds).  The project itself is not changed: the result
    carries a calibrated copy (``CalibrationResult.project``).

    materials: "auto" (default) every material with a finite penetration loss used by
        some building or wall (ray tracing only); a list of material ids; or None.
        The parameter is the loss per crossing at the reference frequency of the
        project; a material with a linear law keeps its slope.  Materials that no
        measured ray crosses are left as they are.
    tx_power: "per_tx" (one offset per transmitter with measurements), "common" (one
        offset for every enabled transmitter: e.g. the gain of the measuring device),
        "none", or "auto" (default): "per_tx" if every measurement names its
        transmitter, "common" otherwise.  The offset is added to ``tx_power_dbm``.
    exponent: in log-distance mode, also fit the path-loss exponent.
    progress: optional callable(evaluations) called after each evaluation of the model.

    A parameter that fewer than 3 measured points depend on (a material that almost
    no measured ray crosses, a transmitter with almost no points) is left as it is:
    those points cannot tell its value apart from the measurement error.
    """
    ms = _measurements(project, measurements)
    base = copy.deepcopy(project)
    base.validate()
    frequency = base.reference_frequency_hz()
    mode = base.settings.mode
    pts = _points(base, ms)
    enabled = [t.tx_id for t in base.enabled_transmitters()]
    tx_index = _tx_index(ms, enabled)

    # --- parameters
    params: List[CalibrationParameter] = []
    if materials is not None and mode == "raytracing":
        material_map = base.material_map()
        ids = _used_materials(base) if materials == "auto" else list(materials)
        for mid in ids:
            if mid not in material_map:
                raise MeasurementError(f"calibración: no existe el material {mid!r}")
            loss = material_map[mid].penetration_loss_db(frequency)
            if loss is None or math.isinf(loss):
                if materials != "auto":
                    raise MeasurementError(f"calibración: el material {mid!r} no tiene una pérdida finita "
                                           "que ajustar (no tapa o es opaco)")
                continue
            params.append(CalibrationParameter("material", mid, material_map[mid].name or mid, loss))
    if exponent and mode == "logdistance":
        params.append(CalibrationParameter("exponent", "exponent", "Exponente de pérdida",
                                           float(base.settings.log_distance_exponent), unit=""))
    if tx_power == "auto":
        tx_power = "per_tx" if all(m.tx_id is not None for m in ms) else "common"
    if tx_power not in ("per_tx", "common", "none"):
        raise MeasurementError(f"calibración: tx_power debe ser per_tx, common, none o auto, no {tx_power!r}")
    names = {t.tx_id: (t.name or f"Tx {t.tx_id}") for t in base.transmitters}
    if tx_power == "per_tx":
        for tx in enabled:
            if any(m.tx_id == tx for m in ms):
                params.append(CalibrationParameter("tx_power", str(tx), f"Potencia de {names[tx]}", 0.0))
    elif tx_power == "common" and enabled:
        params.append(CalibrationParameter("tx_power", "*", "Potencia de todos los transmisores", 0.0))
    if not params:
        raise MeasurementError("calibración: no hay nada que ajustar (sin materiales con pérdida finita "
                               "en ray tracing ni potencias)")

    nonlinear = [p for p in params if p.kind in ("material", "exponent")]
    offsets_of = [p for p in params if p.kind == "tx_power"]

    def build(theta_nl: np.ndarray):
        candidate = copy.deepcopy(base)
        mm = candidate.material_map()
        for p, value in zip(nonlinear, theta_nl):
            if p.kind == "material":
                _set_material_loss(mm[p.key], frequency, value)
            else:
                candidate.settings.log_distance_exponent = float(value)
        return candidate

    evaluations = 0
    cache: Dict[Tuple[float, ...], np.ndarray] = {}

    def predict(theta_nl: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        key = tuple(np.round(theta_nl, 9))
        if key not in cache:
            candidate = build(theta_nl)
            cache[key] = predict_points(candidate, pts)
            evaluations += 1
            if progress is not None:
                progress(evaluations)
        return cache[key]

    def offsets_vector(theta_off: np.ndarray) -> np.ndarray:
        o = np.zeros(len(enabled))
        for p, value in zip(offsets_of, theta_off):
            if p.key == "*":
                o += value
            else:
                o[enabled.index(int(p.key))] += value
        return o

    # Points usable at the start (a signal, not too close, transmitter enabled)
    pred0 = predict(np.array([p.before for p in nonlinear]))
    value0, rows0 = _attribute(pred0, tx_index)
    usable = np.isfinite(value0)
    measured = np.array([m.rssi_dbm for m in ms])[usable]
    n_use = int(usable.sum())
    if n_use < MIN_POINTS_PER_PARAMETER * len(params) + 1:
        raise MeasurementError(f"calibración: hay {n_use} puntos utilizables para {len(params)} parámetro(s); "
                               f"hacen falta al menos {MIN_POINTS_PER_PARAMETER * len(params) + 1}. "
                               "Mide más puntos o ajusta menos parámetros")

    theta0 = np.array([p.before for p in params], dtype=float)
    lo = np.array([{"material": MATERIAL_LOSS_BOUNDS_DB, "tx_power": TX_OFFSET_BOUNDS_DB,
                    "exponent": EXPONENT_BOUNDS}[p.kind][0] for p in params])
    hi = np.array([{"material": MATERIAL_LOSS_BOUNDS_DB, "tx_power": TX_OFFSET_BOUNDS_DB,
                    "exponent": EXPONENT_BOUNDS}[p.kind][1] for p in params])
    n_nl = len(nonlinear)
    steps = np.array([0.5 if p.kind == "material" else 0.05 for p in nonlinear])

    def residuals(theta: np.ndarray):
        pred = predict(theta[:n_nl])
        value, rows = _attribute(pred, tx_index, offsets_vector(theta[n_nl:]))
        r = value[usable] - measured
        # A point that lost its signal on the way (should not happen with finite losses) weighs a lot but finitely
        r = np.where(np.isfinite(r), r, 100.0)
        return r, rows[usable]

    def cost(theta):
        r, _ = residuals(theta)
        return float(r @ r)

    def jacobian(theta, r, rows):
        J = np.zeros((len(r), len(params)))
        for j in range(n_nl):
            step = steps[j] if theta[j] + steps[j] <= hi[j] else -steps[j]
            t = theta.copy()
            t[j] += step
            rj, _ = residuals(t)
            J[:, j] = (rj - r) / step
        for k, p in enumerate(offsets_of):
            col = n_nl + k
            if p.key == "*":
                J[:, col] = 1.0
            else:
                J[:, col] = (rows == enabled.index(int(p.key))).astype(float)
        return J

    theta = theta0.copy()
    # Parameters that too few points depend on stay fixed
    r, rows = residuals(theta)
    J = jacobian(theta, r, rows)
    depends = np.count_nonzero(np.abs(J) > 1e-3, axis=0)
    active = depends >= MIN_DEPENDENT_POINTS
    if not active.any():
        raise MeasurementError("calibración: ningún parámetro depende de al menos "
                               f"{MIN_DEPENDENT_POINTS} puntos medidos")
    current = cost(theta)
    lam = 1e-3
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if iterations > 1:
            r, rows = residuals(theta)
            J = jacobian(theta, r, rows)
        Ja = J[:, active]
        A = Ja.T @ Ja
        g = Ja.T @ r
        improved = False
        while lam < 1e8:
            try:
                step = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-9)), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            delta = np.zeros(len(params))
            delta[active] = step
            trial = np.clip(theta + delta, lo, hi)
            trial_cost = cost(trial)
            if trial_cost < current - 1e-12:
                step_size = float(np.max(np.abs(trial - theta)))
                theta, current = trial, trial_cost
                lam = max(lam / 3.0, 1e-7)
                improved = True
                break
            lam *= 4.0
        if not improved or step_size < 0.01:
            break

    # --- result: 1-sigma uncertainties from the curvature (Gauss-Newton) and the residual scatter
    r, rows = residuals(theta)
    J = jacobian(theta, r, rows)
    Ja = J[:, active]
    dof = max(1, len(r) - int(active.sum()))
    sigma2 = float(r @ r) / dof
    unc = np.full(len(params), np.nan)
    try:
        unc[active] = np.sqrt(np.maximum(np.diag(np.linalg.inv(Ja.T @ Ja)) * sigma2, 0.0))
    except np.linalg.LinAlgError:
        pass
    notes = []
    for j, p in enumerate(params):
        p.after = float(theta[j])
        p.points = int(depends[j])
        p.uncertainty = float(unc[j]) if math.isfinite(unc[j]) else None
        if not active[j]:
            p.after = p.before
            p.uncertainty = None
            notes.append(f"{p.name}: ningún punto medido depende de él; se deja igual" if p.points == 0 else
                         f"{p.name}: solo {p.points} punto(s) medido(s) dependen de él "
                         f"(hacen falta {MIN_DEPENDENT_POINTS}); se deja igual")
        elif theta[j] <= lo[j] + 1e-6 or theta[j] >= hi[j] - 1e-6:
            notes.append(f"{p.name}: quedó en el límite del rango permitido ({p.after:.3g}); "
                         "revisa las mediciones o el modelo")

    calibrated = build(np.array([p.after for p in nonlinear]))
    for p in offsets_of:
        for tx in calibrated.transmitters:
            if tx.enabled and (p.key == "*" or tx.tx_id == int(p.key)):
                tx.tx_power_dbm = float(tx.tx_power_dbm + p.after)
    calibrated.measurements = list(ms)
    calibrated.validate()
    before = _comparison(base, ms, pred0, enabled)
    after = _comparison(calibrated, ms, predict_points(calibrated, pts), enabled)
    evaluations += 1
    return CalibrationResult(parameters=params, before=before, after=after, project=calibrated,
                             iterations=iterations, evaluations=evaluations, notes=notes)
