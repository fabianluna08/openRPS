"""
OpenRPS - Coverage Map & Grid Simulation Engine
Simulates multi-transmitter radio propagation over 2D geographic/indoor grids.
"""

import csv
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .models import PropagationModel
from .multipath import ChannelImpulseResponse, circular_angular_spread, rms_delay_spread
from .raytracer import RayTracer, Transmitter, PlanarReflector

# Upper bound of receiver points traced per numpy batch (keeps memory bounded on big grids)
_MAX_CHUNK_POINTS = 200_000


def _fmt_coord(value: float) -> str:
    """Coordinate with at least one decimal and no precision loss on fine (indoor) grids."""
    text = f"{value:.4f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _finite_stat(func, data: np.ndarray) -> Optional[float]:
    """func over the finite values of data, or None if there are none (JSON null)."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    return float(func(values)) if values.size else None


@dataclass
class SimulationResult:
    """
    Stores the output grids and metrics of an OpenRPS simulation.

    The first axis of ``rx_power_dbm`` / ``path_loss_db`` follows the order in which
    the transmitters were added (``tx_ids`` gives the tx_id of each position); it is
    NOT indexed by tx_id.  ``path_loss_db`` is the link loss P_tx - P_rx, i.e. it
    includes the transmitter antenna gain.
    """
    x_coords: np.ndarray
    y_coords: np.ndarray
    rx_power_dbm: np.ndarray       # Shape: [noOfTx, Ny, Nx]
    path_loss_db: np.ndarray       # Shape: [noOfTx, Ny, Nx]
    best_server_id: np.ndarray     # Shape: [Ny, Nx] (tx_id with maximum received power, -1 = none)
    max_rx_power_dbm: np.ndarray   # Shape: [Ny, Nx]
    rms_delay_spread_ns: np.ndarray# Shape: [Ny, Nx] (best server; NaN when not available)
    angular_spread_bs_phi: np.ndarray # Shape: [Ny, Nx] (best server; NaN when not available)
    tx_ids: Optional[List[int]] = None              # tx_id of each position of the Tx axis
    best_server_index: Optional[np.ndarray] = None  # Shape: [Ny, Nx] position along the Tx axis (-1 = none)
    tx_positions: Optional[List[Tuple[float, float, float]]] = None  # (x, y, z) per Tx

    def best_server_indices(self) -> np.ndarray:
        """
        [Ny, Nx] position of the best server along the Tx axis of ``rx_power_dbm`` /
        ``path_loss_db`` (-1 where there is no server).
        """
        if self.best_server_index is not None:
            return np.asarray(self.best_server_index)
        ids = np.asarray(self.best_server_id)
        if self.tx_ids is None:
            # Result built by hand without metadata: assume tx_id == position
            return ids
        index = np.full(ids.shape, -1, dtype=int)
        for position, tx_id in enumerate(self.tx_ids):
            index[ids == tx_id] = position
        return index

    def best_server_path_loss_db(self) -> np.ndarray:
        """[Ny, Nx] path loss towards the best server (NaN where there is none)."""
        index = self.best_server_indices()
        num_tx = self.path_loss_db.shape[0]
        ok = (index >= 0) & (index < num_tx)
        if num_tx == 0:
            return np.full(index.shape, np.nan)
        picked = np.take_along_axis(self.path_loss_db, np.clip(index, 0, num_tx - 1)[None, ...], axis=0)[0]
        return np.where(ok, picked, np.nan)

    def export_csv(self, filename: str):
        """Export simulation results as a flattened CSV table (X, Y, BestTx, RxPower_dBm, PathLoss_dB, DelaySpread_ns)."""
        ny, nx = self.best_server_id.shape
        best_pl = self.best_server_path_loss_db()
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_m", "y_m", "best_tx_id", "rx_power_dbm", "path_loss_db", "rms_delay_spread_ns", "angular_spread_phi_deg"])
            for j in range(ny):
                y = _fmt_coord(float(self.y_coords[j]))
                for i in range(nx):
                    x = _fmt_coord(float(self.x_coords[i]))
                    best_tx = int(self.best_server_id[j, i])
                    p_rx = float(self.max_rx_power_dbm[j, i])
                    pl = float(best_pl[j, i])
                    ds = float(self.rms_delay_spread_ns[j, i])
                    asp = float(self.angular_spread_bs_phi[j, i])
                    writer.writerow([x, y, best_tx, f"{p_rx:.2f}", f"{pl:.2f}", f"{ds:.2f}", f"{asp:.2f}"])

    def summary(self) -> dict:
        """Grid parameters and statistical summary (non-finite pixels are ignored; None = no data)."""
        return {
            "grid": {
                "x_min": float(self.x_coords[0]),
                "x_max": float(self.x_coords[-1]),
                "y_min": float(self.y_coords[0]),
                "y_max": float(self.y_coords[-1]),
                "nx": len(self.x_coords),
                "ny": len(self.y_coords)
            },
            "tx_ids": None if self.tx_ids is None else [int(t) for t in self.tx_ids],
            "metrics": {
                "max_rx_power_dbm": _finite_stat(np.max, self.max_rx_power_dbm),
                "min_rx_power_dbm": _finite_stat(np.min, self.max_rx_power_dbm),
                "mean_rx_power_dbm": _finite_stat(np.mean, self.max_rx_power_dbm),
                "mean_delay_spread_ns": _finite_stat(np.mean, self.rms_delay_spread_ns),
                "max_delay_spread_ns": _finite_stat(np.max, self.rms_delay_spread_ns),
                "mean_angular_spread_phi_deg": _finite_stat(np.mean, self.angular_spread_bs_phi)
            }
        }

    def export_json(self, filename: str):
        """Exports grid parameters and statistical summary to (strictly valid) JSON."""
        with open(filename, "w") as f:
            json.dump(self.summary(), f, indent=2, allow_nan=False)
            f.write("\n")


def _make_axis(v_range: Tuple[float, float], resolution_m: float, name: str) -> np.ndarray:
    v_min, v_max = float(v_range[0]), float(v_range[1])
    if not (np.isfinite(v_min) and np.isfinite(v_max)) or v_max < v_min:
        raise ValueError(f"{name} must be (min, max) with finite min <= max, got {v_range!r}")
    return np.arange(v_min, v_max + 1e-5, resolution_m)


def _make_grid(x_range, y_range, resolution_m: float) -> Tuple[np.ndarray, np.ndarray]:
    if not (np.isfinite(resolution_m) and resolution_m > 0.0):
        raise ValueError(f"resolution_m must be > 0, got {resolution_m!r}")
    return _make_axis(x_range, resolution_m, "x_range"), _make_axis(y_range, resolution_m, "y_range")


def _select_best_server(rx_power_dbm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Best server per pixel from rx_power_dbm [noOfTx, Ny, Nx].
    Returns (index [Ny, Nx] along the Tx axis, -1 = none; max power [Ny, Nx], -inf = none).
    Ties go to the transmitter added first; NaN powers never win.
    """
    num_tx, ny, nx = rx_power_dbm.shape
    if num_tx == 0:
        return np.full((ny, nx), -1, dtype=int), np.full((ny, nx), -np.inf)
    power = np.where(np.isnan(rx_power_dbm), -np.inf, rx_power_dbm)
    index = np.argmax(power, axis=0)
    max_power = np.take_along_axis(power, index[None, ...], axis=0)[0]
    index = np.where(max_power > -np.inf, index, -1)
    return index, max_power


class PropagationSimulator:
    """
    Main OpenRPS simulation orchestrator.
    Supports both Ray-Tracing multipath simulation and empirical models.
    """
    def __init__(self):
        self.transmitters: List[Transmitter] = []
        self.ray_tracer = RayTracer()

    def add_transmitter(self, tx: Transmitter):
        if tx.tx_id < 0:
            raise ValueError("tx_id must be >= 0 (-1 is reserved for 'no server')")
        if any(t.tx_id == tx.tx_id for t in self.transmitters):
            raise ValueError(f"duplicate tx_id {tx.tx_id}: every transmitter needs a unique tx_id")
        self.transmitters.append(tx)

    def add_reflector(self, wall: PlanarReflector):
        self.ray_tracer.add_reflector(wall)

    def _build_result(self, x_coords, y_coords, rx_power_dbm, path_loss_db,
                      best_index, max_rx_power_dbm, delay_spread, angular_spread) -> SimulationResult:
        tx_ids = [tx.tx_id for tx in self.transmitters]
        id_lookup = np.array(tx_ids + [-1], dtype=int)   # position -1 -> id -1
        return SimulationResult(
            x_coords=x_coords,
            y_coords=y_coords,
            rx_power_dbm=rx_power_dbm,
            path_loss_db=path_loss_db,
            best_server_id=id_lookup[best_index],
            max_rx_power_dbm=max_rx_power_dbm,
            rms_delay_spread_ns=delay_spread,
            angular_spread_bs_phi=angular_spread,
            tx_ids=tx_ids,
            best_server_index=best_index,
            tx_positions=[(tx.x, tx.y, tx.z) for tx in self.transmitters],
        )

    def run_raytracing_simulation(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution_m: float = 10.0,
        rx_height_m: float = 1.5,
        vectorized: bool = True
    ) -> SimulationResult:
        """
        Executes a 2D grid ray-tracing multipath simulation across all transmitters.

        vectorized: True (default) traces the whole grid with numpy
            (``RayTracer.trace_points``).  False runs the reference implementation,
            one ``RayTracer.trace_paths`` call per pixel and transmitter: much slower,
            same results; use it if you subclass RayTracer and override trace_paths.
        """
        x_coords, y_coords = _make_grid(x_range, y_range, resolution_m)
        if vectorized:
            return self._raytracing_vectorized(x_coords, y_coords, rx_height_m)
        return self._raytracing_pointwise(x_coords, y_coords, rx_height_m)

    def _raytracing_vectorized(self, x_coords, y_coords, rx_height_m: float) -> SimulationResult:
        nx, ny, num_tx = len(x_coords), len(y_coords), len(self.transmitters)
        rx_power_dbm = np.full((num_tx, ny, nx), float("-inf"))
        path_loss_db = np.full((num_tx, ny, nx), float("nan"))
        delay_spread_tx = np.full((num_tx, ny, nx), float("nan"))
        angular_spread_tx = np.full((num_tx, ny, nx), float("nan"))

        rows_per_chunk = max(1, _MAX_CHUNK_POINTS // max(1, nx))
        for j0 in range(0, ny, rows_per_chunk):
            rows = slice(j0, min(ny, j0 + rows_per_chunk))
            points = np.empty((len(y_coords[rows]), nx, 3))
            points[..., 0] = x_coords[None, :]
            points[..., 1] = y_coords[rows, None]
            points[..., 2] = rx_height_m

            for tx_idx, tx in enumerate(self.transmitters):
                rays = self.ray_tracer.trace_points(tx, points)
                power = rays.power_linear                    # [K, rows, nx] mW
                p_total = power.sum(axis=0)
                with np.errstate(divide="ignore"):
                    p_dbm = 10.0 * np.log10(p_total)         # -inf where no ray arrives
                rx_power_dbm[tx_idx, rows] = p_dbm
                path_loss_db[tx_idx, rows] = np.where(p_total > 0.0, tx.tx_power_dbm - p_dbm, np.nan)
                delay_spread_tx[tx_idx, rows] = rms_delay_spread(power, rays.tau_ns)
                angular_spread_tx[tx_idx, rows] = circular_angular_spread(power, rays.tx_phi_deg)[1]

        best_index, max_rx_power_dbm = _select_best_server(rx_power_dbm)
        if num_tx:
            pick = np.clip(best_index, 0, num_tx - 1)[None, ...]
            has_server = best_index >= 0
            delay_spread = np.where(has_server, np.take_along_axis(delay_spread_tx, pick, axis=0)[0], np.nan)
            angular_spread = np.where(has_server, np.take_along_axis(angular_spread_tx, pick, axis=0)[0], np.nan)
        else:
            delay_spread = np.full((ny, nx), float("nan"))
            angular_spread = np.full((ny, nx), float("nan"))

        return self._build_result(x_coords, y_coords, rx_power_dbm, path_loss_db,
                                  best_index, max_rx_power_dbm, delay_spread, angular_spread)

    def _raytracing_pointwise(self, x_coords, y_coords, rx_height_m: float) -> SimulationResult:
        """Reference implementation: one trace_paths() call per pixel and transmitter."""
        nx, ny, num_tx = len(x_coords), len(y_coords), len(self.transmitters)
        rx_power_dbm = np.full((num_tx, ny, nx), float("-inf"))
        path_loss_db = np.full((num_tx, ny, nx), float("nan"))
        rms_delay_spread_ns = np.full((ny, nx), float("nan"))
        angular_spread_phi = np.full((ny, nx), float("nan"))
        best_index = np.full((ny, nx), -1, dtype=int)
        max_rx_power_dbm = np.full((ny, nx), float("-inf"))

        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                rx_pos = (x, y, rx_height_m)
                pixel_best_idx = -1
                pixel_max_p_dbm = float("-inf")
                best_cir: Optional[ChannelImpulseResponse] = None

                for tx_idx, tx in enumerate(self.transmitters):
                    cir = self.ray_tracer.trace_paths(tx, rx_pos)
                    p_rx_dbm = cir.total_received_power_dbm()
                    rx_power_dbm[tx_idx, j, i] = p_rx_dbm

                    if p_rx_dbm > float("-inf"):
                        path_loss_db[tx_idx, j, i] = tx.tx_power_dbm - p_rx_dbm

                    if p_rx_dbm > pixel_max_p_dbm:
                        pixel_max_p_dbm = p_rx_dbm
                        pixel_best_idx = tx_idx
                        best_cir = cir

                max_rx_power_dbm[j, i] = pixel_max_p_dbm
                best_index[j, i] = pixel_best_idx

                if best_cir is not None:
                    rms_delay_spread_ns[j, i] = best_cir.rms_delay_spread()
                    _, phi_spread = best_cir.angular_spread(plane="phi", station="bs")
                    angular_spread_phi[j, i] = phi_spread

        return self._build_result(x_coords, y_coords, rx_power_dbm, path_loss_db,
                                  best_index, max_rx_power_dbm, rms_delay_spread_ns, angular_spread_phi)

    def run_empirical_simulation(
        self,
        model: PropagationModel,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution_m: float = 10.0,
        rx_height_m: float = 1.5
    ) -> SimulationResult:
        """
        Executes a 2D grid simulation using an empirical model (e.g. Okumura-Hata, TwoRay, FSPL).

        For every transmitter the model is called as
            model.path_loss(distance_m, frequency_hz, ht_m=tx.z, hr_m=rx_height_m,
                            tx_gain_dbi=tx.antenna_gain_dbi)
        and P_rx = P_tx - path_loss.  ``distance_m`` is the horizontal distance if
        ``model.uses_ground_distance`` is True, the 3D distance otherwise.  Models
        with ``supports_arrays = True`` receive the whole grid at once; any other
        model (e.g. a user subclass written for scalars) is evaluated pixel by pixel.

        Empirical models carry no multipath information: the delay-spread and
        angular-spread grids of the result are NaN.
        """
        x_coords, y_coords = _make_grid(x_range, y_range, resolution_m)
        nx, ny, num_tx = len(x_coords), len(y_coords), len(self.transmitters)

        rx_power_dbm = np.full((num_tx, ny, nx), float("-inf"))
        path_loss_db = np.full((num_tx, ny, nx), float("nan"))
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        use_ground = bool(getattr(model, "uses_ground_distance", False))
        use_arrays = bool(getattr(model, "supports_arrays", False))

        for tx_idx, tx in enumerate(self.transmitters):
            distance = np.hypot(grid_x - tx.x, grid_y - tx.y)
            if not use_ground:
                distance = np.sqrt(distance ** 2 + (rx_height_m - tx.z) ** 2)
            kwargs = dict(frequency_hz=tx.frequency_hz, ht_m=tx.z, hr_m=rx_height_m,
                          tx_gain_dbi=tx.antenna_gain_dbi)
            if use_arrays:
                pl = np.asarray(model.path_loss(distance_m=distance, **kwargs), dtype=float)
            else:
                pl = np.array([model.path_loss(distance_m=float(d), **kwargs) for d in distance.ravel()],
                              dtype=float).reshape(distance.shape)
            path_loss_db[tx_idx] = pl
            rx_power_dbm[tx_idx] = tx.tx_power_dbm - pl

        best_index, max_rx_power_dbm = _select_best_server(rx_power_dbm)
        not_available = np.full((ny, nx), float("nan"))
        return self._build_result(x_coords, y_coords, rx_power_dbm, path_loss_db,
                                  best_index, max_rx_power_dbm, not_available, not_available.copy())
