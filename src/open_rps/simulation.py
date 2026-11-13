"""
OpenRPS - Coverage Map & Grid Simulation Engine
Simulates multi-transmitter radio propagation over 2D geographic/indoor grids.
"""

import csv
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .models import PropagationModel
from .multipath import ChannelImpulseResponse, circular_angular_spread, rms_delay_spread
from .raytracer import RayTracer, Transmitter, PlanarReflector

# Upper bound of receiver points traced per numpy batch (keeps memory bounded on big grids)
_MAX_CHUNK_POINTS = 200_000

BOLTZMANN_J_PER_K = 1.380649e-23
# Pixels closer than this (3D) to a transmitter antenna are not evaluated by default.
DEFAULT_MIN_DISTANCE_M = 1.0


def thermal_noise_dbm(bandwidth_hz: float, noise_figure_db: float = 0.0, temperature_k: float = 290.0) -> float:
    """
    Noise power of a receiver in dBm: N = k * T * B * F  (thermal noise kTB times the
    noise figure).  At 290 K this is -173.98 dBm/Hz + 10*log10(B) + NF.
    """
    if not (np.isfinite(bandwidth_hz) and bandwidth_hz > 0.0):
        raise ValueError(f"bandwidth_hz must be > 0, got {bandwidth_hz!r}")
    if not (np.isfinite(temperature_k) and temperature_k > 0.0):
        raise ValueError(f"temperature_k must be > 0, got {temperature_k!r}")
    if not np.isfinite(noise_figure_db):
        raise ValueError(f"noise_figure_db must be finite, got {noise_figure_db!r}")
    return 10.0 * math.log10(BOLTZMANN_J_PER_K * temperature_k * bandwidth_hz * 1e3) + float(noise_figure_db)


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
    includes the transmitter antenna gain (and pattern) and the shadow fading.

    Values without data: ``-inf`` dBm means that no power arrives (e.g. every ray of
    that transmitter is blocked by opaque walls, see ``PlanarReflector.transmission_loss_db``);
    ``NaN`` means that the value is not reliable or not available (pixel closer to
    the antenna than ``min_distance_m``, metric not provided by the model, no
    server).  A pixel that no transmitter reaches has no server: ``best_server_id``
    -1, ``max_rx_power_dbm`` -inf, and NaN path loss, spreads and SINR.

    ``sinr_db``, ``interference_plus_noise_dbm`` and ``sinr_settings`` stay ``None``
    until ``compute_sinr()`` is called.
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
    tx_frequencies_hz: Optional[List[float]] = None  # carrier frequency per Tx (used by the SINR)
    sinr_db: Optional[np.ndarray] = None                      # Shape: [Ny, Nx], filled by compute_sinr()
    interference_plus_noise_dbm: Optional[np.ndarray] = None  # Shape: [Ny, Nx], filled by compute_sinr()
    sinr_settings: Optional[dict] = None                      # noise parameters used by compute_sinr()

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

    def sinr_map(self, noise_figure_db: float = 7.0, bandwidth_hz: float = 20e6,
                 temperature_k: float = 290.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downlink SINR of the best server and interference-plus-noise power, per pixel:

            SINR = P_best / (I + N),   I = sum of the powers (in mW) of the other
            co-channel transmitters,   N = k * T * B * 10^(NF/10)

        Returns ``(sinr_db, interference_plus_noise_dbm)``, both [Ny, Nx], without
        storing anything (see ``compute_sinr``).

        * Only transmitters on the *same frequency* as the best server interfere
          (``tx_frequencies_hz``); the others are ignored completely (ideal filters,
          no adjacent-channel leakage).  A result built by hand without
          ``tx_frequencies_hz`` is treated as fully co-channel.
        * Every transmitter is assumed to radiate all the time (full load) and the
          powers are the per-transmitter grids ``rx_power_dbm`` (which already
          include antenna pattern, shadow fading and coherent sum if they were used).
        * Pixels without server are NaN.  So are pixels where the power of some
          co-channel interferer is unknown (NaN: closer to it than ``min_distance_m``).
        """
        noise_mw = 10.0 ** (thermal_noise_dbm(bandwidth_hz, noise_figure_db, temperature_k) / 10.0)
        index = self.best_server_indices()
        num_tx = self.rx_power_dbm.shape[0]
        served = (index >= 0) & (index < num_tx)
        interference_mw = np.full(index.shape, np.nan)
        if num_tx:
            power_mw = 10.0 ** (np.asarray(self.rx_power_dbm, dtype=float) / 10.0)   # -inf -> 0, NaN stays
            if self.tx_frequencies_hz is None:
                freqs = np.zeros(num_tx)
            else:
                freqs = np.asarray(self.tx_frequencies_hz, dtype=float)
            positions = np.arange(num_tx)[:, None, None]
            for freq in np.unique(freqs):
                group = np.isclose(freqs, freq, rtol=1e-9, atol=0.0)          # co-channel transmitters
                in_group = served & group[np.clip(index, 0, num_tx - 1)]
                # Sum of the group excluding the best server itself (no subtraction: no cancellation)
                others = np.where(positions != index[None, ...], power_mw, 0.0)[group].sum(axis=0)
                interference_mw = np.where(in_group, others, interference_mw)
        total_mw = interference_mw + noise_mw
        with np.errstate(invalid="ignore"):
            i_plus_n_dbm = 10.0 * np.log10(total_mw)
            sinr = np.where(served, np.asarray(self.max_rx_power_dbm, dtype=float), np.nan) - i_plus_n_dbm
        return sinr, i_plus_n_dbm

    def compute_sinr(self, noise_figure_db: float = 7.0, bandwidth_hz: float = 20e6,
                     temperature_k: float = 290.0) -> np.ndarray:
        """
        Compute the SINR map (see ``sinr_map``) and keep it in the result: fills
        ``sinr_db``, ``interference_plus_noise_dbm`` and ``sinr_settings``.  From then
        on ``export_csv`` / ``export_json`` / ``summary`` include the SINR.
        Returns ``sinr_db`` ([Ny, Nx], NaN where there is no server).
        """
        sinr, i_plus_n = self.sinr_map(noise_figure_db, bandwidth_hz, temperature_k)
        self.sinr_db = sinr
        self.interference_plus_noise_dbm = i_plus_n
        self.sinr_settings = {
            "noise_figure_db": float(noise_figure_db),
            "bandwidth_hz": float(bandwidth_hz),
            "temperature_k": float(temperature_k),
            "noise_power_dbm": thermal_noise_dbm(bandwidth_hz, noise_figure_db, temperature_k),
        }
        return sinr

    def export_csv(self, filename: str):
        """
        Export simulation results as a flattened CSV table with the columns
        x_m, y_m, best_tx_id, rx_power_dbm, path_loss_db, rms_delay_spread_ns, angular_spread_phi_deg.
        If ``compute_sinr()`` was called before, two more columns are appended at the
        end: sinr_db and interference_plus_noise_dbm.
        """
        ny, nx = self.best_server_id.shape
        best_pl = self.best_server_path_loss_db()
        header = ["x_m", "y_m", "best_tx_id", "rx_power_dbm", "path_loss_db", "rms_delay_spread_ns", "angular_spread_phi_deg"]
        with_sinr = self.sinr_db is not None and self.interference_plus_noise_dbm is not None
        if with_sinr:
            header += ["sinr_db", "interference_plus_noise_dbm"]
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for j in range(ny):
                y = _fmt_coord(float(self.y_coords[j]))
                for i in range(nx):
                    x = _fmt_coord(float(self.x_coords[i]))
                    best_tx = int(self.best_server_id[j, i])
                    p_rx = float(self.max_rx_power_dbm[j, i])
                    pl = float(best_pl[j, i])
                    ds = float(self.rms_delay_spread_ns[j, i])
                    asp = float(self.angular_spread_bs_phi[j, i])
                    row = [x, y, best_tx, f"{p_rx:.2f}", f"{pl:.2f}", f"{ds:.2f}", f"{asp:.2f}"]
                    if with_sinr:
                        row += [f"{float(self.sinr_db[j, i]):.2f}", f"{float(self.interference_plus_noise_dbm[j, i]):.2f}"]
                    writer.writerow(row)

    def summary(self) -> dict:
        """
        Grid parameters and statistical summary (non-finite pixels are ignored; None = no data).
        After ``compute_sinr()`` it also has a "sinr" block (noise settings and statistics).
        """
        summary = self._base_summary()
        if self.sinr_db is not None:
            summary["sinr"] = {
                **(self.sinr_settings or {}),
                "mean_sinr_db": _finite_stat(np.mean, self.sinr_db),
                "min_sinr_db": _finite_stat(np.min, self.sinr_db),
                "max_sinr_db": _finite_stat(np.max, self.sinr_db),
                "mean_interference_plus_noise_dbm": (
                    None if self.interference_plus_noise_dbm is None
                    else _finite_stat(np.mean, self.interference_plus_noise_dbm)),
            }
        return summary

    def _base_summary(self) -> dict:
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
    Returns (index [Ny, Nx] along the Tx axis, -1 = none; max power [Ny, Nx]).
    Ties go to the transmitter added first.  NaN powers (not reliable, e.g. pixel
    closer to that transmitter than ``min_distance_m``) never win: the best server
    is chosen among the remaining transmitters.  Where there is no server the max
    power is -inf if no power arrives from any transmitter, and NaN if the power
    of some transmitter is unknown (NaN).
    """
    num_tx, ny, nx = rx_power_dbm.shape
    if num_tx == 0:
        return np.full((ny, nx), -1, dtype=int), np.full((ny, nx), -np.inf)
    unknown = np.isnan(rx_power_dbm)
    power = np.where(unknown, -np.inf, rx_power_dbm)
    index = np.argmax(power, axis=0)
    max_power = np.take_along_axis(power, index[None, ...], axis=0)[0]
    no_server = ~(max_power > -np.inf)
    index = np.where(no_server, -1, index)
    max_power = np.where(no_server & unknown.any(axis=0), np.nan, max_power)
    return index, max_power


def _validate_min_distance(min_distance_m: Optional[float]) -> float:
    """None or 0 disable the minimum-distance mask; returns the threshold in meters (0 = disabled)."""
    if min_distance_m is None:
        return 0.0
    value = float(min_distance_m)
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(f"min_distance_m must be None or a finite distance >= 0, got {min_distance_m!r}")
    return value


def _shadow_fading_db(shape: Tuple[int, int, int], sigma_db: float, seed: Optional[int]) -> Optional[np.ndarray]:
    """
    Log-normal shadow fading: one independent N(0, sigma^2) dB sample per
    (transmitter, pixel), or None when sigma is 0.  The whole [noOfTx, Ny, Nx] array
    is drawn in a single call so that the result only depends on ``seed`` and on the
    grid, not on how the simulation is evaluated (vectorized, chunked or pointwise).
    """
    sigma = float(sigma_db)
    if not (np.isfinite(sigma) and sigma >= 0.0):
        raise ValueError(f"shadow_fading_sigma_db must be a finite value >= 0, got {sigma_db!r}")
    if sigma == 0.0:
        return None
    return np.random.default_rng(seed).normal(0.0, sigma, size=shape)


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
            tx_frequencies_hz=[float(tx.frequency_hz) for tx in self.transmitters],
        )

    def _too_close_mask(self, x_coords, y_coords, rx_height_m: float, min_distance: float) -> Optional[np.ndarray]:
        """
        [noOfTx, Ny, Nx] bool grid, True where the pixel is closer (3D) to that
        transmitter's antenna than ``min_distance``; None if no pixel is.
        Shared by every simulation mode so that they all mask the same pixels.
        """
        if not (min_distance > 0.0) or not self.transmitters:
            return None
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        mask = np.stack([np.sqrt((grid_x - tx.x) ** 2 + (grid_y - tx.y) ** 2 + (rx_height_m - tx.z) ** 2) < min_distance
                         for tx in self.transmitters])
        return mask if mask.any() else None

    def run_raytracing_simulation(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution_m: float = 10.0,
        rx_height_m: float = 1.5,
        vectorized: bool = True,
        coherent: bool = False,
        min_distance_m: Optional[float] = DEFAULT_MIN_DISTANCE_M,
        shadow_fading_sigma_db: float = 0.0,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Executes a 2D grid ray-tracing multipath simulation across all transmitters.

        Walls added with ``add_reflector`` reflect, and those that have a
        ``transmission_loss_db`` also attenuate (or block, if it is ``inf``) every ray
        that crosses them, in both implementations and with both kinds of sum.  A
        blocked ray has zero power: it does not count for the received power nor for
        the delay / angular spreads.  Where every ray of a transmitter is blocked its
        power is ``-inf`` dBm (path loss NaN) and it cannot be the best server.

        vectorized: True (default) traces the whole grid with numpy
            (``RayTracer.trace_points``).  False runs the reference implementation,
            one ``RayTracer.trace_paths`` call per pixel and transmitter: much slower,
            same results; use it if you subclass RayTracer and override trace_paths.
        coherent: False (default) adds the ray powers, P = sum |a_k|^2 (local mean
            power, no fast fading).  True adds the complex ray amplitudes of each
            transmitter, P = |sum a_k|^2: constructive/destructive interference
            (fast fading) appears, and with only the direct and ground rays the
            result equals ``TwoRayGroundReflection``.  The fading pattern has a
            spatial period of the order of the wavelength, so a grid coarser than
            that only *samples* it.  The delay and angular spreads are not affected:
            they always weight each ray by its own power |a_k|^2.
        min_distance_m: pixels whose 3D distance to a transmitter antenna is smaller
            than this are not reliable for that transmitter (the far-field Friis
            formula does not hold next to the antenna): its power and path loss are
            set to NaN there and it cannot be the best server of the pixel.
            Default 1 m; ``None`` or 0 disables the check.
        shadow_fading_sigma_db: standard deviation of the log-normal shadow fading.
            If > 0, an independent N(0, sigma^2) dB sample per (transmitter, pixel) is
            added to the path loss (no spatial correlation).  Default 0 = none.
        seed: seed of the ``numpy.random.Generator`` used for the shadow fading
            (same seed and grid -> same map).  ``None`` = different every run.
        """
        x_coords, y_coords = _make_grid(x_range, y_range, resolution_m)
        too_close = self._too_close_mask(x_coords, y_coords, rx_height_m, _validate_min_distance(min_distance_m))
        shadow = _shadow_fading_db((len(self.transmitters), len(y_coords), len(x_coords)),
                                   shadow_fading_sigma_db, seed)
        if vectorized:
            return self._raytracing_vectorized(x_coords, y_coords, rx_height_m, coherent, too_close, shadow)
        return self._raytracing_pointwise(x_coords, y_coords, rx_height_m, coherent, too_close, shadow)

    def _raytracing_vectorized(self, x_coords, y_coords, rx_height_m: float, coherent: bool = False,
                               too_close: Optional[np.ndarray] = None,
                               shadow: Optional[np.ndarray] = None) -> SimulationResult:
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
                p_total = rays.total_power_linear(coherent=True) if coherent else power.sum(axis=0)
                with np.errstate(divide="ignore"):
                    p_dbm = 10.0 * np.log10(p_total)         # -inf where no ray arrives
                if shadow is not None:
                    p_dbm = p_dbm - shadow[tx_idx, rows]
                rx_power_dbm[tx_idx, rows] = p_dbm
                path_loss_db[tx_idx, rows] = np.where(p_total > 0.0, tx.tx_power_dbm - p_dbm, np.nan)
                # The spreads always weight each ray by its own power |a_k|^2
                delay_spread_tx[tx_idx, rows] = rms_delay_spread(power, rays.tau_ns)
                angular_spread_tx[tx_idx, rows] = circular_angular_spread(power, rays.tx_phi_deg)[1]

        if too_close is not None:
            # Next to an antenna the far-field result is not reliable: unknown (NaN) for that transmitter
            for grid in (rx_power_dbm, path_loss_db, delay_spread_tx, angular_spread_tx):
                grid[too_close] = np.nan

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

    def _raytracing_pointwise(self, x_coords, y_coords, rx_height_m: float, coherent: bool = False,
                              too_close: Optional[np.ndarray] = None,
                              shadow: Optional[np.ndarray] = None) -> SimulationResult:
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
                pixel_has_unknown = False
                best_cir: Optional[ChannelImpulseResponse] = None

                for tx_idx, tx in enumerate(self.transmitters):
                    if too_close is not None and too_close[tx_idx, j, i]:
                        # Too close to this antenna: power not reliable, it cannot serve the pixel
                        rx_power_dbm[tx_idx, j, i] = float("nan")
                        pixel_has_unknown = True
                        continue

                    cir = self.ray_tracer.trace_paths(tx, rx_pos)
                    p_rx_dbm = cir.total_received_power_dbm(coherent=True) if coherent else cir.total_received_power_dbm()
                    if shadow is not None:
                        p_rx_dbm = p_rx_dbm - shadow[tx_idx, j, i]
                    rx_power_dbm[tx_idx, j, i] = p_rx_dbm

                    if p_rx_dbm > float("-inf"):
                        path_loss_db[tx_idx, j, i] = tx.tx_power_dbm - p_rx_dbm

                    if p_rx_dbm > pixel_max_p_dbm:
                        pixel_max_p_dbm = p_rx_dbm
                        pixel_best_idx = tx_idx
                        best_cir = cir

                if pixel_best_idx < 0 and pixel_has_unknown:
                    pixel_max_p_dbm = float("nan")
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
        rx_height_m: float = 1.5,
        min_distance_m: Optional[float] = DEFAULT_MIN_DISTANCE_M,
        shadow_fading_sigma_db: float = 0.0,
        seed: Optional[int] = None
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

        Sector antennas (``Transmitter(pattern="sector")``): the attenuation of the
        pattern towards the straight direction Tx -> Rx is added to the path loss
        (for Two-Ray, the same value is used for the direct and the ground ray).

        min_distance_m, shadow_fading_sigma_db, seed: same meaning as in
        ``run_raytracing_simulation`` (the distance checked is always the 3D one).

        Empirical models carry no multipath information: the delay-spread and
        angular-spread grids of the result are NaN.  They also ignore the walls
        completely: reflectors neither reflect nor obstruct here (their
        ``transmission_loss_db`` only acts in ``run_raytracing_simulation``).
        """
        x_coords, y_coords = _make_grid(x_range, y_range, resolution_m)
        nx, ny, num_tx = len(x_coords), len(y_coords), len(self.transmitters)
        too_close = self._too_close_mask(x_coords, y_coords, rx_height_m, _validate_min_distance(min_distance_m))
        shadow = _shadow_fading_db((num_tx, ny, nx), shadow_fading_sigma_db, seed)

        rx_power_dbm = np.full((num_tx, ny, nx), float("-inf"))
        path_loss_db = np.full((num_tx, ny, nx), float("nan"))
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        use_ground = bool(getattr(model, "uses_ground_distance", False))
        use_arrays = bool(getattr(model, "supports_arrays", False))

        for tx_idx, tx in enumerate(self.transmitters):
            ground_distance = np.hypot(grid_x - tx.x, grid_y - tx.y)
            distance = ground_distance
            if not use_ground:
                distance = np.sqrt(distance ** 2 + (rx_height_m - tx.z) ** 2)
            kwargs = dict(frequency_hz=tx.frequency_hz, ht_m=tx.z, hr_m=rx_height_m,
                          tx_gain_dbi=tx.antenna_gain_dbi)
            if use_arrays:
                pl = np.asarray(model.path_loss(distance_m=distance, **kwargs), dtype=float)
            else:
                pl = np.array([model.path_loss(distance_m=float(d), **kwargs) for d in distance.ravel()],
                              dtype=float).reshape(distance.shape)
            if not tx.is_isotropic:
                # Pattern towards the direction Tx -> Rx (same angle convention as the ray tracer)
                phi = np.degrees(np.arctan2(grid_y - tx.y, grid_x - tx.x))
                theta = np.degrees(np.arctan2(rx_height_m - tx.z, np.maximum(1e-6, ground_distance)))
                pl = pl - tx.pattern_gain_db(phi, theta)
            if shadow is not None:
                pl = pl + shadow[tx_idx]
            if too_close is not None:
                pl = np.where(too_close[tx_idx], np.nan, pl)
            path_loss_db[tx_idx] = pl
            rx_power_dbm[tx_idx] = tx.tx_power_dbm - pl

        best_index, max_rx_power_dbm = _select_best_server(rx_power_dbm)
        not_available = np.full((ny, nx), float("nan"))
        return self._build_result(x_coords, y_coords, rx_power_dbm, path_loss_db,
                                  best_index, max_rx_power_dbm, not_available, not_available.copy())
