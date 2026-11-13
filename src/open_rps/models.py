"""
OpenRPS - Propagation Models
Implementation of standard radiowave propagation models.

Conventions shared by every built-in model
------------------------------------------
* ``basic_path_loss(distance_m, frequency_hz, ...)`` returns the loss in dB
  between two *isotropic* antennas (no antenna gains).
* ``path_loss(distance_m, frequency_hz, tx_gain_dbi=0, rx_gain_dbi=0, ...)``
  returns the link loss ``basic_path_loss - tx_gain_dbi - rx_gain_dbi``, so that
  ``P_rx[dBm] = P_tx[dBm] - path_loss`` holds for every model.  The result may
  be negative when the gains exceed the basic loss (very short links).
* ``distance_m`` may be a float or a numpy array (the result has the same
  shape).  Its meaning depends on ``uses_ground_distance``:
  ``False`` -> 3D (slant) distance between the antennas (FSPL, Log-Distance);
  ``True``  -> horizontal distance along the ground (Two-Ray, Hata, COST-231),
  the antenna heights being given separately through ``ht_m`` / ``hr_m``.
"""

import cmath
import math
import warnings
from typing import Union

import numpy as np

SPEED_OF_LIGHT = 299792458.0   # m/s
EPS_0 = 8.854187817e-12        # F/m

ArrayLike = Union[float, np.ndarray]


class ModelRangeWarning(UserWarning):
    """A model is being evaluated outside the range in which it was fitted/validated."""


def _validate_frequency(frequency_hz: float) -> float:
    try:
        f = float(frequency_hz)
    except (TypeError, ValueError):
        raise ValueError(f"frequency_hz must be a positive scalar, got {frequency_hz!r}") from None
    if not (np.isfinite(f) and f > 0.0):
        raise ValueError(f"frequency_hz must be a positive finite number, got {frequency_hz!r}")
    return f


def _validate_distance(distance_m: ArrayLike) -> np.ndarray:
    d = np.asarray(distance_m, dtype=float)
    if np.any(d < 0.0):
        raise ValueError("distance_m must be >= 0")
    return d


def _validate_height(value: float, name: str) -> float:
    h = float(value)
    if not (np.isfinite(h) and h > 0.0):
        raise ValueError(f"{name} must be a positive finite height in meters, got {value!r}")
    return h


def _match_input(result: np.ndarray, like: ArrayLike) -> ArrayLike:
    """Return a Python float for scalar inputs, an ndarray otherwise."""
    return float(result) if np.ndim(like) == 0 else result


def fresnel_reflection_coefficient(
    grazing_angle_rad: ArrayLike,
    frequency_hz: float,
    eps_r: float,
    sigma: float = 0.0,
    polarization: str = "vertical",
) -> Union[complex, np.ndarray]:
    """
    Fresnel reflection coefficient of a smooth, flat, homogeneous half-space.

    grazing_angle_rad: angle between the incident ray and the surface (0 = grazing).
    eps_r, sigma:      relative permittivity and conductivity [S/m] of the medium.
    polarization:      'vertical'   -> E field in the plane of incidence (TM, parallel);
                       'horizontal' -> E field parallel to the surface (TE, perpendicular).
    """
    pol = polarization.lower()
    if pol not in ("vertical", "horizontal"):
        raise ValueError(f"polarization must be 'vertical' or 'horizontal', got {polarization!r}")
    f = _validate_frequency(frequency_hz)
    eps_c = complex(eps_r, -sigma / (2.0 * math.pi * f * EPS_0))

    if isinstance(grazing_angle_rad, (int, float)):
        # Scalar fast path (keeps the per-point ray tracer free of numpy overhead)
        sin_psi = math.sin(grazing_angle_rad)
        sqrt_term = cmath.sqrt(eps_c - math.cos(grazing_angle_rad) ** 2)
    else:
        psi = np.asarray(grazing_angle_rad, dtype=float)
        sin_psi = np.sin(psi)
        sqrt_term = np.sqrt(eps_c - np.cos(psi) ** 2)
    if pol == "horizontal":
        gamma = (sin_psi - sqrt_term) / (sin_psi + sqrt_term)
    else:
        gamma = (eps_c * sin_psi - sqrt_term) / (eps_c * sin_psi + sqrt_term)
    return complex(gamma) if np.ndim(grazing_angle_rad) == 0 else gamma


class PropagationModel:
    """
    Base class for radio propagation models.

    Subclasses implement ``basic_path_loss`` (isotropic loss, no gains) and inherit
    ``path_loss`` (which applies the antenna gains).  Overriding ``path_loss``
    directly, as in OpenRPS 1.0, is still supported.

    Class attributes read by ``PropagationSimulator.run_empirical_simulation``:
      supports_arrays      -- True if the model accepts numpy arrays as ``distance_m``
                              (otherwise it is evaluated point by point).
      uses_ground_distance -- True if ``distance_m`` means horizontal distance.
    """
    supports_arrays = False
    uses_ground_distance = False

    def basic_path_loss(self, distance_m: ArrayLike, frequency_hz: float, **kwargs) -> ArrayLike:
        raise NotImplementedError

    def path_loss(self, distance_m: ArrayLike, frequency_hz: float,
                  tx_gain_dbi: float = 0.0, rx_gain_dbi: float = 0.0, **kwargs) -> ArrayLike:
        """Link loss in dB including antenna gains: basic_path_loss - G_tx - G_rx."""
        return self.basic_path_loss(distance_m, frequency_hz, **kwargs) - tx_gain_dbi - rx_gain_dbi


class FreeSpacePathLoss(PropagationModel):
    """
    Free Space Path Loss (FSPL) using Friis transmission equation:
    PL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi / c)

    ``distance_m`` is the 3D distance between the antennas.  The basic loss is
    clamped at 0 dB (Friis is a far-field formula; for d < lambda/4pi it would
    predict a gain), and d = 0 returns 0 dB.
    """
    SPEED_OF_LIGHT = SPEED_OF_LIGHT
    supports_arrays = True
    uses_ground_distance = False

    def basic_path_loss(self, distance_m: ArrayLike, frequency_hz: float, **kwargs) -> ArrayLike:
        f = _validate_frequency(frequency_hz)
        d = _validate_distance(distance_m)
        wavelength = self.SPEED_OF_LIGHT / f
        with np.errstate(divide="ignore"):
            fspl_db = 20.0 * np.log10(4.0 * np.pi * d / wavelength)
        return _match_input(np.maximum(0.0, fspl_db), distance_m)


class TwoRayGroundReflection(PropagationModel):
    """
    Two-Ray Ground Reflection Model (Direct + Ground specular reflection).
    Exact coherent (phase) addition of both rays; for d >> 4*ht*hr/lambda it
    tends to the well-known asymptote PL = 40*log10(d) - 20*log10(ht*hr).

    ``distance_m`` is the horizontal (ground) distance between the antennas.
    """
    SPEED_OF_LIGHT = SPEED_OF_LIGHT
    supports_arrays = True
    uses_ground_distance = True

    def __init__(self, eps_r: float = 15.0, sigma: float = 0.005, polarization: str = "vertical"):
        polarization = polarization.lower()
        if polarization not in ("vertical", "horizontal"):
            raise ValueError(f"polarization must be 'vertical' or 'horizontal', got {polarization!r}")
        self.eps_r = eps_r
        self.sigma = sigma
        self.polarization = polarization

    def reflection_coefficient(self, grazing_angle_rad: ArrayLike, frequency_hz: float) -> Union[complex, np.ndarray]:
        return fresnel_reflection_coefficient(grazing_angle_rad, frequency_hz,
                                              self.eps_r, self.sigma, self.polarization)

    def basic_path_loss(self, distance_m: ArrayLike, frequency_hz: float,
                        ht_m: float = 10.0, hr_m: float = 1.5, **kwargs) -> ArrayLike:
        f = _validate_frequency(frequency_hz)
        d = _validate_distance(distance_m)
        if ht_m < 0 or hr_m < 0:
            raise ValueError("ht_m and hr_m must be >= 0")

        wavelength = self.SPEED_OF_LIGHT / f
        k = 2.0 * np.pi / wavelength

        d_los = np.sqrt(d ** 2 + (ht_m - hr_m) ** 2)
        d_ref = np.sqrt(d ** 2 + (ht_m + hr_m) ** 2)
        grazing_angle = np.arctan2(ht_m + hr_m, d)
        gamma = np.asarray(self.reflection_coefficient(grazing_angle, f))

        # Complex field sum (relative to the free-space field at 1 m)
        with np.errstate(divide="ignore", invalid="ignore"):
            e_total = np.exp(-1j * k * d_los) / d_los + gamma * np.exp(-1j * k * d_ref) / d_ref
            pl_linear = (wavelength / (4.0 * np.pi)) ** 2 * np.abs(e_total) ** 2
            pl_db = -10.0 * np.log10(pl_linear)
        # Perfect cancellation (deep null) -> cap at 200 dB; coincident antennas -> 0 dB
        pl_db = np.where(pl_linear > 0.0, pl_db, 200.0)
        pl_db = np.where(d_los > 0.0, pl_db, 0.0)
        return _match_input(pl_db, distance_m)


class LogDistancePathLoss(PropagationModel):
    """
    Log-Distance Path Loss Model:
    PL(d) = PL(d0) + 10 * n * log10(d / d0),   with PL(d0) = FSPL(d0)

    ``distance_m`` is the 3D distance; for d <= d0 the loss is held at PL(d0).
    """
    supports_arrays = True
    uses_ground_distance = False

    def __init__(self, path_loss_exponent: float = 3.0, d0_m: float = 1.0):
        if not d0_m > 0:
            raise ValueError(f"d0_m must be > 0, got {d0_m!r}")
        self.n = path_loss_exponent
        self.d0 = d0_m
        self.fspl = FreeSpacePathLoss()

    def basic_path_loss(self, distance_m: ArrayLike, frequency_hz: float, **kwargs) -> ArrayLike:
        d = _validate_distance(distance_m)
        pl_d0 = self.fspl.basic_path_loss(self.d0, frequency_hz)
        pl = pl_d0 + 10.0 * self.n * np.log10(np.maximum(d, self.d0) / self.d0)
        return _match_input(pl, distance_m)


def _warn_out_of_range(model_name: str, f_mhz: float, f_range, ht_m: float, hr_m: float, d_km: np.ndarray):
    """Emit a single ModelRangeWarning listing every parameter outside the model's validity range."""
    problems = []
    if not (f_range[0] <= f_mhz <= f_range[1]):
        problems.append(f"frequency {f_mhz:g} MHz outside {f_range[0]:g}-{f_range[1]:g} MHz")
    if not (30.0 <= ht_m <= 200.0):
        problems.append(f"base antenna height {ht_m:g} m outside 30-200 m")
    if not (1.0 <= hr_m <= 10.0):
        problems.append(f"mobile antenna height {hr_m:g} m outside 1-10 m")
    if d_km.size and (np.nanmin(d_km) < 1.0 or np.nanmax(d_km) > 20.0):
        problems.append("distance outside 1-20 km")
    if problems:
        warnings.warn(
            f"{model_name} evaluated outside its validity range ({'; '.join(problems)}); "
            "the result is an extrapolation and may be badly wrong.",
            ModelRangeWarning, stacklevel=5)


class OkumuraHata(PropagationModel):
    """
    Okumura-Hata empirical macrocell propagation model.
    Validity range: 150-1500 MHz, base antenna 30-200 m, mobile antenna 1-10 m,
    distance 1-20 km.  Outside that range a ``ModelRangeWarning`` is emitted and
    the formula is extrapolated (use ``Cost231Hata`` for 1500-2000 MHz).
    Environment types: 'urban_large', 'urban_medium', 'suburban', 'open' (rural).

    ``distance_m`` is the horizontal distance; it is clamped to a minimum of 10 m.
    """
    ENVIRONMENTS = ("urban_large", "urban_medium", "suburban", "open")
    FREQ_RANGE_MHZ = (150.0, 1500.0)
    supports_arrays = True
    uses_ground_distance = True

    def __init__(self, environment: str = "urban_medium"):
        env = environment.lower()
        if env not in self.ENVIRONMENTS:
            raise ValueError(f"environment must be one of {self.ENVIRONMENTS}, got {environment!r}")
        self.env = env

    def _mobile_height_correction(self, f_mhz: float, hr_m: float) -> float:
        """a(h_r): correction factor for the mobile antenna height, in dB."""
        if self.env == "urban_large":
            if f_mhz >= 300:
                return 3.2 * (np.log10(11.75 * hr_m)) ** 2 - 4.97
            return 8.29 * (np.log10(1.54 * hr_m)) ** 2 - 1.1
        return (1.1 * np.log10(f_mhz) - 0.7) * hr_m - (1.56 * np.log10(f_mhz) - 0.8)

    def _prepare(self, distance_m: ArrayLike, frequency_hz: float, ht_m: float, hr_m: float):
        f_mhz = _validate_frequency(frequency_hz) / 1e6
        ht_m = _validate_height(ht_m, "ht_m")
        hr_m = _validate_height(hr_m, "hr_m")
        d_km = np.maximum(0.01, _validate_distance(distance_m) / 1000.0)
        _warn_out_of_range(type(self).__name__, f_mhz, self.FREQ_RANGE_MHZ, ht_m, hr_m, d_km)
        return f_mhz, ht_m, hr_m, d_km

    def basic_path_loss(self, distance_m: ArrayLike, frequency_hz: float,
                        ht_m: float = 30.0, hr_m: float = 1.5, **kwargs) -> ArrayLike:
        f_mhz, ht_m, hr_m, d_km = self._prepare(distance_m, frequency_hz, ht_m, hr_m)
        a_hr = self._mobile_height_correction(f_mhz, hr_m)

        # Base urban loss
        pl = (69.55 + 26.16 * np.log10(f_mhz) - 13.82 * np.log10(ht_m) -
              a_hr + (44.9 - 6.55 * np.log10(ht_m)) * np.log10(d_km))

        if self.env == "suburban":
            pl = pl - (2.0 * (np.log10(f_mhz / 28.0)) ** 2 + 5.4)
        elif self.env == "open":
            pl = pl - (4.78 * (np.log10(f_mhz)) ** 2 - 18.33 * np.log10(f_mhz) + 40.94)

        return _match_input(np.maximum(0.0, pl), distance_m)


class Cost231Hata(OkumuraHata):
    """
    COST-231 Hata model: extension of Okumura-Hata to 1500-2000 MHz.
    PL = 46.3 + 33.9*log10(f) - 13.82*log10(ht) - a(hr) + (44.9 - 6.55*log10(ht))*log10(d) + C
    with C = 3 dB for metropolitan centres ('urban_large', which also uses the
    large-city a(hr)) and C = 0 dB for 'urban_medium' and 'suburban'.
    Same height and distance ranges as Okumura-Hata.
    """
    ENVIRONMENTS = ("urban_large", "urban_medium", "suburban")
    FREQ_RANGE_MHZ = (1500.0, 2000.0)

    def basic_path_loss(self, distance_m: ArrayLike, frequency_hz: float,
                        ht_m: float = 30.0, hr_m: float = 1.5, **kwargs) -> ArrayLike:
        f_mhz, ht_m, hr_m, d_km = self._prepare(distance_m, frequency_hz, ht_m, hr_m)
        a_hr = self._mobile_height_correction(f_mhz, hr_m)
        c_m = 3.0 if self.env == "urban_large" else 0.0

        pl = (46.3 + 33.9 * np.log10(f_mhz) - 13.82 * np.log10(ht_m) -
              a_hr + (44.9 - 6.55 * np.log10(ht_m)) * np.log10(d_km) + c_m)
        return _match_input(np.maximum(0.0, pl), distance_m)
