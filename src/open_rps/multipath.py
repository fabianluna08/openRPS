"""
OpenRPS - Multipath & Channel Metrics
Implementation of Channel Impulse Response, Delay Spread, and Angular Spread.

By default power is combined *incoherently* (P = sum |a_k|^2): the phase carried
by each ray amplitude is ignored and small-scale (fast) fading does not appear.
With ``coherent=True`` the received power is P = |sum a_k|^2 instead (complex
sum, constructive/destructive interference between the rays of a transmitter).

The delay-spread and angular-spread metrics always weight each ray by its own
power |a_k|^2, whatever the value of ``coherent``: they describe how the energy
is distributed among the rays, not the narrowband sum of the rays.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def rms_delay_spread(power: np.ndarray, tau_ns: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Power-weighted RMS delay spread along ``axis`` (the ray axis):
        sigma_tau = sqrt( sum(P_k * (tau_k - tau_bar)^2) / sum(P_k) )
    Rays with zero power are ignored (their delay may be anything, even NaN).
    Returns NaN where the total power is zero.
    """
    power = np.asarray(power, dtype=float)
    tau = np.where(power > 0.0, np.asarray(tau_ns, dtype=float), 0.0)
    p_tot = np.sum(power, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_mean = np.sum(power * tau, axis=axis, keepdims=True) / p_tot
        variance = np.sum(power * (tau - tau_mean) ** 2, axis=axis, keepdims=True) / p_tot
        spread = np.sqrt(np.maximum(0.0, variance))
    return np.squeeze(np.where(p_tot > 0.0, spread, np.nan), axis=axis)


def circular_angular_spread(power: np.ndarray, angle_deg: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Power-weighted circular mean angle and RMS angular spread (degrees) along ``axis``.
    The mean is the direction of the power-weighted resultant vector and the
    deviations are wrapped into [-180, 180) around it, so rays at +179 and -179
    degrees are 2 degrees apart, not 358.
    Returns (mean_deg, spread_deg); both NaN where the total power is zero.
    """
    power = np.asarray(power, dtype=float)
    ang = np.where(power > 0.0, np.asarray(angle_deg, dtype=float), 0.0)
    p_tot = np.sum(power, axis=axis, keepdims=True)
    rad = np.radians(ang)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_mean = np.sum(power * np.sin(rad), axis=axis, keepdims=True) / p_tot
        cos_mean = np.sum(power * np.cos(rad), axis=axis, keepdims=True) / p_tot
        mean_deg = np.degrees(np.arctan2(sin_mean, cos_mean))
        diff = (ang - mean_deg + 180.0) % 360.0 - 180.0
        spread = np.sqrt(np.maximum(0.0, np.sum(power * diff ** 2, axis=axis, keepdims=True) / p_tot))
    ok = p_tot > 0.0
    return (np.squeeze(np.where(ok, mean_deg, np.nan), axis=axis),
            np.squeeze(np.where(ok, spread, np.nan), axis=axis))


@dataclass
class MultipathRay:
    """
    Represents a single propagation ray from a transmitter to a receiver point.
    """
    tx_id: int
    magnitude: complex     # Complex amplitude a_k, normalized so that |a_k|^2 is the ray power in mW
    tau_ns: float          # Propagation delay in nanoseconds
    tx_phi_deg: float      # Azimuth angle of departure at Transmitter / BS (-180 to 180)
    tx_theta_deg: float    # Elevation angle of departure at Transmitter / BS (-90 to 90)
    rx_phi_deg: float      # Azimuth angle of arrival at Receiver / MS (-180 to 180)
    rx_theta_deg: float    # Elevation angle of arrival at Receiver / MS (-90 to 90)

    @property
    def power_linear(self) -> float:
        """Received power per ray in mW (P_k = |a_k|^2)."""
        return abs(self.magnitude)**2


@dataclass
class ChannelImpulseResponse:
    """
    Channel Impulse Response (CIR) for a specific receiver location:
    h(t) = sum_k a_k * exp(j * theta_k) * delta(t - tau_k)
    """
    rays: List[MultipathRay] = field(default_factory=list)

    def add_ray(self, ray: MultipathRay):
        self.rays.append(ray)

    def _select(self, tx_id: Optional[int]) -> List[MultipathRay]:
        return [r for r in self.rays if tx_id is None or r.tx_id == tx_id]

    def total_received_power_linear(self, tx_id: Optional[int] = None, coherent: bool = False) -> float:
        """
        Total received power in mW from a specific transmitter (or all transmitters).

        coherent=False (default): incoherent sum, P_r = sum_k |a_k|^2.
        coherent=True: complex (phasor) sum of the rays of each transmitter,
            P_r = |sum_k a_k|^2, which shows fast fading.  Rays of *different*
            transmitters are independent signals, so with ``tx_id=None`` the
            coherent powers of the transmitters are added incoherently.
        """
        rays = self._select(tx_id)
        if not coherent:
            return float(sum(r.power_linear for r in rays))
        field_by_tx = {}
        for r in rays:
            field_by_tx[r.tx_id] = field_by_tx.get(r.tx_id, 0j) + r.magnitude
        return float(sum(abs(field) ** 2 for field in field_by_tx.values()))

    def total_received_power_dbm(self, tx_id: Optional[int] = None, coherent: bool = False) -> float:
        """
        Received power in dBm (10 * log10(P_r_mW)); -inf if there is no power
        (no rays, or a perfect cancellation with ``coherent=True``).
        """
        p_linear = self.total_received_power_linear(tx_id, coherent=coherent)
        if p_linear <= 0:
            return float("-inf")
        return 10.0 * math.log10(p_linear)

    def mean_excess_delay(self, tx_id: Optional[int] = None, relative_to_first: bool = False) -> float:
        """
        Power-weighted mean delay (tau_bar) in nanoseconds:
        tau_bar = sum(P_k * tau_k) / sum(P_k)

        relative_to_first=False (default, behaviour of OpenRPS <= 1.1): tau_k are the
            absolute propagation delays (d_k / c), so the result is the mean
            *absolute* delay.
        relative_to_first=True: the delay of the first arrival (the ray with the
            smallest delay among those carrying power) is subtracted, which gives
            the mean excess delay in the strict sense (0 for a single ray).
        """
        target_rays = self._select(tx_id)
        p_tot = sum(r.power_linear for r in target_rays)
        if p_tot <= 0:
            return float("nan")
        mean_delay = sum(r.power_linear * r.tau_ns for r in target_rays) / p_tot
        if relative_to_first:
            mean_delay -= min(r.tau_ns for r in target_rays if r.power_linear > 0)
        return mean_delay

    def rms_delay_spread(self, tx_id: Optional[int] = None) -> float:
        """
        Root Mean Square (RMS) Delay Spread (sigma_tau) in nanoseconds:
        sigma_tau = sqrt( sum(P_k * tau_k^2)/P_tot - (tau_bar)^2 )
        (evaluated in the equivalent, numerically safer centered form).
        The weights are always the ray powers P_k = |a_k|^2 (never a coherent sum).
        """
        target_rays = self._select(tx_id)
        p_tot = sum(r.power_linear for r in target_rays)
        if p_tot <= 0:
            return float("nan")
        tau_mean = sum(r.power_linear * r.tau_ns for r in target_rays) / p_tot
        variance = sum(r.power_linear * (r.tau_ns - tau_mean) ** 2 for r in target_rays) / p_tot
        return math.sqrt(max(0.0, variance))

    def angular_spread(self, plane: str = "phi", station: str = "bs", tx_id: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculates the mean angle and circular RMS angular spread (in degrees).
        plane: 'phi' (azimuth) or 'theta' (elevation)
        station: 'bs' (transmitter) or 'ms' (receiver)

        Uses circular angle wrapping in [-180, 180] around mean angle, weighting
        every ray by its own power |a_k|^2 (never a coherent sum).
        Returns: (mean_angle_deg, rms_angular_spread_deg)
        """
        plane = plane.lower()
        station = station.lower()
        if plane not in ("phi", "theta"):
            raise ValueError(f"plane must be 'phi' or 'theta', got {plane!r}")
        if station not in ("bs", "ms"):
            raise ValueError(f"station must be 'bs' or 'ms', got {station!r}")

        target_rays = self._select(tx_id)
        p_tot = sum(r.power_linear for r in target_rays)
        if p_tot <= 0:
            return (float("nan"), float("nan"))

        attr = ("tx_" if station == "bs" else "rx_") + plane + "_deg"
        powers = [r.power_linear for r in target_rays]
        angles = [getattr(r, attr) for r in target_rays]

        # Weighted circular mean (resultant vector), which has no discontinuity at +/-180
        sin_mean = sum(p * math.sin(math.radians(a)) for p, a in zip(powers, angles)) / p_tot
        cos_mean = sum(p * math.cos(math.radians(a)) for p, a in zip(powers, angles)) / p_tot
        mean_angle_deg = math.degrees(math.atan2(sin_mean, cos_mean))

        # Deviations wrapped into [-180, 180) around the mean, then weighted RMS
        variance = sum(p * ((a - mean_angle_deg + 180.0) % 360.0 - 180.0) ** 2
                       for p, a in zip(powers, angles)) / p_tot
        return (mean_angle_deg, math.sqrt(max(0.0, variance)))
