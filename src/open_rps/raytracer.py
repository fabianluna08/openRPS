"""
OpenRPS - 3D Ray-Tracing Engine
Computes multipath propagation rays (Direct, Ground Reflected, and Wall Reflected)
with the image method.

Two equivalent entry points are provided:
  * ``RayTracer.trace_paths``  -- one receiver point, returns a ChannelImpulseResponse.
  * ``RayTracer.trace_points`` -- many receiver points at once (numpy), returns a RayField.

Physical scope (see README, "Limitaciones fisicas"): single-bounce specular
reflections only, no obstruction/shadowing, no diffraction, isotropic antennas.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .models import SPEED_OF_LIGHT, fresnel_reflection_coefficient
from .multipath import MultipathRay, ChannelImpulseResponse


@dataclass
class PlanarReflector:
    """
    Represents a flat reflecting wall or building face.

    The wall lies in the plane through ``p0`` with normal ``normal``.  Both faces
    of the plane reflect.  A specular reflection exists only when transmitter and
    receiver are on the same side of the plane.

    bounds: optional axis-aligned box (xmin, xmax, ymin, ymax, zmin, zmax) that
        limits the reflecting patch: the reflection point must fall inside it.
        Use +/-inf for an open side.  ``None`` (default) means an infinite plane.
    reflectivity: constant amplitude reflection coefficient magnitude (0..1).  The
        reflected ray gets a fixed 180 deg phase shift, i.e. coefficient
        ``-reflectivity`` (the sign of a TE reflection off a denser medium).
    eps_r, sigma: if ``eps_r`` is given, ``reflectivity`` is ignored and the
        angle-dependent Fresnel coefficient of a smooth dielectric wall is used
        instead, for the E field parallel to the wall surface (TE).  This is the
        usual case of vertical polarization on a vertical wall with near-horizontal
        propagation; it is an approximation for any other geometry.

    Reflectors never block rays: the direct, ground and other wall rays pass
    through them unattenuated.
    """
    p0: np.ndarray      # Reference point on plane [x, y, z]
    normal: np.ndarray  # Normal vector of the plane [nx, ny, nz] (normalized on creation)
    reflectivity: float = 0.5  # Amplitude reflection coefficient (0.0 to 1.0)
    name: str = "Wall"
    bounds: Optional[Sequence[float]] = None  # (xmin, xmax, ymin, ymax, zmin, zmax)
    eps_r: Optional[float] = None             # Relative permittivity (enables Fresnel)
    sigma: float = 0.0                        # Conductivity [S/m], used with eps_r

    def __post_init__(self):
        self.p0 = np.array(self.p0, dtype=float)
        self.normal = np.array(self.normal, dtype=float)
        if self.p0.shape != (3,) or self.normal.shape != (3,):
            raise ValueError("p0 and normal must be 3-element vectors [x, y, z]")
        norm = float(np.linalg.norm(self.normal))
        if not (np.isfinite(norm) and norm > 0.0):
            raise ValueError("normal must be a non-zero finite vector")
        self.normal = self.normal / norm
        if not (0.0 <= self.reflectivity <= 1.0):
            raise ValueError(f"reflectivity must be within [0, 1], got {self.reflectivity!r}")
        if self.eps_r is not None and not self.eps_r >= 1.0:
            raise ValueError(f"eps_r must be >= 1, got {self.eps_r!r}")
        if self.bounds is not None:
            b = tuple(float(v) for v in self.bounds)
            if len(b) != 6:
                raise ValueError("bounds must be (xmin, xmax, ymin, ymax, zmin, zmax)")
            if b[0] > b[1] or b[2] > b[3] or b[4] > b[5]:
                raise ValueError("bounds must satisfy xmin <= xmax, ymin <= ymax, zmin <= zmax")
            self.bounds = b

    def contains(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """True where ``points`` (shape [..., 3]) fall inside ``bounds`` (always True if unbounded)."""
        points = np.asarray(points, dtype=float)
        if self.bounds is None:
            return np.ones(points.shape[:-1], dtype=bool)
        b = self.bounds
        inside = np.ones(points.shape[:-1], dtype=bool)
        for axis in range(3):
            inside &= (points[..., axis] >= b[2 * axis] - tol) & (points[..., axis] <= b[2 * axis + 1] + tol)
        return inside

    def reflection_coefficient(self, sin_grazing, frequency_hz: float):
        """
        Complex amplitude reflection coefficient for rays hitting the wall.
        sin_grazing: sine of the angle between the ray and the wall surface (float or array).
        """
        scalar = isinstance(sin_grazing, (int, float))
        if self.eps_r is None:
            # Constant magnitude, fixed 180 deg phase flip
            if scalar:
                return complex(-self.reflectivity)
            return np.full(np.shape(sin_grazing), -self.reflectivity, dtype=complex)
        if scalar:
            psi = math.asin(min(1.0, max(0.0, sin_grazing)))
        else:
            psi = np.arcsin(np.clip(np.asarray(sin_grazing, dtype=float), 0.0, 1.0))
        return fresnel_reflection_coefficient(psi, frequency_hz, self.eps_r, self.sigma, "horizontal")


@dataclass
class Transmitter:
    """
    Represents a Base Station / Transmitter.
    The antenna is isotropic: ``antenna_gain_dbi`` is applied equally in every
    direction (no radiation pattern, no sectorization).
    """
    tx_id: int
    x: float
    y: float
    z: float
    tx_power_dbm: float = 43.0  # Default 20 Watts (43 dBm)
    frequency_hz: float = 2.4e9 # Default 2.4 GHz
    antenna_gain_dbi: float = 14.0 # Applied in all directions

    def __post_init__(self):
        if not (np.isfinite(self.frequency_hz) and self.frequency_hz > 0.0):
            raise ValueError(f"frequency_hz must be a positive finite number, got {self.frequency_hz!r}")

    @property
    def power_mw(self) -> float:
        return 10.0 ** (self.tx_power_dbm / 10.0)


@dataclass
class RayField:
    """
    Rays from one transmitter to many receiver points (output of ``RayTracer.trace_points``).

    Every array has shape [K, *points_shape], K being the number of candidate rays:
    index 0 = direct, 1 = ground reflection, 2.. = one per reflector (in the order
    they were added).  Where a ray does not exist ``valid`` is False, its amplitude
    is 0 and its delay/angles are NaN.
    """
    tx_id: int
    amplitude: np.ndarray     # complex; |a|^2 is the ray power in mW
    tau_ns: np.ndarray
    tx_phi_deg: np.ndarray
    tx_theta_deg: np.ndarray
    rx_phi_deg: np.ndarray
    rx_theta_deg: np.ndarray
    valid: np.ndarray         # bool

    @property
    def power_linear(self) -> np.ndarray:
        """Power per ray in mW (0 where the ray does not exist)."""
        return np.abs(self.amplitude) ** 2


def _angles_deg(p_src: np.ndarray, p_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized azimuth/elevation (degrees) of the direction src -> dst; inputs broadcast to [..., 3]."""
    v = p_dst - p_src
    dist_2d = np.hypot(v[..., 0], v[..., 1])
    phi = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    theta = np.degrees(np.arctan2(v[..., 2], np.maximum(1e-6, dist_2d)))
    return phi, theta


class RayTracer:
    """
    Simulates radio ray paths between transmitters and receivers.
    The ground is the plane z = 0 (smooth, homogeneous, vertical polarization).
    """
    def __init__(self, ground_eps_r: float = 15.0, ground_sigma: float = 0.005):
        self.ground_eps_r = ground_eps_r
        self.ground_sigma = ground_sigma
        self.reflectors: List[PlanarReflector] = []

    def add_reflector(self, reflector: PlanarReflector):
        self.reflectors.append(reflector)

    def _calc_angles(self, p_src: np.ndarray, p_dst: np.ndarray) -> Tuple[float, float]:
        """Calculates azimuth (phi: -180..180) and elevation (theta: -90..90) from src to dst."""
        v = p_dst - p_src
        dist_2d = math.hypot(v[0], v[1])
        phi = math.degrees(math.atan2(v[1], v[0]))
        theta = math.degrees(math.atan2(v[2], max(1e-6, dist_2d)))
        return phi, theta

    def _fresnel_ground_reflection(self, theta_grazing_rad, freq_hz: float):
        """Fresnel reflection coefficient for ground plane (vertical polarization, standard for cellular)."""
        return fresnel_reflection_coefficient(theta_grazing_rad, freq_hz,
                                              self.ground_eps_r, self.ground_sigma, "vertical")

    def trace_paths(self, tx: Transmitter, rx_pos: Tuple[float, float, float]) -> ChannelImpulseResponse:
        """
        Traces multipath rays from transmitter to receiver position:
        1. Direct Line of Sight (LoS) Ray
        2. Ground Reflected Ray
        3. Specular Reflections from Planar Reflectors / Walls
        """
        cir = ChannelImpulseResponse()
        tx_p = np.array([tx.x, tx.y, tx.z], dtype=float)
        rx_p = np.array(rx_pos, dtype=float)
        wavelength = SPEED_OF_LIGHT / tx.frequency_hz
        k = 2.0 * math.pi / wavelength
        # Field amplitude at 1 m scaled so that |a|^2 is power in mW (Friis):
        # |a| = sqrt(P_t * G_t) * lambda / (4*pi*d)
        amp_1m = math.sqrt(tx.power_mw * (10**(tx.antenna_gain_dbi/10.0))) * wavelength / (4.0 * math.pi)

        # --- 1. Direct LoS Ray ---
        d_direct = float(np.linalg.norm(rx_p - tx_p))
        if d_direct > 0:
            tau_direct = (d_direct / SPEED_OF_LIGHT) * 1e9 # ns
            amp_direct = amp_1m / d_direct
            phase_direct = -k * d_direct
            mag_direct = amp_direct * complex(math.cos(phase_direct), math.sin(phase_direct))

            tx_phi, tx_theta = self._calc_angles(tx_p, rx_p)
            rx_phi, rx_theta = self._calc_angles(rx_p, tx_p)

            cir.add_ray(MultipathRay(
                tx_id=tx.tx_id,
                magnitude=mag_direct,
                tau_ns=tau_direct,
                tx_phi_deg=tx_phi,
                tx_theta_deg=tx_theta,
                rx_phi_deg=rx_phi,
                rx_theta_deg=rx_theta
            ))

        # --- 2. Ground Reflected Ray ---
        # Image method for flat ground at z=0: Tx image is at (tx.x, tx.y, -tx.z).
        # Needs both antennas on or above the ground and at least one strictly above.
        tx_image_ground = np.array([tx.x, tx.y, -tx.z], dtype=float)
        d_ground = float(np.linalg.norm(rx_p - tx_image_ground))
        if d_ground > 0 and tx.z >= 0 and rx_p[2] >= 0 and (tx.z + rx_p[2]) > 0:
            tau_ground = (d_ground / SPEED_OF_LIGHT) * 1e9
            grazing_ang = math.atan2(tx.z + rx_p[2], max(1e-6, math.hypot(rx_p[0] - tx.x, rx_p[1] - tx.y)))
            gamma_ground = self._fresnel_ground_reflection(grazing_ang, tx.frequency_hz)

            amp_ground = (amp_1m / d_ground) * abs(gamma_ground)
            gamma_phase = math.atan2(gamma_ground.imag, gamma_ground.real)
            phase_ground = -k * d_ground + gamma_phase
            mag_ground = amp_ground * complex(math.cos(phase_ground), math.sin(phase_ground))

            # The bounce point is where the straight line Tx_image -> Rx crosses z = 0:
            #   p_bounce = tx_image + t * (rx - tx_image),  t = tx.z / (tx.z + rx.z)
            # The ray leaves Tx towards p_bounce, i.e. towards the mirror image of Rx,
            # and reaches Rx coming from p_bounce, i.e. from the mirror image of Tx.
            # Using the images keeps the angles well defined even when one antenna
            # sits exactly on the ground (p_bounce then coincides with it).
            rx_image_ground = np.array([rx_p[0], rx_p[1], -rx_p[2]], dtype=float)
            tx_phi_g, tx_theta_g = self._calc_angles(tx_p, rx_image_ground)
            rx_phi_g, rx_theta_g = self._calc_angles(rx_p, tx_image_ground)

            cir.add_ray(MultipathRay(
                tx_id=tx.tx_id,
                magnitude=mag_ground,
                tau_ns=tau_ground,
                tx_phi_deg=tx_phi_g,
                tx_theta_deg=tx_theta_g,
                rx_phi_deg=rx_phi_g,
                rx_theta_deg=rx_theta_g
            ))

        # --- 3. Wall Specular Reflections ---
        for refl in self.reflectors:
            # Signed distances of Tx and Rx to the plane
            s_tx = float(np.dot(tx_p - refl.p0, refl.normal))
            s_rx = float(np.dot(rx_p - refl.p0, refl.normal))

            # A specular reflection needs Tx and Rx strictly on the same side of the wall
            if s_tx * s_rx <= 0.0 or abs(s_tx + s_rx) < 1e-6:
                continue

            # Reflection image of Tx across plane: Tx_img = Tx - 2 * ((Tx - P0) . N) * N
            tx_image_wall = tx_p - 2.0 * s_tx * refl.normal
            d_wall = float(np.linalg.norm(rx_p - tx_image_wall))

            # Reflection point: where the line Tx_img -> Rx crosses the plane (0 < t < 1)
            t_intersect = s_tx / (s_tx + s_rx)
            p_wall_hit = tx_image_wall + t_intersect * (rx_p - tx_image_wall)
            if refl.bounds is not None and not refl.contains(p_wall_hit):
                continue

            tau_wall = (d_wall / SPEED_OF_LIGHT) * 1e9
            sin_grazing = abs(s_tx + s_rx) / d_wall
            gamma_wall = refl.reflection_coefficient(sin_grazing, tx.frequency_hz)
            phase_wall = -k * d_wall
            mag_wall = (amp_1m / d_wall) * gamma_wall * complex(math.cos(phase_wall), math.sin(phase_wall))

            tx_phi_w, tx_theta_w = self._calc_angles(tx_p, p_wall_hit)
            rx_phi_w, rx_theta_w = self._calc_angles(rx_p, p_wall_hit)

            cir.add_ray(MultipathRay(
                tx_id=tx.tx_id,
                magnitude=mag_wall,
                tau_ns=tau_wall,
                tx_phi_deg=tx_phi_w,
                tx_theta_deg=tx_theta_w,
                rx_phi_deg=rx_phi_w,
                rx_theta_deg=rx_theta_w
            ))

        return cir

    def trace_points(self, tx: Transmitter, points: np.ndarray) -> RayField:
        """
        Vectorized equivalent of ``trace_paths`` for an array of receiver points.

        points: array of shape [..., 3] with the receiver coordinates (x, y, z).
        Returns a RayField whose arrays have shape [2 + n_reflectors, ...].
        """
        pts = np.asarray(points, dtype=float)
        if pts.shape[-1] != 3:
            raise ValueError("points must have shape [..., 3]")
        shape = pts.shape[:-1]
        n_rays = 2 + len(self.reflectors)

        amplitude = np.zeros((n_rays,) + shape, dtype=complex)
        tau_ns = np.full((n_rays,) + shape, np.nan)
        angles = np.full((4, n_rays) + shape, np.nan)   # tx_phi, tx_theta, rx_phi, rx_theta
        valid = np.zeros((n_rays,) + shape, dtype=bool)

        tx_p = np.array([tx.x, tx.y, tx.z], dtype=float)
        wavelength = SPEED_OF_LIGHT / tx.frequency_hz
        k = 2.0 * math.pi / wavelength
        amp_1m = math.sqrt(tx.power_mw * (10**(tx.antenna_gain_dbi/10.0))) * wavelength / (4.0 * math.pi)

        def store(idx, ok, dist, coeff, tx_target, rx_target):
            """Fill ray ``idx``: path length ``dist``, reflection coefficient ``coeff``, aim points."""
            with np.errstate(divide="ignore", invalid="ignore"):
                amp = (amp_1m / dist) * coeff * np.exp(-1j * k * dist)
            amplitude[idx] = np.where(ok, amp, 0.0)
            tau_ns[idx] = np.where(ok, dist / SPEED_OF_LIGHT * 1e9, np.nan)
            tx_phi, tx_theta = _angles_deg(tx_p, tx_target)
            rx_phi, rx_theta = _angles_deg(pts, rx_target)
            for a, values in enumerate((tx_phi, tx_theta, rx_phi, rx_theta)):
                angles[a, idx] = np.where(ok, values, np.nan)
            valid[idx] = ok

        # --- 1. Direct LoS Ray ---
        d_direct = np.linalg.norm(pts - tx_p, axis=-1)
        store(0, d_direct > 0.0, d_direct, 1.0, pts, tx_p)

        # --- 2. Ground Reflected Ray (image method, ground at z = 0) ---
        rz = pts[..., 2]
        tx_image_ground = np.array([tx.x, tx.y, -tx.z], dtype=float)
        rx_image_ground = pts * np.array([1.0, 1.0, -1.0])
        d_ground = np.linalg.norm(pts - tx_image_ground, axis=-1)
        ok_ground = (d_ground > 0.0) & (rz >= 0.0) & ((tx.z + rz) > 0.0) & (tx.z >= 0.0)
        horiz = np.hypot(pts[..., 0] - tx.x, pts[..., 1] - tx.y)
        grazing = np.arctan2(tx.z + rz, np.maximum(1e-6, horiz))
        gamma_ground = self._fresnel_ground_reflection(grazing, tx.frequency_hz)
        store(1, ok_ground, d_ground, gamma_ground, rx_image_ground, tx_image_ground)

        # --- 3. Wall Specular Reflections ---
        for r, refl in enumerate(self.reflectors):
            s_tx = float(np.dot(tx_p - refl.p0, refl.normal))
            s_rx = (pts - refl.p0) @ refl.normal
            ok_wall = (s_tx * s_rx > 0.0) & (np.abs(s_tx + s_rx) >= 1e-6)

            tx_image_wall = tx_p - 2.0 * s_tx * refl.normal
            d_wall = np.linalg.norm(pts - tx_image_wall, axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_intersect = s_tx / (s_tx + s_rx)
                p_wall_hit = tx_image_wall + t_intersect[..., None] * (pts - tx_image_wall)
                sin_grazing = np.abs(s_tx + s_rx) / d_wall
            p_wall_hit = np.where(ok_wall[..., None], p_wall_hit, 0.0)
            ok_wall &= refl.contains(p_wall_hit)
            gamma_wall = refl.reflection_coefficient(np.where(ok_wall, sin_grazing, 0.0), tx.frequency_hz)
            gamma_wall = np.asarray(gamma_wall)
            store(2 + r, ok_wall, d_wall, gamma_wall, p_wall_hit, p_wall_hit)

        return RayField(
            tx_id=tx.tx_id,
            amplitude=amplitude,
            tau_ns=tau_ns,
            tx_phi_deg=angles[0],
            tx_theta_deg=angles[1],
            rx_phi_deg=angles[2],
            rx_theta_deg=angles[3],
            valid=valid,
        )
