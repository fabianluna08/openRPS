"""
OpenRPS - 3D Ray-Tracing Engine
Computes multipath propagation rays (Direct, Ground Reflected, and Wall Reflected)
with the image method.

Two equivalent entry points are provided:
  * ``RayTracer.trace_paths``  -- one receiver point, returns a ChannelImpulseResponse.
  * ``RayTracer.trace_points`` -- many receiver points at once (numpy), returns a RayField.

Physical scope (see README, "Limitaciones fisicas"): single-bounce specular
reflections only, no diffraction.  Walls do not block rays unless they are given a
``transmission_loss_db`` (see ``PlanarReflector``): then every ray leg that crosses
the wall is attenuated by that loss (``inf`` = opaque wall, the ray carries no power).
Antennas are isotropic unless ``Transmitter(pattern="sector")`` is used.  The tracer
returns one complex amplitude per ray; whether they are added incoherently (default)
or coherently is decided by whoever sums them (``RayField.total_power_linear``,
``ChannelImpulseResponse.total_received_power_linear``, the grid simulation).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .models import SPEED_OF_LIGHT, fresnel_reflection_coefficient
from .multipath import MultipathRay, ChannelImpulseResponse

# Obstruction by walls: an end of a ray leg closer than this to the plane of a wall
# is *on* the wall and the leg does not cross it (the leg is an open segment).
OBSTRUCTION_TOL_M = 1e-6
# Margin of the ``bounds`` box of a finite wall (the patch is closed: its edge belongs to it)
_BOUNDS_TOL_M = 1e-9
# A ray that bends exactly on the plane of a wall: fraction of the way from the bend point
# towards the chord Tx -> Rx at which the wall patch is probed (see ``_crosses_at_bend``)
_BEND_STEP = 1e-6


def _opposite_sides(s_a, s_b, tol: float) -> np.ndarray:
    """True where two signed distances are strictly on opposite sides of a plane, both farther than ``tol`` from it."""
    if np.ndim(s_a) == 0:
        if s_a > tol:
            return np.asarray(s_b < -tol)
        if s_a < -tol:
            return np.asarray(s_b > tol)
        return np.zeros(np.shape(s_b), dtype=bool)
    return (np.minimum(s_a, s_b) < -tol) & (np.maximum(s_a, s_b) > tol)


def _same_side(s_tx: float, s_rx, tol: float) -> np.ndarray:
    """
    True where the receiver(s) are strictly on the same side of a plane as the
    transmitter, both farther than ``tol`` from it (``s_tx`` float, ``s_rx`` float or array).
    """
    if s_tx > tol:
        return np.asarray(s_rx > tol)
    if s_tx < -tol:
        return np.asarray(s_rx < -tol)
    return np.zeros(np.shape(s_rx), dtype=bool)


def _in_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray, tol: float = _BOUNDS_TOL_M) -> np.ndarray:
    """
    True where the points (x, y) fall inside the simple polygon ``polygon`` ([N, 2]),
    its edges included (closed patch: a point within ``tol`` of an edge is inside).
    Even-odd rule for the interior.  Same arithmetic as ``_in_polygon_scalar``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    on_edge = np.zeros_like(inside)
    n = len(polygon)
    for i in range(n):
        x1, y1 = float(polygon[i][0]), float(polygon[i][1])
        x2, y2 = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        straddles = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddles & (x < x_cross)
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / length2, 0.0, 1.0) if length2 > 0.0 else 0.0
        on_edge |= (x - (x1 + t * dx)) ** 2 + (y - (y1 + t * dy)) ** 2 <= tol * tol
    return inside | on_edge


def _in_polygon_scalar(x: float, y: float, polygon, tol: float = _BOUNDS_TOL_M) -> bool:
    """Scalar reference of ``_in_polygon`` (same rules, same arithmetic)."""
    inside = False
    on_edge = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i][0], polygon[i][1]
        x2, y2 = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy
        t = min(1.0, max(0.0, ((x - x1) * dx + (y - y1) * dy) / length2)) if length2 > 0.0 else 0.0
        if (x - (x1 + t * dx)) ** 2 + (y - (y1 + t * dy)) ** 2 <= tol * tol:
            on_edge = True
    return inside or on_edge


def _take(values, rows: np.ndarray, axis: Optional[int] = None):
    """
    Entries ``rows`` (flat indices over the points) of a per-point quantity, or the
    quantity itself if it is shared by all the points.  ``axis`` = None for scalars
    per point, 0..2 for that coordinate of points of shape [3] or [..., 3].
    """
    values = np.asarray(values)
    if axis is None:
        return values if values.ndim == 0 else values.reshape(-1)[rows]
    return values[axis] if values.ndim == 1 else values.reshape(-1, 3)[rows, axis]


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

    transmission_loss_db: penetration loss of the wall in dB (new in 1.3.0).
        ``None`` (default): the wall blocks nothing, every ray passes through it
        unattenuated (behaviour of OpenRPS <= 1.2).  A finite value L > 0: the power
        of every ray is reduced by L dB each time one of its legs crosses the wall
        (amplitude x 10^(-L/20), no phase change).  ``float("inf")``: opaque wall,
        a ray that crosses it carries no power.  0 dB is equivalent to ``None``.
        What is tested is each straight leg of each ray (direct: Tx->Rx; ground:
        Tx->bounce point and bounce point->Rx; wall: Tx->reflection point and
        reflection point->Rx) against the wall patch (``bounds``, or the whole
        plane); see ``crosses`` for the exact rule.  The losses of all the walls
        crossed along a ray add up in dB.  A wall never obstructs its own reflected
        ray.  If a ray bounces exactly on the plane of another wall (at its foot, or
        on the corner of a building), that wall is crossed once when it stands
        between the two legs of the ray (Tx and Rx on opposite sides of it).
        The loss does not depend on ``reflectivity`` / ``eps_r`` (no energy balance
        between the reflected and the transmitted wave is enforced), on the angle
        of incidence or on the frequency, and the wall has no thickness.

    footprint: optional polygon [(x0, y0), (x1, y1), ...] (new in 1.4.0) that limits a
        *horizontal* patch (normal along z), e.g. the roof of a building whose
        footprint is not an axis-aligned rectangle.  A point belongs to the patch if
        it is inside the polygon, edges included (same closed-patch rule and 1e-9 m
        margin as ``bounds``).  The polygon must be simple (no self-intersections);
        it is closed automatically.  Without ``bounds`` the box is set to the
        bounding box of the polygon at the height of the plane.  It acts everywhere
        a patch is tested: reflection point (``contains``) and obstruction
        (``crosses``), in both implementations of the tracer.
    """
    p0: np.ndarray      # Reference point on plane [x, y, z]
    normal: np.ndarray  # Normal vector of the plane [nx, ny, nz] (normalized on creation)
    reflectivity: float = 0.5  # Amplitude reflection coefficient (0.0 to 1.0)
    name: str = "Wall"
    bounds: Optional[Sequence[float]] = None  # (xmin, xmax, ymin, ymax, zmin, zmax)
    eps_r: Optional[float] = None             # Relative permittivity (enables Fresnel)
    sigma: float = 0.0                        # Conductivity [S/m], used with eps_r
    transmission_loss_db: Optional[float] = None  # Penetration loss [dB]; None = blocks nothing, inf = opaque
    footprint: Optional[Sequence[Sequence[float]]] = None  # Polygon [(x, y), ...] of a horizontal patch

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
        if self.transmission_loss_db is not None:
            try:
                loss = float(self.transmission_loss_db)
            except (TypeError, ValueError):
                loss = float("nan")
            if not loss >= 0.0:     # also rejects NaN
                raise ValueError("transmission_loss_db must be None, a loss >= 0 in dB or float('inf') "
                                 f"(opaque wall), got {self.transmission_loss_db!r}")
            self.transmission_loss_db = loss
        if self.footprint is not None:
            polygon = np.array(self.footprint, dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.all(np.isfinite(polygon)):
                raise ValueError("footprint must be a list of at least 3 finite (x, y) vertices")
            if len(polygon) > 3 and np.array_equal(polygon[0], polygon[-1]):
                polygon = polygon[:-1]          # explicitly closed polygon: drop the repeated vertex
            if self.normal[0] != 0.0 or self.normal[1] != 0.0:
                raise ValueError("footprint is only supported on horizontal planes (normal along z)")
            area2 = float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1]))
            if area2 == 0.0:
                raise ValueError("footprint polygon has zero area")
            self.footprint = polygon
            if self.bounds is None:
                z = float(self.p0[2])
                self.bounds = (float(polygon[:, 0].min()), float(polygon[:, 0].max()),
                               float(polygon[:, 1].min()), float(polygon[:, 1].max()), z, z)

    @property
    def obstructs(self) -> bool:
        """True if rays crossing the wall are attenuated (``transmission_loss_db`` > 0)."""
        return self.transmission_loss_db is not None and self.transmission_loss_db > 0.0

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance in meters from ``points`` (shape [..., 3]) to the plane (> 0 on the side of ``normal``)."""
        points = np.asarray(points, dtype=float)
        # (x - x0) * nx + (y - y0) * ny + (z - z0) * nz, skipping the null components of the
        # normal (they add exactly 0: same value, and most walls are vertical or axis-aligned)
        total = None
        for axis in range(3):
            if self.normal[axis] != 0.0:
                term = (points[..., axis] - self.p0[axis]) * self.normal[axis]
                total = term if total is None else total + term
        return total

    def crosses(self, p_start: np.ndarray, p_end: np.ndarray, tol: float = OBSTRUCTION_TOL_M) -> np.ndarray:
        """
        True where the straight segment ``p_start`` -> ``p_end`` crosses the wall
        (inputs of shape [..., 3], broadcast against each other).  Purely geometric:
        it does not depend on ``transmission_loss_db``.

        Convention (the one used for the obstruction of rays):
          * The segment is *open*: it crosses only if its two ends are strictly on
            opposite sides of the plane, each farther than ``tol`` (1 micrometer) from
            it.  A segment that starts or ends on the plane (an antenna mounted on the
            wall, the reflection point of the wall itself or of a coplanar wall), that
            lies in the plane or that is parallel to it does not cross.
          * The wall patch is *closed*: with ``bounds``, the crossing point must fall
            inside the box, edges included (same 1e-9 m margin as the reflection
            point).  A segment through the common edge of two walls crosses both.
        """
        a, b = np.broadcast_arrays(np.asarray(p_start, dtype=float), np.asarray(p_end, dtype=float))
        if a.shape[-1] != 3:
            raise ValueError("p_start and p_end must have shape [..., 3]")
        return self._crosses(self.signed_distance(a), self.signed_distance(b), a, b, tol)

    def _crosses(self, s_a, s_b, a: np.ndarray, b: np.ndarray, tol: float = OBSTRUCTION_TOL_M) -> np.ndarray:
        """
        ``crosses`` with the signed distances of the ends already computed.  ``s_a`` /
        ``s_b`` are floats or arrays of the points shape; ``a`` / ``b`` have shape
        [3] or [*shape, 3].  The crossing point is only computed for finite walls
        and only where the ends are on opposite sides.
        """
        crossed = _opposite_sides(s_a, s_b, tol)
        if self.bounds is None or not crossed.any():
            return crossed
        return self._keep_inside(crossed, s_a, s_b, a, b)

    def _crosses_at_bend(self, s_before, s_bend, s_after, before: np.ndarray, bend: np.ndarray, after: np.ndarray,
                         tol: float = OBSTRUCTION_TOL_M) -> np.ndarray:
        """
        Crossing of a two-leg path ``before`` -> ``bend`` -> ``after`` exactly at its
        bend point (the point where a ray is reflected), which neither leg can see
        because both end there.  True where the bend lies on the plane (within
        ``tol``), the other ends of the two legs are strictly on opposite sides, and
        the wall stands between the two legs: the point probed against the patch is
        the bend point moved a fraction ``_BEND_STEP`` towards the point where the
        straight chord ``before`` -> ``after`` meets the plane.  So a ray that bounces
        at the foot of a wall, or on the corner of a building towards its inside, is
        obstructed; one that only grazes the end of a wall from outside is not.
        """
        crossed = np.abs(s_bend) <= tol
        if crossed.any():
            crossed = crossed & _opposite_sides(s_before, s_after, tol)
            if self.bounds is not None and crossed.any():
                crossed = self._keep_inside(crossed, s_before, s_after, before, after, bend)
        return crossed

    def _keep_inside(self, crossed: np.ndarray, s_a, s_b, a: np.ndarray, b: np.ndarray,
                     bend: Optional[np.ndarray] = None) -> np.ndarray:
        """
        The entries of ``crossed`` whose crossing point x = a + t * (b - a), with
        t = s_a / (s_a - s_b), falls inside ``bounds`` (same test as ``contains``).
        With ``bend`` the point tested is bend + _BEND_STEP * (x - bend) instead.
        Evaluated one axis at a time and only for the candidates that remain.
        """
        candidates = np.flatnonzero(crossed)
        s_a, s_b = _take(s_a, candidates), _take(s_b, candidates)
        t = np.atleast_1d(s_a / (s_a - s_b))               # 0 < t < 1; the denominator is > 2 * tol

        def crossing_point(axis: int) -> np.ndarray:
            """Coordinate ``axis`` of the point tested, for the remaining candidates."""
            a_axis, b_axis = _take(a, candidates, axis), _take(b, candidates, axis)
            point = a_axis + t * (b_axis - a_axis)
            if bend is not None:
                bend_axis = _take(bend, candidates, axis)
                point = bend_axis + _BEND_STEP * (point - bend_axis)
            return point

        for axis in range(3):
            lo, hi = self.bounds[2 * axis] - _BOUNDS_TOL_M, self.bounds[2 * axis + 1] + _BOUNDS_TOL_M
            if lo == -np.inf and hi == np.inf:
                continue
            point = crossing_point(axis)
            inside = (point >= lo) & (point <= hi)
            candidates, t = candidates[inside], t[inside]
        if self.footprint is not None and candidates.size:
            inside = _in_polygon(crossing_point(0), crossing_point(1), self.footprint)
            candidates = candidates[inside]
        result = np.zeros(crossed.size, dtype=bool)
        result[candidates] = True
        return result.reshape(crossed.shape)

    def contains(self, points: np.ndarray, tol: float = _BOUNDS_TOL_M) -> np.ndarray:
        """
        True where ``points`` (shape [..., 3]) fall inside ``bounds`` and, if there is
        one, inside ``footprint`` (always True if unbounded).
        """
        points = np.asarray(points, dtype=float)
        if self.bounds is None:
            return np.ones(points.shape[:-1], dtype=bool)
        b = self.bounds
        inside = np.ones(points.shape[:-1], dtype=bool)
        for axis in range(3):
            inside &= (points[..., axis] >= b[2 * axis] - tol) & (points[..., axis] <= b[2 * axis + 1] + tol)
        if self.footprint is not None and inside.any():
            inside &= _in_polygon(points[..., 0], points[..., 1], self.footprint, tol)
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


ANTENNA_PATTERNS = ("isotropic", "sector")
# Limits of the 3GPP parabolic sector pattern (TR 38.901, Table 7.3-1): A_m = SLA_V = A_max
SECTOR_MAX_ATTENUATION_DB = 30.0


@dataclass
class Transmitter:
    """
    Represents a Base Station / Transmitter.

    pattern: "isotropic" (default): ``antenna_gain_dbi`` is applied equally in every
        direction.  "sector": ``antenna_gain_dbi`` is the boresight (maximum) gain and
        the 3GPP parabolic pattern (TR 36.814 / TR 38.901) attenuates every other
        direction of departure:
            A_H(phi)      = -min(12 * (phi / beamwidth_az_deg)^2, 30)
            A_V(theta)    = -min(12 * ((theta - tilt) / beamwidth_el_deg)^2, 30)
            A(phi, theta) = -min(-(A_H + A_V), 30)                       [dB]
        phi being the azimuth measured from ``azimuth_deg`` and theta - tilt the
        elevation measured from the (down-tilted) boresight.
    azimuth_deg: azimuth of the boresight, same convention as the ray angles
        (0 = +x axis, 90 = +y axis, counter-clockwise).
    beamwidth_az_deg, beamwidth_el_deg: 3 dB beamwidths (the gain is 3 dB down at
        half the beamwidth off boresight).  Defaults: 65 deg (TR 38.901 element).
    downtilt_deg: positive values point the beam below the horizon.
    The last four fields are ignored when ``pattern="isotropic"``.
    """
    tx_id: int
    x: float
    y: float
    z: float
    tx_power_dbm: float = 43.0  # Default 20 Watts (43 dBm)
    frequency_hz: float = 2.4e9 # Default 2.4 GHz
    antenna_gain_dbi: float = 14.0 # Applied in all directions (isotropic) or on boresight (sector)
    azimuth_deg: float = 0.0
    beamwidth_az_deg: float = 65.0
    beamwidth_el_deg: float = 65.0
    downtilt_deg: float = 0.0
    pattern: str = "isotropic"

    def __post_init__(self):
        if not (np.isfinite(self.frequency_hz) and self.frequency_hz > 0.0):
            raise ValueError(f"frequency_hz must be a positive finite number, got {self.frequency_hz!r}")
        if not isinstance(self.pattern, str) or self.pattern.lower() not in ANTENNA_PATTERNS:
            raise ValueError(f"pattern must be one of {ANTENNA_PATTERNS}, got {self.pattern!r}")
        self.pattern = self.pattern.lower()
        for name in ("beamwidth_az_deg", "beamwidth_el_deg"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a positive finite angle in degrees, got {value!r}")
        for name in ("azimuth_deg", "downtilt_deg"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite angle in degrees, got {getattr(self, name)!r}")

    @property
    def power_mw(self) -> float:
        return 10.0 ** (self.tx_power_dbm / 10.0)

    @property
    def is_isotropic(self) -> bool:
        return self.pattern == "isotropic"

    def pattern_gain_db(self, phi_deg, theta_deg):
        """
        Gain of the radiation pattern relative to boresight, in dB (always <= 0), for
        a direction of departure given by its azimuth ``phi_deg`` (-180..180) and
        elevation ``theta_deg`` (-90..90, positive above the horizon), i.e. the
        ``tx_phi_deg`` / ``tx_theta_deg`` of a ray.  Floats or numpy arrays.
        Isotropic antennas return 0 dB everywhere.
        """
        scalar = np.ndim(phi_deg) == 0 and np.ndim(theta_deg) == 0
        phi = np.asarray(phi_deg, dtype=float)
        theta = np.asarray(theta_deg, dtype=float)
        if self.is_isotropic:
            gain = np.zeros(np.broadcast(phi, theta).shape)
        else:
            limit = SECTOR_MAX_ATTENUATION_DB
            d_phi = (phi - self.azimuth_deg + 180.0) % 360.0 - 180.0     # wrapped into [-180, 180)
            d_theta = theta + self.downtilt_deg                          # elevation off the tilted boresight
            a_h = np.minimum(12.0 * (d_phi / self.beamwidth_az_deg) ** 2, limit)
            a_v = np.minimum(12.0 * (d_theta / self.beamwidth_el_deg) ** 2, limit)
            gain = 0.0 - np.minimum(a_h + a_v, limit)                    # (0.0 - x: boresight gives 0.0, not -0.0)
        return float(gain) if scalar else gain

    def gain_dbi(self, phi_deg, theta_deg):
        """Antenna gain in dBi towards a direction of departure: antenna_gain_dbi + pattern_gain_db."""
        return self.antenna_gain_dbi + self.pattern_gain_db(phi_deg, theta_deg)


@dataclass
class RayField:
    """
    Rays from one transmitter to many receiver points (output of ``RayTracer.trace_points``).

    Every array has shape [K, *points_shape], K being the number of candidate rays:
    index 0 = direct, 1 = ground reflection, 2.. = one per reflector (in the order
    they were added).  Where a ray does not exist ``valid`` is False, its amplitude
    is 0 and its delay/angles are NaN.

    Obstruction (walls with ``transmission_loss_db``) never changes which rays exist:
    a ray attenuated or blocked by walls keeps ``valid`` = True, its delay and its
    angles; only its amplitude is scaled (to exactly 0 behind an opaque wall).
    ``obstruction_loss_db`` gives the loss applied to each ray: 0 where it crosses
    no obstructing wall (or does not exist), ``inf`` where an opaque wall blocks it.
    """
    tx_id: int
    amplitude: np.ndarray     # complex; |a|^2 is the ray power in mW
    tau_ns: np.ndarray
    tx_phi_deg: np.ndarray
    tx_theta_deg: np.ndarray
    rx_phi_deg: np.ndarray
    rx_theta_deg: np.ndarray
    valid: np.ndarray         # bool
    obstruction_loss_db: Optional[np.ndarray] = None   # dB lost through walls, already included in amplitude

    @property
    def power_linear(self) -> np.ndarray:
        """Power per ray in mW (0 where the ray does not exist)."""
        return np.abs(self.amplitude) ** 2

    def total_power_linear(self, coherent: bool = False) -> np.ndarray:
        """
        Total received power per point in mW (shape = points_shape).
        coherent=False (default): sum of the ray powers, sum_k |a_k|^2.
        coherent=True: power of the complex sum of the rays, |sum_k a_k|^2 (fast fading).
        """
        if coherent:
            return np.abs(self.amplitude.sum(axis=0)) ** 2
        return self.power_linear.sum(axis=0)


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
    Every ray amplitude includes the transmitter gain towards the direction in
    which that ray leaves the antenna (``Transmitter.gain_dbi`` of its AoD) and
    the loss through the walls that the ray crosses, for the walls that have a
    ``transmission_loss_db`` (none by default; see ``PlanarReflector``).
    """
    def __init__(self, ground_eps_r: float = 15.0, ground_sigma: float = 0.005):
        self.ground_eps_r = ground_eps_r
        self.ground_sigma = ground_sigma
        self.reflectors: List[PlanarReflector] = []

    def add_reflector(self, reflector: PlanarReflector):
        self.reflectors.append(reflector)

    def obstruction_loss_db(self, p_start, p_end) -> np.ndarray:
        """
        Total penetration loss in dB of the straight segment ``p_start`` -> ``p_end``
        (shape [..., 3], broadcast) through the walls that have a
        ``transmission_loss_db``: sum of the losses of the walls crossed (rule in
        ``PlanarReflector.crosses``), 0 with a clear path, ``inf`` if an opaque wall
        is crossed.  E.g. ``tracer.obstruction_loss_db(tx_xyz, rx_xyz) == 0`` tells
        whether there is line of sight.
        """
        a, b = np.broadcast_arrays(np.asarray(p_start, dtype=float), np.asarray(p_end, dtype=float))
        loss = np.zeros(a.shape[:-1])
        for refl in self.reflectors:
            if refl.obstructs:
                loss = loss + np.where(refl.crosses(a, b), refl.transmission_loss_db, 0.0)
        return loss

    @staticmethod
    def _attenuate(magnitude: complex, loss_db: float) -> complex:
        """Ray amplitude after ``loss_db`` dB of penetration loss: x 10^(-L/20), exactly 0 for an opaque wall."""
        factor = 10.0 ** (-loss_db / 20.0)
        return magnitude * factor if factor > 0.0 else 0j

    def _scalar_obstacles(self) -> list:
        """
        Obstructing walls as plain floats, for the point-by-point tracer:
        (index, loss, p0, normal, bounds, footprint as a list of [x, y] or None).
        """
        return [(index, refl.transmission_loss_db, refl.p0.tolist(), refl.normal.tolist(), refl.bounds,
                 None if refl.footprint is None else refl.footprint.tolist())
                for index, refl in enumerate(self.reflectors) if refl.obstructs]

    @staticmethod
    def _path_loss_db(obstacles: list, a: Sequence[float], b: Sequence[float],
                      via: Optional[Sequence[float]] = None, own: int = -1) -> float:
        """
        Scalar reference of the obstruction: loss in dB of the path a -> b, or
        a -> via -> b for a reflected ray, through the walls of ``_scalar_obstacles``
        (``own`` = index of the wall on which the ray is reflected, never an obstacle
        for it).  Same rules and same arithmetic as ``PlanarReflector.crosses``.
        """
        tol = OBSTRUCTION_TOL_M

        def crossings_of_leg(s_p, s_q, p, q, bounds, footprint) -> int:
            if not ((s_p > tol and s_q < -tol) or (s_p < -tol and s_q > tol)):
                return 0
            if bounds is not None:
                t = s_p / (s_p - s_q)
                if not inside([p[axis] + t * (q[axis] - p[axis]) for axis in range(3)], bounds, footprint):
                    return 0
            return 1

        def inside(point, bounds, footprint) -> bool:
            if not all(bounds[2 * axis] - _BOUNDS_TOL_M <= point[axis] <= bounds[2 * axis + 1] + _BOUNDS_TOL_M
                       for axis in range(3)):
                return False
            return footprint is None or _in_polygon_scalar(point[0], point[1], footprint)

        loss = 0.0
        for index, wall_loss, p0, n, bounds, footprint in obstacles:
            if index == own:
                continue
            s_a = (a[0] - p0[0]) * n[0] + (a[1] - p0[1]) * n[1] + (a[2] - p0[2]) * n[2]
            s_b = (b[0] - p0[0]) * n[0] + (b[1] - p0[1]) * n[1] + (b[2] - p0[2]) * n[2]
            if via is None:
                crossings = crossings_of_leg(s_a, s_b, a, b, bounds, footprint)
            else:
                s_via = (via[0] - p0[0]) * n[0] + (via[1] - p0[1]) * n[1] + (via[2] - p0[2]) * n[2]
                if abs(s_via) <= tol:
                    # The ray bends exactly on the plane of this wall: it goes through it if a and b are on
                    # opposite sides and the wall stands between the two legs (see _crosses_at_bend)
                    crossings = 0
                    if (s_a > tol and s_b < -tol) or (s_a < -tol and s_b > tol):
                        crossings = 1
                        if bounds is not None:
                            t = s_a / (s_a - s_b)
                            probe = [via[axis] + _BEND_STEP * ((a[axis] + t * (b[axis] - a[axis])) - via[axis])
                                     for axis in range(3)]
                            crossings = int(inside(probe, bounds, footprint))
                else:
                    crossings = (crossings_of_leg(s_a, s_via, a, via, bounds, footprint)
                                 + crossings_of_leg(s_via, s_b, via, b, bounds, footprint))
            if crossings:
                loss += crossings * wall_loss
        return loss

    def _calc_angles(self, p_src: np.ndarray, p_dst: np.ndarray) -> Tuple[float, float]:
        """Calculates azimuth (phi: -180..180) and elevation (theta: -90..90) from src to dst."""
        v = p_dst - p_src
        dist_2d = math.hypot(v[0], v[1])
        phi = math.degrees(math.atan2(v[1], v[0]))
        theta = math.degrees(math.atan2(v[2], max(1e-6, dist_2d)))
        return phi, theta

    @staticmethod
    def _pattern_factor(tx: Transmitter, phi_deg, theta_deg):
        """Amplitude factor (<= 1) of the transmitter pattern towards a direction of departure."""
        return 10.0 ** (tx.pattern_gain_db(phi_deg, theta_deg) / 20.0)

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

        The complex amplitude of each ray carries its phase (-k*d plus the phase of
        the reflection coefficient), so the returned CIR can be summed incoherently
        or coherently (``cir.total_received_power_dbm(coherent=True)``).

        Walls with a ``transmission_loss_db`` attenuate every ray whose legs cross
        them (amplitude x 10^(-L/20), phase untouched).  The list of rays does not
        change: a ray blocked by an opaque wall stays in the CIR with magnitude 0
        (and ``obstruction_loss_db`` = inf), and every metric of the CIR ignores it.
        """
        cir = ChannelImpulseResponse()
        tx_p = np.array([tx.x, tx.y, tx.z], dtype=float)
        rx_p = np.array(rx_pos, dtype=float)
        # Walls that attenuate the rays crossing them (none unless transmission_loss_db is used)
        obstacles = self._scalar_obstacles()
        tx_xyz, rx_xyz = (tx_p.tolist(), rx_p.tolist()) if obstacles else (None, None)
        wavelength = SPEED_OF_LIGHT / tx.frequency_hz
        k = 2.0 * math.pi / wavelength
        # Field amplitude at 1 m scaled so that |a|^2 is power in mW (Friis):
        # |a| = sqrt(P_t * G_t) * lambda / (4*pi*d)
        amp_1m = math.sqrt(tx.power_mw * (10**(tx.antenna_gain_dbi/10.0))) * wavelength / (4.0 * math.pi)
        # Sector antennas: every ray is also scaled by the pattern towards its own AoD
        sector = not tx.is_isotropic

        # --- 1. Direct LoS Ray ---
        d_direct = float(np.linalg.norm(rx_p - tx_p))
        if d_direct > 0:
            tau_direct = (d_direct / SPEED_OF_LIGHT) * 1e9 # ns
            amp_direct = amp_1m / d_direct
            phase_direct = -k * d_direct
            mag_direct = amp_direct * complex(math.cos(phase_direct), math.sin(phase_direct))

            tx_phi, tx_theta = self._calc_angles(tx_p, rx_p)
            rx_phi, rx_theta = self._calc_angles(rx_p, tx_p)
            if sector:
                mag_direct *= self._pattern_factor(tx, tx_phi, tx_theta)
            loss_direct = self._path_loss_db(obstacles, tx_xyz, rx_xyz) if obstacles else 0.0
            if loss_direct > 0.0:
                mag_direct = self._attenuate(mag_direct, loss_direct)

            cir.add_ray(MultipathRay(
                tx_id=tx.tx_id,
                magnitude=mag_direct,
                tau_ns=tau_direct,
                tx_phi_deg=tx_phi,
                tx_theta_deg=tx_theta,
                rx_phi_deg=rx_phi,
                rx_theta_deg=rx_theta,
                obstruction_loss_db=loss_direct,
                kind="direct",
                path=(tuple(tx_p.tolist()), tuple(rx_p.tolist())),
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
            if sector:
                mag_ground *= self._pattern_factor(tx, tx_phi_g, tx_theta_g)
            loss_ground = 0.0
            # Bounce point of the ray: Tx -> bounce point -> Rx
            t_bounce = tx.z / (tx.z + rx_p[2])
            bounce_xyz = (tx_image_ground + t_bounce * (rx_p - tx_image_ground)).tolist()
            if obstacles:
                # Both legs of the ray
                loss_ground = self._path_loss_db(obstacles, tx_xyz, rx_xyz, via=bounce_xyz)
                if loss_ground > 0.0:
                    mag_ground = self._attenuate(mag_ground, loss_ground)

            cir.add_ray(MultipathRay(
                tx_id=tx.tx_id,
                magnitude=mag_ground,
                tau_ns=tau_ground,
                tx_phi_deg=tx_phi_g,
                tx_theta_deg=tx_theta_g,
                rx_phi_deg=rx_phi_g,
                rx_theta_deg=rx_theta_g,
                obstruction_loss_db=loss_ground,
                kind="ground",
                path=(tuple(tx_p.tolist()), tuple(bounce_xyz), tuple(rx_p.tolist())),
            ))

        # --- 3. Wall Specular Reflections ---
        for wall_index, refl in enumerate(self.reflectors):
            # Signed distances of Tx and Rx to the plane
            s_tx = float(np.dot(tx_p - refl.p0, refl.normal))
            s_rx = float(np.dot(rx_p - refl.p0, refl.normal))

            # A specular reflection needs Tx and Rx strictly on the same side of the wall, each one
            # farther than OBSTRUCTION_TOL_M from the plane (a point on the plane has no reflection)
            if not _same_side(s_tx, s_rx, OBSTRUCTION_TOL_M):
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
            if sector:
                mag_wall *= self._pattern_factor(tx, tx_phi_w, tx_theta_w)
            loss_wall = 0.0
            if obstacles:
                # Both legs, Tx -> reflection point -> Rx, against every *other* wall
                hit_xyz = p_wall_hit.tolist()
                loss_wall = self._path_loss_db(obstacles, tx_xyz, rx_xyz, via=hit_xyz, own=wall_index)
                if loss_wall > 0.0:
                    mag_wall = self._attenuate(mag_wall, loss_wall)

            cir.add_ray(MultipathRay(
                tx_id=tx.tx_id,
                magnitude=mag_wall,
                tau_ns=tau_wall,
                tx_phi_deg=tx_phi_w,
                tx_theta_deg=tx_theta_w,
                rx_phi_deg=rx_phi_w,
                rx_theta_deg=rx_theta_w,
                obstruction_loss_db=loss_wall,
                kind="wall",
                path=(tuple(tx_p.tolist()), tuple(p_wall_hit.tolist()), tuple(rx_p.tolist())),
                reflector_index=wall_index,
            ))

        return cir

    def trace_points(self, tx: Transmitter, points: np.ndarray) -> RayField:
        """
        Vectorized equivalent of ``trace_paths`` for an array of receiver points.

        points: array of shape [..., 3] with the receiver coordinates (x, y, z).
        Returns a RayField whose arrays have shape [2 + n_reflectors, ...].
        Obstruction by walls with a ``transmission_loss_db`` is applied exactly as in
        ``trace_paths``; ``RayField.obstruction_loss_db`` holds the loss of each ray.
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
        sector = not tx.is_isotropic

        # Walls that attenuate the rays crossing them, with the signed distances of Tx and
        # of every receiver point to their planes (empty unless transmission_loss_db is used)
        obstacles = [(index, refl, float(refl.signed_distance(tx_p)), refl.signed_distance(pts))
                     for index, refl in enumerate(self.reflectors) if refl.obstructs]
        loss_db = np.zeros((n_rays,) + shape)

        def obstruction(ok, via, own):
            """
            Loss in dB of the path Tx -> ``via`` -> points (Tx -> points if ``via`` is
            None) through the obstructing walls, except the wall number ``own`` (the
            one the ray is reflected on).  0 where the ray does not exist.
            """
            loss = np.zeros(shape)
            for index, wall, s_tx, s_rx in obstacles:
                if index == own:
                    continue
                if via is None:
                    crossings = (wall._crosses(s_tx, s_rx, tx_p, pts),)
                else:
                    # The two legs, plus the case of a ray that bends exactly on the plane of this wall
                    s_via = wall.signed_distance(via)
                    crossings = (wall._crosses(s_tx, s_via, tx_p, via), wall._crosses(s_via, s_rx, via, pts),
                                 wall._crosses_at_bend(s_tx, s_via, s_rx, tx_p, via, pts))
                for crossed in crossings:
                    np.add(loss, wall.transmission_loss_db, out=loss, where=crossed)
            return np.where(ok, loss, 0.0)

        def store(idx, ok, dist, coeff, tx_target, rx_target, via=None, own=-1):
            """
            Fill ray ``idx``: path length ``dist``, reflection coefficient ``coeff``, aim
            points; ``via`` is the point where the ray bounces (None: straight ray).
            """
            tx_phi, tx_theta = _angles_deg(tx_p, tx_target)
            rx_phi, rx_theta = _angles_deg(pts, rx_target)
            with np.errstate(divide="ignore", invalid="ignore"):
                amp = (amp_1m / dist) * coeff * np.exp(-1j * k * dist)
                if sector:
                    amp = amp * self._pattern_factor(tx, tx_phi, tx_theta)
                if obstacles:
                    loss_db[idx] = loss = obstruction(ok, via, own)
                    if loss.any():
                        amp = amp * 10.0 ** (-loss / 20.0)          # x1 (exact) where nothing is crossed
            amplitude[idx] = np.where(ok, amp, 0.0)
            tau_ns[idx] = np.where(ok, dist / SPEED_OF_LIGHT * 1e9, np.nan)
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
        p_bounce = None
        if obstacles:
            # Bounce point, as in trace_paths: p = tx_image + t * (rx - tx_image), t = tx.z / (tx.z + rx.z)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_bounce = tx.z / (tx.z + rz)
                p_bounce = tx_image_ground + t_bounce[..., None] * (pts - tx_image_ground)
            p_bounce = np.where(ok_ground[..., None], p_bounce, 0.0)
        store(1, ok_ground, d_ground, gamma_ground, rx_image_ground, tx_image_ground, via=p_bounce)

        # --- 3. Wall Specular Reflections ---
        for r, refl in enumerate(self.reflectors):
            s_tx = float(np.dot(tx_p - refl.p0, refl.normal))
            s_rx = (pts - refl.p0) @ refl.normal
            ok_wall = _same_side(s_tx, s_rx, OBSTRUCTION_TOL_M)

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
            store(2 + r, ok_wall, d_wall, gamma_wall, p_wall_hit, p_wall_hit, via=p_wall_hit, own=r)

        return RayField(
            tx_id=tx.tx_id,
            amplitude=amplitude,
            tau_ns=tau_ns,
            tx_phi_deg=angles[0],
            tx_theta_deg=angles[1],
            rx_phi_deg=angles[2],
            rx_theta_deg=angles[3],
            valid=valid,
            obstruction_loss_db=loss_db,
        )
