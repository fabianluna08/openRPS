"""
OpenRPS - 3D Ray-Tracing Engine
Computes multipath propagation rays (Direct, Ground Reflected, and Wall Reflected)
with the image method.

Two equivalent entry points are provided:
  * ``RayTracer.trace_paths``  -- one receiver point, returns a ChannelImpulseResponse.
  * ``RayTracer.trace_points`` -- many receiver points at once (numpy), returns a RayField.

Physical scope (see README, "Limitaciones fisicas"): direct ray, ground reflection
and one specular reflection per wall; optionally (new in 1.5.0) one diffraction per
edge (``DiffractionEdge``, UTD) and every double reflection wall-wall / wall-ground
(``RayTracer(max_reflections=2)``); new in 1.6.0, rays diffracted by two edges in a
row (``max_diffractions=2``) and rays that combine one reflection with one
diffraction (``mixed_interactions=True``).  Walls do not block rays unless they are given a
``transmission_loss_db`` (see ``PlanarReflector``): then every ray leg that crosses
the wall is attenuated by that loss (``inf`` = opaque wall, the ray carries no power).
Antennas are isotropic unless ``Transmitter(pattern="sector")`` is used.  The tracer
returns one complex amplitude per ray; whether they are added incoherently (default)
or coherently is decided by whoever sums them (``RayField.total_power_linear``,
``ChannelImpulseResponse.total_received_power_linear``, the grid simulation).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .diffraction import utd_coefficient
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


EDGE_POLARIZATIONS = ("soft", "hard")
# Angular tolerance (rad) of the exterior region of a wedge
_ANGLE_TOL = 1e-12


@dataclass
class DiffractionEdge:
    """
    A straight edge of a wedge that diffracts rays (new in 1.5.0), with the
    coefficient of the Uniform Theory of Diffraction (``diffraction.utd_coefficient``).

    Geometry: the edge is the segment ``start`` -> ``end``.  ``face_direction`` (t0)
    points from the edge into face 0 of the wedge and ``face_normal`` (n0) is the
    normal of face 0 on the side of the exterior (free space); both perpendicular
    to the edge (components along it are removed on creation).  Angles around the
    edge are measured from face 0 towards n0: the exterior of the wedge is
    0 <= phi <= ``wedge_n`` * pi, and face n is face 0 rotated by that angle.
    ``wedge_n`` = 2 is a half plane (a thin screen: the top or the end of a wall),
    1.5 a right-angled corner (a building corner or a roof edge); it must lie in (1, 2].

    polarization: "soft" (E parallel to the edge: vertical polarization on a vertical
        edge) or "hard" (E perpendicular to the edge: vertical polarization on a
        horizontal edge such as a roof edge).
    reflectivity / eps_r / sigma: the faces, as in ``PlanarReflector``.  Constant
        reflectivity rho gives the Luebbers face coefficients -rho (soft) or +rho
        (hard): rho = 1 is a perfectly conducting wedge.  With ``eps_r`` the Fresnel
        coefficients are used (TE for soft, TM for hard), at the grazing angles phi'
        on face 0 and n*pi - phi on face n.
    shadow_loss_db: penetration loss of the incident field just inside the shadow
        of the edge (``inf``, default: an opaque wedge, the standard UTD).  A finite
        value L scales the two incident-shadow-boundary terms by 1 - 10^(-L/20), so
        that the total field (transmitted + diffracted) stays continuous at the
        shadow boundary of a wedge that lets part of the field through; 0 removes
        them (a wedge that does not obstruct only diffracts its reflections).

    A diffracted ray exists where both Tx and Rx are in the exterior of the wedge
    (and off the edge line) and the diffraction point given by Keller's law (equal
    angles with the edge) falls on the segment, ends included.
    """
    start: np.ndarray
    end: np.ndarray
    face_direction: np.ndarray
    face_normal: np.ndarray
    wedge_n: float = 2.0
    polarization: str = "soft"
    reflectivity: float = 1.0
    eps_r: Optional[float] = None
    sigma: float = 0.0
    shadow_loss_db: Optional[float] = float("inf")
    name: str = "Edge"

    def __post_init__(self):
        self.start = np.array(self.start, dtype=float)
        self.end = np.array(self.end, dtype=float)
        if self.start.shape != (3,) or self.end.shape != (3,):
            raise ValueError("start and end must be 3-element vectors [x, y, z]")
        axis = self.end - self.start
        length = float(np.linalg.norm(axis))
        if not (np.isfinite(length) and length > 0.0):
            raise ValueError("an edge needs two distinct, finite end points")
        self.length = length
        self.axis = axis / length
        t0 = np.array(self.face_direction, dtype=float)
        n0 = np.array(self.face_normal, dtype=float)
        if t0.shape != (3,) or n0.shape != (3,):
            raise ValueError("face_direction and face_normal must be 3-element vectors")
        t0 = t0 - np.dot(t0, self.axis) * self.axis
        if not np.linalg.norm(t0) > 1e-9:
            raise ValueError("face_direction must not be parallel to the edge")
        t0 = t0 / np.linalg.norm(t0)
        n0 = n0 - np.dot(n0, self.axis) * self.axis - np.dot(n0, t0) * t0
        if not np.linalg.norm(n0) > 1e-9:
            raise ValueError("face_normal must not lie in the plane of face 0")
        self.face_direction = t0
        self.face_normal = n0 / np.linalg.norm(n0)
        n = float(self.wedge_n)
        if not (np.isfinite(n) and 1.0 < n <= 2.0):
            raise ValueError(f"wedge_n (exterior angle / pi) must be in (1, 2], got {self.wedge_n!r}")
        self.wedge_n = n
        if self.polarization not in EDGE_POLARIZATIONS:
            raise ValueError(f"polarization must be one of {EDGE_POLARIZATIONS}, got {self.polarization!r}")
        if not (0.0 <= self.reflectivity <= 1.0):
            raise ValueError(f"reflectivity must be within [0, 1], got {self.reflectivity!r}")
        if self.eps_r is not None and not self.eps_r >= 1.0:
            raise ValueError(f"eps_r must be >= 1, got {self.eps_r!r}")
        loss = 0.0 if self.shadow_loss_db is None else float(self.shadow_loss_db)
        if not loss >= 0.0:
            raise ValueError(f"shadow_loss_db must be >= 0 dB or inf, got {self.shadow_loss_db!r}")
        self.shadow_loss_db = loss

    @property
    def exterior_angle(self) -> float:
        """Exterior angle of the wedge in radians (n * pi)."""
        return self.wedge_n * math.pi

    @property
    def isb_scale(self) -> float:
        """Factor of the incident-shadow-boundary terms: 1 - (amplitude let through the wedge)."""
        return 1.0 - 10.0 ** (-self.shadow_loss_db / 20.0)

    def cylindrical(self, points):
        """
        Coordinates of ``points`` ([..., 3]) around the edge: (s, r, phi) with s the
        position along the edge measured from ``start``, r the distance to the edge
        line and phi the angle from face 0 (0..2 pi; tiny negative angles, i.e. points
        on face 0 up to rounding, give 0).
        """
        v = np.asarray(points, dtype=float) - self.start
        s = v @ self.axis
        perp = v - s[..., None] * self.axis if np.ndim(s) else v - s * self.axis
        r = np.linalg.norm(perp, axis=-1)
        phi = np.arctan2(perp @ self.face_normal, perp @ self.face_direction)
        phi = np.where(phi < -_ANGLE_TOL, phi + 2.0 * math.pi, np.maximum(phi, 0.0))
        return s, r, phi

    def face_coefficients(self, phi_prime, phi, frequency_hz: float):
        """Luebbers' reflection coefficients (r0, rn) of the faces: grazing angles phi' and n pi - phi."""
        if self.eps_r is None:
            rho = -self.reflectivity if self.polarization == "soft" else self.reflectivity
            shape = np.broadcast(np.asarray(phi_prime), np.asarray(phi)).shape
            return np.full(shape, rho, dtype=complex), np.full(shape, rho, dtype=complex)
        pol = "horizontal" if self.polarization == "soft" else "vertical"

        def grazing(angle):
            a = np.clip(np.asarray(angle, dtype=float), 0.0, math.pi)
            return np.minimum(a, math.pi - a)

        psi0 = grazing(phi_prime) + np.zeros(np.shape(phi))
        psin = grazing(self.exterior_angle - np.asarray(phi, dtype=float))
        return (np.asarray(fresnel_reflection_coefficient(psi0, frequency_hz, self.eps_r, self.sigma, pol)),
                np.asarray(fresnel_reflection_coefficient(psin, frequency_hz, self.eps_r, self.sigma, pol)))


@dataclass
class RayField:
    """
    Rays from one transmitter to many receiver points (output of ``RayTracer.trace_points``).

    Every array has shape [K, *points_shape], K being the number of candidate rays:
    index 0 = direct, 1 = ground reflection, 2.. = one per reflector (in the order
    they were added), then one diffracted ray per edge and, with
    ``max_reflections=2``, one per ordered pair of planes (see
    ``RayTracer.candidate_rays``).  Where a ray does not exist ``valid`` is False,
    its amplitude is 0 and its delay/angles are NaN.

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
    labels: Optional[list] = None   # (kind, interactions) of each row of the axis K (new in 1.6.0)

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


def _segment_distance(p1, q1, p2, q2) -> float:
    """Minimum distance between the segments p1-q1 and p2-q2 (3D)."""
    p1, q1, p2, q2 = (np.asarray(v, dtype=float) for v in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    denom = a * e - b * b
    s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom > 1e-12 * a * e else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, min(1.0, max(0.0, -c / a))
    elif t > 1.0:
        t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _plane_ref(index: int) -> Tuple[str, int]:
    """Interaction of a reflection on plane ``index`` (-1 = ground)."""
    return ("ground", -1) if index < 0 else ("wall", index)


class _Rays:
    """One candidate ray of ``RayTracer._extra_rays`` evaluated on a flat array of points."""
    __slots__ = ("kind", "interactions", "idx", "amplitude", "tau_ns", "angles", "loss_db", "vertices")

    def __init__(self, kind: str, interactions):
        self.kind = kind
        self.interactions = interactions
        self.idx = np.zeros(0, dtype=np.intp)          # points where the ray exists
        self.amplitude = self.tau_ns = self.loss_db = None
        self.angles = None                              # (tx_phi, tx_theta, rx_phi, rx_theta)
        self.vertices = None                            # interior vertices of the path, [m, 3] each

    def fill(self, idx, amplitude, length_m, angles, loss_db, vertices):
        self.idx = idx
        self.amplitude = amplitude
        self.tau_ns = length_m / SPEED_OF_LIGHT * 1e9
        self.angles = angles
        self.loss_db = loss_db
        self.vertices = vertices
        return self


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
    def __init__(self, ground_eps_r: float = 15.0, ground_sigma: float = 0.005, max_reflections: int = 1,
                 max_diffractions: int = 1, mixed_interactions: bool = False):
        self.ground_eps_r = ground_eps_r
        self.ground_sigma = ground_sigma
        self.reflectors: List[PlanarReflector] = []
        self.edges: List[DiffractionEdge] = []
        self.max_reflections = max_reflections
        self.max_diffractions = max_diffractions
        self.mixed_interactions = mixed_interactions
        self._visibility_cache = (None, None)
        self._active_cache = (None, None)

    @property
    def max_diffractions(self) -> int:
        """1 (default): one diffraction per ray.  2: also rays diffracted by two edges in a row (new in 1.6.0)."""
        return self._max_diffractions

    @max_diffractions.setter
    def max_diffractions(self, value: int):
        if isinstance(value, bool) or value not in (1, 2):
            raise ValueError(f"max_diffractions must be 1 or 2, got {value!r}")
        self._max_diffractions = int(value)

    @property
    def mixed_interactions(self) -> bool:
        """True: also rays reflected and then diffracted, or diffracted and then reflected (new in 1.6.0)."""
        return self._mixed_interactions

    @mixed_interactions.setter
    def mixed_interactions(self, value: bool):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"mixed_interactions must be True or False, got {value!r}")
        self._mixed_interactions = bool(value)

    @property
    def max_reflections(self) -> int:
        """1 (default): one specular reflection per ray.  2: also every double reflection (new in 1.5.0)."""
        return self._max_reflections

    @max_reflections.setter
    def max_reflections(self, value: int):
        if isinstance(value, bool) or value not in (1, 2):
            raise ValueError(f"max_reflections must be 1 or 2, got {value!r}")
        self._max_reflections = int(value)

    def add_reflector(self, reflector: PlanarReflector):
        self.reflectors.append(reflector)

    def add_edge(self, edge: DiffractionEdge):
        """Add an edge that diffracts rays (UTD; new in 1.5.0)."""
        self.edges.append(edge)

    def _plane_pairs(self) -> List[Tuple[int, int]]:
        """Ordered pairs (a, b), a != b, of the planes of a double reflection: -1 = ground, 0.. = reflectors."""
        if self.max_reflections < 2:
            return []
        planes = range(-1, len(self.reflectors))
        return [(a, b) for a in planes for b in planes if a != b]

    def _higher_order(self) -> List[Tuple[str, int, int]]:
        """
        Candidate rays with two interactions, in the order of ``candidate_rays``:
        ("double", a, b) double reflections, ("rd", a, e) reflection then diffraction,
        ("dr", e, b) diffraction then reflection and ("dd", i, j) two diffractions
        (planes: -1 = ground, 0.. = reflectors; edges: 0..).
        """
        specs = [("double", a, b) for a, b in self._plane_pairs()]
        planes = range(-1, len(self.reflectors))
        edges = range(len(self.edges))
        if self.mixed_interactions:
            specs += [("rd", a, e) for a in planes for e in edges]
            specs += [("dr", e, b) for e in edges for b in planes]
        if self.max_diffractions >= 2:
            specs += [("dd", i, j) for i in edges for j in edges if i != j]
        return specs

    @staticmethod
    def _spec_label(spec) -> Tuple[str, Tuple[Tuple[str, int], ...]]:
        kind, a, b = spec
        if kind == "double":
            return "double", (_plane_ref(a), _plane_ref(b))
        if kind == "rd":
            return "reflection-diffraction", (_plane_ref(a), ("edge", b))
        if kind == "dr":
            return "diffraction-reflection", (("edge", a), _plane_ref(b))
        return "double-diffraction", (("edge", a), ("edge", b))

    def candidate_rays(self) -> List[Tuple[str, Tuple[Tuple[str, int], ...]]]:
        """
        (kind, interactions) of every candidate ray, in the order of the axis K of
        ``trace_points`` (and of the rays of ``trace_paths``).  Interactions are
        ("ground", -1), ("wall", index of the reflector) or ("edge", index of the edge).
        Kinds: "direct", "ground", "wall", "diffraction", "double" (two reflections) and,
        new in 1.6.0, "reflection-diffraction", "diffraction-reflection" and
        "double-diffraction".
        """
        rays = [("direct", ()), ("ground", (("ground", -1),))]
        rays += [("wall", (("wall", r),)) for r in range(len(self.reflectors))]
        rays += [("diffraction", (("edge", e),)) for e in range(len(self.edges))]
        rays += [self._spec_label(spec) for spec in self._higher_order()]
        return rays

    # ------------------------------------------------------------------ visibility (pruning, new in 1.6.0)
    def _scene_key(self):
        """Fingerprint of the geometry that the visibility depends on (the reflectors can be edited in place)."""
        walls = tuple((id(r), r.p0.tobytes(), r.normal.tobytes(), r.bounds, r.transmission_loss_db,
                       None if r.footprint is None else np.asarray(r.footprint).tobytes()) for r in self.reflectors)
        edges = tuple((id(e), e.start.tobytes(), e.end.tobytes(), e.face_direction.tobytes(), e.wedge_n)
                      for e in self.edges)
        return walls, edges

    @staticmethod
    def _patch_samples(refl: PlanarReflector) -> Optional[np.ndarray]:
        """A few points of a bounded wall patch (None for unbounded walls, which are always visible)."""
        box = refl.bounds
        if box is None or not all(np.isfinite(box)):
            return None
        fractions = (1.0 / 6.0, 0.5, 5.0 / 6.0)
        grid = np.array([[box[0] + (box[1] - box[0]) * fx, box[2] + (box[3] - box[2]) * fy,
                          box[4] + (box[5] - box[4]) * fz] for fx in fractions for fy in fractions for fz in fractions])
        grid = grid - refl.signed_distance(grid)[:, None] * refl.normal
        grid = grid[refl.contains(grid)]
        if not len(grid):
            return None
        return np.unique(np.round(grid, 9), axis=0)

    @staticmethod
    def _edge_samples(edge: DiffractionEdge) -> np.ndarray:
        return edge.start + np.array([0.02, 0.25, 0.5, 0.75, 0.98])[:, None] * (edge.end - edge.start)

    def _opaque_walls(self) -> List[PlanarReflector]:
        return [r for r in self.reflectors if r.transmission_loss_db is not None and np.isinf(r.transmission_loss_db)]

    @staticmethod
    def _blocked(opaque: List[PlanarReflector], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where the segments a -> b ([S, 3]) cross an opaque wall."""
        blocked = np.zeros(len(a), dtype=bool)
        for wall in opaque:
            todo = np.flatnonzero(~blocked)
            if not todo.size:
                break
            blocked[todo] = wall.crosses(a[todo], b[todo])
        return blocked

    def _pair_visibility(self, opaque, samples_a: list, samples_b: list) -> np.ndarray:
        """[len(a), len(b)] True if some segment between the samples of a and of b is not blocked."""
        visible = np.ones((len(samples_a), len(samples_b)), dtype=bool)
        seg_a, seg_b, owner = [], [], []
        for i, sa in enumerate(samples_a):
            for j, sb in enumerate(samples_b):
                if sa is None or sb is None:
                    continue
                visible[i, j] = False
                seg_a.append(np.repeat(sa, len(sb), axis=0))
                seg_b.append(np.tile(sb, (len(sa), 1)))
                owner.append(np.full(len(sa) * len(sb), i * len(samples_b) + j))
        if seg_a:
            a, b, owner = np.concatenate(seg_a), np.concatenate(seg_b), np.concatenate(owner)
            clear = ~self._blocked(opaque, a, b)
            flat = visible.reshape(-1)
            flat[np.unique(owner[clear])] = True
        return visible

    def _visibility(self) -> Dict[str, Any]:
        """Sampled visibility between walls and edges through opaque walls (cached per geometry)."""
        key = self._scene_key()
        if self._visibility_cache[0] == key:
            return self._visibility_cache[1]
        opaque = self._opaque_walls()
        wall_samples = [self._patch_samples(r) for r in self.reflectors]
        edge_samples = [self._edge_samples(e) for e in self.edges]
        data = {"opaque": opaque, "wall_samples": wall_samples, "edge_samples": edge_samples, "dd_geometry": {},
                "patch_corners": [self._patch_corners(r) for r in self.reflectors]}
        if self.edges and opaque and (self.mixed_interactions or self.max_diffractions >= 2):
            data["wall_edge"] = self._pair_visibility(opaque, wall_samples, edge_samples) if self.mixed_interactions \
                else np.ones((len(self.reflectors), len(self.edges)), dtype=bool)
            data["edge_edge"] = self._pair_visibility(opaque, edge_samples, edge_samples) \
                if self.max_diffractions >= 2 else np.ones((len(self.edges),) * 2, dtype=bool)
        else:
            data["wall_edge"] = np.ones((len(self.reflectors), len(self.edges)), dtype=bool)
            data["edge_edge"] = np.ones((len(self.edges),) * 2, dtype=bool)
        self._visibility_cache = (key, data)
        return data

    def _mirror_reaches(self, a: int, e: int, image: np.ndarray, s_tx: float) -> bool:
        """
        False if no point of edge e can be seen from the image of Tx through the patch
        of wall a (exact): the reflection points lie on the segment where the triangle
        (image, edge) meets the plane, after clipping the edge to the side of Tx.
        """
        wall, edge = self.reflectors[a], self.edges[e]
        box = wall.bounds
        if box is None:
            return True
        tol = OBSTRUCTION_TOL_M
        ends = np.array([edge.start, edge.end])
        dist = np.sign(s_tx) * wall.signed_distance(ends)
        if dist.min() < tol:                         # clip the edge where it crosses the plane
            if dist.max() <= tol:
                return False
            t = (dist[0] - tol) / (dist[0] - dist[1])
            cut = ends[0] + t * (ends[1] - ends[0])
            ends = np.array([ends[int(np.argmax(dist))], cut])
        s_image = -s_tx
        hits = []
        for c in ends:
            s_c = float(wall.signed_distance(c))
            hits.append(image + (s_image / (s_image - s_c)) * (c - image))
        lo, hi = np.min(hits, axis=0), np.max(hits, axis=0)
        margin = 1e-6
        return all(hi[axis] >= box[2 * axis] - margin and lo[axis] <= box[2 * axis + 1] + margin for axis in range(3))

    @staticmethod
    def _angular_range(about: DiffractionEdge, points):
        """
        Angles (in the convention of ``about.cylindrical``) under which the convex hull
        of ``points`` ([N, 3]: the ends of a segment, the corners of a wall patch) is seen
        from the line of edge ``about``, as a list of (lo, hi) intervals of [0, 2 pi];
        None when the hull meets that line or comes too close to it, where every angle
        is possible.  A convex set that misses the line lies in a half-plane around it,
        so its angles fill the circle except the largest gap between the angles of the
        points, which then exceeds pi.  The intervals are widened by 1e-4 rad (points on
        the cut phi = 0 may count as 0 or 2 pi, and patches are tested 1 nm wider).
        """
        pts = np.asarray(points, dtype=float) - about.start
        perp = pts - (pts @ about.axis)[:, None] * about.axis
        x, y = perp @ about.face_direction, perp @ about.face_normal
        if np.any(np.hypot(x, y) <= 1e-3):
            return None
        two_pi, delta = 2.0 * math.pi, 1e-4
        phi = np.sort(np.mod(np.arctan2(y, x), two_pi))
        gaps = np.diff(np.append(phi, phi[0] + two_pi))
        g = int(np.argmax(gaps))
        if gaps[g] <= math.pi + 2.0 * delta:
            return None
        lo, hi = float(phi[(g + 1) % len(phi)]) - delta, float(phi[g]) + delta   # counter-clockwise lo -> hi
        if g == len(phi) - 1:                       # the gap is the one that contains the cut: no wrap
            intervals = [(lo, hi)]
        else:                                       # the arc crosses the cut
            intervals = [(lo, two_pi), (0.0, hi)]
        out = []
        for a, b in intervals:                      # fold what the widening pushed across the cut
            if a < 0.0:
                out += [(0.0, b), (two_pi + a, two_pi)]
            elif b > two_pi:
                out += [(a, two_pi), (0.0, b - two_pi)]
            else:
                out.append((a, b))
        return out

    @staticmethod
    def _zone_reachable(intervals, phi_in, phi_out, n: float, margin):
        """
        Whether some pair of angles, one taken from ``intervals`` (a list of (lo, hi))
        and the other given, passes ``_transition_zone`` with the given margin (or a
        larger one): ``phi_in`` None means the incidence angle ranges over the intervals
        (``phi_out`` an array, the result too), ``phi_out`` None that the observation
        angle does (``phi_in`` and the result scalars).
        """
        pi = math.pi
        rsb = 2.0 * n - 1.0
        if phi_out is None:
            for lo, hi in intervals:
                if hi >= phi_in + pi - margin or lo <= phi_in - pi + margin:
                    return True
                if any(lo <= b + margin and hi >= b - margin for b in (pi - phi_in, rsb * pi - phi_in)):
                    return True
            return False
        ok = np.zeros(np.shape(phi_out), dtype=bool)
        for lo, hi in intervals:
            ok |= (phi_out >= lo + pi - margin) | (phi_out <= hi - pi + margin)
            ok |= (phi_out >= pi - hi - margin) & (phi_out <= pi - lo + margin)
            ok |= (phi_out >= rsb * pi - hi - margin) & (phi_out <= rsb * pi - lo + margin)
        return ok

    @staticmethod
    def _keller_fits(edge: DiffractionEdge, s_p, r_p, s_range, r_min):
        """
        Necessary condition for the point of edge ``edge`` that ends the shortest path
        from P (coordinates s_p, r_p around the edge, scalars or arrays) to some point Q
        of a segment to lie on the edge.  Given Q, that point is exactly
        s = s_p + (s_q - s_p) w with w = r_p / (r_p + r_q) (the unfolded path), so it is
        bounded with the range of s_q over the segment (``s_range``) and w <= r_p / (r_p
        + r_min), r_min the distance from the segment to the edge line.
        """
        slack = _BOUNDS_TOL_M + 1e-7
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(r_p + r_min > 0, r_p / (r_p + r_min), 1.0)
        above = s_p - edge.length - slack
        below = -s_p - slack
        return (((above <= 0) | (w * np.maximum(0.0, s_p - s_range[0]) >= above))
                & ((below <= 0) | (w * np.maximum(0.0, s_range[1] - s_p) >= below)))

    @staticmethod
    def _patch_corners(refl: PlanarReflector) -> Optional[np.ndarray]:
        """
        Corners of the patch of a bounded wall: where its plane meets the edges of the
        bounding box (a convex polygon that contains the patch); None if unbounded.
        """
        box = refl.bounds
        if box is None or not all(np.isfinite(box)):
            return None
        corners = np.array([[x, y, z] for x in box[0:2] for y in box[2:4] for z in box[4:6]])
        dist = refl.signed_distance(corners)
        points = list(corners[np.abs(dist) <= 1e-12])
        for a in range(8):
            for b in range(a + 1, 8):
                if np.count_nonzero(corners[a] != corners[b]) != 1 or dist[a] * dist[b] >= 0.0:
                    continue
                t = dist[a] / (dist[a] - dist[b])
                points.append(corners[a] + t * (corners[b] - corners[a]))
        return np.array(points) if points else None

    @staticmethod
    def _box_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Distance between the bounding boxes of two point sets (a lower bound of their distance)."""
        gap = np.maximum(0.0, np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0)))
        return float(np.linalg.norm(gap))

    def _dr_pair_possible(self, e: int, b: int, tx_p: np.ndarray, k: float) -> bool:
        """
        Necessary condition of a ray Tx -> edge e -> wall b -> Rx to pass the
        transition-zone test at the edge for some receiver: it leaves the edge towards
        the reflection point, so its angle is one of those under which the edge sees the
        wall patch (``_angular_range``), and L = r_t^2 s2 / (s1 (s1 + s2)) with s1 at most
        the distance from Tx to the farther end of the edge and s2 at least the distance
        from the edge to the patch.
        """
        corners = self._visibility()["patch_corners"][b]
        if corners is None:
            return True
        edge = self.edges[e]
        intervals = self._angular_range(edge, corners)
        if intervals is None:
            return True
        _, r_t, phi_t = (float(v) for v in edge.cylindrical(tx_p))
        far = max(float(np.linalg.norm(edge.start - tx_p)), float(np.linalg.norm(edge.end - tx_p)))
        near = self._box_distance(np.array([edge.start, edge.end]), corners)
        low = r_t * r_t * near / (far * (far + near))
        margin = min(math.pi, math.sqrt(20.0 / (k * max(low, 1e-300))))
        return margin >= math.pi or bool(self._zone_reachable(intervals, phi_t, None, edge.wedge_n, margin))

    @staticmethod
    def _segment_from_line(edge: DiffractionEdge, start, end):
        """(s range along the edge, distance to the edge line) of the segment start-end."""
        ends = np.array([start, end], dtype=float) - edge.start
        s_ends = ends @ edge.axis
        perp = ends - s_ends[:, None] * edge.axis
        v = perp[1] - perp[0]
        vv = float(v @ v)
        t = 0.0 if vv == 0.0 else min(1.0, max(0.0, -float(perp[0] @ v) / vv))
        return (float(s_ends.min()), float(s_ends.max())), float(np.linalg.norm(perp[0] + t * v))

    def _dd_pair_possible(self, ei: DiffractionEdge, ej: DiffractionEdge, tx_p: np.ndarray, gap: float,
                          k: float) -> bool:
        """
        Necessary conditions, for some receiver, of a doubly diffracted ray Tx -> edge i
        -> edge j: the point on edge i can lie on the segment (``_keller_fits``) and the
        ray can pass the transition-zone test at edge i (``_transition_zone``) for some
        point of edge j.  Exact on the angles (``_angular_range``) and with the smallest
        possible distance parameter, L >= r_t^2 gap / (d (d + gap)), d the farthest
        distance from Tx to edge i (L = r_t^2 l2 / (l1 (l1 + l2)), l2 >= gap, l1 <= d).
        """
        s_t, r_t, phi_t = (float(v) for v in ei.cylindrical(tx_p))
        s_range, r_min = self._segment_from_line(ei, ej.start, ej.end)
        if not bool(self._keller_fits(ei, s_t, r_t, s_range, r_min)):
            return False
        far = max(float(np.linalg.norm(ei.start - tx_p)), float(np.linalg.norm(ei.end - tx_p)))
        low = r_t * r_t * gap / (far * (far + gap))
        margin = min(math.pi, math.sqrt(20.0 / (k * max(low, 1e-300))))
        intervals = self._angular_range(ei, [ej.start, ej.end])
        if margin >= math.pi or intervals is None:
            return True
        return bool(self._zone_reachable(intervals, phi_t, None, ei.wedge_n, margin))

    def _dd_geometry(self, i: int, j: int) -> dict:
        """
        What the per-receiver pruning of the doubly diffracted rays i -> j needs, which
        depends only on the two edges (cached with the scene): the distance between
        them, the angles under which edge j sees edge i, and the position and distance
        of edge i relative to the line of edge j.
        """
        cache = self._visibility()["dd_geometry"]
        if (i, j) not in cache:
            ei, ej = self.edges[i], self.edges[j]
            s_range, r_min = self._segment_from_line(ej, ei.start, ei.end)
            cache[(i, j)] = {"gap": _segment_distance(ei.start, ei.end, ej.start, ej.end),
                             "angles": self._angular_range(ej, [ei.start, ei.end]),
                             "s_range": s_range, "r_min": r_min}
        return cache[(i, j)]

    def _active_higher_order(self, tx: Transmitter) -> List[Tuple[int, Tuple[str, int, int]]]:
        """
        (position in ``candidate_rays``, spec) of the two-interaction candidates that
        can exist for this transmitter.  Exact geometric tests (Tx or images on a
        plane, patches entirely behind a plane, Tx inside a wedge, edges inside each
        other's wedge or touching) and, for the rays with a diffraction (new in 1.6.0),
        a sampled visibility test: an object pair (or Tx-object pair) is dropped when
        all the segments between their sample points cross an opaque wall.
        """
        specs = self._higher_order()
        if not specs:
            return []
        tx_p = np.array([tx.x, tx.y, tx.z], dtype=float)
        key = (tx_p.tobytes(), self._scene_key(), self.max_reflections, self.max_diffractions, self.mixed_interactions)
        if self._active_cache[0] == key:
            return self._active_cache[1]
        tol = OBSTRUCTION_TOL_M
        first = 2 + len(self.reflectors) + len(self.edges)
        vis = self._visibility()
        opaque = vis["opaque"]
        need_tx_vis = any(spec[0] != "double" for spec in specs)
        tx_sees_wall = np.ones(len(self.reflectors), dtype=bool)
        tx_sees_edge = np.ones(len(self.edges), dtype=bool)
        if need_tx_vis and opaque:
            tx_sees_wall = self._pair_visibility(opaque, [tx_p[None, :]], vis["wall_samples"])[0]
            tx_sees_edge = self._pair_visibility(opaque, [tx_p[None, :]], vis["edge_samples"])[0]
        s_tx = {a: float(self._plane_distance(a, tx_p)) for a in range(-1, len(self.reflectors))}
        tx_in_wedge = []
        for edge in self.edges:
            _, r, phi = (float(v) for v in edge.cylindrical(tx_p))
            tx_in_wedge.append(r > tol and phi <= edge.exterior_angle + _ANGLE_TOL)

        def edge_side(a, e):
            """Signed distances of the ends of edge e to plane a."""
            edge = self.edges[e]
            return self._plane_distance(a, np.array([edge.start, edge.end]))

        def outside_wedge(e, points):
            """True if some of ``points`` lies in the exterior of wedge e (or on its edge line)."""
            _, r, phi = self.edges[e].cylindrical(np.asarray(points, dtype=float))
            return bool(np.any((phi <= self.edges[e].exterior_angle + _ANGLE_TOL) | (r <= tol)))

        active = []
        for offset, spec in enumerate(specs):
            kind, a, b = spec
            if kind == "double":
                if not abs(s_tx[a]) > tol:
                    continue
                image = self._mirror(a, tx_p, s_tx[a])
                s_image = float(self._plane_distance(b, image))
                if not abs(s_image) > tol:
                    continue
                if b >= 0:
                    box = self.reflectors[b].bounds
                    if box is not None and all(np.isfinite(box)):
                        corners = np.array([[x, y, z] for x in box[0:2] for y in box[2:4] for z in box[4:6]])
                        if np.all(np.sign(s_tx[a]) * self._plane_distance(a, corners) < -tol):
                            continue
            elif kind == "rd":
                e = b
                if not abs(s_tx[a]) > tol:
                    continue
                ends = edge_side(a, e)
                if np.all(np.abs(ends) <= tol) or not np.max(np.sign(s_tx[a]) * ends) > tol:
                    continue
                if a >= 0 and not (tx_sees_wall[a] and vis["wall_edge"][a, e]):
                    continue
                image = self._mirror(a, tx_p, s_tx[a])
                _, r_i, phi_i = (float(v) for v in self.edges[e].cylindrical(image))
                if not (r_i > tol and phi_i <= self.edges[e].exterior_angle + _ANGLE_TOL):
                    continue
                if a >= 0 and not self._mirror_reaches(a, e, image, s_tx[a]):
                    continue
            elif kind == "dr":
                e = a
                if not (tx_in_wedge[e] and tx_sees_edge[e]):
                    continue
                if np.all(np.abs(edge_side(b, e)) <= tol):
                    continue
                if b >= 0 and not vis["wall_edge"][b, e]:
                    continue
                if b >= 0 and not self._dr_pair_possible(e, b, tx_p, 2.0 * math.pi * tx.frequency_hz / SPEED_OF_LIGHT):
                    continue
            else:                                                       # "dd"
                i, j = a, b
                if not (tx_in_wedge[i] and tx_sees_edge[i] and vis["edge_edge"][i, j]):
                    continue
                ei, ej = self.edges[i], self.edges[j]
                gap = _segment_distance(ei.start, ei.end, ej.start, ej.end)
                if gap <= tol:
                    continue                                            # edges that touch: a vertex, not two edges
                if not outside_wedge(i, [ej.start, ej.end]) or not outside_wedge(j, [ei.start, ei.end]):
                    continue
                if not self._dd_pair_possible(ei, ej, tx_p, gap, 2.0 * math.pi * tx.frequency_hz / SPEED_OF_LIGHT):
                    continue
            active.append((first + offset, spec))
        self._active_cache = (key, active)
        return active

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

    # ------------------------------------------------------------------ diffraction and double reflections
    def _plane_distance(self, index: int, points) -> np.ndarray:
        """Signed distance to plane ``index`` (-1 = ground z = 0)."""
        points = np.asarray(points, dtype=float)
        return points[..., 2] if index < 0 else self.reflectors[index].signed_distance(points)

    def _mirror(self, index: int, point: np.ndarray, distance: float) -> np.ndarray:
        """Image of ``point`` across plane ``index`` (its signed distance given)."""
        if index < 0:
            return point * np.array([1.0, 1.0, -1.0])
        return point - 2.0 * distance * self.reflectors[index].normal

    def _plane_contains(self, index: int, points: np.ndarray) -> np.ndarray:
        if index < 0:
            return np.ones(points.shape[:-1], dtype=bool)
        return self.reflectors[index].contains(points)

    def _plane_coefficient(self, index: int, sin_grazing: np.ndarray, frequency_hz: float) -> np.ndarray:
        if index < 0:
            psi = np.arcsin(np.clip(sin_grazing, 0.0, 1.0))
            return np.asarray(self._fresnel_ground_reflection(psi, frequency_hz))
        return np.asarray(self.reflectors[index].reflection_coefficient(sin_grazing, frequency_hz))

    @staticmethod
    def _polyline_obstruction(obstacles: list, idx: np.ndarray, vertices: list, bend_owns: list) -> np.ndarray:
        """
        Loss in dB ([m]) of the polylines vertices[0] (Tx, [3]) -> vertices[1] -> ... ->
        vertices[-1] (Rx, [m, 3]) for the points ``idx`` through the obstructing walls
        (``obstacles`` as built in ``_extra_rays``).  Every leg is tested as in
        ``PlanarReflector.crosses``.  bend_owns gives, per interior vertex, None (no
        test at that bend: a diffraction point, where the ray legitimately turns
        around the edge) or the index of the plane the ray is reflected on there
        (-1 = ground), which is the only wall not tested at that bend.
        """
        loss = np.zeros(len(idx))
        if not obstacles:
            return loss
        # Bounding boxes of the vertices (over all the points): a leg whose box misses the box of a
        # wall (with its 1 nm margin) cannot cross it, so that wall is skipped for that leg
        lows = np.array([np.min(np.reshape(v, (-1, 3)), axis=0) for v in vertices])
        highs = np.array([np.max(np.reshape(v, (-1, 3)), axis=0) for v in vertices])
        leg_lo, leg_hi = np.minimum(lows[:-1], lows[1:]), np.maximum(highs[:-1], highs[1:])
        bend_lo, bend_hi = np.minimum(leg_lo[:-1], leg_lo[1:]), np.maximum(leg_hi[:-1], leg_hi[1:])
        unbounded = (-np.inf, np.inf) * 3
        boxes = np.array([wall.bounds if wall.bounds is not None else unbounded for _, wall, _, _ in obstacles])
        wall_lo, wall_hi = boxes[:, 0::2] - _BOUNDS_TOL_M, boxes[:, 1::2] + _BOUNDS_TOL_M
        leg_hit = ~np.any((leg_hi[None] < wall_lo[:, None]) | (leg_lo[None] > wall_hi[:, None]), axis=-1)
        bend_hit = ~np.any((bend_hi[None] < wall_lo[:, None]) | (bend_lo[None] > wall_hi[:, None]), axis=-1)

        for w, (index, wall, s_tx, s_rx) in enumerate(obstacles):
            legs = np.flatnonzero(leg_hit[w]).tolist()
            bends = [(b, own) for b, own in enumerate(bend_owns, start=1)
                     if own is not None and own != index and bend_hit[w, b - 1]]
            if not legs and not bends:
                continue
            dist = [s_tx] + [wall.signed_distance(v) for v in vertices[1:-1]] + [s_rx[idx]]
            for leg in legs:
                crossed = wall._crosses(dist[leg], dist[leg + 1], vertices[leg], vertices[leg + 1])
                np.add(loss, wall.transmission_loss_db, out=loss, where=crossed)
            for b, own in bends:
                crossed = wall._crosses_at_bend(dist[b - 1], dist[b], dist[b + 1],
                                                vertices[b - 1], vertices[b], vertices[b + 1])
                np.add(loss, wall.transmission_loss_db, out=loss, where=crossed)
        return loss

    def _extra_rays(self, tx: Transmitter, pts: np.ndarray, k: float, amp_1m: float, sector: bool,
                    want_vertices: bool) -> List[Tuple[int, _Rays]]:
        """
        The rays added since 1.5.0, as (position in ``candidate_rays``, rays) in that
        order, for the flat array of receiver points ``pts`` ([M, 3]): one diffraction
        per edge, then the two-interaction candidates that can exist for this
        transmitter (``_active_higher_order``).  Shared by ``trace_paths`` (one point)
        and ``trace_points``, so both give the same rays.
        """
        tx_p = np.array([tx.x, tx.y, tx.z], dtype=float)
        obstacles = [(index, refl, float(refl.signed_distance(tx_p)), refl.signed_distance(pts))
                     for index, refl in enumerate(self.reflectors) if refl.obstructs]
        first = 2 + len(self.reflectors)
        out = [(first + e, self._diffracted_ray(tx, tx_p, pts, e, edge, k, amp_1m, sector, obstacles, want_vertices))
               for e, edge in enumerate(self.edges)]
        cache = {}

        def rx_distance(index):
            if index not in cache:
                cache[index] = self._plane_distance(index, pts)
            return cache[index]

        edge_coords = {}

        def rx_cylindrical(e):
            if e not in edge_coords:
                edge_coords[e] = self.edges[e].cylindrical(pts)
            return edge_coords[e]

        edge_far = {}

        def rx_far(e):
            """Distance from each receiver to the farther end of edge e."""
            if e not in edge_far:
                edge = self.edges[e]
                edge_far[e] = np.maximum(np.linalg.norm(pts - edge.start, axis=-1),
                                         np.linalg.norm(pts - edge.end, axis=-1))
            return edge_far[e]

        for position, (kind, a, b) in self._active_higher_order(tx):
            if kind == "double":
                ray = self._double_ray(tx, tx_p, pts, a, b, k, amp_1m, sector, obstacles, rx_distance, want_vertices)
            elif kind == "rd":
                ray = self._reflect_diffract_ray(tx, tx_p, pts, a, b, k, amp_1m, obstacles, rx_cylindrical,
                                                 want_vertices)
            elif kind == "dr":
                ray = self._diffract_reflect_ray(tx, tx_p, pts, a, b, k, amp_1m, obstacles, rx_distance,
                                                 want_vertices)
            else:
                ray = self._double_diffraction_ray(tx, tx_p, pts, a, b, k, amp_1m, obstacles, rx_cylindrical,
                                                   rx_far, want_vertices)
            out.append((position, ray))
        return out

    def _finish(self, ray: _Rays, tx: Transmitter, tx_p, idx, amp, length, first_point, last_point, rx,
                obstacles, vertices, bend_owns, want_vertices) -> _Rays:
        """Angles, antenna pattern and obstruction common to the extra rays."""
        tx_phi, tx_theta = _angles_deg(tx_p, first_point)
        rx_phi, rx_theta = _angles_deg(rx, last_point)
        if not tx.is_isotropic:
            amp = amp * self._pattern_factor(tx, tx_phi, tx_theta)
        loss = np.zeros(len(idx))
        if obstacles:
            loss = self._polyline_obstruction(obstacles, idx, [tx_p] + vertices + [rx], bend_owns)
            if loss.any():
                amp = amp * 10.0 ** (-loss / 20.0)
        return ray.fill(idx, amp, length, (tx_phi, tx_theta, rx_phi, rx_theta), loss,
                        vertices if want_vertices else None)

    def _diffracted_ray(self, tx, tx_p, pts, e, edge: DiffractionEdge, k, amp_1m, sector, obstacles,
                        want_vertices) -> _Rays:
        """Tx -> Q (on the edge, Keller's law) -> Rx, with the UTD coefficient of the wedge."""
        ray = _Rays("diffraction", (("edge", e),))
        s_t, r_t, phi_t = (float(v) for v in edge.cylindrical(tx_p))
        ext = edge.exterior_angle
        if not (r_t > OBSTRUCTION_TOL_M and phi_t <= ext + _ANGLE_TOL):
            return ray
        s_r, r_r, phi_r = edge.cylindrical(pts)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_q = s_t + (s_r - s_t) * (r_t / (r_t + r_r))
        ok = ((r_r > OBSTRUCTION_TOL_M) & (phi_r <= ext + _ANGLE_TOL)
              & (s_q >= -_BOUNDS_TOL_M) & (s_q <= edge.length + _BOUNDS_TOL_M))
        idx = np.flatnonzero(ok)
        if not idx.size:
            return ray
        s_q, s_r, r_r = s_q[idx], s_r[idx], r_r[idx]
        phi_r = np.minimum(phi_r[idx], ext)
        phi_t = min(phi_t, ext)
        rx = pts[idx]
        q = edge.start + s_q[:, None] * edge.axis
        s1 = np.sqrt(r_t * r_t + (s_q - s_t) ** 2)          # Tx -> Q
        s2 = np.sqrt(r_r * r_r + (s_r - s_q) ** 2)          # Q -> Rx
        total = s1 + s2
        sin_beta0 = r_t / s1
        distance_parameter = s1 * s2 * sin_beta0 ** 2 / total
        r0, rn = edge.face_coefficients(phi_t, phi_r, tx.frequency_hz)
        d = np.asarray(utd_coefficient(phi_r, phi_t, edge.wedge_n, k, distance_parameter, sin_beta0, r0, rn,
                                       edge.isb_scale))
        # Field at Q (spherical wave from Tx) times D and the spreading factor of the diffracted wave
        amp = (amp_1m / s1) * d * np.sqrt(s1 / (s2 * total)) * np.exp(-1j * k * total)
        return self._finish(ray, tx, tx_p, idx, amp, total, q, q, rx, obstacles, [q], [None], want_vertices)

    def _double_ray(self, tx, tx_p, pts, a, b, k, amp_1m, sector, obstacles, rx_distance, want_vertices) -> _Rays:
        """Tx -> P1 (plane a) -> P2 (plane b) -> Rx by the image method (-1 = ground)."""
        tol = OBSTRUCTION_TOL_M
        ray = _Rays("double", (_plane_ref(a), _plane_ref(b)))
        s_a_tx = float(self._plane_distance(a, tx_p))
        if not abs(s_a_tx) > tol:
            return ray
        image_a = self._mirror(a, tx_p, s_a_tx)
        s_b_image = float(self._plane_distance(b, image_a))
        if not abs(s_b_image) > tol:
            return ray
        if b >= 0:
            # P2 must be on the Tx side of plane a: impossible if the whole patch b is behind it
            box = self.reflectors[b].bounds
            if box is not None and all(np.isfinite(box)):
                corners = np.array([[x, y, z] for x in box[0:2] for y in box[2:4] for z in box[4:6]])
                if np.all(np.sign(s_a_tx) * self._plane_distance(a, corners) < -tol):
                    return ray
        image_ab = self._mirror(b, image_a, s_b_image)
        s_b_rx = rx_distance(b)
        idx = np.flatnonzero(_same_side(s_b_image, s_b_rx, tol))      # Rx on the side of plane b of image_a
        if not idx.size:
            return ray
        rx = pts[idx]
        s_b_ab = -s_b_image
        t2 = s_b_ab / (s_b_ab - s_b_rx[idx])
        p2 = image_ab + t2[:, None] * (rx - image_ab)
        s_a_p2 = self._plane_distance(a, p2)
        keep = self._plane_contains(b, p2) & _same_side(s_a_tx, s_a_p2, tol)
        if not keep.any():
            return ray
        idx, rx, p2, s_a_p2 = idx[keep], rx[keep], p2[keep], s_a_p2[keep]
        t1 = -s_a_tx / (-s_a_tx - s_a_p2)
        p1 = image_a + t1[:, None] * (p2 - image_a)
        seg = p2 - p1
        seg_len = np.linalg.norm(seg, axis=-1)
        keep = self._plane_contains(a, p1) & (seg_len > tol)
        if not keep.any():
            return ray
        idx, rx, p1, p2, seg, seg_len = idx[keep], rx[keep], p1[keep], p2[keep], seg[keep], seg_len[keep]
        leg1 = p1 - tx_p
        normal_a = np.array([0.0, 0.0, 1.0]) if a < 0 else self.reflectors[a].normal
        normal_b = np.array([0.0, 0.0, 1.0]) if b < 0 else self.reflectors[b].normal
        sin_a = np.abs(leg1 @ normal_a) / np.linalg.norm(leg1, axis=-1)
        sin_b = np.abs(seg @ normal_b) / seg_len
        gamma = (self._plane_coefficient(a, sin_a, tx.frequency_hz)
                 * self._plane_coefficient(b, sin_b, tx.frequency_hz))
        length = np.linalg.norm(rx - image_ab, axis=-1)
        amp = (amp_1m / length) * gamma * np.exp(-1j * k * length)
        return self._finish(ray, tx, tx_p, idx, amp, length, p1, p2, rx, obstacles, [p1, p2], [a, b],
                            want_vertices)

    def _utd(self, edge: DiffractionEdge, phi_in, phi_out, k, s_in, s_out, sin_beta0, frequency_hz,
             grazing_leg: Optional[str] = None):
        """
        UTD coefficient of ``edge`` for incidence phi_in from distance s_in and observation phi_out at s_out.
        grazing_leg ("in" or "out"): the incident or the diffracted ray is the leg between two edges of a
        double diffraction; where that leg runs along a face (within 1e-6 rad), the coefficient of that face
        is the one of a perfect conductor (-1 soft, +1 hard): Luebbers' heuristic, which gives -1 at grazing
        incidence for every lossy face, is not reliable there and would cancel the field along a roof.
        """
        ext = edge.exterior_angle
        phi_in = np.minimum(phi_in, ext)
        phi_out = np.minimum(phi_out, ext)
        distance_parameter = s_in * s_out * sin_beta0 ** 2 / (s_in + s_out)
        r0, rn = edge.face_coefficients(phi_in, phi_out, frequency_hz)
        if grazing_leg is not None:
            leg = phi_in if grazing_leg == "in" else phi_out
            pec = -1.0 if edge.polarization == "soft" else 1.0
            r0 = np.where(leg < 1e-6, pec, r0)
            rn = np.where(leg > ext - 1e-6, pec, rn)
        return np.asarray(utd_coefficient(phi_out, phi_in, edge.wedge_n, k, distance_parameter, sin_beta0, r0, rn,
                                          edge.isb_scale))

    @staticmethod
    def _transition_zone(phi_in, phi_out, n: float, kL) -> np.ndarray:
        """
        True where the observation angle is in the shadow of the wedge or near one of
        its shadow boundaries (incident: phi_in +- pi; reflected: pi - phi_in and
        (2n - 1) pi - phi_in): where kL a / 2 < 10, i.e. |phi - boundary| < sqrt(20 / kL),
        the transition function F differs from 1 by more than ~5 %.  Elsewhere a
        diffracted ray is a correction of the order of -40 dB to the rays that reach
        the point without it (used for the two-interaction rays with a diffraction).
        """
        margin = np.minimum(math.pi, np.sqrt(20.0 / np.maximum(kL, 1e-300)))
        isb = phi_in + math.pi
        return ((phi_out > isb - margin) | (phi_out < phi_in - math.pi + margin)
                | (np.abs(phi_out - (math.pi - phi_in)) < margin)
                | (np.abs(phi_out - ((2.0 * n - 1.0) * math.pi - phi_in)) < margin))

    def _reflect_diffract_ray(self, tx, tx_p, pts, a, e, k, amp_1m, obstacles, rx_cylindrical, want_vertices) -> _Rays:
        """Tx -> P (plane a) -> Q (edge e) -> Rx: the edge is lit by the image of Tx across plane a."""
        tol = OBSTRUCTION_TOL_M
        ray = _Rays("reflection-diffraction", (_plane_ref(a), ("edge", e)))
        edge = self.edges[e]
        s_a_tx = float(self._plane_distance(a, tx_p))
        image = self._mirror(a, tx_p, s_a_tx)
        s_i, r_i, phi_i = (float(v) for v in edge.cylindrical(image))
        s_r, r_r, phi_r = rx_cylindrical(e)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_q = s_i + (s_r - s_i) * (r_i / (r_i + r_r))
        ok = ((r_r > tol) & (phi_r <= edge.exterior_angle + _ANGLE_TOL)
              & (s_q >= -_BOUNDS_TOL_M) & (s_q <= edge.length + _BOUNDS_TOL_M))
        idx = np.flatnonzero(ok)
        if not idx.size:
            return ray
        s_q, s_r, r_r, phi_r = s_q[idx], s_r[idx], r_r[idx], phi_r[idx]
        q = edge.start + s_q[:, None] * edge.axis
        s_a_q = self._plane_distance(a, q)
        keep = _same_side(s_a_tx, s_a_q, tol)                         # Q in front of the mirror, like Tx
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -s_a_tx / (-s_a_tx - s_a_q)
            p = image + t[:, None] * (q - image)
        keep &= self._plane_contains(a, p)
        if not keep.any():
            return ray
        idx, q, p, s_q, s_r, r_r, phi_r = idx[keep], q[keep], p[keep], s_q[keep], s_r[keep], r_r[keep], phi_r[keep]
        s1 = np.sqrt(r_i * r_i + (s_q - s_i) ** 2)                    # image -> Q = Tx -> P -> Q
        s2 = np.sqrt(r_r * r_r + (s_r - s_q) ** 2)
        zone = self._transition_zone(phi_i, phi_r, edge.wedge_n, k * s1 * s2 * (r_i / s1) ** 2 / (s1 + s2))
        if not zone.any():
            return ray
        idx, q, p, s1, s2, phi_r = idx[zone], q[zone], p[zone], s1[zone], s2[zone], phi_r[zone]
        rx = pts[idx]
        d = self._utd(edge, phi_i, phi_r, k, s1, s2, r_i / s1, tx.frequency_hz)
        leg1 = p - tx_p
        normal = np.array([0.0, 0.0, 1.0]) if a < 0 else self.reflectors[a].normal
        gamma = self._plane_coefficient(a, np.abs(leg1 @ normal) / np.linalg.norm(leg1, axis=-1), tx.frequency_hz)
        total = s1 + s2
        amp = (amp_1m / s1) * gamma * d * np.sqrt(s1 / (s2 * total)) * np.exp(-1j * k * total)
        return self._finish(ray, tx, tx_p, idx, amp, total, p, q, rx, obstacles, [p, q], [a, None], want_vertices)

    def _diffract_reflect_ray(self, tx, tx_p, pts, e, b, k, amp_1m, obstacles, rx_distance, want_vertices) -> _Rays:
        """Tx -> Q (edge e) -> P (plane b) -> Rx: the edge is observed from the image of Rx across plane b."""
        tol = OBSTRUCTION_TOL_M
        ray = _Rays("diffraction-reflection", (("edge", e), _plane_ref(b)))
        edge = self.edges[e]
        s_t, r_t, phi_t = (float(v) for v in edge.cylindrical(tx_p))
        s_b_rx = rx_distance(b)
        idx = np.flatnonzero(np.abs(s_b_rx) > tol)
        if not idx.size:
            return ray
        rx = pts[idx]
        normal = np.array([0.0, 0.0, 1.0]) if b < 0 else self.reflectors[b].normal
        images = rx * np.array([1.0, 1.0, -1.0]) if b < 0 else rx - 2.0 * s_b_rx[idx][:, None] * normal
        s_j, r_j, phi_j = edge.cylindrical(images)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_q = s_t + (s_j - s_t) * (r_t / (r_t + r_j))
        keep = ((r_j > tol) & (phi_j <= edge.exterior_angle + _ANGLE_TOL)
                & (s_q >= -_BOUNDS_TOL_M) & (s_q <= edge.length + _BOUNDS_TOL_M))
        if not keep.any():
            return ray
        idx, rx, images, s_j, r_j, phi_j, s_q = idx[keep], rx[keep], images[keep], s_j[keep], r_j[keep], phi_j[keep], s_q[keep]
        q = edge.start + s_q[:, None] * edge.axis
        s_b_q = self._plane_distance(b, q)
        s_rx = s_b_rx[idx]
        keep = (((s_b_q > tol) & (s_rx > tol)) | ((s_b_q < -tol) & (s_rx < -tol)))   # Q in front of the mirror, like Rx
        with np.errstate(divide="ignore", invalid="ignore"):
            t = s_b_q / (s_b_q + s_rx)                                 # s_b(image) = -s_b(Rx)
            p = q + t[:, None] * (images - q)
        keep &= self._plane_contains(b, p)
        if not keep.any():
            return ray
        idx, rx, images, s_j, r_j, phi_j, s_q, q, p = (v[keep] for v in (idx, rx, images, s_j, r_j, phi_j, s_q, q, p))
        s1 = np.sqrt(r_t * r_t + (s_q - s_t) ** 2)
        s2 = np.sqrt(r_j * r_j + (s_j - s_q) ** 2)                    # Q -> image = Q -> P -> Rx
        zone = self._transition_zone(phi_t, phi_j, edge.wedge_n, k * s1 * s2 * (r_t / s1) ** 2 / (s1 + s2))
        if not zone.any():
            return ray
        idx, rx, q, p, s1, s2, phi_j = idx[zone], rx[zone], q[zone], p[zone], s1[zone], s2[zone], phi_j[zone]
        d = self._utd(edge, phi_t, phi_j, k, s1, s2, r_t / s1, tx.frequency_hz)
        leg3 = rx - p
        gamma = self._plane_coefficient(b, np.abs(leg3 @ normal) / np.linalg.norm(leg3, axis=-1), tx.frequency_hz)
        total = s1 + s2
        amp = (amp_1m / s1) * d * np.sqrt(s1 / (s2 * total)) * gamma * np.exp(-1j * k * total)
        return self._finish(ray, tx, tx_p, idx, amp, total, q, p, rx, obstacles, [q, p], [None, b], want_vertices)

    def _double_diffraction_ray(self, tx, tx_p, pts, i, j, k, amp_1m, obstacles, rx_cylindrical, rx_far,
                                want_vertices) -> _Rays:
        """
        Tx -> Q1 (edge i) -> Q2 (edge j) -> Rx.  Q1 and Q2 minimize the path length
        (the generalized Keller law at both edges), found per receiver by Newton's
        method on the two edge parameters (a convex problem); the path exists if both
        lie on their segments.  Cascaded UTD: the wave diffracted by edge i has its
        caustics at Q1 and at distance s1 behind it, which gives
        E = E0 * D1 * D2 / sqrt(s1 s2 s3 (s1 + s2 + s3)), with L1 = s1 s2 sin^2 / (s1 + s2)
        and L2 = s2 s3 sin^2 / (s2 + s3) (exact for parallel edges); D2 is halved when
        the ray reaches edge j grazing one of its faces (Kouyoumjian & Pathak).

        Before Newton's method, two necessary conditions per receiver drop most points:
        the point on edge j can lie on the segment (``_keller_fits``), and the receiver
        can be in the transition zone of edge j for some direction of arrival from edge
        i, with the smallest possible distance parameter L >= r_rx^2 gap / (d (d + gap)),
        d the distance to the farther end of edge j.
        """
        tol = OBSTRUCTION_TOL_M
        ray = _Rays("double-diffraction", (("edge", i), ("edge", j)))
        e1, e2 = self.edges[i], self.edges[j]
        _, r_t, phi_t = (float(v) for v in e1.cylindrical(tx_p))
        s_r, r_r, phi_r = rx_cylindrical(j)
        geo = self._dd_geometry(i, j)
        ok = ((r_r > tol) & (phi_r <= e2.exterior_angle + _ANGLE_TOL)
              & self._keller_fits(e2, s_r, r_r, geo["s_range"], geo["r_min"]))
        idx = np.flatnonzero(ok)
        if idx.size and geo["angles"] is not None:
            far, gap = rx_far(j)[idx], geo["gap"]
            low = r_r[idx] ** 2 * gap / (far * (far + gap))
            margin = np.minimum(math.pi, np.sqrt(20.0 / (k * np.maximum(low, 1e-300))))
            idx = idx[(margin >= math.pi) | self._zone_reachable(geo["angles"], None, phi_r[idx], e2.wedge_n, margin)]
        if not idx.size:
            return ray
        rx = pts[idx]
        u1, u2 = e1.axis, e2.axis
        if abs(float(u1 @ u2)) > 1.0 - 1e-12:
            t1, t2, converged = self._parallel_double_keller(e1, e2, tx_p, rx)
        else:
            t1, t2, converged = self._newton_double_keller(e1, e2, tx_p, rx, r_t, s_r[idx], r_r[idx])
        keep = (converged & (t1 >= -_BOUNDS_TOL_M) & (t1 <= e1.length + _BOUNDS_TOL_M)
                & (t2 >= -_BOUNDS_TOL_M) & (t2 <= e2.length + _BOUNDS_TOL_M))
        if not keep.any():
            return ray
        idx, rx, t1, t2 = idx[keep], rx[keep], t1[keep], t2[keep]
        return self._double_diffraction_amplitude(ray, tx, tx_p, idx, rx, e1, e2, t1, t2, r_t, phi_t, k, amp_1m,
                                                  obstacles, want_vertices)

    @staticmethod
    def _parallel_double_keller(e1, e2, tx_p, rx):
        """
        Diffraction points on two parallel edges, in closed form: in the plane normal
        to the edges the path is fixed (Tx, edge 1, edge 2, Rx project to points), and
        along the edges the shortest path splits the advance in proportion to the
        normal distances r1, d12, r3 (the straight line of the unfolded path).
        """
        u = e1.axis
        sign = 1.0 if float(u @ e2.axis) > 0 else -1.0
        s_t = float((tx_p - e1.start) @ u)
        s_r = (rx - e1.start) @ u
        s_a2 = float((e2.start - e1.start) @ u)
        r1 = float(np.linalg.norm((tx_p - e1.start) - s_t * u))
        d12 = float(np.linalg.norm((e2.start - e1.start) - s_a2 * u))
        r3 = np.linalg.norm((rx - e2.start) - ((rx - e2.start) @ u)[:, None] * u, axis=-1)
        total = r1 + d12 + r3
        with np.errstate(divide="ignore", invalid="ignore"):
            s_q1 = s_t + (s_r - s_t) * (r1 / total)
            s_q2 = s_t + (s_r - s_t) * ((r1 + d12) / total)
        t1 = s_q1
        t2 = sign * (s_q2 - s_a2)
        converged = np.isfinite(t1) & np.isfinite(t2) & (total > 0)
        return t1, t2, converged

    @staticmethod
    def _newton_double_keller(e1, e2, tx_p, rx, r_t, s_r, r_r):
        """Diffraction points on two skew edges: Newton's method on the (convex) path length."""
        u1, u2 = e1.axis, e2.axis
        # Initial guess: Keller point on edge i towards the middle of edge j, then on edge j from there
        mid = 0.5 * (e2.start + e2.end)
        s_m, r_m, _ = (float(v) for v in e1.cylindrical(mid))
        s_t = float((tx_p - e1.start) @ u1)
        t1 = np.full(len(rx), s_t + (s_m - s_t) * r_t / (r_t + r_m))
        q1 = e1.start + t1[:, None] * u1
        s_q1, r_q1, _ = e2.cylindrical(q1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t2 = np.where(r_q1 + r_r > 0, s_q1 + (s_r - s_q1) * r_q1 / (r_q1 + r_r), 0.5 * e2.length)
        scale = e1.length + e2.length + 1.0
        cos12 = float(u1 @ u2)
        converged = np.zeros(len(rx), dtype=bool)
        # Each point is iterated until its own gradient reaches round-off (or its step stops shrinking)
        # and then frozen, so its result does not depend on the other points of the batch: vectorized and
        # point-by-point runs agree exactly, and both directions of a path meet at the same points.
        # A point is accepted when its gradient is below 1e-10 by then.
        busy = np.arange(len(rx))
        last_step = np.full(len(rx), np.inf)
        # By components (the arrays are long, the vectors 3-D): Q1 = A1 + t1 u1, Q2 = A2 + t2 u2
        (a1x, a1y, a1z), (u1x, u1y, u1z) = e1.start, u1
        (a2x, a2y, a2z), (u2x, u2y, u2z) = e2.start, u2
        tx_x, tx_y, tx_z = (float(v) for v in tx_p)
        rx_x, rx_y, rx_z = rx[:, 0], rx[:, 1], rx[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            for iteration in range(41):
                a, b = t1[busy], t2[busy]
                q1x, q1y, q1z = a1x + a * u1x, a1y + a * u1y, a1z + a * u1z
                q2x, q2y, q2z = a2x + b * u2x, a2y + b * u2y, a2z + b * u2z
                d1x, d1y, d1z = q1x - tx_x, q1y - tx_y, q1z - tx_z
                d2x, d2y, d2z = q2x - q1x, q2y - q1y, q2z - q1z
                d3x, d3y, d3z = rx_x[busy] - q2x, rx_y[busy] - q2y, rx_z[busy] - q2z
                l1 = np.sqrt(d1x * d1x + d1y * d1y + d1z * d1z)
                l2 = np.sqrt(d2x * d2x + d2y * d2y + d2z * d2z)
                l3 = np.sqrt(d3x * d3x + d3y * d3y + d3z * d3z)
                c1a = (d1x * u1x + d1y * u1y + d1z * u1z) / l1
                c1b = (d2x * u1x + d2y * u1y + d2z * u1z) / l2
                c2a = (d2x * u2x + d2y * u2y + d2z * u2z) / l2
                c2b = (d3x * u2x + d3y * u2y + d3z * u2z) / l3
                g1 = c1a - c1b
                g2 = c2a - c2b
                grad = np.abs(g1) + np.abs(g2)
                ok = (grad < 1e-10) & np.isfinite(l2)
                done = (ok & ((grad < 1e-14) | (last_step[busy] < 1e-12 * scale))) | (iteration == 40)
                converged[busy[done]] = ok[done]
                go = ~done
                if not go.any():
                    break
                if not go.all():
                    busy, a, b = busy[go], a[go], b[go]
                    g1, g2, l1, l2, l3 = g1[go], g2[go], l1[go], l2[go], l3[go]
                    c1a, c1b, c2a, c2b = c1a[go], c1b[go], c2a[go], c2b[go]
                h11 = (1.0 - c1a * c1a) / l1 + (1.0 - c1b * c1b) / l2
                h22 = (1.0 - c2a * c2a) / l2 + (1.0 - c2b * c2b) / l3
                h12 = -(cos12 - c1b * c2a) / l2
                det = h11 * h22 - h12 * h12
                good = np.isfinite(det) & (det > 1e-14 * np.maximum(h11 * h22, 1e-300))
                dt1 = np.where(good, -(h22 * g1 - h12 * g2) / det, -g1 / np.maximum(h11, 1e-12))
                dt2 = np.where(good, -(h11 * g2 - h12 * g1) / det, -g2 / np.maximum(h22, 1e-12))
                dt1 = np.clip(np.nan_to_num(dt1), -scale, scale)
                dt2 = np.clip(np.nan_to_num(dt2), -scale, scale)
                last_step[busy] = np.abs(dt1) + np.abs(dt2)
                t1[busy] = a + dt1
                t2[busy] = b + dt2
        return t1, t2, converged

    def _double_diffraction_amplitude(self, ray, tx, tx_p, idx, rx, e1, e2, t1, t2, r_t, phi_t, k, amp_1m,
                                      obstacles, want_vertices) -> _Rays:
        tol = OBSTRUCTION_TOL_M
        u1, u2 = e1.axis, e2.axis
        q1 = e1.start + t1[:, None] * u1
        q2 = e2.start + t2[:, None] * u2
        _, r_q2, phi_q2 = e1.cylindrical(q2)           # leaving edge i towards Q2
        _, r_q1, phi_q1 = e2.cylindrical(q1)           # reaching edge j from Q1
        keep = ((r_q2 > tol) & (phi_q2 <= e1.exterior_angle + _ANGLE_TOL)
                & (r_q1 > tol) & (phi_q1 <= e2.exterior_angle + _ANGLE_TOL))
        if not keep.any():
            return ray
        idx, rx, q1, q2, phi_q2, phi_q1 = idx[keep], rx[keep], q1[keep], q2[keep], phi_q2[keep], phi_q1[keep]
        _, r_rx, phi_rx = e2.cylindrical(rx)
        l1 = np.linalg.norm(q1 - tx_p, axis=-1)
        l2 = np.linalg.norm(q2 - q1, axis=-1)
        l3 = np.linalg.norm(rx - q2, axis=-1)
        # Only where the ray is in the shadow or transition zone of both edges: elsewhere the ray diffracted
        # by one of them alone reaches the point with the full field (symmetric, so reciprocal)
        zone = (self._transition_zone(phi_q1, phi_rx, e2.wedge_n, k * l2 * l3 * (r_rx / l3) ** 2 / (l2 + l3))
                & self._transition_zone(phi_t, phi_q2, e1.wedge_n, k * l1 * l2 * (r_t / l1) ** 2 / (l1 + l2)))
        # Soft polarization along an opaque face (two corners of the same facade): the coefficient of that
        # edge is exactly 0 (the field tangent to the face vanishes; with the face coefficient -1 the four
        # UTD terms cancel in pairs), so the ray carries no power and is left out instead of being kept
        # as a round-off residue of ~1e-36
        along_i = (phi_q2 < 1e-6) | (phi_q2 > e1.exterior_angle - 1e-6)
        along_j = (phi_q1 < 1e-6) | (phi_q1 > e2.exterior_angle - 1e-6)
        if e1.polarization == "soft" and e1.isb_scale == 1.0:
            zone &= ~along_i
        if e2.polarization == "soft" and e2.isb_scale == 1.0:
            zone &= ~along_j
        if not zone.any():
            return ray
        idx, rx, q1, q2, phi_q2, phi_q1, r_rx, phi_rx, l1, l2, l3 = (
            v[zone] for v in (idx, rx, q1, q2, phi_q2, phi_q1, r_rx, phi_rx, l1, l2, l3))
        d_1 = self._utd(e1, phi_t, phi_q2, k, l1, l2, r_t / l1, tx.frequency_hz, grazing_leg="out")
        d_2 = self._utd(e2, phi_q1, phi_rx, k, l2, l3, r_rx / l3, tx.frequency_hz, grazing_leg="in")
        grazing = (phi_q1 < 1e-6) | (phi_q1 > e2.exterior_angle - 1e-6)
        d_2 = np.where(grazing, 0.5 * d_2, d_2)
        total = l1 + l2 + l3
        amp = amp_1m * d_1 * d_2 / np.sqrt(l1 * l2 * l3 * total) * np.exp(-1j * k * total)
        return self._finish(ray, tx, tx_p, idx, amp, total, q1, q2, rx, obstacles, [q1, q2], [None, None],
                            want_vertices)

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
                interactions=(),
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
                interactions=(("ground", -1),),
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
                interactions=(("wall", wall_index),),
            ))

        # --- 4. Diffraction by edges and 5. double reflections (new in 1.5.0) ---
        if self.edges or self.max_reflections >= 2:
            for _, extra in self._extra_rays(tx, rx_p[None, :], k, amp_1m, sector, want_vertices=True):
                if not extra.idx.size:
                    continue
                tx_phi_e, tx_theta_e, rx_phi_e, rx_theta_e = (float(v[0]) for v in extra.angles)
                cir.add_ray(MultipathRay(
                    tx_id=tx.tx_id,
                    magnitude=complex(extra.amplitude[0]),
                    tau_ns=float(extra.tau_ns[0]),
                    tx_phi_deg=tx_phi_e,
                    tx_theta_deg=tx_theta_e,
                    rx_phi_deg=rx_phi_e,
                    rx_theta_deg=rx_theta_e,
                    obstruction_loss_db=float(extra.loss_db[0]),
                    kind=extra.kind,
                    path=(tuple(tx_p.tolist()),) + tuple(tuple(v[0].tolist()) for v in extra.vertices)
                         + (tuple(rx_p.tolist()),),
                    interactions=extra.interactions,
                ))

        return cir

    def trace_points(self, tx: Transmitter, points: np.ndarray, compact: bool = False) -> RayField:
        """
        Vectorized equivalent of ``trace_paths`` for an array of receiver points.

        points: array of shape [..., 3] with the receiver coordinates (x, y, z).
        Returns a RayField whose arrays have shape [K, ...], K following
        ``candidate_rays()`` (``RayField.labels`` says which ray each row is).
        compact=True (new in 1.6.0) leaves out the two-interaction candidates that
        cannot exist for this transmitter (see ``_active_higher_order``): fewer rows,
        the same rays and the same totals.
        Obstruction by walls with a ``transmission_loss_db`` is applied exactly as in
        ``trace_paths``; ``RayField.obstruction_loss_db`` holds the loss of each ray.
        """
        pts = np.asarray(points, dtype=float)
        if pts.shape[-1] != 3:
            raise ValueError("points must have shape [..., 3]")
        shape = pts.shape[:-1]
        n_first = 2 + len(self.reflectors)
        labels = self.candidate_rays()
        row_of = None
        if compact:
            active = self._active_higher_order(tx)
            n_fixed = n_first + len(self.edges)
            row_of = {position: n_fixed + row for row, (position, _) in enumerate(active)}
            labels = labels[:n_fixed] + [labels[position] for position, _ in active]
        n_rays = len(labels)

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

        # --- 4. Diffraction by edges and 5. two-interaction rays (new in 1.5.0 / 1.6.0) ---
        if n_rays > n_first:
            flat = pts.reshape(-1, 3)
            for position, extra in self._extra_rays(tx, flat, k, amp_1m, sector, want_vertices=False):
                if not extra.idx.size:
                    continue
                r = position if row_of is None or position < n_first + len(self.edges) else row_of[position]
                amplitude[r].reshape(-1)[extra.idx] = extra.amplitude
                tau_ns[r].reshape(-1)[extra.idx] = extra.tau_ns
                for a, values in enumerate(extra.angles):
                    angles[a, r].reshape(-1)[extra.idx] = values
                valid[r].reshape(-1)[extra.idx] = True
                loss_db[r].reshape(-1)[extra.idx] = extra.loss_db

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
            labels=labels,
        )
