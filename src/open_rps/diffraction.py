"""
OpenRPS - Diffraction by wedges: Uniform Theory of Diffraction (UTD)

Building blocks of the diffracted rays of the ray tracer (new in 1.5.0), written
with numpy only (no scipy) and vectorized:

* ``fresnel_integrals(t)``: C(t) and S(t), C(t) = int_0^t cos(pi s^2 / 2) ds.
* ``fresnel_tail(t)``: int_t^inf exp(-j pi s^2 / 2) ds, computed without the
  cancellation of 1/2 - C(t) for large t.
* ``utd_transition(X)``: the transition function of Kouyoumjian and Pathak,
  F(X) = 2j sqrt(X) exp(jX) int_sqrt(X)^inf exp(-j tau^2) dtau.
* ``utd_coefficient(...)``: diffraction coefficient of a wedge of exterior angle
  n*pi (Kouyoumjian & Pathak 1974), with the heuristic reflection coefficients of
  Luebbers (1984) for lossy faces.
* ``knife_edge_field(nu)``: Fresnel-Kirchhoff field of a single knife edge
  (ITU-R P.526), the classic reference the UTD is compared with in the tests.

Time convention exp(+j w t): a ray of length d carries the phase exp(-j k d), as
everywhere in OpenRPS.
"""

import math

import numpy as np

# Below this argument the Fresnel integrals use their power series, above it the
# continued fraction of Lentz (Numerical Recipes, "frenel")
_SERIES_LIMIT = 1.5
_SERIES_TERMS = 40
_CF_ITERATIONS = 120
# Points within this angle (rad) of a shadow boundary are evaluated just off it (see utd_coefficient)
_BOUNDARY_TOL = 1e-9
_BOUNDARY_STEP = 1e-7


def _series(t: np.ndarray):
    """C(t), S(t) by their power series (|t| < 1.5): sum of x u^k / (k! (2k + 1)), u = pi t^2 / 2."""
    u = 0.5 * math.pi * t * t
    c = np.zeros_like(t)
    s = np.zeros_like(t)
    term = t.copy()                       # t * u^k / k!
    for k in range(_SERIES_TERMS):
        value = term / (2 * k + 1)
        sign = 1.0 if (k // 2) % 2 == 0 else -1.0
        if k % 2 == 0:
            c = c + sign * value
        else:
            s = s + sign * value
        term = term * u / (k + 1)
    return c, s


def _continued_fraction(t: np.ndarray) -> np.ndarray:
    """
    h' of the continued fraction (|t| >= 1.5):  (1/2 - C) + j (1/2 - S) = (1 + j)/2 * exp(j pi t^2 / 2) * h'.
    Lentz's method with a fixed number of iterations (it converges in a few tens).
    """
    pix2 = math.pi * t * t
    b = 1.0 - 1j * pix2
    cc = np.full(t.shape, 1e300, dtype=complex)
    d = 1.0 / b
    h = d.copy()
    n = -1
    for _ in range(_CF_ITERATIONS):
        n += 2
        a = -float(n * (n + 1))
        b = b + 4.0
        d = 1.0 / (a * d + b)
        cc = b + a / cc
        h = h * (cc * d)
    return (t - 1j * t) * h


def fresnel_integrals(t):
    """Fresnel integrals C(t) and S(t) (odd functions), float or array; relative error ~1e-13."""
    x = np.atleast_1d(np.asarray(t, dtype=float))
    ax = np.abs(x)
    c = np.empty_like(ax)
    s = np.empty_like(ax)
    small = ax < _SERIES_LIMIT
    if small.any():
        c[small], s[small] = _series(ax[small])
    if (~small).any():
        big = ax[~small]
        tail = (0.5 + 0.5j) * np.exp(0.5j * math.pi * big * big) * _continued_fraction(big)
        c[~small] = 0.5 - tail.real
        s[~small] = 0.5 - tail.imag
    c, s = np.sign(x) * c, np.sign(x) * s
    if np.ndim(t) == 0:
        return float(c[0]), float(s[0])
    return c.reshape(np.shape(t)), s.reshape(np.shape(t))


def fresnel_tail(t):
    """int_t^inf exp(-j pi s^2 / 2) ds for t >= 0 (at t = 0 it is (1 - j) / 2)."""
    x = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(x < 0):
        raise ValueError("fresnel_tail needs t >= 0")
    out = np.empty(x.shape, dtype=complex)
    small = x < _SERIES_LIMIT
    if small.any():
        c, s = _series(x[small])
        out[small] = (0.5 - c) - 1j * (0.5 - s)
    if (~small).any():
        big = x[~small]
        out[~small] = (0.5 - 0.5j) * np.exp(-0.5j * math.pi * big * big) * np.conj(_continued_fraction(big))
    return complex(out[0]) if np.ndim(t) == 0 else out.reshape(np.shape(t))


def utd_transition(X):
    """
    Transition function of the UTD, F(X) = 2j sqrt(X) exp(jX) int_sqrt(X)^inf exp(-j tau^2) dtau, X >= 0.
    F(0) = 0, F(X) -> 1 for large X, F(X) ~ sqrt(pi X) exp(j pi / 4) for small X.
    """
    x = np.atleast_1d(np.asarray(X, dtype=float))
    if np.any(x < 0):
        raise ValueError("utd_transition needs X >= 0")
    t = np.sqrt(2.0 * x / math.pi)                        # int_sqrt(X)^inf exp(-j tau^2) = sqrt(pi/2) fresnel_tail(t)
    out = np.empty(x.shape, dtype=complex)
    pref = 2j * np.sqrt(x) * math.sqrt(0.5 * math.pi)
    small = t < _SERIES_LIMIT
    if small.any():
        c, s = _series(t[small])
        tail = (0.5 - c) - 1j * (0.5 - s)
        out[small] = pref[small] * np.exp(1j * x[small]) * tail
    if (~small).any():
        # exp(jX) * tail = (1 - j)/2 * conj(h'): the large phase cancels exactly
        out[~small] = pref[~small] * (0.5 - 0.5j) * np.conj(_continued_fraction(t[~small]))
    return complex(out[0]) if np.ndim(X) == 0 else out.reshape(np.shape(X))


def knife_edge_field(nu):
    """
    Field of a single knife edge relative to free space (Fresnel-Kirchhoff, ITU-R P.526):
    E/E0 = (1 + j)/2 * int_nu^inf exp(-j pi t^2 / 2) dt.   |E/E0| = 0.5 (-6.02 dB) at nu = 0.
    """
    v = np.asarray(nu, dtype=float)
    tail = fresnel_tail(np.abs(v))
    integral = np.where(v >= 0, tail, (1.0 - 1.0j) - tail)
    out = (0.5 + 0.5j) * integral
    return complex(out) if np.ndim(nu) == 0 else out


def _cot_f(beta, sign: int, n: float, kL):
    """cot((pi + sign * beta) / (2n)) * F(kL a^sign(beta)), a^+-(beta) = 2 cos^2((2 n pi N^+- - beta) / 2)."""
    N = np.round((beta + sign * math.pi) / (2.0 * n * math.pi))
    a = 2.0 * np.cos((2.0 * n * math.pi * N - beta) / 2.0) ** 2
    angle = (math.pi + sign * beta) / (2.0 * n)
    return np.cos(angle) / np.sin(angle) * utd_transition(kL * a)


def utd_coefficient(phi, phi_prime, n: float, k: float, L, sin_beta0, r0=-1.0, rn=-1.0, isb_scale=1.0):
    """
    UTD diffraction coefficient of a wedge (Kouyoumjian & Pathak 1974) with the
    heuristic face reflection coefficients of Luebbers (1984):

        D = -exp(-j pi/4) / (2 n sqrt(2 pi k) sin(beta0)) * [ isb_scale * (T1 + T2) + r0 * T3 + rn * T4 ]
        T1 = cot((pi + (phi - phi'))/2n) F(kL a+(phi - phi'))     T2 = cot((pi - (phi - phi'))/2n) F(kL a-(phi - phi'))
        T3 = cot((pi - (phi + phi'))/2n) F(kL a-(phi + phi'))     T4 = cot((pi + (phi + phi'))/2n) F(kL a+(phi + phi'))

    phi_prime / phi: angles (rad) of the source and of the observer, measured from
    face 0 through the exterior of the wedge (0..n pi); n: exterior angle / pi (2 =
    half plane); L: distance parameter s s' sin^2(beta0) / (s + s').
    r0, rn: reflection coefficients of face 0 (for the grazing angle phi') and of
    face n (for n pi - phi): -1 / -1 is the perfectly conducting wedge for the
    "soft" polarization (E parallel to the edge), +1 / +1 for the "hard" one.
    isb_scale: factor of the two incident-shadow-boundary terms.  1 for an opaque
    wedge; 1 - tau for a wedge that lets an amplitude fraction tau of the incident
    field through, so that the total field stays continuous at the shadow boundary.

    Exactly at a shadow boundary the cotangent has a pole that F cancels; the
    coefficient is then evaluated 1e-7 rad off the boundary, on the shadow side of
    the incident field and on the lit side of the reflected fields (the side where
    the tracer, whose patches are closed, keeps or removes those rays).
    """
    phi = np.array(phi, dtype=float, copy=True)
    phi_p = np.asarray(phi_prime, dtype=float)
    # Nudge the observation angle off the shadow boundaries (see docstring)
    for boundary, step in ((math.pi, +1), (-math.pi, -1)):                 # incident shadow boundaries
        at = np.abs((phi - phi_p) - boundary) < _BOUNDARY_TOL
        phi = np.where(at, phi + step * _BOUNDARY_STEP, phi)
    for boundary, step in ((math.pi, -1), ((2.0 * n - 1.0) * math.pi, +1)):  # reflection boundaries of faces 0 and n
        at = np.abs((phi + phi_p) - boundary) < _BOUNDARY_TOL
        phi = np.where(at, phi + step * _BOUNDARY_STEP, phi)
    kL = k * np.asarray(L, dtype=float)
    beta_m = phi - phi_p
    beta_p = phi + phi_p
    t1 = _cot_f(beta_m, +1, n, kL)
    t2 = _cot_f(beta_m, -1, n, kL)
    t3 = _cot_f(beta_p, -1, n, kL)
    t4 = _cot_f(beta_p, +1, n, kL)
    factor = -np.exp(-0.25j * math.pi) / (2.0 * n * math.sqrt(2.0 * math.pi * k) * np.asarray(sin_beta0, dtype=float))
    d = factor * (isb_scale * (t1 + t2) + r0 * t3 + rn * t4)
    return complex(d) if np.ndim(d) == 0 else d
