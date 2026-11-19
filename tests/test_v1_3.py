"""Tests of OpenRPS 1.3.0: walls that obstruct rays (``PlanarReflector.transmission_loss_db``)."""
import cmath
import contextlib
import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
from unittest import mock

import numpy as np

from open_rps import (
    ChannelImpulseResponse,
    FreeSpacePathLoss,
    MapVisualizer,
    OkumuraHata,
    PlanarReflector,
    PropagationSimulator,
    RayTracer,
    Transmitter,
    TwoRayGroundReflection,
    cli,
    thermal_noise_dbm,
)
from open_rps.raytracer import OBSTRUCTION_TOL_M

INF = float("inf")
SVG = "{http://www.w3.org/2000/svg}"
AREA = ((0.0, 400.0), (0.0, 300.0))
RESULT_GRIDS = ("rx_power_dbm", "path_loss_db", "best_server_id", "best_server_index", "max_rx_power_dbm",
                "rms_delay_spread_ns", "angular_spread_bs_phi")


def make_tx(x=0.0, y=0.0, z=10.0, tx_id=0, **kwargs):
    kwargs = {"tx_power_dbm": 30.0, "frequency_hz": 2.4e9, "antenna_gain_dbi": 6.0, **kwargs}
    return Transmitter(tx_id=tx_id, x=x, y=y, z=z, **kwargs)


def make_tracer(*walls):
    tracer = RayTracer()
    for wall in walls:
        tracer.add_reflector(wall)
    return tracer


def wall_x(x, loss=None, **kwargs):
    """Vertical wall on the plane x = const."""
    return PlanarReflector(p0=[x, 0.0, 0.0], normal=[1.0, 0.0, 0.0], transmission_loss_db=loss, **kwargs)


def wall_y(y, loss=None, **kwargs):
    """Vertical wall on the plane y = const."""
    return PlanarReflector(p0=[0.0, y, 0.0], normal=[0.0, 1.0, 0.0], transmission_loss_db=loss, **kwargs)


def building(x0, x1, y0, y1, height, loss=INF, reflectivity=0.5):
    """Four finite vertical walls (no roof): the faces of a rectangular building."""
    return [
        PlanarReflector(p0=[x0, y0, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=reflectivity, name="West",
                        bounds=(x0, x0, y0, y1, 0.0, height), transmission_loss_db=loss),
        PlanarReflector(p0=[x1, y0, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=reflectivity, name="East",
                        bounds=(x1, x1, y0, y1, 0.0, height), transmission_loss_db=loss),
        PlanarReflector(p0=[x0, y0, 0.0], normal=[0.0, 1.0, 0.0], reflectivity=reflectivity, name="South",
                        bounds=(x0, x1, y0, y0, 0.0, height), transmission_loss_db=loss),
        PlanarReflector(p0=[x0, y1, 0.0], normal=[0.0, 1.0, 0.0], reflectivity=reflectivity, name="North",
                        bounds=(x0, x1, y1, y1, 0.0, height), transmission_loss_db=loss),
    ]


def db(power_ratio):
    return 10.0 * math.log10(power_ratio)


def assert_same_result(a, b, names=RESULT_GRIDS):
    for name in names:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def assert_close_result(test, a, b, tol=1e-9):
    """Same non-finite values in the same pixels, finite values within ``tol`` (dB, ns or degrees)."""
    for name in RESULT_GRIDS:
        x, y = np.asarray(getattr(a, name), dtype=float), np.asarray(getattr(b, name), dtype=float)
        np.testing.assert_array_equal(np.isnan(x), np.isnan(y), err_msg=name)
        np.testing.assert_array_equal(np.isinf(x), np.isinf(y), err_msg=name)
        finite = np.isfinite(x)
        if finite.any():
            test.assertLess(float(np.max(np.abs(x[finite] - y[finite]))), tol, name)


def segment_clip_box_2d(ax, ay, bx, by, x0, x1, y0, y1):
    """Liang-Barsky: (t_in, t_out) of the part of the segment a -> b inside the rectangle, or None."""
    t_in, t_out = 0.0, 1.0
    for p, q in ((-(bx - ax), ax - x0), (bx - ax, x1 - ax), (-(by - ay), ay - y0), (by - ay, y1 - ay)):
        if p == 0.0:
            if q < 0.0:
                return None
        elif p < 0.0:
            t_in = max(t_in, q / p)
        else:
            t_out = min(t_out, q / p)
    return (t_in, t_out) if t_in < t_out else None


class BothPathsMixin:
    """Trace one receiver point with trace_paths and with trace_points and check that they agree."""

    def trace(self, tracer, tx, rx):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cir = tracer.trace_paths(tx, rx)
            field = tracer.trace_points(tx, np.array(rx, dtype=float))
        existing = np.flatnonzero(field.valid)
        self.assertEqual(len(existing), len(cir.rays))
        for k, ray in zip(existing, cir.rays):
            self.assertEqual(ray.obstruction_loss_db, float(field.obstruction_loss_db[k]))
            self.assertAlmostEqual(abs(ray.magnitude - complex(field.amplitude[k])), 0.0,
                                   delta=1e-10 * max(abs(ray.magnitude), 1e-300))
            self.assertAlmostEqual(ray.tau_ns, float(field.tau_ns[k]), delta=1e-9)
            self.assertTrue(math.isfinite(ray.tau_ns) and math.isfinite(ray.tx_phi_deg))
            self.assertFalse(cmath.isnan(ray.magnitude))
        self.assertFalse(np.isnan(field.amplitude).any())
        self.assertFalse(np.isnan(field.obstruction_loss_db).any())
        return cir, field


# --------------------------------------------------------------------------------------------
class TransmissionLossParameterTests(unittest.TestCase):
    def test_default_is_no_obstruction(self):
        wall = PlanarReflector(p0=[0, 0, 0], normal=[1, 0, 0])
        self.assertIsNone(wall.transmission_loss_db)
        self.assertFalse(wall.obstructs)

    def test_accepted_values(self):
        for value, obstructs in ((0, False), (0.0, False), (3, True), (12.5, True), (INF, True), (np.float64(7.0), True)):
            wall = wall_x(0.0, value)
            self.assertIsInstance(wall.transmission_loss_db, float)
            self.assertEqual(wall.transmission_loss_db, float(value))
            self.assertEqual(wall.obstructs, obstructs)

    def test_invalid_values_raise_value_error(self):
        for value in (-0.1, -3, float("nan"), -INF, "abc", [1.0]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "transmission_loss_db"):
                wall_x(0.0, value)

    def test_other_parameters_are_still_validated_and_independent(self):
        wall = PlanarReflector(p0=[1, 2, 3], normal=[0, 2, 0], reflectivity=0.25, bounds=(0, 5, 2, 2, 0, 9),
                               transmission_loss_db=6.0)
        self.assertEqual((wall.reflectivity, wall.bounds, wall.transmission_loss_db), (0.25, (0, 5, 2, 2, 0, 9), 6.0))
        np.testing.assert_array_equal(wall.normal, [0.0, 1.0, 0.0])
        self.assertEqual(wall.reflection_coefficient(0.5, 2.4e9), complex(-0.25))
        with self.assertRaises(ValueError):
            PlanarReflector(p0=[0, 0, 0], normal=[1, 0, 0], reflectivity=1.5, transmission_loss_db=6.0)


# --------------------------------------------------------------------------------------------
class SegmentWallCrossingTests(unittest.TestCase):
    """Geometry of ``PlanarReflector.crosses`` (the rule used to decide which rays a wall obstructs)."""

    def test_infinite_plane(self):
        wall = wall_x(50.0)
        self.assertTrue(wall.crosses([0, 0, 1], [100, 0, 1]))
        self.assertTrue(wall.crosses([100, 7, 3], [0, -4000, 900]))        # both directions, anywhere on the plane
        self.assertFalse(wall.crosses([0, 0, 1], [49, 500, 1]))            # same side
        self.assertFalse(wall.crosses([51, 0, 1], [300, 0, 1]))
        self.assertFalse(wall.crosses([10, 0, 1], [10, 80, 40]))           # parallel to the plane
        self.assertFalse(wall.crosses([50, 0, 1], [50, 80, 40]))           # lying in the plane
        self.assertFalse(wall.crosses([3, 3, 3], [3, 3, 3]))               # zero-length segment

    def test_segment_is_open_ends_on_the_plane_do_not_cross(self):
        wall = wall_x(50.0)
        self.assertFalse(wall.crosses([0, 0, 1], [50, 0, 1]))              # ends on the plane
        self.assertFalse(wall.crosses([50, 0, 1], [100, 0, 1]))            # starts on the plane
        self.assertEqual(OBSTRUCTION_TOL_M, 1e-6)
        self.assertFalse(wall.crosses([0, 0, 1], [50 + 0.5e-6, 0, 1]))     # closer than the tolerance: still "on" it
        self.assertFalse(wall.crosses([50 - 0.5e-6, 0, 1], [100, 0, 1]))
        self.assertTrue(wall.crosses([0, 0, 1], [50 + 2e-6, 0, 1]))        # just beyond the tolerance
        self.assertTrue(wall.crosses([50 - 2e-6, 0, 1], [100, 0, 1]))
        self.assertTrue(wall.crosses([0, 0, 1], [50.001, 0, 1], tol=1e-4))
        self.assertFalse(wall.crosses([0, 0, 1], [50.001, 0, 1], tol=1e-2))

    def test_finite_wall_needs_the_crossing_point_inside_the_patch(self):
        wall = wall_x(50.0, bounds=(50, 50, -10, 10, 0, 20))
        self.assertTrue(wall.crosses([0, 0, 5], [100, 0, 5]))
        self.assertTrue(wall.crosses([0, -30, 0], [100, 40, 30]))          # hits (50, 5, 15)
        self.assertFalse(wall.crosses([0, 0, 5], [100, 30, 5]))            # hits (50, 15, 5): beside the wall
        self.assertFalse(wall.crosses([0, 0, 30], [100, 0, 30]))           # passes over the wall
        self.assertFalse(wall.crosses([0, 0, 50], [100, 0, 1]))            # hits (50, 0, 25.5): over the top
        self.assertTrue(wall.crosses([0, 0, 30], [100, 0, 1]))             # hits (50, 0, 15.5)
        self.assertFalse(wall.crosses([0, 0, 5], [40, 0, 5]))              # inside the box in y, z but same side

    def test_patch_is_closed_a_crossing_on_its_edge_counts(self):
        wall = wall_x(50.0, bounds=(50, 50, -10, 10, 0, 20))
        self.assertTrue(wall.crosses([0, 10, 5], [100, 10, 5]))            # exactly on the side edge y = 10
        self.assertTrue(wall.crosses([0, 0, 20], [100, 0, 20]))            # exactly on the top edge z = 20
        self.assertTrue(wall.crosses([0, 10, 20], [100, 10, 20]))          # exactly on the corner
        self.assertFalse(wall.crosses([0, 10 + 1e-6, 5], [100, 10 + 1e-6, 5]))     # one micrometer outside
        self.assertFalse(wall.crosses([0, 0, 20 + 1e-6], [100, 0, 20 + 1e-6]))
        self.assertTrue(wall.crosses([0, 10 - 1e-6, 5], [100, 10 - 1e-6, 5]))

    def test_oblique_wall_and_signed_distance(self):
        wall = PlanarReflector(p0=[10.0, 0.0, 0.0], normal=[3.0, 4.0, 0.0])            # 3x + 4y = 30
        np.testing.assert_allclose(wall.signed_distance([[10, 0, 0], [0, 0, 7], [10, 5, -2], [13, 4, 0]]),
                                   [0.0, -6.0, 4.0, 5.0], atol=1e-12)
        self.assertEqual(float(wall.signed_distance(np.array([0.0, 0.0, 0.0]))), -6.0)
        self.assertTrue(wall.crosses([0, 0, 0], [10, 5, 0]))
        self.assertFalse(wall.crosses([0, 0, 0], [5, 3, 9]))               # 3*5 + 4*3 = 27 < 30
        self.assertFalse(wall.crosses([0, 7.5, 0], [10, 0, 0]))            # along the wall: both ends on the plane
        finite = PlanarReflector(p0=[10.0, 0.0, 0.0], normal=[3.0, 4.0, 0.0], bounds=(2, 10, 0, 6, 0, 10))
        self.assertTrue(finite.crosses([0, 0, 1], [12, 6, 1]))             # crosses at (6, 3, 1)
        self.assertFalse(finite.crosses([0, 0, 1], [1.2, 12, 1]))          # crosses at (0.652, 6.52, 1): outside in x, y
        self.assertFalse(finite.crosses([0, 0, 11], [12, 6, 11]))          # above the patch

    def test_broadcasting_and_shapes(self):
        wall = wall_x(50.0, bounds=(50, 50, -10, 10, 0, 20))
        ends = np.array([[100.0, 0, 5], [100, 30, 5], [40, 0, 5], [50, 0, 5]])
        np.testing.assert_array_equal(wall.crosses([0, 0, 5], ends), [True, False, False, False])
        np.testing.assert_array_equal(wall.crosses(ends, [0, 0, 5]), [True, False, False, False])
        grid = np.zeros((3, 4, 3))
        grid[..., 0] = 100.0
        grid[..., 1] = np.arange(4) * 6.0                                    # y = 0, 6, 12, 18 -> crossing at y / 2
        grid[..., 2] = np.array([1.0, 30.0, 60.0])[:, None]                  # crossing at (5 + z) / 2
        result = wall.crosses(np.array([0.0, 0.0, 5.0]), grid)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_equal(result, [[True] * 4, [True] * 4, [False] * 4])
        single = wall.crosses(np.array([0.0, 0.0, 5.0]), np.array([100.0, 0.0, 5.0]))
        self.assertEqual(np.shape(single), ())
        with self.assertRaises(ValueError):
            wall.crosses([0, 0], [1, 1])

    def test_random_segments_against_a_plain_python_reference(self):
        rng = np.random.default_rng(42)
        crossings = {False: 0, True: 0}                 # per kind of wall (infinite / finite)
        for n in range(40):
            p0 = rng.uniform(-50, 50, size=3)
            normal = rng.normal(size=3)
            bounds = None
            if n % 2:
                half = rng.uniform(5, 60, size=3)
                bounds = tuple(float(v) for axis in range(3) for v in (p0[axis] - half[axis], p0[axis] + half[axis]))
            wall = PlanarReflector(p0=p0, normal=normal, bounds=bounds)
            a = rng.uniform(-100, 100, size=(60, 3))
            b = rng.uniform(-100, 100, size=(60, 3))
            expected = []
            for start, end in zip(a, b):
                s_a = float(np.dot(start - wall.p0, wall.normal))
                s_b = float(np.dot(end - wall.p0, wall.normal))
                crossing = s_a * s_b < 0.0
                if crossing and bounds is not None:
                    hit = start + s_a / (s_a - s_b) * (end - start)
                    crossing = all(bounds[2 * i] <= hit[i] <= bounds[2 * i + 1] for i in range(3))
                expected.append(crossing)
            np.testing.assert_array_equal(wall.crosses(a, b), expected)
            self.assertLess(sum(expected), 60)
            crossings[bounds is not None] += sum(expected)
        self.assertGreater(crossings[False], 300)
        self.assertGreater(crossings[True], 50)

    def test_ray_tracer_segment_loss(self):
        tracer = make_tracer(wall_x(50.0, 7.0), wall_x(60.0, 5.0, bounds=(60, 60, -10, 10, 0, 20)),
                             wall_x(70.0), wall_x(80.0, 0.0), wall_x(200.0, INF))
        self.assertEqual(float(tracer.obstruction_loss_db([0, 0, 5], [100, 0, 5])), 12.0)
        self.assertEqual(float(tracer.obstruction_loss_db([0, 50, 5], [100, 50, 5])), 7.0)     # beside the finite wall
        self.assertEqual(float(tracer.obstruction_loss_db([0, 0, 5], [40, 0, 5])), 0.0)        # line of sight
        self.assertEqual(float(tracer.obstruction_loss_db([0, 0, 5], [300, 0, 5])), INF)
        np.testing.assert_array_equal(
            tracer.obstruction_loss_db([0, 0, 5], [[55, 0, 5], [65, 0, 5], [250, 90, 5]]), [7.0, 12.0, INF])
        self.assertEqual(float(make_tracer(wall_x(50.0)).obstruction_loss_db([0, 0, 5], [100, 0, 5])), 0.0)


# --------------------------------------------------------------------------------------------
class ObstructedRayTests(BothPathsMixin, unittest.TestCase):
    """Per-ray effect of the walls, on both tracer implementations."""

    TX = make_tx(0.0, 0.0, 10.0)
    RX = (100.0, 10.0, 1.5)

    def scene(self, loss):
        """
        A raised screen between Tx and Rx (x = 50, |y| <= 10, 5 <= z <= 30) and a side wall (y = 40) that only
        reflects.  The direct ray hits the screen at (50, 5, 5.75).  The ground ray bounces at x = 86.96 and
        passes under the screen (z = 4.25 at x = 50).  The ray reflected on the side wall (at x = 57.1) passes
        beside it (y = 35 at x = 50).
        """
        screen = wall_x(50.0, loss, bounds=(50, 50, -10, 10, 5, 30), reflectivity=0.5)
        side = wall_y(40.0, reflectivity=0.7)
        return make_tracer(screen, side)

    def test_opaque_wall_removes_the_direct_ray_and_leaves_the_others_untouched(self):
        clear, _ = self.trace(self.scene(None), self.TX, self.RX)
        cir, field = self.trace(self.scene(INF), self.TX, self.RX)
        self.assertEqual(len(clear.rays), 3)                      # direct, ground, side wall (no ray off the screen)
        self.assertEqual(len(cir.rays), 3)                        # obstruction never changes the list of rays
        direct, ground, side = cir.rays
        self.assertEqual(direct.magnitude, 0j)
        self.assertEqual(direct.power_linear, 0.0)
        self.assertEqual(direct.obstruction_loss_db, INF)
        self.assertEqual((direct.tau_ns, direct.tx_phi_deg, direct.rx_theta_deg),
                         (clear.rays[0].tau_ns, clear.rays[0].tx_phi_deg, clear.rays[0].rx_theta_deg))
        for kept, reference in ((ground, clear.rays[1]), (side, clear.rays[2])):
            self.assertEqual(kept, reference)                     # bit-identical ray, loss 0
            self.assertEqual(kept.obstruction_loss_db, 0.0)
        # Every metric of the CIR is the one of the two surviving rays
        survivors = ChannelImpulseResponse(rays=[clear.rays[1], clear.rays[2]])
        self.assertEqual(cir.total_received_power_linear(), survivors.total_received_power_linear())
        self.assertEqual(cir.total_received_power_linear(coherent=True),
                         survivors.total_received_power_linear(coherent=True))
        self.assertEqual(cir.rms_delay_spread(), survivors.rms_delay_spread())
        self.assertEqual(cir.mean_excess_delay(), survivors.mean_excess_delay())
        self.assertEqual(cir.mean_excess_delay(relative_to_first=True),
                         survivors.mean_excess_delay(relative_to_first=True))     # first arrival = ground ray
        self.assertEqual(cir.angular_spread(), survivors.angular_spread())
        self.assertLess(cir.total_received_power_dbm(), clear.total_received_power_dbm() - 3.0)
        # Vectorized field: amplitude exactly 0, the ray still "exists" geometrically
        np.testing.assert_array_equal(field.valid, [True, True, False, True])
        self.assertEqual(complex(field.amplitude[0]), 0j)
        np.testing.assert_array_equal(field.obstruction_loss_db, [INF, 0.0, 0.0, 0.0])
        self.assertTrue(np.isfinite(field.tau_ns[0]))
        self.assertAlmostEqual(float(field.total_power_linear()), survivors.total_received_power_linear(), delta=1e-18)

    def test_finite_loss_attenuates_exactly_l_db_without_touching_the_phase(self):
        clear, clear_field = self.trace(self.scene(None), self.TX, self.RX)
        for loss in (0.5, 3.0, 10.0, 20.0, 37.25, 150.0):
            with self.subTest(loss=loss):
                cir, field = self.trace(self.scene(loss), self.TX, self.RX)
                self.assertEqual([r.obstruction_loss_db for r in cir.rays], [loss, 0.0, 0.0])
                self.assertAlmostEqual(db(clear.rays[0].power_linear / cir.rays[0].power_linear), loss, delta=1e-9)
                ratio = cir.rays[0].magnitude / clear.rays[0].magnitude
                self.assertAlmostEqual(ratio.real, 10.0 ** (-loss / 20.0), delta=1e-15)
                self.assertAlmostEqual(ratio.imag, 0.0, delta=1e-15)                 # amplitude only, no phase
                self.assertEqual(cir.rays[1:], clear.rays[1:])
                self.assertEqual(cir.rays[0].tau_ns, clear.rays[0].tau_ns)
                ratio_field = complex(field.amplitude[0]) / complex(clear_field.amplitude[0])
                self.assertAlmostEqual(-20.0 * math.log10(abs(ratio_field)), loss, delta=1e-9)
                self.assertAlmostEqual(cmath.phase(ratio_field), 0.0, delta=1e-12)
                # Coherent sum = complex sum of the attenuated amplitudes
                expected = abs(clear.rays[0].magnitude * 10.0 ** (-loss / 20.0) + clear.rays[1].magnitude +
                               clear.rays[2].magnitude) ** 2
                self.assertAlmostEqual(cir.total_received_power_linear(coherent=True), expected, delta=1e-12 * expected)
                self.assertAlmostEqual(float(field.total_power_linear(coherent=True)), expected, delta=1e-9 * expected)

    def test_zero_db_is_the_same_as_no_obstruction(self):
        clear, clear_field = self.trace(self.scene(None), self.TX, self.RX)
        cir, field = self.trace(self.scene(0.0), self.TX, self.RX)
        self.assertEqual(cir.rays, clear.rays)
        np.testing.assert_array_equal(field.amplitude, clear_field.amplitude)
        np.testing.assert_array_equal(field.obstruction_loss_db, 0.0)

    def test_losses_of_several_walls_add_up_in_db(self):
        tx, rx = make_tx(0.0, 0.0, 10.0), (100.0, 0.0, 10.0)
        clear, _ = self.trace(make_tracer(), tx, rx)
        cases = [((7.0, 5.0), 12.0), ((7.0, 5.0, 2.5, 0.25), 14.75), ((7.0, None, 5.0), 12.0), ((7.0, 0.0), 7.0),
                 ((7.0, INF), INF), ((INF, INF), INF)]
        for losses, total in cases:
            with self.subTest(losses=losses):
                walls = [wall_x(20.0 + 10.0 * n, loss, reflectivity=0.0) for n, loss in enumerate(losses)]
                cir, _ = self.trace(make_tracer(*walls), tx, rx)
                direct = cir.rays[0]
                self.assertEqual(direct.obstruction_loss_db, total)
                if math.isinf(total):
                    self.assertEqual(direct.power_linear, 0.0)
                else:
                    self.assertAlmostEqual(db(clear.rays[0].power_linear / direct.power_linear), total, delta=1e-9)
                # The ground ray crosses the same vertical planes: same loss
                self.assertEqual(cir.rays[1].obstruction_loss_db, total)

    def test_reflected_ray_crossing_the_same_wall_twice_loses_twice(self):
        """Tx and Rx west of a 6 dB wall (x = 50); a mirror at x = 100: the reflected ray goes through and back."""
        tx, rx = make_tx(0.0, 0.0, 10.0), (20.0, 30.0, 10.0)
        clear, _ = self.trace(make_tracer(wall_x(50.0, None, reflectivity=0.0), wall_x(100.0, reflectivity=0.8)), tx, rx)
        cir, _ = self.trace(make_tracer(wall_x(50.0, 6.0, reflectivity=0.0), wall_x(100.0, reflectivity=0.8)), tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, 0.0, 0.0, 12.0])
        self.assertEqual(cir.rays[:3], clear.rays[:3])
        self.assertAlmostEqual(db(clear.rays[3].power_linear / cir.rays[3].power_linear), 12.0, delta=1e-9)

    def test_a_wall_never_obstructs_its_own_reflection(self):
        tx, rx = make_tx(0.0, 0.0, 10.0), (30.0, 40.0, 1.5)
        clear, clear_field = self.trace(make_tracer(wall_x(50.0, None, reflectivity=0.6)), tx, rx)
        for loss in (9.0, INF):
            cir, field = self.trace(make_tracer(wall_x(50.0, loss, reflectivity=0.6)), tx, rx)
            self.assertEqual(len(cir.rays), 3)
            self.assertEqual(cir.rays, clear.rays)                          # nothing changes, not a single bit
            np.testing.assert_array_equal(field.amplitude, clear_field.amplitude)
            self.assertGreater(cir.rays[2].power_linear, 0.0)
        # A second, coplanar patch of the same wall does not block the reflection point either
        halves = [wall_x(50.0, INF, reflectivity=0.6, bounds=(50, 50, -1000, 20, 0, 100)),
                  wall_x(50.0, INF, reflectivity=0.6, bounds=(50, 50, 20, 1000, 0, 100))]
        cir, _ = self.trace(make_tracer(*halves), tx, rx)                   # reflection point at y = 23.5: second patch
        self.assertEqual(len(cir.rays), 3)
        self.assertEqual(cir.rays[2].obstruction_loss_db, 0.0)
        self.assertAlmostEqual(cir.rays[2].power_linear, clear.rays[2].power_linear, delta=1e-18)

    def test_a_reflected_ray_is_obstructed_by_another_wall(self):
        """Mirror at y = 40; an opaque screen at x = 20 stands in the first leg of the reflected ray only."""
        tx, rx = make_tx(0.0, 0.0, 10.0), (100.0, 10.0, 10.0)
        mirror = wall_y(40.0, reflectivity=0.7)
        # Reflection point: t = 40 / (40 + 30) -> (57.14, 40).  First leg at x = 20: y = 14.  Direct ray at x = 20: y = 2.
        clear, _ = self.trace(make_tracer(mirror, wall_x(20.0, None, reflectivity=0.0, bounds=(20, 20, 8, 20, 0, 30))), tx, rx)
        for loss in (4.0, INF):
            screen = wall_x(20.0, loss, reflectivity=0.0, bounds=(20, 20, 8, 20, 0, 30))
            cir, _ = self.trace(make_tracer(mirror, screen), tx, rx)
            self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, 0.0, loss])
            self.assertEqual(cir.rays[:2], clear.rays[:2])
            if math.isinf(loss):
                self.assertEqual(cir.rays[2].power_linear, 0.0)
            else:
                self.assertAlmostEqual(db(clear.rays[2].power_linear / cir.rays[2].power_linear), loss, delta=1e-9)
        # Screen in the second leg instead (x = 80: second leg at y = 24, direct ray at y = 8)
        screen = wall_x(80.0, INF, reflectivity=0.0, bounds=(80, 80, 20, 30, 0, 30))
        cir, _ = self.trace(make_tracer(mirror, screen), tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, 0.0, INF])
        # An obstructing mirror is itself crossed by the rays of other walls: mirror 2 at y = -40 beyond wall y = -20
        tracer = make_tracer(wall_y(-40.0, reflectivity=0.7), wall_y(-20.0, 3.0, reflectivity=0.5))
        cir, _ = self.trace(tracer, tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, 0.0, 6.0, 0.0])

    def test_walls_that_no_ray_crosses_change_nothing(self):
        tx, rx = make_tx(0.0, 0.0, 10.0), (100.0, 10.0, 1.5)
        cases = {
            "behind the receiver": lambda loss: [wall_x(150.0, loss)],
            "behind the transmitter": lambda loss: [wall_x(-10.0, loss)],
            "parallel to the path, beside it": lambda loss: [wall_y(25.0, loss), wall_y(-5.0, loss)],
            "finite wall in the plane between them, out of the way": lambda loss: [
                wall_x(50.0, loss, bounds=(50, 50, 30, 90, 0, 40))],
            "roof above everything": lambda loss: [
                PlanarReflector(p0=[0, 0, 30.0], normal=[0, 0, 1.0], transmission_loss_db=loss)],
        }
        for name, walls in cases.items():
            with self.subTest(name):
                clear, clear_field = self.trace(make_tracer(*walls(None)), tx, rx)
                for loss in (15.0, INF):
                    cir, field = self.trace(make_tracer(*walls(loss)), tx, rx)
                    self.assertEqual(cir.rays, clear.rays)
                    np.testing.assert_array_equal(field.amplitude, clear_field.amplitude)
                    np.testing.assert_array_equal(field.tau_ns, clear_field.tau_ns)
                    self.assertFalse(field.obstruction_loss_db.any())

    def test_everything_blocked_gives_no_power_and_nan_metrics_without_warnings(self):
        tx, rx = make_tx(0.0, 0.0, 10.0), (100.0, 10.0, 1.5)
        cir, field = self.trace(make_tracer(wall_x(50.0, INF)), tx, rx)
        self.assertEqual(len(cir.rays), 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(cir.total_received_power_linear(), 0.0)
            self.assertEqual(cir.total_received_power_linear(coherent=True), 0.0)
            self.assertEqual(cir.total_received_power_dbm(), -INF)
            self.assertEqual(cir.total_received_power_dbm(coherent=True), -INF)
            self.assertTrue(math.isnan(cir.rms_delay_spread()))
            self.assertTrue(math.isnan(cir.mean_excess_delay()))
            self.assertTrue(math.isnan(cir.mean_excess_delay(relative_to_first=True)))
            self.assertTrue(all(math.isnan(v) for v in cir.angular_spread()))
            self.assertEqual(float(field.total_power_linear()), 0.0)
            self.assertEqual(float(field.total_power_linear(coherent=True)), 0.0)

    def test_sector_pattern_and_obstruction_combine(self):
        tx = make_tx(0.0, 0.0, 10.0, pattern="sector", azimuth_deg=30.0, downtilt_deg=3.0)
        clear, _ = self.trace(self.scene(None), tx, self.RX)
        cir, _ = self.trace(self.scene(8.0), tx, self.RX)
        self.assertAlmostEqual(db(clear.rays[0].power_linear / cir.rays[0].power_linear), 8.0, delta=1e-9)
        self.assertEqual(cir.rays[1:], clear.rays[1:])


# --------------------------------------------------------------------------------------------
class EdgeCaseTests(BothPathsMixin, unittest.TestCase):
    def test_transmitter_on_the_plane_of_a_wall_is_not_obstructed_by_it(self):
        wall = wall_x(0.0, INF)
        for tx_x in (0.0, 0.4e-6, -0.4e-6):                     # exactly on the plane or closer than the tolerance
            tx = make_tx(tx_x, 0.0, 10.0)
            for rx in ((50.0, 5.0, 1.5), (-50.0, 5.0, 1.5), (0.0, 30.0, 1.5)):
                cir, _ = self.trace(make_tracer(wall), tx, rx)
                self.assertEqual([r.obstruction_loss_db for r in cir.rays[:2]], [0.0, 0.0])
                self.assertEqual(cir.rays[:2], make_tracer().trace_paths(tx, rx).rays)     # as if there were no wall
                self.assertTrue(math.isfinite(cir.total_received_power_dbm()))
        # One hundredth of a millimeter off the plane it is on one side: the other side is blocked
        tx = make_tx(-1e-5, 0.0, 10.0)
        cir, _ = self.trace(make_tracer(wall), tx, (50.0, 5.0, 1.5))
        self.assertEqual(cir.total_received_power_dbm(), -INF)
        cir, _ = self.trace(make_tracer(wall), tx, (-50.0, 5.0, 1.5))
        self.assertEqual(cir.rays[0].obstruction_loss_db, 0.0)

    def test_receiver_on_the_plane_of_a_wall_is_not_obstructed_by_it(self):
        tx = make_tx(0.0, 0.0, 10.0)
        clear, _ = self.trace(make_tracer(wall_x(50.0)), tx, (50.0, 5.0, 1.5))
        cir, _ = self.trace(make_tracer(wall_x(50.0, INF)), tx, (50.0, 5.0, 1.5))
        self.assertEqual(cir.rays, clear.rays)
        self.assertEqual(len(cir.rays), 2)                       # and no reflection: the receiver is on the wall
        # ... but it is obstructed by any other wall in between
        cir, _ = self.trace(make_tracer(wall_x(50.0, INF), wall_x(20.0, 4.0)), tx, (50.0, 5.0, 1.5))
        self.assertEqual([r.obstruction_loss_db for r in cir.rays[:2]], [4.0, 4.0])

    def test_rays_parallel_to_a_wall_or_lying_in_its_plane(self):
        tx = make_tx(0.0, 0.0, 10.0)
        in_plane = wall_y(0.0, INF)                              # Tx, Rx and the whole path lie in the wall plane
        cir, _ = self.trace(make_tracer(in_plane), tx, (100.0, 0.0, 10.0))
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, 0.0])
        parallel = wall_y(5.0, INF)                              # path parallel to the wall, 5 m away
        cir, _ = self.trace(make_tracer(parallel), tx, (100.0, 0.0, 10.0))
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, 0.0, 0.0])
        ceiling = PlanarReflector(p0=[0, 0, 10.0], normal=[0, 0, 1.0], transmission_loss_db=INF)
        cir, _ = self.trace(make_tracer(ceiling), tx, (100.0, 0.0, 10.0))      # both antennas on a horizontal plane
        self.assertEqual(cir.rays[0].obstruction_loss_db, 0.0)

    def test_degenerate_geometries_do_not_crash(self):
        walls = [wall_x(50.0, INF), wall_y(20.0, 6.0, bounds=(0, 80, 20, 20, 0, 5)),
                 PlanarReflector(p0=[0, 0, 0], normal=[0, 0, 1], transmission_loss_db=INF)]     # a "wall" on the ground
        tracer = make_tracer(*walls)
        for tx_z, rx in ((10.0, (0.0, 0.0, 10.0)),             # receiver on the antenna: no direct ray
                         (0.0, (100.0, 10.0, 1.5)),            # antenna on the ground: the bounce point is the antenna
                         (10.0, (100.0, 10.0, 0.0)),           # receiver on the ground
                         (0.0, (100.0, 10.0, 0.0)),            # both on the ground: no ground ray
                         (10.0, (0.0, 0.0, 1.5)),              # receiver right below the antenna
                         (10.0, (50.0, 20.0, 0.0))):           # receiver on the corner of two walls and the ground
            with self.subTest(tx_z=tx_z, rx=rx):
                cir, field = self.trace(tracer, make_tx(0.0, 0.0, tx_z), rx)
                self.assertFalse(math.isnan(cir.total_received_power_linear()))
                self.assertFalse(np.isnan(field.total_power_linear(coherent=True)).any())

    def test_ground_bounce_exactly_at_the_foot_of_a_wall(self):
        """Tx and Rx at the same height, wall half-way: the ground ray bends on the wall plane and must not leak."""
        tx, rx = make_tx(0.0, 0.0, 20.0), (100.0, 0.0, 20.0)
        clear, _ = self.trace(make_tracer(), tx, rx)
        cir, _ = self.trace(make_tracer(wall_x(50.0, INF, reflectivity=0.0)), tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [INF, INF])
        cir, _ = self.trace(make_tracer(wall_x(50.0, 9.0, reflectivity=0.0)), tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [9.0, 9.0])          # crossed once, not twice
        self.assertAlmostEqual(db(clear.rays[1].power_linear / cir.rays[1].power_linear), 9.0, delta=1e-9)
        # A wall raised 1 m above the ground lets the ground ray through under it; the direct ray is still blocked
        raised = wall_x(50.0, INF, reflectivity=0.0, bounds=(50, 50, -1e3, 1e3, 1.0, 1e3))
        cir, _ = self.trace(make_tracer(raised), tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [INF, 0.0])
        # A finite wall standing on the ground blocks it as well (the patch is probed just above the bounce point)
        standing = wall_x(50.0, INF, reflectivity=0.0, bounds=(50, 50, -5, 5, 0.0, 3.0))
        cir, _ = self.trace(make_tracer(standing), tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, INF])          # (the direct ray passes above)
        beside = wall_x(50.0, INF, reflectivity=0.0, bounds=(50, 50, 2, 9, 0.0, 3.0))
        cir, _ = self.trace(make_tracer(beside), tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, 0.0])
        # Bounce at the foot with Tx and Rx on the same side (wall beyond both): the ray only touches it
        cir, _ = self.trace(make_tracer(wall_y(0.0, INF, reflectivity=0.0)), tx, rx)
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [0.0, 0.0])

    def test_reflection_exactly_on_the_corner_of_a_building(self):
        """Tx on the diagonal of a square building: the ray to its NW corner must not enter through the edge."""
        tx = make_tx(300.0, 300.0, 25.0)
        rx_inside = (500.0, 400.0, 1.5)
        for loss, expected in ((INF, [INF, INF, INF, INF]), (10.0, [10.0, 10.0, 10.0, 10.0])):
            with self.subTest(loss=loss):
                cir, _ = self.trace(make_tracer(*building(450.0, 600.0, 300.0, 450.0, 50.0, loss)), tx, rx_inside)
                # Rays: direct, ground, east face (from inside), north face (reflection point = the NW corner)
                self.assertEqual(len(cir.rays), 4)
                self.assertEqual([r.obstruction_loss_db for r in cir.rays], expected)
        cir, _ = self.trace(make_tracer(*building(450.0, 600.0, 300.0, 450.0, 50.0, INF)), tx, rx_inside)
        self.assertEqual(cir.total_received_power_dbm(), -INF)

    def test_reflection_on_the_end_of_a_wall_that_only_grazes_the_next_one(self):
        """
        Reflection on the west face of a building exactly at its NW corner, towards the outside: the ray touches
        the end of the north face without going through it (the wall is not between the two legs of the ray).
        """
        tx = make_tx(200.0, 500.0, 25.0)                       # image across x = 450: (700, 500)
        rx = (200.0, 700.0, 1.5)                               # image -> rx crosses x = 450 at y = 600: the corner
        clear, _ = self.trace(make_tracer(*building(450.0, 550.0, 400.0, 600.0, 40.0, None)), tx, rx)
        cir, _ = self.trace(make_tracer(*building(450.0, 550.0, 400.0, 600.0, 40.0, INF)), tx, rx)
        self.assertEqual(len(cir.rays), 4)                     # direct, ground, west face, east face (from inside)
        self.assertEqual(cir.rays[:3], clear.rays[:3])
        self.assertEqual(cir.rays[2].obstruction_loss_db, 0.0)
        self.assertGreater(cir.rays[2].power_linear, 0.0)
        self.assertEqual(cir.rays[3].obstruction_loss_db, INF)   # this one had to go through the west face
        # Same corner, but now the north face is the mirror (inner side) and the ray ends inside the building:
        # Tx (350, 500) -> image (350, 700) -> reflection at (450, 600) -> Rx (500, 550).  The west face is between the legs.
        cir, _ = self.trace(make_tracer(*building(450.0, 550.0, 400.0, 600.0, 40.0, INF)), make_tx(350.0, 500.0, 25.0),
                            (500.0, 550.0, 1.5))
        self.assertEqual(len(cir.rays), 4)                     # direct, ground, east face, north face
        self.assertEqual([r.obstruction_loss_db for r in cir.rays], [INF] * 4)
        self.assertEqual(cir.total_received_power_dbm(), -INF)

    def test_direct_ray_through_the_edge_shared_by_two_walls_crosses_both(self):
        tx = make_tx(300.0, 300.0, 25.0)
        tracer = make_tracer(*building(450.0, 600.0, 300.0, 450.0, 50.0, 10.0))
        cir, _ = self.trace(tracer, tx, (600.0, 600.0, 25.0))            # passes exactly through the corner (450, 450)
        self.assertEqual(cir.rays[0].obstruction_loss_db, 20.0)
        cir, _ = self.trace(tracer, tx, (600.0, 601.0, 25.0))            # a hair to the north: misses the building
        self.assertEqual(cir.rays[0].obstruction_loss_db, 0.0)
        cir, _ = self.trace(tracer, tx, (600.0, 599.0, 25.0))            # a hair to the south: in by the west, out by the north
        self.assertEqual(cir.rays[0].obstruction_loss_db, 20.0)


# --------------------------------------------------------------------------------------------
def obstructed_sim(sector=False, seed=0):
    """Two transmitters and six oblique / finite walls with every kind of transmission loss."""
    rng = np.random.default_rng(seed)
    sim = PropagationSimulator()
    antenna = dict(pattern="sector", azimuth_deg=40.0, downtilt_deg=4.0) if sector else {}
    sim.add_transmitter(Transmitter(tx_id=3, x=60.0, y=70.0, z=25.0, **antenna))
    sim.add_transmitter(Transmitter(tx_id=8, x=330.0, y=210.0, z=12.0, tx_power_dbm=40.0, **antenna))
    losses = [INF, 10.0, 3.5, None, INF, 20.0]
    for n, loss in enumerate(losses):
        p0 = rng.uniform([50, 50, 0], [350, 250, 20])
        normal = rng.normal(size=3) * [1.0, 1.0, 0.3]
        bounds = None
        if n % 2 == 0:
            half = rng.uniform(20, 80, size=3)
            bounds = (p0[0] - half[0], p0[0] + half[0], p0[1] - half[1], p0[1] + half[1], 0.0, 15.0 + half[2])
        sim.add_reflector(PlanarReflector(p0=p0, normal=normal, reflectivity=0.5, bounds=bounds,
                                          transmission_loss_db=loss))
    return sim


class VectorizedVsPointwiseTests(unittest.TestCase):
    def test_grid_simulation_with_oblique_and_finite_walls(self):
        blocked_pixels = 0
        for seed in (0, 3, 4):
            for coherent in (False, True):
                with self.subTest(seed=seed, coherent=coherent):
                    sim = obstructed_sim(sector=(seed == 3), seed=seed)
                    options = dict(resolution_m=16.0, coherent=coherent)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        fast = sim.run_raytracing_simulation(*AREA, **options)
                        slow = sim.run_raytracing_simulation(*AREA, vectorized=False, **options)
                    np.testing.assert_array_equal(fast.best_server_id, slow.best_server_id)
                    assert_close_result(self, fast, slow, tol=1e-9)
                    blocked_pixels += int(np.isneginf(fast.rx_power_dbm).sum())
                    # The scenario really obstructs: it differs from the same walls without transmission loss
                    for wall in sim.ray_tracer.reflectors:
                        wall.transmission_loss_db = None
                    clear = sim.run_raytracing_simulation(*AREA, **options)
                    self.assertTrue(np.isfinite(clear.rx_power_dbm).all())
                    self.assertGreater(float(np.nanmax(np.abs(clear.rx_power_dbm - fast.rx_power_dbm))), 10.0)
        self.assertGreater(blocked_pixels, 200)          # pixels where every ray of a transmitter is blocked

    def test_grid_aligned_scene_full_of_degenerate_rays(self):
        """
        Building and wall aligned with the grid, a station on the building diagonal and in the plane of one face,
        receivers at the height of the antennas: many pixels lie on wall planes and many ground rays bounce exactly
        at the foot of a wall.  Both implementations must take the same decision in every one of them.
        """
        for loss in (INF, 10.0):
            with self.subTest(loss=loss):
                sim = PropagationSimulator()
                sim.add_transmitter(Transmitter(tx_id=0, x=300.0, y=300.0, z=25.0))
                sim.add_transmitter(Transmitter(tx_id=1, x=700.0, y=700.0, z=25.0, tx_power_dbm=40.0))
                for wall in building(450.0, 600.0, 300.0, 450.0, 50.0, loss):
                    sim.add_reflector(wall)
                sim.add_reflector(wall_x(500.0, loss, reflectivity=0.3))
                area = ((0.0, 1000.0), (0.0, 1000.0))
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    fast = sim.run_raytracing_simulation(*area, resolution_m=50.0, rx_height_m=25.0)
                    slow = sim.run_raytracing_simulation(*area, resolution_m=50.0, rx_height_m=25.0, vectorized=False)
                assert_close_result(self, fast, slow)
                np.testing.assert_array_equal(fast.best_server_id, slow.best_server_id)
                # The scene is as degenerate as intended
                points = np.empty(fast.max_rx_power_dbm.shape + (3,))
                points[..., 0], points[..., 1], points[..., 2] = fast.x_coords[None, :], fast.y_coords[:, None], 25.0
                on_a_plane = sum(int((np.abs(wall.signed_distance(points)) <= OBSTRUCTION_TOL_M).sum())
                                 for wall in sim.ray_tracer.reflectors)
                bounce = points * [1.0, 1.0, 0.0] + (np.array([300.0, 300.0, 0.0]) - points * [1.0, 1.0, 0.0]) / 2.0
                bends = sum(int((np.abs(wall.signed_distance(bounce)) <= OBSTRUCTION_TOL_M).sum())
                            for wall in sim.ray_tracer.reflectors)
                self.assertGreater(on_a_plane, 50)
                self.assertGreater(bends, 50)
                if math.isinf(loss):
                    self.assertGreater(int((fast.best_server_id < 0).sum()), 3)

    def test_with_shadow_fading_minimum_distance_and_sinr(self):
        sim = obstructed_sim(seed=0)
        sim.add_transmitter(Transmitter(tx_id=5, x=200.0, y=150.0, z=1.5))       # on a grid node, at receiver height
        options = dict(resolution_m=10.0, shadow_fading_sigma_db=5.0, seed=11)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fast = sim.run_raytracing_simulation(*AREA, **options)
            slow = sim.run_raytracing_simulation(*AREA, vectorized=False, **options)
            sinr_fast, sinr_slow = fast.compute_sinr(), slow.compute_sinr()
        assert_close_result(self, fast, slow, tol=1e-9)
        np.testing.assert_array_equal(np.isnan(sinr_fast), np.isnan(sinr_slow))
        ok = np.isfinite(sinr_fast)
        self.assertLess(float(np.max(np.abs(sinr_fast[ok] - sinr_slow[ok]))), 1e-9)
        self.assertTrue(np.isnan(fast.rx_power_dbm[2, 15, 20]))                  # min_distance_m still applies
        self.assertTrue(np.isneginf(fast.rx_power_dbm).any())                    # shadow fading keeps -inf as -inf

    def test_ray_field_matches_the_cir_ray_by_ray(self):
        sim = obstructed_sim(seed=4)
        tracer = sim.ray_tracer
        points = np.random.default_rng(9).uniform([0, 0, 0.5], [400, 300, 30], size=(300, 3))
        crossed = 0
        for tx in sim.transmitters:
            field = tracer.trace_points(tx, points)
            self.assertEqual(field.obstruction_loss_db.shape, field.amplitude.shape)
            for n, point in enumerate(points):
                cir = tracer.trace_paths(tx, tuple(point))
                existing = np.flatnonzero(field.valid[:, n])
                self.assertEqual(len(existing), len(cir.rays))
                for k, ray in zip(existing, cir.rays):
                    self.assertEqual(ray.obstruction_loss_db, field.obstruction_loss_db[k, n])
                    self.assertAlmostEqual(abs(ray.magnitude - field.amplitude[k, n]), 0.0,
                                           delta=1e-10 * max(abs(ray.magnitude), 1e-300))
                    crossed += ray.obstruction_loss_db > 0.0
            # Where a ray does not exist the loss is 0 and the amplitude 0
            self.assertFalse(field.obstruction_loss_db[~field.valid].any())
            # The loss of the direct ray is the loss of the straight segment Tx -> Rx
            np.testing.assert_array_equal(field.obstruction_loss_db[0],
                                          tracer.obstruction_loss_db([tx.x, tx.y, tx.z], points))
        self.assertGreater(crossed, 300)

    def test_chunked_evaluation_gives_the_same_grid(self):
        sim = obstructed_sim(seed=3)
        whole = sim.run_raytracing_simulation(*AREA, resolution_m=10.0)
        with mock.patch("open_rps.simulation._MAX_CHUNK_POINTS", 150):
            chunked = sim.run_raytracing_simulation(*AREA, resolution_m=10.0)
        assert_same_result(whole, chunked)


# --------------------------------------------------------------------------------------------
# Scenario without any transmission loss, written with the 1.2.0 API only: it is evaluated by this version
# and, optionally, by a reference copy of OpenRPS 1.2.0 in another process (see RegressionAgainst120Tests).
REFERENCE_SCENARIO = r'''
import warnings
import numpy as np
from open_rps import (PlanarReflector, PropagationSimulator, Transmitter, FreeSpacePathLoss,
                      TwoRayGroundReflection, OkumuraHata)

def reference_walls():
    return [dict(p0=[200.0, 150.0, 0.0], normal=[1.0, 0.2, 0.0], reflectivity=0.6),
            dict(p0=[250.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=0.4,
                 bounds=(250.0, 250.0, 50.0, 180.0, 0.0, 30.0)),
            dict(p0=[0.0, 260.0, 0.0], normal=[0.1, 1.0, 0.05], eps_r=5.0, sigma=0.01),
            dict(p0=[0.0, 0.0, 40.0], normal=[0.0, 0.0, 1.0], reflectivity=0.3,
                 bounds=(50.0, 350.0, 50.0, 250.0, 40.0, 40.0))]

def reference_sim(wall_options=None, **tx_options):
    sim = PropagationSimulator()
    sim.add_transmitter(Transmitter(tx_id=0, x=100.0, y=100.0, z=25.0, tx_power_dbm=43.0,
                                    frequency_hz=2.4e9, antenna_gain_dbi=12.0, **tx_options))
    sim.add_transmitter(Transmitter(tx_id=1, x=300.0, y=200.0, z=20.0, tx_power_dbm=40.0,
                                    frequency_hz=1.8e9, antenna_gain_dbi=10.0, **tx_options))
    for wall in reference_walls():
        sim.add_reflector(PlanarReflector(**wall, **(wall_options or {})))
    return sim

def reference_arrays(wall_options=None):
    """Every array that 1.2.0 produced for the scenario, in a dict name -> ndarray."""
    area = ((0.0, 400.0), (0.0, 300.0))
    fields = ("rx_power_dbm", "path_loss_db", "best_server_id", "best_server_index", "max_rx_power_dbm",
              "rms_delay_spread_ns", "angular_spread_bs_phi")
    out = {}
    def put(name, result):
        for field in fields:
            out[name + "/" + field] = np.asarray(getattr(result, field))
        out[name + "/sinr"], out[name + "/i_plus_n"] = result.sinr_map()
    runs = (("default", dict(resolution_m=5.0)), ("coherent", dict(resolution_m=5.0, coherent=True)),
            ("shadow", dict(resolution_m=5.0, shadow_fading_sigma_db=6.0, seed=3)),
            ("ground_level", dict(resolution_m=5.0, rx_height_m=0.0, min_distance_m=None)),
            ("pointwise", dict(resolution_m=20.0, vectorized=False)),
            ("pointwise_coherent", dict(resolution_m=20.0, vectorized=False, coherent=True)))
    for name, options in runs:
        put(name, reference_sim(wall_options).run_raytracing_simulation(*area, **options))
    put("sector", reference_sim(wall_options, pattern="sector", azimuth_deg=30.0, downtilt_deg=5.0)
        .run_raytracing_simulation(*area, resolution_m=5.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model in (FreeSpacePathLoss(), TwoRayGroundReflection(), OkumuraHata("suburban")):
            put("empirical_" + type(model).__name__,
                reference_sim(wall_options).run_empirical_simulation(model, *area, resolution_m=10.0))
    sim = reference_sim(wall_options)
    points = np.random.default_rng(5).uniform([0, 0, 0], [400, 300, 30], size=(400, 3))
    for t, tx in enumerate(sim.transmitters):
        rays = sim.ray_tracer.trace_points(tx, points)
        for field in ("amplitude", "tau_ns", "tx_phi_deg", "tx_theta_deg", "rx_phi_deg", "rx_theta_deg", "valid"):
            out["field%d/%s" % (t, field)] = getattr(rays, field)
        values = []
        for point in points[:80]:
            cir = sim.ray_tracer.trace_paths(tx, tuple(point))
            for ray in cir.rays:
                values += [ray.magnitude.real, ray.magnitude.imag, ray.tau_ns, ray.tx_phi_deg, ray.tx_theta_deg,
                           ray.rx_phi_deg, ray.rx_theta_deg]
            values += [cir.rms_delay_spread(), cir.mean_excess_delay(), cir.total_received_power_linear(coherent=True)]
        out["cir%d" % t] = np.array(values)
    return out
'''
_reference = {}
exec(REFERENCE_SCENARIO, _reference)
reference_sim, reference_arrays = _reference["reference_sim"], _reference["reference_arrays"]
reference_walls = _reference["reference_walls"]


def _wall_plane_pixels(name, shape):
    """
    Mask (broadcast to ``shape``) of the grid pixels of the run ``name`` of ``reference_arrays``
    whose receiver lies within OBSTRUCTION_TOL_M of the plane of a reference wall; None for the
    arrays that are not ray-tracing grids (random points, CIRs, empirical models).
    """
    run = name.split("/")[0]
    resolution = {"default": 5.0, "coherent": 5.0, "shadow": 5.0, "ground_level": 5.0, "sector": 5.0,
                  "pointwise": 20.0, "pointwise_coherent": 20.0}.get(run)
    if resolution is None:
        return None
    xs, ys = np.arange(0.0, 400.0 + 1e-5, resolution), np.arange(0.0, 300.0 + 1e-5, resolution)
    gx, gy = np.meshgrid(xs, ys)
    points = np.stack([gx, gy, np.full(gx.shape, 0.0 if run == "ground_level" else 1.5)], axis=-1)
    mask = np.zeros(gx.shape, dtype=bool)
    for wall in reference_walls():
        mask |= np.abs(PlanarReflector(**wall).signed_distance(points)) <= OBSTRUCTION_TOL_M
    return np.broadcast_to(mask, shape)


class RegressionAgainst120Tests(unittest.TestCase):
    """Without ``transmission_loss_db`` nothing changes with respect to OpenRPS 1.2.0."""
    PIXELS = [(0, 0), (5, 5), (10, 15), (15, 20), (7, 13), (3, 9)]

    def test_reference_values_computed_with_1_2_0(self):
        r = reference_sim().run_raytracing_simulation(*AREA, resolution_m=20.0)
        expected_power = [
            [-27.375381310400453, -11.344541173775845, -30.59784604054353,
             -34.79212242929275, -28.69998011194937, -23.444338419876743],
            [-36.489789102304485, -32.877285893036714, -11.81308381399193,
             -29.839087424325193, -23.851716014299228, -31.684558539860927]]
        expected_ds = [210.87693338383662, 51.944361419355815, 35.119159808545405,
                       147.8246895622708, 79.04398570896537, 116.68526277233713]
        expected_as = [37.08366570987445, 2.327102615235318, 11.731179140209075,
                       29.834370893162102, 27.74497098602377, 10.636335996205103]
        for n, (j, i) in enumerate(self.PIXELS):
            for t in range(2):
                self.assertAlmostEqual(r.rx_power_dbm[t, j, i], expected_power[t][n], delta=1e-10)
            self.assertAlmostEqual(r.rms_delay_spread_ns[j, i], expected_ds[n], delta=1e-9)
            self.assertAlmostEqual(r.angular_spread_bs_phi[j, i], expected_as[n], delta=1e-9)
        self.assertAlmostEqual(float(r.rx_power_dbm.sum()), -19174.47797661359, delta=1e-7)
        self.assertAlmostEqual(float(r.rms_delay_spread_ns.sum()), 36709.913990143454, delta=1e-6)
        self.assertAlmostEqual(float(r.angular_spread_bs_phi.sum()), 6978.908830863562, delta=1e-7)
        self.assertEqual(int(r.best_server_id.sum()), 150)
        coherent = reference_sim().run_raytracing_simulation(*AREA, resolution_m=20.0, coherent=True)
        expected_coherent = [-27.746167048699512, -9.493675177558487, -27.697205821567824,
                             -31.91810963604383, -28.245963963260174, -32.97398613043542]
        for n, (j, i) in enumerate(self.PIXELS):
            self.assertAlmostEqual(coherent.rx_power_dbm[0, j, i], expected_coherent[n], delta=1e-6)
        self.assertEqual(int(coherent.best_server_id.sum()), 155)
        cir = reference_sim().ray_tracer.trace_paths(reference_sim().transmitters[0], (150.0, 120.0, 1.5))
        self.assertEqual(len(cir.rays), 6)
        self.assertAlmostEqual(cir.rays[0].tau_ns, 195.98842200525783, delta=1e-10)
        self.assertAlmostEqual(abs(cir.rays[0].magnitude), abs(-0.06608351891929463 - 0.06843958416221688j), delta=1e-15)
        self.assertAlmostEqual(abs(cir.rays[5].magnitude), abs(0.007314121247669794 - 0.020845553484046313j), delta=1e-15)
        self.assertEqual([ray.obstruction_loss_db for ray in cir.rays], [0.0] * 6)

    def test_none_and_zero_db_are_bit_identical_to_the_default(self):
        base = reference_arrays()
        for options in (dict(transmission_loss_db=None), dict(transmission_loss_db=0.0)):
            other = reference_arrays(options)
            self.assertEqual(sorted(base), sorted(other))
            for name in base:
                self.assertEqual(base[name].tobytes(), other[name].tobytes(), f"{options}: {name}")

    def test_obstructing_walls_that_are_never_crossed_are_bit_identical(self):
        """Opaque walls outside the scene switch the obstruction code on; no number may move."""
        def run(extra_loss, **options):
            sim = reference_sim()
            sim.add_reflector(PlanarReflector(p0=[-500.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=0.2,
                                              transmission_loss_db=extra_loss))
            sim.add_reflector(PlanarReflector(p0=[0.0, 0.0, 300.0], normal=[0.0, 0.0, 1.0], reflectivity=0.0,
                                              bounds=(0.0, 400.0, 0.0, 300.0, 300.0, 300.0),
                                              transmission_loss_db=extra_loss))
            return sim.run_raytracing_simulation(*AREA, **{"resolution_m": 10.0, **options})
        for options in (dict(), dict(coherent=True), dict(vectorized=False, resolution_m=25.0),
                        dict(shadow_fading_sigma_db=4.0, seed=2)):
            with self.subTest(options=options):
                assert_same_result(run(None, **options), run(INF, **options))
                assert_same_result(run(None, **options), run(25.0, **options))

    @unittest.skipUnless(os.environ.get("OPEN_RPS_REFERENCE_SRC"),
                         "set OPEN_RPS_REFERENCE_SRC to the src/ folder of an OpenRPS 1.2.0 checkout "
                         "to compare every array bit by bit against it")
    def test_bit_identical_to_a_reference_checkout_of_1_2_0(self):
        reference_src = os.environ["OPEN_RPS_REFERENCE_SRC"]
        script = (REFERENCE_SCENARIO + "\nimport sys, open_rps\n"
                  "assert open_rps.__version__ == '1.2.0', open_rps.__version__\n"
                  "np.savez(sys.argv[1], **reference_arrays())\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reference.npz")
            env = dict(os.environ, PYTHONPATH=reference_src)
            subprocess.run([sys.executable, "-c", script, path], check=True, env=env, cwd=tmp)
            with np.load(path) as data:
                old = {name: data[name] for name in data.files}
        new = reference_arrays()
        self.assertEqual(sorted(old), sorted(new))
        self.assertGreater(sum(a.size for a in new.values()), 300_000)
        # 1.4.0 fixed one degenerate case: a receiver lying on the plane of a wall (within 1 um) no
        # longer gets a reflection from that wall (1.2.0 / 1.3.0 gave one or not depending on the
        # rounding of its signed distance, and it duplicated the direct ray).  Those pixels are
        # excluded; every other number must still be bit-identical to 1.2.0.
        masked_pixels = 0
        for name in new:
            self.assertEqual(old[name].dtype, new[name].dtype, name)
            a, b = old[name], new[name]
            mask = _wall_plane_pixels(name, a.shape)
            if mask is not None:
                masked_pixels += int(mask.sum())
                a, b = np.where(mask, 0, a), np.where(mask, 0, b)
            self.assertEqual(a.tobytes(), b.tobytes(), name)
        self.assertGreater(masked_pixels, 0)

    def test_empirical_models_ignore_the_obstruction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for model in (FreeSpacePathLoss(), TwoRayGroundReflection(), OkumuraHata("suburban")):
                base = reference_sim().run_empirical_simulation(model, *AREA, resolution_m=20.0)
                opaque = reference_sim(dict(transmission_loss_db=INF)).run_empirical_simulation(
                    model, *AREA, resolution_m=20.0)
                assert_same_result(base, opaque)
                self.assertTrue(np.isfinite(opaque.rx_power_dbm).all())


# --------------------------------------------------------------------------------------------
class BuildingShadowTests(unittest.TestCase):
    """A rectangular building made of four opaque finite walls casts a geometric shadow."""
    BOX = (118.6, 161.2, 72.4, 131.8)        # x0, x1, y0, y1 (off the grid nodes, so no pixel lies on a face)
    TX_XY = (37.3, 61.7)

    def grid(self, step=4.0, rx_height=1.5):
        xs, ys = np.arange(0.0, 300.0 + 1e-9, step), np.arange(0.0, 200.0 + 1e-9, step)
        points = np.empty((len(ys), len(xs), 3))
        points[..., 0], points[..., 1], points[..., 2] = xs[None, :], ys[:, None], rx_height
        return xs, ys, points

    def shadow_mask(self, xs, ys, tx_z, rx_z, height):
        """Independent geometry: the direct ray is blocked if it leaves (or enters) the footprint below the roof."""
        mask = np.zeros((len(ys), len(xs)), dtype=bool)
        inside = np.zeros_like(mask)
        x0, x1, y0, y1 = self.BOX
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                clip = segment_clip_box_2d(self.TX_XY[0], self.TX_XY[1], x, y, x0, x1, y0, y1)
                inside[j, i] = x0 < x < x1 and y0 < y < y1
                if clip is None:
                    continue
                # Height of the ray where it crosses the faces: it descends from tx_z to rx_z
                z_in, z_out = (tx_z + t * (rx_z - tx_z) for t in clip)
                mask[j, i] = z_in <= height if inside[j, i] else (z_in <= height or z_out <= height)
        return mask, inside

    def test_shadow_of_a_tall_building_matches_the_geometry(self):
        tx = make_tx(*self.TX_XY, 25.0)
        xs, ys, points = self.grid()
        tracer = make_tracer(*building(*self.BOX, height=40.0))
        clear_tracer = make_tracer(*building(*self.BOX, height=40.0, loss=None))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            field, clear = tracer.trace_points(tx, points), clear_tracer.trace_points(tx, points)
        shadow, inside = self.shadow_mask(xs, ys, 25.0, 1.5, 40.0)
        direct_power = field.power_linear[0]
        self.assertGreater(int(shadow.sum()), 500)
        self.assertGreater(int((~shadow).sum()), 1500)
        self.assertGreater(int(inside.sum()), 100)
        # Direct ray: no power anywhere in the shadow (behind the building and inside it) ...
        np.testing.assert_array_equal(direct_power == 0.0, shadow)
        np.testing.assert_array_equal(np.isinf(field.obstruction_loss_db[0]), shadow)
        self.assertTrue(shadow[inside].all())
        # ... and bit-identical to the unobstructed ray outside it
        np.testing.assert_array_equal(field.amplitude[0][~shadow], clear.amplitude[0][~shadow])
        self.assertTrue((direct_power[~shadow] > 0.0).all())
        # The building is taller than the antenna: the ground ray is shadowed in the same pixels
        np.testing.assert_array_equal(field.power_linear[1] == 0.0, shadow)
        # Nothing at all gets inside (every ray would have to cross a face)
        self.assertEqual(float(field.total_power_linear()[inside].max()), 0.0)
        # Lit side: the reflection off the west face survives where the direct ray does
        west = field.power_linear[2]
        self.assertGreater(int((west > 0.0).sum()), 100)
        self.assertFalse((west[shadow] > 0.0).any())
        # Geometry of the rays is untouched everywhere
        np.testing.assert_array_equal(field.valid, clear.valid)
        np.testing.assert_array_equal(field.tau_ns, clear.tau_ns)
        # The shadow is a wedge: far behind the building, on the axis Tx -> building, it is still dark
        self.assertTrue(shadow[np.argmin(np.abs(ys - 132.0)), np.argmin(np.abs(xs - 296.0))])
        self.assertFalse(shadow[np.argmin(np.abs(ys - 20.0)), np.argmin(np.abs(xs - 296.0))])

    def test_a_building_lower_than_the_antenna_casts_a_shadow_of_finite_length(self):
        tx = make_tx(*self.TX_XY, 45.0)
        xs, ys, points = self.grid()
        field = make_tracer(*building(*self.BOX, height=12.0)).trace_points(tx, points)
        shadow, inside = self.shadow_mask(xs, ys, 45.0, 1.5, 12.0)
        np.testing.assert_array_equal(field.power_linear[0] == 0.0, shadow)
        tall, _ = self.shadow_mask(xs, ys, 45.0, 1.5, 60.0)
        self.assertLess(int(shadow.sum()), int(tall.sum()) - 300)     # the ray passes over the roof further away
        self.assertTrue((tall | ~shadow).all())
        self.assertTrue(shadow.any())
        self.assertFalse(shadow[np.argmin(np.abs(ys - 132.0)), np.argmin(np.abs(xs - 296.0))])
        # No roof is modelled: part of the interior is lit from above (document it; add a horizontal wall as roof)
        self.assertTrue((field.power_linear[0][inside] > 0.0).any())
        roof = PlanarReflector(p0=[0.0, 0.0, 12.0], normal=[0.0, 0.0, 1.0], reflectivity=0.0,
                               bounds=(self.BOX[0], self.BOX[1], self.BOX[2], self.BOX[3], 12.0, 12.0),
                               transmission_loss_db=INF)
        with_roof = make_tracer(*building(*self.BOX, height=12.0), roof).trace_points(tx, points)
        self.assertEqual(float(with_roof.total_power_linear()[inside].max()), 0.0)

    def test_penetration_loss_of_the_faces_adds_up_behind_the_building(self):
        tx = make_tx(*self.TX_XY, 25.0)
        xs, ys, points = self.grid()
        clear = make_tracer(*building(*self.BOX, height=40.0, loss=None)).trace_points(tx, points)
        field = make_tracer(*building(*self.BOX, height=40.0, loss=8.0)).trace_points(tx, points)
        shadow, inside = self.shadow_mask(xs, ys, 25.0, 1.5, 40.0)
        loss = field.obstruction_loss_db[0]
        np.testing.assert_array_equal(loss[~shadow], 0.0)
        np.testing.assert_array_equal(loss[inside], 8.0)                  # one face to get in
        np.testing.assert_array_equal(loss[shadow & ~inside], 16.0)       # two faces to go through
        with np.errstate(divide="ignore"):
            measured = 10.0 * np.log10(clear.power_linear[0] / field.power_linear[0])
        np.testing.assert_allclose(measured, loss, atol=1e-9)

    def simulate(self, **options):
        sim = PropagationSimulator()
        sim.add_transmitter(make_tx(*self.TX_XY, 25.0, tx_id=4))
        for wall in building(*self.BOX, height=40.0):
            sim.add_reflector(wall)
        return sim, sim.run_raytracing_simulation((0.0, 300.0), (0.0, 200.0), resolution_m=4.0, **options)

    def test_grid_simulation_marks_unreachable_pixels_as_no_data(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sim, result = self.simulate()
            _, slow = self.simulate(vectorized=False)
            _, coherent = self.simulate(coherent=True)
        assert_close_result(self, result, slow)
        xs, ys = result.x_coords, result.y_coords
        shadow, inside = self.shadow_mask(xs, ys, 25.0, 1.5, 40.0)
        dark = np.isneginf(result.rx_power_dbm[0])
        self.assertTrue(dark[inside].all())                              # nothing enters the building
        self.assertTrue((dark <= shadow).all())                          # no ray at all only inside the shadow
        self.assertGreater(int(dark.sum()), 400)
        self.assertEqual(int(np.isnan(result.rx_power_dbm).sum()), 0)
        # Conventions of a pixel without power: -inf dBm, no server, NaN for everything derived from it
        np.testing.assert_array_equal(result.best_server_id == -1, dark)
        np.testing.assert_array_equal(result.best_server_id[~dark], 4)
        np.testing.assert_array_equal(np.isneginf(result.max_rx_power_dbm), dark)
        for grid in (result.path_loss_db[0], result.rms_delay_spread_ns, result.angular_spread_bs_phi,
                     result.best_server_path_loss_db()):
            np.testing.assert_array_equal(np.isnan(grid), dark)
        np.testing.assert_array_equal(np.isneginf(coherent.rx_power_dbm[0]), dark)
        # Outside the shadow nothing changes with respect to the same building without obstruction
        clear_sim = PropagationSimulator()
        clear_sim.add_transmitter(make_tx(*self.TX_XY, 25.0, tx_id=4))
        for wall in building(*self.BOX, height=40.0, loss=None):
            clear_sim.add_reflector(wall)
        clear = clear_sim.run_raytracing_simulation((0.0, 300.0), (0.0, 200.0), resolution_m=4.0)
        self.assertTrue(np.isfinite(clear.rx_power_dbm).all())
        # (except for the rays that, without obstruction, were reflected on the inner side of the far faces:
        # they are weak, so the lit pixels lose less than 1 dB)
        drop = clear.rx_power_dbm[0][~shadow] - result.rx_power_dbm[0][~shadow]
        self.assertGreaterEqual(float(drop.min()), 0.0)
        self.assertLess(float(drop.max()), 1.0)
        self.assertGreater(int((drop == 0.0).sum()), 1000)

    def test_exports_and_maps_handle_the_shadow(self):
        sim, result = self.simulate()
        dark = int(np.isneginf(result.max_rx_power_dbm).sum())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sinr = result.compute_sinr()
            summary = result.summary()
            ascii_map = MapVisualizer.render_ascii_heatmap(result.max_rx_power_dbm)
            with tempfile.TemporaryDirectory() as tmp:
                result.export_csv(os.path.join(tmp, "r.csv"))
                result.export_json(os.path.join(tmp, "r.json"))
                with open(os.path.join(tmp, "r.json")) as f:
                    text = f.read()
                with open(os.path.join(tmp, "r.csv"), newline="") as f:
                    rows = list(csv.DictReader(f))
                svgs = {}
                for metric in ("rx_power", "delay_spread", "best_server", "sinr", "interference"):
                    for theme in ("light", "dark"):
                        path = os.path.join(tmp, f"{metric}_{theme}.svg")
                        MapVisualizer.export_svg(result, path, metric=metric, theme=theme)
                        with open(path, encoding="utf-8") as f:
                            svgs[metric, theme] = f.read()
        self.assertEqual(int(np.isnan(sinr).sum()), dark)
        self.assertEqual(json.loads(text), summary)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        for key in ("max_rx_power_dbm", "min_rx_power_dbm", "mean_rx_power_dbm", "mean_delay_spread_ns"):
            self.assertTrue(math.isfinite(summary["metrics"][key]))
        self.assertTrue(math.isfinite(summary["sinr"]["min_sinr_db"]))
        no_server = [row for row in rows if row["best_tx_id"] == "-1"]
        self.assertEqual(len(no_server), dark)
        self.assertEqual({(row["rx_power_dbm"], row["path_loss_db"], row["rms_delay_spread_ns"], row["sinr_db"])
                          for row in no_server}, {("-inf", "nan", "nan", "nan")})
        self.assertEqual(len(ascii_map.splitlines()), len(result.y_coords) + 4)
        for (metric, theme), text in svgs.items():
            root = ET.fromstring(text)
            labels = [el.text for el in root.iter(SVG + "text")]
            self.assertIn("No server" if metric == "best_server" else "No data", labels, (metric, theme))
            self.assertNotIn("nan", text.lower())
            self.assertNotIn("inf", text.replace("Interference", "").lower())
            cells = [el for el in root.iter(SVG + "rect") if el.find(SVG + "title") is not None]
            self.assertEqual(len(cells), result.best_server_id.size - dark, (metric, theme))

    def test_a_map_where_nothing_arrives_at_all(self):
        """Transmitter enclosed in an opaque box: every pixel outside it is without data; nothing crashes."""
        sim = PropagationSimulator()
        sim.add_transmitter(make_tx(140.0, 100.0, 25.0))
        for wall in building(*self.BOX, height=40.0):
            sim.add_reflector(wall)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = sim.run_raytracing_simulation((170.0, 300.0), (0.0, 200.0), resolution_m=10.0)
            slow = sim.run_raytracing_simulation((170.0, 300.0), (0.0, 200.0), resolution_m=10.0, vectorized=False)
            sinr = result.compute_sinr()
            summary = result.summary()
            ascii_map = MapVisualizer.render_ascii_heatmap(result.max_rx_power_dbm)
            with tempfile.TemporaryDirectory() as tmp:
                result.export_json(os.path.join(tmp, "r.json"))
                result.export_csv(os.path.join(tmp, "r.csv"))
                for metric in ("rx_power", "delay_spread", "best_server", "sinr"):
                    MapVisualizer.export_svg(result, os.path.join(tmp, "m.svg"), metric=metric)
                    with open(os.path.join(tmp, "m.svg"), encoding="utf-8") as f:
                        self.assertIn("No data" if metric != "best_server" else "No server", f.read())
        assert_close_result(self, result, slow)
        self.assertTrue(np.isneginf(result.rx_power_dbm).all())
        self.assertTrue((result.best_server_id == -1).all())
        self.assertTrue(np.isnan(sinr).all())
        self.assertIsNone(summary["metrics"]["max_rx_power_dbm"])
        self.assertIn("no data", ascii_map)


# --------------------------------------------------------------------------------------------
class SimulationWithObstructionTests(unittest.TestCase):
    def two_cells(self, loss):
        """Two co-channel stations on either side of an infinite wall on x = 205 (between grid nodes)."""
        sim = PropagationSimulator()
        sim.add_transmitter(Transmitter(tx_id=0, x=100.0, y=150.0, z=25.0, frequency_hz=2.4e9))
        sim.add_transmitter(Transmitter(tx_id=1, x=300.0, y=150.0, z=25.0, frequency_hz=2.4e9))
        sim.add_reflector(PlanarReflector(p0=[205.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=0.5,
                                          transmission_loss_db=loss))
        return sim.run_raytracing_simulation(*AREA, resolution_m=10.0)

    def test_opaque_wall_splits_the_cells_and_removes_the_interference(self):
        clear, opaque = self.two_cells(None), self.two_cells(INF)
        west = np.broadcast_to(opaque.x_coords[None, :] < 205.0, opaque.best_server_id.shape)
        # Each station only reaches its own side, where its power is the unobstructed one
        np.testing.assert_array_equal(np.isneginf(opaque.rx_power_dbm[0]), ~west)
        np.testing.assert_array_equal(np.isneginf(opaque.rx_power_dbm[1]), west)
        np.testing.assert_array_equal(opaque.rx_power_dbm[0][west], clear.rx_power_dbm[0][west])
        np.testing.assert_array_equal(opaque.rx_power_dbm[1][~west], clear.rx_power_dbm[1][~west])
        np.testing.assert_array_equal(opaque.best_server_id, np.where(west, 0, 1))
        self.assertTrue(np.isfinite(opaque.max_rx_power_dbm).all())
        # Best server, spreads: those of the only station that arrives
        self.assertFalse(np.isnan(opaque.rms_delay_spread_ns).any())
        # SINR: the blocked interferer contributes 0 mW, so SINR = P - N exactly; without obstruction it is lower
        noise = thermal_noise_dbm(20e6, 7.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sinr = opaque.compute_sinr()
        np.testing.assert_allclose(sinr, opaque.max_rx_power_dbm - noise, atol=1e-9)
        np.testing.assert_allclose(opaque.interference_plus_noise_dbm, noise, atol=1e-9)
        self.assertTrue((sinr > clear.compute_sinr() + 10.0).all())

    def test_finite_loss_reduces_the_interference_by_that_amount(self):
        clear, lossy = self.two_cells(None), self.two_cells(12.0)
        west = np.broadcast_to(clear.x_coords[None, :] < 205.0, clear.best_server_id.shape)
        # Every ray of the far station crosses the wall once: its total power drops exactly 12 dB
        np.testing.assert_allclose(clear.rx_power_dbm[1][west] - lossy.rx_power_dbm[1][west], 12.0, atol=1e-9)
        np.testing.assert_allclose(clear.rx_power_dbm[0][~west] - lossy.rx_power_dbm[0][~west], 12.0, atol=1e-9)
        np.testing.assert_array_equal(clear.rx_power_dbm[0][west], lossy.rx_power_dbm[0][west])
        np.testing.assert_allclose(clear.path_loss_db[1][west] + 12.0, lossy.path_loss_db[1][west], atol=1e-9)

    def test_best_server_falls_back_to_the_station_that_is_not_blocked(self):
        sim = PropagationSimulator()
        sim.add_transmitter(Transmitter(tx_id=0, x=100.0, y=150.0, z=25.0, tx_power_dbm=46.0))   # strong, behind the wall
        sim.add_transmitter(Transmitter(tx_id=1, x=390.0, y=150.0, z=25.0, tx_power_dbm=20.0))   # weak, in sight
        sim.add_reflector(PlanarReflector(p0=[205.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=0.0,
                                          transmission_loss_db=INF))
        result = sim.run_raytracing_simulation(*AREA, resolution_m=10.0)
        east = np.broadcast_to(result.x_coords[None, :] > 205.0, result.best_server_id.shape)
        np.testing.assert_array_equal(result.best_server_id, np.where(east, 1, 0))
        np.testing.assert_array_equal(result.max_rx_power_dbm[east], result.rx_power_dbm[1][east])

    def test_coherent_sum_uses_the_attenuated_amplitudes(self):
        sim = obstructed_sim(seed=4)
        tx = sim.transmitters[0]
        result = sim.run_raytracing_simulation(*AREA, resolution_m=10.0, coherent=True, min_distance_m=None)
        points = np.empty(result.max_rx_power_dbm.shape + (3,))
        points[..., 0], points[..., 1], points[..., 2] = result.x_coords[None, :], result.y_coords[:, None], 1.5
        field = sim.ray_tracer.trace_points(tx, points)
        for wall in sim.ray_tracer.reflectors:
            wall.transmission_loss_db = None
        clear = sim.ray_tracer.trace_points(tx, points)
        # Same phases as without obstruction, each amplitude scaled by 10^(-L/20)
        scaled = clear.amplitude * 10.0 ** (-field.obstruction_loss_db / 20.0)
        np.testing.assert_allclose(field.amplitude, scaled, rtol=1e-12, atol=0.0)
        with np.errstate(divide="ignore"):
            expected = 10.0 * np.log10(np.abs(scaled.sum(axis=0)) ** 2)
        np.testing.assert_allclose(result.rx_power_dbm[0], expected, atol=1e-9)
        self.assertGreater(int((field.obstruction_loss_db > 0).sum()), 1000)


# --------------------------------------------------------------------------------------------
class CliObstructionTests(unittest.TestCase):
    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), warnings.catch_warnings():
                warnings.simplefilter("error")
                code = cli.main(["--size", "400", "--res", "20", "--outdir", tmp, *args])
            self.assertEqual((code, stderr.getvalue()), (0, ""))
            with open(os.path.join(tmp, "simulation_results.csv"), newline="") as f:
                rows = list(csv.DictReader(f))
            with open(os.path.join(tmp, "simulation_summary.json")) as f:
                summary = json.load(f)
            files = sorted(os.listdir(tmp))
        return stdout.getvalue(), rows, summary, files

    def test_help_documents_the_new_options(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as ctx:
            cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--wall-loss DB", stdout.getvalue())
        self.assertIn("--building", stdout.getvalue())

    def test_default_run_is_unchanged_and_zero_loss_is_the_same(self):
        out, rows, summary, _ = self.run_cli()
        self.assertIn("ray tracing (reflector on)", out)
        self.assertNotIn("Walls:", out)                           # same banner as 1.2.0
        out_zero, rows_zero, summary_zero, _ = self.run_cli("--wall-loss", "0")
        self.assertEqual(rows_zero, rows)
        self.assertEqual(summary_zero, summary)
        self.assertIn("infinite wall on x = 200 m; 0 dB of penetration loss", out_zero)

    def test_wall_loss_on_the_demo_wall(self):
        _, base, base_summary, _ = self.run_cli()
        out, lossy, lossy_summary, _ = self.run_cli("--wall-loss", "15")
        self.assertIn("15 dB of penetration loss per wall crossed", out)
        self.assertNotEqual(lossy, base)
        self.assertLess(lossy_summary["metrics"]["mean_rx_power_dbm"], base_summary["metrics"]["mean_rx_power_dbm"])
        out, opaque, opaque_summary, files = self.run_cli("--wall-loss", "inf", "--sinr")
        self.assertIn("opaque", out)
        self.assertIn("coverage_sinr.svg", files)
        # Tx0 (120, 120) and Tx1 (280, 280) are on either side of the wall (x = 200): each one serves its side
        # (the pixels at x = 200 lie on the wall itself and see both stations)
        for row in opaque:
            x = float(row["x_m"])
            if x != 200.0:
                self.assertEqual(row["best_tx_id"], "0" if x < 200.0 else "1", row)
                self.assertGreater(float(row["sinr_db"]), 30.0, row)         # no co-channel interference left
                self.assertAlmostEqual(float(row["interference_plus_noise_dbm"]), thermal_noise_dbm(20e6, 7.0), delta=0.006)
        self.assertLessEqual(opaque_summary["metrics"]["mean_rx_power_dbm"],
                             lossy_summary["metrics"]["mean_rx_power_dbm"])

    def test_building_scenario(self):
        out, plain, _, files = self.run_cli("--building")
        self.assertIn("building of 4 walls, x = 180-240 m, y = 140-200 m, 50 m tall", out)
        self.assertIn("blocks nothing", out)
        self.assertTrue(all(row["best_tx_id"] != "-1" for row in plain))
        out, shadow, summary, files = self.run_cli("--building", "--wall-loss", "inf")
        self.assertIn("opaque", out)
        self.assertEqual(files, ["coverage_best_server.svg", "coverage_delay_spread.svg", "coverage_rx_power.svg",
                                 "simulation_results.csv", "simulation_summary.json"])
        inside = [row for row in shadow if 180.0 < float(row["x_m"]) < 240.0 and 140.0 < float(row["y_m"]) < 200.0]
        self.assertEqual(len(inside), 4)                                    # pixels at x = 200, 220 and y = 160, 180
        for row in inside:
            self.assertEqual((row["best_tx_id"], row["rx_power_dbm"], row["path_loss_db"]), ("-1", "-inf", "nan"))
        outside = [row for row in shadow if row not in inside]
        self.assertTrue(all(row["best_tx_id"] in ("0", "1") for row in outside))
        self.assertTrue(math.isfinite(summary["metrics"]["min_rx_power_dbm"]))
        # East of the building Tx0 (120, 120) is in shadow: Tx1 takes over the pixels that Tx0 served, with less power
        before = {(row["x_m"], row["y_m"]): row for row in plain}
        behind = [row for row in shadow if float(row["x_m"]) >= 240.0 and row["y_m"] == "160.0"]
        self.assertEqual(len(behind), 9)
        self.assertEqual({row["best_tx_id"] for row in behind}, {"1"})
        taken_over = [row for row in behind if before[row["x_m"], row["y_m"]]["best_tx_id"] == "0"]
        self.assertEqual([row["x_m"] for row in taken_over], ["240.0", "260.0", "280.0", "300.0"])
        for row in taken_over:
            self.assertLess(float(row["rx_power_dbm"]), float(before[row["x_m"], row["y_m"]]["rx_power_dbm"]) - 0.4)
        _, lossy, _, _ = self.run_cli("--building", "--wall-loss", "10")
        self.assertTrue(all(math.isfinite(float(row["rx_power_dbm"])) for row in lossy))

    def test_options_without_effect_are_reported(self):
        _, base_fspl, _, _ = self.run_cli("--mode", "fspl")
        out, fspl, _, _ = self.run_cli("--mode", "fspl", "--building", "--wall-loss", "inf")
        self.assertIn("--building and --wall-loss are ignored by empirical models", out)
        self.assertEqual(fspl, base_fspl)
        _, base_off, _, _ = self.run_cli("--reflector", "off")
        out, off, _, _ = self.run_cli("--reflector", "off", "--building", "--wall-loss", "inf")
        self.assertIn("no effect with --reflector off", out)
        self.assertEqual(off, base_off)

    def test_invalid_wall_loss_is_a_usage_error(self):
        for value in ("-3", "nan", "-inf", "abc"):
            stderr = io.StringIO()
            with self.subTest(value=value), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
                cli.main(["--wall-loss", value])
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("--wall-loss", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
