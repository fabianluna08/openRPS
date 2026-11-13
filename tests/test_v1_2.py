"""Tests of the features added in OpenRPS 1.2.0 (and regression against 1.1.0 for the defaults)."""
import csv
import json
import math
import os
import re
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from open_rps import (
    ChannelImpulseResponse,
    Cost231Hata,
    FreeSpacePathLoss,
    LogDistancePathLoss,
    MapVisualizer,
    ModelRangeWarning,
    MultipathRay,
    OkumuraHata,
    PlanarReflector,
    PropagationSimulator,
    RayTracer,
    SimulationResult,
    Transmitter,
    TwoRayGroundReflection,
    thermal_noise_dbm,
)
from open_rps.visualizer import OKABE_ITO, _THEMES, _merged_rectangles, _ramp_color, _ramp_keys

SVG = "{http://www.w3.org/2000/svg}"
AREA = ((0.0, 400.0), (0.0, 300.0))
RESULT_GRIDS = ("rx_power_dbm", "path_loss_db", "best_server_id", "best_server_index", "max_rx_power_dbm",
                "rms_delay_spread_ns", "angular_spread_bs_phi")


def build_sim(reflector=True, **tx_kwargs):
    """Same scenario as tests/test_simulation.py (the 1.1.0 reference values below were computed on it)."""
    sim = PropagationSimulator()
    sim.add_transmitter(Transmitter(tx_id=0, x=100.0, y=100.0, z=25.0, tx_power_dbm=43.0,
                                    frequency_hz=2.4e9, antenna_gain_dbi=12.0, **tx_kwargs))
    sim.add_transmitter(Transmitter(tx_id=1, x=300.0, y=200.0, z=20.0, tx_power_dbm=40.0,
                                    frequency_hz=1.8e9, antenna_gain_dbi=10.0, **tx_kwargs))
    if reflector:
        sim.add_reflector(PlanarReflector(p0=[200.0, 150.0, 0.0], normal=[1.0, 0.2, 0.0], reflectivity=0.6))
    return sim


def single_tx_sim(x=50.0, y=50.0, z=25.0, **kwargs):
    sim = PropagationSimulator()
    sim.add_transmitter(Transmitter(tx_id=0, x=x, y=y, z=z, **kwargs))
    return sim


def ray(power_mw, tau_ns, phase=0.0, tx_id=0, tx_phi=0.0):
    amplitude = math.sqrt(power_mw) * complex(math.cos(phase), math.sin(phase))
    return MultipathRay(tx_id, amplitude, tau_ns, tx_phi, 0.0, 0.0, 0.0)


def assert_same_result(test, a, b, exact=True, names=RESULT_GRIDS):
    for name in names:
        if exact:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        else:
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-9, atol=1e-7,
                                       equal_nan=True, err_msg=name)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class RegressionAgainst110Tests(unittest.TestCase):
    """With the default arguments the numbers must be those of OpenRPS 1.1.0 (values computed with 1.1.0)."""
    PIXELS = [(0, 0), (5, 5), (10, 15), (15, 20), (7, 13), (3, 9)]

    def test_raytracing_defaults(self):
        r = build_sim().run_raytracing_simulation(*AREA, resolution_m=20.0)
        expected_power = [
            [-27.824528450003356, -11.409506739640902, -31.564139583090608,
             -35.07845764134123, -29.275462445026626, -23.845999947451972],
            [-37.313035965653604, -33.787870415619395, -11.892995196296646,
             -30.131170174852492, -24.69367085735927, -32.36719953516479]]
        expected_ds = [155.67787041973583, 37.14712538159285, 30.311433343044822,
                       152.87897301819618, 79.59855320114971, 67.4656353103528]
        expected_as = [31.464893545746605, 0.6490823946947089, 7.703235685395519,
                       30.81937972885918, 6.702300608031496, 5.381409474001827]
        for n, (j, i) in enumerate(self.PIXELS):
            for t in range(2):
                self.assertAlmostEqual(r.rx_power_dbm[t, j, i], expected_power[t][n], delta=1e-10)
            self.assertAlmostEqual(r.rms_delay_spread_ns[j, i], expected_ds[n], delta=1e-9)
            self.assertAlmostEqual(r.angular_spread_bs_phi[j, i], expected_as[n], delta=1e-9)
        self.assertEqual([int(r.best_server_id[j, i]) for (j, i) in self.PIXELS], [0, 0, 1, 1, 1, 0])
        self.assertAlmostEqual(float(r.rx_power_dbm.sum()), -19609.51651074393, delta=1e-7)
        self.assertAlmostEqual(float(r.rms_delay_spread_ns.sum()), 28765.658995329213, delta=1e-6)
        self.assertAlmostEqual(float(r.angular_spread_bs_phi.sum()), 4464.295063106383, delta=1e-7)
        self.assertEqual(int(r.best_server_id.sum()), 147)
        self.assertTrue(np.isfinite(r.rx_power_dbm).all())      # the default min_distance_m masks nothing here

    def test_empirical_defaults(self):
        expected = {
            "FreeSpacePathLoss": ([-29.484990047088658, -24.990094444531564], -20040.800643369403),
            "TwoRayGroundReflection": ([-27.989847190092334, -24.751639781028274], -20053.756908190902),
            "OkumuraHata": ([-42.75219905747694, -33.20225905588167], -28644.036764217406),
            "Cost231Hata": ([-58.53842349656358, -47.086624145829916], -38613.35484597314),
            "LogDistancePathLoss": ([-56.14477924167255, -47.452211116880804], -37588.06436716994),
        }
        models = [FreeSpacePathLoss(), TwoRayGroundReflection(), OkumuraHata("suburban"), Cost231Hata(),
                  LogDistancePathLoss(3.2)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ModelRangeWarning)
            for model in models:
                r = build_sim().run_empirical_simulation(model, *AREA, resolution_m=20.0)
                pixel, total = expected[type(model).__name__]
                for t in range(2):
                    self.assertAlmostEqual(r.rx_power_dbm[t, 7, 13], pixel[t], delta=1e-10, msg=type(model).__name__)
                self.assertAlmostEqual(float(r.rx_power_dbm.sum()), total, delta=1e-7, msg=type(model).__name__)

    def test_explicit_defaults_change_nothing(self):
        """Every new argument at its 'off' value gives bit-identical grids, files and SINR-free exports."""
        sim = build_sim()
        base = sim.run_raytracing_simulation(*AREA, resolution_m=20.0)
        explicit = sim.run_raytracing_simulation(*AREA, resolution_m=20.0, coherent=False, min_distance_m=None,
                                                 shadow_fading_sigma_db=0.0, seed=123)
        assert_same_result(self, base, explicit)
        isotropic = build_sim(pattern="isotropic", azimuth_deg=77.0, beamwidth_az_deg=10.0, downtilt_deg=12.0)
        assert_same_result(self, base, isotropic.run_raytracing_simulation(*AREA, resolution_m=20.0))
        emp = sim.run_empirical_simulation(TwoRayGroundReflection(), *AREA, resolution_m=20.0)
        emp_explicit = isotropic.run_empirical_simulation(TwoRayGroundReflection(), *AREA, resolution_m=20.0,
                                                          min_distance_m=0, shadow_fading_sigma_db=0.0, seed=5)
        assert_same_result(self, emp, emp_explicit)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.csv")
            base.export_csv(path)
            with open(path) as f:
                header = f.readline().strip()
            self.assertEqual(header, "x_m,y_m,best_tx_id,rx_power_dbm,path_loss_db,rms_delay_spread_ns,angular_spread_phi_deg")
        self.assertEqual(sorted(base.summary()), ["grid", "metrics", "tx_ids"])


class MeanExcessDelayTests(unittest.TestCase):
    def test_relative_to_first_by_hand(self):
        """1 mW at 100 ns and 0.25 mW at 200 ns: mean 120 ns absolute, 120 - 100 = 20 ns of excess."""
        cir = ChannelImpulseResponse([ray(0.25, 200.0), ray(1.0, 100.0)])
        self.assertAlmostEqual(cir.mean_excess_delay(), 120.0, places=9)                       # default unchanged
        self.assertAlmostEqual(cir.mean_excess_delay(relative_to_first=False), 120.0, places=9)
        self.assertAlmostEqual(cir.mean_excess_delay(relative_to_first=True), 20.0, places=9)

    def test_single_ray_filter_and_empty(self):
        self.assertEqual(ChannelImpulseResponse([ray(1.0, 333.0)]).mean_excess_delay(relative_to_first=True), 0.0)
        cir = ChannelImpulseResponse([ray(1.0, 100.0, tx_id=3), ray(0.25, 200.0, tx_id=3), ray(5.0, 10.0, tx_id=8)])
        self.assertAlmostEqual(cir.mean_excess_delay(tx_id=3, relative_to_first=True), 20.0, places=9)
        self.assertTrue(math.isnan(ChannelImpulseResponse().mean_excess_delay(relative_to_first=True)))
        self.assertTrue(math.isnan(cir.mean_excess_delay(tx_id=99, relative_to_first=True)))

    def test_first_arrival_is_a_ray_with_power(self):
        cir = ChannelImpulseResponse([ray(0.0, 10.0), ray(1.0, 100.0), ray(1.0, 300.0)])
        self.assertAlmostEqual(cir.mean_excess_delay(relative_to_first=True), 100.0, places=9)

    def test_ray_tracer_excess_delay_does_not_depend_on_range(self):
        cir = RayTracer().trace_paths(Transmitter(tx_id=0, x=0.0, y=0.0, z=25.0), (800.0, 0.0, 1.5))
        excess = cir.mean_excess_delay(relative_to_first=True)
        self.assertGreater(cir.mean_excess_delay(), 2600.0)          # absolute: ~800 m / c
        self.assertTrue(0.0 < excess < 1.0)                          # the ground ray is only ~9 cm longer
        self.assertAlmostEqual(excess, cir.mean_excess_delay() - min(r.tau_ns for r in cir.rays), places=9)


class MinimumDistanceTests(unittest.TestCase):
    """Tx at the receiver height on a grid node: that pixel is at d = 0 from the antenna."""

    def setUp(self):
        self.sim = single_tx_sim(50.0, 50.0, 1.5)
        self.area = ((0.0, 100.0), (0.0, 100.0))

    def run_all(self, sim, **kwargs):
        results = [sim.run_raytracing_simulation(*self.area, resolution_m=10.0, **kwargs),
                   sim.run_raytracing_simulation(*self.area, resolution_m=10.0, vectorized=False, **kwargs)]
        for model in (FreeSpacePathLoss(), TwoRayGroundReflection(), LogDistancePathLoss()):
            results.append(sim.run_empirical_simulation(model, *self.area, resolution_m=10.0, **kwargs))
        return results

    def test_pixel_at_the_antenna_is_nan_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = self.run_all(self.sim)
            for r in results:
                masked = np.isnan(r.rx_power_dbm[0])
                self.assertEqual(masked.sum(), 1)
                self.assertTrue(masked[5, 5])
                self.assertTrue(np.isnan(r.path_loss_db[0, 5, 5]))
                self.assertTrue(np.isfinite(np.delete(r.rx_power_dbm[0].ravel(), 5 * 11 + 5)).all())
                # Only transmitter unreliable there: no server, and "unknown" (NaN) rather than "no power" (-inf)
                self.assertEqual(r.best_server_id[5, 5], -1)
                self.assertEqual(r.best_server_index[5, 5], -1)
                self.assertTrue(np.isnan(r.max_rx_power_dbm[5, 5]))
                self.assertTrue(np.isnan(r.rms_delay_spread_ns[5, 5]))
                self.assertEqual((r.best_server_id == 0).sum(), 120)
                # Statistics and exports ignore the pixel
                self.assertAlmostEqual(r.summary()["metrics"]["max_rx_power_dbm"], float(np.nanmax(r.max_rx_power_dbm)))
                with tempfile.TemporaryDirectory() as tmp:
                    r.export_json(os.path.join(tmp, "s.json"))
                    r.export_csv(os.path.join(tmp, "s.csv"))
                    rows = read_csv(os.path.join(tmp, "s.csv"))
                    self.assertEqual((rows[60]["x_m"], rows[60]["y_m"], rows[60]["best_tx_id"], rows[60]["rx_power_dbm"]),
                                     ("50.0", "50.0", "-1", "nan"))
                    for metric in ("rx_power", "best_server", "sinr"):
                        MapVisualizer.export_svg(r, os.path.join(tmp, "m.svg"), metric=metric)
                        ET.parse(os.path.join(tmp, "m.svg"))
                MapVisualizer.render_ascii_heatmap(r.max_rx_power_dbm)
        assert_same_result(self, results[0], results[1], exact=False)

    def test_can_be_disabled(self):
        for off in (None, 0, 0.0):
            for r in self.run_all(self.sim, min_distance_m=off):
                self.assertFalse(np.isnan(r.rx_power_dbm).any())
                self.assertEqual(r.best_server_id[5, 5], 0)
        # ... which brings back the absurd 1.1.0 value: 0 dB of basic loss at d = 0 (P_rx = P_tx + G)
        r = self.sim.run_empirical_simulation(FreeSpacePathLoss(), *self.area, resolution_m=10.0, min_distance_m=None)
        self.assertAlmostEqual(r.rx_power_dbm[0, 5, 5], 43.0 + 14.0, places=9)

    def test_best_server_falls_back_to_another_transmitter(self):
        self.sim.add_transmitter(Transmitter(tx_id=9, x=90.0, y=90.0, z=20.0, tx_power_dbm=20.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = self.run_all(self.sim)
        for r in results:
            self.assertTrue(np.isnan(r.rx_power_dbm[0, 5, 5]))
            self.assertEqual(r.best_server_id[5, 5], 9)
            self.assertEqual(r.best_server_index[5, 5], 1)
            self.assertEqual(r.max_rx_power_dbm[5, 5], r.rx_power_dbm[1, 5, 5])
            self.assertEqual(r.best_server_id[5, 4], 0)          # 10 m away the near transmitter wins again
            self.assertTrue(np.isfinite(r.max_rx_power_dbm).all())
        self.assertFalse(np.isnan(results[0].rms_delay_spread_ns).any())
        assert_same_result(self, results[0], results[1], exact=False)

    def test_threshold_is_the_3d_distance_and_strict(self):
        for z, masked in ((2.4, True), (2.5, False), (0.6, True), (0.5, False)):      # |z - 1.5| = 0.9 or 1.0 m
            for r in self.run_all(single_tx_sim(50.0, 50.0, z)):
                self.assertEqual(bool(np.isnan(r.rx_power_dbm[0, 5, 5])), masked, msg=f"z={z}")

    def test_custom_threshold(self):
        for r in self.run_all(single_tx_sim(50.0, 50.0, 10.0), min_distance_m=22.0):
            d = np.sqrt((r.x_coords[None, :] - 50.0) ** 2 + (r.y_coords[:, None] - 50.0) ** 2 + 8.5 ** 2)
            np.testing.assert_array_equal(np.isnan(r.rx_power_dbm[0]), d < 22.0)
            np.testing.assert_array_equal(r.best_server_id == -1, d < 22.0)
            self.assertEqual((d < 22.0).sum(), 13)

    def test_invalid_threshold(self):
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.sim.run_raytracing_simulation(*self.area, resolution_m=10.0, min_distance_m=bad)
            with self.assertRaises(ValueError):
                self.sim.run_empirical_simulation(FreeSpacePathLoss(), *self.area, resolution_m=10.0, min_distance_m=bad)


class StrictHataTests(unittest.TestCase):
    ARGS = dict(ht_m=30.0, hr_m=1.5)

    def test_inside_the_range_strict_changes_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = np.array([1000.0, 5000.0, 20000.0])
            np.testing.assert_array_equal(OkumuraHata("suburban", strict=True).path_loss(d, 900e6, **self.ARGS),
                                          OkumuraHata("suburban").path_loss(d, 900e6, **self.ARGS))
            self.assertEqual(Cost231Hata(strict=True).path_loss(2000.0, 1800e6, **self.ARGS),
                             Cost231Hata().path_loss(2000.0, 1800e6, **self.ARGS))

    def test_outside_the_range_raises_value_error_with_a_clear_message(self):
        model = OkumuraHata(strict=True)
        cases = [
            (dict(distance_m=2000.0, frequency_hz=2.4e9, **self.ARGS), "frequency 2400 MHz outside 150-1500 MHz"),
            (dict(distance_m=2000.0, frequency_hz=900e6, ht_m=10.0, hr_m=1.5), "base antenna height 10 m outside 30-200 m"),
            (dict(distance_m=2000.0, frequency_hz=900e6, ht_m=30.0, hr_m=12.0), "mobile antenna height 12 m outside 1-10 m"),
            (dict(distance_m=100.0, frequency_hz=900e6, **self.ARGS), "distance outside 1-20 km"),
            (dict(distance_m=np.array([2000.0, 50e3]), frequency_hz=900e6, **self.ARGS), "distance outside 1-20 km"),
        ]
        for kwargs, problem in cases:
            with self.subTest(problem=problem), warnings.catch_warnings():
                warnings.simplefilter("error")           # it must raise, not warn
                with self.assertRaises(ValueError) as ctx:
                    model.path_loss(**kwargs)
                message = str(ctx.exception)
                self.assertIn(problem, message)
                self.assertIn("OkumuraHata", message)
                self.assertIn("strict=False", message)
                with self.assertRaises(ValueError):
                    model.basic_path_loss(kwargs["distance_m"], kwargs["frequency_hz"],
                                          ht_m=kwargs["ht_m"], hr_m=kwargs["hr_m"])

    def test_every_problem_is_listed(self):
        with self.assertRaises(ValueError) as ctx:
            Cost231Hata(strict=True).path_loss(500.0, 2.4e9, ht_m=25.0, hr_m=1.5)
        message = str(ctx.exception)
        for problem in ("Cost231Hata", "frequency 2400 MHz outside 1500-2000 MHz",
                        "base antenna height 25 m outside 30-200 m", "distance outside 1-20 km"):
            self.assertIn(problem, message)

    def test_default_still_warns_and_extrapolates(self):
        self.assertFalse(OkumuraHata().strict)
        with self.assertWarns(ModelRangeWarning):
            value = OkumuraHata().path_loss(100.0, 900e6, **self.ARGS)
        self.assertTrue(math.isfinite(value))
        with self.assertWarns(ModelRangeWarning):
            self.assertTrue(math.isfinite(OkumuraHata("open", strict=False).path_loss(100.0, 900e6, **self.ARGS)))

    def test_grid_simulation_propagates_the_error(self):
        sim = single_tx_sim(z=30.0, frequency_hz=900e6)
        with self.assertRaises(ValueError):
            sim.run_empirical_simulation(OkumuraHata(strict=True), (0.0, 500.0), (0.0, 500.0), resolution_m=100.0)
        # A grid entirely inside 1-20 km works
        r = sim.run_empirical_simulation(OkumuraHata(strict=True), (2000.0, 5000.0), (2000.0, 5000.0), resolution_m=1000.0)
        self.assertTrue(np.isfinite(r.rx_power_dbm).all())


class CoherentSumTests(unittest.TestCase):
    def test_cir_by_hand(self):
        in_phase = ChannelImpulseResponse([ray(1.0, 10.0), ray(1.0, 20.0)])
        self.assertAlmostEqual(in_phase.total_received_power_linear(), 2.0, places=12)                 # 1 + 1
        self.assertAlmostEqual(in_phase.total_received_power_linear(coherent=True), 4.0, places=12)    # |1 + 1|^2
        self.assertAlmostEqual(in_phase.total_received_power_dbm(coherent=True), 10 * math.log10(4.0), places=12)
        opposite = ChannelImpulseResponse([ray(1.0, 10.0), ray(1.0, 20.0, phase=math.pi)])
        self.assertAlmostEqual(opposite.total_received_power_linear(coherent=True), 0.0, places=12)
        quadrature = ChannelImpulseResponse([ray(4.0, 10.0), ray(1.0, 20.0, phase=math.pi / 2)])
        self.assertAlmostEqual(quadrature.total_received_power_linear(coherent=True), 5.0, places=12)  # |2 + 1j|^2
        self.assertEqual(ChannelImpulseResponse().total_received_power_dbm(coherent=True), float("-inf"))
        exact_null = ChannelImpulseResponse([MultipathRay(0, 1 + 0j, 1.0, 0, 0, 0, 0), MultipathRay(0, -1 + 0j, 2.0, 0, 0, 0, 0)])
        self.assertEqual(exact_null.total_received_power_dbm(coherent=True), float("-inf"))
        self.assertAlmostEqual(exact_null.total_received_power_dbm(), 10 * math.log10(2.0), places=12)

    def test_cir_never_mixes_the_fields_of_different_transmitters(self):
        cir = ChannelImpulseResponse([ray(1.0, 10.0, tx_id=1), ray(1.0, 20.0, tx_id=1),
                                      ray(1.0, 30.0, tx_id=2, phase=math.pi)])
        self.assertAlmostEqual(cir.total_received_power_linear(tx_id=1, coherent=True), 4.0, places=12)
        self.assertAlmostEqual(cir.total_received_power_linear(tx_id=2, coherent=True), 1.0, places=12)
        self.assertAlmostEqual(cir.total_received_power_linear(coherent=True), 5.0, places=12)     # 4 + 1, not |1 + 1 - 1|^2

    def test_spreads_ignore_the_phase(self):
        a = ChannelImpulseResponse([ray(1.0, 100.0, tx_phi=10.0), ray(0.25, 200.0, tx_phi=50.0)])
        b = ChannelImpulseResponse([ray(1.0, 100.0, tx_phi=10.0), ray(0.25, 200.0, tx_phi=50.0, phase=math.pi)])
        self.assertAlmostEqual(a.rms_delay_spread(), 40.0, places=9)
        self.assertEqual(a.rms_delay_spread(), b.rms_delay_spread())
        self.assertEqual(a.mean_excess_delay(), b.mean_excess_delay())
        self.assertEqual(a.angular_spread(), b.angular_spread())

    def test_ray_field_total_power(self):
        tracer = RayTracer()
        tracer.add_reflector(PlanarReflector(p0=[60.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=0.5))
        tx = Transmitter(tx_id=0, x=0.0, y=0.0, z=10.0)
        points = np.array([[30.0, 5.0, 1.5], [45.0, -20.0, 1.5], [80.0, 3.0, 1.5]])
        field = tracer.trace_points(tx, points)
        np.testing.assert_array_equal(field.total_power_linear(), field.power_linear.sum(axis=0))
        for p, point in enumerate(points):
            cir = tracer.trace_paths(tx, tuple(point))
            self.assertAlmostEqual(field.total_power_linear(coherent=True)[p] / cir.total_received_power_linear(coherent=True),
                                   1.0, delta=1e-9)
            self.assertAlmostEqual(cir.total_received_power_linear(coherent=True) / abs(sum(r.magnitude for r in cir.rays)) ** 2,
                                   1.0, delta=1e-12)

    def test_direct_plus_ground_equals_the_two_ray_model(self):
        """
        Ray tracing without walls + coherent sum is the Two-Ray model: same Fresnel coefficient (eps_r 15,
        sigma 0.005, vertical polarization), heights, gain and distances.  Only rounding remains (~1e-10 dB).
        """
        worst = 0.0
        for f, z, gain, hr, res, size in ((2.4e9, 25.0, 12.0, 1.5, 5.0, 1000.0), (900e6, 30.0, 0.0, 2.0, 50.0, 5000.0),
                                          (3.5e9, 10.0, 17.0, 1.5, 1.0, 100.0)):
            sim = single_tx_sim(size / 2, size / 2, z, tx_power_dbm=43.0, frequency_hz=f, antenna_gain_dbi=gain)
            area = ((0.0, size), (0.0, size))
            coherent = sim.run_raytracing_simulation(*area, resolution_m=res, rx_height_m=hr, coherent=True)
            two_ray = sim.run_empirical_simulation(TwoRayGroundReflection(), *area, resolution_m=res, rx_height_m=hr)
            self.assertTrue(np.isfinite(coherent.path_loss_db).all())
            diff = np.abs(coherent.path_loss_db - two_ray.path_loss_db)
            worst = max(worst, float(diff.max()))
            np.testing.assert_allclose(coherent.rx_power_dbm, two_ray.rx_power_dbm, rtol=0.0, atol=1e-8)
            # ... while the default incoherent sum is a different thing (no fades)
            incoherent = sim.run_raytracing_simulation(*area, resolution_m=res, rx_height_m=hr)
            self.assertGreater(float(np.abs(incoherent.path_loss_db - two_ray.path_loss_db).max()), 3.0)
        self.assertLess(worst, 1e-8)

    def test_two_ray_with_other_ground_parameters(self):
        sim = single_tx_sim(0.0, 0.0, 12.0, frequency_hz=1.8e9, antenna_gain_dbi=5.0)
        sim.ray_tracer.ground_eps_r, sim.ray_tracer.ground_sigma = 4.0, 0.02
        area = ((10.0, 2000.0), (0.0, 0.0))
        coherent = sim.run_raytracing_simulation(*area, resolution_m=1.0, coherent=True)
        two_ray = sim.run_empirical_simulation(TwoRayGroundReflection(eps_r=4.0, sigma=0.02), *area, resolution_m=1.0)
        np.testing.assert_allclose(coherent.path_loss_db, two_ray.path_loss_db, rtol=0.0, atol=1e-8)
        default_ground = sim.run_empirical_simulation(TwoRayGroundReflection(), *area, resolution_m=1.0)
        self.assertGreater(float(np.abs(coherent.path_loss_db - default_ground.path_loss_db).max()), 0.5)

    def test_fast_fading_around_the_incoherent_power(self):
        sim = build_sim()
        incoherent = sim.run_raytracing_simulation(*AREA, resolution_m=4.0)
        coherent = sim.run_raytracing_simulation(*AREA, resolution_m=4.0, coherent=True)
        delta = coherent.rx_power_dbm - incoherent.rx_power_dbm
        # Three rays: at most (|a1|+|a2|+|a3|)^2 <= 3 * sum |a_k|^2  ->  +4.77 dB; fades go much deeper
        self.assertLessEqual(float(delta.max()), 10 * math.log10(3.0) + 1e-9)
        self.assertGreater(float(delta.max()), 1.0)
        self.assertLess(float(delta.min()), -10.0)
        # Averaged in linear units over many fades the coherent power tends to the incoherent one
        ratio = 10 ** (delta / 10.0)
        self.assertAlmostEqual(float(ratio.mean()), 1.0, delta=0.1)
        # The spreads still use |a_k|^2: same per-transmitter values wherever the best server did not change
        same = coherent.best_server_index == incoherent.best_server_index
        self.assertGreater(same.mean(), 0.5)
        np.testing.assert_array_equal(coherent.rms_delay_spread_ns[same], incoherent.rms_delay_spread_ns[same])
        np.testing.assert_array_equal(coherent.angular_spread_bs_phi[same], incoherent.angular_spread_bs_phi[same])

    def test_vectorized_matches_pointwise(self):
        sim = build_sim()
        vectorized = sim.run_raytracing_simulation(*AREA, resolution_m=20.0, coherent=True)
        pointwise = sim.run_raytracing_simulation(*AREA, resolution_m=20.0, coherent=True, vectorized=False)
        np.testing.assert_array_equal(vectorized.best_server_index, pointwise.best_server_index)
        assert_same_result(self, vectorized, pointwise, exact=False)
        self.assertGreater(float(np.abs(vectorized.rx_power_dbm - sim.run_raytracing_simulation(
            *AREA, resolution_m=20.0).rx_power_dbm).max()), 1.0)


class ShadowFadingTests(unittest.TestCase):
    def test_zero_sigma_has_no_effect(self):
        sim = build_sim()
        base = sim.run_raytracing_simulation(*AREA, resolution_m=20.0)
        assert_same_result(self, base, sim.run_raytracing_simulation(*AREA, resolution_m=20.0, shadow_fading_sigma_db=0.0, seed=1))
        emp = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=20.0)
        assert_same_result(self, emp, sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=20.0,
                                                                   shadow_fading_sigma_db=0.0, seed=1))

    def test_same_seed_is_reproducible(self):
        sim = build_sim()
        kwargs = dict(resolution_m=20.0, shadow_fading_sigma_db=6.0)
        a = sim.run_raytracing_simulation(*AREA, seed=42, **kwargs)
        b = sim.run_raytracing_simulation(*AREA, seed=42, **kwargs)
        c = sim.run_raytracing_simulation(*AREA, seed=43, **kwargs)
        assert_same_result(self, a, b)
        self.assertGreater(float(np.abs(a.rx_power_dbm - c.rx_power_dbm).mean()), 3.0)
        ea = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, seed=42, **kwargs)
        eb = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, seed=42, **kwargs)
        assert_same_result(self, ea, eb)
        # Without seed two runs differ
        u1 = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, **kwargs)
        u2 = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, **kwargs)
        self.assertFalse(np.array_equal(u1.rx_power_dbm, u2.rx_power_dbm))

    def test_same_samples_in_every_mode(self):
        """The (Tx, pixel) sample depends only on seed and grid: vectorized, chunked, pointwise and empirical agree."""
        from open_rps import simulation
        sim = build_sim()
        kwargs = dict(resolution_m=20.0, shadow_fading_sigma_db=8.0, seed=7)
        plain = sim.run_raytracing_simulation(*AREA, resolution_m=20.0)
        vectorized = sim.run_raytracing_simulation(*AREA, **kwargs)
        pointwise = sim.run_raytracing_simulation(*AREA, vectorized=False, **kwargs)
        np.testing.assert_array_equal(vectorized.best_server_index, pointwise.best_server_index)
        assert_same_result(self, vectorized, pointwise, exact=False)
        original = simulation._MAX_CHUNK_POINTS
        simulation._MAX_CHUNK_POINTS = 50
        try:
            chunked = sim.run_raytracing_simulation(*AREA, **kwargs)
        finally:
            simulation._MAX_CHUNK_POINTS = original
        assert_same_result(self, vectorized, chunked)
        shadow = vectorized.path_loss_db - plain.path_loss_db
        emp_plain = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=20.0)
        emp = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, **kwargs)
        np.testing.assert_allclose(emp.path_loss_db - emp_plain.path_loss_db, shadow, rtol=0.0, atol=1e-9)
        expected = np.random.default_rng(7).normal(0.0, 8.0, size=(2, 16, 21))
        np.testing.assert_allclose(shadow, expected, rtol=0.0, atol=1e-9)
        # P_rx + PL = P_tx still holds, and the spreads of each transmitter do not change
        np.testing.assert_allclose(vectorized.rx_power_dbm[0] + vectorized.path_loss_db[0], 43.0)
        same = vectorized.best_server_index == plain.best_server_index
        np.testing.assert_array_equal(vectorized.rms_delay_spread_ns[same], plain.rms_delay_spread_ns[same])

    def test_statistics_on_a_large_grid(self):
        """200 x 200 pixels x 2 transmitters: mean ~ 0 dB and standard deviation ~ sigma, independent per transmitter."""
        sim = build_sim(reflector=False)
        area = ((0.0, 995.0), (0.0, 995.0))
        sigma = 8.0
        for run in (lambda **kw: sim.run_raytracing_simulation(*area, resolution_m=5.0, **kw),
                    lambda **kw: sim.run_empirical_simulation(FreeSpacePathLoss(), *area, resolution_m=5.0, **kw)):
            plain = run()
            faded = run(shadow_fading_sigma_db=sigma, seed=2024)
            shadow = faded.path_loss_db - plain.path_loss_db            # [2, 200, 200] dB
            np.testing.assert_allclose(faded.rx_power_dbm - plain.rx_power_dbm, -shadow, rtol=0.0, atol=1e-9)
            for t in range(2):
                n = shadow[t].size
                self.assertEqual(n, 40000)
                self.assertAlmostEqual(float(shadow[t].mean()), 0.0, delta=4 * sigma / math.sqrt(n))        # 4 std errors
                self.assertAlmostEqual(float(shadow[t].std()), sigma, delta=4 * sigma / math.sqrt(2 * n))
                # Gaussian in dB (log-normal): ~68.3 % within one sigma, no spatial correlation between neighbours
                self.assertAlmostEqual(float((np.abs(shadow[t]) < sigma).mean()), 0.6827, delta=0.012)
                self.assertLess(abs(float(np.corrcoef(shadow[t][:, 1:].ravel(), shadow[t][:, :-1].ravel())[0, 1])), 0.03)
            self.assertLess(abs(float(np.corrcoef(shadow[0].ravel(), shadow[1].ravel())[0, 1])), 0.03)

    def test_invalid_sigma(self):
        sim = build_sim()
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sim.run_raytracing_simulation(*AREA, resolution_m=50.0, shadow_fading_sigma_db=bad)
            with self.assertRaises(ValueError):
                sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=50.0, shadow_fading_sigma_db=bad)


class SectorAntennaTests(unittest.TestCase):
    def sector(self, **kwargs):
        defaults = dict(tx_id=0, x=0.0, y=0.0, z=25.0, antenna_gain_dbi=17.0, pattern="sector")
        defaults.update(kwargs)
        return Transmitter(**defaults)

    def test_pattern_values(self):
        tx = self.sector()
        self.assertEqual((tx.azimuth_deg, tx.beamwidth_az_deg, tx.beamwidth_el_deg, tx.downtilt_deg), (0.0, 65.0, 65.0, 0.0))
        self.assertEqual(tx.pattern_gain_db(0.0, 0.0), 0.0)                                # boresight
        self.assertAlmostEqual(tx.pattern_gain_db(32.5, 0.0), -3.0, places=12)             # half the 3 dB beamwidth
        self.assertAlmostEqual(tx.pattern_gain_db(-32.5, 0.0), -3.0, places=12)
        self.assertAlmostEqual(tx.pattern_gain_db(0.0, 32.5), -3.0, places=12)
        self.assertAlmostEqual(tx.pattern_gain_db(0.0, -32.5), -3.0, places=12)
        self.assertAlmostEqual(tx.pattern_gain_db(32.5, 32.5), -6.0, places=12)            # A_H + A_V
        self.assertAlmostEqual(tx.pattern_gain_db(65.0, 0.0), -12.0, places=12)            # 12 * (65/65)^2
        self.assertAlmostEqual(tx.pattern_gain_db(20.0, -10.0), -12.0 * (20 / 65) ** 2 - 12.0 * (10 / 65) ** 2, places=12)
        self.assertAlmostEqual(tx.gain_dbi(32.5, 0.0), 14.0, places=12)
        self.assertIsInstance(tx.pattern_gain_db(10.0, 0.0), float)

    def test_saturation_at_30_db(self):
        tx = self.sector()
        self.assertEqual(tx.pattern_gain_db(180.0, 0.0), -30.0)           # back lobe: A_m
        self.assertEqual(tx.pattern_gain_db(-180.0, 0.0), -30.0)
        self.assertEqual(tx.pattern_gain_db(0.0, -90.0), -12.0 * (90 / 65) ** 2)   # -23.0 dB straight down, 65 deg beam
        self.assertEqual(self.sector(beamwidth_el_deg=10.0).pattern_gain_db(0.0, -90.0), -30.0)    # SLA_V
        self.assertEqual(tx.pattern_gain_db(180.0, 90.0), -30.0)          # A_H + A_V = 53 dB, limited by A_max
        self.assertEqual(tx.pattern_gain_db(103.0, 0.0), -30.0)           # 12 * (103/65)^2 = 30.1 dB
        self.assertAlmostEqual(tx.pattern_gain_db(102.0, 0.0), -12.0 * (102 / 65) ** 2, places=12)   # -29.5 dB, just before
        phi, theta = np.meshgrid(np.linspace(-180.0, 180.0, 73), np.linspace(-90.0, 90.0, 37))
        gain = tx.pattern_gain_db(phi, theta)
        self.assertEqual(gain.shape, phi.shape)
        self.assertEqual(float(gain.max()), 0.0)
        self.assertEqual(float(gain.min()), -30.0)

    def test_azimuth_wraps_and_downtilt_moves_the_boresight(self):
        tx = self.sector(azimuth_deg=170.0, beamwidth_az_deg=40.0, beamwidth_el_deg=10.0, downtilt_deg=6.0)
        self.assertEqual(tx.pattern_gain_db(170.0, -6.0), 0.0)
        self.assertAlmostEqual(tx.pattern_gain_db(-170.0, -6.0), -3.0, places=12)      # 20 deg across the +/-180 seam
        self.assertAlmostEqual(tx.pattern_gain_db(150.0, -6.0), -3.0, places=12)
        self.assertAlmostEqual(tx.pattern_gain_db(170.0, -11.0), -3.0, places=12)      # 5 deg below the tilted beam
        self.assertAlmostEqual(tx.pattern_gain_db(170.0, -1.0), -3.0, places=12)
        self.assertAlmostEqual(tx.pattern_gain_db(170.0, 0.0), -12.0 * 0.36, places=12)   # horizon: 6 deg off
        self.assertEqual(self.sector(azimuth_deg=-540.0).pattern_gain_db(180.0, 0.0), 0.0)

    def test_isotropic_and_validation(self):
        iso = Transmitter(tx_id=0, x=0.0, y=0.0, z=10.0, azimuth_deg=30.0, downtilt_deg=5.0)
        self.assertEqual(iso.pattern, "isotropic")
        self.assertEqual(iso.pattern_gain_db(123.0, -45.0), 0.0)
        self.assertEqual(iso.gain_dbi(123.0, -45.0), 14.0)
        self.assertTrue((iso.pattern_gain_db(np.array([0.0, 90.0, 180.0]), 0.0) == 0.0).all())
        self.assertEqual(Transmitter(tx_id=0, x=0, y=0, z=10, pattern="Sector").pattern, "sector")
        for bad in (dict(pattern="omni"), dict(pattern=None), dict(beamwidth_az_deg=0.0), dict(beamwidth_el_deg=-5.0),
                    dict(azimuth_deg=float("nan")), dict(downtilt_deg=float("inf"))):
            with self.assertRaises(ValueError, msg=str(bad)):
                Transmitter(tx_id=0, x=0, y=0, z=10, **bad)

    def test_each_ray_is_weighted_towards_its_own_departure_angle(self):
        tracer = RayTracer()
        tracer.add_reflector(PlanarReflector(p0=[0.0, 60.0, 0.0], normal=[0.0, 1.0, 0.0], reflectivity=0.7))
        kwargs = dict(x=0.0, y=0.0, z=20.0, azimuth_deg=20.0, beamwidth_az_deg=50.0, beamwidth_el_deg=12.0, downtilt_deg=8.0)
        tx_sector = self.sector(**kwargs)
        tx_iso = Transmitter(tx_id=0, antenna_gain_dbi=17.0, **kwargs)
        points = np.array([[120.0, 30.0, 1.5], [40.0, -25.0, 1.5], [-80.0, 10.0, 1.5], [15.0, 5.0, 10.0]])
        field = tracer.trace_points(tx_sector, points)
        attenuations = set()
        for p, point in enumerate(points):
            rays_sector = tracer.trace_paths(tx_sector, tuple(point)).rays
            rays_iso = tracer.trace_paths(tx_iso, tuple(point)).rays
            self.assertEqual(len(rays_sector), 3)
            for k, (rs, ri) in enumerate(zip(rays_sector, rays_iso)):
                self.assertEqual((rs.tau_ns, rs.tx_phi_deg, rs.tx_theta_deg), (ri.tau_ns, ri.tx_phi_deg, ri.tx_theta_deg))
                att = tx_sector.pattern_gain_db(rs.tx_phi_deg, rs.tx_theta_deg)
                attenuations.add(round(att, 6))
                self.assertAlmostEqual(10 * math.log10(rs.power_linear / ri.power_linear), att, places=9)
                # Same phase as the isotropic ray; the vectorized tracer agrees
                self.assertAlmostEqual(abs(rs.magnitude / ri.magnitude - 10 ** (att / 20.0)), 0.0, places=12)
                self.assertAlmostEqual(abs(field.amplitude[k, p] - rs.magnitude) / abs(rs.magnitude), 0.0, delta=1e-9)
        self.assertGreaterEqual(len(attenuations), 6)     # direct, ground and wall rays really get different gains

    def test_coverage_map_is_consistent_with_the_pattern(self):
        """Sector looking east from the middle of the map: front lobe strong, back 30 dB down, symmetric about the axis."""
        def make(pattern):
            sim = PropagationSimulator()
            sim.add_transmitter(Transmitter(tx_id=0, x=500.0, y=500.0, z=25.0, antenna_gain_dbi=17.0,
                                            azimuth_deg=0.0, pattern=pattern))
            return sim

        area = ((0.0, 1000.0), (0.0, 1000.0))
        for coherent in (False, True):
            iso = make("isotropic").run_raytracing_simulation(*area, resolution_m=50.0, coherent=coherent)
            sec = make("sector").run_raytracing_simulation(*area, resolution_m=50.0, coherent=coherent)
            pointwise = make("sector").run_raytracing_simulation(*area, resolution_m=50.0, coherent=coherent, vectorized=False)
            assert_same_result(self, sec, pointwise, exact=False)
            delta = (sec.rx_power_dbm - iso.rx_power_dbm)[0]
            self.assertLessEqual(float(delta.max()), 1e-9)                 # a pattern only attenuates
            np.testing.assert_allclose(delta[10, 0], -30.0, atol=1e-9)     # (0, 500): straight behind
            self.assertGreater(delta[10, 20], -0.05)                       # (1000, 500): boresight, ~3 deg below the horizon
            np.testing.assert_allclose(delta[::-1], delta, atol=1e-9)      # mirror symmetry about y = 500
            self.assertAlmostEqual(delta[20, 10], -12.0 * (90 / 65) ** 2, delta=0.1)   # (500, 1000): 90 deg off, 23 dB
            self.assertLess(delta[10, 0], delta[20, 10] - 5.0)             # back < side
            self.assertLess(delta[20, 10], delta[10, 20] - 10.0)           # side << front
            # Without walls the two rays leave at almost the same angle far from the mast:
            # the map is the isotropic one plus the pattern towards the pixel (direct-ray direction)
            tx = make("sector").transmitters[0]
            phi = np.degrees(np.arctan2(sec.y_coords[:, None] - 500.0, sec.x_coords[None, :] - 500.0))
            dist = np.hypot(sec.x_coords[None, :] - 500.0, sec.y_coords[:, None] - 500.0)
            theta = np.degrees(np.arctan2(1.5 - 25.0, np.maximum(1e-6, dist)))
            far = dist > 300.0
            np.testing.assert_allclose(delta[far], tx.pattern_gain_db(phi, theta)[far], atol=0.5)

    def test_empirical_models_use_the_direction_to_the_pixel(self):
        sim_iso, sim_sec = PropagationSimulator(), PropagationSimulator()
        kwargs = dict(tx_id=3, x=200.0, y=100.0, z=30.0, frequency_hz=900e6, azimuth_deg=120.0,
                      beamwidth_az_deg=70.0, beamwidth_el_deg=10.0, downtilt_deg=4.0)
        sim_iso.add_transmitter(Transmitter(**kwargs))
        sim_sec.add_transmitter(Transmitter(pattern="sector", **kwargs))
        tx = sim_sec.transmitters[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ModelRangeWarning)
            for model in (FreeSpacePathLoss(), TwoRayGroundReflection(), OkumuraHata(), LogDistancePathLoss()):
                iso = sim_iso.run_empirical_simulation(model, *AREA, resolution_m=20.0)
                sec = sim_sec.run_empirical_simulation(model, *AREA, resolution_m=20.0)
                for (j, i) in ((0, 0), (5, 10), (15, 20), (12, 3), (5, 11)):
                    dx, dy = sec.x_coords[i] - 200.0, sec.y_coords[j] - 100.0
                    phi = math.degrees(math.atan2(dy, dx))
                    theta = math.degrees(math.atan2(1.5 - 30.0, max(1e-6, math.hypot(dx, dy))))
                    self.assertAlmostEqual(sec.path_loss_db[0, j, i] - iso.path_loss_db[0, j, i],
                                           -tx.pattern_gain_db(phi, theta), places=9, msg=type(model).__name__)
                    self.assertAlmostEqual(sec.rx_power_dbm[0, j, i] + sec.path_loss_db[0, j, i], 43.0, places=9)
                self.assertTrue((sec.rx_power_dbm <= iso.rx_power_dbm + 1e-9).all())
        # FSPL with a sector equals the direct ray of the ray tracer
        direct = RayTracer().trace_paths(tx, (300.0, 240.0, 1.5)).rays[0]
        fspl = sim_sec.run_empirical_simulation(FreeSpacePathLoss(), (300.0, 300.0), (240.0, 240.0), resolution_m=10.0)
        self.assertAlmostEqual(10 * math.log10(direct.power_linear), fspl.rx_power_dbm[0, 0, 0], places=9)


def hand_result(rx_power_dbm, frequencies=None):
    """SimulationResult built by hand from per-transmitter powers [noOfTx, 1, Nx]."""
    power = np.array(rx_power_dbm, dtype=float)
    num_tx, ny, nx = power.shape
    clean = np.where(np.isnan(power), -np.inf, power)
    index = np.where(clean.max(axis=0) > -np.inf, clean.argmax(axis=0), -1)
    tx_ids = [10 * (t + 1) for t in range(num_tx)]
    return SimulationResult(
        x_coords=np.arange(nx) * 10.0, y_coords=np.arange(ny) * 10.0, rx_power_dbm=power, path_loss_db=40.0 - power,
        best_server_id=np.array(tx_ids + [-1])[index], max_rx_power_dbm=clean.max(axis=0),
        rms_delay_spread_ns=np.full((ny, nx), np.nan), angular_spread_bs_phi=np.full((ny, nx), np.nan),
        tx_ids=tx_ids, best_server_index=index, tx_positions=[(5.0 * t, 0.0, 10.0) for t in range(num_tx)],
        tx_frequencies_hz=frequencies)


class SinrTests(unittest.TestCase):
    def test_thermal_noise(self):
        """kTB at 290 K = -173.975 dBm/Hz; 20 MHz -> +73.010 dB; NF 7 dB  =>  -93.965 dBm."""
        self.assertAlmostEqual(thermal_noise_dbm(1.0), -173.975, places=3)
        self.assertAlmostEqual(thermal_noise_dbm(20e6, 7.0), -93.965, places=3)
        self.assertAlmostEqual(thermal_noise_dbm(20e6, 7.0), 10 * math.log10(1.380649e-23 * 290 * 20e6 * 1e3) + 7.0, places=12)
        self.assertAlmostEqual(thermal_noise_dbm(1e6, 0.0, temperature_k=580.0) - thermal_noise_dbm(1e6), 10 * math.log10(2), places=12)
        for bad in (dict(bandwidth_hz=0.0), dict(bandwidth_hz=-1.0), dict(bandwidth_hz=1e6, temperature_k=0.0),
                    dict(bandwidth_hz=1e6, noise_figure_db=float("nan"))):
            with self.assertRaises(ValueError):
                thermal_noise_dbm(**bad)

    def test_by_hand(self):
        """
        Three co-channel transmitters, noise N = -93.965 dBm = 4.0133e-10 mW (20 MHz, NF 7 dB).
        px 0: P = (-60, -70, -80) dBm = (1e-6, 1e-7, 1e-8) mW.  I = 1.1e-7 mW; I+N = 1.10401e-7 mW = -69.570 dBm;
              SINR = -60 + 69.570 = 9.570 dB.
        px 1: P = (-90, -inf, -inf): no interference, I+N = N; SNR = -90 + 93.965 = 3.965 dB.
        px 2: P = (-100, -100, -100) dBm = 1e-10 mW each: I = 2e-10; I+N = 6.0133e-10 mW = -92.209 dBm; SINR = -7.791 dB.
        px 3: nothing arrives: no server -> NaN.
        """
        inf = float("inf")
        result = hand_result([[[-60.0, -90.0, -100.0, -inf]], [[-70.0, -inf, -100.0, -inf]], [[-80.0, -inf, -100.0, -inf]]],
                             frequencies=[2.4e9, 2.4e9, 2.4e9])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sinr, i_plus_n = result.sinr_map()
        self.assertEqual(sinr.shape, (1, 4))
        self.assertAlmostEqual(sinr[0, 0], 9.570, places=3)
        self.assertAlmostEqual(i_plus_n[0, 0], -69.570, places=3)
        self.assertAlmostEqual(sinr[0, 1], 3.965, places=3)
        self.assertAlmostEqual(i_plus_n[0, 1], -93.965, places=3)
        self.assertAlmostEqual(sinr[0, 2], -7.791, places=3)
        self.assertAlmostEqual(i_plus_n[0, 2], -92.209, places=3)
        self.assertTrue(np.isnan(sinr[0, 3]) and np.isnan(i_plus_n[0, 3]))
        # Exact formula on the first pixel
        noise = 1.380649e-23 * 290.0 * 20e6 * 1e3 * 10 ** 0.7
        self.assertAlmostEqual(sinr[0, 0], 10 * math.log10(1e-6 / (1e-7 + 1e-8 + noise)), places=9)
        # Noise settings: 1 MHz and NF 5 dB -> N = -108.975 dBm; px 1 gets 18.975 dB, px 0 barely moves
        sinr2, _ = result.sinr_map(noise_figure_db=5.0, bandwidth_hz=1e6)
        self.assertAlmostEqual(sinr2[0, 1], 18.975, places=3)
        self.assertAlmostEqual(sinr2[0, 0], 10 * math.log10(1e-6 / (1.1e-7 + 10 ** -10.8975)), places=6)
        # sinr_map does not store anything
        self.assertIsNone(result.sinr_db)

    def test_only_co_channel_transmitters_interfere(self):
        """Tx 20 moves to another carrier: px 0 keeps only the -80 dBm interferer -> 10*log10(1e-6/(1e-8+N)) = 19.829 dB."""
        powers = [[[-60.0, -75.0]], [[-70.0, -65.0]], [[-80.0, -85.0]]]
        result = hand_result(powers, frequencies=[2.4e9, 2.6e9, 2.4e9])
        sinr, _ = result.sinr_map()
        self.assertAlmostEqual(sinr[0, 0], 19.829, places=3)
        # px 1 is served by the 2.6 GHz transmitter (-65 dBm), alone on its carrier: SNR = -65 + 93.965 = 28.965 dB
        self.assertEqual(result.best_server_id[0, 1], 20)
        self.assertAlmostEqual(sinr[0, 1], 28.965, places=3)
        # All on the same carrier (or no frequency metadata): everybody interferes
        for frequencies in ([2.4e9] * 3, None):
            sinr_all, _ = hand_result(powers, frequencies=frequencies).sinr_map()
            self.assertAlmostEqual(sinr_all[0, 0], 9.570, places=3)
            self.assertAlmostEqual(sinr_all[0, 1], 10 * math.log10(10 ** -6.5 / (10 ** -7.5 + 10 ** -8.5 + 4.0133e-10)), places=3)

    def test_unknown_interferer_gives_nan(self):
        nan = float("nan")
        result = hand_result([[[-60.0, -60.0, nan]], [[nan, -70.0, nan]]], frequencies=[1e9, 1e9])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sinr = result.compute_sinr()
        self.assertTrue(np.isnan(sinr[0, 0]))                  # interferer of unknown power (near field)
        self.assertAlmostEqual(sinr[0, 1], 10 * math.log10(1e-6 / (1e-7 + 4.0133e-10)), places=3)
        self.assertTrue(np.isnan(sinr[0, 2]))                  # no server
        # On another carrier the unknown transmitter does not matter
        other = hand_result([[[-60.0, -60.0, nan]], [[nan, -70.0, nan]]], frequencies=[1e9, 2e9])
        self.assertAlmostEqual(other.sinr_map()[0][0, 0], -60.0 + 93.965, places=3)

    def test_compute_sinr_stores_and_exports(self):
        result = hand_result([[[-60.0, -90.0, -float("inf")]], [[-70.0, -float("inf"), -float("inf")]]], frequencies=[1e9, 1e9])
        with tempfile.TemporaryDirectory() as tmp:
            before_csv, after_csv = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
            before_json, after_json = os.path.join(tmp, "a.json"), os.path.join(tmp, "b.json")
            result.export_csv(before_csv)
            result.export_json(before_json)
            returned = result.compute_sinr(noise_figure_db=5.0, bandwidth_hz=10e6)
            self.assertIs(returned, result.sinr_db)
            self.assertEqual(result.interference_plus_noise_dbm.shape, (1, 3))
            self.assertEqual(result.sinr_settings["noise_figure_db"], 5.0)
            self.assertEqual(result.sinr_settings["bandwidth_hz"], 10e6)
            self.assertAlmostEqual(result.sinr_settings["noise_power_dbm"], -98.975, places=3)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result.export_csv(after_csv)
                result.export_json(after_json)
            before, after = read_csv(before_csv), read_csv(after_csv)
            with open(before_json) as f:
                summary_before = json.load(f)
            with open(after_json) as f:
                text = f.read()
        old_columns = ["x_m", "y_m", "best_tx_id", "rx_power_dbm", "path_loss_db", "rms_delay_spread_ns", "angular_spread_phi_deg"]
        self.assertEqual(list(before[0]), old_columns)
        self.assertEqual(list(after[0]), old_columns + ["sinr_db", "interference_plus_noise_dbm"])
        for row_before, row_after in zip(before, after):
            self.assertEqual(row_before, {k: row_after[k] for k in old_columns})       # existing columns untouched
        self.assertEqual(after[0]["sinr_db"], f"{10 * math.log10(1e-6 / (1e-7 + 10 ** -9.8975)):.2f}")
        self.assertEqual(after[1]["sinr_db"], "8.98")                                   # -90 + 98.975
        self.assertEqual(after[1]["interference_plus_noise_dbm"], "-98.98")
        self.assertEqual((after[2]["best_tx_id"], after[2]["sinr_db"]), ("-1", "nan"))
        self.assertNotIn("sinr", summary_before)
        self.assertNotIn("NaN", text)
        summary = json.loads(text)
        self.assertEqual({k: summary[k] for k in summary_before}, summary_before)       # existing keys untouched
        self.assertEqual(summary["sinr"]["bandwidth_hz"], 10e6)
        self.assertAlmostEqual(summary["sinr"]["max_sinr_db"], float(np.nanmax(result.sinr_db)))
        self.assertAlmostEqual(summary["sinr"]["min_sinr_db"], 8.975, places=3)

    def test_simulation_result_end_to_end(self):
        sim = PropagationSimulator()
        sim.add_transmitter(Transmitter(tx_id=0, x=100.0, y=150.0, z=25.0, tx_power_dbm=43.0))
        sim.add_transmitter(Transmitter(tx_id=1, x=300.0, y=150.0, z=25.0, tx_power_dbm=43.0))
        sim.add_transmitter(Transmitter(tx_id=2, x=200.0, y=20.0, z=25.0, tx_power_dbm=43.0, frequency_hz=3.5e9))
        result = sim.run_raytracing_simulation(*AREA, resolution_m=20.0)
        self.assertEqual(result.tx_frequencies_hz, [2.4e9, 2.4e9, 3.5e9])
        sinr = result.compute_sinr()
        self.assertTrue(np.isfinite(sinr).all())
        power_mw = 10 ** (result.rx_power_dbm / 10.0)
        noise_mw = 10 ** (thermal_noise_dbm(20e6, 7.0) / 10.0)
        for (j, i) in ((0, 0), (7, 5), (7, 10), (1, 10), (15, 20)):
            best = result.best_server_index[j, i]
            interference = 0.0 if best == 2 else power_mw[1 - best, j, i]
            self.assertAlmostEqual(sinr[j, i], 10 * math.log10(power_mw[best, j, i] / (interference + noise_mw)), places=9)
        # Halfway between the two co-channel stations the SINR is ~0 dB; next to one of them it is high
        self.assertAlmostEqual(sinr[9, 10], 0.0, delta=0.5)
        self.assertGreater(sinr[7, 5], 15.0)
        self.assertEqual(result.best_server_id[1, 10], 2)
        self.assertGreater(sinr[1, 10], 60.0)                    # alone on its carrier: limited only by noise
        # Empirical mode carries the frequencies too
        emp = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=20.0)
        self.assertEqual(emp.tx_frequencies_hz, [2.4e9, 2.4e9, 3.5e9])
        self.assertTrue(np.isfinite(emp.compute_sinr()).all())

    def test_svg_maps(self):
        result = build_sim().run_raytracing_simulation(*AREA, resolution_m=20.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.svg")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                MapVisualizer.export_svg(result, path, metric="sinr")           # not computed yet: default settings
                self.assertIsNone(result.sinr_db)                                # ... and the result is not modified
                labels_default = [el.text for el in ET.parse(path).getroot().iter(SVG + "text")]
                sinr = result.compute_sinr(bandwidth_hz=200e3)
                MapVisualizer.export_svg(result, path, metric="sinr", theme="dark")
                root = ET.parse(path).getroot()
                labels = [el.text for el in root.iter(SVG + "text")]
                MapVisualizer.export_svg(result, path, metric="interference")
                labels_i = [el.text for el in ET.parse(path).getroot().iter(SVG + "text")]
        self.assertIn("SINR of the Best Server Map", labels)
        self.assertIn("dB", labels)
        self.assertIn(f"{sinr.max():.1f}", labels)
        self.assertIn(f"{sinr.min():.1f}", labels)
        self.assertIn(f"{result.sinr_map()[0].max():.1f}", labels_default)
        self.assertNotEqual(labels, labels_default)
        cells = [el for el in root.iter(SVG + "rect") if el.find(SVG + "title") is not None]
        self.assertEqual(len(cells), sinr.size)
        self.assertTrue(cells[0].find(SVG + "title").text.endswith(" dB"))
        self.assertIn("Interference + Noise (I + N) Map", labels_i)
        self.assertIn("dBm", labels_i)


class OkabeItoPaletteTests(unittest.TestCase):
    def simulate(self, n):
        sim = PropagationSimulator()
        for t in range(n):
            sim.add_transmitter(Transmitter(tx_id=100 + t, x=30.0 + 50.0 * t, y=30.0 + 37.0 * (t % 3), z=20.0))
        return sim.run_raytracing_simulation((0.0, 600.0), (0.0, 150.0), resolution_m=10.0)

    def export(self, result, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.svg")
            MapVisualizer.export_svg(result, path, metric="best_server", **kwargs)
            return ET.parse(path).getroot()

    def test_palette_is_okabe_ito(self):
        self.assertEqual(len(OKABE_ITO), 8)
        self.assertEqual(sorted(OKABE_ITO), sorted(["#000000", "#e69f00", "#56b4e9", "#009e73",
                                                    "#f0e442", "#0072b2", "#d55e00", "#cc79a7"]))
        self.assertEqual(_THEMES["light"]["categorical"], OKABE_ITO)
        # Dark theme: same colors, black swapped for white (it would not show on a dark background)
        self.assertEqual(_THEMES["dark"]["categorical"][:7], OKABE_ITO[:7])
        self.assertEqual(_THEMES["dark"]["categorical"][7], "#ffffff")
        for theme in _THEMES.values():
            self.assertNotIn(theme["other"], theme["categorical"])
            self.assertNotIn(theme["nodata"], theme["categorical"])

    def test_legend_and_cells_for_n_transmitters(self):
        for n in (1, 2, 3, 8, 9, 11):
            for theme in ("light", "dark"):
                with self.subTest(n=n, theme=theme):
                    palette, other = _THEMES[theme]["categorical"], _THEMES[theme]["other"]
                    result = self.simulate(n)
                    root = self.export(result, theme=theme)
                    # Legend: 12x12 swatches followed by their label, in order
                    swatches = [el.get("fill") for el in root.iter(SVG + "rect") if el.get("rx") == "2"]
                    labels = [el.text for el in root.iter(SVG + "text")]
                    legend = labels[labels.index("Best server") + 1:]
                    shown = min(n, 8)
                    self.assertEqual(legend[:shown], [f"Tx {100 + t}" for t in range(shown)])
                    self.assertEqual(swatches[:shown], palette[:shown])
                    if n > 8:
                        self.assertEqual(legend[shown:], [f"Other ({n - 8} Tx)"])
                        self.assertEqual(swatches[shown:], [other])
                    else:
                        self.assertEqual(len(legend), shown)
                        self.assertEqual(len(swatches), shown)
                    # Cells: transmitter number t is drawn with palette slot t (gray from the 9th on)
                    for cell in root.iter(SVG + "rect"):
                        title = cell.find(SVG + "title")
                        if title is not None:
                            slot = int(title.text.rsplit("Tx ", 1)[1]) - 100
                            self.assertEqual(cell.get("fill"), palette[slot] if slot < 8 else other)
                    served = {int(v) - 100 for v in np.unique(result.best_server_id)}
                    self.assertEqual(served, set(range(n)))        # every transmitter really appears on the map


def paint(root):
    """Axis-aligned rectangles drawn by the <path> elements of an exported map: {(x0, x1, y0, y1): fill}."""
    canvas = {}
    for path in root.iter(SVG + "path"):
        fill = path.get("fill")
        if fill in (None, "none"):
            continue
        for x0, y0, x1, y1, x2 in re.findall(r"M([\d.]+) ([\d.]+)H([\d.]+)V([\d.]+)H([\d.]+)z", path.get("d")):
            assert x0 == x2
            canvas[(float(x0), float(x1), float(y0), float(y1))] = fill
    return canvas


class SvgMergeTests(unittest.TestCase):
    def test_merged_rectangles_cover_exactly_the_cells(self):
        rng = np.random.default_rng(5)
        grids = [rng.integers(-1, 3, size=(40, 60)), np.full((7, 9), 4), np.full((5, 5), -1),
                 np.repeat(np.repeat(rng.integers(0, 4, size=(6, 8)), 5, axis=0), 7, axis=1),
                 np.arange(12).reshape(3, 4), np.array([[1]]), np.array([[2, 2, -1, 2, 2]])]
        for keys in grids:
            rectangles = _merged_rectangles(keys)
            rebuilt = np.full(keys.shape, -1, dtype=np.int64)
            covered = np.zeros(keys.shape, dtype=int)
            for i0, i1, j0, j1, key in rectangles:
                self.assertTrue(0 <= i0 < i1 <= keys.shape[1] and 0 <= j0 < j1 <= keys.shape[0])
                rebuilt[j0:j1, i0:i1] = key
                covered[j0:j1, i0:i1] += 1
            np.testing.assert_array_equal(rebuilt, np.where(keys >= 0, keys, -1))
            self.assertTrue((covered[keys >= 0] == 1).all())           # no overlaps
            self.assertTrue((covered[keys < 0] == 0).all())            # nothing drawn where there is no data
            self.assertLessEqual(len(rectangles), int((keys >= 0).sum()))
        self.assertEqual(_merged_rectangles(np.full((7, 9), 4)), [(0, 9, 0, 7, 4)])
        self.assertLessEqual(len(_merged_rectangles(grids[3])), 48)   # 6 x 8 blocks of 5 x 7 cells -> at most 48

    def test_vectorized_ramp_matches_the_scalar_one(self):
        rng = np.random.default_rng(11)
        norm = np.concatenate([rng.uniform(-0.2, 1.2, 5000), [0.0, 1.0, 0.5, 1 / 3, np.nan]])
        has_data = np.isfinite(norm)
        for theme in _THEMES.values():
            keys = _ramp_keys(theme["ramp"], norm, has_data)
            self.assertEqual(keys[-1], -1)
            for value, key in zip(norm[:-1], keys[:-1]):
                self.assertEqual(f"#{key:06x}", _ramp_color(theme["ramp"], float(value)))

    def large_result(self):
        sim = PropagationSimulator()
        sim.add_transmitter(Transmitter(tx_id=4, x=300.0, y=400.0, z=25.0))
        sim.add_transmitter(Transmitter(tx_id=9, x=1500.0, y=1100.0, z=25.0, tx_power_dbm=40.0))
        sim.add_reflector(PlanarReflector(p0=[1000.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=0.6))
        result = sim.run_raytracing_simulation((0.0, 2090.0), (0.0, 1990.0), resolution_m=10.0)     # 210 x 200 = 42 000 cells
        result.max_rx_power_dbm[150:, 180:] = np.nan              # a block without data
        result.best_server_id[150:, 180:] = -1
        return result

    def test_large_grid_is_merged_without_changing_the_picture(self):
        result = self.large_result()
        ny, nx = result.best_server_id.shape
        self.assertGreater(nx * ny, 40_000)
        with tempfile.TemporaryDirectory() as tmp:
            sizes = {}
            for metric in ("rx_power", "best_server"):
                path = os.path.join(tmp, f"{metric}.svg")
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    MapVisualizer.export_svg(result, path, metric=metric)
                sizes[metric] = os.path.getsize(path)
                root = ET.parse(path).getroot()
                self.assertEqual(len(list(root.iter(SVG + "title"))), 0)       # no tooltips on large grids
                # Rebuild the color of every cell from the merged shapes and compare with the expected one
                background = next(el for el in root.iter(SVG + "rect") if el.get("fill") == _THEMES["light"]["nodata"])
                left, top = float(background.get("x")), float(background.get("y"))
                cell = float(background.get("width")) / nx
                shapes = paint(root)
                colors = np.full((ny, nx), "", dtype=object)
                count = np.zeros((ny, nx), dtype=int)
                for (x0, x1, y0, y1), fill in shapes.items():
                    i0, i1 = round((x0 - left) / cell), round((x1 - left) / cell)
                    j1, j0 = ny - round((y0 - top) / cell), ny - round((y1 - top) / cell)
                    colors[j0:j1, i0:i1] = fill
                    count[j0:j1, i0:i1] += 1
                if metric == "best_server":
                    expected = np.full((ny, nx), "", dtype=object)
                    expected[result.best_server_id == 4] = OKABE_ITO[0]
                    expected[result.best_server_id == 9] = OKABE_ITO[1]
                    has_data = result.best_server_id >= 0
                else:
                    data = result.max_rx_power_dbm
                    has_data = np.isfinite(data)
                    lo, hi = np.nanmin(data), np.nanmax(data)
                    expected = np.full((ny, nx), "", dtype=object)
                    for j, i in zip(*np.nonzero(has_data)):
                        expected[j, i] = _ramp_color(_THEMES["light"]["ramp"], (float(data[j, i]) - lo) / (hi - lo))
                np.testing.assert_array_equal(colors, expected, err_msg=metric)
                np.testing.assert_array_equal(count, has_data.astype(int), err_msg=metric)
                self.assertLess(len(shapes), has_data.sum() / 2, msg=metric)
            # One 70-byte <rect> per cell (1.1.0) would be ~2.9 MB
            self.assertLess(sizes["rx_power"], 0.25 * 70 * nx * ny)
            self.assertLess(sizes["best_server"], 0.01 * 70 * nx * ny)

    def test_small_grids_keep_one_rect_and_tooltip_per_cell(self):
        result = build_sim().run_raytracing_simulation(*AREA, resolution_m=20.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.svg")
            MapVisualizer.export_svg(result, path, metric="best_server")
            root = ET.parse(path).getroot()
        self.assertEqual(len(list(root.iter(SVG + "title"))), result.best_server_id.size)
        self.assertEqual(paint(root), {})


if __name__ == "__main__":
    unittest.main()
