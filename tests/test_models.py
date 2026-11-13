import math
import unittest
import warnings

import numpy as np

from open_rps import (
    Cost231Hata,
    FreeSpacePathLoss,
    LogDistancePathLoss,
    ModelRangeWarning,
    OkumuraHata,
    TwoRayGroundReflection,
    fresnel_reflection_coefficient,
)


class FreeSpaceTests(unittest.TestCase):
    def test_known_values(self):
        fspl = FreeSpacePathLoss()
        # 20*log10(d_km) + 20*log10(f_MHz) + 32.45
        self.assertAlmostEqual(fspl.path_loss(1000.0, 2.4e9), 100.05, places=2)
        self.assertAlmostEqual(fspl.path_loss(1000.0, 900e6), 91.53, places=2)
        self.assertAlmostEqual(fspl.path_loss(10.0, 2.4e9), 60.05, places=2)

    def test_inverse_square_law(self):
        fspl = FreeSpacePathLoss()
        self.assertAlmostEqual(fspl.path_loss(2000.0, 1e9) - fspl.path_loss(1000.0, 1e9), 20 * math.log10(2), places=9)
        self.assertAlmostEqual(fspl.path_loss(500.0, 2e9) - fspl.path_loss(500.0, 1e9), 20 * math.log10(2), places=9)

    def test_gains_are_subtracted_without_clamping(self):
        fspl = FreeSpacePathLoss()
        base = fspl.path_loss(100.0, 2.4e9)
        self.assertAlmostEqual(fspl.path_loss(100.0, 2.4e9, tx_gain_dbi=12.0, rx_gain_dbi=3.0), base - 15.0, places=9)
        # Link loss may be negative with large gains on a short link (it used to be clamped at 0)
        self.assertAlmostEqual(fspl.path_loss(1.0, 100e6, tx_gain_dbi=30.0), fspl.path_loss(1.0, 100e6) - 30.0, places=9)
        self.assertLess(fspl.path_loss(1.0, 100e6, tx_gain_dbi=30.0), 0.0)

    def test_degenerate_distances(self):
        fspl = FreeSpacePathLoss()
        self.assertEqual(fspl.path_loss(0.0, 2.4e9), 0.0)
        self.assertEqual(fspl.path_loss(1e-6, 2.4e9), 0.0)   # near field: basic loss clamped at 0 dB

    def test_scalar_and_array(self):
        fspl = FreeSpacePathLoss()
        d = np.array([[0.0, 1.0, 10.0], [100.0, 1000.0, 5000.0]])
        out = fspl.path_loss(d, 1.8e9, tx_gain_dbi=5.0)
        self.assertEqual(out.shape, d.shape)
        self.assertIsInstance(fspl.path_loss(10.0, 1.8e9), float)
        for value, dist in zip(out.ravel(), d.ravel()):
            self.assertAlmostEqual(value, fspl.path_loss(float(dist), 1.8e9, tx_gain_dbi=5.0), places=9)

    def test_invalid_inputs(self):
        fspl = FreeSpacePathLoss()
        for bad_freq in (0.0, -1e9, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fspl.path_loss(100.0, bad_freq)
        with self.assertRaises(ValueError):
            fspl.path_loss(-1.0, 1e9)
        with self.assertRaises(ValueError):
            fspl.path_loss(np.array([10.0, -5.0]), 1e9)


class TwoRayTests(unittest.TestCase):
    def test_asymptotic_d4_law(self):
        """Far beyond the breakpoint 4*ht*hr/lambda the loss tends to 40log(d) - 20log(ht*hr)."""
        ht, hr, f = 30.0, 2.0, 900e6     # breakpoint ~ 720 m
        for polarization, tol in (("horizontal", 0.05), ("vertical", 0.1)):
            model = TwoRayGroundReflection(polarization=polarization)
            for d in (20e3, 40e3):
                expected = 40 * math.log10(d) - 20 * math.log10(ht * hr)
                self.assertAlmostEqual(model.path_loss(d, f, ht_m=ht, hr_m=hr), expected, delta=tol)
            # 40 dB/decade -> 12.04 dB per doubling of the distance
            slope = model.path_loss(40e3, f, ht_m=ht, hr_m=hr) - model.path_loss(20e3, f, ht_m=ht, hr_m=hr)
            self.assertAlmostEqual(slope, 40 * math.log10(2), delta=0.05)

    def test_asymptote_is_independent_of_frequency(self):
        model = TwoRayGroundReflection(polarization="horizontal")
        self.assertAlmostEqual(model.path_loss(50e3, 450e6, ht_m=30, hr_m=2),
                               model.path_loss(50e3, 900e6, ht_m=30, hr_m=2), delta=0.05)

    def test_oscillates_around_free_space_at_short_range(self):
        """Before the breakpoint the two rays interfere: the loss stays within +6 dB gain of free space."""
        model = TwoRayGroundReflection()
        fspl = FreeSpacePathLoss()
        d = np.linspace(20.0, 200.0, 400)
        diff = model.path_loss(d, 2.4e9, ht_m=25.0, hr_m=1.5) - fspl.path_loss(np.hypot(d, 23.5), 2.4e9)
        self.assertGreaterEqual(diff.min(), -6.1)     # at most field doubling (+6.02 dB)
        self.assertGreater(diff.max(), 1.0)           # and real destructive fades

    def test_reflection_coefficient_limits(self):
        lossless = dict(eps_r=4.0, sigma=0.0)
        normal = math.pi / 2
        self.assertAlmostEqual(TwoRayGroundReflection(polarization="horizontal", **lossless)
                               .reflection_coefficient(normal, 1e9), -1 / 3, places=12)
        self.assertAlmostEqual(TwoRayGroundReflection(polarization="vertical", **lossless)
                               .reflection_coefficient(normal, 1e9), 1 / 3, places=12)
        for polarization in ("horizontal", "vertical"):
            gamma = TwoRayGroundReflection(polarization=polarization).reflection_coefficient(1e-9, 1e9)
            self.assertAlmostEqual(gamma, -1.0, places=6)     # grazing incidence
        # Brewster angle (vertical polarization, lossless): tan(psi_B) = 1/sqrt(eps_r)
        gamma_b = fresnel_reflection_coefficient(math.atan(1 / 2.0), 1e9, 4.0, 0.0, "vertical")
        self.assertAlmostEqual(abs(gamma_b), 0.0, places=12)

    def test_scalar_matches_array_and_zero_distance_is_finite(self):
        model = TwoRayGroundReflection()
        d = np.array([0.0, 1.0, 37.5, 400.0, 12345.0])
        out = model.path_loss(d, 2.4e9, ht_m=25.0, hr_m=1.5)
        for value, dist in zip(out, d):
            self.assertAlmostEqual(value, model.path_loss(float(dist), 2.4e9, ht_m=25.0, hr_m=1.5), places=9)
        # Directly below the mast the loss is that of a ~23.5 m link, not 0 dB
        self.assertGreater(out[0], 50.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            TwoRayGroundReflection(polarization="circular")
        with self.assertRaises(ValueError):
            TwoRayGroundReflection().path_loss(100.0, 0.0)
        with self.assertRaises(ValueError):
            TwoRayGroundReflection().path_loss(-100.0, 1e9)


class HataTests(unittest.TestCase):
    ARGS = dict(ht_m=30.0, hr_m=1.5)

    def test_reference_values(self):
        """Hand-computed from Hata's formulas: f = 900 MHz, hb = 30 m, hm = 1.5 m, d = 1 km."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # inside the validity range: no warning allowed
            self.assertAlmostEqual(OkumuraHata("urban_medium").path_loss(1000.0, 900e6, **self.ARGS), 126.40, delta=0.01)
            self.assertAlmostEqual(OkumuraHata("suburban").path_loss(1000.0, 900e6, **self.ARGS), 116.46, delta=0.01)
            self.assertAlmostEqual(OkumuraHata("open").path_loss(1000.0, 900e6, **self.ARGS), 97.90, delta=0.01)
            self.assertAlmostEqual(OkumuraHata("urban_large").path_loss(5000.0, 900e6, ht_m=50.0, hr_m=1.5), 146.96, delta=0.01)

    def test_distance_slope(self):
        model = OkumuraHata()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            slope = model.path_loss(10e3, 900e6, **self.ARGS) - model.path_loss(1e3, 900e6, **self.ARGS)
        self.assertAlmostEqual(slope, 44.9 - 6.55 * math.log10(30.0), places=9)

    def test_gain_is_applied(self):
        model = OkumuraHata()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            base = model.path_loss(2000.0, 900e6, **self.ARGS)
            self.assertAlmostEqual(model.path_loss(2000.0, 900e6, tx_gain_dbi=12.0, **self.ARGS), base - 12.0, places=9)

    def test_warns_outside_validity_range(self):
        model = OkumuraHata()
        with self.assertWarns(ModelRangeWarning) as ctx:
            model.path_loss(2000.0, 2.4e9, **self.ARGS)
        self.assertIn("2400 MHz", str(ctx.warning))
        with self.assertWarns(ModelRangeWarning):
            model.path_loss(2000.0, 900e6, ht_m=10.0, hr_m=1.5)    # base antenna too low
        with self.assertWarns(ModelRangeWarning):
            model.path_loss(100.0, 900e6, **self.ARGS)             # closer than 1 km
        with self.assertWarns(ModelRangeWarning):
            model.path_loss(np.array([2000.0, 50e3]), 900e6, **self.ARGS)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            OkumuraHata("jungle")
        model = OkumuraHata()
        for kwargs in (dict(frequency_hz=0.0), dict(frequency_hz=-900e6),
                       dict(frequency_hz=900e6, ht_m=0.0), dict(frequency_hz=900e6, hr_m=-1.0)):
            with self.assertRaises(ValueError):
                model.path_loss(2000.0, **kwargs)

    def test_cost231_reference_and_range(self):
        """46.3 + 33.9*log10(1800) - 13.82*log10(30) - a(1.5) = 136.20 dB at 1 km."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            medium = Cost231Hata("urban_medium").path_loss(1000.0, 1800e6, **self.ARGS)
            metro = Cost231Hata("urban_large").path_loss(1000.0, 1800e6, **self.ARGS)
        self.assertAlmostEqual(medium, 136.20, delta=0.01)
        self.assertAlmostEqual(metro, 139.24, delta=0.01)
        with self.assertWarns(ModelRangeWarning):
            Cost231Hata().path_loss(2000.0, 2.4e9, **self.ARGS)
        with self.assertWarns(ModelRangeWarning):
            Cost231Hata().path_loss(2000.0, 900e6, **self.ARGS)
        with self.assertRaises(ValueError):
            Cost231Hata("open")

    def test_array_matches_scalar(self):
        d = np.array([1000.0, 2500.0, 19000.0])
        for model in (OkumuraHata("suburban"), Cost231Hata()):
            f = 900e6 if type(model) is OkumuraHata else 1900e6
            out = model.path_loss(d, f, **self.ARGS)
            for value, dist in zip(out, d):
                self.assertAlmostEqual(value, model.path_loss(float(dist), f, **self.ARGS), places=9)


class LogDistanceTests(unittest.TestCase):
    def test_reference_and_slope(self):
        model = LogDistancePathLoss(path_loss_exponent=3.5, d0_m=10.0)
        fspl = FreeSpacePathLoss()
        self.assertAlmostEqual(model.path_loss(10.0, 2.4e9), fspl.path_loss(10.0, 2.4e9), places=9)
        self.assertAlmostEqual(model.path_loss(1000.0, 2.4e9), fspl.path_loss(10.0, 2.4e9) + 70.0, places=9)
        self.assertAlmostEqual(model.path_loss(1.0, 2.4e9), fspl.path_loss(10.0, 2.4e9), places=9)   # held below d0
        self.assertAlmostEqual(model.path_loss(1000.0, 2.4e9, tx_gain_dbi=7.0), model.path_loss(1000.0, 2.4e9) - 7.0, places=9)
        with self.assertRaises(ValueError):
            LogDistancePathLoss(d0_m=0.0)


class GainConventionTests(unittest.TestCase):
    def test_every_model_applies_antenna_gains_the_same_way(self):
        models = [(FreeSpacePathLoss(), 900e6), (TwoRayGroundReflection(), 900e6), (LogDistancePathLoss(), 900e6),
                  (OkumuraHata(), 900e6), (Cost231Hata(), 1800e6)]
        for model, f in models:
            with self.subTest(model=type(model).__name__):
                base = model.path_loss(3000.0, f, ht_m=40.0, hr_m=1.5)
                with_gains = model.path_loss(3000.0, f, ht_m=40.0, hr_m=1.5, tx_gain_dbi=12.0, rx_gain_dbi=2.0)
                self.assertAlmostEqual(with_gains, base - 14.0, places=9)
                self.assertAlmostEqual(model.basic_path_loss(3000.0, f, ht_m=40.0, hr_m=1.5), base, places=9)
