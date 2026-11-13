import math
import unittest

import numpy as np

from open_rps import ChannelImpulseResponse, MultipathRay
from open_rps.multipath import circular_angular_spread, rms_delay_spread


def ray(power_mw, tau_ns, tx_phi=0.0, tx_theta=0.0, rx_phi=0.0, rx_theta=0.0, tx_id=0, phase=0.3):
    amplitude = math.sqrt(power_mw) * complex(math.cos(phase), math.sin(phase))
    return MultipathRay(tx_id, amplitude, tau_ns, tx_phi, tx_theta, rx_phi, rx_theta)


class DelaySpreadTests(unittest.TestCase):
    def test_two_rays_by_hand(self):
        """P = 1 mW at 100 ns and 0.25 mW at 200 ns:
        mean = (100 + 50)/1.25 = 120 ns;  E[tau^2] = (1e4 + 1e4)/1.25 = 16000;  sigma = sqrt(16000 - 14400) = 40 ns."""
        cir = ChannelImpulseResponse([ray(1.0, 100.0), ray(0.25, 200.0)])
        self.assertAlmostEqual(cir.total_received_power_linear(), 1.25, places=12)
        self.assertAlmostEqual(cir.total_received_power_dbm(), 10 * math.log10(1.25), places=12)
        self.assertAlmostEqual(cir.mean_excess_delay(), 120.0, places=9)
        self.assertAlmostEqual(cir.rms_delay_spread(), 40.0, places=9)

    def test_equal_power_rays(self):
        cir = ChannelImpulseResponse([ray(2.0, 0.0), ray(2.0, 100.0)])
        self.assertAlmostEqual(cir.rms_delay_spread(), 50.0, places=9)

    def test_single_ray_and_common_offset(self):
        self.assertEqual(ChannelImpulseResponse([ray(1.0, 333.0)]).rms_delay_spread(), 0.0)
        # The spread does not depend on a common absolute delay, even a huge one
        cir = ChannelImpulseResponse([ray(1.0, 1e7 + 100.0), ray(0.25, 1e7 + 200.0)])
        self.assertAlmostEqual(cir.rms_delay_spread(), 40.0, places=6)

    def test_empty_cir(self):
        cir = ChannelImpulseResponse()
        self.assertEqual(cir.total_received_power_linear(), 0.0)
        self.assertEqual(cir.total_received_power_dbm(), float("-inf"))
        self.assertTrue(math.isnan(cir.mean_excess_delay()))
        self.assertTrue(math.isnan(cir.rms_delay_spread()))
        self.assertTrue(all(math.isnan(v) for v in cir.angular_spread()))

    def test_tx_filter(self):
        cir = ChannelImpulseResponse([ray(1.0, 100.0, tx_id=3), ray(0.25, 200.0, tx_id=3), ray(5.0, 900.0, tx_id=8)])
        self.assertAlmostEqual(cir.rms_delay_spread(tx_id=3), 40.0, places=9)
        self.assertAlmostEqual(cir.total_received_power_linear(tx_id=8), 5.0, places=12)
        self.assertTrue(math.isnan(cir.rms_delay_spread(tx_id=99)))


class AngularSpreadTests(unittest.TestCase):
    def test_wraparound_at_180_degrees(self):
        """Two equal rays at +179 and -179 deg are 2 deg apart: mean = +/-180, spread = 1 deg (not 179)."""
        cir = ChannelImpulseResponse([ray(1.0, 0.0, tx_phi=179.0), ray(1.0, 0.0, tx_phi=-179.0)])
        mean, spread = cir.angular_spread(plane="phi", station="bs")
        self.assertAlmostEqual(abs(mean), 180.0, places=9)
        self.assertAlmostEqual(spread, 1.0, places=9)

    def test_spread_is_invariant_under_rotation(self):
        powers = [3.0, 1.0, 0.5]
        base = [-10.0, 10.0, 25.0]
        reference = ChannelImpulseResponse([ray(p, 0.0, tx_phi=a) for p, a in zip(powers, base)]).angular_spread()
        for shift in (170.0, 180.0, -175.0, 359.0):
            shifted = [((a + shift + 180.0) % 360.0) - 180.0 for a in base]
            mean, spread = ChannelImpulseResponse([ray(p, 0.0, tx_phi=a) for p, a in zip(powers, shifted)]).angular_spread()
            self.assertAlmostEqual(spread, reference[1], places=9)
            delta = (mean - reference[0] - shift + 180.0) % 360.0 - 180.0
            self.assertAlmostEqual(delta, 0.0, places=9)

    def test_by_hand_away_from_the_wrap(self):
        """Equal rays at 0 and 20 deg -> mean 10, spread 10."""
        cir = ChannelImpulseResponse([ray(1.0, 0.0, rx_phi=0.0, rx_theta=-5.0), ray(1.0, 0.0, rx_phi=20.0, rx_theta=5.0)])
        mean, spread = cir.angular_spread(plane="phi", station="ms")
        self.assertAlmostEqual(mean, 10.0, places=9)
        self.assertAlmostEqual(spread, 10.0, places=9)
        mean, spread = cir.angular_spread(plane="theta", station="ms")
        self.assertAlmostEqual(mean, 0.0, places=9)
        self.assertAlmostEqual(spread, 5.0, places=9)

    def test_invalid_arguments(self):
        cir = ChannelImpulseResponse([ray(1.0, 0.0)])
        with self.assertRaises(ValueError):
            cir.angular_spread(plane="azimuth")
        with self.assertRaises(ValueError):
            cir.angular_spread(station="tx")


class VectorizedMetricsTests(unittest.TestCase):
    def test_match_the_per_point_implementation(self):
        rng = np.random.default_rng(7)
        n_rays, n_points = 5, 60
        power = rng.uniform(0.0, 2.0, (n_rays, n_points))
        tau = rng.uniform(0.0, 5000.0, (n_rays, n_points))
        phi = rng.uniform(-180.0, 180.0, (n_rays, n_points))
        # Missing rays: zero power and NaN delay/angle must be ignored
        missing = rng.random((n_rays, n_points)) < 0.3
        missing[:, 0] = True                    # a point with no rays at all
        power[missing], tau[missing], phi[missing] = 0.0, np.nan, np.nan

        ds = rms_delay_spread(power, tau)
        mean, spread = circular_angular_spread(power, phi)
        self.assertEqual(ds.shape, (n_points,))
        self.assertTrue(np.isnan(ds[0]) and np.isnan(mean[0]) and np.isnan(spread[0]))
        for p in range(1, n_points):
            keep = ~missing[:, p]
            if not keep.any():
                self.assertTrue(np.isnan(ds[p]))
                continue
            cir = ChannelImpulseResponse([ray(power[k, p], tau[k, p], tx_phi=phi[k, p]) for k in np.flatnonzero(keep)])
            self.assertAlmostEqual(ds[p], cir.rms_delay_spread(), delta=1e-7)
            ref_mean, ref_spread = cir.angular_spread()
            self.assertAlmostEqual(spread[p], ref_spread, delta=1e-7)
            self.assertAlmostEqual((mean[p] - ref_mean + 180.0) % 360.0 - 180.0, 0.0, delta=1e-7)
