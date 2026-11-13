import cmath
import math
import unittest

import numpy as np

from open_rps import FreeSpacePathLoss, PlanarReflector, RayTracer, Transmitter
from open_rps.models import SPEED_OF_LIGHT


def make_tx(x=0.0, y=0.0, z=10.0, f=2.4e9, power=30.0, gain=6.0, tx_id=0):
    return Transmitter(tx_id=tx_id, x=x, y=y, z=z, tx_power_dbm=power, frequency_hz=f, antenna_gain_dbi=gain)


class DirectAndGroundRayTests(unittest.TestCase):
    def test_direct_ray_matches_friis(self):
        tx = make_tx()
        rx = (300.0, 400.0, 1.5)
        direct = RayTracer().trace_paths(tx, rx).rays[0]
        d = math.dist((tx.x, tx.y, tx.z), rx)
        expected_dbm = tx.tx_power_dbm + tx.antenna_gain_dbi - FreeSpacePathLoss().path_loss(d, tx.frequency_hz)
        self.assertAlmostEqual(10 * math.log10(direct.power_linear), expected_dbm, places=9)
        self.assertAlmostEqual(direct.tau_ns, d / SPEED_OF_LIGHT * 1e9, places=9)
        self.assertAlmostEqual(direct.tx_phi_deg, math.degrees(math.atan2(400.0, 300.0)), places=9)
        self.assertAlmostEqual(direct.rx_phi_deg, math.degrees(math.atan2(-400.0, -300.0)), places=9)
        self.assertAlmostEqual(direct.tx_theta_deg, -direct.rx_theta_deg, places=9)
        self.assertLess(direct.tx_theta_deg, 0.0)       # the base station looks down at the mobile

    def test_ground_ray_geometry(self):
        """Tx (0,0,10), Rx (30,0,5): the ray bounces at x = 30*10/(10+5) = 20 m on the ground.
        Both ends see the bounce point 26.565 deg below the horizon (tan = 10/20 = 5/10)."""
        tx = make_tx(z=10.0)
        cir = RayTracer().trace_paths(tx, (30.0, 0.0, 5.0))
        self.assertEqual(len(cir.rays), 2)
        ground = cir.rays[1]
        self.assertAlmostEqual(ground.tau_ns, math.hypot(30.0, 15.0) / SPEED_OF_LIGHT * 1e9, places=9)
        expected = -math.degrees(math.atan2(10.0, 20.0))
        self.assertAlmostEqual(ground.tx_theta_deg, expected, places=9)
        self.assertAlmostEqual(ground.rx_theta_deg, expected, places=9)
        self.assertAlmostEqual(ground.tx_phi_deg, 0.0, places=9)
        self.assertAlmostEqual(abs(ground.rx_phi_deg), 180.0, places=9)

    def test_ground_ray_points_at_the_bounce_point(self):
        """General case: AoD/AoA of the ground ray are the directions to p = tx_image + t*(rx - tx_image)."""
        tx = make_tx(x=12.0, y=-7.0, z=23.0)
        rx = np.array([140.0, 95.0, 1.5])
        ground = RayTracer().trace_paths(tx, tuple(rx)).rays[1]
        tx_image = np.array([tx.x, tx.y, -tx.z])
        t = tx.z / (tx.z + rx[2])
        bounce = tx_image + t * (rx - tx_image)
        self.assertAlmostEqual(bounce[2], 0.0, places=12)
        for origin, phi, theta in ((np.array([tx.x, tx.y, tx.z]), ground.tx_phi_deg, ground.tx_theta_deg),
                                   (rx, ground.rx_phi_deg, ground.rx_theta_deg)):
            v = bounce - origin
            self.assertAlmostEqual(phi, math.degrees(math.atan2(v[1], v[0])), places=9)
            self.assertAlmostEqual(theta, math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))), places=9)
            self.assertLess(theta, 0.0)

    def test_ground_ray_amplitude_uses_fresnel(self):
        tracer = RayTracer()
        tx = make_tx(z=10.0)
        direct, ground = tracer.trace_paths(tx, (500.0, 0.0, 1.5)).rays
        d_dir, d_gnd = math.hypot(500.0, 8.5), math.hypot(500.0, 11.5)
        gamma = tracer._fresnel_ground_reflection(math.atan2(11.5, 500.0), tx.frequency_hz)
        self.assertAlmostEqual(abs(ground.magnitude) / abs(direct.magnitude), abs(gamma) * d_dir / d_gnd, places=9)

    def test_no_ground_ray_below_ground(self):
        self.assertEqual(len(RayTracer().trace_paths(make_tx(), (30.0, 0.0, -2.0)).rays), 1)


class WallReflectionTests(unittest.TestCase):
    def setUp(self):
        self.tx = make_tx(x=-10.0, y=0.0, z=5.0)
        self.tracer = RayTracer()

    def wall(self, **kwargs):
        return PlanarReflector(p0=[0.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], **kwargs)

    def test_geometry_and_phase(self):
        """Tx (-10,0,5), Rx (-10,20,5), wall x = 0: hit point (0,10,5), path 2*sqrt(200) m."""
        self.tracer.add_reflector(self.wall(reflectivity=0.6))
        rays = self.tracer.trace_paths(self.tx, (-10.0, 20.0, 5.0)).rays
        self.assertEqual(len(rays), 3)
        direct, wall = rays[0], rays[2]
        d_wall = 2 * math.sqrt(200.0)
        self.assertAlmostEqual(wall.tau_ns, d_wall / SPEED_OF_LIGHT * 1e9, places=9)
        self.assertAlmostEqual(wall.tx_phi_deg, 45.0, places=9)
        self.assertAlmostEqual(wall.rx_phi_deg, -45.0, places=9)
        self.assertAlmostEqual(wall.tx_theta_deg, 0.0, places=9)
        self.assertAlmostEqual(abs(wall.magnitude) / abs(direct.magnitude), 0.6 * 20.0 / d_wall, places=9)
        # Coefficient -reflectivity: 180 deg on top of the propagation phase -k*d
        k = 2 * math.pi * self.tx.frequency_hz / SPEED_OF_LIGHT
        extra_phase = cmath.phase(wall.magnitude * cmath.exp(1j * (k * d_wall % (2 * math.pi))))
        self.assertAlmostEqual(abs(extra_phase), math.pi, places=6)

    def test_requires_tx_and_rx_on_the_same_side(self):
        self.tracer.add_reflector(self.wall())
        self.assertEqual(len(self.tracer.trace_paths(self.tx, (-5.0, 20.0, 5.0)).rays), 3)
        self.assertEqual(len(self.tracer.trace_paths(self.tx, (+5.0, 20.0, 5.0)).rays), 2)   # behind the wall
        self.assertEqual(len(self.tracer.trace_paths(self.tx, (0.0, 20.0, 5.0)).rays), 2)    # on the wall

    def test_both_faces_reflect(self):
        self.tracer.add_reflector(PlanarReflector(p0=[0.0, 0.0, 0.0], normal=[-1.0, 0.0, 0.0]))
        self.assertEqual(len(self.tracer.trace_paths(self.tx, (-5.0, 20.0, 5.0)).rays), 3)

    def test_bounds_limit_the_reflecting_patch(self):
        rx = (-10.0, 20.0, 5.0)     # hit point (0, 10, 5)
        inf = float("inf")
        cases = [((-1, 1, 0, 15, 0, 20), 3), ((-1, 1, 0, 5, 0, 20), 2),
                 ((-1, 1, -inf, inf, 6, 20), 2), ((-inf, inf, -inf, inf, -inf, inf), 3)]
        for bounds, expected in cases:
            with self.subTest(bounds=bounds):
                tracer = RayTracer()
                tracer.add_reflector(self.wall(bounds=bounds))
                self.assertEqual(len(tracer.trace_paths(self.tx, rx).rays), expected)

    def test_fresnel_wall_at_normal_incidence(self):
        """Lossless eps_r = 4: Gamma = (1 - 2)/(1 + 2) = -1/3 at normal incidence, whatever the reflectivity."""
        self.tracer.add_reflector(self.wall(reflectivity=0.9, eps_r=4.0))
        rays = self.tracer.trace_paths(self.tx, (-20.0, 0.0, 5.0)).rays    # path Tx -> wall -> Rx = 30 m, direct = 10 m
        self.assertAlmostEqual(abs(rays[2].magnitude) / abs(rays[0].magnitude), (1 / 3) * 10.0 / 30.0, places=9)
        # ... and it grows towards 1 at grazing incidence
        grazing = self.tracer.trace_paths(self.tx, (-10.0, 4000.0, 5.0)).rays
        d_wall = math.hypot(4000.0, 20.0)
        self.assertGreater(abs(grazing[2].magnitude) / abs(grazing[0].magnitude) * d_wall / 4000.0, 0.95)

    def test_invalid_reflectors(self):
        with self.assertRaises(ValueError):
            PlanarReflector(p0=[0, 0, 0], normal=[0, 0, 0])
        with self.assertRaises(ValueError):
            self.wall(reflectivity=1.5)
        with self.assertRaises(ValueError):
            self.wall(bounds=(0, 1, 0, 1))
        with self.assertRaises(ValueError):
            self.wall(bounds=(1, 0, 0, 1, 0, 1))
        with self.assertRaises(ValueError):
            Transmitter(tx_id=0, x=0, y=0, z=10, frequency_hz=0.0)


class VectorizedTracingTests(unittest.TestCase):
    def test_trace_points_matches_trace_paths(self):
        tracer = RayTracer()
        tracer.add_reflector(PlanarReflector(p0=[400.0, 300.0, 0.0], normal=[1.0, 0.3, 0.0], reflectivity=0.7))
        tracer.add_reflector(PlanarReflector(p0=[100.0, 600.0, 0.0], normal=[0.0, 1.0, 0.0], reflectivity=0.4,
                                             bounds=(0.0, 500.0, 590.0, 610.0, 0.0, 30.0)))
        tracer.add_reflector(PlanarReflector(p0=[700.0, 700.0, 10.0], normal=[0.5, 0.5, 0.2], eps_r=5.0, sigma=0.01))
        rng = np.random.default_rng(42)
        points = np.column_stack([rng.uniform(-100, 1000, 300), rng.uniform(-100, 1000, 300), rng.uniform(0.0, 40, 300)])
        points[0] = [120.0, 80.0, 30.0]      # exactly at the first transmitter
        points[1] = [50.0, 50.0, -3.0]       # below ground
        transmitters = [make_tx(120.0, 80.0, 30.0, 900e6), make_tx(610.0, 420.0, 18.0, 3.5e9, tx_id=1)]

        n_rays_seen = set()
        for tx in transmitters:
            field = tracer.trace_points(tx, points)
            self.assertEqual(field.amplitude.shape, (5, 300))
            for p, point in enumerate(points):
                rays = tracer.trace_paths(tx, tuple(point)).rays
                present = np.flatnonzero(field.valid[:, p])
                self.assertEqual(len(present), len(rays))
                n_rays_seen.add(len(rays))
                for k, ray in zip(present, rays):
                    self.assertAlmostEqual(abs(field.amplitude[k, p] - ray.magnitude) / abs(ray.magnitude), 0.0, delta=1e-8)
                    self.assertAlmostEqual(field.tau_ns[k, p], ray.tau_ns, delta=1e-9)
                    for name in ("tx_phi_deg", "tx_theta_deg", "rx_phi_deg", "rx_theta_deg"):
                        self.assertAlmostEqual(getattr(field, name)[k, p], getattr(ray, name), delta=1e-8)
                absent = ~field.valid[:, p]
                self.assertTrue(np.all(field.amplitude[absent, p] == 0.0))
                self.assertTrue(np.all(np.isnan(field.tau_ns[absent, p])))
        self.assertGreaterEqual(len(n_rays_seen), 4)     # the sample really exercises missing rays

    def test_grid_shaped_input(self):
        tracer = RayTracer()
        tracer.add_reflector(PlanarReflector(p0=[50.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0]))
        points = np.zeros((4, 6, 3))
        points[..., 0] = np.linspace(0.0, 100.0, 6)[None, :]
        points[..., 1] = np.linspace(0.0, 60.0, 4)[:, None]
        points[..., 2] = 1.5
        field = tracer.trace_points(make_tx(10.0, 10.0, 20.0), points)
        self.assertEqual(field.power_linear.shape, (3, 4, 6))
        self.assertTrue(field.valid[:2].all())
        self.assertTrue(field.valid[2][:, :3].all() and not field.valid[2][:, 3:].any())
        with self.assertRaises(ValueError):
            tracer.trace_points(make_tx(), np.zeros((5, 2)))
